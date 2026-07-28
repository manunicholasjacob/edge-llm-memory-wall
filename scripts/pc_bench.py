"""Cross-platform roofline on x86: run llama-bench on each model, record peak decode throughput
(over a thread sweep) and prefill, and compute effective decode bandwidth = tok/s * model_bytes.
Compares directly against the Raspberry Pi 5 (Cortex-A76) roofline."""
import os, json, subprocess

BIN = r"C:\llmpc\bin\llama-bench.exe"
MODELS_DIR = r"C:\llmpc\models"
PEAK_BW = 42.1  # measured achievable read GB/s (mem_bw.py); Pi5 peak = 13.98 GB/s
# name -> (file, params_B). Same 7 as the Pi, plus larger PC-only models.
ZOO = {
    "qwen0.5b-q2k":  ("qwen0.5b-q2k.gguf", 0.5),
    "qwen0.5b-q4km": ("qwen0.5b-q4km.gguf", 0.5),
    "qwen0.5b-q8":   ("qwen0.5b-q8.gguf", 0.5),
    "llama1b-q4km":  ("llama1b-q4km.gguf", 1.24),
    "qwen1.5b-q4km": ("qwen1.5b-q4km.gguf", 1.5),
    "qwen3b-q4km":   ("qwen3b-q4km.gguf", 3.0),
    "qwen7b-q4km":   ("qwen7b-q4km.gguf", 7.0),
}
THREADS = [8, 14]   # 14 = peak decode on the 6P+8E i7-12700H (20 threads over-subscribes E-cores)

def bench(path, t):
    cmd = [BIN, "-m", path, "-p", "128", "-n", "128", "-t", str(t), "-r", "3", "-o", "json"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        data = json.loads(out)
    except Exception as e:
        return None, None
    pp = tg = None
    for r in data:
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0: pp = r.get("avg_ts")
        elif r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0: tg = r.get("avg_ts")
    return pp, tg

def main():
    out = {"platform": "x86 Intel i7-12700H, DDR5-4800 dual-channel",
           "peak_read_GBs": PEAK_BW, "models": {}}
    for name, (fn, pB) in ZOO.items():
        path = os.path.join(MODELS_DIR, fn)
        if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            print(f"[skip {name}] missing"); continue
        mb = os.path.getsize(path) / 1e6
        best_tg = 0.0; best_pp = 0.0; best_t = None
        for t in THREADS:
            pp, tg = bench(path, t)
            if tg and tg > best_tg:
                best_tg, best_pp, best_t = tg, pp, t
        if not best_tg:
            print(f"[{name}] bench failed"); continue
        bw_eff = best_tg * os.path.getsize(path) / 1e9
        out["models"][name] = {"file_mb": mb, "params_B": pB, "decode_ts": best_tg,
                               "prefill_ts": best_pp, "best_threads": best_t,
                               "decode_bw_GBs": bw_eff, "bw_util_pct": 100 * bw_eff / PEAK_BW}
        print(f"[{name}] {mb:.0f}MB decode={best_tg:.1f} tok/s (t={best_t}) "
              f"prefill={best_pp:.0f} BW_eff={bw_eff:.1f} GB/s ({100*bw_eff/PEAK_BW:.0f}% peak)", flush=True)
        json.dump(out, open(r"C:\llmpc\pc_roofline.json", "w"), indent=2)
    print("DONE PC BENCH")

if __name__ == "__main__":
    main()
