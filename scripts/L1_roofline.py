"""
L1 — Prefill/decode roofline + thread scaling (the memory-wall centerpiece for LLMs).

Decode (token generation) reads the entire model from DRAM per token, so decode tok/s should be
bandwidth-bound: tok/s ~= BW_eff / bytes_per_token, with bytes_per_token ~= model file size.
Prefill (prompt processing) is a GEMM over many tokens -> compute-bound and thread-scalable.
We run llama-bench across model sizes (fixed Q4_K_M) and thread counts, and compute the effective
decode bandwidth = tg_tok/s * model_bytes, comparing to the platform DRAM peak (~13.98 GB/s from
the memory-wall study). Frequency pinned 2.4 GHz.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

SIZE_SWEEP = ["qwen0.5b-q4km","llama1b-q4km","qwen1.5b-q4km"]
THREADS = [1,2,3,4]
PEAK_BW_GBs = 13.98  # measured DRAM read peak (memory-wall paper)

def main():
    L.set_freq_khz(2400000)
    print("pinned", L.cur_freqs(), flush=True)
    out={"peak_bw_GBs":PEAK_BW_GBs,"models":{}}
    for name in SIZE_SWEEP:
        if not os.path.exists(L.mpath(name)):
            print(f"[skip {name}] missing", flush=True); continue
        mb=L.msize_bytes(name)/1e6
        out["models"][name]={"file_mb":mb,"params_B":L.ZOO[name][1],"threads":{}}
        for t in THREADS:
            try:
                r=L.run_bench(name, p=512, n=128, threads=t, reps=3)
            except Exception as e:
                print(f"[{name} t={t}] ERR {e}", flush=True); continue
            tg=r["tg_ts"]; pp=r["pp_ts"]
            bw_eff = (tg*L.msize_bytes(name)/1e9) if tg else None   # GB/s streamed at decode
            out["models"][name]["threads"][str(t)]={
                "pp_ts":pp,"tg_ts":tg,"decode_bw_GBs":bw_eff,
                "bw_util_pct":(100*bw_eff/PEAK_BW_GBs) if bw_eff else None}
            print(f"[{name} {mb:.0f}MB t={t}] prefill={pp:.1f} decode={tg:.2f} tok/s "
                  f"BW={bw_eff:.2f} GB/s ({100*bw_eff/PEAK_BW_GBs:.0f}% peak)" if tg else
                  f"[{name} t={t}] no tg", flush=True)
            json.dump(out, open(os.path.expanduser("~/llm/results/L1_roofline.json"),"w"), indent=2)
    L.set_governor("schedutil")
    print("DONE L1", flush=True)

if __name__=="__main__":
    main()
