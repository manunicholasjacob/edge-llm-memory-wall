"""
L2 — Quantization Pareto: speed, memory, (and quality) across Q2..Q8 for Qwen2.5-0.5B, plus the
size sweep at Q4 for reference. Decode tok/s should scale ~inversely with model bytes (bandwidth
bound), so lower-bit quant = faster decode AND smaller footprint. Quality (perplexity) is measured
separately if a corpus is present. Frequency pinned 2.4 GHz, threads=4.
"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

QUANTS = ["qwen0.5b-q2k","qwen0.5b-q3km","qwen0.5b-q4km","qwen0.5b-q5km","qwen0.5b-q8"]

def perplexity(name, corpus):
    """Optional: wikitext perplexity via llama-perplexity, short run. Returns ppl or None."""
    if not os.path.exists(corpus): return None
    cmd=[os.path.join(L.BIN,"llama-perplexity"),"-m",L.mpath(name),"-f",corpus,"-t","4","-c","512"]
    try:
        out=subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=900).decode()
        import re
        m=re.findall(r"[Ff]inal.*?([0-9]+\.[0-9]+)", out) or re.findall(r"PPL.*?([0-9]+\.[0-9]+)", out)
        return float(m[-1]) if m else None
    except Exception:
        return None

def main():
    L.set_freq_khz(2400000)
    corpus=os.path.expanduser("~/llm/wikitext.txt")
    out={"models":{}}
    for name in QUANTS:
        if not os.path.exists(L.mpath(name)):
            print(f"[skip {name}] missing", flush=True); continue
        mb=L.msize_bytes(name)/1e6
        try:
            r=L.run_bench(name, p=512, n=128, threads=4, reps=3)
        except Exception as e:
            print(f"[{name}] ERR {e}", flush=True); continue
        ppl=perplexity(name, corpus)
        out["models"][name]={"quant":L.ZOO[name][2],"file_mb":mb,
                             "pp_ts":r["pp_ts"],"tg_ts":r["tg_ts"],"ppl":ppl}
        print(f"[{name}] {L.ZOO[name][2]:7s} {mb:.0f}MB decode={r['tg_ts']:.2f} tok/s "
              f"prefill={r['pp_ts']:.1f} ppl={ppl}", flush=True)
        json.dump(out, open(os.path.expanduser("~/llm/results/L2_quant.json"),"w"), indent=2)
    L.set_governor("schedutil")
    print("DONE L2", flush=True)

if __name__=="__main__":
    main()
