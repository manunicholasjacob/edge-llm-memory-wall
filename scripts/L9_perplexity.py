"""
L9 — Quality axis for the quantization Pareto. We measure perplexity (llama-perplexity) of each
Qwen2.5-0.5B quantization on a fixed English corpus, completing the speed/size/quality trade-off:
lower-bit quantization is faster and smaller (L2) but degrades quality; this quantifies by how much.
Absolute perplexity depends on the corpus; the RELATIVE ordering across bit-widths is the result.
"""
import os, sys, json, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

QUANTS=["qwen0.5b-q2k","qwen0.5b-q3km","qwen0.5b-q4km","qwen0.5b-q5km","qwen0.5b-q8"]
CORPUS=os.path.expanduser("~/llm/wiki.txt")

def ppl(name):
    cmd=[os.path.join(L.BIN,"llama-perplexity"),"-m",L.mpath(name),"-f",CORPUS,"-t","4","-c","512"]
    try:
        out=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=1800).stdout.decode()
    except Exception as e:
        return None, str(e)[:80]
    # final line: "Final estimate: PPL = 12.34 +/- ..."
    m=re.findall(r"PPL\s*=\s*([0-9]+\.[0-9]+)", out)
    if m: return float(m[-1]), None
    m=re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9]+\.[0-9]+)", out)
    return (float(m[-1]),None) if m else (None,"no PPL parsed")

def main():
    L.set_freq_khz(2400000)
    if not os.path.exists(CORPUS) or os.path.getsize(CORPUS)<1000:
        print("no corpus; skipping L9",flush=True); return
    out={"corpus_bytes":os.path.getsize(CORPUS),"models":{}}
    for name in QUANTS:
        if not os.path.exists(L.mpath(name)): print(f"[skip {name}]",flush=True); continue
        p,err=ppl(name)
        out["models"][name]={"quant":L.ZOO[name][2],"file_mb":L.msize_bytes(name)/1e6,"ppl":p}
        print(f"[{name}] {L.ZOO[name][2]:7s} {L.msize_bytes(name)/1e6:.0f}MB PPL={p} {err or ''}",flush=True)
        json.dump(out,open(os.path.expanduser("~/llm/results/L9_perplexity.json"),"w"),indent=2)
    L.set_governor("schedutil")
    print("DONE L9",flush=True)

if __name__=="__main__": main()
