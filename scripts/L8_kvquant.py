"""
L8 — KV-cache quantization extends the capacity wall. The KV cache is half-precision (f16) by
default; quantizing it to q8_0 halves its bytes, so on a fixed 2 GB budget a model can reach a
longer context before weights + KV exhaust memory. We sweep context for a model with f16 vs q8_0
KV cache and record the max context that fits (prefill succeeds without swap). Freq pinned.
"""
import os, sys, json, subprocess, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

MODELS=["llama1b-q4km","qwen1.5b-q4km"]
CTX=[2048,4096,8192,16384,32768]
KV_TYPES=["f16","q8_0"]
SAFE_AVAIL_MB=250   # abort a point if MemAvailable would dip below this (swap-death guard)

def avail_mb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"): return int(ln.split()[1])//1024
    return 0

def run_ctx(name, ctx, kv):
    """llama-bench prefill of `ctx` tokens with KV cache type kv. Returns (pp_ts, min_avail, ok)."""
    cmd=[os.path.join(L.BIN,"llama-bench"),"-m",L.mpath(name),"-t","4","-r","1","-o","json",
         "-p",str(ctx),"-n","0","-ctk",kv,"-ctv",kv]
    box=[10**9]; stop=threading.Event()
    def mon():
        while not stop.is_set(): box[0]=min(box[0],avail_mb()); time.sleep(0.15)
    th=threading.Thread(target=mon,daemon=True); th.start()
    try:
        out=subprocess.check_output(cmd, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=600).decode()
        import json as J; data=J.loads(out)
        pp=next((r.get("avg_ts") for r in data if r.get("n_prompt",0)>0), None); ok=pp is not None
    except Exception:
        pp=None; ok=False
    stop.set(); th.join(timeout=1)
    return pp, box[0], ok

def main():
    L.set_freq_khz(2400000)
    out={"models":{}}
    for name in MODELS:
        if not os.path.exists(L.mpath(name)): print(f"[skip {name}]",flush=True); continue
        out["models"][name]={"file_mb":L.msize_bytes(name)/1e6,"kv":{}}
        for kv in KV_TYPES:
            maxctx=0; rows=[]
            for c in CTX:
                if avail_mb() < SAFE_AVAIL_MB+200: break
                pp,mavail,ok = run_ctx(name,c,kv)
                rows.append({"ctx":c,"pp_ts":pp,"min_avail_mb":mavail,"ok":ok})
                print(f"[{name} kv={kv} ctx={c}] prefill={pp} min_avail={mavail}MB ok={ok}",flush=True)
                if ok and mavail>SAFE_AVAIL_MB: maxctx=c
                else: break
            out["models"][name]["kv"][kv]={"max_ctx":maxctx,"curve":rows}
            json.dump(out,open(os.path.expanduser("~/llm/results/L8_kvquant.json"),"w"),indent=2)
        f16=out["models"][name]["kv"].get("f16",{}).get("max_ctx",0)
        q8=out["models"][name]["kv"].get("q8_0",{}).get("max_ctx",0)
        print(f"[{name}] max ctx: f16={f16} q8_0={q8} (KV-q8 extends by {q8/f16:.1f}x)" if f16 else f"[{name}] f16 max=0",flush=True)
    L.set_governor("schedutil")
    print("DONE L8",flush=True)

if __name__=="__main__": main()
