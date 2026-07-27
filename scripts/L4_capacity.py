"""L4 — KV-cache capacity wall. Context is driven by the prompt length -p: processing P prompt
tokens requires a KV cache for P tokens, so sweeping -p grows KV until weights + KV exceed RAM.
We record prefill throughput and min MemAvailable at each context, stopping before the estimated
footprint risks swap-death. Freq pinned 2.4 GHz, threads=4."""
import os, sys, json, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

CTX=[512, 1024, 2048, 4096, 8192, 16384]
MODELS=["qwen0.5b-q4km","llama1b-q4km","qwen1.5b-q4km"]
SAFE_BUDGET_MB=1350   # tighter: keep KV+weights well clear of the 2GB limit (avoid swap-thrash)
def kv_est_mb(name,ctx): return ctx/1024.0*L.ZOO[name][1]*60.0
def avail_mb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"): return int(ln.split()[1])//1024
    return 0
def swap_mb():
    t=f=0
    for ln in open("/proc/meminfo"):
        if ln.startswith("SwapTotal"): t=int(ln.split()[1])//1024
        if ln.startswith("SwapFree"): f=int(ln.split()[1])//1024
    return t-f
def monitor(stop,box):
    while not stop.is_set(): box[0]=min(box[0],avail_mb()); time.sleep(0.2)

def main():
    L.set_freq_khz(2400000)
    out={"models":{}}
    for name in MODELS:
        if not os.path.exists(L.mpath(name)): print(f"[skip {name}]",flush=True); continue
        out["models"][name]={"file_mb":L.msize_bytes(name)/1e6,"curve":[]}
        for c in CTX:
            est=L.msize_bytes(name)/1e6 + kv_est_mb(name,c)
            if est>SAFE_BUDGET_MB:
                print(f"[{name} ctx={c}] SKIP est {est:.0f}MB>{SAFE_BUDGET_MB} -> capacity wall",flush=True)
                out["models"][name]["curve"].append({"ctx":c,"pp_ts":None,"skipped_oom":True,"est_mb":est})
                json.dump(out,open(os.path.expanduser("~/llm/results/L4_capacity.json"),"w"),indent=2); break
            box=[10**9]; stop=threading.Event(); th=threading.Thread(target=monitor,args=(stop,box),daemon=True); th.start()
            try:
                r=L.run_bench(name, p=c, n=0, threads=4, reps=1); pp=r["pp_ts"]; ok=True
            except Exception as e:
                pp=None; ok=False; print(f"[{name} ctx={c}] ERR {e}",flush=True)
            stop.set(); th.join(timeout=1)
            rec={"ctx":c,"pp_ts":pp,"min_avail_mb":box[0],"swap_mb":swap_mb(),"ok":ok,"est_mb":est}
            out["models"][name]["curve"].append(rec)
            print(f"[{name} ctx={c}] prefill={pp} min_avail={box[0]}MB swap={rec['swap_mb']}MB ok={ok}",flush=True)
            json.dump(out,open(os.path.expanduser("~/llm/results/L4_capacity.json"),"w"),indent=2)
            if not ok: break
    L.set_governor("schedutil")
    print("DONE L4",flush=True)

if __name__=="__main__": main()
