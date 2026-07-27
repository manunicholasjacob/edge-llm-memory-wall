"""
L7 — Batch-size sweep: validate the memory-wall MECHANISM by the intervention that breaks it.
Single-stream decode is bandwidth-bound because each token reads all weights for one sequence.
Batching B sequences amortizes each weight read across B tokens, raising arithmetic intensity and
lifting decode off the bandwidth slope. We run llama-batched-bench at increasing batch sizes and
record aggregate decode throughput (S_TG): if the memory-wall model is right, aggregate decode
tok/s should rise with batch (until compute- or capacity-bound), even though per-sequence rate
falls. Freq pinned 2.4 GHz, threads=4.
"""
import os, sys, json, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

BB=os.path.join(L.BIN,"llama-batched-bench")
MODELS=["qwen0.5b-q4km","llama1b-q4km"]
BATCHES=[1,2,4,8]
PP=64; TG=128

def run_bb(name):
    npl=",".join(str(b) for b in BATCHES)
    cmd=[BB,"-m",L.mpath(name),"-t","4","-npp",str(PP),"-ntg",str(TG),"-npl",npl,"-c","4096"]
    out=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL).stdout.decode()
    rows={}
    # table rows: | PP | TG | B | N_KV | T_PP | S_PP | T_TG | S_TG | T | S |
    for line in out.splitlines():
        parts=[p.strip() for p in line.split("|") if p.strip()]
        if len(parts)>=10:
            try:
                pp=int(parts[0]); tg=int(parts[1]); B=int(parts[2])
                s_pp=float(parts[5]); s_tg=float(parts[7]); s_tot=float(parts[9])
                rows[B]={"S_pp":s_pp,"S_tg_aggregate":s_tg,"S_tg_per_seq":s_tg/B,"S_total":s_tot}
            except (ValueError, IndexError):
                continue
    return rows

def main():
    L.set_freq_khz(2400000)
    out={"pp":PP,"tg":TG,"models":{}}
    for name in MODELS:
        if not os.path.exists(L.mpath(name)): print(f"[skip {name}]",flush=True); continue
        try:
            rows=run_bb(name)
        except Exception as e:
            print(f"[{name}] ERR {e}",flush=True); continue
        out["models"][name]={"file_mb":L.msize_bytes(name)/1e6,"batches":rows}
        for B in sorted(rows):
            r=rows[B]
            print(f"[{name} B={B}] decode_aggregate={r['S_tg_aggregate']:.1f} tok/s "
                  f"per_seq={r['S_tg_per_seq']:.1f} prefill={r['S_pp']:.1f}",flush=True)
        json.dump(out,open(os.path.expanduser("~/llm/results/L7_batch.json"),"w"),indent=2)
    L.set_governor("schedutil")
    print("DONE L7",flush=True)

if __name__=="__main__": main()
