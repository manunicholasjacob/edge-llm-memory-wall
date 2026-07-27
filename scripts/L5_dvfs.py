"""L5 — DVFS for decode. Sweep CPU frequency; throughput measured unperturbed, power via low-rate
sampler. Decode is bandwidth-bound so throughput should rise little with clock while power climbs,
placing the energy-optimal clock below maximum. threads=4."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

FREQS=[1500000,1800000,2100000,2400000]
MODELS=["qwen0.5b-q4km","llama1b-q4km"]

def main():
    out={"models":{}}
    for name in MODELS:
        if not os.path.exists(L.mpath(name)): print(f"[skip {name}]",flush=True); continue
        out["models"][name]={"file_mb":L.msize_bytes(name)/1e6,"freqs":{}}
        for f in FREQS:
            L.set_freq_khz(f); time.sleep(0.3)
            r=L.decode_energy(name, threads=4, n=200, reps=2)
            tg,P=r["tg"],r["P"]
            if not tg or not P: print(f"[{name} {f//1000}] FAIL",flush=True); continue
            etok=P/tg
            out["models"][name]["freqs"][str(f)]={"decode_ts":tg,"P_W":P,"energy_per_tok_mJ":etok*1e3,"edp":etok/tg}
            print(f"[{name} {f//1000}MHz] tok/s={tg:.2f} P={P:.2f}W E/tok={etok*1e3:.0f}mJ",flush=True)
            json.dump(out,open(os.path.expanduser("~/llm/results/L5_dvfs.json"),"w"),indent=2)
        fr=out["models"][name]["freqs"]; valid={k:v for k,v in fr.items() if v["energy_per_tok_mJ"]}
        if valid:
            eo=min(valid,key=lambda k:valid[k]["energy_per_tok_mJ"]); out["models"][name]["energy_opt_freq_mhz"]=int(eo)//1000
            print(f"[{name}] energy-opt freq={int(eo)//1000}MHz",flush=True)
    L.set_governor("schedutil")
    json.dump(out,open(os.path.expanduser("~/llm/results/L5_dvfs.json"),"w"),indent=2)
    print("DONE L5",flush=True)

if __name__=="__main__": main()
