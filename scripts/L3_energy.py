"""L3 — Energy per token (PMIC). Throughput measured unperturbed; power via a low-rate sampler in
a separate run. E/token = P_decode / decode_tok_s. Reports DDR-rail share. Freq pinned 2.4 GHz."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L

MODELS=["qwen0.5b-q4km","llama1b-q4km","qwen1.5b-q4km","qwen0.5b-q2k","qwen0.5b-q8"]

def main():
    L.set_freq_khz(2400000)
    # idle power (short, low-rate sampler)
    ps=L.PowerSampler(0.1); ps.start(); t0=L.now(); time.sleep(4.0); pid,_=ps.mean_power(t0,L.now(),0); ps.stop()
    out={"_meta":{"P_idle_W":pid},"models":{}}
    print(f"P_idle={pid:.2f}W",flush=True)
    for name in MODELS:
        if not os.path.exists(L.mpath(name)): print(f"[skip {name}]",flush=True); continue
        r=L.decode_energy(name, threads=4, n=256, reps=2)
        tg,P,Pddr=r["tg"],r["P"],r["Pddr"]
        if not tg or not P: print(f"[{name}] measure failed tg={tg} P={P}",flush=True); continue
        etok=P/tg
        rec={"file_mb":L.msize_bytes(name)/1e6,"decode_ts":tg,"P_decode_W":P,"P_core_W":r["Pcore"],
             "P_ddr_W":Pddr,"ddr_share_pct":(100*Pddr/P) if Pddr else None,
             "energy_per_tok_J":etok,"energy_per_tok_mJ":etok*1e3,
             "dyn_energy_per_tok_mJ":((P-pid)/tg)*1e3 if pid else None}
        out["models"][name]=rec
        print(f"[{name}] decode={tg:.2f} tok/s P={P:.2f}W (ddr {Pddr:.2f}) E/tok={etok*1e3:.0f}mJ ddr%={rec['ddr_share_pct']:.0f}",flush=True)
        json.dump(out,open(os.path.expanduser("~/llm/results/L3_energy.json"),"w"),indent=2)
    L.set_governor("schedutil")
    print("DONE L3",flush=True)

if __name__=="__main__": main()
