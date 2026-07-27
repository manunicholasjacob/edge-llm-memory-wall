"""Analysis on measured LLM data (local): (1) decode roofline predictive model,
(2) energy-optimal clock policy from L5, (3) headline stats. Produces fig7_policy + STATS.txt."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
FIG=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","paper","figs")
def L(n):
    p=os.path.join(RES,n); return json.load(open(p)) if os.path.exists(p) else None
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})
PEAK=13.98
S=[]
def P(x): S.append(x); print(x)

def roofline_model(e1):
    ms=list(e1["models"].keys())
    bytes_=np.array([e1["models"][m]["file_mb"]*1e6 for m in ms])
    # use PEAK decode throughput across thread counts = the true bandwidth ceiling
    def peak_tg(m):
        return max(r["tg_ts"] for r in e1["models"][m]["threads"].values() if r["tg_ts"])
    tg=np.array([peak_tg(m) for m in ms])
    # decode_ts = BW_eff / bytes  ->  tg = BW_eff * (1/bytes); fit slope through origin
    x=1.0/bytes_
    BW=float(np.sum(x*tg)/np.sum(x*x))    # bytes/s
    pred=BW*x
    r2=1-np.sum((tg-pred)**2)/np.sum((tg-tg.mean())**2)
    P(f"[roofline] BW_eff={BW/1e9:.2f} GB/s = {100*BW/1e9/PEAK:.0f}% of {PEAK} GB/s peak; "
      f"decode_ts = {BW/1e9:.2f}e9 / model_bytes ; R^2={r2:.3f}")
    util=[max(r["bw_util_pct"] for r in e1["models"][m]["threads"].values() if r["bw_util_pct"]) for m in ms]
    P(f"[roofline] per-model peak BW utilization: {min(util):.0f}%..{max(util):.0f}%")
    return {"BW_eff_GBs":BW/1e9,"r2":r2,"util_lo":min(util),"util_hi":max(util)}

def energy_policy(e5):
    """For a throughput SLO, pick the lowest-energy frequency that still meets it (from L5).
    Compare energy/token vs always-max-clock."""
    out={}
    for m in e5["models"]:
        fr=e5["models"][m]["freqs"]
        items=[(int(f), fr[f]["decode_ts"], fr[f]["energy_per_tok_mJ"]) for f in fr
               if fr[f]["decode_ts"] and fr[f]["energy_per_tok_mJ"]]
        if not items: continue
        items.sort()
        maxf=max(items,key=lambda x:x[0]); ts_max=maxf[1]; e_max=maxf[2]
        # SLO sweep as fractions of max throughput
        rows=[]
        for frac in [0.7,0.8,0.9,0.95,1.0]:
            slo=frac*ts_max
            feas=[it for it in items if it[1]>=slo*0.999]
            if not feas: continue
            best=min(feas,key=lambda x:x[2])   # min energy meeting SLO
            rows.append({"slo_frac":frac,"slo_tok_s":slo,"freq_mhz":best[0]//1000,
                         "energy_mJ":best[2],"vs_maxclock_pct":100*(e_max-best[2])/e_max})
        out[m]={"ts_max":ts_max,"e_maxclock_mJ":e_max,"rows":rows,
                "energy_opt_freq_mhz":min(items,key=lambda x:x[2])[0]//1000}
        best_sav=max((r["vs_maxclock_pct"] for r in rows),default=0)
        P(f"[policy {m}] energy-opt freq={out[m]['energy_opt_freq_mhz']}MHz; "
          f"up to {best_sav:.0f}% energy saved at relaxed SLO vs max-clock")
    # figure
    if out:
        fig,ax=plt.subplots(figsize=(6,3.4))
        for m in out:
            rows=out[m]["rows"]
            ax.plot([r["slo_frac"]*100 for r in rows],[r["vs_maxclock_pct"] for r in rows],
                    marker="o",label=m)
        ax.set_xlabel("Throughput SLO (% of max)"); ax.set_ylabel("Energy saved vs max-clock (%)")
        ax.set_title("Energy-optimal clock policy for decode")
        ax.legend(frameon=False,fontsize=8)
        fig.savefig(os.path.join(FIG,"fig7_policy.png")); plt.close(fig)
    return out

def quant_stats(e2):
    ms=list(e2["models"].keys())
    tg=[e2["models"][m]["tg_ts"] for m in ms]; sz=[e2["models"][m]["file_mb"] for m in ms]
    P(f"[quant] decode {min(tg):.1f}..{max(tg):.1f} tok/s across {min(sz):.0f}..{max(sz):.0f} MB "
      f"(Q2..Q8); speedup Q8->Q2 = {max(tg)/min(tg):.2f}x")
    ppl=[(e2["models"][m]["quant"],e2["models"][m]["ppl"]) for m in ms if e2["models"][m].get("ppl")]
    if ppl: P(f"[quant] perplexity: "+", ".join(f"{q}={p:.2f}" for q,p in ppl))

def main():
    e1=L("L1_roofline.json"); e2=L("L2_quant.json"); e5=L("L5_dvfs.json")
    out={}
    if e1: out["roofline"]=roofline_model(e1)
    if e2: quant_stats(e2)
    if e5: out["policy"]=energy_policy(e5)
    json.dump(out,open(os.path.join(RES,"analysis.json"),"w"),indent=2)
    open(os.path.join(RES,"STATS.txt"),"w").write("\n".join(S))
    print("wrote analysis.json, STATS.txt")

if __name__=="__main__": main()
