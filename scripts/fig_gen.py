"""Generate all paper figures from LLM result JSONs. Run locally."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
FIG=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","paper","figs"); os.makedirs(FIG,exist_ok=True)
def L(n):
    p=os.path.join(RES,n); return json.load(open(p)) if os.path.exists(p) else None
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})
PEAK=13.98
NAME={"qwen0.5b-q4km":"Qwen2.5-0.5B","llama1b-q4km":"Llama-3.2-1B","qwen1.5b-q4km":"Qwen2.5-1.5B",
      "qwen0.5b-q2k":"0.5B Q2_K","qwen0.5b-q3km":"0.5B Q3_K_M","qwen0.5b-q5km":"0.5B Q5_K_M","qwen0.5b-q8":"0.5B Q8_0"}

def fig_roofline(e1, e2=None):
    import json as _j
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    # fit BW_eff over all points
    def peak_tg(m): return max(r["tg_ts"] for r in e1["models"][m]["threads"].values() if r["tg_ts"])
    size_pts=[(e1["models"][m]["file_mb"]/1000, peak_tg(m), NAME.get(m,m)) for m in e1["models"]]
    quant_pts=[]
    if e2:
        sn=set(e1["models"].keys())
        for m in e2["models"]:
            if m in sn: continue
            quant_pts.append((e2["models"][m]["file_mb"]/1000, e2["models"][m]["tg_ts"], e2["models"][m]["quant"]))
    allb=np.array([p[0] for p in size_pts+quant_pts]); allt=np.array([p[1] for p in size_pts+quant_pts])
    x=1.0/(allb*1e9); BW=float(np.sum(x*allt)/np.sum(x*x)); r2=1-np.sum((allt-BW*x)**2)/np.sum((allt-allt.mean())**2)
    xs=np.linspace(0.25,1.05,100)
    ax.plot(xs, (BW/1e9)/xs, 'k--', lw=1, label=f"tg = {BW/1e9:.1f} GB/s / bytes  ($R^2$={r2:.3f})")
    ax.scatter([p[0] for p in size_pts],[p[1] for p in size_pts], s=70, color="#2166ac", zorder=5, label="size sweep (Q4_K_M)")
    for gb,tg,lab in size_pts:
        ax.annotate(lab,(gb,tg),fontsize=6.5,xytext=(5,3),textcoords="offset points")
    if quant_pts:
        ax.scatter([p[0] for p in quant_pts],[p[1] for p in quant_pts], s=55, color="#c51b7d", marker="D", zorder=5, label="quant sweep (0.5B, Q2..Q8)")
    ax.set_xlabel("Model size (GB)"); ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Decode is bandwidth-bound: throughput $=$ BW$_\\mathrm{eff}$ / model bytes")
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(os.path.join(FIG,"fig1_roofline.png")); plt.close(fig)

def fig_threads(e1):
    fig,ax=plt.subplots(1,2,figsize=(8.2,3.3))
    for m in e1["models"]:
        th=sorted(e1["models"][m]["threads"].keys(),key=int)
        pp=[e1["models"][m]["threads"][t]["pp_ts"] for t in th]
        tg=[e1["models"][m]["threads"][t]["tg_ts"] for t in th]
        ax[0].plot([int(t) for t in th],pp,marker="o",label=NAME.get(m,m))
        ax[1].plot([int(t) for t in th],tg,marker="s",label=NAME.get(m,m))
    ax[0].set_title("(a) Prefill scales with threads"); ax[1].set_title("(b) Decode does not")
    for a in ax: a.set_xlabel("threads"); a.set_ylabel("tok/s")
    ax[0].legend(frameon=False,fontsize=7.5)
    fig.savefig(os.path.join(FIG,"fig2_threads.png")); plt.close(fig)

def fig_quant(e2):
    ms=list(e2["models"].keys())
    sz=[e2["models"][m]["file_mb"] for m in ms]; tg=[e2["models"][m]["tg_ts"] for m in ms]
    lab=[e2["models"][m]["quant"] for m in ms]
    fig,ax=plt.subplots(figsize=(6,3.4))
    ax.plot(sz,tg,"o-",color="#c51b7d")
    for s,t,l in zip(sz,tg,lab): ax.annotate(l,(s,t),fontsize=8,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("Model size (MB)"); ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Quantization frontier: fewer bits $\\to$ faster decode $+$ smaller")
    fig.savefig(os.path.join(FIG,"fig3_quant.png")); plt.close(fig)

def fig_quant2(e2, e9):
    """2-panel quant frontier: (a) decode tok/s vs size, (b) perplexity vs quant."""
    ms=list(e2["models"].keys())
    sz=[e2["models"][m]["file_mb"] for m in ms]; tg=[e2["models"][m]["tg_ts"] for m in ms]
    lab=[e2["models"][m]["quant"] for m in ms]
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.3))
    ax[0].plot(sz,tg,"o-",color="#c51b7d")
    for s,t,l in zip(sz,tg,lab): ax[0].annotate(l,(s,t),fontsize=7,xytext=(4,4),textcoords="offset points")
    ax[0].set_xlabel("Model size (MB)"); ax[0].set_ylabel("Decode tok/s"); ax[0].set_title("(a) Speed vs size")
    order=["Q2_K","Q3_K_M","Q4_K_M","Q5_K_M","Q8_0"]
    qm={e9["models"][m]["quant"]:e9["models"][m] for m in e9["models"] if e9["models"][m].get("ppl")}
    xs=[q for q in order if q in qm]; ppl=[qm[q]["ppl"] for q in xs]
    ax[1].plot(range(len(xs)),ppl,"s-",color="#2166ac")
    ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(xs,rotation=30,ha="right",fontsize=8)
    ax[1].set_ylabel("Perplexity (lower=better)"); ax[1].set_title("(b) Quality vs bit-width")
    fig.savefig(os.path.join(FIG,"fig3_quant.png")); plt.close(fig)

def fig_kvquant(e8):
    ms=list(e8["models"].keys())
    fig,ax=plt.subplots(figsize=(6,3.3))
    x=np.arange(len(ms)); w=0.38
    f16=[e8["models"][m]["kv"].get("f16",{}).get("max_ctx",0) for m in ms]
    q8=[e8["models"][m]["kv"].get("q8_0",{}).get("max_ctx",0) for m in ms]
    ax.bar(x-w/2,f16,w,color="#999999",label="f16 KV (default)")
    ax.bar(x+w/2,q8,w,color="#c51b7d",label="q8_0 KV")
    ax.set_xticks(x); ax.set_xticklabels([NAME.get(m,m) for m in ms])
    ax.set_ylabel("Max usable context (tokens)"); ax.set_title("KV-cache quantization extends context on 2\\,GB")
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(os.path.join(FIG,"fig8_kvquant.png")); plt.close(fig)

def fig_energy(e3):
    ms=[m for m in e3["models"]]
    et=[e3["models"][m]["energy_per_tok_mJ"] for m in ms]
    ddr=[e3["models"][m]["ddr_share_pct"] for m in ms]
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.3))
    ax[0].bar(range(len(ms)),et,color="#2166ac")
    ax[0].set_xticks(range(len(ms))); ax[0].set_xticklabels([NAME.get(m,m) for m in ms],rotation=40,ha="right",fontsize=7)
    ax[0].set_ylabel("Energy per token (mJ)"); ax[0].set_title("(a) Decode energy per token")
    ax[1].bar(range(len(ms)),ddr,color="#01665e")
    ax[1].set_xticks(range(len(ms))); ax[1].set_xticklabels([NAME.get(m,m) for m in ms],rotation=40,ha="right",fontsize=7)
    ax[1].set_ylabel("DDR rail share of power (%)"); ax[1].set_title("(b) DRAM-rail share during decode")
    fig.savefig(os.path.join(FIG,"fig4_energy.png")); plt.close(fig)

def fig_capacity(e4):
    fig,ax=plt.subplots(figsize=(6.2,3.4))
    for m in e4["models"]:
        cur=e4["models"][m]["curve"]
        ctx=[c["ctx"] for c in cur if c.get("decode_ts")]
        ts=[c["decode_ts"] for c in cur if c.get("decode_ts")]
        ax.plot(ctx,ts,marker="o",label=f"{NAME.get(m,m)} ({e4['models'][m]['file_mb']:.0f}MB)")
        # mark cap wall (skipped)
        for c in cur:
            if c.get("skipped_oom"): ax.axvline(c["ctx"],ls=":",alpha=0.4)
    ax.set_xscale("log",base=2); ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("Decode tok/s")
    ax.set_title("KV-cache capacity wall on 2\\,GB"); ax.legend(frameon=False,fontsize=7.5)
    fig.savefig(os.path.join(FIG,"fig5_capacity.png")); plt.close(fig)

def fig_dvfs(e5):
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.3))
    for m in e5["models"]:
        fr=sorted(e5["models"][m]["freqs"].keys(),key=int)
        mhz=[int(f)//1000 for f in fr]
        ts=[e5["models"][m]["freqs"][f]["decode_ts"] for f in fr]
        et=[e5["models"][m]["freqs"][f]["energy_per_tok_mJ"] for f in fr]
        ax[0].plot(mhz,ts,marker="o",label=NAME.get(m,m))
        ax[1].plot(mhz,et,marker="s",label=NAME.get(m,m))
    ax[0].set_title("(a) Throughput barely rises with clock"); ax[0].set_ylabel("decode tok/s")
    ax[1].set_title("(b) Energy/token minimized below max clock"); ax[1].set_ylabel("mJ/token")
    for a in ax: a.set_xlabel("CPU frequency (MHz)")
    ax[0].legend(frameon=False,fontsize=7.5)
    fig.savefig(os.path.join(FIG,"fig6_dvfs.png")); plt.close(fig)

def main():
    e1=L("L1_roofline.json"); e2=L("L2_quant.json"); e3=L("L3_energy.json")
    e4=L("L4_capacity.json"); e5=L("L5_dvfs.json")
    if e1: fig_roofline(e1, e2); fig_threads(e1); print("fig1,2 ok")
    e9=L("L9_perplexity.json"); e8kv=L("L8_kvquant.json")
    if e2 and e9 and any(e9["models"][m].get("ppl") for m in e9["models"]): fig_quant2(e2,e9); print("fig3(quality) ok")
    elif e2: fig_quant(e2); print("fig3 ok")
    if e8kv: fig_kvquant(e8kv); print("fig8 ok")
    if e3: fig_energy(e3); print("fig4 ok")
    if e4: fig_capacity(e4); print("fig5 ok")
    if e5: fig_dvfs(e5); print("fig6 ok")
    print("FIGS DONE")

if __name__=="__main__": main()
