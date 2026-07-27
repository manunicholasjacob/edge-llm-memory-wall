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

def fig_roofline(e1):
    ms=list(e1["models"].keys())
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    xs=np.linspace(0.2,2.2,100)  # model GB
    ax.plot(xs, PEAK/xs, 'k--', lw=1, label=f"roofline: BW$_\\mathrm{{peak}}$/bytes ({PEAK} GB/s)")
    for m in ms:
        thr=e1["models"][m]["threads"]
        d=max((r for r in thr.values() if r["tg_ts"]), key=lambda r:r["tg_ts"])  # peak decode
        gb=e1["models"][m]["file_mb"]/1000
        if d["tg_ts"]:
            ax.scatter(gb, d["tg_ts"], s=60, zorder=5)
            ax.annotate(f"{NAME.get(m,m)}\n{d['tg_ts']:.1f} tok/s\n({d['bw_util_pct']:.0f}% BW)",
                        (gb,d["tg_ts"]), fontsize=7, xytext=(6,0), textcoords="offset points", va="center")
    ax.set_xlabel("Model size (GB, Q4_K_M)"); ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Decode is bandwidth-bound: throughput $=$ BW / model bytes")
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
    if e1: fig_roofline(e1); fig_threads(e1); print("fig1,2 ok")
    if e2: fig_quant(e2); print("fig3 ok")
    if e3: fig_energy(e3); print("fig4 ok")
    if e4: fig_capacity(e4); print("fig5 ok")
    if e5: fig_dvfs(e5); print("fig6 ok")
    print("FIGS DONE")

if __name__=="__main__": main()
