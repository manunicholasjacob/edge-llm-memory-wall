"""Cross-platform roofline: fit the decode roofline (tok/s = BW_eff/bytes) independently on the
Raspberry Pi 5 (Cortex-A76) and an x86 CPU (i7-12700H), and plot both. The roofline SHAPE holding
on both platforms -- with an effective bandwidth that tracks each platform's DRAM bandwidth --
shows the memory wall governs edge LLM decode regardless of ISA."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
FIG=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","paper","figs")
PC_JSON=r"C:\llmpc\pc_roofline.json"
def load(n):
    p=os.path.join(RES,n); return json.load(open(p)) if os.path.exists(p) else None
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})

def pi_points():
    e1=load("L1_roofline.json"); e2=load("L2_quant.json"); pts=[]
    def peak(m): return max(r["tg_ts"] for r in e1["models"][m]["threads"].values() if r["tg_ts"])
    for m in e1["models"]: pts.append((e1["models"][m]["file_mb"]/1000, peak(m)))
    sn=set(e1["models"])
    for m in e2["models"]:
        if m not in sn: pts.append((e2["models"][m]["file_mb"]/1000, e2["models"][m]["tg_ts"]))
    return pts

def pc_points():
    if not os.path.exists(PC_JSON): return [], None
    d=json.load(open(PC_JSON)); pts=[(d["models"][m]["file_mb"]/1000, d["models"][m]["decode_ts"]) for m in d["models"]]
    return pts, d.get("peak_read_GBs")

def fit(pts):
    b=np.array([p[0] for p in pts]); t=np.array([p[1] for p in pts]); x=1.0/(b*1e9)
    BW=float(np.sum(x*t)/np.sum(x*x)); r2=1-np.sum((t-BW*x)**2)/np.sum((t-t.mean())**2)
    return BW/1e9, r2

def main():
    pi=pi_points(); pc,pc_peak=pc_points()
    bw_pi,r2_pi=fit(pi)
    print(f"[Pi5 A76]  BW_eff={bw_pi:.2f} GB/s  R^2={r2_pi:.3f}  ({len(pi)} pts, peak 13.98 GB/s, util {100*bw_pi/13.98:.0f}%)")
    if pc:
        bw_pc,r2_pc=fit(pc)
        print(f"[x86 12700H] BW_eff={bw_pc:.2f} GB/s  R^2={r2_pc:.3f}  ({len(pc)} pts, peak {pc_peak} GB/s, util {100*bw_pc/pc_peak:.0f}%)")
        print(f"[cross] x86 decode is {bw_pc/bw_pi:.1f}x the Pi's effective bandwidth; both follow tok/s=BW/bytes")
    # figure
    fig,ax=plt.subplots(figsize=(6.6,4.0))
    xs=np.linspace(0.25,5.0,200)
    ax.scatter([p[0] for p in pi],[p[1] for p in pi],s=60,color="#2166ac",zorder=5,label=f"Raspberry Pi 5 (A76)")
    ax.plot(xs,bw_pi/xs,'--',color="#2166ac",lw=1,label=f"  {bw_pi:.1f} GB/s / bytes ($R^2$={r2_pi:.3f})")
    if pc:
        ax.scatter([p[0] for p in pc],[p[1] for p in pc],s=60,color="#b2182b",marker="D",zorder=5,label="x86 i7-12700H")
        ax.plot(xs,bw_pc/xs,'--',color="#b2182b",lw=1,label=f"  {bw_pc:.1f} GB/s / bytes ($R^2$={r2_pc:.3f})")
    ax.set_xlabel("Model size (GB)"); ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("The decode roofline is platform-independent")
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(os.path.join(FIG,"fig9_crossplatform.png")); plt.close(fig)
    print("wrote fig9_crossplatform.png")
    json.dump({"pi":{"bw":bw_pi,"r2":r2_pi},"pc":({"bw":bw_pc,"r2":r2_pc,"peak":pc_peak} if pc else None)},
              open(os.path.join(RES,"cross_platform.json"),"w"),indent=2)

if __name__=="__main__": main()
