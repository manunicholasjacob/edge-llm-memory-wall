"""
llm_common.py — harness for the Edge LLM memory-wall campaign (Raspberry Pi 5 / Cortex-A76).
Wraps llama.cpp (llama-bench, llama-cli) with PMIC power sampling and DVFS control.
"""
import os, sys, time, json, subprocess, threading, re
import numpy as np

BIN = os.path.expanduser("~/llm/llama.cpp/build/bin")
MODELS = os.path.expanduser("~/llm/models")

# name -> (file, params_B, quant). Filled as models download.
ZOO = {
    "qwen0.5b-q4km": ("qwen0.5b-q4km.gguf", 0.5, "Q4_K_M"),
    "llama1b-q4km":  ("llama1b-q4km.gguf",  1.24, "Q4_K_M"),
    "qwen1.5b-q4km": ("qwen1.5b-q4km.gguf", 1.5, "Q4_K_M"),
    "qwen0.5b-q2k":  ("qwen0.5b-q2k.gguf",  0.5, "Q2_K"),
    "qwen0.5b-q3km": ("qwen0.5b-q3km.gguf", 0.5, "Q3_K_M"),
    "qwen0.5b-q5km": ("qwen0.5b-q5km.gguf", 0.5, "Q5_K_M"),
    "qwen0.5b-q8":   ("qwen0.5b-q8.gguf",   0.5, "Q8_0"),
}
def mpath(name): return os.path.join(MODELS, ZOO[name][0])
def msize_bytes(name): return os.path.getsize(mpath(name))

# ---------- privileged (NOPASSWD sudo) ----------
def set_governor(gov):
    subprocess.run(["sudo","sh","-c",
        f"for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {gov} > $c; done"],check=True)
def set_freq_khz(khz):
    set_governor("userspace")
    subprocess.run(["sudo","sh","-c",
        f"for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed; do echo {khz} > $c; done"],check=True)
def cur_freqs():
    return [int(x) for x in subprocess.check_output(
        ["sh","-c","cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"]).decode().split()]

# ---------- PMIC power sampler ----------
class PowerSampler:
    def __init__(self, period=0.02):
        self.period=period; self._stop=threading.Event(); self.samples=[]; self._thr=None
    @staticmethod
    def _read():
        out=subprocess.check_output(["vcgencmd","pmic_read_adc"]).decode()
        v,a={},{}
        for ln in out.strip().splitlines():
            ln=ln.strip()
            if "volt(" in ln: v[ln.split()[0][:-2]]=float(ln.split("=")[1].rstrip("V"))
            elif "current(" in ln: a[ln.split()[0][:-2]]=float(ln.split("=")[1].rstrip("A"))
        tot=core=ddr=0.0
        for k in v:
            if k in a:
                p=v[k]*a[k]; tot+=p
                if k.startswith("VDD_CORE"): core+=p
                elif k.startswith("DDR"): ddr+=p
        return tot,core,ddr
    def _loop(self):
        while not self._stop.is_set():
            t=time.perf_counter()
            try: tot,core,ddr=self._read(); self.samples.append((t,tot,core,ddr))
            except Exception: pass
            time.sleep(self.period)
    def start(self): self.samples=[]; self._stop.clear(); self._thr=threading.Thread(target=self._loop,daemon=True); self._thr.start()
    def stop(self):
        self._stop.set()
        if self._thr: self._thr.join(timeout=2)
    def mean_power(self, t0, t1, rail=0):
        vals=[s[1+rail] for s in self.samples if t0<=s[0]<=t1]
        return (float(np.mean(vals)), len(vals)) if vals else (None,0)
    def energy(self, t0, t1):
        pts=[s for s in self.samples if t0<=s[0]<=t1]
        if len(pts)<2: return None
        E=Ec=Ed=0.0
        for i in range(1,len(pts)):
            dt=pts[i][0]-pts[i-1][0]
            E+=0.5*(pts[i][1]+pts[i-1][1])*dt; Ec+=0.5*(pts[i][2]+pts[i-1][2])*dt; Ed+=0.5*(pts[i][3]+pts[i-1][3])*dt
        return {"E_J":E,"E_core_J":Ec,"E_ddr_J":Ed,"dur":pts[-1][0]-pts[0][0]}

# ---------- llama.cpp wrappers ----------
def run_bench(name, p=512, n=128, threads=4, reps=3, ctx=None):
    """Return {'pp_ts':..,'tg_ts':..} tokens/s via llama-bench JSON."""
    cmd=[os.path.join(BIN,"llama-bench"),"-m",mpath(name),"-t",str(threads),"-r",str(reps),"-o","json"]
    if p is not None: cmd += ["-p",str(p)]
    if n is not None: cmd += ["-n",str(n)]
    if ctx: cmd += ["-c",str(ctx)]
    out=subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
    data=json.loads(out)
    res={"pp_ts":None,"tg_ts":None}
    for row in data:
        ts=row.get("avg_ts")
        if row.get("n_prompt",0)>0 and row.get("n_gen",0)==0: res["pp_ts"]=ts
        elif row.get("n_gen",0)>0 and row.get("n_prompt",0)==0: res["tg_ts"]=ts
    return res

def bench_with_power(name, p, n, threads, reps, ps, ctx=None):
    """Run llama-bench (non-interactive) while sampling PMIC. Returns (pp_ts, tg_ts, P_mean_W,
    P_ddr_W, wall_s, ok). Power is averaged over the active window (skipping model load)."""
    cmd=[os.path.join(BIN,"llama-bench"),"-m",mpath(name),"-t",str(threads),"-r",str(reps),"-o","json"]
    if p is not None: cmd += ["-p",str(p)]
    if n is not None: cmd += ["-n",str(n)]
    if ctx: cmd += ["-c",str(ctx)]
    t0=time.perf_counter()
    r=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    t1=time.perf_counter()
    if r.returncode!=0:
        return None,None,None,None,t1-t0,False
    try:
        data=json.loads(r.stdout.decode())
    except Exception:
        return None,None,None,None,t1-t0,False
    pp=tg=None
    for row in data:
        if row.get("n_prompt",0)>0 and row.get("n_gen",0)==0: pp=row.get("avg_ts")
        elif row.get("n_gen",0)>0 and row.get("n_prompt",0)==0: tg=row.get("avg_ts")
    P,_=ps.mean_power(t0+1.5, t1-0.1, 0) if ps else (None,0)
    Pddr,_=ps.mean_power(t0+1.5, t1-0.1, 2) if ps else (None,0)
    return pp, tg, P, Pddr, (t1-t0), True

def decode_energy(name, threads=4, n=256, reps=2, period=0.1):
    """Clean energy-per-token: throughput measured WITHOUT the sampler (unperturbed), then power
    measured in a separate run with a low-rate (10 Hz) sampler to minimize CPU contention.
    Returns {tg, P, Pcore, Pddr}. E/token = P / tg."""
    # NOTE: llama-bench '-p 0' yields an anomalous low tg; use a small real prompt so tg_ts is the
    # true decode rate (reported separately from prefill).
    tg = run_bench(name, p=128, n=n, threads=threads, reps=reps).get("tg_ts")
    ps = PowerSampler(period); ps.start(); time.sleep(0.3)
    t0 = time.perf_counter()
    subprocess.run([os.path.join(BIN,"llama-bench"),"-m",mpath(name),"-p","128","-n",str(n),
                    "-t",str(threads),"-r",str(reps),"-o","json"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    t1 = time.perf_counter()
    ps.stop()
    # skip first ~4s (load + 128-token prefill); the remainder is decode-dominated
    P,_=ps.mean_power(t0+4.0,t1-0.1,0); Pc,_=ps.mean_power(t0+4.0,t1-0.1,1); Pd,_=ps.mean_power(t0+4.0,t1-0.1,2)
    return {"tg":tg,"P":P,"Pcore":Pc,"Pddr":Pd}

def now(): return time.perf_counter()
