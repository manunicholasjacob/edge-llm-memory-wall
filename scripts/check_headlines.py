"""Recompute the README headline numbers from the released data in results/ and fail on drift.

Run after scripts/analysis.py. Uses the same fitting code as the analysis scripts, so a change to
either the data or the fit that moves a published number makes this exit non-zero.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402
import cross_platform  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FAIL = []


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def check(label, got, want, ndigits):
    ok = round(got, ndigits) == round(want, ndigits)
    print(f"{'ok  ' if ok else 'FAIL'} {label}: recomputed {got:.{max(ndigits, 0) + 3}f}, published {want}")
    if not ok:
        FAIL.append(label)


e1, e2, e3 = load("L1_roofline.json"), load("L2_quant.json"), load("L3_energy.json")

# Finding 1: one roofline over all seven Pi 5 configurations, and the three-size sweep alone.
r = analysis.roofline_model(e1, e2)
check("F1 seven-config fit R^2", r["r2"], 0.98, 2)
check("F1 effective bandwidth GB/s", r["BW_eff_GBs"], 10.69, 2)
check("F1 utilization low %", r["util_lo"], 73, 0)
check("F1 utilization high %", r["util_hi"], 86, 0)
check("F1 config count", r["n_points"], 7, 0)
size_only = analysis.roofline_model(e1, None)
check("F1 size-sweep R^2", size_only["r2"], 0.994, 3)

# Finding 3: quantization frontier on the 0.5 B model.
tg = [m["tg_ts"] for m in e2["models"].values()]
mb = [m["file_mb"] for m in e2["models"].values()]
check("F3 slowest decode tok/s", min(tg), 19.1, 1)
check("F3 fastest decode tok/s", max(tg), 31.1, 1)
check("F3 smallest file MB", min(mb), 339, 0)
check("F3 largest file MB", max(mb), 531, 0)

# Finding 4: energy per token = decode power / decode rate; DRAM rail share = DDR power / decode power.
m3 = e3["models"]
for name, want in [("qwen0.5b-q4km", 253), ("llama1b-q4km", 453), ("qwen1.5b-q4km", 576)]:
    check(f"F4 energy per token mJ, {name}", 1000 * m3[name]["P_decode_W"] / m3[name]["decode_ts"], want, 0)
shares = [100 * v["P_ddr_W"] / v["P_decode_W"] for v in m3.values()]
check("F4 DRAM rail share low %", min(shares), 4, 0)
check("F4 DRAM rail share high %", max(shares), 5, 0)

# Finding 6: decode throughput lift from 1.5 to 2.4 GHz on the 0.5 B model. Timing only: the 1500 MHz
# PMIC power readings are withdrawn, so no energy number is checked or published for this finding.
f5 = load("L5_dvfs.json")["models"]["qwen0.5b-q4km"]["freqs"]
lo, hi = f5["1500000"]["decode_ts"], f5["2400000"]["decode_ts"]
check("F6 decode tok/s at 1.5 GHz", lo, 22.67, 2)
check("F6 decode tok/s at 2.4 GHz", hi, 26.74, 2)
check("F6 decode lift 1.5 to 2.4 GHz %", 100 * (hi / lo - 1), 18, 0)

# Finding 8: the same one-line roofline on the x86 laptop CPU.
pc = load("pc_roofline.json")
pc_pts = [(v["file_mb"] / 1000, v["decode_ts"]) for v in pc["models"].values()]
bw_pc, r2_pc = cross_platform.fit(pc_pts)
bw_pi, r2_pi = cross_platform.fit(cross_platform.pi_points())
check("F8 Pi effective bandwidth GB/s", bw_pi, 10.7, 1)
check("F8 x86 effective bandwidth GB/s", bw_pc, 35.7, 1)
check("F8 Pi R^2", r2_pi, 0.98, 2)
check("F8 x86 R^2", r2_pc, 0.98, 2)
check("F8 x86 / Pi effective bandwidth", bw_pc / bw_pi, 3.3, 1)

if FAIL:
    print(f"\n{len(FAIL)} headline number(s) drifted: {', '.join(FAIL)}")
    sys.exit(1)
print("\nall headline numbers reproduce from results/")
