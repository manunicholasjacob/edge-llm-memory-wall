# Paper 12 — The Memory Wall at the Edge of Language

**Original title:** The Memory Wall at the Edge of Language: Characterizing and Optimizing LLM Inference on a 2 GB CPU
**Author:** Manu Nicholas Jacob
**Current title:** One Law, Two Memory Systems: A Transferable Decode Roofline for Edge LLM Inference
**Venue:** under review at the IEEE Internet of Things Journal, `IoT-75150-2026`, submitted 22 September 2026.
Previously `TC-2026-08-0905` at IEEE Transactions on Computers, rejected 8 September 2026; TC bars
resubmission there, including of modified versions.
**Platform:** Raspberry Pi 5 (quad-core ARM Cortex-A76, 2 GB LPDDR4X), llama.cpp CPU backend
**Public artifact:** https://github.com/manunicholasjacob/edge-llm-memory-wall

## Summary
LLM decode reads the entire model from DRAM once per token, so on a memory-constrained edge CPU it
is bandwidth-bound — the memory wall governs it even more strictly than it governs CNNs. This paper
characterizes prefill vs. decode, builds a size-only decode roofline, measures energy per token,
maps the KV-cache capacity wall on 2 GB, and derives a configuration policy for a throughput target.

## Headline results (all hardware-measured)
- **Decode roofline:** tok/s = effective bandwidth / model_bytes: 10.7 GB/s, **R^2 = 0.98** over seven
  configurations (R^2 = 0.994 for the three-size sweep), sustaining **73 to 86% of peak DRAM bandwidth**.
  Prefill is compute-bound and scales ~4x with cores; decode saturates at 2 to 3 threads.
- **Quantization frontier:** Q2→Q8 = 31→19 tok/s and 339→531 MB (faster + smaller together).
- **Energy/token:** 253/453/576 mJ (0.5/1/1.5 B); DRAM rail only 4–5% → **core-stall energy**.
- **Capacity wall:** 0.5 B → 16 K ctx, 1 B/1.5 B → 4–8 K on 2 GB.
- **DVFS:** 1.5 to 2.4 GHz lifts 0.5 B decode by ~18% (22.67 to 26.74 tok/s, timing only). No energy
  figure is given for 1.5 GHz: a later control experiment on the same board found the power monitor
  unreliable at that clock, so the earlier "~24% more energy at max clock" is withdrawn (13 Oct 2026).
- **Policy:** run at the lowest clock (and quant/threads) that meets the throughput target; the energy it
  saves must be measured with a power monitor validated at low clocks. The earlier "up to 23%" is withdrawn.

## Contents
- `paper/` — LaTeX (IEEEtran compsoc), refs.bib, compiled `main.pdf`, cover letter
- `scripts/` — L1–L5 measurement + analysis + figure generation
- `results/` — raw measurement JSONs + STATS
- `LAB_NOTEBOOK.md` — full campaign log incl. the memory/OOM battles on the 2 GB box
