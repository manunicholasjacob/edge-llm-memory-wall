# The Memory Wall at the Edge of Language — LLM Inference on a 2 GB CPU

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21844855.svg)](https://doi.org/10.5281/zenodo.21844855)

Artifact for the study that is now the manuscript **"One Law, Two Memory Systems: A Transferable
Decode Roofline for Edge LLM Inference"**.

## Status, 27 September 2026

The manuscript is **under review at the IEEE Internet of Things Journal**, `IoT-75150-2026`,
submitted 22 September 2026.

It was previously submitted to *IEEE Transactions on Computers* as `TC-2026-08-0905` under the
earlier title "The Memory Wall at the Edge of Language: Characterizing and Optimizing LLM Inference
on a 2 GB CPU". **TC rejected it on 8 September 2026** and its policy bars resubmission there,
including of modified versions. Earlier versions of this README described the work as targeting
TC; that is no longer true and the line has been corrected rather than quietly deleted.

The repository keeps its own title and DOI, which were minted against the artifact rather than
against the paper, so `CITATION.cff` and the Zenodo record are unchanged.

LLMs are moving onto edge devices, but the tools we use to reason about their cost were built for
CNNs and datacenter GPUs. This work shows that on a memory-constrained edge CPU the **memory wall**
is the organizing principle — and it governs LLM inference *even more strictly* than it governs
CNNs. All measurements are on a **Raspberry Pi 5** (quad-core ARM Cortex-A76, **2 GB** LPDDR4X)
running `llama.cpp`.

## Headline findings (hardware-measured)

| # | Finding |
|---|---------|
| 1 | **Decode is bandwidth-bound**: throughput ≈ effective bandwidth ÷ model_bytes. The size sweep fits at **R² = 0.994** and all seven measured configurations at **R² = 0.98**; decode sustains **73–86% of peak DRAM bandwidth**. |
| 2 | **Prefill scales ~4× with cores; decode saturates at 2–3 threads** — opposite ends of the roofline (prefill is 3–5× faster). |
| 3 | **Quantization is the master lever**: Q2→Q8 spans 31→19 tok/s and 339→531 MB — fewer bits = faster decode *and* smaller footprint together. |
| 4 | **Energy per token** scales with model size (253/453/576 mJ for 0.5/1/1.5 B) and falls with fewer bits; the **DRAM rail is only 4–5% of power** — the memory-wall energy is *core stalls*, not DRAM-interface power. |
| 5 | **KV-cache capacity wall**: on 2 GB the 0.5 B model reaches 16 K context, 1 B/1.5 B hit the wall by 4–8 K — model size trades against context length. |
| 6 | **Energy-optimal decode clock = 1.5 GHz (minimum)**: max-clock costs ~24% more energy for ~18% throughput. |
| 7 | **Energy-optimal configuration policy** (quant/threads/clock for a throughput SLO) saves up to **23% energy/token** vs always-max-clock, at no throughput cost. |
| 8 | **The law transfers between CPUs**: an x86 part with 3.3× the bandwidth fits the same one-line roofline, R² = 0.98 on both, with only the constant changing (10.7 GB/s effective on the Pi against 35.7 GB/s on x86). |

### Two corrections to earlier versions of this table

**The bandwidth figures changed when the fit did.** An earlier README quoted 11.34 GB/s and a
80 to 86% band. Those came from the three-model size sweep. The manuscript now fits seven
configurations, three parameter sizes at Q4_K_M plus five quantizations of the 0.5 B model, and
reports 73 to 86%. The numbers above are the manuscript's.

**"Transfers between CPUs" is not "platform-independent."** Finding 8 is measured on two CPUs and
it holds on both. It is not a claim about all hardware. Separate work by the same author, across
six memory systems from 17 to 1555 GB/s, finds that the through-origin form of this law describes
a regime rather than a machine: it fits well wherever every model tested is larger than the point
at which fixed per-token cost equals memory time, and degrades badly where it is not. Read
Finding 8 as a CPU result.

## Repository layout

```
scripts/
  llm_common.py     harness (llama.cpp wrappers, PMIC power, DVFS, clean decode-energy)
  L1_roofline.py    prefill/decode roofline + thread scaling
  L2_quant.py       quantization frontier (Q2..Q8)
  L3_energy.py      energy per token (PMIC)
  L4_capacity.py    KV-cache capacity wall
  L5_dvfs.py        energy-optimal decode frequency
  analysis.py       roofline predictive model + energy-optimal policy
  fig_gen.py        regenerates all figures
results/            raw measurement JSONs + STATS.txt
paper/              LaTeX (IEEEtran compsoc), refs.bib, compiled PDF, cover letter
```

## Reproduce

On a Raspberry Pi 5 with `llama.cpp` (built `-DGGML_NATIVE=ON`; the CPU backend uses the A76
`asimddp` dot-product kernels at runtime) and GGUF models in `~/llm/models/`, run each `LN_*.py`,
then pull `results/*.json` and run `analysis.py` and `fig_gen.py` locally. `sudo` is needed for
`drop_caches` and DVFS; the scripts pin the clock and control the governor.

## Platform

Raspberry Pi 5 (Broadcom BCM2712, quad-core Cortex-A76 @ up to 2.4 GHz, 2 GB LPDDR4X), Debian 12 /
Linux 6.12, `llama.cpp` (CPU backend). Peak DRAM read 13.98 GB/s and compute ceiling 102.8 GFLOP/s
measured in the companion memory-wall study. Energy via `vcgencmd pmic_read_adc`.

## Author

Manu Nicholas Jacob. Part of an edge-AI measurement portfolio on the Pi 5; extends the
memory-wall roofline from CNNs (*The Memory Wall Governs Edge DNN Inference*) to generative LLM
inference, with companion energy and cold-start studies.

## Archived version

This artifact is archived on Zenodo. The concept DOI
[10.5281/zenodo.21844855](https://doi.org/10.5281/zenodo.21844855)
always resolves to the latest release, and `CITATION.cff` carries the full metadata,
which is what GitHub's "Cite this repository" button renders.
