# Lab Notebook — The Memory Wall at the Edge of Language (Paper 12)

Platform: Raspberry Pi 5, 2 GB, Cortex-A76 (asimddp), Debian 12 / Linux 6.12. Engine:
`llama.cpp` (CPU backend, native ARM build). Models (GGUF): Qwen2.5-0.5B, Llama-3.2-1B,
Qwen2.5-1.5B at Q4_K_M (size sweep); Qwen2.5-0.5B at Q2_K/Q3_K_M/Q4_K_M/Q5_K_M/Q8_0 (quant sweep).
Peak DRAM read 13.98 GB/s (from the memory-wall study). Clock pinned 2.4 GHz except in L5.

## Experiments
- **L1 roofline + threads** (`llama-bench`). Decode tok/s = 11.34 GB/s / model_bytes, R²=0.994,
  80–86% of peak BW at optimal threads. Prefill scales ~4× with cores; decode peaks at 2–3 threads.
- **L2 quantization** (`llama-bench`). Q2→Q8: 31.1→19.1 tok/s across 339→531 MB (1.63× speedup);
  decode ∝ 1/size. Prefill (compute-bound) varies non-monotonically with quant kernel efficiency.
- **L3 energy/token** (PMIC). 253/453/576 mJ (0.5/1/1.5 B), 208–297 mJ across Q2–Q8; DRAM rail
  only 4–5% → core-stall energy (matches the CNN energy study).
- **L4 capacity wall.** 0.5 B reaches 16 K ctx; 1 B/1.5 B hit ~4–8 K on 2 GB (weights + KV). Prefill
  throughput also drops with context (attention cost).
- **L5 DVFS.** Energy-optimal decode clock = 1.5 GHz (min): 1.5→2.4 GHz gives +18% tok/s for +31%
  energy/token, so max-clock wastes ~24% energy.
- **Analysis.** Size-only decode roofline (R²=0.994); energy-optimal clock/config policy (up to 23%
  energy saved at a throughput SLO, 0.5 B).

## TC expansion (2026-07-27)
- **Roofline strengthened to 7 points**: the 3-size sweep PLUS the 5 quantized 0.5B models all
  lie on decode = BW_eff/bytes (R²=0.980; size-only R²=0.994). Byte reductions from fewer params
  or fewer bits are equivalent on a bandwidth-bound loop. No new models needed.
- **L9 perplexity** (quality Pareto): Q2=22.8, Q3=20.2, Q4=19.7, Q5=19.5, Q8=19.3 — Q4 recovers
  near-Q8 quality at 75% footprint; Q2 pays a steep penalty. Corpus: Project Gutenberg English prose
  (wikitext URL was dead). Completes the speed/size/quality trade-off.
- **L8 KV-quant → honest null**: q8_0 KV barely extends context for 0.5–1.5B models on 2GB because
  WEIGHTS dominate the capacity budget, not KV. (Also: llama.cpp needs `-fa` for quantized KV; the
  -fa re-measure was too slow — 9min/16K-prefill — so reported as the weight-dominated finding.)
- **L7 batch DROPPED**: llama-batched-bench build needs a disk-heavy full recompile that the
  100%-full disk wouldn't allow; the batching mechanism is already argued analytically (single-user
  edge can't batch → stuck on the BW slope, per Pope et al.).
- Disk hell: a stray 620MB partial SmolLM2 download (missed by an earlier rm on a dropped conn)
  filled the disk to 0 and silently broke a whole phase-3 run; deleting it fixed it. Lesson: verify
  deletes landed, and `df` before every run on this box.

## Build & ops battles (the 2 GB Pi is right at the edge — fitting for the thesis)
- **Build OOM:** `cmake --build -j3` at -O3 exhausts 2 GB compiling large files (server.cpp) →
  swap-death, Pi unreachable. Fix: **-j1** (single-file compilation), target only
  llama-bench/llama-cli/llama-perplexity. No power-cycle needed once builds serialized.
- **Disk:** apt/pip caches were hiding 3.5 GB; `apt-get clean` + drop pip cache freed it. apt list
  file was corrupted (from an earlier disk-full) → `rm -rf /var/lib/apt/lists/*; apt-get update`.
  cmake wasn't installed; installed via apt.
- **llama-cli hang:** `llama-cli … -no-cnv` entered interactive mode and hung 77 min on one call.
  Fix: use `llama-bench` (non-interactive) for all decode measurements.
- **`-p 0` anomaly:** `llama-bench -p 0` reports a bogus low tg (~2 tok/s); use `-p 128` and read
  the separately-reported `tg_ts`.
- **PMIC sampler contention:** a 50 Hz `vcgencmd` sampler steals CPU from the 4-thread workload and
  slowed decode ~6×. Fix: measure throughput unperturbed, power in a separate run at 10 Hz.
- **Swap-thrash from leftovers:** processes from killed runs (a hung llama-cli, a 976 MB
  32K-context bench) filled RAM → weights pushed to swap → decode 13× slower (streaming from SD).
  Fix: kill leftovers by PID, `swapoff -a; swapon -a`, drop caches; runner clears swap first and
  caps context at 16 K. Lesson: on a 2 GB box, always kill cleanly and watch MemAvailable.
