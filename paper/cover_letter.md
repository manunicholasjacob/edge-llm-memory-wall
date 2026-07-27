# Cover Letter — IEEE Transactions on Computers

**Manuscript:** *The Memory Wall at the Edge of Language: Characterizing and Optimizing LLM Inference on a 2 GB CPU*
**Author:** Manu Nicholas Jacob

Dear Editors,

Please consider the attached manuscript for publication in *IEEE Transactions on Computers*.

Large language models are moving onto edge and embedded devices, but the frameworks engineers use
to reason about their cost are inherited from convolutional-network inference on mobile SoCs and
from LLM serving on datacenter GPUs — neither of which describes a single-user LLM on a
memory-constrained CPU. This paper characterizes that regime on a Raspberry Pi 5 (ARM Cortex-A76,
2 GB) and shows that the **memory wall** is its organizing principle.

**Contributions:**
1. A **size-only decode roofline**: autoregressive decode reads the whole model per token, so its
   throughput equals DRAM bandwidth divided by model bytes — predicting measured throughput across
   models with **R² = 0.994** at **80–86% of peak DRAM bandwidth**. Prefill, by contrast, is
   compute-bound and core-scalable.
2. The **energy of a token** from on-board PMIC telemetry: energy scales with model size, and the
   DRAM rail is only 4–5% of power — the memory-wall energy is paid as core stalls, a result that
   mirrors our earlier CNN findings and is counter-intuitive.
3. The **KV-cache capacity wall**: on 2 GB, usable context is bounded by weights plus KV cache, and
   quantization is the single lever that relaxes both the bandwidth and the capacity wall.
4. A deployable **energy-optimal configuration policy** (quantization, threads, clock for a
   throughput SLO), demonstrated on the device, saving up to 23% energy per token.

The work is a rigorous, energy-instrumented, roofline-grounded characterization of an increasingly
important workload on commodity hardware, with an actionable operating policy, and is fully
reproducible from an open artifact. It extends a validated architectural model (the roofline) to a
new and topical workload, which we believe fits the scope and readership of TC.

This manuscript is original, has not been published previously, and is not under consideration
elsewhere. The author declares no conflicts of interest.

Sincerely,
Manu Nicholas Jacob
