# LLM Inference Optimization — Llama 3.2 on RTX 3060

Real inference engineering on consumer hardware. Not a tutorial — actual measurements, unexpected findings, and documented bottlenecks on a single RTX 3060 (SM86, Ampere, 12GB VRAM).

---

## What This Is

I benchmarked Llama 3.2 (1B and 3B) across the full inference stack — from naive PyTorch baseline to production-style vLLM serving with Locust load testing and Nsight Systems profiling. Every number here was measured, not estimated.

**Hardware:** RTX 3060 | 360 GB/s HBM | 12GB VRAM | SM86 (Ampere)  
**Models:** Llama 3.2 1B, 3B, 3B-Instruct  
**Tools:** PyTorch, vLLM, Locust, NVIDIA Nsight Systems, Nsight Compute (NCU)

---

## The Core Finding

Going from PyTorch baseline to vLLM isn't just a framework swap. It's a fundamentally different execution model:

| What Changed | PyTorch | vLLM | Why |
|---|---|---|---|
| Throughput (1B, BS=1) | 29 tok/s | 101 tok/s | CUDA graphs eliminate CPU overhead |
| Throughput (1B, BS=16) | 379 tok/s | 1222 tok/s | Continuous batching saturates GPU |
| VRAM behavior | Dynamic (2.39→3.71GB) | Flat (9GB always) | PagedAttention pre-allocates pool |
| GPU utilization | 35% | ~80%+ | No idle time between requests |

The flat VRAM line in vLLM isn't waste — it's PagedAttention reserving a fixed pool of KV cache blocks upfront. The dynamic PyTorch allocation looks more "efficient" but actually wastes memory through fragmentation.

---

## Tail Latency Results (vLLM, Llama 3.2 3B, FP16)

This is the part most benchmarks skip. Throughput numbers look great. What actually matters in production is whether your P99 holds up.

| Batch Size | Throughput | TTFT | Avg TTOPT | P90 TTOPT | P99 TTOPT |
|---|---|---|---|---|---|
| 1 | 43 tok/s | 192ms | 22.5ms | 23.5ms | 24.2ms |
| 2 | 75 tok/s | 193ms | 26.1ms | 26.6ms | 27.9ms |
| 4 | 147 tok/s | 297ms | 26.1ms | 26.7ms | 27.3ms |
| 8 | 272 tok/s | 532ms | 27.4ms | 28.6ms | 29.7ms |
| **16** | **639 tok/s** | 1057ms | **21.0ms** | 30.2ms | **31.1ms** |

**The interesting thing at BS=16:** TTOPT actually *dropped* from 22.5ms to 21.0ms while throughput went up 15×. This is the batch size 16 "free lunch" — at BS=1 the GPU finishes the computation and sits idle waiting for the CPU. At BS=16, it's running 16 sequences in parallel for the same memory bandwidth cost as 1.

The P99 gap is only 10ms from average. That narrow distribution proves the memory management isn't causing surprise stalls — which is exactly what you want to see before trusting a system with real traffic.

---

## Hardware Ceiling Analysis

Before running any optimization, I calculated the theoretical ceiling so I'd know when I was done.

```
RTX 3060 HBM bandwidth: 360 GB/s

Llama 3.2 3B (FP16, 6.42GB):
  Theoretical max throughput: 360/6.42 = ~56 tok/s per sequence
  Minimum latency per token:  (6.42/360) × 1000 = 17.8ms

My measured TTOPT at BS=16: 21.0ms
Model Bandwidth Utilization (MBU): (17.8/21.0) × 100 = 84.9%

The 3B model is operating at 85% of the theoretical hardware ceiling.
Further software optimization has diminishing returns here.
```

The 1B model tells a different story — MBU sits at ~26% at BS=1. The bottleneck there isn't memory bandwidth, it's CPU framework overhead. The GPU finishes the math and waits for Python to send the next command.

---

## Base Model vs Instruct: Does Fine-Tuning Change Inference?

I ran Llama 3.2 3B and 3B-Instruct through identical benchmarks to isolate the effect of instruction tuning on inference performance.

**Short answer:** Same hardware cost, different behavior.

- Identical VRAM footprint (same weights, different formatting)
- TTFT is slightly higher for Instruct because the chat template adds tokens to every request
- The Instruct model follows stop tokens more reliably, which actually shortens outputs in practice

The architecture is identical — GQA with 8 KV heads, SwiGLU activations, RoPE, 128K context window. The difference is entirely in the prompt format and training objective, not the compute graph.

---

## Nsight Systems Teardown

I profiled the full vLLM lifecycle on Llama 3.2 3B to understand what actually happens between "start the server" and "first token".

**Phase 1 — Weight Loading (0s to ~0.85s)**  
~6.4GB of FP16 weights streaming from disk through CPU RAM across PCIe to VRAM. SMs are completely idle during this. Running in WSL2 forces `pin_memory=False`, adding an extra staging buffer copy. Nothing to optimize here — it's PCIe bandwidth.

**Phase 2 — Warmup & KV Cache Allocation (0.85s to ~12s)**  
vLLM runs a mock forward pass to measure how much VRAM is left after weights load. Then `fill_reverse_indices_kernel` runs for ~1 second to build the PagedAttention block tables. This is a one-time cost — mandatory engineering overhead that makes all subsequent requests fast.

**Phase 3 — CUDA Graph Capture (~11.2s to ~12s)**  
The perfectly repeating blue blocks in the Nsight timeline are vLLM capturing CUDA graphs for every possible batch size. This is why the first request takes longer — after capture, every decode step runs from a pre-compiled graph with zero CPU launch overhead.

**Phase 4 — Prefill (compute-bound)**  
Large matrix multiplications across all prompt tokens simultaneously. SM throughput is high here. MFU (Model FLOP Utilization) during prefill: **24.88 TFLOPS / 98.3 peak = 25.3%**. The remaining gap is framework overhead and memory access latency.

**Phase 5 — Decode (memory-bandwidth-bound)**  
One token per forward pass. Each step loads 6.4GB of weights from HBM. At 360 GB/s that's a hard floor of 17.8ms per token. My measured 21ms = 85% bandwidth utilization. The remaining 15% is scheduling and kernel launch overhead.

---

## Why the Shifting Bottleneck Matters

At **BS=1 with the 1B model**: CPU framework overhead dominates. The GPU finishes in microseconds and waits. MBU is 26%. Adding more compute won't help — you need to reduce driver calls.

At **BS=16 with the 3B model**: HBM bandwidth dominates. The memory bus is saturated at 85% moving KV cache weights. MBU is 85%. Adding more memory bandwidth (H100 has 3.35 TB/s) would directly translate to proportionally higher throughput.

Same GPU. Same framework. Different model size. Completely different bottleneck.

---

## Repo Structure

```
llm-inference-benchmarks/
├── pytorch/
│   └── benchmark.py          baseline inference scripts
├── vllm/
│   └── vllm_benchmark.py     vLLM serving benchmarks
├── analysis/
│   ├── tail_latencies.xlsx   P90/P99 data across batch sizes
│   ├── bottleneck_analysis.md shifting bottleneck findings
│   └── nsight_analysis.md    Nsight Systems phase breakdown
└── results/
    └── vllm_vs_pytorch.xlsx  full comparison table
```

---

## What I'd Do With Better Hardware

On an H100 (3.35 TB/s HBM, 80GB VRAM):

- The 3B model's 85% MBU would translate to ~530 tok/s per sequence (vs 56 tok/s ceiling on 3060)
- The 1B model's CPU bottleneck becomes even more visible — more bandwidth doesn't help if the GPU is waiting for Python
- Tensor parallelism across multiple H100s becomes viable since NVLink (900 GB/s) dwarfs PCIe (64 GB/s)
- FP8 precision becomes available (SM89+) — my SM86 doesn't support it, so I couldn't measure it directly

The bottleneck analysis here is hardware-independent. The *ratios* and *mechanisms* hold at any scale.

---

## Hardware

```
GPU:       NVIDIA GeForce RTX 3060 (Ampere, SM86)
VRAM:      12GB GDDR6
Bandwidth: 360 GB/s
CUDA:      12.x
OS:        Ubuntu 22.04 (WSL2)
```
