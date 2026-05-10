# Phase 3 Report
## Memory Bandwidth Modeling for BitNet Inference
### BitNet b1.58 2B4T — Roofline Analysis Across Chips and Precisions

---

## Executive Summary

Phase 1 and 2 measured **peak compute speedup** (gate-count reduction) for
ternary hardware.  Phase 3 closes the gap to **effective tokens per second**
by modeling memory bandwidth limits.

**The headline shift:** ternary's advantage is *larger* in effective throughput
than in peak compute, but the mechanism changes from compute efficiency to
memory efficiency.  At batch=1 decode — the dominant use case for interactive
inference — the model is **deeply memory-bound** on every chip.  Ternary wins
primarily by loading ~10× fewer bytes from memory (trit weights at 0.2 bytes
vs fp16 at 2.0 bytes), not by executing fewer gate operations.

| Scenario | Config | TPS (L=2048) | vs H100 fp16 |
|:---------|:-------|-------------:|-------------:|
| Batch=1 decode | H100 fp16 baseline | 804 | 1.00× |
| Batch=1 decode | H100 trit-w/int4-a (hypothetical) | 8,037 | 10.0× |
| Batch=1 decode | TH100 drop-in ternary | 8,037 | 10.0× |
| Batch=1 decode | TPB purpose-built ternary | 85,195 | 106× |

The 106× effective speedup from the purpose-built chip (TPB) dwarfs the
14.9× peak compute speedup from Phase 2.  The extra gain comes from
eliminating the weight-loading bottleneck by storing all model weights (417 MB
trit-packed) in on-chip SRAM.

---

## 1. Why Inference Is Memory-Bound

The roofline model says: throughput = min(compute_ceiling, bandwidth × intensity).
For compute to win, arithmetic intensity must exceed the ridge point:

| Chip | Peak compute | Peak BW | Ridge point |
|:-----|-------------:|--------:|------------:|
| H100 SXM | 990 TFLOPS | 3.35 TB/s | 148 ops/byte |
| MI300X | 1,307 TFLOPS | 5.3 TB/s | 123 ops/byte |
| TH100 (proj.) | 14,714 TFLOPS | 3.35 TB/s | 2064 ops/byte |

Decode arithmetic intensity (batch=1, one new token processed):
- **fp16 model, BitLinear**: 0.5 ops/byte  ← deeply memory-bound (ridge: 148)
- **trit-weight, int4-act, BitLinear**: 5.0 ops/byte  ← still memory-bound
- **trit-weight, batch=32, BitLinear**: 158.8 ops/byte  ← above H100 ridge!
- **Attention decode (trit, int4, any L)**: 8.0 ops/byte  ← memory-bound

At batch=1, NOTHING is compute-bound.  The chip's 148 ops/byte ridge point
is unreachable for single-token generation.  The only knob that matters is
how fast you can load weights and KV cache from memory.

---

## 2. Weight Loading: Ternary's Primary Advantage

Model weight footprint:
- fp16: **4,168 MB** (30 layers × 138.9 MB/layer)
- trit:  **417 MB** (30 layers × 13.9 MB/layer)  ← 10.0× smaller

At H100's 3.35 TB/s HBM3 bandwidth, the time to stream all weights for one
decode token:
- fp16:  1244 μs → 804 tokens/sec
- trit:   124 μs → 8,037 tokens/sec

This ~10× difference in memory traffic is the primary source of
ternary's effective throughput advantage for decode.  Phase 2's gate-count
speedup (14.9×) understates the real benefit for memory-bound workloads.

---

## 3. The KV Cache: The Hidden Second Bottleneck

KV cache memory footprint (all 30 layers, one sequence):

```
KV cache total (MB) — all 30 layers, batch=1:
Config                     L=128       L=512      L=1024      L=2048      L=4096
--------------------------------------------------------------------------------
fp16w/fp16a                  9.8        39.3        78.6       157.3       314.6
tritw/int8a                  4.9        19.7        39.3        78.6       157.3
tritw/int4a                  2.5         9.8        19.7        39.3        78.6
```

Context length at which KV cache bytes EQUAL weight loading bytes:
- fp16 weights / fp16 KV:  L* = 54,272 tokens  (practically unreachable)
- trit weights / int8 KV:  L* = 10,854 tokens
- trit weights / int4 KV:  L* = 21,709 tokens

**For ternary models, KV cache dominates weight loading at L ≈ 21,709 tokens.**
For fp16 models, the crossover is at L ≈ 54,272 tokens — far outside any
current model's context window.

This is a structural consequence of trit weight packing: the smaller the weight
footprint, the sooner KV becomes the bandwidth bottleneck.  For ternary hardware
targeting 128K context windows, KV compression (quantization, sparse attention,
sliding window) becomes as important as weight precision.

HBM capacity limit on H100 (80 GB):
- fp16 model: max context = 987,394 tokens (model takes 4168 MB, remainder for KV)
- trit model: max context = 4,144,957 tokens (model takes only 417 MB)

---

## 4. Effective Throughput: Batch=1 (Interactive Inference)

```
Decode tokens/sec and bottleneck (batch=1):
Config                                        L=128        L=512       L=1024       L=2048       L=4096
-------------------------------------------------------------------------------------------------------
H100 fp16 baseline                           804(W)       804(W)       804(W)       804(W)       804(W)
H100 trit-w/int8-a (hypothetical)           8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)
H100 trit-w/int4-a (hypothetical)           8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)
MI300X fp16 baseline                        1.3K(W)      1.3K(W)      1.3K(W)      1.3K(W)      1.3K(W)
MI300X trit-w/int4-a (hypothetical)        12.7K(W)     12.7K(W)     12.7K(W)     12.7K(W)     12.7K(W)
TH100 drop-in ternary                       8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)
TPB purpose-built ternary                 239.9K(W)    239.9K(W)    170.4K(K)     85.2K(K)     42.6K(K)
Bottleneck key: W=weight-loading, K=KV-cache, C=compute
```

At every configuration and sequence length, the bottleneck is memory — either
weight loading (W) or KV cache (K).  Compute (C) never limits at batch=1.

Key observations:
- H100 fp16: weight-bound at all sequence lengths.
- H100 with trit weights: still weight-bound at short L; KV-bound at L≥21,708.
- TH100 drop-in: same memory hierarchy as H100 → same bottleneck, same effective TPS.
  The compute advantage (14.87×) is completely invisible here.
- TPB purpose-built: weights move to SRAM → KV cache becomes the binding constraint.
  At L=128, TPB is 299× faster than H100 fp16.
  At L=2048, TPB is 106× faster.
  At L=4096, TPB is 53× faster.

---

## 5. Effective Throughput: Batch=32 (Serving)

```
Decode tokens/sec and bottleneck (batch=32):
Config                                        L=128        L=512       L=1024       L=2048       L=4096
-------------------------------------------------------------------------------------------------------
H100 fp16 baseline                        823.0K(W)    823.0K(W)    823.0K(W)    681.6K(K)    340.8K(K)
H100 trit-w/int8-a (hypothetical)           8.2M(W)      5.5M(K)      2.7M(K)      1.4M(K)    681.6K(K)
H100 trit-w/int4-a (hypothetical)           8.2M(W)      8.2M(W)      5.5M(K)      2.7M(K)      1.4M(K)
MI300X fp16 baseline                        1.3M(W)      1.3M(W)      1.3M(W)      1.1M(K)    539.1K(K)
MI300X trit-w/int4-a (hypothetical)        13.0M(W)     13.0M(W)      8.6M(K)      4.3M(K)      2.2M(K)
TH100 drop-in ternary                       8.2M(W)      8.2M(W)      5.5M(K)      2.7M(K)      1.4M(K)
TPB purpose-built ternary                  43.6M(K)     10.9M(K)      5.5M(K)      2.7M(K)      1.4M(K)
Bottleneck key: W=weight-loading, K=KV-cache, C=compute
```

With batch=32, weight bytes are amortized (one weight load serves 32 tokens),
but each of the 32 sequences loads its own KV cache.  This shifts the bottleneck:

- fp16 batch=32: weight intensity rises to 16 ops/byte → still memory-bound (ridge: 148)
- trit+int4 batch=32: weight intensity = 159 ops/byte → **above H100 ridge** → compute-bound for BitLinear!
  But attention KV intensity = 8.0 ops/byte → still memory-bound for attention.
  At batch=32, L=2048, bottleneck shifts to KV cache loading.

H100 fp16 (batch=32): 681,559 tokens/sec
TH100 trit-int4 (batch=32): 2,726,237 tokens/sec (4.0× vs H100 fp16)

The batch=32 result is more nuanced: ternary still wins, but the gap narrows
because KV cache (which doesn't benefit from weight packing) dominates at large batch.

---

## 6. Peak Compute vs Effective Throughput Gap

| Config | Phase 2 peak (L=2048) | Phase 3 effective (B=1) | Ratio |
|:-------|----------------------:|------------------------:|------:|
| H100 fp16 → H100 trit-int4 | 14.9× | 10.0× | 0.67× |
| H100 fp16 → TH100 drop-in | 14.9× | 10.0× | 0.67× |
| H100 fp16 → TPB purpose-built | 14.9× | 106× | 7.13× |

**The drop-in chip (TH100) delivers only 10.0×** despite a 14.9× compute advantage.
This is the Phase 2 → Phase 3 correction: a fast chip with slow memory is still
a slow chip for memory-bound workloads.

**The purpose-built chip (TPB) delivers 106×** — exceeding the Phase 2
compute speedup — by eliminating the weight-loading bottleneck entirely.
This is the finding that should drive silicon investment decisions.

---

## 7. SRAM-Native Chips: Groq and Cerebras

```
SRAM-only chip analysis:

  BitNet 2B4T model size — trit: 417 MB, fp16: 4168 MB
  Groq LPU SRAM:     230 MB  → trit model (417 MB) needs 2 chips
  Cerebras WSE-3:    44 GB → trit model fits with 106× headroom

  Cerebras WSE-3 (all weights in SRAM, compute-bound):
    L=  128: 59.4M tokens/sec
    L=  512: 57.8M tokens/sec
    L= 1024: 55.8M tokens/sec
    L= 2048: 52.1M tokens/sec
    L= 4096: 46.1M tokens/sec

  Groq LPU (2-chip for model, 80 TB/s SRAM BW, no HBM for KV):
  → KV cache grows until it overflows SRAM; not modeled for long L.
  → At L=128, KV fits; at L>1000, KV approaches remaining SRAM capacity.
  → Available SRAM after weights (2 chips): 43 MB for KV + activations.
```

Both Groq and Cerebras illustrate the purpose-built design principle at
extreme scale.  Groq's 230 MB SRAM barely fits a 2B trit model (417 MB
needs 2 chips); Cerebras' 44 GB SRAM fits models up to ~220B parameters
at trit precision.  For BitNet 2B4T, Cerebras becomes nearly compute-bound
at decode because weights and even KV cache fit entirely on-chip.

The TPB concept (512 MB SRAM weight store, 24 GB HBM for KV) is a middle
path: smaller die area than Cerebras, tailored specifically to the 2B-7B
model range that dominates edge inference.

---

## 8. Hardware Investment Recommendations

### For a ternary accelerator startup

1. **Don't build a fast chip with slow memory** (TH100 anti-pattern).
   A ternary chip with H100-class HBM3 delivers only 10.0× vs H100 for
   interactive inference — not a compelling hardware investment thesis.

2. **Store weights on-chip** (TPB pattern).  417 MB of SRAM holds all
   of BitNet 2B4T.  An aggressive 512 MB on-chip SRAM at 100 TB/s costs
   die area but eliminates the #1 bottleneck.  Effective speedup jumps to 106×.

3. **Optimize KV cache next**.  Once weights are on-chip, KV is the next wall.
   Options: int4 KV (2× less traffic), sliding window attention, KV compression
   (not modeled here).  Each halves the KV loading time.

4. **Target interactive inference (batch=1) first**.  Ternary's memory advantage
   is largest at batch=1 (106× for TPB).  For serving at batch=32,
   the gap narrows because KV loading dominates.

5. **Market first to edge devices, not datacenter**.  Edge devices have tight
   power and memory budgets where the 10× weight reduction transforms
   feasibility (a BitNet 2B model that requires 4 GB of fp16 weights fits in
   400 MB of trit-packed weights — RAM that many edge SoCs have on-chip).

---

## 9. Caveats and Assumptions

1. **Roofline assumes perfect overlap of compute and memory**.  Real hardware
   achieves 70-90% of roofline due to scheduling overhead, DRAM latency,
   bank conflicts.  Multiply all TPS numbers by 0.7-0.9 for real estimates.

2. **Ternary chip compute specs are gate-count projections**, not silicon
   measurements.  Phase 2's 14.87× gate advantage assumes same die area and
   same technology node as H100.  Real silicon has layout overhead, power,
   routing.  A conservative estimate is 10-12× effective compute advantage.

3. **Clock frequency penalty (25%) for 3-level logic** is an industry estimate,
   not a measurement for ternary-specific circuits.  Applied uniformly here;
   real designs may optimize critical paths.

4. **Weight distribution assumed uniform (1/3 each trit)**.  Real BitNet models
   vary by layer (attention layers tend to 50% nonzero, FFN layers to 70%).
   Higher nonzero fraction increases effective gate cost by up to 15%.

5. **KV cache assumed uncompressed fp16/int8/int4**.  Production systems use
   KV quantization and eviction; this model represents the worst case.

6. **Memory-bound analysis only**.  Power, area, yield, and manufacturing cost
   are all outside scope.  These are real barriers to a ternary startup.

---

## 10. Summary Table: Where Each Chip's Bottleneck Sits

```
Decode tokens/sec and bottleneck (batch=1):
Config                                        L=128        L=512       L=1024       L=2048       L=4096
-------------------------------------------------------------------------------------------------------
H100 fp16 baseline                           804(W)       804(W)       804(W)       804(W)       804(W)
H100 trit-w/int8-a (hypothetical)           8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)
H100 trit-w/int4-a (hypothetical)           8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)
MI300X fp16 baseline                        1.3K(W)      1.3K(W)      1.3K(W)      1.3K(W)      1.3K(W)
MI300X trit-w/int4-a (hypothetical)        12.7K(W)     12.7K(W)     12.7K(W)     12.7K(W)     12.7K(W)
TH100 drop-in ternary                       8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)      8.0K(W)
TPB purpose-built ternary                 239.9K(W)    239.9K(W)    170.4K(K)     85.2K(K)     42.6K(K)
Bottleneck key: W=weight-loading, K=KV-cache, C=compute
```

```
Decode tokens/sec and bottleneck (batch=32):
Config                                        L=128        L=512       L=1024       L=2048       L=4096
-------------------------------------------------------------------------------------------------------
H100 fp16 baseline                        823.0K(W)    823.0K(W)    823.0K(W)    681.6K(K)    340.8K(K)
H100 trit-w/int8-a (hypothetical)           8.2M(W)      5.5M(K)      2.7M(K)      1.4M(K)    681.6K(K)
H100 trit-w/int4-a (hypothetical)           8.2M(W)      8.2M(W)      5.5M(K)      2.7M(K)      1.4M(K)
MI300X fp16 baseline                        1.3M(W)      1.3M(W)      1.3M(W)      1.1M(K)    539.1K(K)
MI300X trit-w/int4-a (hypothetical)        13.0M(W)     13.0M(W)      8.6M(K)      4.3M(K)      2.2M(K)
TH100 drop-in ternary                       8.2M(W)      8.2M(W)      5.5M(K)      2.7M(K)      1.4M(K)
TPB purpose-built ternary                  43.6M(K)     10.9M(K)      5.5M(K)      2.7M(K)      1.4M(K)
Bottleneck key: W=weight-loading, K=KV-cache, C=compute
```

---

## Methodology

- Architecture: BitNet b1.58 2B4T (hidden=2560, FFN=6912, 30 layers, 20 heads, GQA 4:1)
- Weight packing: trit = 5 trits/byte (0.2 bytes/weight); int8 = 1 byte; fp16 = 2 bytes
- Activation packing: int4 = 0.5 bytes; int8 = 1 byte; fp16 = 2 bytes
- Roofline: ideal overlap (compute and memory fully pipelined)
- KV cache: fully loaded from HBM each decode step; no compression
- Batch=1 decode: one new token generated, full KV context re-read
- Compute rates: H100 non-sparse published specs; ternary = Phase 2 speedup × H100
- Clock penalty for 3-level logic: 25% (ternary gates slower per switch cycle)
- Sources:
  - H100: nvidia.com/en-us/data-center/h100 (989.5 TFLOPS fp16, 3.35 TB/s)
  - MI300X: amd.com/en/products/accelerators/instinct/mi300/mi300x.html (1307.4 TFLOPS, 5.3 TB/s)
  - Groq LPU: groq.com/lpu-architecture (750 TOPS int8, 230 MB SRAM, 80 TB/s)
  - Cerebras WSE-3: cerebras.ai/chip (125 PFLOPS, 44 GB SRAM, 21 PB/s)
  - Phase 2 speedups: ternary/phase2/DESIGN_MEMO.md

*Reproducibility:* run `python run_phase3.py` from this directory.
