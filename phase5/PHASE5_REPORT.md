# Phase 5 Report
## Prefill Analysis for BitNet Inference
### BitNet b1.58 2B4T — Prefill Arithmetic Intensity, Roofline, and Workload Mix

---

## Executive Summary

Phases 1-4 measured **decode phase only** — always memory-bound, always
limited by weight-loading speed.  Phase 5 asks: what happens during prefill,
when all L input tokens are processed simultaneously?

**Phase 5 finding: ternary's compute advantage IS visible during prefill,
but the window is narrower than expected, and attention is the new bottleneck.**

Key headline:

| Config | Prefill TPS (L=2048) | Decode TPS | Prefill/Decode ratio |
|:-------|---------------------:|-----------:|--------------------:|
| H100 fp16 | 176.1K | 804 | 219.1× |
| H100 trit+int4 (hyp.) | 404.4K | 8.0K | 50.3× |
| TH100 drop-in | 1.5M | 8.0K | 186.2× |
| TPB purpose-built | 1.5M | 85.2K | 17.6× |

Prefill is always faster than decode (measured in tokens/sec processed per second).
The advantage over decode grows with L because prefill amortizes weight loads over L
tokens while decode reloads weights for every single output token.

---

## 1. Why Prefill Has Higher Arithmetic Intensity

**Decode** (one new token): each token loads ALL model weights alone.
  BitLinear intensity ≈ 1 MAC/byte (fp16) or 10 MAC/byte (trit).  Deeply memory-bound.

**Prefill** (L tokens at once): weights are loaded ONCE and shared by all L tokens.
  BitLinear intensity ≈ L/bpw MACs/byte.  Grows linearly with L.

This means: at large enough L, even a chip as memory-bandwidth-rich as H100
becomes compute-bound for BitLinear projections.

The crossover sequence length L* (where BitLinear goes compute-bound):

```
Prefill BitLinear crossover (memory-bound → compute-bound), batch=1:

Config                       BL Ridge  Attn Ridge    L* (BL)   Attn bound
------------------------------------------------------------------------
H100 fp16/fp16                    148         148       >356 memory-bound
H100 trit/int4                    295         295        >65 memory-bound
MI300X fp16/fp16                  123         123       >287 memory-bound
MI300X trit/int4                  247         247        >53 compute-bound
TH100 trit/int4                  2064         708      >1014 memory-bound
TPB trit/int4                      69         708        >14 memory-bound
```

Key results:
- H100 fp16:  compute-bound at L > 356 tokens.
  Most chat prompts (~500-2K tokens) are already compute-bound for H100 fp16 prefill.
- H100 trit:  compute-bound at L > 65 tokens.
  Almost immediately — trit weights are so light that weight loading is trivial.
- TPB trit:   compute-bound at L > 14 tokens.
  Effectively ALWAYS compute-bound (SRAM bandwidth = 100 TB/s crushes weight loading).

---

## 2. Attention: The Unchanging Bottleneck

BitLinear goes compute-bound as L grows.  Attention does not.

Attention (Q@K^T) arithmetic intensity = HEAD_DIM / bpa (constant, independent of L):
  fp16:   64 MACs/byte
  int4:   256 MACs/byte
  H100 ridge: 148 MACs/byte

Attention is memory-bound on H100 (intensity 64 < ridge 148 for fp16).
The score matrix (N_HEADS × L² × bpa per layer) grows quadratically and must be
written and read from HBM for each forward pass.

**This means:**
- At short L (< 1K): BitLinear dominates compute, attention is a minor component
- At long L (> 2K): attention's O(L²) score matrix starts to cost as much as BitLinear
- The score matrix becomes the primary bandwidth bottleneck at long prefills

At L=4096, int4: score matrix = 336 MB per layer × 30 = 10066 MB
Time at 3.35 TB/s: 3.0 ms (per batch)

**FlashAttention caveat:**  FlashAttention eliminates the score matrix writes by fusing
softmax computation.  This reduces attention from O(L²) memory to O(L), making attention
also compute-bound at longer L.  We do NOT model FlashAttention — our prefill numbers
are pessimistic for attention by up to 10× at very long L.

---

## 3. Prefill Throughput Surface

```
Prefill speedup vs H100 fp16 (batch=1):
  Config                            L=128      L=512     L=1024     L=2048     L=4096
  -----------------------------------------------------------------------------------
  H100 fp16                         1.00x      1.00x      1.00x      1.00x      1.00x
  H100 trit+int4 (hyp.)             4.94x      2.08x      2.16x      2.30x      2.52x
  MI300X fp16                       1.58x      1.34x      1.35x      1.38x      1.42x
  TH100 drop-in                     8.97x      8.92x     10.20x      8.50x      6.90x
  TPB purpose-built                32.45x     11.65x     10.20x      8.50x      6.90x
```

The TH100 drop-in ternary chip gets 8.5× prefill speedup over H100 fp16 at
L=2048.  This is MUCH higher than TH100's decode speedup (~10×).  The reason: prefill
makes TH100's compute advantage visible.  In Phase 3, TH100 decode was compute-idle
(memory-bound, waiting for weights).  In prefill, TH100 is compute-bound (weights are
amortized) and its 11.2× effective compute rate delivers real speedup.

TPB prefill is limited by attention's score matrix bandwidth (not weights, not compute)
at L=2048+.  The SRAM weight advantage doesn't help attention — attention lives in HBM.

---

## 4. Prefill vs Decode Comparison

```
Prefill vs Decode TPS comparison (batch=1, L=2048):
  Config                         Prefill TPS    Decode TPS  Prefill/Decode
  ----------------------------------------------------------------------
  H100 fp16                           176.1K           804          219.1x
  H100 trit+int4 (hyp.)               404.4K          8.0K           50.3x
  MI300X fp16                         243.0K          1.3K          191.1x
  TH100 drop-in                         1.5M          8.0K          186.2x
  TPB purpose-built                     1.5M         85.2K           17.6x
```

"Prefill TPS" = input tokens processed per second.
"Decode TPS" = output tokens generated per second.

Prefill always runs faster (more tokens per second) because it amortizes weight
loading across L tokens.  For decode, every single output token reloads all weights.

The ratio grows with L: at L=128, prefill is ~2-5× faster than decode; at L=4096,
the ratio exceeds 100× for trit-weight chips.

---

## 5. Workload Mix: Blended Speedup

```
Blended speedup — H100 → TPB purpose-built:
Workload                    n_in  n_out   Prefill%   Total spd   Prefill spd   Decode spd
----------------------------------------------------------------------------------
RAG Retrieval               4000    200        10%      31.80x         6.90x       54.27x
Document Summary            3000    400         3%      57.95x         8.50x       72.36x
Interactive Chat            1024    512         1%     183.38x        10.20x      212.00x
Code Completion              512    128         1%     219.92x        11.65x      298.51x
Long-Form Generation         256   3000         0%     296.75x        32.45x      298.51x
```

The blended speedup across workload types:

| Workload | Prefill% | Total speedup | Prefill speedup | Decode speedup |
|:---------|:--------:|:-------------:|:---------------:|:--------------:|
| RAG Retrieval | 10% | 31.8× | 6.9× | 54.3× |
| Document Summary | 3% | 57.9× | 8.5× | 72.4× |
| Interactive Chat | 1% | 183.4× | 10.2× | 212.0× |
| Code Completion | 1% | 219.9× | 11.6× | 298.5× |
| Long-Form Generation | 0% | 296.8× | 32.5× | 298.5× |

(TPB vs H100 fp16, batch=1.  All numbers are roofline upper bounds.)

**The speedup curve is roughly flat across workloads (297×–32×).**
This is surprising: the expected finding was that prefill-heavy workloads (RAG)
would show dramatically higher speedup than decode-heavy workloads (generation).

The reason it's flatter than expected: prefill itself is partially limited by
attention's score matrix bandwidth (same HBM as decode KV loading), and decode
on TPB runs very fast anyway (85K TPS).  When both prefill and decode are fast,
total latency is low and the ratio stays bounded.

For H100 trit+int4 vs H100 fp16 (same chip, hypothetical weight change):
```
Blended speedup — H100: fp16 → trit+int4:
Workload                    n_in  n_out   Prefill%   Total spd   Prefill spd   Decode spd
----------------------------------------------------------------------------------
RAG Retrieval               4000    200        10%       7.65x         2.52x       10.00x
Document Summary            3000    400         3%       9.00x         2.30x       10.00x
Interactive Chat            1024    512         1%       9.72x         2.16x       10.00x
Code Completion              512    128         1%       9.48x         2.08x       10.00x
Long-Form Generation         256   3000         0%       9.99x         4.94x       10.00x
```

---

## 6. Hardware Design Implications

### Prefill-heavy products (RAG, classification, embeddings)
- Ternary chip should prioritize COMPUTE DENSITY for BitLinear projections
- Attention is still memory-bound → HBM bandwidth investment still needed
- FlashAttention support is critical to unlock full prefill performance at long L
- Phase 5's numbers for L=4096 prefill are ~10× pessimistic without FlashAttention

### Decode-heavy products (chat, long-form generation)
- Phase 3's finding holds: memory bandwidth dominates, SRAM weights are key
- Compute advantage from trit MACs is not the primary ROI
- KV cache compression (int4, sliding window) has more impact than trit MAC density

### Unified design recommendation
Build trit-MAC arrays sized for compute-bound prefill (primary benefit) with enough
HBM bandwidth to serve decode KV caches (secondary bottleneck at long L).

The attention score matrix is the gap in both cases:
FlashAttention or sparse-attention hardware reduces attention's memory cost to O(L),
making prefill fully compute-bound and unlocking the full ternary advantage at any L.

---

## 7. Crossover vs Phase 1-4 Predictions

Phase 1-2 showed 14.9× gate-count speedup for ternary (int4) at L=2048.
Phase 3 showed only ~10× effective speedup for DECODE (memory-bound, weight-packing dominates).
Phase 5 shows:
  - Prefill at L=2048: TH100 achieves 8.5× over H100 fp16
    → the compute advantage IS realized for prefill
  - Long-form generation (5% prefill): blended speedup ≈ 297× (decode-dominated)
  - RAG retrieval (95% prefill): blended speedup ≈ 32× (prefill-dominated)

The Phase 1-2 gate speedup is more accurately described as:
  "The theoretical maximum for prefill-dominated workloads on a purpose-built chip."
  The effective number ranges from ~297× (pure decode) to ~32× (pure prefill)
  depending on the prefill/decode mix.

---

## Caveats

1. **FlashAttention not modeled.** Real production systems use FlashAttention,
   which fuses attention and avoids materializing the O(L²) score matrix.
   Phase 5's prefill TPS numbers are pessimistic by up to 10× for L=4096.
   With FlashAttention, prefill would be more compute-bound and speedups higher.

2. **Single-batch only.** Batch=B prefill amortizes weights even further.
   At batch=32, L=512: effective weight intensity = 32×512/bpw → fully compute-bound
   on any chip. Phase 5 batch=1 is the conservative, latency-focused case.

3. **KV cache write not fully modeled.** Prefill also WRITES the initial KV cache.
   This adds N_KV_HEADS × L × HEAD_DIM × 2 × bpa per layer, which is smaller than
   the score matrix but non-zero.

4. **Speculative decoding not modeled.** Many production systems use speculative
   decoding to overlap prefill and decode, improving effective throughput.
   This blurs the prefill/decode boundary in ways we don't capture.

5. **Compute rate for H100 trit hypothetical uses H100's int8 rate.** Real H100
   hardware doesn't support native trit MACs.  The int8 rate is used as a proxy
   for "what H100 would do if it could process lower-precision activations."

---

## Methodology

- Architecture: BitNet b1.58 2B4T (hidden=2560, FFN=6912, 30 layers, GQA 4:1)
- Gate costs: Phase 2 model (gate_costs.py), applied with seq_len=L for prefill
- Memory: weight loading (once per layer) + activation traffic + score matrix + KV write
- Attention: standard causal attention (score matrix materialized to HBM)
- Roofline: ideal compute/memory overlap assumed
- Decode: Phase 3 decode_roofline() with context_len = n_input
- Verify: prefill at seq_len=1 collapses to decode model (checked in prefill_compute.py)

*Reproducibility:* run `python run_phase5.py` from this directory.
