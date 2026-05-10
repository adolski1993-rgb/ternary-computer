# Empirical Report
## Weight Distribution Validation for BitNet b1.58 2B4T
### Testing the Phase 1-5 Uniform Distribution Assumption

---

## Executive Summary

**Meaningful deviation found.** Phases 1-5 assumed trit weights are uniformly distributed (1/3 each).
The empirical measurement shows:

| Metric | Assumed (Phase 1-5) | Measured (empirical) | Delta |
|:-------|:-------------------:|:--------------------:|:------:|
| p_nonzero (all layers) | 0.667 | 0.578 | -0.089 |
| p_zero | 0.333 | 0.422 | +0.089 |

Effect on BitLinear gate speedup (the primary gate-count metric):
- Int8 activations: 9.71x assumed -> 11.12x actual (+14.6%)
- Int4 activations: 18.63x assumed -> 21.21x actual (+13.9%)

The deviation is outside 5% — see corrected numbers below.

---

## 1. What Was Measured

Source: `microsoft/bitnet-b1.58-2B-4T-bf16` (BF16 master weights)
Method: absmean quantization per BitNet b1.58 paper
  scale = mean(|W|) per weight matrix
  trits = clip(round(W / scale), -1, +1)

Tensors measured: 210 BitLinear weight matrices
  (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj x 30 layers)
Tensors skipped: embedding, lm_head, norms, biases (not BitLinear)

---

## 2. Per-Component Distribution

| Component | p(zero) | p(+1) | p(-1) | p(nonzero) | Int8 speedup delta |
|:---------|:-------:|:------:|:------:|:----------:|:------------------:|
| Attention output       | 0.416 | 0.292 | 0.292 | 0.584 | +13.4% |
| FFN down               | 0.414 | 0.293 | 0.293 | 0.586 | +13.1% |
| FFN gate               | 0.423 | 0.288 | 0.288 | 0.577 | +14.8% |
| FFN up                 | 0.417 | 0.292 | 0.292 | 0.583 | +13.6% |
| QKV projection         | 0.447 | 0.277 | 0.277 | 0.553 | +19.4% |

Assumed (uniform): p(zero) = 0.333, p(nonzero) = 0.667

The component-level variation reveals whether specific layer types deviate more
than others. Components where p_nonzero >> 2/3 have denser weights (more gates
per MAC). Components where p_nonzero << 2/3 are sparser (fewer gates, faster).

---

## 3. Layer-Depth Distribution

Per-layer p_nonzero (all components merged):

| Layer | p_nonzero | p_zero |
|------:|:---------:|:------:|
|   0 | 0.5901 | 0.4099 |
|   1 | 0.4411 | 0.5589 |
|   2 | 0.5096 | 0.4904 |
|   3 | 0.5533 | 0.4467 |
|   4 | 0.5768 | 0.4232 |
|   5 | 0.5787 | 0.4213 |
|   6 | 0.5910 | 0.4090 |
|   7 | 0.6030 | 0.3970 |
|   8 | 0.6007 | 0.3993 |
|   9 | 0.5971 | 0.4029 |
|  10 | 0.5797 | 0.4203 |
|  11 | 0.5926 | 0.4074 |
|  12 | 0.5748 | 0.4252 |
|  13 | 0.5856 | 0.4144 |
|  14 | 0.5679 | 0.4321 |
|  15 | 0.5822 | 0.4178 |
|  16 | 0.5830 | 0.4170 |
|  17 | 0.5901 | 0.4099 |
|  18 | 0.5909 | 0.4091 |
|  19 | 0.5894 | 0.4106 |
|  20 | 0.5673 | 0.4327 |
|  21 | 0.5960 | 0.4040 |
|  22 | 0.5826 | 0.4174 |
|  23 | 0.5953 | 0.4047 |
|  24 | 0.5850 | 0.4150 |
|  25 | 0.5980 | 0.4020 |
|  26 | 0.5970 | 0.4030 |
|  27 | 0.5836 | 0.4164 |
|  28 | 0.5886 | 0.4114 |
|  29 | 0.5719 | 0.4281 |

A flat table indicates the distribution is stable across depth.
A U-shaped or monotone trend indicates layer-position sensitivity
(consistent with GPTQ's finding for weight quantization).

---

## 4. Corrected Phase 1-5 Numbers

All corrections are for BitLinear gate counts only. Attention, norms,
RoPE, and residuals are UNCHANGED (they operate on activations, not weights).

The full-layer speedup corrections are proportionally smaller because
attention (~13% of binary cost at L=2048) is not affected.

### BitLinear speedup correction (per-component):

| Component | p_nz assumed | p_nz actual | Int8 speedup | Int4 speedup |
|:----------|:------------:|:-----------:|:------------:|:------------:|
| Attention output       | 0.667 | 0.584 | 9.71x -> 11.02x (+13.4%) | 18.63x -> 21.02x (+12.8%) |
| FFN down               | 0.667 | 0.586 | 9.71x -> 10.98x (+13.1%) | 18.63x -> 20.96x (+12.5%) |
| FFN gate               | 0.667 | 0.577 | 9.71x -> 11.15x (+14.8%) | 18.63x -> 21.25x (+14.1%) |
| FFN up                 | 0.667 | 0.583 | 9.71x -> 11.04x (+13.6%) | 18.63x -> 21.05x (+13.0%) |
| QKV projection         | 0.667 | 0.553 | 9.71x -> 11.60x (+19.4%) | 18.63x -> 22.08x (+18.5%) |

### Full-layer speedup (Phase 1 and Phase 2 headline numbers):

Phase 1 (BitLinear at int8, L=2048): assumed **6.72x** -> corrected **~7.70x**
Phase 2 (BitLinear at int4, L=2048): assumed **14.87x** -> corrected **~16.94x**

Note: full-layer correction is smaller than BitLinear-only correction because
attention (fixed, no trit weights) dilutes the BitLinear change.

---

## 5. Verdict: Does This Change the Conclusions?

### The assumption deviated — minor updates recommended.

Phase 1-5 findings that are NOT affected by this measurement:
- The structural advantage of balanced ternary (free subtraction, three-way compare)
- The L/d attention crossover (attention uses activations, not weights)
- The memory bandwidth analysis (weight bytes depend on precision, not distribution)
- Hardware design recommendations (same qualitative conclusions)

The BitLinear speedup numbers should be updated by +14.6% (int8) and +13.9% (int4). This is within the stated caveats for Phase 1-5.

---

## Caveats

1. This measurement uses BF16 master weights before any fine-tuning. A model
   fine-tuned with BitNet quantization-aware training might have a slightly
   different distribution.

2. The absmean scale is computed per weight matrix. Production inference
   might use per-row or per-group scales, which could change the zero fraction.

3. This validates one assumption out of many. Gate costs per primitive,
   hardware bandwidth specs, and roofline model assumptions are still modeled.
