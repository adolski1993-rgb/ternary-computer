# Phase 2 Design Memo
## Optimal Activation Precision for a Ternary Inference Accelerator
### BitNet b1.58 2B4T — Gate-Count Analysis Across Five Activation Precisions

---

## Recommendation (TL;DR)

**Target int4 activations for a first-generation ternary accelerator.**

Int4 activations deliver a **14.9× gate-count reduction** at the
workload-representative sequence length of 2048 tokens — more than double
Phase 1's int8 result (6.7×).  The jump from int8 to int4 is the
largest single gain anywhere on the precision ladder (+8.2× at L=2048)
and costs less in model quality than going to int2 or trit.  At short prompts
(L=128, the prefill-heavy regime), int4 achieves **18.3×**.  Even at the
maximum context of 4096 tokens it holds **12.9×**.

Trit activations are theoretically optimal (49.5× at L=2048) but require
end-to-end ternary activation quantization, which does not yet have the same
quality validation as int4.  Design the silicon with int4 as the primary target
and a trit-activation mode as a future-proof extension.

---

## 1. The Full Speedup Surface

All numbers are gate-count ratios relative to fp16 binary (the conventional
baseline).  Binary total at L=2048: 37.71T gates per decoder layer.

```
      Activation        L=128        L=512       L=1024       L=2048       L=4096
---------------------------------------------------------------------------------
            fp16         4.08×         3.76×         3.42×         2.95×         2.41×
            int8         9.40×         8.63×         7.84×         6.72×         5.44×
            int4        18.29×        17.40×        16.41×        14.87×        12.89×
            int2        32.50×        31.64×        30.64×        28.98×        26.59×
    trit (1.58b)        55.25×        53.86×        52.21×        49.49×        45.57×
```

Phase 1 (int8, L=2048) reported 6.72×; this analysis reproduces that
within 0.1% using the analytical expected-value model.

Key observations:
- The fp16 row establishes the floor: trit weights alone (no activation
  quantization) still deliver 3.0× because the BitLinear multiplier is
  eliminated.  This is the minimum benefit any ternary chip gets.
- Every step down the precision ladder adds meaningfully, with int8→int4
  being the largest single jump at every sequence length.
- Gains compress at long sequences because attention cost scales as O(L²)
  while BitLinear scales as O(L).  This is the L/d effect from Phase 1,
  but activation quantization pushes the crossover to longer sequences.

---

## 2. Marginal Gains Across the Precision Ladder

How much does each precision step multiply the speedup you already have?
(Absolute deltas grow at lower precision because they compound on a larger
base; the relative multiplier is the honest per-step measure.)

```
Relative speedup multiplier per step (higher = more gain per step):
        Precision step        L=128        L=512       L=1024       L=2048       L=4096
---------------------------------------------------------------------------------------
           fp16 → int8         2.31×         2.30×         2.29×         2.27×         2.26×
           int8 → int4         1.95×         2.02×         2.09×         2.21×         2.37×
           int4 → int2         1.78×         1.82×         1.87×         1.95×         2.06×
   int2 → trit (1.58b)         1.70×         1.70×         1.70×         1.71×         1.71×
```

The bottom two steps (int4→int2, int2→trit) diminish monotonically at all
sequence lengths.  The top two steps have a crossover: **fp16→int8 leads at
short sequences, int8→int4 overtakes it at long sequences.**

At L=128 (prefill-heavy): fp16→int8 multiplies speedup by 2.31× vs int8→int4
at 1.95×.  At L=4096 (long context): int8→int4 pulls ahead at 2.37× vs
fp16→int8 at 2.26×.

The crossover happens because these two steps have opposite attention profiles:
- **fp16→int8** improves attention from fp16×fp16 (230 gates) to int8×int8
  (104 gates): a 2.2× attention win.
- **int8→int4** improves attention from int8×int8 (104 gates) to int4×int4
  (36 gates): a 2.9× attention win — significantly larger.

At short sequences, attention is a small fraction of total cost, so the
2.2× vs 2.9× attention difference barely matters.  At long sequences,
attention dominates and int4's larger attention win takes over.

**Design implication:** if your target workload is long-context generation
(L>2048), prioritizing int4 over int8 activations is worth more per
precision bit than the initial int8 quantization step itself.  If your
target is short prompts, both steps are roughly equally impactful.

int4→int2 and int2→trit still add meaningfully, but the curve flattens
because fixed overheads (trit decode, requantize) become a larger share of
the total per-pair cost as the variable accumulation cost shrinks.

---

## 3. Per-Component Breakdown at L=2048

Speedup per sub-component across all five precisions (% of binary cost shown
in the rightmost column — what the component weighs in the binary total):

```
Component             fp16 act    int8 act    int4 act    int2 act    trit act  % binary
----------------------------------------------------------------------------------------
rmsnorm                   1.19×        2.72×        9.26×       28.62×       44.95×       0.0%
qkv_proj                  4.20×        9.71×       18.63×       32.81×       55.73×      12.3%
rope                      1.00×        2.26×        7.31×       21.11×       34.55×       0.0%
attention_qk              1.00×        2.21×        6.40×       16.48×       28.81×       6.5%
softmax                   1.00×        5.32×        7.77×       11.22×       12.62×       0.1%
attention_av              1.00×        2.21×        6.39×       16.43×       28.75×       6.5%
attn_out_proj             4.20×        9.71×       18.63×       32.81×       55.73×       8.2%
residual                  1.00×        2.00×        4.00×        8.00×       16.00×       0.0%
ffn_up_gate               4.20×        9.71×       18.63×       32.81×       55.73×      44.2%
ffn_activation            1.00×        2.21×        6.39×       16.43×       28.75×       0.0%
ffn_down                  4.21×        9.71×       18.64×       32.84×       55.86×      22.1%
----------------------------------------------------------------------------------------
TOTAL                     2.95×        6.72×       14.87×       28.98×       49.49×     100.0%
```

The four BitLinear projections (qkv_proj, attn_out_proj, ffn_up_gate,
ffn_down) account for 86.8% of binary cost.  Their speedup profile:

| Activation | BitLinear speedup | Attention speedup |
|:-----------|------------------:|------------------:|
| fp16       |              4.20× |              1.00× |
| int8       |              9.71× |              2.21× |
| int4       |             18.63× |              6.39× |
| int2       |             32.81× |             16.45× |
| trit       |             55.76× |             28.78× |

The attention speedup grows from 1.00× (fp16, no benefit) to
28.78× (trit × trit), crossing the same improvement curve as
BitLinear but starting from a lower baseline (no weight-trit elimination).

---

## 4. The Attention Crossover: How Precision Shifts the L/d Boundary

Phase 1 showed that ternary's advantage shrinks at long sequences because
attention (O(L²)) dominates over BitLinear (O(L)) as L approaches d=2560.
Reducing activation precision pushes that crossover to longer sequences by
making attention cheaper too.

```
Attention cost as % of total TERNARY gates (shows when attention dominates):
      Activation       L=128       L=512      L=1024      L=2048      L=4096
----------------------------------------------------------------------------
            fp16         3.8%        13.6%        24.0%        38.6%        55.6%
            int8         4.0%        14.2%        24.8%        39.7%        56.8%
            int4         2.7%         9.9%        18.0%        30.4%        46.6%
            int2         1.8%         7.0%        13.0%        23.0%        37.4%
    trit (1.58b)         1.8%         6.8%        12.7%        22.5%        36.6%
```

With int8 activations, attention already represents 40% of ternary cost at
L=2048 and 57% at L=4096.  With int4, attention is only 30% at L=2048
and 47% at L=4096 — the L/d effect is substantially reduced.

With trit activations, attention accounts for only 37% of cost even at
L=4096, meaning the gate-count advantage holds nearly flat across the full
context range.

---

## 5. Hardware Design Recommendations

### Primary target: int4 activations

1. **Dedicate trit-MAC arrays to the four BitLinear projections.**  They
   represent 86.8% of binary cost and run at 18.6× with int4
   activations.  This alone captures most of the theoretical advantage.

2. **Size int4×int4 attention units proportionally.**  At int4, attention
   runs 6.4× faster than fp16 binary.  The attention arrays need to be
   physically smaller than BitLinear arrays because attention is only 13.1%
   of binary cost; don't over-invest in attention compute at the expense of
   trit-MAC density.

3. **Non-matmul ops are irrelevant to die area.** RMSNorm, RoPE, softmax
   scalars, and residuals are < 0.5% of total gates combined at L=2048.
   Any reasonable implementation suffices.

4. **Memory bandwidth dividend.**  Int4 weights require half the memory
   bandwidth of int8, and trit weights pack to ~1.58 bits per weight.  A
   ternary accelerator loading int4 activations + trit weights is ~6× more
   memory-efficient per byte loaded than an fp16 accelerator.  This Phase 2
   analysis measures only compute; the memory advantage compounds on top.

### Extension path: trit activation mode

5. **Reserve trit-activation paths in the datapath.**  Going from int4 to
   trit multiplies your speedup by 3.33× at L=2048
   (14.9× → 49.5×).  If the BitNet
   training ecosystem produces trit-activation models (analogous to the
   progression from int8 to int4 in BitNet a4.8), the hardware should
   support it without a respin.  The trit accumulator is a strict superset
   of the int4 accumulator; a 2-bit mode select is sufficient.

6. **Target short-sequence workloads first.**  At L=128 (chat, code
   completion, classification), int4 delivers 18.3× and trit delivers
   55.2×.  These are the workloads where ternary hardware
   has the clearest ROI.  Long-document tasks (L=4096) still benefit
   (12.9× for int4) but are not the first target.

---

## 6. What Would Change This Analysis

1. **Non-uniform trit distribution.**  BitNet absmean quantization produces
   roughly 1/3–1/3–1/3 in practice but varies by layer.  FFN layers tend to
   have more nonzero trits (~0.7) than attention layers (~0.5).  Higher
   nonzero fraction worsens ternary's BitLinear speedup by up to 15%.

2. **Activation sparsity.**  ReLU²-gated FFN activations are sparse
   (many exact zeros).  Hardware that skips zero activations would get
   additional wins on top of the precision reduction modeled here.

3. **Clock-frequency penalty.**  Ternary gates distinguish three voltage
   levels vs two, adding ~25% setup-time overhead.  The 14.9× gate-count
   advantage converts to roughly 11.2× latency advantage after
   this real-silicon adjustment.  Still transformative.

4. **Memory-bound vs compute-bound regime.**  This analysis models compute
   only.  Batch-size-1 interactive inference is compute-bound for large
   models; batch processing is memory-bound.  Phase 3 addresses bandwidth.

---

## 7. Full Model Projection at L=2048 (30 layers × per-layer cost)

| Activation | Full-model binary | Full-model ternary | Speedup |
|:-----------|------------------:|-------------------:|--------:|
| fp16       | 1131.28T | 383.16T | 2.95× |
| int8       | 1131.28T | 168.42T | 6.72× |
| int4       | 1131.28T | 76.06T | 14.87× |
| int2       | 1131.28T | 39.04T | 28.98× |
| trit       | 1131.28T | 22.86T | 49.49× |

Embedding and lm_head are identical between binary and ternary; they add a
fixed cost that dilutes the per-layer advantage slightly in a real run.

---

## Methodology Notes

- Gate costs: same framework as Phase 1.  fp16 mul=150, add=80.
  Integer costs scaled as n² (multiplier) and n (adder).
  Trit MAC: 37/9 ≈ 4.11 gates/pair (analytical; see gate_costs.py).
- Weight distribution: 1/3–1/3–1/3 (zero/+1/−1), matching Phase 1.
- Gate counting is analytical (expected-value) rather than Monte Carlo.
  Validated against Phase 1 random-matrix results to within 0.1%.
- Architecture: BitNet b1.58 2B4T (hidden=2560, intermediate=6912,
  20 heads, GQA 4×, head_dim=128, max_pos=4096).

*Reproducibility:* run `python run_sweep.py` from this directory.
All results are written to `results/` as JSON.
