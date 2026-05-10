"""
Phase 3: Memory Bandwidth Modeling — main script.

Generates:
  results/arithmetic_intensity.json
  results/roofline_data.json
  results/hardware_comparison.json
  PHASE3_REPORT.md
  visualization.html
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from hardware_specs import (
    H100_SXM, MI300X, TH100, TPB, GROQ_LPU, CEREBRAS_WSE3,
    PHASE2_SPEEDUP_INT4_BITLINEAR, PHASE2_SPEEDUP_INT4_ATTENTION,
    PHASE2_SPEEDUP_INT4_OVERALL, TERNARY_CLOCK_PENALTY,
)
from memory_model import (
    total_weight_bytes, layer_weight_bytes, decode_bytes_per_token,
    bitlinear_intensity, attention_decode_intensity, attention_prefill_intensity,
    LAYER_WEIGHT_ELEMENTS, N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE,
    ACT_BYTES, WEIGHT_BYTES,
)
from kv_cache import (
    kv_bytes_total, kv_weight_crossover_length, hbm_capacity_max_context, kv_table,
)
from roofline import decode_roofline, ridge_point, build_roofline_data
from effective_throughput import (
    CONFIGS, SEQUENCE_LENGTHS, BATCH_SIZES, throughput_matrix,
    bottleneck_table, speedup_vs_baseline, groq_cerebras_analysis,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Run all calculations
# ---------------------------------------------------------------------------

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Arithmetic intensity for each config
    intensity_data = {}
    for wprec, aprec in [('fp16','fp16'), ('trit','int8'), ('trit','int4'), ('trit','trit')]:
        label = f"{wprec}w/{aprec}a"
        intensity_data[label] = {
            "bitlinear_b1":   bitlinear_intensity(wprec, aprec, batch=1),
            "bitlinear_b8":   bitlinear_intensity(wprec, aprec, batch=8),
            "bitlinear_b32":  bitlinear_intensity(wprec, aprec, batch=32),
            "bitlinear_b128": bitlinear_intensity(wprec, aprec, batch=128),
            "attn_decode": {str(L): attention_decode_intensity(aprec, L) for L in SEQUENCE_LENGTHS},
            "attn_prefill": {str(L): attention_prefill_intensity(aprec, L) for L in SEQUENCE_LENGTHS},
        }
    with open(os.path.join(RESULTS_DIR, "arithmetic_intensity.json"), "w") as f:
        json.dump(intensity_data, f, indent=2)

    # 2. Roofline data
    roofline_data = build_roofline_data()
    with open(os.path.join(RESULTS_DIR, "roofline_data.json"), "w") as f:
        json.dump(roofline_data, f, indent=2)

    # 3. Hardware comparison: tokens/sec for all configs, batches, lengths
    hw_comparison = {}
    for batch in BATCH_SIZES:
        hw_comparison[str(batch)] = {}
        matrix = throughput_matrix(batch)
        for chip, wprec, aprec, label in CONFIGS:
            hw_comparison[str(batch)][label] = {}
            for L in SEQUENCE_LENGTHS:
                r = matrix[label][L]
                hw_comparison[str(batch)][label][str(L)] = {
                    "tokens_per_sec": r["tokens_per_sec"],
                    "bottleneck":     r["bottleneck"],
                    "compute_time_s": r["compute_time_s"],
                    "weight_time_s":  r["weight_time_s"],
                    "kv_time_s":      r["kv_time_s"],
                }
    with open(os.path.join(RESULTS_DIR, "hardware_comparison.json"), "w") as f:
        json.dump(hw_comparison, f, indent=2)

    return intensity_data, roofline_data, hw_comparison


# ---------------------------------------------------------------------------
# Derived numbers for the report
# ---------------------------------------------------------------------------

def tps(chip, wprec, aprec, L, batch=1):
    return decode_roofline(chip, wprec, aprec, L, batch)["tokens_per_sec"]

def bn(chip, wprec, aprec, L, batch=1):
    return decode_roofline(chip, wprec, aprec, L, batch)["bottleneck"]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report():
    # Pre-compute key numbers
    h_fp16_128   = tps(H100_SXM, 'fp16', 'fp16', 128,  1)
    h_fp16_2048  = tps(H100_SXM, 'fp16', 'fp16', 2048, 1)
    h_fp16_4096  = tps(H100_SXM, 'fp16', 'fp16', 4096, 1)
    h_ti4_128    = tps(H100_SXM, 'trit', 'int4', 128,  1)
    h_ti4_2048   = tps(H100_SXM, 'trit', 'int4', 2048, 1)
    h_ti4_4096   = tps(H100_SXM, 'trit', 'int4', 4096, 1)
    th_ti4_2048  = tps(TH100,    'trit', 'int4', 2048, 1)
    tpb_ti4_128  = tps(TPB,      'trit', 'int4', 128,  1)
    tpb_ti4_2048 = tps(TPB,      'trit', 'int4', 2048, 1)
    tpb_ti4_4096 = tps(TPB,      'trit', 'int4', 4096, 1)

    sp_h_ti4   = h_ti4_2048  / h_fp16_2048
    sp_th_ti4  = th_ti4_2048 / h_fp16_2048
    sp_tpb_ti4 = tpb_ti4_2048/ h_fp16_2048

    # Batch=32
    h_fp16_b32  = tps(H100_SXM, 'fp16', 'fp16', 2048, 32)
    th_b32      = tps(TH100,    'trit', 'int4', 2048, 32)
    tpb_b32     = tps(TPB,      'trit', 'int4', 2048, 32)

    # KV crossover lengths
    cross_fp16 = kv_weight_crossover_length('fp16', 'fp16')
    cross_ti4  = kv_weight_crossover_length('trit', 'int4')
    cross_ti8  = kv_weight_crossover_length('trit', 'int8')

    # Arithmetic intensities
    bl_int_fp16 = bitlinear_intensity('fp16', 'fp16', batch=1)
    bl_int_ti4  = bitlinear_intensity('trit', 'int4', batch=1)
    bl_int_ti4_b32 = bitlinear_intensity('trit', 'int4', batch=32)
    attn_int_ti4_2048 = attention_decode_intensity('int4', 2048)
    ridge_h100 = ridge_point(H100_SXM, 'fp16')
    ridge_th   = ridge_point(TH100,    'bitlinear')

    # Weight sizes
    w_fp16_mb = total_weight_bytes('fp16') / 1e6
    w_trit_mb = total_weight_bytes('trit') / 1e6

    # HBM capacity
    hbm_max_fp16 = hbm_capacity_max_context('fp16', 'fp16', 80.0, 1)
    hbm_max_ti4  = hbm_capacity_max_context('trit', 'int4', 80.0, 1)

    # Phase 2 speedup for contrast
    p2_ti4 = PHASE2_SPEEDUP_INT4_OVERALL

    report = f"""\
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
| Batch=1 decode | H100 fp16 baseline | {h_fp16_2048:,.0f} | 1.00× |
| Batch=1 decode | H100 trit-w/int4-a (hypothetical) | {h_ti4_2048:,.0f} | {sp_h_ti4:.1f}× |
| Batch=1 decode | TH100 drop-in ternary | {th_ti4_2048:,.0f} | {sp_th_ti4:.1f}× |
| Batch=1 decode | TPB purpose-built ternary | {tpb_ti4_2048:,.0f} | {sp_tpb_ti4:.0f}× |

The {sp_tpb_ti4:.0f}× effective speedup from the purpose-built chip (TPB) dwarfs the
{p2_ti4:.1f}× peak compute speedup from Phase 2.  The extra gain comes from
eliminating the weight-loading bottleneck by storing all model weights ({w_trit_mb:.0f} MB
trit-packed) in on-chip SRAM.

---

## 1. Why Inference Is Memory-Bound

The roofline model says: throughput = min(compute_ceiling, bandwidth × intensity).
For compute to win, arithmetic intensity must exceed the ridge point:

| Chip | Peak compute | Peak BW | Ridge point |
|:-----|-------------:|--------:|------------:|
| H100 SXM | {H100_SXM.tflops_fp16:,.0f} TFLOPS | {H100_SXM.hbm_bandwidth_tbs:.2f} TB/s | {ridge_h100:.0f} ops/byte |
| MI300X | {MI300X.tflops_fp16:,.0f} TFLOPS | {MI300X.hbm_bandwidth_tbs:.1f} TB/s | {ridge_point(MI300X,'fp16'):.0f} ops/byte |
| TH100 (proj.) | {TH100.tflops_fp16:,.0f} TFLOPS | {TH100.hbm_bandwidth_tbs:.2f} TB/s | {ridge_th:.0f} ops/byte |

Decode arithmetic intensity (batch=1, one new token processed):
- **fp16 model, BitLinear**: {bl_int_fp16:.1f} ops/byte  ← deeply memory-bound (ridge: {ridge_h100:.0f})
- **trit-weight, int4-act, BitLinear**: {bl_int_ti4:.1f} ops/byte  ← still memory-bound
- **trit-weight, batch=32, BitLinear**: {bl_int_ti4_b32:.1f} ops/byte  ← above H100 ridge!
- **Attention decode (trit, int4, any L)**: {attn_int_ti4_2048:.1f} ops/byte  ← memory-bound

At batch=1, NOTHING is compute-bound.  The chip's {ridge_h100:.0f} ops/byte ridge point
is unreachable for single-token generation.  The only knob that matters is
how fast you can load weights and KV cache from memory.

---

## 2. Weight Loading: Ternary's Primary Advantage

Model weight footprint:
- fp16: **{w_fp16_mb:,.0f} MB** (30 layers × {layer_weight_bytes('fp16')/1e6:.1f} MB/layer)
- trit:  **{w_trit_mb:,.0f} MB** (30 layers × {layer_weight_bytes('trit')/1e6:.1f} MB/layer)  ← {w_fp16_mb/w_trit_mb:.1f}× smaller

At H100's 3.35 TB/s HBM3 bandwidth, the time to stream all weights for one
decode token:
- fp16:  {total_weight_bytes('fp16') / (H100_SXM.hbm_bandwidth_tbs * 1e12) * 1e6:.0f} μs → {1/(total_weight_bytes('fp16')/(H100_SXM.hbm_bandwidth_tbs*1e12)):,.0f} tokens/sec
- trit:   {total_weight_bytes('trit') / (H100_SXM.hbm_bandwidth_tbs * 1e12) * 1e6:.0f} μs → {1/(total_weight_bytes('trit')/(H100_SXM.hbm_bandwidth_tbs*1e12)):,.0f} tokens/sec

This ~{w_fp16_mb/w_trit_mb:.0f}× difference in memory traffic is the primary source of
ternary's effective throughput advantage for decode.  Phase 2's gate-count
speedup ({p2_ti4:.1f}×) understates the real benefit for memory-bound workloads.

---

## 3. The KV Cache: The Hidden Second Bottleneck

KV cache memory footprint (all 30 layers, one sequence):

```
{kv_table()}
```

Context length at which KV cache bytes EQUAL weight loading bytes:
- fp16 weights / fp16 KV:  L* = {cross_fp16:,.0f} tokens  (practically unreachable)
- trit weights / int8 KV:  L* = {cross_ti8:,.0f} tokens
- trit weights / int4 KV:  L* = {cross_ti4:,.0f} tokens

**For ternary models, KV cache dominates weight loading at L ≈ {cross_ti4:,.0f} tokens.**
For fp16 models, the crossover is at L ≈ {cross_fp16:,.0f} tokens — far outside any
current model's context window.

This is a structural consequence of trit weight packing: the smaller the weight
footprint, the sooner KV becomes the bandwidth bottleneck.  For ternary hardware
targeting 128K context windows, KV compression (quantization, sparse attention,
sliding window) becomes as important as weight precision.

HBM capacity limit on H100 (80 GB):
- fp16 model: max context = {hbm_max_fp16:,} tokens (model takes {w_fp16_mb:.0f} MB, remainder for KV)
- trit model: max context = {hbm_max_ti4:,} tokens (model takes only {w_trit_mb:.0f} MB)

---

## 4. Effective Throughput: Batch=1 (Interactive Inference)

```
{bottleneck_table(batch=1)}
```

At every configuration and sequence length, the bottleneck is memory — either
weight loading (W) or KV cache (K).  Compute (C) never limits at batch=1.

Key observations:
- H100 fp16: weight-bound at all sequence lengths.
- H100 with trit weights: still weight-bound at short L; KV-bound at L≥{int(cross_ti4):,}.
- TH100 drop-in: same memory hierarchy as H100 → same bottleneck, same effective TPS.
  The compute advantage (14.87×) is completely invisible here.
- TPB purpose-built: weights move to SRAM → KV cache becomes the binding constraint.
  At L=128, TPB is {tpb_ti4_128/h_fp16_128:.0f}× faster than H100 fp16.
  At L=2048, TPB is {tpb_ti4_2048/h_fp16_2048:.0f}× faster.
  At L=4096, TPB is {tpb_ti4_4096/h_fp16_4096:.0f}× faster.

---

## 5. Effective Throughput: Batch=32 (Serving)

```
{bottleneck_table(batch=32)}
```

With batch=32, weight bytes are amortized (one weight load serves 32 tokens),
but each of the 32 sequences loads its own KV cache.  This shifts the bottleneck:

- fp16 batch=32: weight intensity rises to {bitlinear_intensity('fp16','fp16',32):.0f} ops/byte → still memory-bound (ridge: {ridge_h100:.0f})
- trit+int4 batch=32: weight intensity = {bl_int_ti4_b32:.0f} ops/byte → **above H100 ridge** → compute-bound for BitLinear!
  But attention KV intensity = {attn_int_ti4_2048:.1f} ops/byte → still memory-bound for attention.
  At batch=32, L=2048, bottleneck shifts to KV cache loading.

H100 fp16 (batch=32): {h_fp16_b32:,.0f} tokens/sec
TH100 trit-int4 (batch=32): {th_b32:,.0f} tokens/sec ({th_b32/h_fp16_b32:.1f}× vs H100 fp16)

The batch=32 result is more nuanced: ternary still wins, but the gap narrows
because KV cache (which doesn't benefit from weight packing) dominates at large batch.

---

## 6. Peak Compute vs Effective Throughput Gap

| Config | Phase 2 peak (L=2048) | Phase 3 effective (B=1) | Ratio |
|:-------|----------------------:|------------------------:|------:|
| H100 fp16 → H100 trit-int4 | {p2_ti4:.1f}× | {sp_h_ti4:.1f}× | {sp_h_ti4/p2_ti4:.2f}× |
| H100 fp16 → TH100 drop-in | {p2_ti4:.1f}× | {sp_th_ti4:.1f}× | {sp_th_ti4/p2_ti4:.2f}× |
| H100 fp16 → TPB purpose-built | {p2_ti4:.1f}× | {sp_tpb_ti4:.0f}× | {sp_tpb_ti4/p2_ti4:.2f}× |

**The drop-in chip (TH100) delivers only {sp_h_ti4:.1f}×** despite a {p2_ti4:.1f}× compute advantage.
This is the Phase 2 → Phase 3 correction: a fast chip with slow memory is still
a slow chip for memory-bound workloads.

**The purpose-built chip (TPB) delivers {sp_tpb_ti4:.0f}×** — exceeding the Phase 2
compute speedup — by eliminating the weight-loading bottleneck entirely.
This is the finding that should drive silicon investment decisions.

---

## 7. SRAM-Native Chips: Groq and Cerebras

```
{groq_cerebras_analysis()}
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
   A ternary chip with H100-class HBM3 delivers only {sp_h_ti4:.1f}× vs H100 for
   interactive inference — not a compelling hardware investment thesis.

2. **Store weights on-chip** (TPB pattern).  {w_trit_mb:.0f} MB of SRAM holds all
   of BitNet 2B4T.  An aggressive 512 MB on-chip SRAM at 100 TB/s costs
   die area but eliminates the #1 bottleneck.  Effective speedup jumps to {sp_tpb_ti4:.0f}×.

3. **Optimize KV cache next**.  Once weights are on-chip, KV is the next wall.
   Options: int4 KV (2× less traffic), sliding window attention, KV compression
   (not modeled here).  Each halves the KV loading time.

4. **Target interactive inference (batch=1) first**.  Ternary's memory advantage
   is largest at batch=1 ({sp_tpb_ti4:.0f}× for TPB).  For serving at batch=32,
   the gap narrows because KV loading dominates.

5. **Market first to edge devices, not datacenter**.  Edge devices have tight
   power and memory budgets where the {w_fp16_mb/w_trit_mb:.0f}× weight reduction transforms
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
{bottleneck_table(batch=1)}
```

```
{bottleneck_table(batch=32)}
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
"""
    with open(os.path.join(os.path.dirname(__file__), "PHASE3_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# Visualization (HTML + Chart.js from CDN)
# ---------------------------------------------------------------------------

def write_visualization(hw_comparison: dict):
    # Build data for charts
    labels_b1 = [label for _, _, _, label in CONFIGS]
    tps_l2048_b1 = [
        hw_comparison["1"][label]["2048"]["tokens_per_sec"]
        for _, _, _, label in CONFIGS
    ]
    tps_l128_b1 = [
        hw_comparison["1"][label]["128"]["tokens_per_sec"]
        for _, _, _, label in CONFIGS
    ]
    tps_l4096_b1 = [
        hw_comparison["1"][label]["4096"]["tokens_per_sec"]
        for _, _, _, label in CONFIGS
    ]

    # Roofline data for H100 and TH100 vs TPB
    chips_plot = ['H100', 'TH100 (drop-in)', 'TPB (purpose-built)']
    intensities = [round(10 ** (x / 4), 4) for x in range(-4, 32)]

    h100_ridge  = ridge_point(H100_SXM, 'fp16')
    th100_ridge = ridge_point(TH100,    'bitlinear')
    tpb_ridge   = ridge_point(TPB,      'bitlinear')

    h100_peak   = H100_SXM.tflops_fp16 * 0.5   # TMACS
    th100_peak  = TH100.tflops_fp16 * 0.5
    tpb_peak    = TPB.tflops_fp16 * 0.5
    h100_bw_tbs = H100_SXM.hbm_bandwidth_tbs   # TB/s

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 3 — Memory Bandwidth Roofline Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #0f0f0f; color: #e0e0e0; }
  h1 { color: #7ec8e3; }
  h2 { color: #a8d8a8; margin-top: 40px; }
  .chart-wrap { background: #1a1a1a; border-radius: 8px; padding: 20px; margin: 24px 0; }
  canvas { max-height: 450px; }
  .note { font-size: 0.85em; color: #888; margin: 8px 0 0 0; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th, td { border: 1px solid #333; padding: 8px 12px; text-align: right; }
  th { background: #222; color: #7ec8e3; }
  td:first-child { text-align: left; }
</style>
</head>
<body>
<h1>Phase 3 — Memory Bandwidth Roofline Analysis</h1>
<p>BitNet b1.58 2B4T &nbsp;|&nbsp; Gate-count baseline: Phase 1+2 &nbsp;|&nbsp; Roofline model: ideal overlap assumed</p>

<h2>1. Roofline Plot: Arithmetic Intensity vs Attainable Performance</h2>
<div class="chart-wrap">
<canvas id="rooflineChart"></canvas>
<p class="note">
  Diagonal lines = memory bandwidth ceiling (slope = BW).  Horizontal lines = peak compute ceiling.
  Points = operations at specific (wprec, aprec, batch, L) configurations.
  Everything left of ridge point is memory-bound.
</p>
</div>

<h2>2. Decode Throughput — Batch=1 (Interactive Inference)</h2>
<div class="chart-wrap">
<canvas id="tpsChart"></canvas>
<p class="note">Tokens/sec for batch=1 decode.  All numbers are roofline upper bounds (ideal compute/memory overlap).</p>
</div>

<h2>3. Throughput vs Sequence Length (Batch=1)</h2>
<div class="chart-wrap">
<canvas id="seqChart"></canvas>
<p class="note">Shows how KV cache growth erodes throughput at long context.</p>
</div>

<script>
// ---- Roofline Chart ----
const intensities = """ + json.dumps(intensities) + """;

function rooflineCurve(peakTmacs, bwTbs, intensities) {
  return intensities.map(i => Math.min(peakTmacs, i * bwTbs * 1000));  // bw in TB/s, peak in TMACS → result in GMACS
}

// Operation points
const opPoints = [
  // label, intensity, peakTmacs (chip-specific), color
  {label:"fp16 BitLinear B=1",  x:""" + str(round(bitlinear_intensity('fp16','fp16',1),2)) + """, chip:"H100",  color:"#4488ff"},
  {label:"trit BitLinear B=1",  x:""" + str(round(bitlinear_intensity('trit','int4',1),2)) + """, chip:"H100",  color:"#44ff88"},
  {label:"trit BitLinear B=32", x:""" + str(round(bitlinear_intensity('trit','int4',32),2)) + """, chip:"H100",  color:"#ffcc44"},
  {label:"Attn decode L=2048",  x:""" + str(round(attention_decode_intensity('int4',2048),2)) + """, chip:"H100",  color:"#ff6644"},
  {label:"Attn prefill L=2048", x:""" + str(round(attention_prefill_intensity('int4',2048),2)) + """, chip:"H100",  color:"#cc44ff"},
];

const h100bw = """ + str(H100_SXM.hbm_bandwidth_tbs) + """;
const h100pk = """ + str(round(H100_SXM.tflops_fp16 * 0.5, 1)) + """;
const th100pk= """ + str(round(TH100.tflops_fp16 * 0.5, 1)) + """;

const roofCtx = document.getElementById('rooflineChart').getContext('2d');
new Chart(roofCtx, {
  type: 'scatter',
  data: {
    datasets: [
      {
        label: 'H100 memory ceiling (3.35 TB/s)',
        data: intensities.map((x,i) => ({x: Math.log10(x), y: Math.log10(Math.min(h100pk, x * h100bw * 1000))})),
        type: 'line', borderColor: '#4488ff', backgroundColor: 'transparent',
        pointRadius: 0, borderWidth: 2,
      },
      {
        label: 'TH100/TPB memory ceiling (same BW)',
        data: intensities.map((x,i) => ({x: Math.log10(x), y: Math.log10(Math.min(th100pk, x * h100bw * 1000))})),
        type: 'line', borderColor: '#44ff88', backgroundColor: 'transparent',
        pointRadius: 0, borderWidth: 2, borderDash: [6,3],
      },
      {
        label: 'H100 compute ceiling',
        data: intensities.map(x => ({x: Math.log10(x), y: Math.log10(h100pk)})),
        type: 'line', borderColor: '#4488ff80', backgroundColor: 'transparent',
        pointRadius: 0, borderWidth: 1, borderDash: [2,4],
      },
      {
        label: 'TH100 compute ceiling (~15× H100)',
        data: intensities.map(x => ({x: Math.log10(x), y: Math.log10(th100pk)})),
        type: 'line', borderColor: '#44ff8880', backgroundColor: 'transparent',
        pointRadius: 0, borderWidth: 1, borderDash: [2,4],
      },
      ...opPoints.map(p => ({
        label: p.label,
        data: [{x: Math.log10(p.x), y: Math.log10(Math.min(h100pk, p.x * h100bw * 1000))}],
        type: 'scatter', backgroundColor: p.color, pointRadius: 8,
      }))
    ]
  },
  options: {
    plugins: { legend: { labels: { color: '#ccc', font: { size: 11 } } } },
    scales: {
      x: { title: { display: true, text: 'Arithmetic Intensity log₁₀(ops/byte)', color: '#aaa' },
           ticks: { color: '#aaa', callback: v => `10^${v}` }, grid: { color: '#333' } },
      y: { title: { display: true, text: 'Attainable log₁₀(GMAC/s)', color: '#aaa' },
           ticks: { color: '#aaa', callback: v => `10^${v}` }, grid: { color: '#333' } },
    }
  }
});

// ---- TPS Bar Chart ----
const configLabels = """ + json.dumps([l for _,_,_,l in CONFIGS]) + """;
const tps_l128  = """ + json.dumps([round(tps_l128_b1[i]) for i in range(len(CONFIGS))]) + """;
const tps_l2048 = """ + json.dumps([round(tps_l2048_b1[i]) for i in range(len(CONFIGS))]) + """;
const tps_l4096 = """ + json.dumps([round(tps_l4096_b1[i]) for i in range(len(CONFIGS))]) + """;

const tpsCtx = document.getElementById('tpsChart').getContext('2d');
new Chart(tpsCtx, {
  type: 'bar',
  data: {
    labels: configLabels.map(l => l.length > 28 ? l.slice(0,28)+'…' : l),
    datasets: [
      {label:'L=128',  data:tps_l128,  backgroundColor:'#4488ff88', borderColor:'#4488ff', borderWidth:1},
      {label:'L=2048', data:tps_l2048, backgroundColor:'#44ff8888', borderColor:'#44ff88', borderWidth:1},
      {label:'L=4096', data:tps_l4096, backgroundColor:'#ff664488', borderColor:'#ff6644', borderWidth:1},
    ]
  },
  options: {
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { ticks: { color: '#aaa', maxRotation: 30 }, grid: { color: '#333' } },
      y: { title: { display: true, text: 'Tokens/sec (log scale)', color: '#aaa' },
           type: 'logarithmic', ticks: { color: '#aaa' }, grid: { color: '#333' } }
    }
  }
});

// ---- Throughput vs Seq Length ----
const seqLengths = """ + json.dumps(SEQUENCE_LENGTHS) + """;
const h100fp16_seq   = """ + json.dumps([round(tps(H100_SXM,'fp16','fp16',L,1)) for L in SEQUENCE_LENGTHS]) + """;
const h100trit_seq   = """ + json.dumps([round(tps(H100_SXM,'trit','int4',L,1)) for L in SEQUENCE_LENGTHS]) + """;
const th100_seq      = """ + json.dumps([round(tps(TH100,'trit','int4',L,1)) for L in SEQUENCE_LENGTHS]) + """;
const tpb_seq        = """ + json.dumps([round(tps(TPB,'trit','int4',L,1)) for L in SEQUENCE_LENGTHS]) + """;

const seqCtx = document.getElementById('seqChart').getContext('2d');
new Chart(seqCtx, {
  type: 'line',
  data: {
    labels: seqLengths,
    datasets: [
      {label:'H100 fp16', data:h100fp16_seq, borderColor:'#4488ff', backgroundColor:'transparent', pointRadius:4},
      {label:'H100 trit-w/int4-a', data:h100trit_seq, borderColor:'#88aaff', backgroundColor:'transparent', pointRadius:4, borderDash:[4,3]},
      {label:'TH100 drop-in', data:th100_seq, borderColor:'#44ff88', backgroundColor:'transparent', pointRadius:4},
      {label:'TPB purpose-built', data:tpb_seq, borderColor:'#ffcc44', backgroundColor:'transparent', pointRadius:4},
    ]
  },
  options: {
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { title: { display: true, text: 'Context length (tokens)', color: '#aaa' },
           ticks: { color: '#aaa' }, grid: { color: '#333' } },
      y: { title: { display: true, text: 'Tokens/sec (log scale)', color: '#aaa' },
           type: 'logarithmic', ticks: { color: '#aaa' }, grid: { color: '#333' } }
    }
  }
});
</script>
</body>
</html>
"""
    path = os.path.join(os.path.dirname(__file__), "visualization.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Phase 3: Memory Bandwidth Modeling — BitNet b1.58 2B4T")
    print("=" * 72)

    # Key sanity-check numbers
    print(f"\nModel weight sizes:")
    print(f"  fp16: {total_weight_bytes('fp16')/1e6:.0f} MB")
    print(f"  trit: {total_weight_bytes('trit')/1e6:.0f} MB  ({total_weight_bytes('fp16')/total_weight_bytes('trit'):.1f}× smaller)")

    print(f"\nRidge points (ops/byte):")
    for chip in [H100_SXM, MI300X, TH100, TPB]:
        r = ridge_point(chip, 'fp16' if chip in (H100_SXM, MI300X) else 'bitlinear')
        print(f"  {chip.short:<26}: {r:.0f}")

    print(f"\nDecode throughput (batch=1, L=2048, tokens/sec):")
    for chip, wprec, aprec, label in CONFIGS:
        r = decode_roofline(chip, wprec, aprec, 2048, 1)
        print(f"  {label:<38}: {r['tokens_per_sec']:>8,.0f}  [{r['bottleneck']}]")

    print()
    intensity_data, roofline_data, hw_comparison = run_all()
    print("JSON results written to results/")

    write_report()
    print("PHASE3_REPORT.md written")

    write_visualization(hw_comparison)
    print("visualization.html written")


if __name__ == "__main__":
    main()
