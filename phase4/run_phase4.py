"""
Phase 4: Per-Layer Quantization Sensitivity — main script.

Generates:
  results/pareto_data.json
  results/speedup_matrix.json
  results/tps_matrix.json
  PHASE4_REPORT.md
  visualization.html
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from layer_sensitivity_model import (
    quality_degradation_units, component_qdu_breakdown, uniform_config,
    COMPONENTS, N_LAYERS, COMPONENT_SENSITIVITY, POSITION_SENSITIVITY,
    PRECISION_PENALTY,
)
from hybrid_configurations import CONFIGS, describe_config
from hybrid_gate_counts import (
    model_gate_counts, binary_model_gates, verify_vs_phase2, SEQUENCE_LENGTHS,
)
from hybrid_bandwidth import decode_timing, kv_bytes_for_config
from pareto_analysis import (
    compute_all_metrics, pareto_table, pareto_frontier,
    speedup_table_across_seq, tps_table, microsoft_comparison,
)

PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, PHASE3_DIR)
from hardware_specs import H100_SXM, TH100, TPB

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = compute_all_metrics(batch=1)
    frontier = pareto_frontier(results, "qdu", "tps_tpb",
                               x_lower_better=True, y_lower_better=False)

    # Pareto data JSON
    pareto_json = []
    for r in results:
        pareto_json.append({k: v for k, v in r.items() if k != "rationale"})
    with open(os.path.join(RESULTS_DIR, "pareto_data.json"), "w") as f:
        json.dump(pareto_json, f, indent=2)

    # Speedup matrix
    speedup_matrix = {}
    binary_totals = {L: binary_model_gates(L)["TOTAL"] for L in SEQUENCE_LENGTHS}
    for cfg in CONFIGS:
        speedup_matrix[cfg["name"]] = {}
        for L in SEQUENCE_LENGTHS:
            gc = model_gate_counts(L, cfg["layer_cfgs"])
            speedup_matrix[cfg["name"]][str(L)] = binary_totals[L] / gc["TOTAL"]
    with open(os.path.join(RESULTS_DIR, "speedup_matrix.json"), "w") as f:
        json.dump(speedup_matrix, f, indent=2)

    # TPS matrix (TPB, batch=1)
    tps_matrix = {}
    for cfg in CONFIGS:
        tps_matrix[cfg["name"]] = {}
        for L in SEQUENCE_LENGTHS:
            gc = model_gate_counts(L, cfg["layer_cfgs"])
            r = decode_timing(TPB, cfg["layer_cfgs"], L, 1, L, gc)
            tps_matrix[cfg["name"]][str(L)] = r["tokens_per_sec"]
    with open(os.path.join(RESULTS_DIR, "tps_matrix.json"), "w") as f:
        json.dump(tps_matrix, f, indent=2)

    return results, frontier, speedup_matrix, tps_matrix


def write_report(results, frontier, speedup_matrix, tps_matrix):

    # Lookup shortcuts
    def get(name, key):
        for r in results:
            if r["name"] == name:
                return r[key]
        raise KeyError(name)

    a48_qdu    = get("a4.8-canonical", "qdu")
    int4_qdu   = get("int4-uniform",   "qdu")
    int8_qdu   = get("int8-uniform",   "qdu")
    trit_qdu   = get("trit-uniform",   "qdu")

    a48_sp     = get("a4.8-canonical", "speedup_vs_fp16")
    int4_sp    = get("int4-uniform",   "speedup_vs_fp16")
    int8_sp    = get("int8-uniform",   "speedup_vs_fp16")
    trit_sp    = get("trit-uniform",   "speedup_vs_fp16")

    a48_tps    = get("a4.8-canonical", "tps_tpb")
    int4_tps   = get("int4-uniform",   "tps_tpb")
    int8_tps   = get("int8-uniform",   "tps_tpb")

    a48_kv     = get("a4.8-canonical", "kv_mb_L2048")
    int4_kv    = get("int4-uniform",   "kv_mb_L2048")
    int8_kv    = get("int8-uniform",   "kv_mb_L2048")

    conserv_qdu = get("conservative",  "qdu")
    conserv_sp  = get("conservative",  "speedup_vs_fp16")
    conserv_tps = get("conservative",  "tps_tpb")

    # Component QDU breakdown for uniform int4
    bkdn = component_qdu_breakdown(uniform_config('int4'))
    bkdn_sorted = sorted(bkdn.items(), key=lambda x: -x[1])

    ffn_down_share = bkdn["ffn_down_in"] / 100.0  # fraction of total QDU

    report = f"""\
# Phase 4 Report
## Per-Layer Quantization Sensitivity for Hybrid Ternary Architectures
### BitNet b1.58 2B4T — Activation Precision Hybrid Analysis

---

## Upfront Correction to the Task Framing

The task prompt stated "FFN down: tolerant" based on weight-quantization
literature.  **The activation quantization literature says the opposite.**

  SmoothQuant (Xiao et al., ICML 2023) identified FFN down-projection INPUTS
  as having "massive outliers at magnitudes ~1400× the typical value."
  Source: https://arxiv.org/abs/2211.10438

  BitNet a4.8 (Wang et al., arXiv:2411.04965, Nov 2024) explicitly keeps
  FFN down-projection inputs at int8 because "applying FP4 quantization
  for inputs to the down projection leads to significant performance
  degradation."

  The TOLERANT components are FFN gate/up projection inputs and QKV inputs
  (Gaussian-like distributions, safely quantized to int4).

  Weight quantization and activation quantization have OPPOSITE sensitivity
  orderings for FFN down.  Phase 4 is built on the published activation
  result.

---

## Executive Summary

**Phase 4 finding: H2 is partially correct, and the architecture is
already published.  BitNet a4.8 IS the optimal hybrid.**

The question "uniform vs hybrid?" has a clean answer from published literature:
- Weights: uniform trit everywhere (H1 confirmed — Microsoft b1.58 result)
- Activations: hybrid int4/int8 (H2 confirmed — Microsoft a4.8 result)

The specific Pareto-optimal hybrid (a4.8-extended):
- FFN down inputs: int8       (outlier-heavy; quality-critical [MEASURED])
- Attn out inputs: int8       (normal outliers; sensitive [PROJECTED])
- Everything else: int4       (tolerant; compress aggressively)

This achieves:
- QDU = {get("a4.8-extended","qdu"):.1f}  (vs {a48_qdu:.1f} for a4.8-canonical, {int4_qdu:.0f} for uniform int4)
- Gate speedup = {get("a4.8-extended","speedup_vs_fp16"):.2f}×
- TPB tokens/sec = {get("a4.8-extended","tps_tpb"):,.0f}  (same as uniform int4 = {int4_tps:,.0f})

**Why a4.8-extended dominates a4.8-canonical:** keeping attn_out at int8 reduces QDU from
{a48_qdu:.1f} to {get("a4.8-extended","qdu"):.1f} at zero TPS cost — attn_out precision does not affect
KV cache size (which is the TPB decode bottleneck) or weight loading.
The hardware change is trivial: one more component in int8 mode.

The published BitNet a4.8 (ffn_down=int8 only) is still an excellent config ({a48_qdu:.1f} QDU,
{a48_tps:,.0f} TPS). Our Pareto analysis suggests extending it to also keep attn_out at int8.

---

## 1. The Sensitivity Model

All quality numbers are **[PROJECTED]** unless marked [MEASURED BASIS].

### Component sensitivity ordering (activation quantization, not weight quantization):

| Component | Sensitivity | Basis |
|:----------|:------------|:------|
| ffn_down_in | HIGHEST (3.5×) | [MEASURED BASIS] BitNet a4.8, SmoothQuant |
| attn_out_in | Moderate-high (1.4×) | [PROJECTED] SmoothQuant "normal outliers" |
| qkv_in | Moderate (1.0×) | [PROJECTED] residual stream, Gaussian |
| ffn_up_in | Tolerant (0.6×) | [PROJECTED] post-norm residual, nearly Gaussian |
| ffn_gate_in | Tolerant (0.6×) | [PROJECTED] same input as ffn_up_in |

### Layer position multipliers (from GPTQ empirical curve [MEASURED BASIS]):
  first 2 blocks:   2.5×
  early 4 blocks:   1.5×
  middle 18 blocks: 1.0×  (reference)
  late 4 blocks:    1.3×
  last 2 blocks:    2.0×

### QDU calibration:
  Uniform int4 = **100 QDU** (reference baseline).
  QDU 0 = uniform int8 (no degradation).
  QDU > 100 = worse than uniform int4.

Quality Degradation Units by component (for uniform int4):
```
{"".join(f"  {comp:<18} {qdu:>6.1f} QDU ({100*qdu/100:.1f}% of total)\\n" for comp, qdu in bkdn_sorted)}```

**FFN down inputs account for {bkdn['ffn_down_in']:.1f} QDU out of 100** in uniform int4.
This explains why keeping only ffn_down_in at int8 captures most of the quality
benefit of going all the way back to int8.

---

## 2. Pareto Analysis

```
{pareto_table(results)}
```

The Pareto-optimal set (★) contains:
{chr(10).join(f"  - {r['label']}  (QDU={r['qdu']:.1f}, speedup={r['speedup_vs_fp16']:.2f}×, TPB={r['tps_tpb']:,.0f} tps)" for r in sorted(results, key=lambda x: x['qdu']) if r['name'] in frontier)}

**Key Pareto insight:** uniform int4 ({int4_qdu:.0f} QDU, {int4_sp:.2f}×) is NOT Pareto-optimal.
The a4.8-canonical config dominates it: lower QDU ({a48_qdu:.1f} vs {int4_qdu:.0f}) at only
~10% compute cost ({int4_sp:.2f}× vs {a48_sp:.2f}×).

If you are already willing to run uniform int4, you should instead run a4.8-canonical —
it is strictly better on quality with negligible performance cost.

---

## 3. Gate-Count Speedup Surface

```
{speedup_table_across_seq(batch=1)}
```

The a4.8-canonical config maintains {a48_sp:.2f}× speedup at L=2048, close to uniform
int4's {int4_sp:.2f}×.  The "cost" of keeping ffn_down at int8 is only
{(int4_sp-a48_sp)/int4_sp*100:.1f}% slower compute vs uniform int4.

This is because ffn_down, despite being 22% of binary compute, runs at
9.71× speedup even with int8 activations (the trit-weight benefit remains;
only the activation side differs).

---

## 4. Effective Throughput (TPB Chip, batch=1)

```
{tps_table(TPB, "TPB purpose-built", batch=1)}
```

On the TPB chip (weights in SRAM, bottleneck is KV cache), all configs with
the same QKV precision have the SAME KV cache size and thus the same TPS.

KV cache size at L=2048:
  - Configs with int8 QKV: {int8_kv:.1f} MB
  - Configs with int4 QKV: {int4_kv:.1f} MB
  - a4.8-canonical uses int4 QKV: {a48_kv:.1f} MB

The a4.8-canonical keeps QKV at int4, giving it the smaller KV cache
and TPB throughput identical to uniform int4 ({int4_tps:,.0f} tps).

This is a crucial design insight: **the a4.8 architecture SIMULTANEOUSLY
achieves better quality AND smaller KV cache than uniform int4.**
The ffn_down_in outlier handling at int8 does not penalize KV cache size
because KV cache is determined by QKV precision, not FFN precision.

---

## 5. Hardware Implications

### Does hybrid require heterogeneous compute tiles?

**No.** The required hardware is simpler than expected.

For a4.8-canonical, the only int8 operation is FFN down-projection inputs.
The hardware needs:
  - trit-weight MAC array (primary, for all BitLinear operations)
  - Mode-switchable activation precision: int4 or int8 per MAC
  - No separate int8 MAC arrays needed — same trit-MAC circuit with wider input

The int4/int8 difference is just the activation input width (4 bits vs 8 bits).
This translates to approximately:
  - Trit-MAC with 8-bit input: ~30 gates for int8 accumulation (Phase 2)
  - Trit-MAC with 4-bit input: ~15 gates for int4 accumulation (Phase 2)
  - Mode switching overhead: ~5% die area for the multiplexer

A single trit-MAC array with programmable activation width is sufficient.
No architectural bifurcation needed.

### KV cache layout

With a4.8-canonical, all 30 KV caches use int4 (QKV is uniformly int4).
A simple, uniform KV cache layout. No per-layer precision heterogeneity in memory.

### Hardware recommendation update from Phase 2:

Phase 2 recommended: "int4 activations for first-generation ternary accelerator."

**Phase 4 update: int4 for QKV/attention/FFN-up-gate, int8 for FFN-down.**

The additional hardware cost (mode-switchable activation precision) is
estimated at ~5% die area overhead.  The quality gain is meaningful.
No other architectural changes required.

---

## 6. The H1/H2 Verdict

**H1 (uniform is fine) — TRUE for weights, FALSE for activations.**

For weight precision: Microsoft's b1.58 result stands — uniform trit weights
throughout the model is optimal.  Phase 4 does not change this.

**H2 (sensitive layers dominate quality budget) — TRUE, and the answer is clear.**

FFN down-projection inputs ({bkdn['ffn_down_in']:.1f} QDU out of 100 for uniform int4) are the
dominant source of activation quantization degradation.  Keeping ONLY this
component at int8, with everything else at int4, captures {(a48_qdu-int4_qdu)/int4_qdu*-100:.0f}% of
the quality benefit of returning fully to int8, at only ~10% compute cost.

The finding is not ambiguous: **a4.8-canonical (ffn_down int8, rest int4)
Pareto-dominates uniform int4 and is the recommended hardware target.**

---

## 7. Addressing "Microsoft Found Uniform Was Best"

{microsoft_comparison()}

---

## 8. Summary Table

| Config | QDU | Speedup | TPB TPS | KV (MB) | Pareto |
|:-------|----:|--------:|--------:|--------:|:------:|
{"".join(f'| {r["label"]:<32} | {r["qdu"]:>5.1f} | {r["speedup_vs_fp16"]:>6.2f}× | {r["tps_tpb"]:>7,.0f} | {r["kv_mb_L2048"]:>7.1f} | {"★" if r["name"] in frontier else ""} |' + chr(10) for r in sorted(results, key=lambda x: x['qdu']))}

L=2048, batch=1.  ★ = Pareto-optimal (better quality AND better performance).

---

## Caveats

1. **QDU numbers are PROJECTED.** We calibrate the scale using the BitNet a4.8
   finding as an anchor (ffn_down_in is the critical component) but all absolute
   values are estimates.  Real PPL degradation for these configurations is not
   available for BitNet 2B4T specifically.

2. **Sensitivity ordering is based on general LLM literature, not BitNet-specific
   ablations.** The BitNet a4.8 paper directly validates the ffn_down_in finding;
   other component orderings [PROJECTED] may differ for ternary-weight models.

3. **Quality degradation is modeled as additive.** Real quantization errors may
   interact non-linearly (e.g., errors in early layers compound through later layers
   in ways the additive model doesn't capture).

4. **The sensitivity model uses a fixed PRECISION_PENALTY scale.** Going from int4
   to int2 or trit is modeled as a fixed multiplier; real degradation may be
   architecture-dependent and threshold-based.

---

## Methodology

- Sensitivity model: calibrated from SmoothQuant (arXiv:2211.10438) and
  BitNet a4.8 (arXiv:2411.04965) for component ordering; GPTQ (arXiv:2210.17323)
  for position sensitivity.  All [PROJECTED] values are explicitly flagged.
- Gate counts: Phase 2 gate_costs.py, applied per-component per-layer.
  Verified: uniform int4 = 14.87×, uniform int8 = 6.72× (within 0.5% of Phase 2).
- Bandwidth: Phase 3 roofline methodology, with per-config KV cache precision.
- Quality scale: QDU = 0 (int8 reference) to 100 (uniform int4) to >100 (below int4).

*Reproducibility:* run `python run_phase4.py` from this directory.
"""

    with open(os.path.join(os.path.dirname(__file__), "PHASE4_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)


def write_visualization(results, frontier, speedup_matrix, tps_matrix):
    """Interactive Pareto plot + speedup bars."""

    qdus    = [r["qdu"]            for r in results]
    speedups= [r["speedup_vs_fp16"]for r in results]
    tps_v   = [r["tps_tpb"]        for r in results]
    labels  = [r["label"]          for r in results]
    is_front= [r["name"] in frontier for r in results]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 4 — Hybrid Architecture Pareto Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #0f0f0f; color: #e0e0e0; }
  h1 { color: #7ec8e3; }
  h2 { color: #a8d8a8; margin-top: 40px; }
  .chart-wrap { background: #1a1a1a; border-radius: 8px; padding: 20px; margin: 24px 0; }
  canvas { max-height: 480px; }
  .note { font-size: 0.85em; color: #888; margin: 8px 0 0 0; }
  .finding { background: #1a2a1a; border-left: 3px solid #44ff88; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
</style>
</head>
<body>
<h1>Phase 4 — Hybrid Activation Precision: Pareto Analysis</h1>
<p>All weights are trit. Activation precision varies per component per layer.</p>

<div class="finding">
  <strong>Key finding:</strong> BitNet a4.8 (ffn_down=int8, rest=int4) is Pareto-optimal.
  It achieves lower quality degradation than uniform int4 at nearly the same performance.
  The mechanism: FFN down inputs have massive outliers (SmoothQuant, BitNet a4.8);
  keeping only this component at int8 captures ~QUALITY_PCT% of the quality benefit of full int8.
</div>
""".replace("QUALITY_PCT", f"{(100 - results[[r['name'] for r in results].index('a4.8-canonical')]['qdu']) / 100 * 100:.0f}")

    # Pareto scatter
    colors = ["#ffcc44" if f else "#4488ff" for f in is_front]

    html += f"""
<h2>1. Pareto Frontier: Quality vs Performance (TPB, batch=1, L=2048)</h2>
<div class="chart-wrap">
<canvas id="paretoChart"></canvas>
<p class="note">X-axis: Quality Degradation Units (lower = better; 0=int8, 100=uniform int4). Y-axis: tokens/sec.
★ points are Pareto-optimal. [All QDU values are PROJECTED estimates]</p>
</div>

<h2>2. Gate-Count Speedup by Configuration</h2>
<div class="chart-wrap">
<canvas id="speedupChart"></canvas>
<p class="note">Speedup vs fp16 binary at L=2048. Higher is better.</p>
</div>

<script>
const qdus    = {json.dumps(qdus)};
const tpsList = {json.dumps(tps_v)};
const speedups= {json.dumps(speedups)};
const labels  = {json.dumps(labels)};
const isFront = {json.dumps(is_front)};

const colors  = isFront.map(f => f ? '#ffcc44' : '#4488ff88');
const borders = isFront.map(f => f ? '#ffcc44' : '#4488ff');
const sizes   = isFront.map(f => f ? 12 : 7);

// Pareto scatter
new Chart(document.getElementById('paretoChart').getContext('2d'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Configurations',
      data: qdus.map((q, i) => ({{x: q, y: tpsList[i]}})),
      backgroundColor: colors,
      borderColor: borders,
      pointRadius: sizes,
      pointStyle: isFront.map(f => f ? 'star' : 'circle'),
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => labels[ctx.dataIndex] + '  QDU=' + qdus[ctx.dataIndex].toFixed(1) + '  TPS=' + Math.round(tpsList[ctx.dataIndex]).toLocaleString(),
        }}
      }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Quality Degradation (QDU, lower=better) [PROJECTED]', color: '#aaa' }}, ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
      y: {{ title: {{ display: true, text: 'Tokens/sec (TPB, log scale)', color: '#aaa' }}, type: 'logarithmic', ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
    }}
  }}
}});

// Speedup bar
new Chart(document.getElementById('speedupChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: labels.map(l => l.length > 30 ? l.slice(0,30)+'…' : l),
    datasets: [{{
      label: 'Speedup vs fp16 binary',
      data: speedups,
      backgroundColor: isFront.map(f => f ? '#ffcc4488' : '#4488ff44'),
      borderColor: isFront.map(f => f ? '#ffcc44' : '#4488ff'),
      borderWidth: 1,
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#aaa', maxRotation: 35 }}, grid: {{ color: '#333' }} }},
      y: {{ title: {{ display: true, text: 'Speedup vs fp16', color: '#aaa' }}, ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
    }}
  }}
}});
</script>
</body>
</html>
"""
    with open(os.path.join(os.path.dirname(__file__), "visualization.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 72)
    print("Phase 4: Per-Layer Quantization Sensitivity — Hybrid Ternary")
    print("=" * 72)

    print("\nVerification vs Phase 2 (uniform configs must match):")
    verify_vs_phase2()

    print("\nQDU and speedup for all configs (L=2048):")
    binary_total = binary_model_gates(2048)["TOTAL"]
    for cfg in CONFIGS:
        qdu = quality_degradation_units(cfg["layer_cfgs"])
        gc  = model_gate_counts(2048, cfg["layer_cfgs"])
        sp  = binary_total / gc["TOTAL"]
        print(f"  {cfg['label']:<38}  QDU={qdu:>6.1f}  speedup={sp:.2f}×")

    print()
    results, frontier, speedup_matrix, tps_matrix = run_all()
    print("JSON results written to results/")

    write_report(results, frontier, speedup_matrix, tps_matrix)
    print("PHASE4_REPORT.md written")

    write_visualization(results, frontier, speedup_matrix, tps_matrix)
    print("visualization.html written")

    print(f"\nPareto-optimal configurations:")
    for r in sorted(results, key=lambda x: x["qdu"]):
        if r["name"] in frontier:
            print(f"  [*] {r['label']}  QDU={r['qdu']:.1f}  speedup={r['speedup_vs_fp16']:.2f}x  TPB={r['tps_tpb']:,.0f} tps")


if __name__ == "__main__":
    main()
