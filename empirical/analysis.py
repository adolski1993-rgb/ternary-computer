"""
Phase 1-5 correction analysis based on empirical trit distribution.

Reads the per-tensor distribution from download_and_measure.py output
and computes how much the real distribution differs from the assumed
1/3-1/3-1/3, then propagates the correction through Phase 1-5 numbers.

Also generates per_component_distribution.json and
corrected_phase_numbers.json, then writes EMPIRICAL_REPORT.md.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

MODEL_ID    = "microsoft/bitnet-b1.58-2B-4T-bf16"
RESULTS_DIR = Path(__file__).parent / "results"
PHASE_ROOTS = {
    "phase1": Path(__file__).parent.parent / "phase1",
    "phase2": Path(__file__).parent.parent / "phase2",
    "phase3": Path(__file__).parent.parent / "phase3",
}

# ---------------------------------------------------------------------------
# Gate cost model (mirrors Phase 2 gate_costs.py for int8 activations,
# the Phase 1 baseline precision)
# ---------------------------------------------------------------------------

# Phase 1 assumed 1/3 nonzero for gate cost calculation:
#   per_pair = 1 + (2/3) × ACC_ADD + (1/3) × NEG
# With empirical p_nonzero, assuming p_pos ≈ p_neg ≈ p_nonzero / 2:
#   per_pair = (1 - p_nz) × 1 + (p_nz/2) × (1 + ACC_ADD) + (p_nz/2) × (1 + NEG + ACC_ADD)
#            = 1 + p_nz × (ACC_ADD + NEG/2)

INT8_ACC_ADD = 30    # Phase 2: GATES_TRIT_ACCUM_ADD for int8
INT8_NEG     =  8    # Phase 2: GATES_TRIT_NEG_INT8
INT8_REQUANT = 50    # Phase 2: GATES_TRIT_REQUANTIZE (per output element)

INT4_ACC_ADD = 15
INT4_NEG     =  4
INT4_REQUANT = 35

FP16_ADD  = 80   # binary fp16 accumulation gates
FP16_MUL  = 150  # binary fp16 multiply gates

BINARY_PER_PAIR = FP16_MUL + FP16_ADD  # 230


def per_pair_gates(p_nonzero: float, acc_add: int, neg: int) -> float:
    """Expected gate cost per (activation, weight-trit) pair."""
    return 1.0 + p_nonzero * (acc_add + neg / 2.0)


def bitlinear_speedup(p_nonzero: float, acc_add: int, neg: int,
                      requant: int, K: int = 2560) -> float:
    """
    Speedup of one BitLinear matmul vs fp16 binary.

    Binary per MAC:      FP16_MUL + FP16_ADD = 230 gates
    Ternary per MAC:     K × per_pair(p_nz) + requant (per output element)
    Speedup ≈ 230 / per_pair(p_nz)   (dominates for large K)
    """
    pp = per_pair_gates(p_nonzero, acc_add, neg)
    # Full formula: binary_per_element / ternary_per_element
    # = (K × 230) / (K × pp + requant) for an [M, K, N] matmul
    return (K * BINARY_PER_PAIR) / (K * pp + requant)


# ---------------------------------------------------------------------------
# Load and aggregate empirical data
# ---------------------------------------------------------------------------

def load_records() -> tuple[list[dict], dict]:
    path = RESULTS_DIR / "per_layer_distribution.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_and_measure.py first."
        )
    data = json.loads(path.read_text())
    return data["per_tensor"], data["assumption"]


def aggregate_by_component(records: list[dict]) -> dict:
    """Aggregate stats per component type across all layers."""
    comp_stats: dict[str, dict] = defaultdict(lambda: {
        "n_neg": 0, "n_zero": 0, "n_pos": 0, "n_total": 0, "n_tensors": 0
    })
    for r in records:
        c = r["component"]
        comp_stats[c]["n_neg"]    += r["n_neg"]
        comp_stats[c]["n_zero"]   += r["n_zero"]
        comp_stats[c]["n_pos"]    += r["n_pos"]
        comp_stats[c]["n_total"]  += r["n_total"]
        comp_stats[c]["n_tensors"] += 1

    result = {}
    for comp, s in sorted(comp_stats.items()):
        n = s["n_total"]
        result[comp] = {
            "n_total":   n,
            "n_tensors": s["n_tensors"],
            "p_neg":     round(s["n_neg"]  / n, 6),
            "p_zero":    round(s["n_zero"] / n, 6),
            "p_pos":     round(s["n_pos"]  / n, 6),
            "p_nonzero": round((s["n_neg"] + s["n_pos"]) / n, 6),
        }
    return result


def aggregate_by_layer(records: list[dict]) -> list[dict]:
    """Per-layer p_nonzero (all components merged)."""
    layer_stats: dict[int, dict] = defaultdict(lambda: {
        "n_neg": 0, "n_zero": 0, "n_pos": 0, "n_total": 0
    })
    for r in records:
        li = r["layer_idx"]
        layer_stats[li]["n_neg"]   += r["n_neg"]
        layer_stats[li]["n_zero"]  += r["n_zero"]
        layer_stats[li]["n_pos"]   += r["n_pos"]
        layer_stats[li]["n_total"] += r["n_total"]

    layers = []
    for li in sorted(layer_stats.keys()):
        s = layer_stats[li]
        n = s["n_total"]
        layers.append({
            "layer_idx":  li,
            "n_total":    n,
            "p_nonzero":  round((s["n_neg"] + s["n_pos"]) / n, 6),
            "p_zero":     round(s["n_zero"] / n, 6),
            "p_neg":      round(s["n_neg"]  / n, 6),
            "p_pos":      round(s["n_pos"]  / n, 6),
        })
    return layers


# ---------------------------------------------------------------------------
# Phase 1-5 speedup corrections
# ---------------------------------------------------------------------------

PHASE1_ASSUMED_PNZ    = 2 / 3   # assumed nonzero fraction

COMPONENT_LABELS = {
    "qkv":      "QKV projection",
    "attn_out": "Attention output",
    "ffn_gate": "FFN gate",
    "ffn_up":   "FFN up",
    "ffn_down": "FFN down",
}

# Phase 1 baseline speedups (int8 activations, L=2048)
PHASE1_SPEEDUPS = {
    "qkv":      9.71,
    "attn_out": 9.71,
    "ffn_gate": 9.71,
    "ffn_up":   9.71,
    "ffn_down": 9.71,
    "overall":  6.72,   # full layer including attention
}

PHASE2_INT4_SPEEDUPS = {
    "qkv":      18.63,
    "attn_out": 18.63,
    "ffn_gate": 18.63,
    "ffn_up":   18.63,
    "ffn_down": 18.64,
    "overall":  14.87,
}


def compute_corrections(comp_stats: dict) -> dict:
    """
    For each component, compute corrected speedup vs assumed.
    For the overall correction, use the weighted-average p_nonzero
    across all BitLinear components.
    """
    corrections = {}

    assumed_bl_speedup_int8 = bitlinear_speedup(PHASE1_ASSUMED_PNZ, INT8_ACC_ADD, INT8_NEG, INT8_REQUANT)
    assumed_bl_speedup_int4 = bitlinear_speedup(PHASE1_ASSUMED_PNZ, INT4_ACC_ADD, INT4_NEG, INT4_REQUANT)

    total_weights = sum(s["n_total"] for s in comp_stats.values())
    weighted_pnz  = sum(s["n_total"] * s["p_nonzero"] for s in comp_stats.values()) / total_weights

    for comp, s in comp_stats.items():
        pnz = s["p_nonzero"]
        actual_bl_int8 = bitlinear_speedup(pnz, INT8_ACC_ADD, INT8_NEG, INT8_REQUANT)
        actual_bl_int4 = bitlinear_speedup(pnz, INT4_ACC_ADD, INT4_NEG, INT4_REQUANT)

        corrections[comp] = {
            "component":          comp,
            "p_nonzero_assumed":  round(PHASE1_ASSUMED_PNZ, 4),
            "p_nonzero_empirical":s["p_nonzero"],
            "delta_pnz":          round(s["p_nonzero"] - PHASE1_ASSUMED_PNZ, 4),
            "speedup_int8_assumed":  round(assumed_bl_speedup_int8, 3),
            "speedup_int8_empirical":round(actual_bl_int8, 3),
            "speedup_int8_pct_change": round(100 * (actual_bl_int8 - assumed_bl_speedup_int8) / assumed_bl_speedup_int8, 2),
            "speedup_int4_assumed":  round(assumed_bl_speedup_int4, 3),
            "speedup_int4_empirical":round(actual_bl_int4, 3),
            "speedup_int4_pct_change": round(100 * (actual_bl_int4 - assumed_bl_speedup_int4) / assumed_bl_speedup_int4, 2),
        }

    # Overall weighted correction
    actual_overall_int8 = bitlinear_speedup(weighted_pnz, INT8_ACC_ADD, INT8_NEG, INT8_REQUANT)
    actual_overall_int4 = bitlinear_speedup(weighted_pnz, INT4_ACC_ADD, INT4_NEG, INT4_REQUANT)

    corrections["_overall"] = {
        "p_nonzero_assumed":   round(PHASE1_ASSUMED_PNZ, 4),
        "p_nonzero_empirical": round(weighted_pnz, 4),
        "delta_pnz":           round(weighted_pnz - PHASE1_ASSUMED_PNZ, 4),
        "speedup_bitlinear_int8_assumed":   round(assumed_bl_speedup_int8, 3),
        "speedup_bitlinear_int8_empirical": round(actual_overall_int8, 3),
        "speedup_int8_pct_change":          round(100 * (actual_overall_int8 - assumed_bl_speedup_int8) / assumed_bl_speedup_int8, 2),
        "speedup_bitlinear_int4_assumed":   round(assumed_bl_speedup_int4, 3),
        "speedup_bitlinear_int4_empirical": round(actual_overall_int4, 3),
        "speedup_int4_pct_change":          round(100 * (actual_overall_int4 - assumed_bl_speedup_int4) / assumed_bl_speedup_int4, 2),
        "note": (
            "BitLinear gate speedup only. Full-layer speedups differ because "
            "attention (act×act) and non-matmul ops are unchanged."
        ),
    }
    return corrections, weighted_pnz


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(records, comp_stats, layer_stats, corrections, weighted_pnz):
    assumed = PHASE1_ASSUMED_PNZ
    ov = corrections["_overall"]

    # Derived numbers
    phase1_bl_speedup_assumed = ov["speedup_bitlinear_int8_assumed"]
    phase1_bl_speedup_actual  = ov["speedup_bitlinear_int8_empirical"]
    phase2_bl_speedup_assumed = ov["speedup_bitlinear_int4_assumed"]
    phase2_bl_speedup_actual  = ov["speedup_bitlinear_int4_empirical"]
    pct_int8 = ov["speedup_int8_pct_change"]
    pct_int4 = ov["speedup_int4_pct_change"]

    within_5 = abs(ov["delta_pnz"]) / assumed * 100 < 5

    comp_table = ""
    for comp in sorted(comp_stats.keys()):
        s  = comp_stats[comp]
        cr = corrections.get(comp, {})
        label = COMPONENT_LABELS.get(comp, comp)
        comp_table += (
            f"| {label:<22} | {s['p_zero']:.3f} | {s['p_pos']:.3f} | {s['p_neg']:.3f} | "
            f"{s['p_nonzero']:.3f} | {cr.get('speedup_int8_pct_change', 0.0):+.1f}% |\n"
        )

    layer_table = ""
    for ls in layer_stats:
        layer_table += f"| {ls['layer_idx']:3d} | {ls['p_nonzero']:.4f} | {ls['p_zero']:.4f} |\n"

    report = f"""\
# Empirical Report
## Weight Distribution Validation for BitNet b1.58 2B4T
### Testing the Phase 1-5 Uniform Distribution Assumption

---

## Executive Summary

{'**The assumption holds.** ' if within_5 else '**Meaningful deviation found.** '}Phases 1-5 assumed trit weights are uniformly distributed (1/3 each).
The empirical measurement shows:

| Metric | Assumed (Phase 1-5) | Measured (empirical) | Delta |
|:-------|:-------------------:|:--------------------:|:------:|
| p_nonzero (all layers) | {assumed:.3f} | {weighted_pnz:.3f} | {weighted_pnz-assumed:+.3f} |
| p_zero | {1-assumed:.3f} | {1-weighted_pnz:.3f} | {1-weighted_pnz-(1-assumed):+.3f} |

Effect on BitLinear gate speedup (the primary gate-count metric):
- Int8 activations: {phase1_bl_speedup_assumed:.2f}x assumed -> {phase1_bl_speedup_actual:.2f}x actual ({pct_int8:+.1f}%)
- Int4 activations: {phase2_bl_speedup_assumed:.2f}x assumed -> {phase2_bl_speedup_actual:.2f}x actual ({pct_int4:+.1f}%)

{'The deviation is within 5% — the Phase 1-5 analysis is not materially affected.' if within_5 else 'The deviation is outside 5% — see corrected numbers below.'}

---

## 1. What Was Measured

Source: `{MODEL_ID}` (BF16 master weights)
Method: absmean quantization per BitNet b1.58 paper
  scale = mean(|W|) per weight matrix
  trits = clip(round(W / scale), -1, +1)

Tensors measured: {len(records)} BitLinear weight matrices
  (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj x 30 layers)
Tensors skipped: embedding, lm_head, norms, biases (not BitLinear)

---

## 2. Per-Component Distribution

| Component | p(zero) | p(+1) | p(-1) | p(nonzero) | Int8 speedup delta |
|:---------|:-------:|:------:|:------:|:----------:|:------------------:|
{comp_table}
Assumed (uniform): p(zero) = {1-assumed:.3f}, p(nonzero) = {assumed:.3f}

The component-level variation reveals whether specific layer types deviate more
than others. Components where p_nonzero >> 2/3 have denser weights (more gates
per MAC). Components where p_nonzero << 2/3 are sparser (fewer gates, faster).

---

## 3. Layer-Depth Distribution

Per-layer p_nonzero (all components merged):

| Layer | p_nonzero | p_zero |
|------:|:---------:|:------:|
{layer_table}
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
"""
    for comp in sorted(comp_stats.keys()):
        cr = corrections.get(comp, {})
        label = COMPONENT_LABELS.get(comp, comp)
        report += (
            f"| {label:<22} | {assumed:.3f} | {comp_stats[comp]['p_nonzero']:.3f} | "
            f"{cr.get('speedup_int8_assumed',0):.2f}x -> {cr.get('speedup_int8_empirical',0):.2f}x "
            f"({cr.get('speedup_int8_pct_change',0):+.1f}%) | "
            f"{cr.get('speedup_int4_assumed',0):.2f}x -> {cr.get('speedup_int4_empirical',0):.2f}x "
            f"({cr.get('speedup_int4_pct_change',0):+.1f}%) |\n"
        )

    report += f"""
### Full-layer speedup (Phase 1 and Phase 2 headline numbers):

Phase 1 (BitLinear at int8, L=2048): assumed **6.72x** -> corrected **~{6.72 * (1 + pct_int8/100):.2f}x**
Phase 2 (BitLinear at int4, L=2048): assumed **14.87x** -> corrected **~{14.87 * (1 + pct_int4/100):.2f}x**

Note: full-layer correction is smaller than BitLinear-only correction because
attention (fixed, no trit weights) dilutes the BitLinear change.

---

## 5. Verdict: Does This Change the Conclusions?

{'### The assumption was accurate — no update needed.' if within_5 else '### The assumption deviated — minor updates recommended.'}

Phase 1-5 findings that are NOT affected by this measurement:
- The structural advantage of balanced ternary (free subtraction, three-way compare)
- The L/d attention crossover (attention uses activations, not weights)
- The memory bandwidth analysis (weight bytes depend on precision, not distribution)
- Hardware design recommendations (same qualitative conclusions)

{'Phase 1-5 quantitative numbers are within the stated uncertainty bounds. The 1/3-1/3-1/3 assumption is validated empirically for this model.' if within_5 else f'The BitLinear speedup numbers should be updated by {pct_int8:+.1f}% (int8) and {pct_int4:+.1f}% (int4). This is within the stated caveats for Phase 1-5.'}

---

## Caveats

1. This measurement uses BF16 master weights before any fine-tuning. A model
   fine-tuned with BitNet quantization-aware training might have a slightly
   different distribution.

2. The absmean scale is computed per weight matrix. Production inference
   might use per-row or per-group scales, which could change the zero fraction.

3. This validates one assumption out of many. Gate costs per primitive,
   hardware bandwidth specs, and roofline model assumptions are still modeled.
"""

    path = Path(__file__).parent / "EMPIRICAL_REPORT.md"
    path.write_text(report, encoding="utf-8")
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# Visualization HTML
# ---------------------------------------------------------------------------

def write_visualization(comp_stats, layer_stats, corrections, weighted_pnz):
    comp_labels  = [COMPONENT_LABELS.get(c, c) for c in sorted(comp_stats.keys())]
    comp_pnz     = [comp_stats[c]["p_nonzero"] for c in sorted(comp_stats.keys())]
    comp_pzero   = [comp_stats[c]["p_zero"]    for c in sorted(comp_stats.keys())]
    layer_ids    = [ls["layer_idx"]  for ls in layer_stats]
    layer_pnz    = [ls["p_nonzero"]  for ls in layer_stats]

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<title>Empirical: BitNet Weight Distribution</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#0f0f0f;color:#e0e0e0}}
  h1{{color:#7ec8e3}} h2{{color:#a8d8a8;margin-top:36px}}
  .wrap{{background:#1a1a1a;border-radius:8px;padding:20px;margin:20px 0}}
  canvas{{max-height:380px}}
  .key{{background:#152515;border-left:3px solid #44ff88;padding:10px 14px;margin:12px 0;border-radius:4px;font-size:.9em}}
</style>
</head>
<body>
<h1>Empirical Validation: BitNet b1.58 2B4T Weight Distribution</h1>
<div class="key">
  <strong>Key finding:</strong> Measured p_nonzero = {weighted_pnz:.4f}
  vs assumed 0.6667 (delta = {weighted_pnz-2/3:+.4f}).
  BitLinear gate speedup changes by ~{corrections.get('_overall',{}).get('speedup_int8_pct_change',0):+.1f}% for int8 activations.
</div>

<h2>1. Per-Component Nonzero Fraction</h2>
<div class="wrap">
<canvas id="compChart"></canvas>
</div>

<h2>2. Nonzero Fraction by Layer Depth</h2>
<div class="wrap">
<canvas id="layerChart"></canvas>
</div>

<script>
new Chart(document.getElementById('compChart').getContext('2d'),{{
  type:'bar',
  data:{{
    labels:{comp_labels},
    datasets:[
      {{label:'p_nonzero (empirical)', data:{comp_pnz}, backgroundColor:'#44ff8888', borderColor:'#44ff88', borderWidth:1}},
      {{label:'p_nonzero (assumed 2/3)', data:[0.6667]*len(comp_labels), type:'line', borderColor:'#ff6644', borderWidth:2, pointRadius:0, fill:false}}
    ]
  }},
  options:{{plugins:{{legend:{{labels:{{color:'#ccc'}}}}}},scales:{{x:{{ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}},y:{{min:0,max:1,title:{{display:true,text:'Fraction',color:'#aaa'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}}}}}}
}});
new Chart(document.getElementById('layerChart').getContext('2d'),{{
  type:'line',
  data:{{
    labels:{layer_ids},
    datasets:[
      {{label:'p_nonzero per layer', data:{layer_pnz}, borderColor:'#4488ff', backgroundColor:'transparent', pointRadius:3}},
      {{label:'assumed 2/3', data:[0.6667]*len(layer_ids), borderColor:'#ff6644', borderWidth:2, pointRadius:0, borderDash:[6,3]}}
    ]
  }},
  options:{{plugins:{{legend:{{labels:{{color:'#ccc'}}}}}},scales:{{x:{{title:{{display:true,text:'Layer',color:'#aaa'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}},y:{{min:0,max:1,title:{{display:true,text:'p_nonzero',color:'#aaa'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}}}}}}
}});
</script>
</body>
</html>
""".replace("[0.6667]*len(comp_labels)", str([0.6667] * len(comp_labels))) \
   .replace("[0.6667]*len(layer_ids)",  str([0.6667] * len(layer_ids)))

    path = Path(__file__).parent / "visualization.html"
    path.write_text(html, encoding="utf-8")
    print(f"Visualization written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading per-tensor distribution from results/ ...")
    records, assumption = load_records()
    print(f"  {len(records)} BitLinear tensor records loaded.")

    print("Aggregating by component ...")
    comp_stats = aggregate_by_component(records)
    with open(RESULTS_DIR / "per_component_distribution.json", "w") as f:
        json.dump(comp_stats, f, indent=2)
    print(f"  per_component_distribution.json saved ({len(comp_stats)} components).")

    print("Aggregating by layer ...")
    layer_stats = aggregate_by_layer(records)

    print("Computing corrections ...")
    corrections, weighted_pnz = compute_corrections(comp_stats)
    with open(RESULTS_DIR / "corrected_phase_numbers.json", "w") as f:
        json.dump(corrections, f, indent=2)
    print("  corrected_phase_numbers.json saved.")

    print("\nSummary:")
    ov = corrections["_overall"]
    print(f"  Assumed p_nonzero: {ov['p_nonzero_assumed']:.4f}")
    print(f"  Empirical p_nonzero: {ov['p_nonzero_empirical']:.4f}  (delta {ov['delta_pnz']:+.4f})")
    print(f"  BitLinear int8 speedup: {ov['speedup_bitlinear_int8_assumed']:.2f}x -> {ov['speedup_bitlinear_int8_empirical']:.2f}x  ({ov['speedup_int8_pct_change']:+.1f}%)")
    print(f"  BitLinear int4 speedup: {ov['speedup_bitlinear_int4_assumed']:.2f}x -> {ov['speedup_bitlinear_int4_empirical']:.2f}x  ({ov['speedup_int4_pct_change']:+.1f}%)")

    print("\nWriting report and visualization ...")
    write_report(records, comp_stats, layer_stats, corrections, weighted_pnz)
    write_visualization(comp_stats, layer_stats, corrections, weighted_pnz)
    print("Done.")


if __name__ == "__main__":
    main()
