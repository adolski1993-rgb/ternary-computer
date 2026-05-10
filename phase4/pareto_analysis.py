"""
Pareto analysis: quality-degradation vs compute/bandwidth cost for hybrid configs.

Plots the tradeoff space and identifies Pareto-optimal configurations.

Axes:
  X: Quality Degradation Units (QDU) — lower is better; 0=int8 baseline, 100=uniform int4
  Y: Gate-count speedup vs fp16 binary — higher is better
  Y': Effective TPS on TPB chip (batch=1, L=2048) — higher is better

A configuration Pareto-dominates another if it is BOTH:
  - Lower QDU (better quality)
  - Higher speedup / TPS (better performance)

If neither dominates the other, both are Pareto-optimal (on the frontier).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from layer_sensitivity_model import quality_degradation_units
from hybrid_gate_counts import model_gate_counts, binary_model_gates, SEQUENCE_LENGTHS
from hybrid_bandwidth import decode_timing, kv_bytes_for_config
from hybrid_configurations import CONFIGS

PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, PHASE3_DIR)
from hardware_specs import H100_SXM, TH100, TPB


def compute_all_metrics(batch: int = 1) -> list[dict]:
    """
    For every configuration, compute all metrics at L=2048.
    Returns a list of dicts with all Pareto-relevant fields.
    """
    results = []
    L = 2048
    binary_total = binary_model_gates(L)["TOTAL"]

    for cfg in CONFIGS:
        name       = cfg["name"]
        label      = cfg["label"]
        layer_cfgs = cfg["layer_cfgs"]

        qdu = quality_degradation_units(layer_cfgs)

        # Gate counts at L=2048
        gc = model_gate_counts(L, layer_cfgs)
        speedup = binary_total / gc["TOTAL"]

        # Effective TPS on TH100 and TPB (batch=1)
        r_th  = decode_timing(TH100, layer_cfgs, L, batch, L, gc)
        r_tpb = decode_timing(TPB,   layer_cfgs, L, batch, L, gc)

        # KV cache size at L=2048
        kv_mb = kv_bytes_for_config(layer_cfgs, L) / 1e6

        results.append({
            "name":             name,
            "label":            label,
            "qdu":              qdu,
            "speedup_vs_fp16":  speedup,
            "tps_th100":        r_th["tokens_per_sec"],
            "tps_tpb":          r_tpb["tokens_per_sec"],
            "bottleneck_th100": r_th["bottleneck"],
            "bottleneck_tpb":   r_tpb["bottleneck"],
            "gate_total_T":     gc["TOTAL"] / 1e12,
            "kv_mb_L2048":      kv_mb,
            "rationale":        cfg["rationale"],
        })

    return results


def pareto_frontier(results: list[dict], x_key: str, y_key: str,
                    x_lower_better: bool = True, y_lower_better: bool = False) -> list[str]:
    """
    Identify Pareto-optimal configurations.

    A config A dominates config B if:
      A is at least as good as B on both dimensions AND strictly better on at least one.
    """
    dominated = set()
    names = [r["name"] for r in results]

    for i, ri in enumerate(results):
        for j, rj in enumerate(results):
            if i == j:
                continue
            xi, yi = ri[x_key], ri[y_key]
            xj, yj = rj[x_key], rj[y_key]

            if x_lower_better:
                x_better_or_equal = xi <= xj
                x_strictly_better = xi < xj
            else:
                x_better_or_equal = xi >= xj
                x_strictly_better = xi > xj

            if y_lower_better:
                y_better_or_equal = yi <= yj
                y_strictly_better = yi < yj
            else:
                y_better_or_equal = yi >= yj
                y_strictly_better = yi > yj

            if x_better_or_equal and y_better_or_equal and (x_strictly_better or y_strictly_better):
                dominated.add(names[j])

    return [n for n in names if n not in dominated]


def pareto_table(results: list[dict]) -> str:
    """Formatted table with Pareto status marked."""
    frontier_names = pareto_frontier(results, "qdu", "tps_tpb",
                                     x_lower_better=True, y_lower_better=False)
    lines = [
        f"{'Config':<34} {'QDU':>7} {'Speedup':>9} {'TPB TPS':>10} {'KV(MB)':>8} {'Pareto':>7}",
        "-" * 82,
    ]
    for r in sorted(results, key=lambda x: x["qdu"]):
        star = " ★" if r["name"] in frontier_names else ""
        lines.append(
            f"{r['label']:<34} {r['qdu']:>7.1f} {r['speedup_vs_fp16']:>8.2f}×"
            f" {r['tps_tpb']:>9,.0f} {r['kv_mb_L2048']:>8.1f}{star}"
        )
    lines.append("★ = Pareto-optimal (better quality AND better TPS than at least one non-★)")
    return "\n".join(lines)


def speedup_table_across_seq(batch: int = 1) -> str:
    """Gate-count speedup vs fp16 binary at each sequence length for each config."""
    col_w = 10
    header = f"{'Config':<34}" + "".join(f"  {'L='+str(L):>{col_w}}" for L in SEQUENCE_LENGTHS)
    lines = [f"Gate-count speedup vs fp16 (batch={batch}):", header, "-" * (34 + (col_w + 2) * 5)]
    binary_totals = {L: binary_model_gates(L)["TOTAL"] for L in SEQUENCE_LENGTHS}
    for cfg in CONFIGS:
        row = f"{cfg['label']:<34}"
        for L in SEQUENCE_LENGTHS:
            gc = model_gate_counts(L, cfg["layer_cfgs"])
            sp = binary_totals[L] / gc["TOTAL"]
            row += f"  {sp:>{col_w}.2f}×"
        lines.append(row)
    return "\n".join(lines)


def tps_table(chip, chip_name: str, batch: int = 1) -> str:
    """TPS at each sequence length for each config on a given chip."""
    col_w = 10
    header = f"{'Config':<34}" + "".join(f"  {'L='+str(L):>{col_w}}" for L in SEQUENCE_LENGTHS)
    lines = [
        f"Tokens/sec on {chip_name} (batch={batch}):",
        header,
        "-" * (34 + (col_w + 2) * 5),
    ]
    for cfg in CONFIGS:
        row = f"{cfg['label']:<34}"
        for L in SEQUENCE_LENGTHS:
            gc = model_gate_counts(L, cfg["layer_cfgs"])
            r = decode_timing(chip, cfg["layer_cfgs"], L, batch, L, gc)
            tps = r["tokens_per_sec"]
            tstr = f"{tps/1e3:.1f}K" if tps < 1e6 else f"{tps/1e6:.1f}M"
            row += f"  {tstr:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def microsoft_comparison() -> str:
    """
    Address the task's reference: "Microsoft tested hybrid configs during training
    and reported uniform was best."

    This refers to WEIGHT quantization uniformity (the BitNet b1.58 paper).
    Our analysis is about ACTIVATION quantization — a different dimension.
    BitNet a4.8 shows activation hybridity IS beneficial.
    """
    return """
Why this doesn't contradict Microsoft's finding:
  Microsoft's "uniform is best" finding (BitNet b1.58 paper) refers to
  WEIGHT precision: they found uniform ternary weights throughout the model
  outperform mixed-precision weight schemes.  This Phase 4 accepts that
  result and keeps all weights at trit.

  Phase 4's question is orthogonal: given trit weights (fixed), which
  ACTIVATION tensors should stay at int8 vs int4?  BitNet a4.8 (also from
  Microsoft) explicitly recommends non-uniform activation precision —
  FFN down inputs at int8, everything else at int4.

  These are consistent: the optimal architecture is:
    Weights:     UNIFORM trit everywhere (Microsoft finding from b1.58)
    Activations: HYBRID int8 for ffn_down_in, int4 elsewhere (a4.8 finding)
"""


if __name__ == "__main__":
    results = compute_all_metrics(batch=1)
    print(pareto_table(results))
    print()
    print(speedup_table_across_seq(batch=1))
    print()
    print(tps_table(TPB, "TPB", batch=1))
