"""
Effective inference throughput: tokens per second under realistic bottlenecks.

Combines compute limits (from Phase 2 gate-count analysis) and memory bandwidth
limits (from memory_model.py + kv_cache.py) into final tokens/sec numbers.

The roofline model assumption: compute and memory overlap perfectly (ideal).
Real hardware achieves 70-90% of roofline; we report the ideal ceiling and note
this in the report.

Two inference modes:
  - decode: generating one token at a time (batch=1 for interactive, batch=B for serving)
  - prefill: processing the input prompt (all tokens at once, compute-bound for large L)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from hardware_specs import (
    ChipSpec, ALL_CHIPS, H100_SXM, MI300X, GROQ_LPU, CEREBRAS_WSE3, TH100, TPB,
    PHASE2_SPEEDUP_INT4_BITLINEAR, PHASE2_SPEEDUP_INT4_ATTENTION, TERNARY_CLOCK_PENALTY,
)
from roofline import decode_roofline

SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 8, 32]


# ---------------------------------------------------------------------------
# Canonical inference configurations to evaluate
# ---------------------------------------------------------------------------

CONFIGS = [
    # (chip,          wprec,  aprec,  label)
    (H100_SXM, 'fp16', 'fp16',  "H100 fp16 baseline"),
    (H100_SXM, 'trit', 'int8',  "H100 trit-w/int8-a (hypothetical)"),
    (H100_SXM, 'trit', 'int4',  "H100 trit-w/int4-a (hypothetical)"),
    (MI300X,   'fp16', 'fp16',  "MI300X fp16 baseline"),
    (MI300X,   'trit', 'int4',  "MI300X trit-w/int4-a (hypothetical)"),
    (TH100,    'trit', 'int4',  "TH100 drop-in ternary"),
    (TPB,      'trit', 'int4',  "TPB purpose-built ternary"),
]

# For Groq and Cerebras we need special handling (SRAM-only, different capacity limits)
SRAM_ONLY_CHIPS = [GROQ_LPU, CEREBRAS_WSE3]


def throughput_matrix(batch: int = 1) -> dict:
    """
    Compute tokens/sec for every (config, sequence_length) combination.

    Returns a dict: { label → { seq_len → decode_roofline_result } }
    """
    results = {}
    for chip, wprec, aprec, label in CONFIGS:
        results[label] = {}
        for L in SEQUENCE_LENGTHS:
            results[label][L] = decode_roofline(chip, wprec, aprec, L, batch=batch)
    return results


def bottleneck_table(batch: int = 1) -> str:
    """
    Formatted table showing bottleneck type at each (config, L).
    """
    matrix = throughput_matrix(batch)
    col_w = 12
    header = f"{'Config':<38}"
    for L in SEQUENCE_LENGTHS:
        header += f" {'L='+str(L):>{col_w}}"
    lines = [
        f"Decode tokens/sec and bottleneck (batch={batch}):",
        header,
        "-" * (38 + (col_w + 1) * len(SEQUENCE_LENGTHS)),
    ]
    for chip, wprec, aprec, label in CONFIGS:
        row = f"{label:<38}"
        for L in SEQUENCE_LENGTHS:
            r = matrix[label][L]
            tps = r['tokens_per_sec']
            bn  = r['bottleneck'][0].upper()   # W, K, or C
            if tps >= 1e6:
                s = f"{tps/1e6:.1f}M({bn})"
            elif tps >= 1e3:
                s = f"{tps/1e3:.1f}K({bn})"
            else:
                s = f"{tps:.0f}({bn})"
            row += f" {s:>{col_w}}"
        lines.append(row)
    lines.append("Bottleneck key: W=weight-loading, K=KV-cache, C=compute")
    return "\n".join(lines)


def speedup_vs_baseline(batch: int = 1, ref_L: int = 2048) -> str:
    """Effective speedup of each config vs H100 fp16 at ref_L."""
    matrix = throughput_matrix(batch)
    baseline_tps = matrix["H100 fp16 baseline"][ref_L]["tokens_per_sec"]
    lines = [
        f"Effective speedup vs H100 fp16 at L={ref_L} (batch={batch}):",
        f"{'Config':<38} {'TPS':>10} {'Speedup':>10} {'Bottleneck':>12}",
        "-" * 74,
    ]
    for chip, wprec, aprec, label in CONFIGS:
        r = matrix[label][ref_L]
        tps = r['tokens_per_sec']
        sp = tps / baseline_tps
        lines.append(
            f"{label:<38} {tps:>10,.0f} {sp:>9.2f}×  {r['bottleneck']:>12}"
        )
    return "\n".join(lines)


def peak_vs_effective_comparison() -> str:
    """
    Key Phase 3 finding: compare Phase 2 PEAK COMPUTE speedup vs
    Phase 3 EFFECTIVE THROUGHPUT speedup (both at L=2048, batch=1).
    """
    matrix = throughput_matrix(batch=1)
    baseline_tps = matrix["H100 fp16 baseline"][2048]["tokens_per_sec"]

    from phase2_speedups import PHASE2_AT_L2048
    lines = [
        "Peak compute speedup (Phase 2) vs Effective throughput speedup (Phase 3)",
        f"at L=2048, batch=1:",
        f"{'Config':<38} {'Peak compute (P2)':>18} {'Effective TPS (P3)':>18} {'Ratio P3/P2':>12}",
        "-" * 88,
    ]
    comparisons = [
        ("trit-int4 on H100 (hypothetical)", "H100 trit-w/int4-a (hypothetical)", 14.87),
        ("trit-int4 TH100 drop-in",          "TH100 drop-in ternary",             14.87),
        ("trit-int4 TPB purpose-built",       "TPB purpose-built ternary",         14.87),
    ]
    for label, key, p2_speedup in comparisons:
        r = matrix[key][2048]
        eff_sp = r["tokens_per_sec"] / baseline_tps
        ratio = eff_sp / p2_speedup
        lines.append(
            f"{label:<38} {p2_speedup:>17.2f}×  {eff_sp:>17.2f}×  {ratio:>11.2f}×"
        )
    return "\n".join(lines)


def groq_cerebras_analysis() -> str:
    """
    Special analysis for SRAM-only chips.
    Groq: 230 MB SRAM, trit BitNet 2B4T = 417 MB → needs 2 chips
    Cerebras: 44 GB SRAM, far exceeds model size → weights effectively free
    """
    from memory_model import total_weight_bytes, LAYER_WEIGHT_ELEMENTS, N_LAYERS
    from kv_cache import kv_bytes_total, hbm_capacity_max_context

    model_mb_trit = total_weight_bytes('trit') / 1e6
    model_mb_fp16 = total_weight_bytes('fp16') / 1e6

    def cerebras_tps(aprec, context_len):
        """Cerebras: compute-bound for small models, SRAM BW for attention."""
        from memory_model import N_HEADS, HEAD_DIM
        from roofline import chip_macs_per_sec
        # Weights: on SRAM at 21 PB/s → negligible
        # KV: also on SRAM (44 GB >> model + KV at these sequence lengths)
        bitlinear_macs = LAYER_WEIGHT_ELEMENTS * N_LAYERS
        attn_macs = 2 * N_HEADS * HEAD_DIM * context_len * N_LAYERS
        total_macs = bitlinear_macs + attn_macs
        # Use INT8 compute rate as proxy (ternary not natively supported)
        compute_rate = chip_macs_per_sec(CEREBRAS_WSE3, 'int8')
        compute_time = total_macs / compute_rate
        # KV from SRAM (21 PB/s):
        kv_bytes = kv_bytes_total(aprec, context_len, 1)
        kv_time = kv_bytes / (CEREBRAS_WSE3.sram_bandwidth_tbs * 1e12)
        total_time = max(compute_time, kv_time)
        return 1.0 / total_time

    lines = [
        "SRAM-only chip analysis:",
        "",
        f"  BitNet 2B4T model size — trit: {model_mb_trit:.0f} MB, fp16: {model_mb_fp16:.0f} MB",
        f"  Groq LPU SRAM:     {GROQ_LPU.sram_mb:.0f} MB  → trit model ({model_mb_trit:.0f} MB) needs 2 chips",
        f"  Cerebras WSE-3:    {CEREBRAS_WSE3.sram_mb/1000:.0f} GB → trit model fits with {CEREBRAS_WSE3.sram_mb*1e6/total_weight_bytes('trit'):.0f}× headroom",
        "",
        "  Cerebras WSE-3 (all weights in SRAM, compute-bound):",
    ]
    for L in SEQUENCE_LENGTHS:
        tps = cerebras_tps('int8', L)
        if tps > 1e6:
            tstr = f"{tps/1e6:.1f}M"
        elif tps > 1e3:
            tstr = f"{tps/1e3:.1f}K"
        else:
            tstr = f"{tps:.0f}"
        lines.append(f"    L={L:5d}: {tstr} tokens/sec")

    lines += [
        "",
        "  Groq LPU (2-chip for model, 80 TB/s SRAM BW, no HBM for KV):",
        "  → KV cache grows until it overflows SRAM; not modeled for long L.",
        "  → At L=128, KV fits; at L>1000, KV approaches remaining SRAM capacity.",
        f"  → Available SRAM after weights (2 chips): {2*GROQ_LPU.sram_mb - model_mb_trit:.0f} MB for KV + activations.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inline reference to Phase 2 speedups (avoid circular import)
# ---------------------------------------------------------------------------

class phase2_speedups:
    PHASE2_AT_L2048 = {
        'fp16': 2.95,
        'int8': 6.72,
        'int4': 14.87,
        'int2': 28.98,
        'trit': 49.49,
    }


if __name__ == "__main__":
    print(bottleneck_table(batch=1))
    print()
    print(speedup_vs_baseline(batch=1, ref_L=2048))
    print()
    print(bottleneck_table(batch=32))
    print()
    print(groq_cerebras_analysis())
