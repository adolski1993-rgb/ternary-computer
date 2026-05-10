"""
Crossover analysis: where does each chip transition between memory-bound and
compute-bound during prefill?

Two crossovers matter:
  1. BitLinear crossover L*: sequence length at which BitLinear projections
     go from memory-bound (weight-limited) to compute-bound.
  2. Attention crossover: attention is always at a fixed intensity = HEAD_DIM/bpa,
     so it is either always memory-bound or always compute-bound.
     For H100 and TH100/TPB without FlashAttention, it's almost always memory-bound.

Summary of expected findings:
  - H100 fp16: L* ≈ 367 (compute-bound for most real prompts)
  - H100 trit: L* ≈ 31 (compute-bound for almost any prompt)
  - TH100 trit: L* ≈ 1,258 (compute-bound only for long contexts)
  - TPB trit:   L* ≈ 14 (effectively always compute-bound for weights from SRAM)

The attention score matrix (without FlashAttention) is typically memory-bound
on H100 and TH100, but becomes a significant fraction of total time at long L.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, PHASE3_DIR)

from prefill_roofline import (
    bitlinear_ridge, attention_ridge, chip_hbm_bandwidth,
    prefill_layer_timing, SEQUENCE_LENGTHS,
)
from prefill_memory import bitlinear_crossover_L, attention_intensity
from hardware_specs import H100_SXM, MI300X, TH100, TPB, ChipSpec


ANALYSIS_CONFIGS = [
    (H100_SXM, 'fp16', 'fp16', "H100 fp16/fp16"),
    (H100_SXM, 'trit', 'int4', "H100 trit/int4"),
    (MI300X,   'fp16', 'fp16', "MI300X fp16/fp16"),
    (MI300X,   'trit', 'int4', "MI300X trit/int4"),
    (TH100,    'trit', 'int4', "TH100 trit/int4"),
    (TPB,      'trit', 'int4', "TPB trit/int4"),
]


def crossover_summary() -> str:
    """Full crossover analysis table."""
    lines = [
        "Prefill BitLinear crossover (memory-bound → compute-bound), batch=1:",
        "",
        f"{'Config':<26} {'BL Ridge':>10} {'Attn Ridge':>11} {'L* (BL)':>10} {'Attn bound':>12}",
        "-" * 72,
    ]
    for chip, wp, ap, label in ANALYSIS_CONFIGS:
        bl_r   = bitlinear_ridge(chip, wp)
        at_r   = attention_ridge(chip, ap)   # use activation precision for rate
        L_star = bitlinear_crossover_L(wp, ap, batch=1, ridge_macs_per_byte=bl_r)
        at_int = attention_intensity(1, ap)   # constant

        at_label = "memory-bound" if at_int < at_r else "compute-bound"
        L_str    = f">{L_star:.0f}" if L_star is not None else "always mem"

        lines.append(
            f"{label:<26} {bl_r:>10.0f} {at_r:>11.0f} {L_str:>10} {at_label:>12}"
        )
    return "\n".join(lines)


def time_breakdown_table(batch: int = 1) -> str:
    """
    For each config and seq_len, show how compute and memory time split.
    Shows when compute-bound vs memory-bound.
    """
    lines = [
        f"Prefill layer time breakdown (ms), batch={batch}:",
        f"  {'Config':<26} {'L':>6}  {'BL-comp':>9}  {'Attn-comp':>10}  {'Score-mem':>10}  {'Bound':>7}",
        "  " + "-" * 70,
    ]
    for chip, wp, ap, label in ANALYSIS_CONFIGS[:4]:  # abbreviated
        for L in [128, 512, 2048, 4096]:
            lt = prefill_layer_timing(chip, wp, ap, L, batch)
            comp = (lt["bl_time_s"] + lt["attn_time_s"]) * 1000
            score = lt["score_time_s"] * 1000
            bl_b  = lt["bl_bottleneck"][0].upper()  # C or M
            at_b  = lt["attn_bottleneck"][0].upper()
            lines.append(
                f"  {label:<26} {L:>6}  "
                f"{lt['bl_time_s']*1000:>9.3f}  "
                f"{lt['attn_time_s']*1000:>10.3f}  "
                f"{score:>10.3f}  "
                f"BL:{bl_b} At:{at_b}"
            )
    return "\n".join(lines)


def crossover_batch_sensitivity() -> str:
    """
    How does increasing batch size change the crossover L*?
    Larger batch → weights amortized more → lower L* (earlier compute-bound).
    """
    header = "  " + f"{'Config':<22}"
    for b in [1, 4, 16, 64]:
        header += f"  {'B='+str(b):>8}"
    lines = [
        "Crossover L* by batch size (H100 configs):",
        header,
        "  " + "-" * 56,
    ]

    for chip, wp, ap, label in [(H100_SXM,'fp16','fp16','H100 fp16'),
                                  (H100_SXM,'trit','int4','H100 trit+int4')]:
        row = f"  {label:<22}"
        ridge = bitlinear_ridge(chip, wp)
        for b in [1, 4, 16, 64]:
            L = bitlinear_crossover_L(wp, ap, batch=b, ridge_macs_per_byte=ridge)
            row += f"  {'>'+str(int(L)):>8}" if L is not None else "  alw.mem"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print(crossover_summary())
    print()
    print(time_breakdown_table(batch=1))
    print()
    print(crossover_batch_sensitivity())
