"""
Roofline analysis for the prefill phase.

For each (chip, wprec, aprec, seq_len, batch):
  - Compute time = max(BitLinear_time, Attention_time)
  - Memory time  = weight_time + activation_time + score_matrix_time + kv_write_time
  - Total time   = max(compute_time, memory_time)  ← roofline assumption
  - TPS          = batch × seq_len / total_time

We separately track BitLinear and attention because they have different
arithmetic intensities and may be on different sides of the roofline.

Chip compute rates (same conventions as Phase 3/4):
  H100/MI300X fp16 path: use fp16 MAC rate
  H100/MI300X trit path: use int8 MAC rate (closest available HW support)
  TH100/TPB: scale H100 by Phase 2 speedup × clock penalty (ternary-specific)

Weight bandwidth for TH100 (HBM3) and TPB (SRAM) differs — same as Phase 3.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
PHASE2_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase2')
sys.path.insert(0, PHASE3_DIR)
sys.path.insert(0, PHASE2_DIR)

from hardware_specs import (
    ChipSpec, H100_SXM, MI300X, TH100, TPB,
    PHASE2_SPEEDUP_INT4_BITLINEAR, PHASE2_SPEEDUP_INT4_ATTENTION,
    TERNARY_CLOCK_PENALTY,
)
from prefill_compute import (
    bitlinear_macs_per_layer, attention_macs_per_layer,
    N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, QKV_OUT_DIM,
    LAYER_WEIGHT_ELEMENTS, SEQUENCE_LENGTHS, BATCH_SIZES,
)
from prefill_memory import (
    weight_bytes_per_layer, activation_bytes_per_layer,
    attention_score_bytes_per_layer, kv_write_bytes_per_layer,
    bitlinear_intensity as bl_intensity, attention_intensity as attn_intensity,
    bitlinear_crossover_L,
)
from memory_model import WEIGHT_BYTES


# ---------------------------------------------------------------------------
# Chip compute rates for prefill
# ---------------------------------------------------------------------------

def chip_bl_mac_rate(chip: ChipSpec, wprec: str) -> float:
    """BitLinear MAC rate for a given chip and weight precision."""
    h100_base = H100_SXM.tflops_fp16 * 0.5 * 1e12  # 494.75 TMAC/s

    if chip in (TH100, TPB):
        return h100_base * PHASE2_SPEEDUP_INT4_BITLINEAR * TERNARY_CLOCK_PENALTY
    if wprec in ('fp16',):
        return chip.tflops_fp16 * 0.5 * 1e12
    return chip.tops_int8 * 0.5 * 1e12  # use int8 rate for quantized weights


def chip_attn_mac_rate(chip: ChipSpec, aprec: str) -> float:
    """Attention (act×act) MAC rate."""
    h100_base = H100_SXM.tflops_fp16 * 0.5 * 1e12

    if chip in (TH100, TPB):
        return h100_base * PHASE2_SPEEDUP_INT4_ATTENTION * TERNARY_CLOCK_PENALTY
    if aprec in ('fp16',):
        return chip.tflops_fp16 * 0.5 * 1e12
    return chip.tops_int8 * 0.5 * 1e12


def chip_weight_bandwidth(chip: ChipSpec) -> float:
    """Bandwidth for loading weights (SRAM for TPB, HBM for others)."""
    if chip.sram_is_weight_store and chip.sram_bandwidth_tbs > 0:
        return chip.sram_bandwidth_tbs * 1e12
    return chip.hbm_bandwidth_tbs * 1e12


def chip_hbm_bandwidth(chip: ChipSpec) -> float:
    """HBM bandwidth for activations, scores, KV."""
    bw = chip.hbm_bandwidth_tbs * 1e12
    if bw > 0:
        return bw
    return chip.sram_bandwidth_tbs * 1e12   # SRAM-only fallback


# ---------------------------------------------------------------------------
# Ridge points
# ---------------------------------------------------------------------------

def bitlinear_ridge(chip: ChipSpec, wprec: str) -> float:
    """
    Ridge point for BitLinear during prefill (MACs/byte).
    Uses WEIGHT bandwidth as the relevant bandwidth (weights dominate small L).
    """
    mac_rate = chip_bl_mac_rate(chip, wprec)
    bw = chip_weight_bandwidth(chip)
    return mac_rate / bw if bw > 0 else float('inf')


def attention_ridge(chip: ChipSpec, aprec: str = 'fp16') -> float:
    """
    Ridge point for attention (MACs/byte) using HBM bandwidth.
    aprec determines the compute rate (int8 proxy for int4/trit on real chips).
    """
    mac_rate = chip_attn_mac_rate(chip, aprec)
    bw = chip_hbm_bandwidth(chip)
    return mac_rate / bw if bw > 0 else float('inf')


# ---------------------------------------------------------------------------
# Per-layer roofline timing
# ---------------------------------------------------------------------------

def prefill_layer_timing(
    chip: ChipSpec,
    wprec: str,
    aprec: str,
    seq_len: int,
    batch: int = 1,
) -> dict:
    """
    Roofline time estimate for one prefill layer.

    Returns times in seconds, plus bottleneck labels.
    """
    # --- Compute rates ---
    bl_rate   = chip_bl_mac_rate(chip, wprec)
    attn_rate = chip_attn_mac_rate(chip, aprec)
    w_bw      = chip_weight_bandwidth(chip)
    hbm_bw    = chip_hbm_bandwidth(chip)

    # --- MACs ---
    bl_macs   = bitlinear_macs_per_layer(seq_len, batch)
    attn_macs = attention_macs_per_layer(seq_len, batch)

    # --- Memory bytes ---
    w_bytes    = weight_bytes_per_layer(wprec)
    act_bytes  = activation_bytes_per_layer(seq_len, aprec, batch)
    score_bytes= attention_score_bytes_per_layer(seq_len, aprec, batch)
    kv_bytes   = kv_write_bytes_per_layer(seq_len, aprec, batch)

    # --- Arithmetic intensity ---
    bl_intensity  = bl_macs / (w_bytes + act_bytes)  if (w_bytes + act_bytes) > 0 else 0
    at_intensity  = attn_intensity(seq_len, aprec)   # HEAD_DIM / bpa (constant)

    # --- Roofline: attainable performance ---
    bl_attainable   = min(bl_rate,   bl_intensity * w_bw)
    attn_attainable = min(attn_rate, at_intensity * hbm_bw)

    # --- Time ---
    bl_time    = bl_macs   / bl_attainable   if bl_attainable   > 0 else 1e9
    attn_time  = attn_macs / attn_attainable if attn_attainable > 0 else 1e9
    # Score matrix + KV write: bound by HBM
    score_time = score_bytes / hbm_bw if hbm_bw > 0 else 1e9
    kv_time    = kv_bytes   / hbm_bw if hbm_bw > 0 else 1e9

    # Total layer time is max of compute and dominant memory terms (overlap assumed)
    mem_time   = max(w_bytes / w_bw + act_bytes / hbm_bw, score_time, kv_time)
    total_time = max(bl_time + attn_time, mem_time)

    # Bottleneck
    if bl_intensity < bitlinear_ridge(chip, wprec):
        bl_bound = "memory"
    else:
        bl_bound = "compute"

    if at_intensity < attention_ridge(chip):
        at_bound = "memory"
    else:
        at_bound = "compute"

    return {
        "bl_time_s":      bl_time,
        "attn_time_s":    attn_time,
        "score_time_s":   score_time,
        "kv_time_s":      kv_time,
        "mem_time_s":     mem_time,
        "total_time_s":   total_time,
        "bl_intensity":   bl_intensity,
        "attn_intensity": at_intensity,
        "bl_bottleneck":  bl_bound,
        "attn_bottleneck":at_bound,
    }


# ---------------------------------------------------------------------------
# Full model prefill TPS
# ---------------------------------------------------------------------------

def prefill_tps(
    chip: ChipSpec,
    wprec: str,
    aprec: str,
    seq_len: int,
    batch: int = 1,
) -> dict:
    """
    Tokens per second for prefill: (batch × seq_len) / total_model_time.

    "Tokens" here = INPUT tokens processed per second.
    """
    lt = prefill_layer_timing(chip, wprec, aprec, seq_len, batch)
    total_model_time = N_LAYERS * lt["total_time_s"]
    tps = batch * seq_len / total_model_time if total_model_time > 0 else 0

    return {
        "tokens_per_sec":  tps,
        "total_time_s":    total_model_time,
        "per_layer":       lt,
        "dominant_bound":  "compute" if lt["bl_bottleneck"] == "compute" else "memory",
    }


# ---------------------------------------------------------------------------
# Crossover tables
# ---------------------------------------------------------------------------

def crossover_table() -> str:
    """L at which prefill BitLinear goes compute-bound on each chip+precision."""
    configs = [
        (H100_SXM, 'fp16', 'fp16'),
        (H100_SXM, 'trit', 'int4'),
        (MI300X,   'fp16', 'fp16'),
        (MI300X,   'trit', 'int4'),
        (TH100,    'trit', 'int4'),
        (TPB,      'trit', 'int4'),
    ]
    lines = [
        "BitLinear prefill compute-bound crossover (batch=1):",
        f"  {'Config':<38} {'Ridge':>8}  {'L_crossover':>12}",
        "  " + "-" * 62,
    ]
    for chip, wp, ap in configs:
        ridge = bitlinear_ridge(chip, wp)
        L_cross = bitlinear_crossover_L(wp, ap, batch=1, ridge_macs_per_byte=ridge)
        L_str = f"L > {L_cross:.0f}" if L_cross is not None else "always memory-bound"
        lines.append(f"  {chip.short} {wp}w/{ap}a        {ridge:>8.0f}  {L_str:>12}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(crossover_table())
    print()
    print("Prefill TPS (batch=1):")
    print(f"  {'Config':<28}", end="")
    for L in SEQUENCE_LENGTHS:
        print(f"  {'L='+str(L):>10}", end="")
    print()

    configs = [
        (H100_SXM, 'fp16', 'fp16', "H100 fp16"),
        (H100_SXM, 'trit', 'int4', "H100 trit+int4"),
        (TH100,    'trit', 'int4', "TH100 drop-in"),
        (TPB,      'trit', 'int4', "TPB purpose-built"),
    ]
    for chip, wp, ap, label in configs:
        print(f"  {label:<28}", end="")
        for L in SEQUENCE_LENGTHS:
            r = prefill_tps(chip, wp, ap, L, 1)
            tps = r["tokens_per_sec"]
            s = f"{tps/1e3:.0f}K" if tps >= 1e3 else f"{tps:.0f}"
            print(f"  {s:>10}", end="")
        print()
