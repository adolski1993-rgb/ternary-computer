"""
Roofline model for BitNet inference on each chip.

The roofline model characterizes each operation by its arithmetic intensity
(MACs / byte of memory traffic) and determines whether it is:
  - Memory-bound:  intensity < ridge_point → throughput = bandwidth × intensity
  - Compute-bound: intensity ≥ ridge_point → throughput = peak_compute

Ridge point = peak_compute / bandwidth  (ops/byte at the crossover)

We track three operation classes separately because they have different intensities:
  1. BitLinear matmuls (trit weights × activations)
  2. Attention (activation × activation, KV cache loading is the bandwidth limiter)
  3. Full layer blended (weighted by compute share)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from memory_model import (
    bitlinear_intensity, attention_decode_intensity, attention_prefill_intensity,
    LAYER_WEIGHT_ELEMENTS, N_HEADS, N_KV_HEADS, HEAD_DIM,
)
from hardware_specs import (
    ChipSpec, ALL_CHIPS, H100_SXM, TH100, TPB,
    PHASE2_SPEEDUP_INT4_BITLINEAR, PHASE2_SPEEDUP_INT4_ATTENTION,
    TERNARY_CLOCK_PENALTY,
)

SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 8, 32, 128]


# ---------------------------------------------------------------------------
# Effective compute rates
#
# For real chips (H100, MI300X, Groq, Cerebras), compute rates are published.
# For hypothetical ternary chips, we scale from H100 using Phase 2 speedups.
#
# "MACs/sec" here means logical MACs (one multiply-accumulate per weight element).
# The ternary chip executes these with fewer hardware gates (the Phase 2 finding),
# so it can sustain more MACs/sec for the same silicon area.
# ---------------------------------------------------------------------------

def chip_macs_per_sec(chip: ChipSpec, op_type: str = 'fp16') -> float:
    """
    Peak logical MACs per second for a given operation type.

    op_type options:
      'fp16'       — standard fp16 matmul
      'int8'       — int8 matmul
      'bitlinear'  — trit-weight × int4-activation (ternary chips only)
      'attention'  — int4 × int4 attention (ternary chips only)
    """
    # 1 TFLOP = 0.5 TMAC (1 MAC = 2 FLOP: multiply + add)
    if op_type == 'fp16':
        return chip.tflops_fp16 * 0.5 * 1e12
    if op_type == 'int8':
        return chip.tops_int8 * 0.5 * 1e12

    # Ternary-specific: scale H100 fp16 compute by Phase 2 speedup × clock penalty
    base_mac_rate = H100_SXM.tflops_fp16 * 0.5 * 1e12
    if op_type == 'bitlinear':
        return base_mac_rate * PHASE2_SPEEDUP_INT4_BITLINEAR * TERNARY_CLOCK_PENALTY
    if op_type == 'attention':
        return base_mac_rate * PHASE2_SPEEDUP_INT4_ATTENTION * TERNARY_CLOCK_PENALTY

    raise ValueError(f"Unknown op_type: {op_type}")


def effective_bandwidth_for_weights(chip: ChipSpec) -> float:
    """
    Bandwidth (bytes/sec) available for loading model weights.

    For chips where model weights fit in on-chip SRAM (sram_is_weight_store=True),
    return SRAM bandwidth.  Otherwise return HBM bandwidth.
    """
    if chip.sram_is_weight_store and chip.sram_bandwidth_tbs > 0:
        return chip.sram_bandwidth_tbs * 1e12
    return chip.hbm_bandwidth_tbs * 1e12


def effective_bandwidth_for_kv(chip: ChipSpec) -> float:
    """
    Bandwidth (bytes/sec) for loading KV cache.
    KV cache always comes from HBM (too large for on-chip SRAM at long L).
    For SRAM-only chips (Groq, Cerebras), KV is managed differently —
    we use SRAM bandwidth as an upper bound but note capacity constraints.
    """
    if chip.hbm_bandwidth_tbs > 0:
        return chip.hbm_bandwidth_tbs * 1e12
    # SRAM-only chip: use SRAM bandwidth but KV may not fit
    return chip.sram_bandwidth_tbs * 1e12


# ---------------------------------------------------------------------------
# Ridge points
# ---------------------------------------------------------------------------

def ridge_point(chip: ChipSpec, op_type: str = 'fp16') -> float:
    """
    Ridge point in MACs/byte (or ops/byte).

    Below ridge: memory-bound.  Above ridge: compute-bound.
    """
    macs_per_sec = chip_macs_per_sec(chip, op_type)
    bw = chip.hbm_bandwidth_tbs * 1e12 if chip.hbm_bandwidth_tbs > 0 else chip.sram_bandwidth_tbs * 1e12
    return macs_per_sec / bw if bw > 0 else float('inf')


# ---------------------------------------------------------------------------
# Per-operation roofline analysis
# ---------------------------------------------------------------------------

def roofline_point(
    intensity: float,
    chip: ChipSpec,
    op_type: str,
    weight_bw: float | None = None,
) -> dict:
    """
    Given arithmetic intensity and chip, return attainable performance and bottleneck.

    Parameters
    ----------
    intensity   : MACs per byte of memory traffic
    chip        : chip spec
    op_type     : compute type ('fp16', 'int8', 'bitlinear', 'attention')
    weight_bw   : override bandwidth for weight loading (SRAM vs HBM)
    """
    macs_per_sec = chip_macs_per_sec(chip, op_type)
    bw = weight_bw if weight_bw is not None else chip.hbm_bandwidth_tbs * 1e12
    if bw <= 0:
        bw = chip.sram_bandwidth_tbs * 1e12

    compute_bound_perf = macs_per_sec
    memory_bound_perf  = intensity * bw
    attainable         = min(compute_bound_perf, memory_bound_perf)
    ridge              = compute_bound_perf / bw if bw > 0 else float('inf')
    bottleneck         = "compute" if intensity >= ridge else "memory"

    return {
        "intensity":          intensity,
        "attainable_macs_ps": attainable,
        "peak_compute_macs_ps": compute_bound_perf,
        "ridge_ops_per_byte": ridge,
        "bottleneck":         bottleneck,
        "memory_util":        min(1.0, intensity / ridge) if ridge > 0 else 0.0,
        "compute_util":       min(1.0, ridge / intensity) if intensity > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Full decode-step roofline (blended across BitLinear + attention)
# ---------------------------------------------------------------------------

def decode_roofline(
    chip: ChipSpec,
    wprec: str,
    aprec: str,
    context_len: int,
    batch: int = 1,
) -> dict:
    """
    Compute and memory time for one full decode step (one new token, all 30 layers).

    Returns timing components and the effective bottleneck for each.
    All times are in seconds per batch.  Divide by batch for per-token times.
    """
    from memory_model import (
        layer_weight_bytes, LAYER_WEIGHT_ELEMENTS, N_LAYERS,
    )
    from kv_cache import kv_bytes_per_layer

    # --- Compute time ---
    # BitLinear MACs per layer per decode token (all tokens compressed to 1 new)
    bitlinear_macs = LAYER_WEIGHT_ELEMENTS   # one MAC per weight element
    # Attention MACs per layer: Q@K + softmax@V
    attn_macs = 2 * N_HEADS * HEAD_DIM * context_len

    # For hypothetical ternary chips, BitLinear and attention have different speedups
    if chip in (TH100, TPB):
        bl_rate   = chip_macs_per_sec(chip, 'bitlinear')
        attn_rate = chip_macs_per_sec(chip, 'attention')
    else:
        # Real chips: use fp16 for fp16 model, int8 for int8 model as proxy
        rate = chip_macs_per_sec(chip, 'fp16' if wprec == 'fp16' else 'int8')
        bl_rate = attn_rate = rate

    compute_per_layer = (bitlinear_macs / bl_rate + attn_macs / attn_rate)
    compute_time_batch = N_LAYERS * compute_per_layer  # total, for one token

    # --- Memory time: weights ---
    w_bw = effective_bandwidth_for_weights(chip)
    w_bytes_per_layer = layer_weight_bytes(wprec)
    weight_time_batch = (N_LAYERS * w_bytes_per_layer) / w_bw / batch

    # --- Memory time: KV cache ---
    kv_bw = effective_bandwidth_for_kv(chip)
    kv_bytes_layer = kv_bytes_per_layer(aprec, context_len)
    kv_time_batch = (N_LAYERS * kv_bytes_layer * batch) / kv_bw / batch
    # Note: KV scales with batch (each seq has own KV) but we're computing per-token time
    # total KV traffic = batch × kv_bytes; total time = batch × kv_bytes / bw
    # per-token time = kv_bytes / bw  (independent of batch — same as batch=1)

    # --- Bottleneck ---
    dominant_memory_time = max(weight_time_batch, kv_time_batch)
    total_time = max(compute_time_batch, dominant_memory_time)

    # Tokens per second
    tps = batch / total_time

    bottleneck = "compute"
    if dominant_memory_time > compute_time_batch:
        bottleneck = "weight" if weight_time_batch >= kv_time_batch else "kv_cache"

    return {
        "chip":               chip.short,
        "wprec":              wprec,
        "aprec":              aprec,
        "context_len":        context_len,
        "batch":              batch,
        "compute_time_s":     compute_time_batch,
        "weight_time_s":      weight_time_batch,
        "kv_time_s":          kv_time_batch,
        "total_time_s":       total_time,
        "tokens_per_sec":     tps,
        "bottleneck":         bottleneck,
        "weight_util":        min(1.0, weight_time_batch / total_time),
        "kv_util":            min(1.0, kv_time_batch / total_time),
        "compute_util":       min(1.0, compute_time_batch / total_time),
    }


# ---------------------------------------------------------------------------
# Roofline data for visualization (intensity vs attainable performance)
# ---------------------------------------------------------------------------

def build_roofline_data() -> dict:
    """
    Build roofline data points for visualization:
    - Chip bandwidth ceiling curves (for sweeping intensity 0.1 → 10000 ops/byte)
    - Operation points for decode at each (wprec, aprec, L, batch)
    """
    import math

    intensities = [10 ** (x / 4) for x in range(-4, 32)]   # 0.1 to ~16,000

    chips_to_plot = [H100_SXM, TH100, TPB]

    ceiling_curves = {}
    for chip in chips_to_plot:
        bw = chip.hbm_bandwidth_tbs * 1e12 if chip.hbm_bandwidth_tbs > 0 else chip.sram_bandwidth_tbs * 1e12
        peak = chip.tflops_fp16 * 0.5 * 1e12
        ceiling_curves[chip.short] = {
            "intensities": intensities,
            "attainable":  [min(peak, i * bw) / 1e12 for i in intensities],  # in TMACS/s
            "ridge_point": peak / bw if bw > 0 else float('inf'),
            "peak_tmacs":  peak / 1e12,
        }

    # Operation points
    op_points = {}
    for wprec, aprec in [('fp16','fp16'), ('trit','int8'), ('trit','int4')]:
        label = f"{wprec}w/{aprec}a"
        op_points[label] = {}
        for L in [128, 2048, 4096]:
            op_points[label][f"L{L}_bl"]   = bitlinear_intensity(wprec, aprec, batch=1)
            op_points[label][f"L{L}_attn"] = attention_decode_intensity(aprec, L)

    return {"ceiling_curves": ceiling_curves, "op_points": op_points}


if __name__ == "__main__":
    print(f"{'Chip':<22} {'Op type':<12} {'Ridge (ops/B)':>14}")
    print("-" * 50)
    for chip in [H100_SXM, TH100, TPB]:
        for op in ['fp16', 'bitlinear', 'attention']:
            r = ridge_point(chip, op)
            print(f"{chip.short:<22} {op:<12} {r:>14.1f}")
    print()
    print("Decode bottleneck analysis (batch=1, L=2048):")
    configs = [
        (H100_SXM, 'fp16', 'fp16'),
        (TH100,    'trit', 'int4'),
        (TPB,      'trit', 'int4'),
    ]
    for chip, wp, ap in configs:
        r = decode_roofline(chip, wp, ap, 2048, batch=1)
        print(f"  {chip.short:<22} {wp}w/{ap}a  "
              f"TPS={r['tokens_per_sec']:,.0f}  bottleneck={r['bottleneck']}")
