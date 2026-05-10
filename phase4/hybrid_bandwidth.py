"""
Memory bandwidth analysis for hybrid activation precision configurations.

Key insight: for BitNet inference with TRIT WEIGHTS (constant),
the main bandwidth variables are:
  1. KV cache precision (determined by QKV activation precision)
  2. Total weight bytes (constant = 417 MB for trit weights)

Activation tensors are mostly transient (stay in SRAM between matmuls),
so the dominant bandwidth terms are weight loading and KV cache loading —
the same decomposition as Phase 3.

The hybrid dimension primarily affects KV cache size (via QKV precision)
and compute time (via gate counts from hybrid_gate_counts.py).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
PHASE2_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase2')
sys.path.insert(0, PHASE3_DIR)
sys.path.insert(0, PHASE2_DIR)

from memory_model import (
    total_weight_bytes, ACT_BYTES, WEIGHT_BYTES,
    N_LAYERS, N_KV_HEADS, HEAD_DIM, LAYER_WEIGHT_ELEMENTS, N_HEADS,
)
from hardware_specs import (
    ChipSpec, H100_SXM, TH100, TPB,
    PHASE2_SPEEDUP_INT4_BITLINEAR, PHASE2_SPEEDUP_INT4_ATTENTION,
    TERNARY_CLOCK_PENALTY,
)
from hybrid_gate_counts import model_gate_counts, binary_model_gates, SEQUENCE_LENGTHS


WPREC = 'trit'   # weights always trit in Phase 4


def kv_bytes_for_config(layer_cfgs: list[dict], context_len: int) -> float:
    """
    Total KV cache bytes for the full model at given context length.

    KV cache precision for each layer = QKV activation precision for that layer
    (the keys and values are computed from QKV activations and stored at that precision).

    For a hybrid config, different layers may have different KV precisions,
    giving a heterogeneous KV cache — valid hardware would handle this per layer.
    """
    total = 0.0
    for cfg in layer_cfgs:
        kv_prec = cfg['qkv_in']    # KV precision = QKV input precision
        kv_b = ACT_BYTES[kv_prec]
        total += 2 * N_KV_HEADS * context_len * HEAD_DIM * kv_b
    return total


def weight_bytes() -> float:
    """Total model weight bytes (constant across all hybrid configs)."""
    return total_weight_bytes(WPREC)


def decode_timing(
    chip: ChipSpec,
    layer_cfgs: list[dict],
    context_len: int,
    batch: int,
    seq_len: int,
    gate_counts: dict | None = None,
) -> dict:
    """
    Roofline timing for one decode step (one new token per sequence, batch=B).

    Returns timing in seconds (per batch) and derived tokens/sec.

    Compute time:
      Combines BitLinear and attention compute, using per-component speedup
      multipliers from Phase 2 (same as Phase 3's decode_roofline).

    Memory time:
      weight_time = weight_bytes / weight_bandwidth / batch
      kv_time     = kv_bytes(context_len) / hbm_bandwidth  (per sequence, not amortized)
    """
    # -- Compute time --
    # DECODE: generates exactly ONE new token regardless of context length.
    # MatMul shape is [1, K] @ [K, N] — M=1, not M=seq_len.
    # We compute with seq_len=1 so gate counts reflect a single decode step.
    # The passed gate_counts (computed at full seq_len) are used ONLY for
    # speedup comparisons in pareto_analysis, NOT for timing.
    decode_gc = model_gate_counts(1, layer_cfgs)

    h100_mac_rate = H100_SXM.tflops_fp16 * 0.5 * 1e12  # 494.75 TMAC/s

    if chip in (TH100, TPB):
        bl_rate   = h100_mac_rate * PHASE2_SPEEDUP_INT4_BITLINEAR * TERNARY_CLOCK_PENALTY
        attn_rate = h100_mac_rate * PHASE2_SPEEDUP_INT4_ATTENTION * TERNARY_CLOCK_PENALTY
    else:
        bl_rate = attn_rate = h100_mac_rate

    bitlinear_components = ['qkv_proj', 'attn_out_proj', 'ffn_gate', 'ffn_up', 'ffn_down']
    attention_components  = ['attention_qk', 'attention_av']

    # Use decode gate counts (M=1) for timing; divide by gates/MAC to get MACs
    bl_gates   = sum(decode_gc.get(c, 0) for c in bitlinear_components)
    attn_gates = sum(decode_gc.get(c, 0) for c in attention_components)

    # Note: attention Q@K^T at decode uses context_len (M=1 query attends to L keys)
    # The decode gate count with seq_len=1 correctly models: seq//2 avg attended len → 0,
    # so we add attention at context_len separately.
    from hybrid_gate_counts import act_act_gates, N_HEADS, HEAD_DIM, N_KV_HEADS
    from gate_costs import ACT_MUL, ACT_ADD
    attn_decode_gates = 0
    for cfg in layer_cfgs:
        qkv_prec = cfg['qkv_in']
        # Q@K: [N_HEADS×1, HEAD_DIM] @ [HEAD_DIM, context_len]
        # AV:  [N_HEADS×1, context_len] @ [context_len, HEAD_DIM]
        attn_decode_gates += (
            act_act_gates(N_HEADS, HEAD_DIM, context_len, qkv_prec) +
            act_act_gates(N_HEADS, context_len, HEAD_DIM, qkv_prec)
        )

    BITLINEAR_GATES_PER_MAC = 12.33
    ATTENTION_GATES_PER_MAC = 36.0

    compute_time = (bl_gates / BITLINEAR_GATES_PER_MAC / bl_rate +
                    attn_decode_gates / ATTENTION_GATES_PER_MAC / attn_rate)

    # -- Weight memory time --
    w_bytes = weight_bytes()
    if chip.sram_is_weight_store and chip.sram_bandwidth_tbs > 0:
        w_bw = chip.sram_bandwidth_tbs * 1e12
    else:
        w_bw = chip.hbm_bandwidth_tbs * 1e12
    weight_time = w_bytes / w_bw / batch

    # -- KV cache time --
    kv_bytes = kv_bytes_for_config(layer_cfgs, context_len)
    kv_bw = chip.hbm_bandwidth_tbs * 1e12 if chip.hbm_bandwidth_tbs > 0 else chip.sram_bandwidth_tbs * 1e12
    kv_time = kv_bytes / kv_bw   # per sequence, same for any batch (each seq has own KV)

    total_time = max(compute_time, weight_time, kv_time)
    bottleneck = max(
        [('compute', compute_time), ('weight', weight_time), ('kv_cache', kv_time)],
        key=lambda x: x[1]
    )[0]

    return {
        "compute_time_s": compute_time,
        "weight_time_s":  weight_time,
        "kv_time_s":      kv_time,
        "total_time_s":   total_time,
        "tokens_per_sec": batch / total_time,
        "bottleneck":     bottleneck,
        "kv_bytes":       kv_bytes,
        "weight_bytes":   w_bytes,
    }


def throughput_for_config(
    chip: ChipSpec,
    layer_cfgs: list[dict],
    batch: int = 1,
) -> dict:
    """Compute decode TPS at all 5 sequence lengths for a given config + chip."""
    results = {}
    for L in SEQUENCE_LENGTHS:
        gc = model_gate_counts(L, layer_cfgs)
        results[L] = decode_timing(chip, layer_cfgs, L, batch, L, gc)
    return results


def kv_summary(layer_cfgs: list[dict]) -> dict:
    """Per-L KV cache size and per-layer precision breakdown."""
    prec_counts = {}
    for cfg in layer_cfgs:
        p = cfg['qkv_in']
        prec_counts[p] = prec_counts.get(p, 0) + 1
    return {
        "layer_kv_precisions": prec_counts,
        "kv_bytes": {L: kv_bytes_for_config(layer_cfgs, L) for L in SEQUENCE_LENGTHS},
    }


if __name__ == "__main__":
    from hybrid_configurations import CONFIGS

    print("KV cache bytes at L=2048 (in MB):")
    for cfg in CONFIGS:
        kv_mb = kv_bytes_for_config(cfg["layer_cfgs"], 2048) / 1e6
        qdu = __import__('layer_sensitivity_model').quality_degradation_units(cfg["layer_cfgs"])
        print(f"  {cfg['label']:<38}  KV={kv_mb:.1f} MB  QDU={qdu:.1f}")

    print()
    print("TPB tokens/sec (batch=1, L=2048):")
    for cfg in CONFIGS:
        r = decode_timing(TPB, cfg["layer_cfgs"], 2048, 1, 2048)
        print(f"  {cfg['label']:<38}  {r['tokens_per_sec']:>8,.0f} tps  [{r['bottleneck']}]")
