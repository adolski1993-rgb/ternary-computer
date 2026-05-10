"""
Memory footprint model for BitNet b1.58 2B4T inference.

Models bytes-per-element for weights and activations at each precision,
then computes total memory traffic per decode token and per prefill token
for each model component.

Packing conventions
-------------------
Trit weights:   5 trits per byte  (log2(3)=1.585 bits/trit theoretical;
                                   5 trits/byte = 1.600 bits/trit = 0.95% overhead vs theoretical.
                                   Packing efficiency 243/256 = 94.9%.)
Int4 elements:  2 elements/byte   (0.5 bytes/element)
Int8 elements:  1 byte/element
FP16 elements:  2 bytes/element

Weight precision choices are independent of activation precision choices.
"""

from typing import Literal

WeightPrec = Literal['trit', 'int8', 'fp16']
ActPrec    = Literal['trit', 'int4', 'int8', 'fp16']

# ---------------------------------------------------------------------------
# Bytes per element for weights and activations
# ---------------------------------------------------------------------------

WEIGHT_BYTES: dict[str, float] = {
    'trit': 1 / 5,    # 5 trits per byte; 0.2 bytes/weight
    'int8': 1.0,
    'fp16': 2.0,
}

ACT_BYTES: dict[str, float] = {
    'trit': 1 / 5,    # same packing as trit weights
    'int2': 0.25,     # 4 elements per byte
    'int4': 0.5,
    'int8': 1.0,
    'fp16': 2.0,
}

# ---------------------------------------------------------------------------
# BitNet b1.58 2B4T architecture constants (from phase1/architecture.py)
# ---------------------------------------------------------------------------
N_LAYERS     = 30
HIDDEN       = 2560
INTERMEDIATE = 6912
N_HEADS      = 20
N_KV_HEADS   = 5
HEAD_DIM     = 128
QKV_OUT_DIM  = HIDDEN + 2 * N_KV_HEADS * HEAD_DIM   # 2560 + 640 + 640 = 3840

# Per-layer weight element counts (BitLinear weights only; norms are negligible)
LAYER_WEIGHT_ELEMENTS = (
    HIDDEN * QKV_OUT_DIM            # QKV projection
    + HIDDEN * HIDDEN               # attention output projection
    + 2 * HIDDEN * INTERMEDIATE     # FFN gate + up (two separate projections)
    + INTERMEDIATE * HIDDEN         # FFN down
)
# = 2560×3840 + 2560×2560 + 2×2560×6912 + 6912×2560 = 69,468,160

# Total model BitLinear weight elements (30 layers)
TOTAL_WEIGHT_ELEMENTS = N_LAYERS * LAYER_WEIGHT_ELEMENTS
# ≈ 2.084 billion (matches "2B" model designation)


# ---------------------------------------------------------------------------
# Weight memory footprint
# ---------------------------------------------------------------------------

def layer_weight_bytes(wprec: WeightPrec) -> float:
    """Bytes of weight memory for one decoder layer."""
    return LAYER_WEIGHT_ELEMENTS * WEIGHT_BYTES[wprec]


def total_weight_bytes(wprec: WeightPrec) -> float:
    """Total weight memory for the full model."""
    return TOTAL_WEIGHT_ELEMENTS * WEIGHT_BYTES[wprec]


# ---------------------------------------------------------------------------
# Activation memory per token per layer
# (the residual stream and key intermediate tensors that flow across layers)
# ---------------------------------------------------------------------------

def layer_act_bytes_per_token(aprec: ActPrec) -> float:
    """
    Bytes of activation memory traffic per layer per token during inference.

    Includes:
    - Reading/writing the residual stream (HIDDEN elements in + out per layer)
    - QKV output (QKV_OUT_DIM elements)
    - Attention output (HIDDEN elements)
    - FFN intermediate (2 × INTERMEDIATE elements: gate + up outputs)
    - FFN output (HIDDEN elements)
    Excludes KV cache (modeled separately in kv_cache.py).

    For batch=B, multiply by B to get total activation traffic.
    In practice, many of these tensors stay in on-chip SRAM and don't hit HBM.
    This is an upper-bound estimate (worst case: no SRAM buffering).
    """
    ab = ACT_BYTES[aprec]
    elements = (
        HIDDEN                  # residual stream input
        + QKV_OUT_DIM           # QKV output
        + HIDDEN                # attention output
        + 2 * INTERMEDIATE      # FFN gate + up outputs
        + HIDDEN                # FFN output / residual out
    )
    return elements * ab
    # ≈ (2560 + 3840 + 2560 + 13824 + 2560) × ab = 25,344 × ab per layer per token


# ---------------------------------------------------------------------------
# Bytes per token for the full forward pass (30 layers, decode step)
# ---------------------------------------------------------------------------

def decode_bytes_per_token(
    wprec: WeightPrec,
    aprec: ActPrec,
    context_len: int,
    batch: int = 1,
) -> dict[str, float]:
    """
    Total memory traffic (bytes) for generating one new token, summed over
    all 30 layers.  Returns a breakdown dict for analysis.

    For decode (generating), each call processes exactly 1 new token but
    must load all weights and the full KV cache from memory.

    Parameters
    ----------
    wprec       : weight precision
    aprec       : activation precision (also used for KV cache elements)
    context_len : number of tokens already in context (KV cache length)
    batch       : number of sequences processed simultaneously
                  - weights are amortized over the batch
                  - each sequence has its own KV cache (not amortized)
    """
    # KV cache bytes per layer per sequence (loaded every decode step)
    # K:  N_KV_HEADS × context_len × HEAD_DIM
    # V:  same
    kv_per_layer_per_seq = 2 * N_KV_HEADS * context_len * HEAD_DIM * ACT_BYTES[aprec]

    # Activation traffic per layer per token (shared across batch items)
    act_per_layer = layer_act_bytes_per_token(aprec)

    weight_bytes_total   = N_LAYERS * layer_weight_bytes(wprec)
    kv_bytes_total       = N_LAYERS * kv_per_layer_per_seq  # per sequence
    act_bytes_total      = N_LAYERS * act_per_layer         # tiny vs weights/KV

    return {
        # Per-token: weight bytes amortized over batch
        "weight_per_token":  weight_bytes_total / batch,
        # Per-token: each sequence loads its own KV (not amortized)
        "kv_per_token":      kv_bytes_total,
        # Per-token: activations (per sequence)
        "act_per_token":     act_bytes_total,
        # Total per token (dominant terms)
        "total_per_token":   weight_bytes_total / batch + kv_bytes_total + act_bytes_total,
        # Total memory traffic for the whole batch (for bandwidth calculation)
        "total_per_batch":   weight_bytes_total + batch * (kv_bytes_total + act_bytes_total),
        # Raw sizes for reporting
        "weight_total_bytes": weight_bytes_total,
        "kv_total_bytes":     kv_bytes_total,
        "act_total_bytes":    act_bytes_total,
    }


# ---------------------------------------------------------------------------
# Arithmetic intensity for each operation class (MACs / byte)
# ---------------------------------------------------------------------------

def bitlinear_intensity(wprec: WeightPrec, aprec: ActPrec, batch: int = 1) -> float:
    """
    Arithmetic intensity for BitLinear matmuls (trit weights × activations).

    MACs ≈ LAYER_WEIGHT_ELEMENTS (one MAC per weight element for decode)
    Bytes: weight_bytes/batch + small activation overhead
    Intensity = MACs / bytes  (ops/byte, where 1 MAC = 1 op for this metric)

    For large weight matrices, activation loading is negligible vs weight loading.
    """
    macs  = LAYER_WEIGHT_ELEMENTS          # per layer, per token
    wb    = layer_weight_bytes(wprec)      # dominant term
    ab_in = HIDDEN * ACT_BYTES[aprec]      # input activation
    ab_out= QKV_OUT_DIM * ACT_BYTES[aprec] # approximate output (largest proj)
    total_bytes = wb / batch + ab_in + ab_out
    return macs / total_bytes


def attention_decode_intensity(aprec: ActPrec, context_len: int) -> float:
    """
    Arithmetic intensity for attention during decode (Q attending to KV cache).

    Each decode step: new Q [N_HEADS, 1, HEAD_DIM] attends to KV [N_KV_HEADS, L, HEAD_DIM].
    MACs: 2 × N_HEADS × HEAD_DIM × context_len  (QK + AV)
    Bytes: 2 × N_KV_HEADS × context_len × HEAD_DIM × act_bytes  (load K and V)

    Note: GQA means N_HEADS Q heads share N_KV_HEADS KV heads (ratio 4:1).
    Intensity = (N_HEADS / N_KV_HEADS) / act_bytes = 4 / act_bytes
    Independent of context_len! (Both numerator and denominator scale with L.)
    """
    macs  = 2 * N_HEADS * HEAD_DIM * context_len
    bytes_kv = 2 * N_KV_HEADS * context_len * HEAD_DIM * ACT_BYTES[aprec]
    return macs / bytes_kv if bytes_kv > 0 else float('inf')


def attention_prefill_intensity(aprec: ActPrec, seq_len: int) -> float:
    """
    Arithmetic intensity for attention during prefill (all tokens at once).

    MACs: N_HEADS × seq_len² × HEAD_DIM  (Q@K and softmax@V)
    Bytes: (N_HEADS + 2×N_KV_HEADS) × seq_len × HEAD_DIM × act_bytes
    Intensity = N_HEADS × seq_len / ((N_HEADS + 2×N_KV_HEADS) × act_bytes)
              = 20 × seq_len / (30 × act_bytes)
    Grows with seq_len → prefill becomes compute-bound for large L.
    """
    macs  = N_HEADS * (seq_len ** 2) * HEAD_DIM
    bytes_ = (N_HEADS + 2 * N_KV_HEADS) * seq_len * HEAD_DIM * ACT_BYTES[aprec]
    return macs / bytes_ if bytes_ > 0 else float('inf')


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_memory_summary(wprec: WeightPrec, aprec: ActPrec):
    """Print a human-readable memory footprint summary."""
    print(f"\nMemory model: weights={wprec}, activations={aprec}")
    print(f"  Layer weight elements: {LAYER_WEIGHT_ELEMENTS:,}")
    print(f"  Layer weight bytes:    {layer_weight_bytes(wprec)/1e6:.2f} MB")
    print(f"  Total model weights:   {total_weight_bytes(wprec)/1e6:.1f} MB")
    for L in [128, 512, 1024, 2048, 4096]:
        d = decode_bytes_per_token(wprec, aprec, L, batch=1)
        print(f"  Decode at L={L:5d}:  "
              f"weight {d['weight_per_token']/1e6:.1f} MB  "
              f"KV {d['kv_per_token']/1e6:.1f} MB  "
              f"total {d['total_per_token']/1e6:.1f} MB/token  "
              f"intensity_bitlinear {bitlinear_intensity(wprec, aprec):.1f} op/B  "
              f"intensity_attn {attention_decode_intensity(aprec, L):.1f} op/B")


if __name__ == "__main__":
    print("=== Memory footprint comparison ===")
    print_memory_summary('fp16', 'fp16')
    print_memory_summary('trit', 'int8')
    print_memory_summary('trit', 'int4')
