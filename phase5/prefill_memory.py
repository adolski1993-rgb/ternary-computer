"""
Memory bandwidth requirements during prefill.

Prefill memory differs from decode in three key ways:

1. Weights amortized over L tokens
   Decode: each token loads all weights alone → intensity ≈ 2 MACs/byte (weight-limited)
   Prefill: L tokens SHARE the weight load → intensity ≈ 2L MACs/byte (grows with L)
   Crossover to compute-bound happens when L × decode_intensity > ridge_point.

2. Activation tensors are O(L) not O(1)
   Decode: one token's activations are negligible vs weights
   Prefill: L tokens generate L× more activation data; at large L, acts > weights

3. Attention score matrix is O(L²)
   This term dominates at long L:
     N_HEADS × L² × bpa bytes per layer (read+write cycle for causal attention)
   Without FlashAttention, this is written to memory and re-read for softmax.
   FlashAttention avoids materializing this matrix (flagged as a caveat).

4. KV cache WRITE (not just read)
   Prefill writes the initial KV cache: N_KV_HEADS × L × HEAD_DIM × 2 × bpa per layer.
   Decode only reads the KV cache. Prefill both writes and (optionally) reads.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, PHASE3_DIR)

from memory_model import (
    WEIGHT_BYTES, ACT_BYTES, LAYER_WEIGHT_ELEMENTS,
    N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, QKV_OUT_DIM,
)

SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 8, 32]

# Activation dimensions per layer per token (input+output of each matmul)
# Approximation: sum of all matmul input+output dimensions
ACT_DIMS_PER_TOKEN = (
    HIDDEN + QKV_OUT_DIM                 # QKV in + out
    + HIDDEN + HIDDEN                    # attn_out in + out
    + HIDDEN + INTERMEDIATE              # FFN up in + out (one direction)
    + HIDDEN + INTERMEDIATE              # FFN gate in + out
    + INTERMEDIATE + HIDDEN             # FFN down in + out
)  # ≈ 25,344 elements per token per layer (matches Phase 3)


# ---------------------------------------------------------------------------
# Individual memory components
# ---------------------------------------------------------------------------

def weight_bytes_per_layer(wprec: str) -> float:
    """Same as decode: weights are the same matrix, loaded once per layer per prompt."""
    return LAYER_WEIGHT_ELEMENTS * WEIGHT_BYTES[wprec]


def activation_bytes_per_layer(seq_len: int, aprec: str, batch: int = 1) -> float:
    """
    Activation tensor traffic per layer.
    Scales as batch × seq_len × ACT_DIMS_PER_TOKEN × act_bytes.
    Much larger than decode (where seq_len=1).
    """
    return batch * seq_len * ACT_DIMS_PER_TOKEN * ACT_BYTES[aprec]


def attention_score_bytes_per_layer(seq_len: int, aprec: str, batch: int = 1) -> float:
    """
    Memory for the L×L attention score matrix.

    Standard (non-FlashAttention) attention materializes N_HEADS × L × L values
    per layer.  With causal masking, the lower triangle only matters, but hardware
    typically allocates the full matrix.  We model full materialization.

    NOTE: FlashAttention fuses the softmax and avoids writing the full score
    matrix, reducing this to O(L) (dominated by Q/K/V).  We do NOT model
    FlashAttention — this term is the primary caveat for long-L prefill.

    Factor of 2: written once (for softmax input) and read back once.
    """
    return 2 * batch * N_HEADS * seq_len * seq_len * ACT_BYTES[aprec]


def kv_write_bytes_per_layer(seq_len: int, aprec: str, batch: int = 1) -> float:
    """
    Bytes written to the KV cache during prefill.
    Decode only reads KV; prefill generates and writes it.
    """
    return batch * 2 * N_KV_HEADS * seq_len * HEAD_DIM * ACT_BYTES[aprec]


def prefill_total_bytes_per_layer(
    seq_len: int, wprec: str, aprec: str, batch: int = 1
) -> dict:
    """
    Full memory traffic breakdown for one prefill layer.

    Returns a dict with each component and the total.
    Note: weights are shared by all batch×seq_len tokens (not multiplied by batch).
    """
    return {
        "weights":      weight_bytes_per_layer(wprec),
        "activations":  activation_bytes_per_layer(seq_len, aprec, batch),
        "attn_scores":  attention_score_bytes_per_layer(seq_len, aprec, batch),
        "kv_write":     kv_write_bytes_per_layer(seq_len, aprec, batch),
    }


def prefill_total_bytes_model(
    seq_len: int, wprec: str, aprec: str, batch: int = 1
) -> dict:
    """Full-model (30-layer) prefill memory traffic breakdown."""
    per_layer = prefill_total_bytes_per_layer(seq_len, wprec, aprec, batch)
    return {k: v * N_LAYERS for k, v in per_layer.items()}


# ---------------------------------------------------------------------------
# Arithmetic intensity for each memory component
# ---------------------------------------------------------------------------

def bitlinear_intensity(seq_len: int, wprec: str, aprec: str, batch: int = 1) -> float:
    """
    Arithmetic intensity for BitLinear matmuls during prefill.

    MACs = batch × seq_len × LAYER_WEIGHT_ELEMENTS  (per layer)
    Bytes = weights + activation input/output  (per layer)

    At small L: weight-dominated → intensity ≈ batch×L / bpw_weight
    At large L: activation-dominated → intensity ≈ K×N / ((K+N) × bpa) ≈ K/2 / bpa
    """
    from prefill_compute import bitlinear_macs_per_layer
    macs = bitlinear_macs_per_layer(seq_len, batch)
    wb   = weight_bytes_per_layer(wprec)
    ab   = activation_bytes_per_layer(seq_len, aprec, batch)
    return macs / (wb + ab) if (wb + ab) > 0 else 0.0


def attention_intensity(seq_len: int, aprec: str) -> float:
    """
    Arithmetic intensity for attention Q@K^T (and softmax@V).

    MACs = 2 × N_HEADS × L² × HEAD_DIM  (per layer, batch=1 for simplicity)
    Bytes = score matrix (dominant) = 2 × N_HEADS × L² × bpa

    Result = 2 × N_HEADS × L² × HEAD_DIM / (2 × N_HEADS × L² × bpa)
           = HEAD_DIM / bpa = 128 / bpa

    CONSTANT w.r.t. L and batch — does not grow with sequence length.
    This means attention is ALWAYS at the same roofline point on any chip,
    regardless of L.  Without FlashAttention, attention is memory-bound
    whenever HEAD_DIM / bpa < ridge_point.
    """
    bpa = ACT_BYTES[aprec]
    return HEAD_DIM / bpa if bpa > 0 else float('inf')


# ---------------------------------------------------------------------------
# Crossover analysis: L at which prefill BitLinear goes compute-bound
# ---------------------------------------------------------------------------

def bitlinear_crossover_L(
    wprec: str, aprec: str, batch: int, ridge_macs_per_byte: float
) -> float | None:
    """
    Sequence length at which prefill BitLinear transitions from memory-bound
    to compute-bound.

    Derived by solving:
        batch × L × LAYER_WEIGHT_ELEMENTS / (LAYER_WEIGHT_ELEMENTS × bpw + batch × L × ACT_DIMS × bpa)
        = ridge

    Let W = LAYER_WEIGHT_ELEMENTS × bpw  (weight bytes per layer)
        A = ACT_DIMS_PER_TOKEN × bpa     (activation bytes per token per layer)
        M = LAYER_WEIGHT_ELEMENTS        (MACs per token per layer)

    Solving for L:
        batch × L × M = ridge × (W + batch × L × A)
        L × (batch × M - ridge × batch × A) = ridge × W
        L = ridge × W / (batch × (M - ridge × A))

    Returns None if no finite crossover exists (always memory- or always compute-bound).
    """
    bpw = WEIGHT_BYTES[wprec]
    bpa = ACT_BYTES[aprec]
    M   = LAYER_WEIGHT_ELEMENTS
    W   = M * bpw
    A   = ACT_DIMS_PER_TOKEN * bpa

    denom = batch * (M - ridge_macs_per_byte * A)
    if denom <= 0:
        return None  # always memory-bound (ridge too high or acts too large)

    return ridge_macs_per_byte * W / denom


if __name__ == "__main__":
    from prefill_compute import SEQUENCE_LENGTHS

    print("Prefill memory breakdown (per layer, batch=1):")
    print(f"\n{'Config':<18}  {'L':>6}  {'Weights':>10}  {'Acts':>10}  {'Scores':>10}  {'KV_wr':>8}  {'Total':>10}")
    print("-" * 80)
    for wprec, aprec in [('fp16', 'fp16'), ('trit', 'int4')]:
        for L in [128, 512, 2048, 4096]:
            d = prefill_total_bytes_per_layer(L, wprec, aprec, 1)
            total = sum(d.values())
            print(f"{wprec}w/{aprec}a        {L:>6}  "
                  f"{d['weights']/1e6:>9.1f}M  "
                  f"{d['activations']/1e6:>9.1f}M  "
                  f"{d['attn_scores']/1e6:>9.1f}M  "
                  f"{d['kv_write']/1e6:>7.1f}M  "
                  f"{total/1e6:>9.1f}M")

    print()
    print("Attention intensity (ops/byte) — constant w.r.t. L:")
    for aprec in ['fp16', 'int8', 'int4', 'int2', 'trit']:
        print(f"  {aprec}: {attention_intensity(1, aprec):.1f} MACs/byte")
