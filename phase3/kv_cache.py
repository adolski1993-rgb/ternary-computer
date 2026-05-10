"""
KV cache sizing and bandwidth analysis.

The KV cache is the second major memory consumer in transformer inference
(after model weights). Unlike weights, KV cache:
  - Grows linearly with context length (O(L) per sequence)
  - Must be read from memory on every decode step (re-read every token generated)
  - Is not compressible via weight precision tricks (stores activations, not weights)
  - Scales with batch size (each sequence has its own KV cache)

For very long contexts or large batches, KV cache loading can dominate
over weight loading even for fp16 models.  For ternary trit-weight models,
where weights are 10× smaller, KV dominates at shorter contexts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from memory_model import N_LAYERS, N_KV_HEADS, HEAD_DIM, ACT_BYTES

SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]


def kv_bytes_per_layer(aprec: str, context_len: int) -> float:
    """Bytes of KV cache for ONE layer ONE sequence at given context length."""
    return 2 * N_KV_HEADS * context_len * HEAD_DIM * ACT_BYTES[aprec]


def kv_bytes_total(aprec: str, context_len: int, batch: int = 1) -> float:
    """Total KV cache bytes: all 30 layers, all batch sequences."""
    return N_LAYERS * kv_bytes_per_layer(aprec, context_len) * batch


def kv_weight_crossover_length(wprec: str, aprec: str) -> float:
    """
    Context length at which KV cache bytes EQUAL total weight bytes.

    Below this length: weights dominate memory traffic.
    Above this length: KV cache dominates.

    Derivation:
        KV_bytes = N_LAYERS × 2 × N_KV_HEADS × L × HEAD_DIM × act_bytes
        W_bytes  = N_LAYERS × weight_elements_per_layer × weight_bytes
        L* = (weight_elements_per_layer × weight_bytes) / (2 × N_KV_HEADS × HEAD_DIM × act_bytes)
    """
    from memory_model import LAYER_WEIGHT_ELEMENTS, WEIGHT_BYTES
    w_per_layer = LAYER_WEIGHT_ELEMENTS * WEIGHT_BYTES[wprec]
    kv_per_token = 2 * N_KV_HEADS * HEAD_DIM * ACT_BYTES[aprec]
    return w_per_layer / kv_per_token


def kv_decode_time_seconds(
    aprec: str,
    context_len: int,
    bandwidth_tbs: float,
    batch: int = 1,
) -> float:
    """
    Time to load all KV cache from memory for one decode step.

    For batch=B, each sequence has its own KV cache (no sharing), so total
    KV bytes scales with batch.  Time = total_bytes / bandwidth.

    Returns time in seconds per BATCH (divide by batch for per-token time).
    """
    total = kv_bytes_total(aprec, context_len, batch)
    return total / (bandwidth_tbs * 1e12)


def weight_decode_time_seconds(
    wprec: str,
    bandwidth_tbs: float,
    batch: int = 1,
) -> float:
    """
    Time to load all model weights from memory for one decode step.

    For batch=B, weight bytes are amortized: one set of weights serves B tokens.
    Time = weight_bytes / bandwidth / batch... actually:
    One forward pass processes B tokens simultaneously; weights are loaded once
    and reused.  So time per forward pass = weight_bytes / bandwidth.
    Time per TOKEN = weight_bytes / bandwidth / batch.
    """
    from memory_model import total_weight_bytes
    return total_weight_bytes(wprec) / (bandwidth_tbs * 1e12) / batch


def hbm_capacity_max_context(
    wprec: str,
    aprec: str,
    hbm_gb: float,
    batch: int = 1,
) -> int:
    """
    Maximum context length that fits in the chip's HBM.

    HBM is used for:
    - Model weights (if not in SRAM)
    - KV cache
    - Working activation buffers (~2× HIDDEN × N_LAYERS × act_bytes = negligible)

    Solves: weight_bytes + batch × kv_bytes_total(L) ≤ hbm_gb × 1e9
    """
    from memory_model import total_weight_bytes, LAYER_WEIGHT_ELEMENTS, WEIGHT_BYTES
    available = hbm_gb * 1e9 - total_weight_bytes(wprec)
    if available <= 0:
        return 0   # weights alone fill HBM
    kv_per_token_all_layers = N_LAYERS * 2 * N_KV_HEADS * HEAD_DIM * ACT_BYTES[aprec]
    return int(available / (batch * kv_per_token_all_layers))


def kv_table() -> str:
    """Human-readable KV cache sizing table."""
    precisions = [('fp16', 'fp16'), ('trit', 'int8'), ('trit', 'int4')]
    header = f"{'Config':<20}" + "".join(f"  {'L='+str(L):>10}" for L in SEQUENCE_LENGTHS)
    lines = ["KV cache total (MB) — all 30 layers, batch=1:", header, "-" * (20 + 12 * 5)]
    for wprec, aprec in precisions:
        label = f"{wprec}w/{aprec}a"
        row = f"{label:<20}"
        for L in SEQUENCE_LENGTHS:
            mb = kv_bytes_total(aprec, L, 1) / 1e6
            row += f"  {mb:>10.1f}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print(kv_table())
    print()
    print("Context length where KV = weight traffic:")
    for wprec, aprec in [('fp16','fp16'), ('trit','int8'), ('trit','int4')]:
        L_cross = kv_weight_crossover_length(wprec, aprec)
        print(f"  {wprec}w/{aprec}a:  L* = {L_cross:,.0f} tokens")
    print()
    print("Max context in H100 80GB HBM (batch=1):")
    for wprec, aprec in [('fp16','fp16'), ('trit','int8'), ('trit','int4')]:
        L_max = hbm_capacity_max_context(wprec, aprec, 80.0, batch=1)
        print(f"  {wprec}w/{aprec}a:  L_max = {L_max:,} tokens")
