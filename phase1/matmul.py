"""
Matrix multiplication: the workhorse of LLM inference.

This is where ternary fundamentally diverges from binary, and where 95%+
of the gate cost lives. We model two versions:

BINARY (fp16 matmul):
  C[i,j] = sum_k A[i,k] * B[k,j]
  Each element multiply: fp16 mantissa multiply (10 bits × 10 bits) +
                          exponent add + normalize + round.
  We approximate as: ~150 gates per fp16 multiply, ~30 gates per fp16 add.
  These are well-established figures from synthesis literature.

TERNARY (trit-weighted matmul):
  C[i,j] = sum_k A[i,k] * W[k,j]
  Where W[k,j] in {-1, 0, +1}. Each element "multiply":
    - if W == 0: skip (1 gate to inspect)
    - if W == +1: add A[i,k] to accumulator
    - if W == -1: subtract A[i,k] from accumulator (free in balanced ternary)
  No multiplier needed. Just a 3-way mux into an int accumulator.

For the ternary case, accumulator is wide (we use ~24-trit accumulator,
then re-quantize to int8 at the end). For binary, accumulator is fp16
throughout (or fp32 with downcast at the end - we model fp16 to be
generous to binary).
"""

from binary_ops import COUNTER
import random


# ---------------------------------------------------------------------------
# Per-element costs (gate counts), based on standard cell library figures
# ---------------------------------------------------------------------------

# Binary fp16:
# - fp16 multiplier: ~150 gates (mantissa mult ~100, exp add ~10, norm ~30, special-case ~10)
# - fp16 adder: ~80 gates (align, add, normalize, round)
# - We use slightly conservative figures favoring binary
GATES_FP16_MUL = 150
GATES_FP16_ADD = 80
GATES_FP16_SQRT = 400      # iterative
GATES_FP16_DIV = 250
GATES_FP16_EXP = 600       # for softmax, expensive

# Binary int8:
# - int8 multiplier: ~64 gates (Wallace tree)
# - int8 add: ~40 gates
# - Used in the int8-activation path on binary side for fairness
GATES_INT8_MUL = 64
GATES_INT8_ADD = 40

# Ternary trit-weighted MAC:
# - decode the trit (1 gate)
# - if nonzero, do an int add of activation to accumulator
# - if -1, the activation gets per-bit/per-trit-negated (cheap, ~5 gates for 8-bit)
# - accumulator add: depends on accumulator width, ~30 gates for 24-trit add
GATES_TRIT_DECODE = 1
GATES_TRIT_NEG_INT8 = 8     # negate an 8-bit value (per-bit NOT + carry chain... or in ternary, free)
GATES_TRIT_ACCUM_ADD = 30   # add 8-bit value to 24-trit accumulator
GATES_TRIT_REQUANTIZE = 50  # at the end: divide by scale, clamp to int8


# ---------------------------------------------------------------------------
# Trit weight matrix generator (random, BitNet-distributed)
# ---------------------------------------------------------------------------

def random_trit_matrix(rows: int, cols: int, seed: int = 0) -> dict:
    """Generate a random trit weight matrix.
    BitNet's absmean quantization tends to produce roughly uniform
    distribution over {-1, 0, +1} after training, so we model that.
    Returns a sparse representation: counts of each value per column,
    which is all we need for gate-counting.
    """
    rng = random.Random(seed)
    # Per-column counts of nonzero trits (this is what matters for gate count)
    nonzero_per_col = []
    pos_per_col = []
    neg_per_col = []
    for _ in range(cols):
        n_pos = sum(1 for _ in range(rows) if rng.random() < 1/3)
        n_neg = sum(1 for _ in range(rows) if rng.random() < 1/3)
        # Cap so we don't exceed row count
        if n_pos + n_neg > rows:
            n_neg = rows - n_pos
        nonzero_per_col.append(n_pos + n_neg)
        pos_per_col.append(n_pos)
        neg_per_col.append(n_neg)
    return {
        "rows": rows,
        "cols": cols,
        "nonzero_per_col": nonzero_per_col,
        "pos_per_col": pos_per_col,
        "neg_per_col": neg_per_col,
    }


# ---------------------------------------------------------------------------
# fp16 matmul: gate-counted
# ---------------------------------------------------------------------------

def binary_matmul_fp16(M: int, K: int, N: int) -> int:
    """Gate count for [M, K] @ [K, N] -> [M, N] in fp16.
    Each output element: K multiplies + (K-1) adds.
    """
    n_muls = M * K * N
    n_adds = M * (K - 1) * N
    gates = n_muls * GATES_FP16_MUL + n_adds * GATES_FP16_ADD
    COUNTER.binary_gates += gates
    return gates


def binary_matmul_int8(M: int, K: int, N: int) -> int:
    """For attention scores (Q @ K^T, both activations are int8).
    Result is wider int (int24-ish) accumulator.
    """
    n_muls = M * K * N
    n_adds = M * (K - 1) * N
    gates = n_muls * GATES_INT8_MUL + n_adds * GATES_INT8_ADD
    COUNTER.binary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# Ternary trit-weighted matmul: gate-counted
# ---------------------------------------------------------------------------

def ternary_matmul_trit_weighted(
    M: int, K: int, N: int, *, seed: int = 0
) -> int:
    """[M, K] (int8 activations) @ [K, N] (trit weights) -> [M, N].

    For each output element C[i, j]:
      - For each k in K: inspect W[k, j]
        - If 0: 1 gate (decode), no accumulator update
        - If +1: GATES_TRIT_ACCUM_ADD (add activation to accumulator)
        - If -1: GATES_TRIT_NEG_INT8 + GATES_TRIT_ACCUM_ADD
      - Final requantize: GATES_TRIT_REQUANTIZE per output element

    With 1/3 zero, 1/3 +1, 1/3 -1 weight distribution:
      avg per element: (1/3)*1 + (1/3)*30 + (1/3)*(8+30) = ~23 gates
    vs binary fp16: 150+80 = ~230 per element. ~10x ratio per MAC.
    """
    weights = random_trit_matrix(K, N, seed=seed)
    gates = 0

    for j in range(N):
        n_zero = K - weights["nonzero_per_col"][j]
        n_pos = weights["pos_per_col"][j]
        n_neg = weights["neg_per_col"][j]

        # Each row of A processes against this column of W
        for _ in range(M):
            # Inspect every trit (decode)
            gates += K * GATES_TRIT_DECODE
            # +1 trits: just add
            gates += n_pos * GATES_TRIT_ACCUM_ADD
            # -1 trits: negate then add
            gates += n_neg * (GATES_TRIT_NEG_INT8 + GATES_TRIT_ACCUM_ADD)
            # Final requantize
            gates += GATES_TRIT_REQUANTIZE

    COUNTER.ternary_gates += gates
    return gates


def ternary_matmul_int8_int8(M: int, K: int, N: int) -> int:
    """Attention Q @ K^T case: both operands are activations (not trits).
    Ternary hardware would still use int8 multipliers here. So this is the
    same cost as binary int8 matmul. INCLUDED so the analysis is honest:
    ternary doesn't win on attention scores.
    """
    n_muls = M * K * N
    n_adds = M * (K - 1) * N
    gates = n_muls * GATES_INT8_MUL + n_adds * GATES_INT8_ADD
    COUNTER.ternary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from architecture import CONFIG

    print("Single QKV-projection matmul cost:")
    print(f"  shape: [seq_len=512, hidden={CONFIG.hidden_size}] @ "
          f"[{CONFIG.hidden_size}, {CONFIG.qkv_out_dim}]")
    print()

    seq_len = 512

    COUNTER.reset()
    b_gates = binary_matmul_fp16(seq_len, CONFIG.hidden_size, CONFIG.qkv_out_dim)
    print(f"  Binary fp16:    {b_gates:>15,} gates")

    COUNTER.reset()
    t_gates = ternary_matmul_trit_weighted(
        seq_len, CONFIG.hidden_size, CONFIG.qkv_out_dim
    )
    print(f"  Ternary (trit): {t_gates:>15,} gates")
    print(f"  Speedup:        {b_gates/t_gates:>15.2f}x")
