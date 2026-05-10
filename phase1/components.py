"""
Non-matmul components of the decoder layer.

These are operations where ternary doesn't fundamentally win, because they
operate on activations only (not weights). We still model them carefully
so the per-component breakdown is honest.

Key insight: for these components, ternary hardware would still use int8
arithmetic, so the cost is approximately the same as binary int8. The
binary baseline uses fp16. So we DO see some advantage here too because
int8 ops are cheaper than fp16, but it's not the dramatic ternary win
seen in matmuls.
"""

from binary_ops import COUNTER
from matmul import (
    GATES_FP16_MUL, GATES_FP16_ADD, GATES_FP16_SQRT, GATES_FP16_DIV, GATES_FP16_EXP,
    GATES_INT8_MUL, GATES_INT8_ADD,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
# RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
# Per token: D multiplies for x^2, D adds for sum, 1 div by D, 1 sqrt,
#            D divs by sqrt, D mul by gamma
# Total ops per token: ~4D mul/add + 1 sqrt + D div

def binary_rmsnorm_fp16(seq_len: int, dim: int) -> int:
    """fp16 RMSNorm."""
    per_token = (
        dim * GATES_FP16_MUL +              # x^2
        (dim - 1) * GATES_FP16_ADD +        # sum
        GATES_FP16_DIV +                    # /D
        GATES_FP16_SQRT +                   # sqrt
        dim * GATES_FP16_DIV +              # x / rms
        dim * GATES_FP16_MUL                # * gamma
    )
    gates = seq_len * per_token
    COUNTER.binary_gates += gates
    return gates


def ternary_rmsnorm_int8(seq_len: int, dim: int) -> int:
    """int8 RMSNorm using fixed-point arithmetic.
    Approx 0.4× the cost of fp16 because int8 ops are cheaper.
    """
    per_token = (
        dim * GATES_INT8_MUL +              # x^2
        (dim - 1) * GATES_INT8_ADD +        # sum (in wider accumulator)
        100 +                                # rsqrt via lookup table
        dim * GATES_INT8_MUL +              # x * rsqrt(scaled)
        dim * GATES_INT8_MUL                # * gamma
    )
    gates = seq_len * per_token
    COUNTER.ternary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------
# Per (token, head): apply 2D rotation to each pair of dimensions in head_dim.
# That's head_dim/2 rotations, each: 4 mul + 2 add (real and imag parts).
# Cosine/sine values are precomputed lookup tables, so we don't count their
# generation cost (but we do count the muls/adds).

def binary_rope_fp16(seq_len: int, num_heads: int, head_dim: int) -> int:
    """fp16 RoPE."""
    rotations_per_token = num_heads * (head_dim // 2)
    cost_per_rotation = 4 * GATES_FP16_MUL + 2 * GATES_FP16_ADD
    gates = seq_len * rotations_per_token * cost_per_rotation
    COUNTER.binary_gates += gates
    return gates


def ternary_rope_int8(seq_len: int, num_heads: int, head_dim: int) -> int:
    """int8 RoPE with int8 cos/sin lookup tables."""
    rotations_per_token = num_heads * (head_dim // 2)
    cost_per_rotation = 4 * GATES_INT8_MUL + 2 * GATES_INT8_ADD
    gates = seq_len * rotations_per_token * cost_per_rotation
    COUNTER.ternary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
# Per row of length L: 1 max search (L-1 comparisons), L exps, L-1 sums, L divs
# We're careful to model this; softmax is sometimes a bottleneck on long
# sequences.

def binary_softmax_fp16(seq_len: int, attended_len: int) -> int:
    """One softmax per (head, row), where row has `attended_len` columns."""
    per_row = (
        (attended_len - 1) * GATES_FP16_ADD +    # max via comparisons (~adds)
        attended_len * GATES_FP16_EXP +          # expensive part
        (attended_len - 1) * GATES_FP16_ADD +    # sum
        attended_len * GATES_FP16_DIV            # normalize
    )
    gates = seq_len * per_row  # one softmax per query position
    COUNTER.binary_gates += gates
    return gates


def ternary_softmax_int8(seq_len: int, attended_len: int) -> int:
    """int8 softmax via lookup tables (standard for int8 inference).
    exp() becomes a lookup (~30 gates per call).
    """
    per_row = (
        (attended_len - 1) * GATES_INT8_ADD +    # max
        attended_len * 30 +                      # exp lookup (cheap!)
        (attended_len - 1) * GATES_INT8_ADD +    # sum
        attended_len * 80                        # int division
    )
    gates = seq_len * per_row
    COUNTER.ternary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# ReLU² activation (squared ReLU, BitNet's choice)
# ---------------------------------------------------------------------------
# Per element: 1 compare (>0?), 1 multiply (x*x if positive)

def binary_relu2_fp16(n_elements: int) -> int:
    gates = n_elements * (GATES_FP16_ADD + GATES_FP16_MUL)
    COUNTER.binary_gates += gates
    return gates


def ternary_relu2_int8(n_elements: int) -> int:
    gates = n_elements * (GATES_INT8_ADD + GATES_INT8_MUL)
    COUNTER.ternary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# Element-wise ops (gating, residuals)
# ---------------------------------------------------------------------------

def binary_elementwise_mul_fp16(n_elements: int) -> int:
    gates = n_elements * GATES_FP16_MUL
    COUNTER.binary_gates += gates
    return gates


def ternary_elementwise_mul_int8(n_elements: int) -> int:
    gates = n_elements * GATES_INT8_MUL
    COUNTER.ternary_gates += gates
    return gates


def binary_elementwise_add_fp16(n_elements: int) -> int:
    """For residual connections."""
    gates = n_elements * GATES_FP16_ADD
    COUNTER.binary_gates += gates
    return gates


def ternary_elementwise_add_int8(n_elements: int) -> int:
    gates = n_elements * GATES_INT8_ADD
    COUNTER.ternary_gates += gates
    return gates


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from architecture import CONFIG

    seq_len = 512
    print("Non-matmul component costs at seq_len=512:")
    print()

    for name, b_fn, t_fn, args in [
        ("RMSNorm",
         lambda: binary_rmsnorm_fp16(seq_len, CONFIG.hidden_size),
         lambda: ternary_rmsnorm_int8(seq_len, CONFIG.hidden_size),
         None),
        ("RoPE (Q heads only)",
         lambda: binary_rope_fp16(seq_len, CONFIG.num_attention_heads, CONFIG.head_dim),
         lambda: ternary_rope_int8(seq_len, CONFIG.num_attention_heads, CONFIG.head_dim),
         None),
        ("Softmax (causal mask, avg L/2)",
         lambda: binary_softmax_fp16(seq_len, seq_len // 2) * CONFIG.num_attention_heads,
         lambda: ternary_softmax_int8(seq_len, seq_len // 2) * CONFIG.num_attention_heads,
         None),
        ("ReLU² (FFN)",
         lambda: binary_relu2_fp16(seq_len * CONFIG.intermediate_size),
         lambda: ternary_relu2_int8(seq_len * CONFIG.intermediate_size),
         None),
    ]:
        COUNTER.reset()
        b = b_fn()
        COUNTER.reset()
        t = t_fn()
        print(f"  {name:<35} binary {b:>13,}  ternary {t:>13,}  ratio {b/t:.2f}x")
