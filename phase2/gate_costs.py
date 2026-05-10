"""
Gate cost constants for all five activation precisions.

Binary baseline is always fp16 (matching Phase 1). The ternary path is
parameterized: we sweep activation precision from fp16 down to trit to
measure the compound benefit of reducing both weight and activation width.

Scaling rationale
-----------------
Integer multiplier gates scale as n² (array/Wallace-tree, n×n partial products).
Integer adder gates scale as n (ripple-carry full-adder chain).
Lookup-table ops (exp, rsqrt) are roughly constant because they're ROM-based;
narrower precision needs fewer address lines but not proportionally fewer gates.

The fp16 baseline numbers (150 mul, 80 add) carry over from Phase 1 and are
consistent with published cell-library area estimates for 28nm synthesis.
"""

from typing import Literal

Precision = Literal['fp16', 'int8', 'int4', 'int2', 'trit']

PRECISIONS: list[Precision] = ['fp16', 'int8', 'int4', 'int2', 'trit']

PRECISION_LABELS: dict[str, str] = {
    'fp16': 'fp16',
    'int8': 'int8',
    'int4': 'int4',
    'int2': 'int2',
    'trit': 'trit (1.58b)',
}

SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]

# ---------------------------------------------------------------------------
# Binary fp16 baseline — unchanged across all Phase 2 comparisons
# ---------------------------------------------------------------------------
FP16_MUL  = 150   # fp16 mantissa multiply + exp add + normalize + round
FP16_ADD  = 80    # fp16 align, add, normalize, round
FP16_SQRT = 400   # iterative Newton–Raphson
FP16_DIV  = 250
FP16_EXP  = 600   # softmax exp: the most expensive scalar op

# ---------------------------------------------------------------------------
# Per-precision gate costs for activation × activation operations
# (used in attention Q@K and softmax@V, where both operands are activations)
# ---------------------------------------------------------------------------

# Multiplier: n-bit array multiplier scales as n²
# 64 (int8) × (n/8)²
ACT_MUL: dict[str, int] = {
    'fp16': FP16_MUL,   # 150
    'int8': 64,         # Phase 1 baseline
    'int4': 16,         # 64 × (4/8)²
    'int2': 4,          # 64 × (2/8)²
    'trit': 3,          # trit×trit: combinational mux-tree (~3 ternary gates)
}

# Adder: ripple-carry, scales as n
# 40 (int8) × (n/8)
ACT_ADD: dict[str, int] = {
    'fp16': FP16_ADD,   # 80
    'int8': 40,         # Phase 1 baseline
    'int4': 20,         # 40 × (4/8)
    'int2': 10,         # 40 × (2/8)
    'trit': 5,          # ternary full-adder equivalent
}

# ---------------------------------------------------------------------------
# Per-precision gate costs for BitLinear (trit_weight × activation) MACs
# ---------------------------------------------------------------------------

# Cost to negate one activation value (for weight == -1 trits).
# fp16: flip sign bit (1 gate).  int: two's complement NOT-then-add-1 chain.
# trit: a single NEG gate (the free negation of balanced ternary).
ACT_NEG: dict[str, int] = {
    'fp16': 1,    # flip sign bit
    'int8': 8,    # Phase 1 TRIT_NEG_INT8
    'int4': 4,
    'int2': 2,
    'trit': 1,    # per-trit NEG: free in balanced ternary
}

# Cost to add one activation value into the wide running accumulator.
# Phase 1 used 30 for int8 (adding an 8-bit value into a 24-trit accumulator).
ACT_ACCUM_ADD: dict[str, int] = {
    'fp16': 80,   # fp16 adder into fp32 accumulator
    'int8': 30,   # Phase 1 TRIT_ACCUM_ADD
    'int4': 15,   # half of int8
    'int2': 8,
    'trit': 5,    # trit value into accumulator (ternary full-adder chain)
}

# Cost per output element to rescale the accumulated result back to the
# target activation precision.
ACT_REQUANT: dict[str, int] = {
    'fp16': 100,  # fp16 dequantize: multiply by scale + round
    'int8': 50,   # Phase 1 TRIT_REQUANTIZE
    'int4': 35,
    'int2': 25,
    'trit': 40,   # rounding to nearest trit needs sign extraction + compare
}

# Decode one weight trit (always 1 gate, independent of activation precision).
TRIT_DECODE = 1

# ---------------------------------------------------------------------------
# Per-precision gate costs for scalar element-wise ops in non-matmul components
# (RMSNorm, RoPE, softmax, residuals — all tiny vs matmuls at inference scale)
# ---------------------------------------------------------------------------
ACT_SQRT: dict[str, int] = {
    'fp16': FP16_SQRT,  # 400 — iterative
    'int8': 100,        # fixed-point rsqrt via lookup
    'int4': 80,
    'int2': 60,
    'trit': 60,
}

ACT_DIV: dict[str, int] = {
    'fp16': FP16_DIV,
    'int8': 80,
    'int4': 60,
    'int2': 40,
    'trit': 40,
}

# exp() — lookup-table cost is roughly constant regardless of precision
ACT_EXP: dict[str, int] = {
    'fp16': FP16_EXP,   # 600 — full fp16 exp
    'int8': 30,         # lookup table
    'int4': 30,
    'int2': 30,
    'trit': 30,
}


# ---------------------------------------------------------------------------
# Convenience: expected gate cost per (activation, weight-trit) pair
# in a BitLinear matmul.  Used analytically (expected value, not Monte Carlo).
# ---------------------------------------------------------------------------

def bitlinear_per_pair(prec: Precision) -> float:
    """
    Expected gate cost for one inner-product step: one activation element
    multiplied by one weight trit, contributing to one output element.

    Weight distribution: 1/3 zero, 1/3 +1, 1/3 -1  (BitNet absmean).

    For non-trit activations:
        E[gates] = P(w=0)·1 + P(w=+1)·(1+ACC) + P(w=-1)·(1+NEG+ACC)
                 = 1 + (2/3)·ACC + (1/3)·NEG

    For trit activations (both operands are trits):
        P(w=0)=1/3             → decode w only: 1 gate
        P(w≠0, a=0)=2/9        → decode both, skip: 2 gates
        P(w=+1, a≠0)=2/9       → decode both + accumulate: 1+1+5 = 7 gates
        P(w=-1, a≠0)=2/9       → decode both + NEG + accumulate: 1+1+1+5 = 8 gates
        E[gates] = 1/3 + 4/9 + 14/9 + 16/9 = 37/9 ≈ 4.11
    """
    if prec == 'trit':
        return 37 / 9  # ≈ 4.111
    acc = ACT_ACCUM_ADD[prec]
    neg = ACT_NEG[prec]
    return 1.0 + (2 / 3) * acc + (1 / 3) * neg


# Pre-computed for quick reference
BITLINEAR_PER_PAIR: dict[str, float] = {p: bitlinear_per_pair(p) for p in PRECISIONS}

# Binary fp16 per-pair cost (for comparison):  FP16_MUL + FP16_ADD = 230
BINARY_PER_PAIR = FP16_MUL + FP16_ADD  # 230
