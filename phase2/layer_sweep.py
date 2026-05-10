"""
Full BitNet b1.58 decoder layer, parameterized by activation precision.

Architecture (same as Phase 1):
  x_in → RMSNorm → QKV (BitLinear) → RoPE → Attention → Out-proj (BitLinear)
       → residual → x_mid
       → RMSNorm → FFN gate+up (BitLinear×2) → ReLU² → FFN down (BitLinear)
       → residual → x_out

The binary baseline is always fp16, identical to Phase 1.
The ternary path is parameterized by `prec`, sweeping five activation precisions.

Gate counting is analytical (expected-value), not Monte Carlo.  The expected
value of Phase 1's random-trit sampling agrees with the analytical formula to
within 0.03% (verified against Phase 1 per_component.json).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gate_costs import (
    Precision, PRECISIONS, SEQUENCE_LENGTHS,
    FP16_MUL, FP16_ADD, FP16_SQRT, FP16_DIV, FP16_EXP,
    ACT_MUL, ACT_ADD, ACT_SQRT, ACT_DIV, ACT_EXP,
    ACT_REQUANT, TRIT_DECODE,
    bitlinear_per_pair,
)

# ---------------------------------------------------------------------------
# BitNet b1.58 2B4T architecture constants (from published config.json)
# ---------------------------------------------------------------------------
HIDDEN       = 2560
INTERMEDIATE = 6912
N_HEADS      = 20
N_KV_HEADS   = 5
HEAD_DIM     = 128
KV_DIM       = N_KV_HEADS * HEAD_DIM    # 640
QKV_OUT_DIM  = HIDDEN + 2 * KV_DIM     # 3840


# ---------------------------------------------------------------------------
# Analytical matmul gate counts
# ---------------------------------------------------------------------------

def binary_matmul_fp16_gates(M: int, K: int, N: int) -> int:
    """fp16 matmul [M,K] @ [K,N].  Same formula as Phase 1 matmul.py."""
    return M * K * N * FP16_MUL + M * (K - 1) * N * FP16_ADD


def ternary_bitlinear_gates(M: int, K: int, N: int, prec: Precision) -> int:
    """
    Trit-weighted matmul [M,K] @ [K,N] with activation precision `prec`.

    Per output element: K pairs each costing bitlinear_per_pair(prec) gates,
    plus ACT_REQUANT[prec] to rescale the accumulator.
    Total: M × N × (K × per_pair + requant).
    """
    per_pair = bitlinear_per_pair(prec)
    requant  = ACT_REQUANT[prec]
    return int(M * N * (K * per_pair + requant))


def ternary_act_act_gates(M: int, K: int, N: int, prec: Precision) -> int:
    """
    Activation × activation matmul [M,K] @ [K,N].
    Both operands have precision `prec`; no trit weights, no sparsity skip.
    Used for Q@K^T and softmax@V.
    """
    return M * K * N * ACT_MUL[prec] + M * (K - 1) * N * ACT_ADD[prec]


# ---------------------------------------------------------------------------
# Full decoder layer — binary fp16 baseline (matches Phase 1 exactly)
# ---------------------------------------------------------------------------

def run_binary_layer(seq_len: int) -> dict[str, int]:
    """Gate counts for one decoder layer in fp16 binary."""
    b: dict[str, int] = {}

    # RMSNorm × 2 (pre-attention + pre-FFN)
    per_tok = (
        HIDDEN * FP16_MUL +           # x²
        (HIDDEN - 1) * FP16_ADD +     # reduce sum
        FP16_DIV + FP16_SQRT +        # /D, sqrt
        HIDDEN * FP16_DIV +           # x / rms
        HIDDEN * FP16_MUL             # × gamma
    )
    b["rmsnorm"] = seq_len * per_tok * 2

    # QKV projection
    b["qkv_proj"] = binary_matmul_fp16_gates(seq_len, HIDDEN, QKV_OUT_DIM)

    # RoPE on Q (N_HEADS) and K (N_KV_HEADS)
    rot = seq_len * (HEAD_DIM // 2) * (4 * FP16_MUL + 2 * FP16_ADD)
    b["rope"] = rot * (N_HEADS + N_KV_HEADS)

    # Attention scores Q @ K^T  (per-head, causal)
    b["attention_qk"] = binary_matmul_fp16_gates(N_HEADS * seq_len, HEAD_DIM, seq_len)

    # Softmax (causal mask → average attended length = seq_len / 2)
    atten_len = seq_len // 2
    per_row = (
        (atten_len - 1) * FP16_ADD +
        atten_len * FP16_EXP +
        (atten_len - 1) * FP16_ADD +
        atten_len * FP16_DIV
    )
    b["softmax"] = seq_len * N_HEADS * per_row

    # Attention output  softmax @ V
    b["attention_av"] = binary_matmul_fp16_gates(N_HEADS * seq_len, seq_len, HEAD_DIM)

    # Attention output projection
    b["attn_out_proj"] = binary_matmul_fp16_gates(seq_len, HIDDEN, HIDDEN)

    # Residual adds × 2
    b["residual"] = seq_len * HIDDEN * FP16_ADD * 2

    # FFN: gate + up projections (two separate BitLinear calls)
    b["ffn_up_gate"] = binary_matmul_fp16_gates(seq_len, HIDDEN, INTERMEDIATE) * 2

    # ReLU² + elementwise mul (gating)
    b["ffn_activation"] = seq_len * INTERMEDIATE * (FP16_ADD + FP16_MUL) * 2

    # FFN down projection
    b["ffn_down"] = binary_matmul_fp16_gates(seq_len, INTERMEDIATE, HIDDEN)

    b["TOTAL"] = sum(v for k, v in b.items() if k != "TOTAL")
    return b


# ---------------------------------------------------------------------------
# Full decoder layer — ternary path, parameterized by activation precision
# ---------------------------------------------------------------------------

def run_ternary_layer(seq_len: int, prec: Precision) -> dict[str, int]:
    """
    Gate counts for one decoder layer with trit weights and activation
    precision `prec`.

    BitLinear layers (QKV, out-proj, FFN up/gate/down) use trit-weighted MACs
    whose cost scales with `prec`.  Attention (Q@K^T, softmax@V) uses
    activation × activation MACs, which also scale with `prec` but receive no
    trit-weight benefit.  All non-matmul ops (RMSNorm, RoPE, softmax scalars,
    residuals) also scale with `prec`.
    """
    t: dict[str, int] = {}
    m = ACT_MUL[prec]
    a = ACT_ADD[prec]

    # RMSNorm × 2
    per_tok = (
        HIDDEN * m +
        (HIDDEN - 1) * a +
        ACT_SQRT[prec] +
        HIDDEN * m +
        HIDDEN * m
    )
    t["rmsnorm"] = seq_len * per_tok * 2

    # QKV: TRIT WEIGHTS — big win
    t["qkv_proj"] = ternary_bitlinear_gates(seq_len, HIDDEN, QKV_OUT_DIM, prec)

    # RoPE: activation ops, scales with prec
    rot = seq_len * (HEAD_DIM // 2) * (4 * m + 2 * a)
    t["rope"] = rot * (N_HEADS + N_KV_HEADS)

    # Attention Q@K^T: BOTH OPERANDS ARE ACTIVATIONS — no trit benefit
    t["attention_qk"] = ternary_act_act_gates(N_HEADS * seq_len, HEAD_DIM, seq_len, prec)

    # Softmax scalars
    atten_len = seq_len // 2
    per_row = (
        (atten_len - 1) * a +
        atten_len * ACT_EXP[prec] +
        (atten_len - 1) * a +
        atten_len * ACT_DIV[prec]
    )
    t["softmax"] = seq_len * N_HEADS * per_row

    # softmax @ V: BOTH OPERANDS ARE ACTIVATIONS
    t["attention_av"] = ternary_act_act_gates(N_HEADS * seq_len, seq_len, HEAD_DIM, prec)

    # Attention output projection: TRIT WEIGHTS
    t["attn_out_proj"] = ternary_bitlinear_gates(seq_len, HIDDEN, HIDDEN, prec)

    # Residual adds × 2
    t["residual"] = seq_len * HIDDEN * a * 2

    # FFN gate + up: TRIT WEIGHTS × 2
    t["ffn_up_gate"] = ternary_bitlinear_gates(seq_len, HIDDEN, INTERMEDIATE, prec) * 2

    # ReLU² + elementwise mul
    t["ffn_activation"] = seq_len * INTERMEDIATE * (a + m) * 2

    # FFN down: TRIT WEIGHTS
    t["ffn_down"] = ternary_bitlinear_gates(seq_len, INTERMEDIATE, HIDDEN, prec)

    t["TOTAL"] = sum(v for k, v in t.items() if k != "TOTAL")
    return t


# ---------------------------------------------------------------------------
# Smoke test — run with `python layer_sweep.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seq_len = 2048
    b = run_binary_layer(seq_len)
    print(f"seq_len={seq_len}")
    print(f"{'Prec':<12} {'Ternary gates':>18} {'Speedup':>10}")
    print("-" * 42)
    for prec in PRECISIONS:
        t = run_ternary_layer(seq_len, prec)
        ratio = b["TOTAL"] / t["TOTAL"]
        print(f"{prec:<12} {t['TOTAL']:>18,} {ratio:>9.2f}x")
    print()
    print(f"Binary fp16 total: {b['TOTAL']:,}")
    print(f"Phase 1 reference (int8): 5,613,054,676,992  →  6.72x")
