"""
One full BitNet b1.58 decoder layer, gate-counted end-to-end.

Architecture (per the BitNet b1.58 2B4T paper):

    x_in  (residual stream)
      |
      RMSNorm  ──────────────────────────┐
      │                                  │
      QKV projection (BitLinear)         │  <- TRIT WEIGHTS
      │                                  │
      Split Q, K, V                      │
      │                                  │
      RoPE on Q, K                       │
      │                                  │
      Attention: softmax(QK^T/√d) V      │  <- INT8 ACTIVATIONS, no trits
      │                                  │
      Output projection (BitLinear)      │  <- TRIT WEIGHTS
      │                                  │
      Residual add  <─────────────────────┘
      │
      x_mid
      │
      RMSNorm  ──────────────────────────┐
      │                                  │
      FFN gate proj (BitLinear)          │  <- TRIT WEIGHTS
      FFN up proj (BitLinear)            │  <- TRIT WEIGHTS
      ReLU²(gate) * up                   │
      FFN down proj (BitLinear)          │  <- TRIT WEIGHTS
      │                                  │
      Residual add  <─────────────────────┘
      │
      x_out

The four BitLinear layers (QKV, output, gate, up, down) are where ternary
hardware shines. Everything else is roughly the same as int8 binary.
"""

from binary_ops import COUNTER
from architecture import CONFIG
from matmul import (
    binary_matmul_fp16, binary_matmul_int8,
    ternary_matmul_trit_weighted, ternary_matmul_int8_int8,
)
from components import (
    binary_rmsnorm_fp16, ternary_rmsnorm_int8,
    binary_rope_fp16, ternary_rope_int8,
    binary_softmax_fp16, ternary_softmax_int8,
    binary_relu2_fp16, ternary_relu2_int8,
    binary_elementwise_mul_fp16, ternary_elementwise_mul_int8,
    binary_elementwise_add_fp16, ternary_elementwise_add_int8,
)


def run_binary_layer(seq_len: int) -> dict:
    """Run one decoder layer in fp16 binary, return per-component gate counts."""
    c = CONFIG
    breakdown = {}

    # ---- Block 1: Attention ----
    COUNTER.reset()
    binary_rmsnorm_fp16(seq_len, c.hidden_size)
    breakdown["rmsnorm_1"] = COUNTER.binary_gates

    COUNTER.reset()
    binary_matmul_fp16(seq_len, c.hidden_size, c.qkv_out_dim)
    breakdown["qkv_proj"] = COUNTER.binary_gates

    COUNTER.reset()
    # RoPE on Q (full heads) and K (kv heads)
    binary_rope_fp16(seq_len, c.num_attention_heads, c.head_dim)
    binary_rope_fp16(seq_len, c.num_key_value_heads, c.head_dim)
    breakdown["rope"] = COUNTER.binary_gates

    COUNTER.reset()
    # Q @ K^T: per head, [seq_len, head_dim] @ [head_dim, seq_len]
    # GQA: each KV head is shared by (n_q_heads / n_kv_heads) Q heads
    # But total work is still per-Q-head
    binary_matmul_fp16(c.num_attention_heads * seq_len, c.head_dim, seq_len)
    # Scale by 1/sqrt(d): seq_len^2 * num_heads multiplies
    breakdown["attention_qk"] = COUNTER.binary_gates

    COUNTER.reset()
    # Softmax: one per (head, query_position)
    # Average attended length under causal mask is L/2
    binary_softmax_fp16(seq_len * c.num_attention_heads, seq_len // 2)
    breakdown["softmax"] = COUNTER.binary_gates

    COUNTER.reset()
    # softmax @ V: [seq_len, seq_len] @ [seq_len, head_dim] per head
    binary_matmul_fp16(c.num_attention_heads * seq_len, seq_len, c.head_dim)
    breakdown["attention_av"] = COUNTER.binary_gates

    COUNTER.reset()
    # Output projection
    binary_matmul_fp16(seq_len, c.hidden_size, c.hidden_size)
    breakdown["attn_out_proj"] = COUNTER.binary_gates

    COUNTER.reset()
    # Residual add
    binary_elementwise_add_fp16(seq_len * c.hidden_size)
    breakdown["residual_1"] = COUNTER.binary_gates

    # ---- Block 2: FFN ----
    COUNTER.reset()
    binary_rmsnorm_fp16(seq_len, c.hidden_size)
    breakdown["rmsnorm_2"] = COUNTER.binary_gates

    COUNTER.reset()
    # Gate and up projection (two separate BitLinear layers)
    binary_matmul_fp16(seq_len, c.hidden_size, c.intermediate_size)
    binary_matmul_fp16(seq_len, c.hidden_size, c.intermediate_size)
    breakdown["ffn_up_gate"] = COUNTER.binary_gates

    COUNTER.reset()
    # ReLU² on gate, then elementwise multiply with up
    binary_relu2_fp16(seq_len * c.intermediate_size)
    binary_elementwise_mul_fp16(seq_len * c.intermediate_size)
    breakdown["ffn_activation"] = COUNTER.binary_gates

    COUNTER.reset()
    # Down projection
    binary_matmul_fp16(seq_len, c.intermediate_size, c.hidden_size)
    breakdown["ffn_down"] = COUNTER.binary_gates

    COUNTER.reset()
    binary_elementwise_add_fp16(seq_len * c.hidden_size)
    breakdown["residual_2"] = COUNTER.binary_gates

    breakdown["TOTAL"] = sum(breakdown.values())
    return breakdown


def run_ternary_layer(seq_len: int) -> dict:
    """Run one decoder layer with trit weights + int8 activations."""
    c = CONFIG
    breakdown = {}

    # ---- Block 1: Attention ----
    COUNTER.reset()
    ternary_rmsnorm_int8(seq_len, c.hidden_size)
    breakdown["rmsnorm_1"] = COUNTER.ternary_gates

    COUNTER.reset()
    # QKV: TRIT-WEIGHTED matmul - this is where ternary wins big
    ternary_matmul_trit_weighted(seq_len, c.hidden_size, c.qkv_out_dim, seed=1)
    breakdown["qkv_proj"] = COUNTER.ternary_gates

    COUNTER.reset()
    ternary_rope_int8(seq_len, c.num_attention_heads, c.head_dim)
    ternary_rope_int8(seq_len, c.num_key_value_heads, c.head_dim)
    breakdown["rope"] = COUNTER.ternary_gates

    COUNTER.reset()
    # Q @ K^T: BOTH operands are activations (int8), not trits
    # Ternary hardware uses int8 multipliers here, same as binary int8
    ternary_matmul_int8_int8(c.num_attention_heads * seq_len, c.head_dim, seq_len)
    breakdown["attention_qk"] = COUNTER.ternary_gates

    COUNTER.reset()
    ternary_softmax_int8(seq_len * c.num_attention_heads, seq_len // 2)
    breakdown["softmax"] = COUNTER.ternary_gates

    COUNTER.reset()
    # softmax @ V: also activation × activation, int8 × int8
    ternary_matmul_int8_int8(c.num_attention_heads * seq_len, seq_len, c.head_dim)
    breakdown["attention_av"] = COUNTER.ternary_gates

    COUNTER.reset()
    # Output proj: TRIT WEIGHTS again
    ternary_matmul_trit_weighted(seq_len, c.hidden_size, c.hidden_size, seed=2)
    breakdown["attn_out_proj"] = COUNTER.ternary_gates

    COUNTER.reset()
    ternary_elementwise_add_int8(seq_len * c.hidden_size)
    breakdown["residual_1"] = COUNTER.ternary_gates

    # ---- Block 2: FFN ----
    COUNTER.reset()
    ternary_rmsnorm_int8(seq_len, c.hidden_size)
    breakdown["rmsnorm_2"] = COUNTER.ternary_gates

    COUNTER.reset()
    # Gate + up: TRIT WEIGHTS
    ternary_matmul_trit_weighted(seq_len, c.hidden_size, c.intermediate_size, seed=3)
    ternary_matmul_trit_weighted(seq_len, c.hidden_size, c.intermediate_size, seed=4)
    breakdown["ffn_up_gate"] = COUNTER.ternary_gates

    COUNTER.reset()
    ternary_relu2_int8(seq_len * c.intermediate_size)
    ternary_elementwise_mul_int8(seq_len * c.intermediate_size)
    breakdown["ffn_activation"] = COUNTER.ternary_gates

    COUNTER.reset()
    # Down: TRIT WEIGHTS
    ternary_matmul_trit_weighted(seq_len, c.intermediate_size, c.hidden_size, seed=5)
    breakdown["ffn_down"] = COUNTER.ternary_gates

    COUNTER.reset()
    ternary_elementwise_add_int8(seq_len * c.hidden_size)
    breakdown["residual_2"] = COUNTER.ternary_gates

    breakdown["TOTAL"] = sum(breakdown.values())
    return breakdown


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seq_len = 512
    print(f"One BitNet b1.58 decoder layer at seq_len={seq_len}")
    print()

    b = run_binary_layer(seq_len)
    t = run_ternary_layer(seq_len)

    print(f"{'Component':<22} {'Binary (fp16)':>18} {'Ternary':>18} {'Ratio':>8}")
    print("-" * 70)
    components = [k for k in b if k != "TOTAL"]
    for k in components:
        ratio = b[k] / t[k] if t[k] > 0 else float('inf')
        print(f"{k:<22} {b[k]:>18,} {t[k]:>18,} {ratio:>7.2f}x")
    print("-" * 70)
    ratio_total = b["TOTAL"] / t["TOTAL"]
    print(f"{'TOTAL':<22} {b['TOTAL']:>18,} {t['TOTAL']:>18,} {ratio_total:>7.2f}x")
    print()
    print(f"Full model (30 layers): "
          f"binary {30*b['TOTAL']:,} vs ternary {30*t['TOTAL']:,}")
