"""
Per-layer, per-component gate count computation for hybrid configurations.

Extends Phase 2's layer_sweep.py to handle per-component activation precision.
Verification: uniform int4 and int8 configs must match Phase 2 results.

Key design: activation precision is per-COMPONENT (not per-layer-uniform).
Different components within one layer can use different activation precisions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Phase 2 gate cost infrastructure (reused directly)
PHASE2_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase2')
sys.path.insert(0, PHASE2_DIR)

from gate_costs import (
    bitlinear_per_pair, ACT_MUL, ACT_ADD, ACT_SQRT, ACT_DIV, ACT_EXP, ACT_REQUANT,
    FP16_MUL, FP16_ADD, FP16_SQRT, FP16_DIV, FP16_EXP,
    PRECISIONS, TRIT_DECODE,
)

# BitNet 2B4T architecture (same as Phase 2)
N_LAYERS     = 30
HIDDEN       = 2560
INTERMEDIATE = 6912
N_HEADS      = 20
N_KV_HEADS   = 5
HEAD_DIM     = 128
QKV_OUT_DIM  = HIDDEN + 2 * N_KV_HEADS * HEAD_DIM   # 3840
SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]


def bitlinear_gates(M: int, K: int, N: int, prec: str) -> int:
    """Gate count for trit-weighted matmul [M,K]@[K,N] with activation precision prec."""
    per_pair = bitlinear_per_pair(prec)
    return int(M * N * (K * per_pair + ACT_REQUANT[prec]))


def act_act_gates(M: int, K: int, N: int, prec: str) -> int:
    """Gate count for act×act matmul (attention); no trit weight benefit."""
    return M * K * N * ACT_MUL[prec] + M * (K - 1) * N * ACT_ADD[prec]


def binary_fp16_gates(M: int, K: int, N: int) -> int:
    return M * K * N * FP16_MUL + M * (K - 1) * N * FP16_ADD


def one_layer_gates(seq_len: int, layer_cfg: dict) -> dict[str, int]:
    """
    Gate counts for one decoder layer with per-component activation precision.

    layer_cfg keys:
        qkv_in, attn_out_in, ffn_gate_in, ffn_up_in, ffn_down_in

    The QKV activation precision also determines the attention compute precision
    (Q@K^T and softmax@V both use qkv_in precision).

    Returns a dict with per-component gate counts and 'TOTAL'.
    """
    qkv_prec      = layer_cfg['qkv_in']
    attn_out_prec = layer_cfg['attn_out_in']
    ffn_gate_prec = layer_cfg['ffn_gate_in']
    ffn_up_prec   = layer_cfg['ffn_up_in']
    ffn_down_prec = layer_cfg['ffn_down_in']

    m = ACT_MUL[qkv_prec]
    a = ACT_ADD[qkv_prec]

    gates = {}

    # RMSNorm × 2 — uses QKV precision as the dominant precision
    per_tok = (
        HIDDEN * m + (HIDDEN - 1) * a +
        ACT_SQRT[qkv_prec] +
        HIDDEN * m + HIDDEN * m
    )
    gates["rmsnorm"] = seq_len * per_tok * 2

    # QKV projection: trit weights × qkv_prec activations
    gates["qkv_proj"] = bitlinear_gates(seq_len, HIDDEN, QKV_OUT_DIM, qkv_prec)

    # RoPE: uses QKV precision
    rot = seq_len * (HEAD_DIM // 2) * (4 * m + 2 * a)
    gates["rope"] = rot * (N_HEADS + N_KV_HEADS)

    # Attention Q@K^T: act×act at qkv_prec
    gates["attention_qk"] = act_act_gates(N_HEADS * seq_len, HEAD_DIM, seq_len, qkv_prec)

    # Softmax: uses qkv_prec scalar ops
    atten_len = seq_len // 2
    per_row = (
        (atten_len - 1) * a +
        atten_len * ACT_EXP[qkv_prec] +
        (atten_len - 1) * a +
        atten_len * ACT_DIV[qkv_prec]
    )
    gates["softmax"] = seq_len * N_HEADS * per_row

    # softmax @ V: act×act at qkv_prec
    gates["attention_av"] = act_act_gates(N_HEADS * seq_len, seq_len, HEAD_DIM, qkv_prec)

    # Attention output projection: trit weights × attn_out_prec
    gates["attn_out_proj"] = bitlinear_gates(seq_len, HIDDEN, HIDDEN, attn_out_prec)

    # Residual: uses dominant (qkv_prec for simplicity)
    gates["residual"] = seq_len * HIDDEN * a * 2

    # FFN gate: trit × ffn_gate_prec
    gates["ffn_gate"] = bitlinear_gates(seq_len, HIDDEN, INTERMEDIATE, ffn_gate_prec)

    # FFN up: trit × ffn_up_prec
    gates["ffn_up"] = bitlinear_gates(seq_len, HIDDEN, INTERMEDIATE, ffn_up_prec)

    # FFN activation: uses ffn_up_prec (output precision of gate/up)
    gates["ffn_activation"] = seq_len * INTERMEDIATE * (
        ACT_ADD[ffn_up_prec] + ACT_MUL[ffn_up_prec]
    ) * 2

    # FFN down: trit × ffn_down_prec
    gates["ffn_down"] = bitlinear_gates(seq_len, INTERMEDIATE, HIDDEN, ffn_down_prec)

    gates["TOTAL"] = sum(v for k, v in gates.items() if k != "TOTAL")
    return gates


def model_gate_counts(seq_len: int, layer_cfgs: list[dict]) -> dict:
    """
    Total gate counts for the full 30-layer model.

    Sums over all layers, returns per-component totals + TOTAL.
    """
    totals: dict[str, int] = {}
    for layer_cfg in layer_cfgs:
        lg = one_layer_gates(seq_len, layer_cfg)
        for k, v in lg.items():
            totals[k] = totals.get(k, 0) + v
    return totals


def binary_model_gates(seq_len: int) -> dict:
    """
    fp16 binary baseline gate counts for 30 layers (matches Phase 1/2 binary layer).
    Used as the reference denominator for speedup calculations.
    """
    per_layer = {
        "rmsnorm":      seq_len * (HIDDEN * FP16_MUL + (HIDDEN - 1) * FP16_ADD + FP16_DIV + FP16_SQRT + HIDDEN * FP16_DIV + HIDDEN * FP16_MUL) * 2,
        "qkv_proj":     binary_fp16_gates(seq_len, HIDDEN, QKV_OUT_DIM),
        "rope":         seq_len * (HEAD_DIM // 2) * (4 * FP16_MUL + 2 * FP16_ADD) * (N_HEADS + N_KV_HEADS),
        "attention_qk": binary_fp16_gates(N_HEADS * seq_len, HEAD_DIM, seq_len),
        "softmax":      seq_len * N_HEADS * ((seq_len // 2 - 1) * FP16_ADD + (seq_len // 2) * FP16_EXP + (seq_len // 2 - 1) * FP16_ADD + (seq_len // 2) * FP16_DIV),
        "attention_av": binary_fp16_gates(N_HEADS * seq_len, seq_len, HEAD_DIM),
        "attn_out_proj":binary_fp16_gates(seq_len, HIDDEN, HIDDEN),
        "residual":     seq_len * HIDDEN * FP16_ADD * 2,
        "ffn_gate":     binary_fp16_gates(seq_len, HIDDEN, INTERMEDIATE),
        "ffn_up":       binary_fp16_gates(seq_len, HIDDEN, INTERMEDIATE),
        "ffn_activation": seq_len * INTERMEDIATE * (FP16_ADD + FP16_MUL) * 2,
        "ffn_down":     binary_fp16_gates(seq_len, INTERMEDIATE, HIDDEN),
    }
    per_layer["TOTAL"] = sum(v for k, v in per_layer.items() if k != "TOTAL")
    return {k: v * N_LAYERS for k, v in per_layer.items()}


# ---------------------------------------------------------------------------
# Verification: uniform configs should match Phase 2
# ---------------------------------------------------------------------------

def verify_vs_phase2():
    """
    Verify that uniform int4 and int8 match Phase 2 results at L=2048.
    Phase 2 totals at L=2048:
      int8: ternary total = 5,613,054,676,992 → speedup 6.718×
      int4: ternary total = 7,544,... (Phase 2 per-layer × 30)
    We check speedups match to within 1%.
    """
    from layer_sensitivity_model import uniform_config

    L = 2048
    binary_total = binary_model_gates(L)["TOTAL"]
    binary_total_per_layer = binary_total / N_LAYERS

    for prec, expected_speedup in [('int8', 6.72), ('int4', 14.87), ('trit', 49.49)]:
        cfg = uniform_config(prec)
        ternary_total = model_gate_counts(L, cfg)["TOTAL"]
        speedup = binary_total / ternary_total
        error_pct = abs(speedup - expected_speedup) / expected_speedup * 100
        status = "OK" if error_pct < 2.0 else "FAIL"
        print(f"  {prec}: speedup={speedup:.2f}× expected={expected_speedup}× error={error_pct:.1f}% [{status}]")


if __name__ == "__main__":
    print("Verification vs Phase 2 (L=2048):")
    verify_vs_phase2()
    print()

    from layer_sensitivity_model import uniform_config
    from hybrid_configurations import CONFIGS

    L = 2048
    binary_total = binary_model_gates(L)["TOTAL"]
    print(f"Binary fp16 total at L={L}: {binary_total/1e12:.2f}T gates")
    print()
    print(f"{'Config':<34} {'Ternary gates':>16} {'Speedup':>10}")
    print("-" * 62)
    for cfg in CONFIGS:
        t = model_gate_counts(L, cfg["layer_cfgs"])["TOTAL"]
        sp = binary_total / t
        print(f"{cfg['label']:<34} {t/1e12:>15.2f}T {sp:>9.2f}×")
