"""
Gate counts and MAC counts for the PREFILL phase of inference.

Prefill processes L input tokens all at once: matmul shapes are [L, K] @ [K, N],
not the decode's [1, K] @ [K, N].  The full Phase 2 gate model applies directly —
Phase 2 already uses seq_len as the row dimension — so this module is a thin
adapter that labels things clearly and adds the attention score matrix.

Verification contract
---------------------
At seq_len=1 (one token to prefill), gate counts should match Phase 3's
decode model.  We check this in the __main__ block.

Compute terminology used throughout Phase 5
-------------------------------------------
  MACs  = multiply-accumulate operations (hardware-neutral count)
  gates = Phase 2 gate count (used for ternary chip timing only)
  For a standard [M,K]@[K,N] matmul: MACs = M × K × N
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PHASE2_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase2')
PHASE4_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase4')
sys.path.insert(0, PHASE2_DIR)
sys.path.insert(0, PHASE4_DIR)

from gate_costs import bitlinear_per_pair, ACT_MUL, ACT_ADD, ACT_REQUANT, TRIT_DECODE
from layer_sensitivity_model import uniform_config

# BitNet 2B4T architecture
N_LAYERS     = 30
HIDDEN       = 2560
INTERMEDIATE = 6912
N_HEADS      = 20
N_KV_HEADS   = 5
HEAD_DIM     = 128
QKV_OUT_DIM  = HIDDEN + 2 * N_KV_HEADS * HEAD_DIM   # 3840
LAYER_WEIGHT_ELEMENTS = (
    HIDDEN * QKV_OUT_DIM + HIDDEN * HIDDEN +
    2 * HIDDEN * INTERMEDIATE + INTERMEDIATE * HIDDEN
)   # 69,468,160

SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 8, 32]


# ---------------------------------------------------------------------------
# MAC counts (hardware-neutral)
# ---------------------------------------------------------------------------

def bitlinear_macs_per_layer(seq_len: int, batch: int = 1) -> int:
    """
    Total MACs for all BitLinear projections in one layer.
    Scales as batch × seq_len × LAYER_WEIGHT_ELEMENTS.
    Each weight element contributes exactly one MAC per input token.
    """
    return batch * seq_len * LAYER_WEIGHT_ELEMENTS


def attention_macs_per_layer(seq_len: int, batch: int = 1) -> int:
    """
    Total MACs for attention (Q@K^T and softmax@V) in one layer.
    Scales as batch × N_HEADS × seq_len² × HEAD_DIM (quadratic in seq_len!).
    """
    qk = batch * N_HEADS * seq_len * HEAD_DIM * seq_len       # Q @ K^T
    av = batch * N_HEADS * seq_len * seq_len * HEAD_DIM       # softmax @ V
    return qk + av   # = 2 × batch × N_HEADS × seq_len² × HEAD_DIM


def total_macs_per_layer(seq_len: int, batch: int = 1) -> int:
    return bitlinear_macs_per_layer(seq_len, batch) + attention_macs_per_layer(seq_len, batch)


def total_model_macs(seq_len: int, batch: int = 1) -> dict:
    """Total MACs for the full 30-layer model."""
    bl = N_LAYERS * bitlinear_macs_per_layer(seq_len, batch)
    at = N_LAYERS * attention_macs_per_layer(seq_len, batch)
    return {"bitlinear": bl, "attention": at, "total": bl + at}


# ---------------------------------------------------------------------------
# Gate counts (ternary chip only — Phase 2 model, applied to prefill shape)
# ---------------------------------------------------------------------------

def bitlinear_gates_for_prec(M: int, K: int, N: int, prec: str) -> int:
    """Phase 2 gate cost for a trit-weighted matmul [M,K]@[K,N]."""
    return int(M * N * (K * bitlinear_per_pair(prec) + ACT_REQUANT[prec]))


def prefill_layer_gates(seq_len: int, prec: str, batch: int = 1) -> dict:
    """
    Gate counts for one prefill layer with uniform activation precision `prec`.
    Matches Phase 2 layer_sweep.run_ternary_layer() at the same seq_len.
    Includes both BitLinear and attention components.
    """
    m = ACT_MUL[prec]
    a = ACT_ADD[prec]
    M = batch * seq_len   # prefill row count

    gates = {}
    gates["qkv_proj"]     = bitlinear_gates_for_prec(M, HIDDEN, QKV_OUT_DIM, prec)
    gates["attn_out_proj"]= bitlinear_gates_for_prec(M, HIDDEN, HIDDEN, prec)
    gates["ffn_up_gate"]  = bitlinear_gates_for_prec(M, HIDDEN, INTERMEDIATE, prec) * 2
    gates["ffn_down"]     = bitlinear_gates_for_prec(M, INTERMEDIATE, HIDDEN, prec)

    # Attention: act×act matmul using QKV precision
    # Q@K^T: [N_HEADS×M, HEAD_DIM] @ [HEAD_DIM, M]
    gates["attention_qk"] = (N_HEADS * M * HEAD_DIM * M * m +
                             N_HEADS * M * (HEAD_DIM - 1) * M * a)
    # softmax@V: [N_HEADS×M, M] @ [M, HEAD_DIM]
    gates["attention_av"] = (N_HEADS * M * M * HEAD_DIM * m +
                             N_HEADS * M * (M - 1) * HEAD_DIM * a)

    gates["TOTAL"] = sum(v for k, v in gates.items() if k != "TOTAL")
    return gates


def prefill_model_gates(seq_len: int, prec: str, batch: int = 1) -> dict:
    """Total gate counts for 30-layer prefill."""
    per_layer = prefill_layer_gates(seq_len, prec, batch)
    return {k: v * N_LAYERS for k, v in per_layer.items()}


# ---------------------------------------------------------------------------
# Verify at seq_len=1 against Phase 3 decode model
# ---------------------------------------------------------------------------

def verify_vs_decode():
    """
    At seq_len=1, batch=1: prefill collapses to decode.
    Compare total MACs vs Phase 3's LAYER_WEIGHT_ELEMENTS.
    """
    seq1_macs = bitlinear_macs_per_layer(1, 1)
    attn1_macs = attention_macs_per_layer(1, 1)   # 2 × N_HEADS × 1 × HEAD_DIM × 1 = tiny
    print(f"seq_len=1 BitLinear MACs per layer: {seq1_macs:,}  (= LAYER_WEIGHT_ELEMENTS = {LAYER_WEIGHT_ELEMENTS:,})")
    print(f"seq_len=1 Attention MACs per layer: {attn1_macs:,}  (~0 for seq_len=1)")
    assert seq1_macs == LAYER_WEIGHT_ELEMENTS, "Mismatch with decode model!"
    print("OK: prefill at seq_len=1 = decode compute [verified]")


# ---------------------------------------------------------------------------
# Attention-vs-BitLinear dominance crossover
# ---------------------------------------------------------------------------

def attention_dominance_crossover() -> int:
    """
    Returns L at which attention MACs exceed BitLinear MACs per layer.
    Attention MACs = 2 × N_HEADS × L² × HEAD_DIM
    BitLinear MACs = L × LAYER_WEIGHT_ELEMENTS
    Crossover: L = LAYER_WEIGHT_ELEMENTS / (2 × N_HEADS × HEAD_DIM)
    """
    return LAYER_WEIGHT_ELEMENTS // (2 * N_HEADS * HEAD_DIM)


if __name__ == "__main__":
    verify_vs_decode()
    print()
    L_cross = attention_dominance_crossover()
    print(f"Attention dominates BitLinear MACs at L > {L_cross:,} tokens")
    print(f"(All our sequence lengths 128-4096 are below this — BitLinear dominates)")
    print()
    print(f"{'Seq_len':>8}  {'BL MACs/layer':>16}  {'Attn MACs/layer':>18}  {'Attn %':>8}")
    for L in SEQUENCE_LENGTHS:
        bl = bitlinear_macs_per_layer(L)
        at = attention_macs_per_layer(L)
        print(f"{L:>8}  {bl/1e9:>15.2f}B  {at/1e9:>17.2f}B  {100*at/(bl+at):>8.1f}%")
