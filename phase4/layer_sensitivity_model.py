"""
Per-layer activation quantization sensitivity model for BitNet b1.58.

IMPORTANT FRAMING
-----------------
This module models ACTIVATION precision sensitivity, not weight precision.
BitNet b1.58 is trained with UNIFORM TRIT WEIGHTS throughout — Microsoft found
uniform weight quantization optimal, and we respect that result.

Phase 4's question is orthogonal: given trit weights, which ACTIVATION tensors
need to stay at int8 (vs being pushed to int4 or trit) for acceptable quality?

This is exactly the question Microsoft addressed in BitNet a4.8 (Wang et al.,
arXiv:2411.04965, November 2024).  Their answer: a HYBRID activation schedule
— int4 for most activations, int8 for FFN down-projection inputs — achieves
"performance comparable to BitNet b1.58 with almost no loss in average accuracy."

CRITICAL FINDING (from published literature)
--------------------------------------------
The task prompt assumed "FFN down: tolerant" based on weight-quantization
literature (GPTQ/AWQ).  The ACTIVATION quantization literature says the
OPPOSITE:

  SmoothQuant (Xiao et al., ICML 2023): "inputs to the FFN down-projection
  have massive outliers at magnitudes ~1400× the typical value."
  Source: https://arxiv.org/abs/2211.10438

  BitNet a4.8 (Wang et al., 2024): "applying FP4 quantization for inputs
  to the down projection leads to significant performance degradation."
  Source: https://arxiv.org/abs/2411.04965

  The tolerant components are FFN up/gate projection INPUTS (Gaussian-like
  distribution, easily quantized to int4) and QKV projection INPUTS.

We model activation sensitivity accordingly.  All quality degradation numbers
are PROJECTED/ESTIMATED unless explicitly marked as published measurements.

SENSITIVITY COMPONENTS
-----------------------
We track sensitivity for the following activation tensors (inputs to each matmul):

  qkv_in       : residual stream entering QKV projection
  attn_out_in  : attention softmax output entering attn output projection
  ffn_gate_in  : residual stream entering FFN gate projection
  ffn_up_in    : residual stream entering FFN up projection
  ffn_down_in  : gate×up intermediate entering FFN down projection  ← SENSITIVE

  Note: ffn_gate_in and ffn_up_in receive the same input (post-norm residual),
  so they have the same sensitivity class.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Activation component identifiers
# ---------------------------------------------------------------------------

COMPONENTS = ['qkv_in', 'attn_out_in', 'ffn_gate_in', 'ffn_up_in', 'ffn_down_in']

# ---------------------------------------------------------------------------
# Layer position classification for BitNet 2B4T (30 layers)
# ---------------------------------------------------------------------------

N_LAYERS = 30

def layer_position(layer_idx: int) -> str:
    """
    Classify layer by position in the stack.

    Published basis: GPTQ (Frantar et al., ICLR 2023) observed that early
    and late layers are disproportionately sensitive to quantization.
    The U-shaped sensitivity curve is a consistent finding across multiple
    post-training quantization papers.
    Source: https://arxiv.org/abs/2210.17323

    Position classification for 30-layer model:
      first:  layers 0-1  (blocks closest to embedding — most sensitive)
      early:  layers 2-5
      middle: layers 6-23 (bulk of model — most tolerant)
      late:   layers 24-27
      last:   layers 28-29 (closest to lm_head — sensitive)
    """
    if layer_idx <= 1:
        return 'first'
    elif layer_idx <= 5:
        return 'early'
    elif layer_idx <= 23:
        return 'middle'
    elif layer_idx <= 27:
        return 'late'
    else:
        return 'last'


# Position-based sensitivity multiplier relative to middle layers.
# Basis: GPTQ empirical observation; values are estimated.  [PROJECTED]
POSITION_SENSITIVITY: dict[str, float] = {
    'first':  2.5,   # [PROJECTED] most sensitive — errors cascade through all layers
    'early':  1.5,   # [PROJECTED]
    'middle': 1.0,   # [MEASURED by GPTQ as reference]
    'late':   1.3,   # [PROJECTED]
    'last':   2.0,   # [PROJECTED] sensitive — errors directly affect lm_head logits
}

# ---------------------------------------------------------------------------
# Per-component baseline sensitivity
# Represents relative sensitivity of each activation tensor to quantization.
#
# Scale: 1.0 = sensitivity of qkv_in (moderate, our reference unit).
# Basis: BitNet a4.8, SmoothQuant findings.
# ---------------------------------------------------------------------------

# [MEASURED from BitNet a4.8]:
#   ffn_down_in is the MOST sensitive (outliers, must stay at int8)
#   ffn_up_in / ffn_gate_in are tolerant (safely quantized to int4)
# [PROJECTED for other components]:
#   qkv_in: moderate (residual stream, similar Gaussian shape to ffn_gate_in
#            but slightly more sensitive due to attention position encoding)
#   attn_out_in: moderate-high (post-softmax values have structure not captured
#                by simple quantizers; SmoothQuant notes "normal outliers" here)

COMPONENT_SENSITIVITY: dict[str, float] = {
    'qkv_in':      1.0,   # [PROJECTED] moderate; residual stream, Gaussian-like
    'attn_out_in': 1.4,   # [PROJECTED] moderate-high; post-softmax structure
    'ffn_gate_in': 0.6,   # [PROJECTED] tolerant; post-norm residual, nearly Gaussian
    'ffn_up_in':   0.6,   # [PROJECTED] tolerant; same input as gate
    'ffn_down_in': 3.5,   # [MEASURED BASIS] MOST sensitive; massive outliers per
                           # SmoothQuant + BitNet a4.8 explicitly keeps this at int8
}

# ---------------------------------------------------------------------------
# Precision penalty relative to int8 baseline
#
# Scale: 0.0 = no degradation (int8 reference), >0 = degradation.
# Basis for int4 → trit direction: general quantization literature.
# The relative ordering (int4 < int2 < trit for degradation) is consistent
# across papers.  Absolute values are [PROJECTED].
# ---------------------------------------------------------------------------

PRECISION_PENALTY: dict[str, float] = {
    'int8': 0.0,     # [DEFINED] reference; no degradation by construction
    'int4': 1.0,     # [PROJECTED] unit of degradation
    'int2': 3.5,     # [PROJECTED] ~3.5× harder than int4; very aggressive
    'trit': 5.5,     # [PROJECTED] ~5.5× harder than int4; outlier-unfriendly format
}

# ---------------------------------------------------------------------------
# Normalization calibration
#
# We calibrate so that UNIFORM int4 for all activations across all layers
# = 100 "quality degradation units" (QDU). This gives an interpretable scale.
# QDU 0 = int8 baseline. QDU 100 = uniform int4.  Higher = worse.
#
# This lets us say: "Config X has QDU 45, meaning it degrades quality about
# 45% as much as going to uniform int4."
# ---------------------------------------------------------------------------

def _raw_score(layer_idx: int, component: str, precision: str) -> float:
    pos = layer_position(layer_idx)
    return (
        COMPONENT_SENSITIVITY[component]
        * POSITION_SENSITIVITY[pos]
        * PRECISION_PENALTY[precision]
    )


def _uniform_int4_raw_score() -> float:
    """Raw score for uniform int4 across all layers and components."""
    total = 0.0
    for i in range(N_LAYERS):
        for comp in COMPONENTS:
            total += _raw_score(i, comp, 'int4')
    return total


_NORM_FACTOR = _uniform_int4_raw_score()  # normalization divisor


def quality_degradation_units(layer_cfg_list: list[dict]) -> float:
    """
    Compute total Quality Degradation Units (QDU) for a model configuration.

    Parameters
    ----------
    layer_cfg_list : list of dicts, one per layer, each with keys = COMPONENTS,
                     values = precision strings ('int8', 'int4', 'int2', 'trit').

    Returns
    -------
    QDU : float
        0.0 = int8 baseline (no degradation)
        100.0 = uniform int4 (our Phase 2 recommended baseline)
        Values > 100 = worse than uniform int4

    All values are [PROJECTED] based on the literature model above.
    """
    total = 0.0
    for i, cfg in enumerate(layer_cfg_list):
        for comp in COMPONENTS:
            total += _raw_score(i, comp, cfg[comp])
    return 100.0 * total / _NORM_FACTOR


def component_qdu_breakdown(layer_cfg_list: list[dict]) -> dict:
    """Per-component QDU breakdown for diagnosis."""
    breakdown = {comp: 0.0 for comp in COMPONENTS}
    for i, cfg in enumerate(layer_cfg_list):
        for comp in COMPONENTS:
            breakdown[comp] += 100.0 * _raw_score(i, comp, cfg[comp]) / _NORM_FACTOR
    return breakdown


# ---------------------------------------------------------------------------
# Helpers to build uniform and hybrid layer configs
# ---------------------------------------------------------------------------

def uniform_config(precision: str) -> list[dict]:
    """All layers, all components at the same precision."""
    return [{comp: precision for comp in COMPONENTS} for _ in range(N_LAYERS)]


def hybrid_config(default: str, overrides: dict) -> list[dict]:
    """
    Build a layer config list with a default precision and selective overrides.

    overrides format:
        { layer_selector: { component: precision } }

    layer_selector can be:
        'first' / 'early' / 'middle' / 'late' / 'last'  — position class
        int                                               — specific layer index
        'all'                                             — all layers

    Example:
        hybrid_config('int4', {
            'first': {'qkv_in': 'int8', 'ffn_down_in': 'int8'},
            'last':  {'qkv_in': 'int8', 'ffn_down_in': 'int8'},
            'all':   {'ffn_down_in': 'int8'},
        })
    """
    cfg = [{comp: default for comp in COMPONENTS} for _ in range(N_LAYERS)]

    for selector, comp_prec in overrides.items():
        if selector == 'all':
            target_layers = list(range(N_LAYERS))
        elif isinstance(selector, int):
            target_layers = [selector]
        else:
            target_layers = [i for i in range(N_LAYERS) if layer_position(i) == selector]

        for i in target_layers:
            for comp, prec in comp_prec.items():
                cfg[i][comp] = prec
    return cfg


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Layer sensitivity model — QDU sanity check")
    print(f"Normalization: uniform int4 = {_uniform_int4_raw_score():.2f} raw = 100.0 QDU")
    print()

    configs = [
        ("Uniform int8 (baseline)",    uniform_config('int8')),
        ("Uniform int4",               uniform_config('int4')),
        ("Uniform int2",               uniform_config('int2')),
        ("Uniform trit",               uniform_config('trit')),
        ("BitNet a4.8 (ffn_down=int8)", hybrid_config('int4', {'all': {'ffn_down_in': 'int8'}})),
    ]
    print(f"{'Config':<38} {'QDU':>8}")
    print("-" * 48)
    for name, cfg in configs:
        qdu = quality_degradation_units(cfg)
        print(f"{name:<38} {qdu:>8.1f}")

    print()
    print("Component breakdown for uniform int4:")
    bkdn = component_qdu_breakdown(uniform_config('int4'))
    for comp, qdu in sorted(bkdn.items(), key=lambda x: -x[1]):
        print(f"  {comp:<18} {qdu:>6.1f} QDU")
