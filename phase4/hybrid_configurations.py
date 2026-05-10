"""
Hybrid activation precision configurations for Phase 4 Pareto analysis.

Each configuration specifies the activation precision per component per layer.
All configurations use TRIT WEIGHTS throughout (BitNet b1.58 standard).

Configuration design rationale
-------------------------------
We span the space from pure int8 (best quality) to pure trit (best compute),
including several hybrid points motivated by published literature or hardware
design logic.

The key literature anchor is BitNet a4.8 (arXiv:2411.04965), which shows:
  - FFN down inputs: int8 (sensitive due to massive outliers)
  - Everything else: int4

We extend this with additional configs exploring:
  - Position-sensitive hybrids (int8 for first/last blocks)
  - Component-selective hybrids (int8 for specific component types)
  - KV-cache-optimized configs (keep QKV at int4 to shrink KV cache)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from layer_sensitivity_model import (
    COMPONENTS, N_LAYERS, uniform_config, hybrid_config,
    quality_degradation_units, layer_position,
)

# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

def make_configs() -> list[dict]:
    """
    Return list of configuration dicts, each with keys:
      name        : short label for tables
      label       : full description
      layer_cfgs  : list[dict] — N_LAYERS × {component: precision}
      rationale   : string explaining the design choice
    """
    configs = []

    # ------- Uniform baselines -------

    configs.append({
        "name": "int8-uniform",
        "label": "Uniform int8 (reference)",
        "layer_cfgs": uniform_config('int8'),
        "rationale": "BitNet b1.58 default activation precision. No degradation vs training.",
    })

    configs.append({
        "name": "int4-uniform",
        "label": "Uniform int4 (Phase 2 rec.)",
        "layer_cfgs": uniform_config('int4'),
        "rationale": "Phase 2 recommendation: best compute/quality tradeoff for uniform schemes.",
    })

    configs.append({
        "name": "int2-uniform",
        "label": "Uniform int2",
        "layer_cfgs": uniform_config('int2'),
        "rationale": "Aggressive uniform compression; likely too much quality loss.",
    })

    configs.append({
        "name": "trit-uniform",
        "label": "Uniform trit (1.58b)",
        "layer_cfgs": uniform_config('trit'),
        "rationale": "Maximum compression uniform; purely theoretical endpoint.",
    })

    # ------- Literature-motivated hybrids -------

    # BitNet a4.8 precisely: ffn_down inputs at int8, all others int4.
    # This is the PUBLISHED optimal configuration from Microsoft Research.
    # Source: Wang et al., arXiv:2411.04965, November 2024.
    configs.append({
        "name": "a4.8-canonical",
        "label": "BitNet a4.8 (ffn_down=int8)",
        "layer_cfgs": hybrid_config('int4', {'all': {'ffn_down_in': 'int8'}}),
        "rationale": (
            "Exact BitNet a4.8 configuration. FFN down inputs kept at int8 "
            "due to massive activation outliers. All other activations int4. "
            "[PUBLISHED] — Wang et al., arXiv:2411.04965."
        ),
    })

    # Extended a4.8: also keep attn_out at int8 (uncertain from paper; we test it).
    configs.append({
        "name": "a4.8-extended",
        "label": "a4.8-extended (ffn_down+attn_out=int8)",
        "layer_cfgs": hybrid_config('int4', {'all': {'ffn_down_in': 'int8',
                                                      'attn_out_in': 'int8'}}),
        "rationale": (
            "Extends a4.8 by also keeping attention output inputs at int8. "
            "SmoothQuant notes 'normal outliers' here. More conservative. [PROJECTED]"
        ),
    })

    # Position-sensitive: first/last two blocks entirely at int8, rest int4.
    # Motivated by GPTQ's U-shaped sensitivity curve.
    configs.append({
        "name": "position-sensitive",
        "label": "Position-sensitive (first/last=int8)",
        "layer_cfgs": hybrid_config('int4', {
            'first': {comp: 'int8' for comp in COMPONENTS},
            'last':  {comp: 'int8' for comp in COMPONENTS},
        }),
        "rationale": (
            "All components in first 2 and last 2 blocks at int8; "
            "middle 26 blocks at int4. Motivated by GPTQ's observation that "
            "errors in early/late layers cascade through the full stack. "
            "Source: Frantar et al., ICLR 2023. [PROJECTED application to activations]"
        ),
    })

    # Conservative hybrid: both position-sensitive AND a4.8 component selection.
    configs.append({
        "name": "conservative",
        "label": "Conservative (pos+comp hybrid)",
        "layer_cfgs": hybrid_config('int4', {
            'first': {comp: 'int8' for comp in COMPONENTS},
            'last':  {comp: 'int8' for comp in COMPONENTS},
            'all':   {'ffn_down_in': 'int8'},
        }),
        "rationale": (
            "First/last blocks fully int8 (GPTQ motivation) + "
            "ffn_down_in at int8 throughout (BitNet a4.8). "
            "Double-conservative. [PROJECTED]"
        ),
    })

    # KV-optimized: QKV stays int4 (keeps KV cache int4 = small);
    # ffn_down at int8. Best bandwidth tradeoff on TPB/TH100.
    configs.append({
        "name": "kv-optimized",
        "label": "KV-optimized (ffn_down=int8, QKV=int4)",
        "layer_cfgs": hybrid_config('int4', {'all': {'ffn_down_in': 'int8'}}),
        "rationale": (
            "Same gate counts as a4.8-canonical. KV cache is int4 (QKV stays int4). "
            "Explicitly designed to minimize KV bandwidth while maintaining quality. "
            "This is structurally identical to a4.8-canonical — we include it "
            "to highlight that a4.8 ALSO happens to be KV-cache optimal."
        ),
    })

    # Aggressive: all trit except ffn_down (int8), matching BitNet a4.8 spirit
    # but pushing all other components to trit.
    configs.append({
        "name": "aggressive-trit",
        "label": "Aggressive (all-trit except ffn_down)",
        "layer_cfgs": hybrid_config('trit', {'all': {'ffn_down_in': 'int8'}}),
        "rationale": (
            "All activations at trit (maximum compression) except ffn_down_in at int8. "
            "Tests whether the literature's 'keep ffn_down at int8' heuristic "
            "also applies when everything else goes to trit. [PROJECTED]"
        ),
    })

    # Minimal hybrid: only ffn_down at int8 for first/last blocks; int4 elsewhere.
    configs.append({
        "name": "minimal",
        "label": "Minimal hybrid (ffn_down first/last=int8)",
        "layer_cfgs": hybrid_config('int4', {
            'first': {'ffn_down_in': 'int8'},
            'last':  {'ffn_down_in': 'int8'},
        }),
        "rationale": (
            "Only the most sensitive component (ffn_down_in) in the most sensitive "
            "positions (first/last 2 blocks) stays at int8. "
            "Cheapest hybrid supporting both position and component sensitivity. [PROJECTED]"
        ),
    })

    return configs


# The canonical list used throughout Phase 4
CONFIGS = make_configs()

# Mapping for quick lookup
CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}


# ---------------------------------------------------------------------------
# Verification: check uniform configs match Phase 2 totals
# ---------------------------------------------------------------------------

def describe_config(cfg: dict) -> str:
    """One-line summary of precision mix in a model config."""
    counts = {}
    for layer in cfg["layer_cfgs"]:
        for prec in layer.values():
            counts[prec] = counts.get(prec, 0) + 1
    total = sum(counts.values())
    parts = [f"{prec}: {n}/{total} ({100*n/total:.0f}%)" for prec, n in sorted(counts.items())]
    return "  |  ".join(parts)


if __name__ == "__main__":
    print("Phase 4 Hybrid Configurations")
    print(f"  {N_LAYERS} layers × {len(COMPONENTS)} components = {N_LAYERS*len(COMPONENTS)} activation decisions per config")
    print()
    for cfg in CONFIGS:
        qdu = quality_degradation_units(cfg["layer_cfgs"])
        print(f"  {cfg['name']:<28}  QDU={qdu:>6.1f}  {describe_config(cfg)}")
