"""
Run full Phase 1 benchmark: one decoder layer at multiple sequence lengths.

Produces:
  results/per_seqlen.json  — total layer cost at each seq length
  results/per_component.json — per-component breakdown
  Console output summarizing findings
"""

import json
import os
from architecture import CONFIG, SEQUENCE_LENGTHS
from layer import run_binary_layer, run_ternary_layer


def main():
    os.makedirs("results", exist_ok=True)

    per_seqlen = {}
    per_component = {}

    print("=" * 78)
    print("BitNet b1.58 2B4T — One Decoder Layer — Gate Count Comparison")
    print(f"  hidden={CONFIG.hidden_size}, intermediate={CONFIG.intermediate_size}, "
          f"heads={CONFIG.num_attention_heads}, kv_heads={CONFIG.num_key_value_heads}")
    print("=" * 78)

    print(f"\n{'Seq Length':>10} {'Binary (gates)':>22} {'Ternary (gates)':>22} {'Speedup':>10}")
    print("-" * 78)

    for L in SEQUENCE_LENGTHS:
        b = run_binary_layer(L)
        t = run_ternary_layer(L)

        per_seqlen[L] = {
            "binary_total": b["TOTAL"],
            "ternary_total": t["TOTAL"],
            "speedup": b["TOTAL"] / t["TOTAL"],
        }
        per_component[L] = {"binary": b, "ternary": t}

        print(f"{L:>10} {b['TOTAL']:>22,} {t['TOTAL']:>22,} "
              f"{b['TOTAL']/t['TOTAL']:>9.2f}x")

    # Save raw data
    with open("results/per_seqlen.json", "w") as f:
        json.dump(per_seqlen, f, indent=2)
    with open("results/per_component.json", "w") as f:
        json.dump(per_component, f, indent=2)

    # Per-component analysis at L=2048 (a realistic inference setting)
    print()
    print("=" * 78)
    print(f"Per-Component Breakdown at seq_len=2048")
    print("=" * 78)
    L = 2048
    b = per_component[L]["binary"]
    t = per_component[L]["ternary"]
    print(f"\n{'Component':<22} {'Binary':>16} {'Ternary':>16} {'Ratio':>8} {'% of total (b)':>15}")
    print("-" * 80)
    components = [k for k in b if k != "TOTAL"]
    for k in components:
        ratio = b[k] / t[k] if t[k] > 0 else float('inf')
        pct = 100 * b[k] / b["TOTAL"]
        print(f"{k:<22} {b[k]:>16,} {t[k]:>16,} {ratio:>7.2f}x {pct:>14.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':<22} {b['TOTAL']:>16,} {t['TOTAL']:>16,} "
          f"{b['TOTAL']/t['TOTAL']:>7.2f}x")

    # Identify ternary's dominant wins and ties
    print()
    print("=" * 78)
    print("Where ternary wins by sub-component (any seq length):")
    print("=" * 78)
    L_ref = 2048
    b = per_component[L_ref]["binary"]
    t = per_component[L_ref]["ternary"]
    # Sort components by ratio
    sorted_components = sorted(
        [(k, b[k]/t[k]) for k in b if k != "TOTAL"],
        key=lambda x: -x[1]
    )
    print(f"\n{'Rank':<6} {'Component':<22} {'Speedup':>10} {'% of binary cost':>18}")
    for rank, (name, ratio) in enumerate(sorted_components, 1):
        pct = 100 * b[name] / b["TOTAL"]
        marker = " <-- TRIT WEIGHTS" if ratio > 5 else ""
        print(f"{rank:<6} {name:<22} {ratio:>9.2f}x {pct:>17.1f}%{marker}")

    # Sequence-length scaling
    print()
    print("=" * 78)
    print("Speedup as sequence length grows:")
    print("=" * 78)
    print(f"\n{'Seq Length':>10} {'Speedup':>10} {'Δ':>10}")
    prev = None
    for L in SEQUENCE_LENGTHS:
        s = per_seqlen[L]["speedup"]
        delta = f"{s - prev:+.2f}x" if prev is not None else ""
        print(f"{L:>10} {s:>9.2f}x {delta:>10}")
        prev = s

    print()
    print("Results written to results/per_seqlen.json and results/per_component.json")
    return per_seqlen, per_component


if __name__ == "__main__":
    main()
