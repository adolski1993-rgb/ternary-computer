"""
Workload mix analysis: blended prefill+decode speedups for real use cases.

Total latency for one request = prefill_time + n_output_tokens × decode_time_per_token

decode_time_per_token: from Phase 3's decode_roofline at context_len ≈ n_input_tokens.
  We use n_input as the context_len for decode (conservative — context grows during
  decode, but the input tokens dominate).

prefill_time: batch=1 (one request), seq_len = n_input_tokens.
  = n_input / prefill_tps(seq_len=n_input)

Blended speedup = fp16_total_latency / ternary_total_latency.

The "prefill fraction" = prefill_time / total_latency tells how much of the
workload is prefill-dominated. When this is high, ternary's compute advantage
activates fully; when it's low, decode's memory advantage dominates.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, PHASE3_DIR)

from prefill_roofline import prefill_tps, SEQUENCE_LENGTHS
from roofline import decode_roofline
from hardware_specs import H100_SXM, TH100, TPB


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------

WORKLOADS = [
    {
        "name":        "rag_retrieval",
        "label":       "RAG Retrieval",
        "n_input":     4000,
        "n_output":    200,
        "description": "Long context (4K), short answer. ~95% prefill time.",
    },
    {
        "name":        "doc_summary",
        "label":       "Document Summary",
        "n_input":     3000,
        "n_output":    400,
        "description": "Long document summarisation. ~88% prefill.",
    },
    {
        "name":        "chat",
        "label":       "Interactive Chat",
        "n_input":     1024,
        "n_output":    512,
        "description": "Medium context, medium output. ~50% prefill.",
    },
    {
        "name":        "code_completion",
        "label":       "Code Completion",
        "n_input":     512,
        "n_output":    128,
        "description": "Short context, short completion. ~30% prefill.",
    },
    {
        "name":        "long_generation",
        "label":       "Long-Form Generation",
        "n_input":     256,
        "n_output":    3000,
        "description": "Short prompt, long output. <8% prefill.",
    },
]


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------

def request_latency(
    chip,
    wprec: str,
    aprec: str,
    n_input: int,
    n_output: int,
) -> dict:
    """
    Total latency for one request = prefill_time + n_output × decode_time.

    For prefill_tps: nearest SEQUENCE_LENGTHS bin to n_input is used.
    For decode: context = n_input (conservative — ignores context growth during decode).
    """
    # Snap n_input to nearest modeled seq_len for prefill
    L_prefill = min(SEQUENCE_LENGTHS, key=lambda x: abs(x - n_input))

    pf = prefill_tps(chip, wprec, aprec, L_prefill, batch=1)
    pf_time = n_input / pf["tokens_per_sec"]

    # Decode: generate n_output tokens, one at a time, context = n_input
    dec = decode_roofline(chip, wprec, aprec, n_input, batch=1)
    dec_time = n_output / dec["tokens_per_sec"]

    total = pf_time + dec_time
    return {
        "prefill_time_s":  pf_time,
        "decode_time_s":   dec_time,
        "total_time_s":    total,
        "prefill_frac":    pf_time / total if total > 0 else 0,
        "prefill_tps":     pf["tokens_per_sec"],
        "decode_tps":      dec["tokens_per_sec"],
        "prefill_bound":   pf["dominant_bound"],
        "decode_bound":    dec["bottleneck"],
    }


def blended_speedup(
    base_chip, base_wprec: str, base_aprec: str,
    tern_chip, tern_wprec: str, tern_aprec: str,
    n_input: int, n_output: int,
) -> dict:
    """Speedup of ternary config over baseline for a given request."""
    base = request_latency(base_chip, base_wprec, base_aprec, n_input, n_output)
    tern = request_latency(tern_chip, tern_wprec, tern_aprec, n_input, n_output)

    total_speedup    = base["total_time_s"]    / tern["total_time_s"]
    prefill_speedup  = base["prefill_time_s"]  / tern["prefill_time_s"]
    decode_speedup   = base["decode_time_s"]   / tern["decode_time_s"]

    return {
        "total_speedup":   total_speedup,
        "prefill_speedup": prefill_speedup,
        "decode_speedup":  decode_speedup,
        "prefill_frac_base": base["prefill_frac"],
        "prefill_frac_tern": tern["prefill_frac"],
        "base_total_ms":   base["total_time_s"] * 1000,
        "tern_total_ms":   tern["total_time_s"] * 1000,
    }


# ---------------------------------------------------------------------------
# Full workload comparison matrix
# ---------------------------------------------------------------------------

COMPARISONS = [
    # (label, base_chip, base_wp, base_ap, tern_chip, tern_wp, tern_ap)
    ("H100: fp16 → trit+int4",  H100_SXM, 'fp16', 'fp16', H100_SXM, 'trit', 'int4'),
    ("H100 → TH100 drop-in",    H100_SXM, 'fp16', 'fp16', TH100,    'trit', 'int4'),
    ("H100 → TPB purpose-built",H100_SXM, 'fp16', 'fp16', TPB,      'trit', 'int4'),
]


def workload_matrix() -> list[dict]:
    """Full matrix of (comparison, workload) → speedup numbers."""
    rows = []
    for comp_label, bc, bw, ba, tc, tw, ta in COMPARISONS:
        for wl in WORKLOADS:
            sp = blended_speedup(bc, bw, ba, tc, tw, ta, wl["n_input"], wl["n_output"])
            rows.append({
                "comparison":     comp_label,
                "workload":       wl["label"],
                "n_input":        wl["n_input"],
                "n_output":       wl["n_output"],
                "description":    wl["description"],
                **sp,
            })
    return rows


def workload_table(chip_label: str, bc, bw, ba, tc, tw, ta) -> str:
    """Summary table for one chip comparison across all workloads."""
    col_w = 11
    lines = [
        f"Blended speedup — {chip_label}:",
        f"{'Workload':<25} {'n_in':>6} {'n_out':>6}  {'Prefill%':>9}  "
        f"{'Total spd':>10}  {'Prefill spd':>12}  {'Decode spd':>11}",
        "-" * 82,
    ]
    for wl in WORKLOADS:
        sp = blended_speedup(bc, bw, ba, tc, tw, ta, wl["n_input"], wl["n_output"])
        lines.append(
            f"{wl['label']:<25} {wl['n_input']:>6} {wl['n_output']:>6}  "
            f"{sp['prefill_frac_base']*100:>8.0f}%  "
            f"{sp['total_speedup']:>9.2f}x  "
            f"{sp['prefill_speedup']:>11.2f}x  "
            f"{sp['decode_speedup']:>10.2f}x"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for label, bc, bw, ba, tc, tw, ta in COMPARISONS:
        print(workload_table(label, bc, bw, ba, tc, tw, ta))
        print()
