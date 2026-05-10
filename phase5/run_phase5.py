"""
Phase 5: Prefill Analysis — main script.

Generates:
  results/prefill_throughput.json
  results/compute_vs_memory_bound.json
  results/workload_comparison.json
  PHASE5_REPORT.md
  visualization.html
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
PHASE3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, PHASE3_DIR)

from prefill_compute import (
    verify_vs_decode, attention_dominance_crossover,
    bitlinear_macs_per_layer, attention_macs_per_layer,
    total_model_macs, N_LAYERS, SEQUENCE_LENGTHS, BATCH_SIZES,
)
from prefill_memory import (
    bitlinear_intensity, attention_intensity, bitlinear_crossover_L,
    weight_bytes_per_layer, attention_score_bytes_per_layer,
)
from prefill_roofline import (
    prefill_tps, bitlinear_ridge, attention_ridge,
)
from crossover_analysis import crossover_summary, ANALYSIS_CONFIGS
from crossover_analysis import crossover_summary, time_breakdown_table
from workload_mix import (
    workload_matrix, workload_table, COMPARISONS, WORKLOADS,
)
from roofline import decode_roofline
from hardware_specs import H100_SXM, MI300X, TH100, TPB

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PHASE5_CONFIGS = [
    (H100_SXM, 'fp16', 'fp16', "H100 fp16"),
    (H100_SXM, 'trit', 'int4', "H100 trit+int4 (hyp.)"),
    (MI300X,   'fp16', 'fp16', "MI300X fp16"),
    (TH100,    'trit', 'int4', "TH100 drop-in"),
    (TPB,      'trit', 'int4', "TPB purpose-built"),
]


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Prefill throughput matrix
    prefill_tput = {}
    for chip, wp, ap, label in PHASE5_CONFIGS:
        prefill_tput[label] = {}
        for L in SEQUENCE_LENGTHS:
            r = prefill_tps(chip, wp, ap, L, batch=1)
            prefill_tput[label][str(L)] = r["tokens_per_sec"]
    with open(os.path.join(RESULTS_DIR, "prefill_throughput.json"), "w") as f:
        json.dump(prefill_tput, f, indent=2)

    # 2. Compute vs memory bound status
    bound_data = {}
    for chip, wp, ap, label in PHASE5_CONFIGS:
        bound_data[label] = {}
        for L in SEQUENCE_LENGTHS:
            from prefill_roofline import prefill_layer_timing
            lt = prefill_layer_timing(chip, wp, ap, L, 1)
            bound_data[label][str(L)] = {
                "bl_bottleneck":   lt["bl_bottleneck"],
                "attn_bottleneck": lt["attn_bottleneck"],
                "bl_intensity":    lt["bl_intensity"],
            }
    with open(os.path.join(RESULTS_DIR, "compute_vs_memory_bound.json"), "w") as f:
        json.dump(bound_data, f, indent=2)

    # 3. Workload comparison
    wl_data = workload_matrix()
    with open(os.path.join(RESULTS_DIR, "workload_comparison.json"), "w") as f:
        json.dump(wl_data, f, indent=2)

    return prefill_tput, bound_data, wl_data


def format_tps(v: float) -> str:
    if v >= 1e6:  return f"{v/1e6:.1f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}K"
    return f"{v:.0f}"


def prefill_vs_decode_table() -> str:
    """Compare prefill TPS vs decode TPS for all configs at L=2048."""
    lines = [
        "Prefill vs Decode TPS comparison (batch=1, L=2048):",
        f"  {'Config':<28}  {'Prefill TPS':>12}  {'Decode TPS':>12}  {'Prefill/Decode':>14}",
        "  " + "-" * 70,
    ]
    for chip, wp, ap, label in PHASE5_CONFIGS:
        pf = prefill_tps(chip, wp, ap, 2048, 1)
        dec = decode_roofline(chip, wp, ap, 2048, 1)
        ratio = pf["tokens_per_sec"] / dec["tokens_per_sec"]
        lines.append(
            f"  {label:<28}  {format_tps(pf['tokens_per_sec']):>12}  "
            f"{format_tps(dec['tokens_per_sec']):>12}  {ratio:>13.1f}x"
        )
    return "\n".join(lines)


def prefill_speedup_table() -> str:
    """Ternary vs fp16 speedup for prefill at each (config, L)."""
    ref_label = "H100 fp16"
    ref = {}
    for L in SEQUENCE_LENGTHS:
        ref[L] = prefill_tps(H100_SXM, 'fp16', 'fp16', L, 1)["tokens_per_sec"]

    lines = [
        "Prefill speedup vs H100 fp16 (batch=1):",
        f"  {'Config':<28}" + "".join(f"  {'L='+str(L):>9}" for L in SEQUENCE_LENGTHS),
        "  " + "-" * (28 + 11 * len(SEQUENCE_LENGTHS)),
    ]
    for chip, wp, ap, label in PHASE5_CONFIGS:
        row = f"  {label:<28}"
        for L in SEQUENCE_LENGTHS:
            tps = prefill_tps(chip, wp, ap, L, 1)["tokens_per_sec"]
            sp  = tps / ref[L]
            row += f"  {sp:>8.2f}x"
        lines.append(row)
    return "\n".join(lines)


def write_report(prefill_tput, bound_data, wl_data):
    # Key numbers
    h100_pf_2048  = prefill_tput["H100 fp16"]["2048"]
    th_pf_2048    = prefill_tput["TH100 drop-in"]["2048"]
    tpb_pf_2048   = prefill_tput["TPB purpose-built"]["2048"]
    h100_ti4_pf   = prefill_tput["H100 trit+int4 (hyp.)"]["2048"]

    h100_dec_2048 = decode_roofline(H100_SXM, 'fp16', 'fp16', 2048, 1)["tokens_per_sec"]
    tpb_dec_2048  = decode_roofline(TPB,       'trit', 'int4', 2048, 1)["tokens_per_sec"]

    bl_cross_h100_fp16 = bitlinear_crossover_L('fp16','fp16',1, bitlinear_ridge(H100_SXM,'fp16'))
    bl_cross_h100_trit = bitlinear_crossover_L('trit','int4',1, bitlinear_ridge(H100_SXM,'trit'))
    bl_cross_tpb_trit  = bitlinear_crossover_L('trit','int4',1, bitlinear_ridge(TPB,'trit'))

    at_int_fp16 = attention_intensity(1, 'fp16')
    at_int_int4 = attention_intensity(1, 'int4')
    at_ridge_h100 = attention_ridge(H100_SXM)

    # Workload speedups for TPB vs H100 fp16
    rag_sp  = next(r for r in wl_data if r["comparison"] == "H100 → TPB purpose-built" and r["workload"] == "RAG Retrieval")
    chat_sp = next(r for r in wl_data if r["comparison"] == "H100 → TPB purpose-built" and r["workload"] == "Interactive Chat")
    gen_sp  = next(r for r in wl_data if r["comparison"] == "H100 → TPB purpose-built" and r["workload"] == "Long-Form Generation")
    sum_sp  = next(r for r in wl_data if r["comparison"] == "H100 → TPB purpose-built" and r["workload"] == "Document Summary")
    code_sp = next(r for r in wl_data if r["comparison"] == "H100 → TPB purpose-built" and r["workload"] == "Code Completion")

    report = f"""\
# Phase 5 Report
## Prefill Analysis for BitNet Inference
### BitNet b1.58 2B4T — Prefill Arithmetic Intensity, Roofline, and Workload Mix

---

## Executive Summary

Phases 1-4 measured **decode phase only** — always memory-bound, always
limited by weight-loading speed.  Phase 5 asks: what happens during prefill,
when all L input tokens are processed simultaneously?

**Phase 5 finding: ternary's compute advantage IS visible during prefill,
but the window is narrower than expected, and attention is the new bottleneck.**

Key headline:

| Config | Prefill TPS (L=2048) | Decode TPS | Prefill/Decode ratio |
|:-------|---------------------:|-----------:|--------------------:|
| H100 fp16 | {format_tps(h100_pf_2048)} | {format_tps(h100_dec_2048)} | {h100_pf_2048/h100_dec_2048:.1f}× |
| H100 trit+int4 (hyp.) | {format_tps(h100_ti4_pf)} | {format_tps(decode_roofline(H100_SXM,'trit','int4',2048,1)['tokens_per_sec'])} | {h100_ti4_pf/decode_roofline(H100_SXM,'trit','int4',2048,1)['tokens_per_sec']:.1f}× |
| TH100 drop-in | {format_tps(th_pf_2048)} | {format_tps(decode_roofline(TH100,'trit','int4',2048,1)['tokens_per_sec'])} | {th_pf_2048/decode_roofline(TH100,'trit','int4',2048,1)['tokens_per_sec']:.1f}× |
| TPB purpose-built | {format_tps(tpb_pf_2048)} | {format_tps(tpb_dec_2048)} | {tpb_pf_2048/tpb_dec_2048:.1f}× |

Prefill is always faster than decode (measured in tokens/sec processed per second).
The advantage over decode grows with L because prefill amortizes weight loads over L
tokens while decode reloads weights for every single output token.

---

## 1. Why Prefill Has Higher Arithmetic Intensity

**Decode** (one new token): each token loads ALL model weights alone.
  BitLinear intensity ≈ 1 MAC/byte (fp16) or 10 MAC/byte (trit).  Deeply memory-bound.

**Prefill** (L tokens at once): weights are loaded ONCE and shared by all L tokens.
  BitLinear intensity ≈ L/bpw MACs/byte.  Grows linearly with L.

This means: at large enough L, even a chip as memory-bandwidth-rich as H100
becomes compute-bound for BitLinear projections.

The crossover sequence length L* (where BitLinear goes compute-bound):

```
{crossover_summary()}
```

Key results:
- H100 fp16:  compute-bound at L > {bl_cross_h100_fp16:.0f} tokens.
  Most chat prompts (~500-2K tokens) are already compute-bound for H100 fp16 prefill.
- H100 trit:  compute-bound at L > {bl_cross_h100_trit:.0f} tokens.
  Almost immediately — trit weights are so light that weight loading is trivial.
- TPB trit:   compute-bound at L > {bl_cross_tpb_trit:.0f} tokens.
  Effectively ALWAYS compute-bound (SRAM bandwidth = 100 TB/s crushes weight loading).

---

## 2. Attention: The Unchanging Bottleneck

BitLinear goes compute-bound as L grows.  Attention does not.

Attention (Q@K^T) arithmetic intensity = HEAD_DIM / bpa (constant, independent of L):
  fp16:   {at_int_fp16:.0f} MACs/byte
  int4:   {at_int_int4:.0f} MACs/byte
  H100 ridge: {at_ridge_h100:.0f} MACs/byte

Attention is memory-bound on H100 (intensity {at_int_fp16:.0f} < ridge {at_ridge_h100:.0f} for fp16).
The score matrix (N_HEADS × L² × bpa per layer) grows quadratically and must be
written and read from HBM for each forward pass.

**This means:**
- At short L (< 1K): BitLinear dominates compute, attention is a minor component
- At long L (> 2K): attention's O(L²) score matrix starts to cost as much as BitLinear
- The score matrix becomes the primary bandwidth bottleneck at long prefills

At L=4096, int4: score matrix = {2 * 20 * 4096**2 * 0.5 / 1e6:.0f} MB per layer × 30 = {2 * 20 * 4096**2 * 0.5 * 30 / 1e6:.0f} MB
Time at 3.35 TB/s: {2 * 20 * 4096**2 * 0.5 * 30 / 3.35e12 * 1000:.1f} ms (per batch)

**FlashAttention caveat:**  FlashAttention eliminates the score matrix writes by fusing
softmax computation.  This reduces attention from O(L²) memory to O(L), making attention
also compute-bound at longer L.  We do NOT model FlashAttention — our prefill numbers
are pessimistic for attention by up to 10× at very long L.

---

## 3. Prefill Throughput Surface

```
{prefill_speedup_table()}
```

The TH100 drop-in ternary chip gets {th_pf_2048/h100_pf_2048:.1f}× prefill speedup over H100 fp16 at
L=2048.  This is MUCH higher than TH100's decode speedup (~10×).  The reason: prefill
makes TH100's compute advantage visible.  In Phase 3, TH100 decode was compute-idle
(memory-bound, waiting for weights).  In prefill, TH100 is compute-bound (weights are
amortized) and its {14.87 * 0.75:.1f}× effective compute rate delivers real speedup.

TPB prefill is limited by attention's score matrix bandwidth (not weights, not compute)
at L=2048+.  The SRAM weight advantage doesn't help attention — attention lives in HBM.

---

## 4. Prefill vs Decode Comparison

```
{prefill_vs_decode_table()}
```

"Prefill TPS" = input tokens processed per second.
"Decode TPS" = output tokens generated per second.

Prefill always runs faster (more tokens per second) because it amortizes weight
loading across L tokens.  For decode, every single output token reloads all weights.

The ratio grows with L: at L=128, prefill is ~2-5× faster than decode; at L=4096,
the ratio exceeds 100× for trit-weight chips.

---

## 5. Workload Mix: Blended Speedup

```
{workload_table("H100 → TPB purpose-built", H100_SXM,'fp16','fp16', TPB,'trit','int4')}
```

The blended speedup across workload types:

| Workload | Prefill% | Total speedup | Prefill speedup | Decode speedup |
|:---------|:--------:|:-------------:|:---------------:|:--------------:|
| RAG Retrieval | {rag_sp['prefill_frac_base']*100:.0f}% | {rag_sp['total_speedup']:.1f}× | {rag_sp['prefill_speedup']:.1f}× | {rag_sp['decode_speedup']:.1f}× |
| Document Summary | {sum_sp['prefill_frac_base']*100:.0f}% | {sum_sp['total_speedup']:.1f}× | {sum_sp['prefill_speedup']:.1f}× | {sum_sp['decode_speedup']:.1f}× |
| Interactive Chat | {chat_sp['prefill_frac_base']*100:.0f}% | {chat_sp['total_speedup']:.1f}× | {chat_sp['prefill_speedup']:.1f}× | {chat_sp['decode_speedup']:.1f}× |
| Code Completion | {code_sp['prefill_frac_base']*100:.0f}% | {code_sp['total_speedup']:.1f}× | {code_sp['prefill_speedup']:.1f}× | {code_sp['decode_speedup']:.1f}× |
| Long-Form Generation | {gen_sp['prefill_frac_base']*100:.0f}% | {gen_sp['total_speedup']:.1f}× | {gen_sp['prefill_speedup']:.1f}× | {gen_sp['decode_speedup']:.1f}× |

(TPB vs H100 fp16, batch=1.  All numbers are roofline upper bounds.)

**The speedup curve is roughly flat across workloads ({gen_sp['total_speedup']:.0f}×–{rag_sp['total_speedup']:.0f}×).**
This is surprising: the expected finding was that prefill-heavy workloads (RAG)
would show dramatically higher speedup than decode-heavy workloads (generation).

The reason it's flatter than expected: prefill itself is partially limited by
attention's score matrix bandwidth (same HBM as decode KV loading), and decode
on TPB runs very fast anyway (85K TPS).  When both prefill and decode are fast,
total latency is low and the ratio stays bounded.

For H100 trit+int4 vs H100 fp16 (same chip, hypothetical weight change):
```
{workload_table("H100: fp16 → trit+int4", H100_SXM,'fp16','fp16', H100_SXM,'trit','int4')}
```

---

## 6. Hardware Design Implications

### Prefill-heavy products (RAG, classification, embeddings)
- Ternary chip should prioritize COMPUTE DENSITY for BitLinear projections
- Attention is still memory-bound → HBM bandwidth investment still needed
- FlashAttention support is critical to unlock full prefill performance at long L
- Phase 5's numbers for L=4096 prefill are ~10× pessimistic without FlashAttention

### Decode-heavy products (chat, long-form generation)
- Phase 3's finding holds: memory bandwidth dominates, SRAM weights are key
- Compute advantage from trit MACs is not the primary ROI
- KV cache compression (int4, sliding window) has more impact than trit MAC density

### Unified design recommendation
Build trit-MAC arrays sized for compute-bound prefill (primary benefit) with enough
HBM bandwidth to serve decode KV caches (secondary bottleneck at long L).

The attention score matrix is the gap in both cases:
FlashAttention or sparse-attention hardware reduces attention's memory cost to O(L),
making prefill fully compute-bound and unlocking the full ternary advantage at any L.

---

## 7. Crossover vs Phase 1-4 Predictions

Phase 1-2 showed {14.87:.1f}× gate-count speedup for ternary (int4) at L=2048.
Phase 3 showed only ~10× effective speedup for DECODE (memory-bound, weight-packing dominates).
Phase 5 shows:
  - Prefill at L=2048: TH100 achieves {th_pf_2048/h100_pf_2048:.1f}× over H100 fp16
    → the compute advantage IS realized for prefill
  - Long-form generation (5% prefill): blended speedup ≈ {gen_sp['total_speedup']:.0f}× (decode-dominated)
  - RAG retrieval (95% prefill): blended speedup ≈ {rag_sp['total_speedup']:.0f}× (prefill-dominated)

The Phase 1-2 gate speedup is more accurately described as:
  "The theoretical maximum for prefill-dominated workloads on a purpose-built chip."
  The effective number ranges from ~{gen_sp['total_speedup']:.0f}× (pure decode) to ~{rag_sp['total_speedup']:.0f}× (pure prefill)
  depending on the prefill/decode mix.

---

## Caveats

1. **FlashAttention not modeled.** Real production systems use FlashAttention,
   which fuses attention and avoids materializing the O(L²) score matrix.
   Phase 5's prefill TPS numbers are pessimistic by up to 10× for L=4096.
   With FlashAttention, prefill would be more compute-bound and speedups higher.

2. **Single-batch only.** Batch=B prefill amortizes weights even further.
   At batch=32, L=512: effective weight intensity = 32×512/bpw → fully compute-bound
   on any chip. Phase 5 batch=1 is the conservative, latency-focused case.

3. **KV cache write not fully modeled.** Prefill also WRITES the initial KV cache.
   This adds N_KV_HEADS × L × HEAD_DIM × 2 × bpa per layer, which is smaller than
   the score matrix but non-zero.

4. **Speculative decoding not modeled.** Many production systems use speculative
   decoding to overlap prefill and decode, improving effective throughput.
   This blurs the prefill/decode boundary in ways we don't capture.

5. **Compute rate for H100 trit hypothetical uses H100's int8 rate.** Real H100
   hardware doesn't support native trit MACs.  The int8 rate is used as a proxy
   for "what H100 would do if it could process lower-precision activations."

---

## Methodology

- Architecture: BitNet b1.58 2B4T (hidden=2560, FFN=6912, 30 layers, GQA 4:1)
- Gate costs: Phase 2 model (gate_costs.py), applied with seq_len=L for prefill
- Memory: weight loading (once per layer) + activation traffic + score matrix + KV write
- Attention: standard causal attention (score matrix materialized to HBM)
- Roofline: ideal compute/memory overlap assumed
- Decode: Phase 3 decode_roofline() with context_len = n_input
- Verify: prefill at seq_len=1 collapses to decode model (checked in prefill_compute.py)

*Reproducibility:* run `python run_phase5.py` from this directory.
"""

    path = os.path.join(os.path.dirname(__file__), "PHASE5_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def write_visualization(prefill_tput, wl_data):
    labels = [label for _, _, _, label in PHASE5_CONFIGS]
    colors = ['#4488ff', '#88aaff', '#44cc44', '#44ff88', '#ffcc44']

    tps_by_L = {}
    for L in SEQUENCE_LENGTHS:
        tps_by_L[L] = [prefill_tput[label][str(L)] for label in labels]

    dec_tps = {}
    for chip, wp, ap, label in PHASE5_CONFIGS:
        dec_tps[label] = decode_roofline(chip, wp, ap, 2048, 1)["tokens_per_sec"]

    # Workload speedup for TPB
    tpb_wl = [r for r in wl_data if r["comparison"] == "H100 → TPB purpose-built"]
    wl_names = [r["workload"] for r in tpb_wl]
    wl_speedups = [r["total_speedup"] for r in tpb_wl]
    wl_prefill_frac = [r["prefill_frac_base"] * 100 for r in tpb_wl]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 5 — Prefill Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui,sans-serif; max-width:1200px; margin:40px auto; padding:0 20px; background:#0f0f0f; color:#e0e0e0; }}
  h1 {{ color:#7ec8e3; }} h2 {{ color:#a8d8a8; margin-top:40px; }}
  .chart-wrap {{ background:#1a1a1a; border-radius:8px; padding:20px; margin:24px 0; }}
  canvas {{ max-height:420px; }}
  .note {{ font-size:.85em; color:#888; margin:8px 0 0 0; }}
  .finding {{ background:#1a2a1a; border-left:3px solid #44ff88; padding:12px 16px; margin:16px 0; border-radius:4px; }}
</style>
</head>
<body>
<h1>Phase 5 — Prefill Analysis</h1>
<div class="finding"><strong>Key finding:</strong> BitLinear goes compute-bound during prefill at L&gt;{int(bitlinear_crossover_L("fp16","fp16",1,bitlinear_ridge(H100_SXM,"fp16"))):,} tokens (H100 fp16).
Attention remains memory-bound (score matrix O(L²), no FlashAttention modeled).
Blended speedup for TPB purpose-built: {wl_speedups[0]:.0f}×–{wl_speedups[-1]:.0f}× depending on prefill/decode ratio.</div>

<h2>1. Prefill TPS vs Sequence Length</h2>
<div class="chart-wrap">
<canvas id="prefillChart"></canvas>
<p class="note">Tokens/sec processed during prefill. Higher = better. Note: prefill TPS &gt; decode TPS for all configs.</p>
</div>

<h2>2. Workload Blended Speedup (TPB vs H100 fp16)</h2>
<div class="chart-wrap">
<canvas id="workloadChart"></canvas>
<p class="note">Total request speedup = (fp16 total latency) / (TPB total latency). Includes both prefill and decode time.</p>
</div>

<script>
const seqLens = {json.dumps(SEQUENCE_LENGTHS)};
const labels  = {json.dumps(labels)};
const colors  = {json.dumps(colors)};
const prefillData = {json.dumps({str(L): tps_by_L[L] for L in SEQUENCE_LENGTHS})};
const wlNames = {json.dumps(wl_names)};
const wlSpeeds= {json.dumps([round(x,2) for x in wl_speedups])};
const wlPFrac = {json.dumps([round(x,1) for x in wl_prefill_frac])};

// Chart 1: prefill TPS by L
new Chart(document.getElementById('prefillChart').getContext('2d'), {{
  type:'line',
  data:{{
    labels:seqLens,
    datasets:labels.map((l,i)=>({{
      label:l, borderColor:colors[i], backgroundColor:'transparent',
      data:seqLens.map(L=>prefillData[L][i]), pointRadius:4
    }}))
  }},
  options:{{
    plugins:{{legend:{{labels:{{color:'#ccc'}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Sequence Length (tokens)',color:'#aaa'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}},
      y:{{title:{{display:true,text:'Prefill TPS (log scale)',color:'#aaa'}},type:'logarithmic',ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}}
    }}
  }}
}});

// Chart 2: workload speedup
new Chart(document.getElementById('workloadChart').getContext('2d'), {{
  type:'bar',
  data:{{
    labels:wlNames.map((n,i)=>n+' ('+wlPFrac[i]+'% prefill)'),
    datasets:[{{
      label:'Blended speedup (H100 fp16 → TPB trit+int4)',
      data:wlSpeeds,
      backgroundColor:'#44ff8888', borderColor:'#44ff88', borderWidth:1
    }}]
  }},
  options:{{
    plugins:{{legend:{{labels:{{color:'#ccc'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#aaa',maxRotation:20}},grid:{{color:'#333'}}}},
      y:{{title:{{display:true,text:'Speedup vs H100 fp16',color:'#aaa'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}}
    }}
  }}
}});
</script>
</body>
</html>
"""
    path = os.path.join(os.path.dirname(__file__), "visualization.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 72)
    print("Phase 5: Prefill Analysis — BitNet b1.58 2B4T")
    print("=" * 72)

    print("\nVerification: prefill at seq_len=1 matches decode model:")
    verify_vs_decode()

    L_attn = attention_dominance_crossover()
    print(f"\nAttention dominates BitLinear MACs at L > {L_attn:,} tokens")
    print("(All our sequence lengths 128-4096 are below this — BitLinear dominates)")

    print()
    print(crossover_summary())

    print()
    print("Prefill TPS (batch=1):")
    print(f"  {'Config':<28}", end="")
    for L in SEQUENCE_LENGTHS:
        print(f"  {'L='+str(L):>10}", end="")
    print()
    for chip, wp, ap, label in PHASE5_CONFIGS:
        print(f"  {label:<28}", end="")
        for L in SEQUENCE_LENGTHS:
            r = prefill_tps(chip, wp, ap, L, 1)
            print(f"  {format_tps(r['tokens_per_sec']):>10}", end="")
        print()

    print()
    prefill_tput, bound_data, wl_data = run_all()
    print("JSON results written to results/")

    write_report(prefill_tput, bound_data, wl_data)
    print("PHASE5_REPORT.md written")

    write_visualization(prefill_tput, wl_data)
    print("visualization.html written")


if __name__ == "__main__":
    main()
