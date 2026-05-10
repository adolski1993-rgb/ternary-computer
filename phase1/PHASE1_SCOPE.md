# Phase 1: End-to-End Gate-Count for One BitNet Decoder Layer

## The question this phase answers

> For a real BitNet b1.58 model running real inference, what is the
> end-to-end gate-operation cost of one decoder layer on native ternary
> hardware vs. fp16 binary hardware, and where does the bottleneck sit?

This is a question nobody has answered in public. Microsoft proved the
*algorithm* works (BitNet b1.58 2B4T, 4T tokens, 2024). The C++ inference
library (bitnet.cpp) shipped a 6.25× speedup on commodity CPUs. But neither
gives a clean architecture-level number for what *purpose-built* ternary
hardware would deliver.

We're going to produce that number for one layer. Honestly, with full
methodology, with the simulator we already built doing the gate counting.

## What we'll prove (and what we won't)

### What we'll prove
1. A concrete gate-count ratio (binary ÷ ternary) for one full BitNet
   decoder layer at typical inference settings.
2. A breakdown by sub-component: attention QKV projection, attention
   scores, attention output projection, FFN up-projection, FFN down-
   projection, normalization, RoPE.
3. Identification of the dominant cost in each path. (Hypothesis: it's
   matmul in both, but the *ratio* of matmul-to-everything-else differs
   sharply between binary and ternary.)
4. A scaling projection: how the gap changes as sequence length grows
   from 128 to 4096 tokens.

### What we won't prove
- Wall-clock latency. We measure gate operations, not nanoseconds. Real
  silicon converts gates to ns based on technology node, fan-out, wire
  delay, etc. We'd need a circuit simulator for that.
- Memory bandwidth limits. (That's Phase 3.)
- Accuracy. We trust BitNet b1.58 2B4T's published quality numbers.
- Training cost. Inference only.
- Anything about non-BitLinear layers (embeddings, lm_head). They're
  unchanged between binary and ternary models.

## The model: BitNet b1.58 2B4T

Real, public, downloadable. Architecture:
```
hidden_size:         2560
intermediate_size:   6912    (FFN expansion: 2.7×)
num_hidden_layers:   30
num_attention_heads: 20
num_key_value_heads: 5       (GQA: 4× compression on K/V)
head_dim:            128     (= 2560 / 20)
max_position:        4096
vocab_size:          128,256
activation:          ReLU²
norm:                subln (RMSNorm variant)
position encoding:   RoPE
biases:              none (BitNet removes all biases)
```

Quantization scheme (per the technical report):
- Weights: ternary {-1, 0, +1} via absmean quantization
- Activations: 8-bit signed integers via per-token absmax quantization

So one BitLinear matmul looks like: `int8_activations @ trit_weights`,
producing an int32-ish accumulator that's then rescaled.

## Methodology

### The fairness contract

Both implementations will:
1. Process the same activation tensor (same shape, same values)
2. Produce numerically equivalent output (within quantization error)
3. Use the same algorithms (no carry-lookahead on one side and ripple-carry
   on the other; no Wallace-tree multipliers on one side and shift-add on
   the other)
4. Count primitive gate operations consistently:
   - Binary: AND, OR, XOR, NOT each = 1 gate
   - Ternary: NEG, MIN, MAX, SUM, CARRY each = 1 gate
5. Operate on numerically-equivalent ranges:
   - Binary: int16 activations (since BitNet uses int8 activations but
     accumulators need wider range), fp16 for the comparison case
   - Ternary: appropriately-sized trit vectors

### What we're modeling

For each of the 7 sub-components of one decoder layer, we implement two
versions:
- `binary_*` — fp16 compute path representing what a standard accelerator
  would do
- `ternary_*` — trit-weight, int8-activation path representing what
  hypothetical native ternary hardware would do

For each, we count gates. We use realistic shapes:
- Batch size 1 (single-user inference, the most relevant case)
- Sequence lengths: 128, 512, 1024, 2048, 4096 tokens
- One layer (we'll multiply by 30 to get full-model numbers)

### Why one layer is enough

All 30 layers are architecturally identical. Per-layer numbers multiply
linearly to full-model numbers (with embedding + lm_head bracketing them,
which we explicitly mark as "unchanged between binary and ternary").

This isn't a shortcut, it's the right unit of analysis. Saying "the model
has 30 layers and each costs X gates" is more informative than saying "the
model costs 30X gates total" because it makes the per-layer breakdown
inspectable.

## The seven sub-components we'll count

For each sub-component, the binary baseline uses fp16 arithmetic. The
ternary version uses int8 × trit MACs.

| Component | Operation | Binary cost driver | Ternary cost driver |
|-----------|-----------|--------------------|--------------------|
| 1. RMSNorm input | Normalize activations | fp16 add/mul/sqrt | int8 normalization (fixed-point sqrt) |
| 2. QKV projection | `[L, 2560] @ [2560, 2560+512+512]` | fp16 matmul | trit matmul (3-way mux MAC) |
| 3. RoPE | Rotate Q, K | fp16 sin/cos/mul | int8 rotation tables |
| 4. Attention | `softmax(Q @ K^T / √d) @ V` | fp16 matmul + softmax | int8 matmul (Q·K NOT trit-weighted because both are activations) + softmax |
| 5. Attention output proj | `[L, 2560] @ [2560, 2560]` | fp16 matmul | trit matmul |
| 6. FFN up + gate | `[L, 2560] @ [2560, 6912]` ×2 | fp16 matmul ×2 | trit matmul ×2 |
| 7. FFN down | `[L, 6912] @ [6912, 2560]` | fp16 matmul | trit matmul |

Note that **attention scores (Q @ K^T) are NOT a trit operation** — both
operands are activations, so it's int8 × int8 in the ternary case, fp16
× fp16 in the binary case. This is a critical detail many casual analyses
miss: ternary advantage applies only to operations where one operand is a
weight.

The trit-weighted matmuls (steps 2, 5, 6, 7) are where ternary dominates.
Steps 1, 3, 4 are roughly equivalent or only modestly different.

## The deliverable

A self-contained directory `phase1/` containing:

```
phase1/
├── PHASE1_SCOPE.md           ← this file
├── architecture.py           ← BitNet config + shapes (no model logic)
├── components/
│   ├── rmsnorm.py            ← binary + ternary versions, gate-counted
│   ├── matmul.py             ← the workhorse: trit @ int8 vs fp16 @ fp16
│   ├── attention.py          ← scaled-dot-product attention
│   ├── rope.py               ← rotary position embedding
│   ├── ffn.py                ← FFN block (up, gate, down)
│   └── layer.py              ← assembles full decoder layer
├── run_layer.py              ← runs one layer, gate-counted, saves results
├── analyze.py                ← processes results, generates breakdown table
├── results/
│   ├── per_seqlen.json       ← gates per layer at each seq length
│   └── per_component.json    ← gates broken down by sub-component
└── REPORT.md                 ← human-readable findings, the actual deliverable
```

## What the report will contain

1. **Executive summary** — one paragraph, with the headline number.
2. **Methodology** — exactly what we counted and why.
3. **Per-component breakdown** — table showing gates for each of the 7
   components, binary vs ternary, with ratios.
4. **Sequence-length scaling** — how the gap changes from L=128 to L=4096.
5. **Where ternary wins, where it doesn't** — explicit identification of
   sub-components where the advantage is small or absent.
6. **Caveats** — gate-count fidelity, optimizations we didn't model,
   real-silicon adjustments needed.
7. **Implications for hardware** — what this suggests about the design
   space (compute-bound vs memory-bound, optimal silicon allocation).
8. **Next steps** — Phase 2 (activation precision) and Phase 3 (memory).

## What this is NOT meant to claim

- Not a paper. A paper requires more rigor: cycle-accurate simulation,
  multiple model sizes, comparison against published accelerator numbers,
  peer review. This is the *foundation* such a paper would build on.
- Not a product spec. Building real hardware requires power/area/wire
  delay modeling we explicitly skip.
- Not a benchmark for software inference. bitnet.cpp already exists and
  is faster on existing hardware. Our numbers are about *hypothetical*
  hardware optimized for the workload.

## Success criteria for Phase 1

We declare success if:
1. All 7 sub-components are implemented in both binary and ternary
2. End-to-end one-layer numbers are produced for at least 3 sequence lengths
3. Sub-component breakdown isolates where the wins come from
4. The report is honest about what we modeled vs. what we approximated
5. Results are reproducible: someone with the repo can re-run `run_layer.py`
   and get the same numbers

If the result is that ternary wins by 5×, we report that.
If it's 1.5×, we report that.
If something we expected to be a win is a wash, we report that.

This is exploratory engineering analysis, not advocacy.

## Time estimate

For Claude Code working on this end-to-end:
- Architecture + matmul implementation: 2-3 hours of focused work
- Full layer assembly + test: 2-3 hours
- Running benchmarks + analysis: 1 hour
- Writing the report: 1-2 hours

Total: a focused weekend.

## Why this matters (one more time, for the report)

If our headline number is something like "5× fewer gates per token, with
the gap growing at long sequences," that's a real, defensible claim that
would interest:
- Hardware startups designing inference chips (Etched, MatX, others)
- Cloud providers planning datacenter capex (Databricks knows this domain)
- Academic compilers/architecture groups (publishable at MLSys workshops)
- Anyone doing on-device LLM inference (battery life is the constraint)

The gate-count number, even rough, anchors the conversation. Without it,
"BitNet might be good for hardware" is a vibe. With it, the conversation
becomes "given X% gate reduction and Y% area savings, is the silicon
investment justified?" That's a question architects can actually engage
with.
