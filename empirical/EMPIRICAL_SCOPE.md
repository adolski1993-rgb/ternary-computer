# Empirical Scope: Weight Distribution Validation

## What We're Testing

Phases 1-5 assumed BitNet b1.58 2B4T weights follow a uniform trit distribution
({-1, 0, +1} each at probability 1/3). This assumption came from BitNet's absmean
quantization producing approximately uniform distributions *in expectation*.

Real trained models may deviate. This empirical test downloads the actual
published weights and measures the real distribution.

## Why It Matters

The Phase 1-2 gate cost formula:
    per_pair = 1 + p_nonzero × (ACC_ADD + NEG / 2)

is sensitive to p_nonzero. If the actual nonzero fraction is 55% instead of 67%,
the BitLinear speedup changes by roughly ±10%. If it's within 5% of 67%, the
Phase 1-5 analysis stands as published.

## What We're NOT Testing

- Activation quantization accuracy (Phase 2 sweep)
- Hardware bandwidth numbers (Phase 3)
- Roofline model assumptions
- KV cache behavior
- Prefill vs decode ratios

## Method

Source: `microsoft/bitnet-b1.58-2B-4T-bf16` (BF16 master weights on HuggingFace)
Quantization: apply absmean quantization per BitNet paper
    scale = mean(|W|)   [per weight matrix, scalar]
    trits = clip(round(W / scale), -1, +1)
Measurement: count -1, 0, +1 per layer and component

The BF16 master weights are the "true" weights before ternary quantization.
Applying absmean to them recovers the ternary representation used at inference time.

## Success Criteria

If |empirical p_nonzero - 2/3| / (2/3) < 5%: assumption validated, no update needed.
If > 5%: report corrected speedup numbers.
If > 15%: flag as material and update Phase 1-5 conclusions.
