"""
Published hardware specifications for AI accelerators used in Phase 3 roofline analysis.

Sources are cited per constant. All values are non-sparse unless noted.
"Hypothetical ternary" chips (TH100, TPB) are parametric projections, not measurements.

Last verified: 2024-2026.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChipSpec:
    name: str
    short: str                  # used in tables and chart labels

    # Compute
    tflops_fp16: float          # peak FP16 TFLOPS, non-sparse
    tops_int8: float            # peak INT8 TOPS, non-sparse
    tops_int4: float            # peak INT4 TOPS (estimated if not published)

    # Memory
    hbm_bandwidth_tbs: float    # off-chip (HBM/DRAM) bandwidth in TB/s; 0 = SRAM-only
    hbm_gb: float               # off-chip memory in GB; 0 = SRAM-only
    sram_mb: float              # on-chip SRAM (L2 / dedicated weight store) in MB
    sram_bandwidth_tbs: float   # on-chip SRAM bandwidth in TB/s; 0 = negligible

    # Physical
    tdp_w: float                # TDP in watts
    die_area_mm2: float         # die area; 0 = not published

    # Flags
    sram_is_weight_store: bool  # True if SRAM large enough to hold model weights

    notes: str = ""


# ---------------------------------------------------------------------------
# NVIDIA H100 SXM5 (80 GB HBM3)
#   Source: https://www.nvidia.com/en-us/data-center/h100/
#           NVIDIA H100 Tensor Core GPU Datasheet (March 2023)
#   FP16 non-sparse: 989.5 TFLOPS  (search result confirms ~989 TFLOPS dense)
#   FP16 sparse:   1,979 TFLOPS
#   INT8 non-sparse: 1,979 TOPS
#   INT8 sparse:   3,958 TOPS
#   HBM3 bandwidth: 3.35 TB/s
#   HBM3 capacity: 80 GB (5 stacks × 16 GB)
#   L2 cache: 50 MB on-chip
#   TDP: 700 W
#   Die area: ~814 mm² (GH100 full die, TSMC 4N)
# ---------------------------------------------------------------------------
H100_SXM = ChipSpec(
    name="NVIDIA H100 SXM5 (80 GB)",
    short="H100",
    tflops_fp16=989.5,
    tops_int8=1979.0,
    tops_int4=3958.0,       # not published; estimated 2× INT8
    hbm_bandwidth_tbs=3.35,
    hbm_gb=80.0,
    sram_mb=50.0,           # L2 cache
    sram_bandwidth_tbs=0.0,
    tdp_w=700.0,
    die_area_mm2=814.0,
    sram_is_weight_store=False,
    notes="Source: nvidia.com/en-us/data-center/h100 — non-sparse numbers used throughout",
)

# ---------------------------------------------------------------------------
# AMD Instinct MI300X (192 GB HBM3)
#   Source: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
#           AMD MI300X data sheet PDF (amd.com/content/dam/amd/...)
#   FP16 non-sparse: 1,307.4 TFLOPS
#   FP16 sparse:     2,614.9 TFLOPS
#   HBM3 bandwidth: 5.3 TB/s
#   HBM3 capacity: 192 GB
#   TDP: 750 W
#   Die area: multi-die chiplet design; GCD ~227 mm² × 3 + HBM
# ---------------------------------------------------------------------------
MI300X = ChipSpec(
    name="AMD Instinct MI300X (192 GB)",
    short="MI300X",
    tflops_fp16=1307.4,
    tops_int8=2614.9,
    tops_int4=5229.8,       # not published; estimated 2× INT8
    hbm_bandwidth_tbs=5.3,
    hbm_gb=192.0,
    sram_mb=256.0,          # AMD Infinity Cache (estimated; not published per chip)
    sram_bandwidth_tbs=0.0,
    tdp_w=750.0,
    die_area_mm2=0.0,       # multi-chiplet; total die area not published as single number
    sram_is_weight_store=False,
    notes="Source: amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
)

# ---------------------------------------------------------------------------
# Groq LPU (GroqChip 1 / TSP)
#   Source: https://groq.com/lpu-architecture
#           Groq blog "Inside the LPU" (groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
#   INT8: 750 TOPS per chip
#   FP16: 188 TFLOPS per chip
#   On-chip SRAM: 230 MB per chip  (weight-primary storage — NOT a cache)
#   SRAM bandwidth: 80 TB/s per chip
#   NO external DRAM on-chip (weights must fit in SRAM or streamed via interconnect)
#   Process: 14 nm
#   Note: BitNet 2B4T at trit precision = ~417 MB; does NOT fit in 1 chip (230 MB).
#         Groq inference typically chains multiple chips for large models.
# ---------------------------------------------------------------------------
GROQ_LPU = ChipSpec(
    name="Groq LPU (GroqChip 1)",
    short="Groq LPU",
    tflops_fp16=188.0,
    tops_int8=750.0,
    tops_int4=1500.0,       # not published; estimated 2× INT8
    hbm_bandwidth_tbs=0.0,  # no off-chip DRAM; relies entirely on SRAM
    hbm_gb=0.0,
    sram_mb=230.0,
    sram_bandwidth_tbs=80.0,
    tdp_w=300.0,            # estimated; not published
    die_area_mm2=0.0,       # not published
    sram_is_weight_store=True,   # SRAM is the weight store by design
    notes=(
        "Source: groq.com/lpu-architecture, groq.com/blog/inside-the-lpu-deconstructing-groq-speed. "
        "No external DRAM; weights must fit in 230 MB SRAM. BitNet 2B4T (trit) = 417 MB — "
        "requires 2 chips or model parallelism."
    ),
)

# ---------------------------------------------------------------------------
# Cerebras WSE-3 (wafer-scale engine)
#   Source: https://www.cerebras.ai/chip
#           Cerebras press release (cerebras.ai/press-release/cerebras-announces-third-generation...)
#           Tom's Hardware: "Cerebras launches 900,000-core 125 PetaFLOPS wafer-scale processor"
#   Peak perf: 125 PFLOPS = 125,000 TFLOPS (mixed precision / AI-optimised)
#   On-chip SRAM: 44 GB
#   On-chip bandwidth: 21 PB/s = 21,000 TB/s
#   Transistors: 4 trillion (TSMC 5nm)
#   NO external DRAM per wafer; CS-3 system can connect to external I/O nodes
#   BitNet 2B4T at trit: 417 MB << 44 GB → ENTIRE model fits comfortably on-chip
# ---------------------------------------------------------------------------
CEREBRAS_WSE3 = ChipSpec(
    name="Cerebras WSE-3",
    short="WSE-3",
    tflops_fp16=125_000.0,    # 125 PFLOPS
    tops_int8=250_000.0,      # estimated 2× FP16
    tops_int4=500_000.0,      # estimated 4× FP16
    hbm_bandwidth_tbs=0.0,    # no off-chip HBM per chip
    hbm_gb=0.0,
    sram_mb=44_000.0,         # 44 GB on-chip SRAM
    sram_bandwidth_tbs=21_000.0,  # 21 PB/s
    tdp_w=23_000.0,           # CS-3 system TDP; chip itself not published separately
    die_area_mm2=46_225.0,    # full 300mm wafer = 46,225 mm² (pi × 150² ≈ 70,686 mm² usable ~46k after edge)
    sram_is_weight_store=True,
    notes=(
        "Source: cerebras.ai/chip, cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine. "
        "44 GB SRAM can hold entire BitNet 2B4T model (trit: 417 MB) many times over. "
        "21 PB/s on-chip bandwidth makes weight loading effectively free."
    ),
)

# ---------------------------------------------------------------------------
# Hypothetical Ternary Chip: "Drop-In Replacement" (TH100)
#
#   Same die area as H100 (814 mm²), same HBM3 bandwidth (3.35 TB/s),
#   same 80 GB HBM3. BUT: replaces fp16 FMA units with trit-int4 MAC units.
#
#   Compute capacity derived from Phase 2 gate-count analysis:
#     fp16 MAC gate cost: ~230 gates  (Phase 1/2 methodology)
#     trit-int4 MAC gate cost: ~12.33 gates for BitLinear (Phase 2)
#                              ~36 gates for attention (int4×int4, no trit benefit)
#   Same die area → proportionally more MAC units.
#   Effective fp16-equivalent TFLOPS (BitLinear): 989.5 × (230/12.33) = 18,424 TFLOPS
#   Effective fp16-equivalent TFLOPS (attention): 989.5 × (230/36)  =  6,316 TFLOPS
#   Phase 2 overall speedup at int4/L=2048: 14.87× → effective 14,714 TFLOPS (blended)
#   We use component-specific multipliers for accuracy.
#
#   CAVEAT: Ternary gates run at ~75% clock rate vs binary (3-level logic overhead).
#   Net compute advantage after clock adjustment: ~14.87 × 0.75 = 11.15× fp16-equivalent.
#   We report both adjusted and unadjusted to bracket the real-silicon range.
# ---------------------------------------------------------------------------
PHASE2_SPEEDUP_INT4_BITLINEAR  = 18.63   # from Phase 2 component table, L=2048
PHASE2_SPEEDUP_INT4_ATTENTION  =  6.39   # from Phase 2 component table, L=2048
PHASE2_SPEEDUP_INT4_OVERALL    = 14.87   # from Phase 2 overall speedup at L=2048
TERNARY_CLOCK_PENALTY          =  0.75   # 3-level logic runs ~25% slower per cycle

TH100 = ChipSpec(
    name="Hypothetical Ternary Chip (H100 area + HBM3)",
    short="TH100 (drop-in)",
    # Unadjusted (theoretical gate-count advantage only):
    tflops_fp16=989.5 * PHASE2_SPEEDUP_INT4_OVERALL,           # 14,714 effective
    tops_int8=989.5 * PHASE2_SPEEDUP_INT4_OVERALL * 2,         # proxy
    tops_int4=989.5 * PHASE2_SPEEDUP_INT4_OVERALL * 2,
    hbm_bandwidth_tbs=3.35,
    hbm_gb=80.0,
    sram_mb=50.0,
    sram_bandwidth_tbs=0.0,
    tdp_w=700.0,
    die_area_mm2=814.0,
    sram_is_weight_store=False,
    notes=(
        "PROJECTION: same area/BW as H100. Compute = Phase 2 speedup × H100 compute. "
        f"Phase 2 int4 speedup: BitLinear {PHASE2_SPEEDUP_INT4_BITLINEAR}×, "
        f"Attention {PHASE2_SPEEDUP_INT4_ATTENTION}×, Overall {PHASE2_SPEEDUP_INT4_OVERALL}×. "
        f"After ~25% clock penalty for 3-level logic: effective ~{PHASE2_SPEEDUP_INT4_OVERALL * TERNARY_CLOCK_PENALTY:.1f}× net compute vs H100."
    ),
)

# ---------------------------------------------------------------------------
# Hypothetical Ternary Chip: "Purpose-Built" (TPB)
#
#   Designed around ternary's small weight footprint.
#   BitNet 2B4T trit weights = ~417 MB total model → fits in 512 MB on-chip SRAM.
#   On-chip SRAM holds ALL model weights; HBM3 is used ONLY for KV cache and activations.
#
#   Design rationale:
#   - 512 MB SRAM at TSMC 5nm is feasible (Cerebras WSE-3 has 44 GB).
#     A chip with 512 MB SRAM dedicated weight store + smaller compute die fits in ~400 mm².
#   - SRAM bandwidth: 100 TB/s (achievable for large on-chip SRAM arrays at 5nm).
#   - HBM3: 24 GB @ 3.35 TB/s — enough for KV cache at up to L=32K + activations.
#
#   This transforms the bottleneck from "weight loading" to "KV cache loading,"
#   since weights are now accessed at SRAM speeds (essentially free vs HBM).
# ---------------------------------------------------------------------------
TPB = ChipSpec(
    name="Hypothetical Ternary Chip (Purpose-Built, weights in SRAM)",
    short="TPB (purpose-built)",
    tflops_fp16=989.5 * PHASE2_SPEEDUP_INT4_OVERALL,
    tops_int8=989.5 * PHASE2_SPEEDUP_INT4_OVERALL * 2,
    tops_int4=989.5 * PHASE2_SPEEDUP_INT4_OVERALL * 2,
    hbm_bandwidth_tbs=3.35,   # for KV cache and activations only
    hbm_gb=24.0,
    sram_mb=512.0,             # dedicated weight store
    sram_bandwidth_tbs=100.0,  # on-chip SRAM bandwidth to weight arrays
    tdp_w=450.0,               # estimated: smaller HBM footprint, same compute density
    die_area_mm2=400.0,        # estimated: less die area needed (smaller HBM controller)
    sram_is_weight_store=True,
    notes=(
        "PROJECTION: purpose-built ternary design. "
        "512 MB on-chip SRAM at 100 TB/s holds all BitNet 2B4T trit weights (417 MB). "
        "Weight loading cost is ~SRAM-speed (negligible vs HBM). "
        "Decode bottleneck shifts from weights to KV cache (HBM3 at 3.35 TB/s). "
        "HBM3 24 GB supports KV cache up to ~L=180K at int8, ~360K at int4."
    ),
)

# Ordered list for iteration
ALL_CHIPS: list[ChipSpec] = [H100_SXM, MI300X, GROQ_LPU, CEREBRAS_WSE3, TH100, TPB]
REFERENCE_CHIP = H100_SXM     # comparison baseline
TERNARY_CHIPS  = [TH100, TPB]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{'Chip':<40} {'FP16 TFLOPS':>14} {'HBM BW (TB/s)':>15} {'SRAM (MB)':>11}")
    print("-" * 82)
    for c in ALL_CHIPS:
        print(f"{c.short:<40} {c.tflops_fp16:>14,.1f} {c.hbm_bandwidth_tbs:>15.2f} {c.sram_mb:>11,.0f}")
    print()
    print(f"Phase 2 speedup at int4: BitLinear {PHASE2_SPEEDUP_INT4_BITLINEAR}× | "
          f"Attention {PHASE2_SPEEDUP_INT4_ATTENTION}× | Overall {PHASE2_SPEEDUP_INT4_OVERALL}×")
    print(f"Clock penalty for 3-level logic: {1-TERNARY_CLOCK_PENALTY:.0%}")
