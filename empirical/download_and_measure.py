"""
Empirical validation: measure actual trit distribution in BitNet b1.58 2B4T.

Downloads microsoft/bitnet-b1.58-2B-4T-bf16 (BF16 master weights, ~5 GB),
applies the same absmean quantization used by BitNet at inference time,
and counts the resulting {-1, 0, +1} distribution per layer and component.

Model cache location (in order of precedence):
  1. TERNARY_MODEL_CACHE environment variable
  2. ~/.cache/ternary  (default)
Override example:  TERNARY_MODEL_CACHE=/data/models python download_and_measure.py

Weights are NOT saved — only the computed statistics (~90 KB of JSON).

Memory: tensors are loaded one at a time from a binary-mapped file.
Peak RAM = max single tensor (~0.65 GB embedding). Fine with 16 GB.

No torch required: BF16->float32 is done via numpy bit manipulation.
Safetensors file format is parsed directly (avoids BF16 dtype issue).
"""

import os, re, json, struct, time
from pathlib import Path

# Resolve cache directory — portable, no hardcoded paths.
_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "ternary")
_CACHE = os.environ.get("TERNARY_MODEL_CACHE", _DEFAULT_CACHE)

# Set HuggingFace cache dirs BEFORE importing hf_hub
os.environ["HF_HOME"]               = _CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_CACHE, "hub")

from huggingface_hub import snapshot_download
import numpy as np

MODEL_ID    = "microsoft/bitnet-b1.58-2B-4T-bf16"
LOCAL_DIR   = os.path.join(_CACHE, "bitnet-bf16")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Safetensors binary reader (no torch, handles BF16 natively)
#
# File layout:
#   bytes [0:8]      uint64 LE — N = length of JSON header
#   bytes [8:8+N]    UTF-8 JSON — {"tensor_name": {"dtype":..., "shape":...,
#                                   "data_offsets": [begin, end]}, ...}
#   bytes [8+N:...]  raw tensor data; offsets are relative to this start
#
# BF16 -> float32: BF16 = float32 with last 16 mantissa bits zeroed.
#   uint16_bits << 16, reinterpret as float32.
# ---------------------------------------------------------------------------

NUMPY_DTYPE = {
    "F32": np.float32, "F16": np.float16,
    "I8":  np.int8,    "U8":  np.uint8,
    "I32": np.int32,   "I64": np.int64,
    "BF16": None,      # handled specially
}


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Parse safetensors header. Returns (header_dict, data_base_offset)."""
    with open(path, "rb") as f:
        n = struct.unpack_from("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw), 8 + n


def load_tensor_f32(path: str, header: dict, data_base: int, key: str) -> np.ndarray:
    """Load one tensor from a safetensors file, returning float32 array."""
    meta       = header[key]
    dtype_str  = meta["dtype"]
    shape      = meta["shape"]
    beg, end   = meta["data_offsets"]

    with open(path, "rb") as f:
        f.seek(data_base + beg)
        raw = f.read(end - beg)

    if dtype_str == "BF16":
        # BF16 stored as big-endian uint16 pairs; shift left 16 bits → float32
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)

    dt = NUMPY_DTYPE.get(dtype_str)
    if dt is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return np.frombuffer(raw, dtype=dt).astype(np.float32).reshape(shape)


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

COMPONENT_MAP = {
    "q_proj":   "qkv",  "k_proj":    "qkv",  "v_proj":  "qkv",
    "qkv_proj": "qkv",  "o_proj":    "attn_out",
    "out_proj": "attn_out",
    "gate_proj":"ffn_gate", "up_proj": "ffn_up", "down_proj":"ffn_down",
}
NOT_BITLINEAR = {"embed_tokens","lm_head","norm","bias","inv_freq","position","rotary"}


def classify(name: str) -> tuple[str | None, int | None]:
    """Return (component, layer_idx) or (None, None) if not a BitLinear weight."""
    if not name.endswith(".weight"):
        return None, None
    if any(s in name for s in NOT_BITLINEAR):
        return None, None
    m = re.search(r'layers?\.(\d+)\b', name)
    if not m:
        return None, None
    layer_idx = int(m.group(1))
    for fragment, comp in COMPONENT_MAP.items():
        if fragment in name:
            return comp, layer_idx
    return None, None


# ---------------------------------------------------------------------------
# Absmean quantization statistics
# ---------------------------------------------------------------------------

def quantize_stats(W: np.ndarray) -> dict:
    """
    Apply BitNet absmean quantization and return trit distribution.
        scale = mean(|W|)
        trits = clip(round(W / scale), -1, +1)
    """
    scale = float(np.abs(W).mean())
    if scale < 1e-8:
        n = W.size
        return {"n_neg":0,"n_zero":n,"n_pos":0,"n_total":n,
                "p_neg":0.,"p_zero":1.,"p_pos":0.,"p_nonzero":0.,
                "scale":0.,"shape":list(W.shape)}

    trits   = np.clip(np.round(W / scale), -1, 1).astype(np.int8)
    n_neg   = int((trits == -1).sum())
    n_zero  = int((trits ==  0).sum())
    n_pos   = int((trits ==  1).sum())
    n_total = trits.size
    return {
        "n_neg":    n_neg,  "n_zero":   n_zero,  "n_pos":  n_pos,
        "n_total":  n_total,
        "p_neg":    round(n_neg   / n_total, 6),
        "p_zero":   round(n_zero  / n_total, 6),
        "p_pos":    round(n_pos   / n_total, 6),
        "p_nonzero":round((n_neg + n_pos) / n_total, 6),
        "scale":    round(scale, 8),
        "shape":    list(W.shape),
    }


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------

def measure_model(model_path: str) -> tuple[list[dict], dict]:
    sf_files = sorted(Path(model_path).glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors in {model_path}")

    print(f"Found {len(sf_files)} safetensors file(s).")
    records, n_bl = [], 0
    t0 = time.time()

    for sf in sf_files:
        print(f"  Parsing header: {sf.name} ...", flush=True)
        header, data_base = read_safetensors_header(str(sf))
        keys = [k for k in header if k != "__metadata__"]
        print(f"  {len(keys)} tensors in file. Measuring BitLinear weights ...",
              flush=True)

        for key in keys:
            comp, layer_idx = classify(key)
            if comp is None:
                continue

            dtype = header[key]["dtype"]
            shape = header[key]["shape"]
            n_elem = 1
            for d in shape:
                n_elem *= d
            size_mb = n_elem * 2 / 1e6   # BF16 = 2 bytes

            W = load_tensor_f32(str(sf), header, data_base, key)
            stats = quantize_stats(W)
            del W   # release immediately

            stats["name"]      = key
            stats["layer_idx"] = layer_idx
            stats["component"] = comp
            records.append(stats)
            n_bl += 1

            if n_bl % 10 == 0:
                elapsed = time.time() - t0
                print(f"    [{n_bl:3d} BitLinear tensors, {elapsed:.0f}s] "
                      f"last: {key.split('.')[-2]} shape={shape} "
                      f"p_nz={stats['p_nonzero']:.3f}", flush=True)

    elapsed = time.time() - t0
    file_stats = {
        "n_bitlinear_tensors": n_bl,
        "elapsed_seconds":     round(elapsed, 1),
        "model_id":            MODEL_ID,
    }
    print(f"  Measured {n_bl} BitLinear tensors in {elapsed:.1f}s")
    return records, file_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Phase 1-5 assumption: p_zero=1/3={1/3:.4f}, p_nonzero=2/3={2/3:.4f}")
    print()

    # Download model (files already present are skipped automatically)
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
    print("(~5 GB download. Re-runs skip already-downloaded files.)")
    model_path = snapshot_download(
        MODEL_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.bin","*.msgpack","flax_model*","tf_model*",
                         "rust_model*","coreml*","*.ot"],
    )
    print(f"Model path: {model_path}")
    print()

    # Measure
    print("Measuring trit distribution ...")
    records, fstats = measure_model(model_path)

    # Save raw data
    out = {
        "metadata":   fstats,
        "assumption": {"p_neg":1/3,"p_zero":1/3,"p_pos":1/3,"p_nonzero":2/3},
        "per_tensor": records,
    }
    out_path = RESULTS_DIR / "per_layer_distribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(records)} tensor records -> {out_path}")

    # Quick summary
    all_pnz = [r["p_nonzero"] for r in records]
    mean_pnz = sum(all_pnz) / len(all_pnz)
    print(f"\nQuick summary:")
    print(f"  Mean p_nonzero (empirical): {mean_pnz:.4f}")
    print(f"  Assumed p_nonzero:          {2/3:.4f}")
    print(f"  Delta:                      {mean_pnz - 2/3:+.4f}  "
          f"({100*(mean_pnz-2/3)/(2/3):+.1f}%)")
    print(f"  Min: {min(all_pnz):.4f}  Max: {max(all_pnz):.4f}")
    print(f"\nRun analysis.py to compute corrected Phase 1-5 numbers.")


if __name__ == "__main__":
    main()
