"""
BitNet b1.58 2B4T architecture parameters.

Source: microsoft/bitnet-b1.58-2B-4T config.json on HuggingFace.

These are CONSTANTS describing the public model. We don't load weights;
we just use the shapes to drive gate-count calculations on representative
random data.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BitNetConfig:
    """BitNet b1.58 2B4T model configuration."""
    hidden_size:         int = 2560      # d_model
    intermediate_size:   int = 6912      # FFN inner dim (~2.7x expansion)
    num_hidden_layers:   int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5         # GQA: 4 query heads share each KV head
    head_dim:            int = 128       # 2560 / 20
    max_position:        int = 4096
    vocab_size:          int = 128_256
    rms_norm_eps:        float = 1e-5
    rope_theta:          float = 500_000.0

    # Quantization
    weight_states:       int = 3         # ternary: -1, 0, +1
    activation_bits:     int = 8         # int8 activations
    fp_compare_bits:     int = 16        # we compare against fp16 binary

    # Derived dimensions
    @property
    def kv_dim(self) -> int:
        """Total dimension of K and V combined (GQA-reduced)."""
        return self.num_key_value_heads * self.head_dim    # 5 * 128 = 640

    @property
    def qkv_out_dim(self) -> int:
        """Output dim of the fused QKV projection."""
        return self.hidden_size + 2 * self.kv_dim  # Q + K + V = 2560 + 640 + 640


CONFIG = BitNetConfig()


# Sequence lengths we'll benchmark at
SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------------------
# Sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    c = CONFIG
    print(f"BitNet b1.58 2B4T configuration:")
    print(f"  hidden_size:         {c.hidden_size}")
    print(f"  intermediate_size:   {c.intermediate_size}")
    print(f"  num_hidden_layers:   {c.num_hidden_layers}")
    print(f"  num_attention_heads: {c.num_attention_heads}")
    print(f"  num_key_value_heads: {c.num_key_value_heads}")
    print(f"  head_dim:            {c.head_dim}")
    print(f"  qkv_out_dim:         {c.qkv_out_dim}")
    print(f"  kv_dim:              {c.kv_dim}")
    print()

    # Compute parameter counts to sanity-check we're modeling the right thing
    qkv_params  = c.hidden_size * c.qkv_out_dim
    o_params    = c.hidden_size * c.hidden_size
    ffn_up      = c.hidden_size * c.intermediate_size  # gate + up
    ffn_down    = c.intermediate_size * c.hidden_size
    per_layer = qkv_params + o_params + 2 * ffn_up + ffn_down

    embed = c.vocab_size * c.hidden_size  # tied with lm_head
    total = c.num_hidden_layers * per_layer + embed
    print(f"Estimated parameter counts:")
    print(f"  per layer (matmul weights): {per_layer:>15,}")
    print(f"    QKV projection:           {qkv_params:>15,}")
    print(f"    Attention out:            {o_params:>15,}")
    print(f"    FFN up + gate:            {2*ffn_up:>15,}")
    print(f"    FFN down:                 {ffn_down:>15,}")
    print(f"  All {c.num_hidden_layers} layers:               "
          f"{c.num_hidden_layers * per_layer:>15,}")
    print(f"  Embedding (tied):           {embed:>15,}")
    print(f"  Total (approx):             {total:>15,}  "
          f"(~{total/1e9:.2f}B)")
