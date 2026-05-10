"""
Head-to-head benchmark: binary vs ternary, counting gate operations.

For fair range comparison:
  N trits cover range +/- (3^N - 1)/2
  M bits cover range +/- 2^(M-1)
  So pick M such that 2^(M-1) ~= 3^N / 2  ->  M ~= N * log2(3) ~= 1.585 N

Pairs we'll use:
  9 trits  (range +/- 9841)   vs  15 bits  (range +/- 16384)
  18 trits (range +/- ~193e6) vs  29 bits  (range +/- 268e6)

We measure on FIVE workloads:
  1. Pure addition          (expected: ~tie)
  2. Subtraction            (expected: ternary advantage - free NEG)
  3. Multiplication         (expected: ternary big advantage - skip zeros)
  4. Comparison             (expected: ternary advantage)
  5. Neural-net dot product (expected: ternary HUGE advantage with trit weights)

Each result is normalized to "gates per operation".
"""

import json
import random
from binary_ops import (
    COUNTER, int_to_bits, bits_to_int,
    b_add, b_subtract, b_multiply, b_compare,
)
from ternary_ops import (
    t_add, t_subtract, t_multiply, t_compare, t_negate, t_NEG,
)
from trit import int_to_trits, trits_to_int


# Width pairs giving comparable numerical range
PAIRS = [
    ("small",  9, 15),    # 9 trits ≈ 15 bits
    ("medium", 18, 29),   # 18 trits ≈ 29 bits
    ("large",  27, 43),   # 27 trits ≈ 43 bits
]


def benchmark_addition(n_trit, n_bit, samples=500):
    """Measure gates per addition."""
    random.seed(42)
    t_max = (3**n_trit - 1) // 4   # stay well within range
    b_max = 2**(n_bit - 2)
    rng = min(t_max, b_max)

    COUNTER.reset()
    for _ in range(samples):
        x = random.randint(-rng, rng)
        y = random.randint(-rng, rng)
        a = int_to_trits(x, n_trit)
        b = int_to_trits(y, n_trit)
        t_add(a, b, n_trit)
    t_gates = COUNTER.ternary_gates / samples

    COUNTER.reset()
    random.seed(42)
    for _ in range(samples):
        x = random.randint(-rng, rng)
        y = random.randint(-rng, rng)
        a = int_to_bits(x, n_bit)
        b = int_to_bits(y, n_bit)
        b_add(a, b, n_bit)
    b_gates = COUNTER.binary_gates / samples

    return b_gates, t_gates


def benchmark_subtraction(n_trit, n_bit, samples=500):
    random.seed(42)
    t_max = (3**n_trit - 1) // 4
    b_max = 2**(n_bit - 2)
    rng = min(t_max, b_max)

    COUNTER.reset()
    for _ in range(samples):
        x = random.randint(-rng, rng); y = random.randint(-rng, rng)
        a = int_to_trits(x, n_trit); b = int_to_trits(y, n_trit)
        t_subtract(a, b, n_trit)
    t_gates = COUNTER.ternary_gates / samples

    COUNTER.reset()
    random.seed(42)
    for _ in range(samples):
        x = random.randint(-rng, rng); y = random.randint(-rng, rng)
        a = int_to_bits(x, n_bit); b = int_to_bits(y, n_bit)
        b_subtract(a, b, n_bit)
    b_gates = COUNTER.binary_gates / samples

    return b_gates, t_gates


def benchmark_multiplication(n_trit, n_bit, samples=200):
    random.seed(42)
    # Keep operands small enough that product fits in width
    t_max = int((3**n_trit) ** 0.5) // 2
    b_max = int((2**n_bit) ** 0.5) // 2
    rng = min(t_max, b_max)

    COUNTER.reset()
    for _ in range(samples):
        x = random.randint(-rng, rng); y = random.randint(-rng, rng)
        a = int_to_trits(x, n_trit); b = int_to_trits(y, n_trit)
        t_multiply(a, b, n_trit)
    t_gates = COUNTER.ternary_gates / samples

    COUNTER.reset()
    random.seed(42)
    for _ in range(samples):
        x = random.randint(-rng, rng); y = random.randint(-rng, rng)
        a = int_to_bits(x, n_bit); b = int_to_bits(y, n_bit)
        b_multiply(a, b, n_bit)
    b_gates = COUNTER.binary_gates / samples

    return b_gates, t_gates


def benchmark_compare(n_trit, n_bit, samples=500):
    random.seed(42)
    t_max = (3**n_trit - 1) // 4
    b_max = 2**(n_bit - 2)
    rng = min(t_max, b_max)

    COUNTER.reset()
    for _ in range(samples):
        x = random.randint(-rng, rng); y = random.randint(-rng, rng)
        a = int_to_trits(x, n_trit); b = int_to_trits(y, n_trit)
        t_compare(a, b, n_trit)
    t_gates = COUNTER.ternary_gates / samples

    COUNTER.reset()
    random.seed(42)
    for _ in range(samples):
        x = random.randint(-rng, rng); y = random.randint(-rng, rng)
        a = int_to_bits(x, n_bit); b = int_to_bits(y, n_bit)
        b_compare(a, b, n_bit)
    b_gates = COUNTER.binary_gates / samples

    return b_gates, t_gates


def benchmark_neural_dot_product(vector_len=64, n_trit=18, n_bit=16):
    """Neural network inner product: y = sum(weights[i] * activations[i]).

    For BINARY: weights are 8-bit ints, activations are 8-bit ints,
                multiplication is full integer multiply per element.
                We use n_bit-wide accumulator.

    For TERNARY: weights are TRITS (-1, 0, +1), activations are 8-bit ints
                 stored in a wider trit format. The 'multiply' is just a
                 3-way mux: skip / add / subtract.
                 This is the BitNet b1.58 architecture.

    This is the modern killer app for ternary.
    """
    random.seed(42)
    # Ternary weights: -1, 0, +1 with equal probability
    weights_ternary = [random.choice([-1, 0, 1]) for _ in range(vector_len)]
    # Equivalent binary 8-bit weights (forced into +/- 1, 0 for fairness)
    weights_binary  = list(weights_ternary)
    # Activations: random 8-bit signed ints
    activations = [random.randint(-127, 127) for _ in range(vector_len)]

    # ---- Binary: full multiplies ----
    COUNTER.reset()
    acc_b = int_to_bits(0, n_bit)
    for w, a_val in zip(weights_binary, activations):
        w_bits = int_to_bits(w, n_bit)
        a_bits = int_to_bits(a_val, n_bit)
        prod = b_multiply(w_bits, a_bits, n_bit)
        acc_b = b_add(acc_b, prod, n_bit)
    b_gates = COUNTER.binary_gates

    # ---- Ternary: trit-mux MAC ----
    COUNTER.reset()
    acc_t = int_to_trits(0, n_trit)
    for w, a_val in zip(weights_ternary, activations):
        # The "multiplication" in ternary is just inspecting the trit weight.
        # In hardware: 1 gate to dispatch (the trit IS the control signal).
        if w == 0:
            COUNTER.ternary_gates += 1  # decode skip
            continue
        a_trits = int_to_trits(a_val, n_trit)
        if w == -1:
            a_trits = t_negate(a_trits)  # width NEG gates
        # accumulate: one ripple add
        acc_t = t_add(acc_t, a_trits, n_trit)
    t_gates = COUNTER.ternary_gates

    # Verify both arrived at the same answer
    expected = sum(w * a for w, a in zip(weights_ternary, activations))
    actual_b = bits_to_int(acc_b)
    actual_t = trits_to_int(acc_t)
    assert actual_b == expected, f"binary wrong: {actual_b} vs {expected}"
    assert actual_t == expected, f"ternary wrong: {actual_t} vs {expected}"

    return b_gates, t_gates


# ---------------------------------------------------------------------------
# Run all benchmarks and report
# ---------------------------------------------------------------------------

def run_all():
    results = {}

    print("=" * 78)
    print(f"{'Benchmark':<28} {'Width':<14} {'Binary':>10} {'Ternary':>10} {'Speedup':>10}")
    print("-" * 78)

    for label, n_trit, n_bit in PAIRS:
        results[label] = {}
        width_str = f"{n_trit}t / {n_bit}b"

        for name, fn in [
            ("Addition",       benchmark_addition),
            ("Subtraction",    benchmark_subtraction),
            ("Multiplication", benchmark_multiplication),
            ("Comparison",     benchmark_compare),
        ]:
            b, t = fn(n_trit, n_bit)
            speedup = b / t
            results[label][name] = {"binary": b, "ternary": t, "speedup": speedup}
            print(f"{name:<28} {width_str:<14} {b:>10.1f} {t:>10.1f} {speedup:>9.2f}x")
        print()

    # Neural network dot product (fixed-size)
    print("-" * 78)
    print("Neural network dot product (BitNet-style ternary weights):")
    for length in [16, 64, 256, 1024]:
        b, t = benchmark_neural_dot_product(vector_len=length, n_trit=18, n_bit=16)
        speedup = b / t
        results.setdefault("nn_dot", {})[length] = {
            "binary": b, "ternary": t, "speedup": speedup,
        }
        print(f"  vector len {length:>5}:        "
              f"{b:>10} {t:>10} {speedup:>9.2f}x")

    print("=" * 78)
    return results


if __name__ == "__main__":
    results = run_all()
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")
