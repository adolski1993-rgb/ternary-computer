"""
Binary arithmetic with gate counting, structurally parallel to arithmetic.py.

We use two's complement signed binary, ripple-carry adders, and shift-and-add
multiplication. Each primitive gate increments a counter so we can compare
fairly against the ternary implementation.
"""

from dataclasses import dataclass

Bit = int  # 0 or 1


@dataclass
class GateCounter:
    """Tracks gate operations. Reset between measurements."""
    binary_gates: int = 0
    ternary_gates: int = 0

    def reset(self):
        self.binary_gates = 0
        self.ternary_gates = 0


COUNTER = GateCounter()


# ---------------------------------------------------------------------------
# Primitive binary gates - each call counts as one gate operation
# ---------------------------------------------------------------------------

def b_AND(a: Bit, b: Bit) -> Bit:
    COUNTER.binary_gates += 1
    return a & b

def b_OR(a: Bit, b: Bit) -> Bit:
    COUNTER.binary_gates += 1
    return a | b

def b_XOR(a: Bit, b: Bit) -> Bit:
    COUNTER.binary_gates += 1
    return a ^ b

def b_NOT(a: Bit) -> Bit:
    COUNTER.binary_gates += 1
    return 1 - a


# ---------------------------------------------------------------------------
# Half- and full-adders (binary)
# ---------------------------------------------------------------------------

def b_half_adder(a: Bit, b: Bit) -> tuple[Bit, Bit]:
    """sum = a XOR b, carry = a AND b. 2 gates."""
    return b_XOR(a, b), b_AND(a, b)


def b_full_adder(a: Bit, b: Bit, cin: Bit) -> tuple[Bit, Bit]:
    """Standard implementation: 5 gates per full adder."""
    s1, c1 = b_half_adder(a, b)         # 2 gates
    s2, c2 = b_half_adder(s1, cin)      # 2 gates
    return s2, b_OR(c1, c2)             # 1 gate


# ---------------------------------------------------------------------------
# Two's complement: signed integer <-> bit list, LSB-first
# ---------------------------------------------------------------------------

def int_to_bits(n: int, width: int) -> list[Bit]:
    """Two's complement encoding, LSB first."""
    if n < 0:
        n = (1 << width) + n
    return [(n >> i) & 1 for i in range(width)]


def bits_to_int(bits: list[Bit]) -> int:
    """Decode two's complement, LSB first."""
    n = sum(b << i for i, b in enumerate(bits))
    if bits[-1] == 1:
        n -= (1 << len(bits))
    return n


# ---------------------------------------------------------------------------
# Multi-bit operations
# ---------------------------------------------------------------------------

def b_add(a: list[Bit], b: list[Bit], width: int) -> list[Bit]:
    """Ripple-carry addition. width full adders = 5*width gates."""
    result = [0] * width
    carry: Bit = 0
    for i in range(width):
        s, carry = b_full_adder(a[i], b[i], carry)
        result[i] = s
    return result


def b_negate(a: list[Bit], width: int) -> list[Bit]:
    """Two's complement: invert all bits, then add 1.
    width NOT gates + a full ripple-carry add of 1.
    """
    inverted = [b_NOT(bit) for bit in a]
    one = [0] * width
    one[0] = 1
    return b_add(inverted, one, width)


def b_subtract(a: list[Bit], b: list[Bit], width: int) -> list[Bit]:
    """A - B = A + NEG(B). Costs: width NOTs + 2 ripple-adds.
    This is the binary tax compared to ternary's free subtraction.
    """
    return b_add(a, b_negate(b, width), width)


def b_multiply(a: list[Bit], b: list[Bit], width: int) -> list[Bit]:
    """Shift-and-add multiplication.
    For each bit of b: if 1, add shifted a to accumulator.
    On average half the bits are 1, so ~width/2 adds, each width gates.
    For signed numbers, we should sign-extend; we keep it simple by
    casting through Python ints for negative cases (still counting gates
    for the positive parts of the algorithm)."""
    # Use Booth-style: handle signs by tracking magnitude, recombine sign at end.
    # Simpler: do the multiply on the magnitude, then negate if signs differ.
    a_int = bits_to_int(a)
    b_int = bits_to_int(b)
    sign_negative = (a_int < 0) ^ (b_int < 0)

    a_mag = int_to_bits(abs(a_int), width)
    b_mag = int_to_bits(abs(b_int), width)

    result = [0] * width
    for i in range(width):
        if b_mag[i] == 1:
            # Shift a_mag left by i, then add
            shifted = ([0] * i + a_mag)[:width]
            result = b_add(result, shifted, width)
        # If b_mag[i] == 0, we still pay one gate to test it (the AND with
        # the shifted operand happens implicitly in real hardware via an
        # array multiplier; we approximate by NOT counting skipped cases,
        # which is favorable to binary -- making this comparison fair).

    if sign_negative:
        result = b_negate(result, width)

    return result


def b_compare(a: list[Bit], b: list[Bit], width: int) -> int:
    """Returns -1, 0, or +1. Implemented as A - B then check sign + zero."""
    diff = b_subtract(a, b, width)
    # zero check: OR all bits, NOT it
    is_zero_acc = 0
    for bit in diff:
        is_zero_acc = b_OR(is_zero_acc, bit)
    is_zero = b_NOT(is_zero_acc)
    if is_zero == 1:
        return 0
    # sign is the MSB
    return -1 if diff[-1] == 1 else 1


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)
    W = 16
    fails = 0
    for _ in range(1000):
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        a = int_to_bits(x, W)
        b = int_to_bits(y, W)
        if bits_to_int(b_add(a, b, W)) != ((x + y) % (1 << W) if (x+y) >= 0 else
                                            (x + y) if -(1<<(W-1)) <= x+y < (1<<(W-1)) else None):
            # use direct check instead
            pass
        if bits_to_int(b_add(a, b, W)) != x + y and -(1<<(W-1)) <= x+y < (1<<(W-1)):
            fails += 1
        if bits_to_int(b_subtract(a, b, W)) != x - y and -(1<<(W-1)) <= x-y < (1<<(W-1)):
            fails += 1
        if abs(x) < 50 and abs(y) < 50:
            if bits_to_int(b_multiply(a, b, W)) != x * y:
                fails += 1
        expected = (x > y) - (x < y)
        if b_compare(a, b, W) != expected:
            fails += 1
    print(f"Binary self-test failures: {fails}")
    assert fails == 0
    print("Binary arithmetic verified.")
