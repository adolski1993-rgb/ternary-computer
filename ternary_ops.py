"""
Instrumented ternary arithmetic with gate counting.
Mirror of arithmetic.py but with COUNTER.ternary_gates incremented per gate.
"""

from binary_ops import COUNTER
from trit import Trit, int_to_trits, trits_to_int


# ---------------------------------------------------------------------------
# Counted primitive gates
# ---------------------------------------------------------------------------

def t_NEG(a: Trit) -> Trit:
    COUNTER.ternary_gates += 1
    return -a

def t_MIN(a: Trit, b: Trit) -> Trit:
    COUNTER.ternary_gates += 1
    return min(a, b)

def t_MAX(a: Trit, b: Trit) -> Trit:
    COUNTER.ternary_gates += 1
    return max(a, b)

def t_SUM(a: Trit, b: Trit) -> Trit:
    COUNTER.ternary_gates += 1
    s = a + b
    if s == -2: return 1
    if s == 2:  return -1
    return s

def t_CARRY(a: Trit, b: Trit) -> Trit:
    COUNTER.ternary_gates += 1
    s = a + b
    if s == 2:  return 1
    if s == -2: return -1
    return 0

def t_OR_combine(a: Trit, b: Trit) -> Trit:
    """For combining carries; integer add since they're guaranteed non-conflicting."""
    COUNTER.ternary_gates += 1
    return a + b


# ---------------------------------------------------------------------------
# Adders
# ---------------------------------------------------------------------------

def t_half_adder(a: Trit, b: Trit) -> tuple[Trit, Trit]:
    """2 gates."""
    return t_SUM(a, b), t_CARRY(a, b)


def t_full_adder(a: Trit, b: Trit, cin: Trit) -> tuple[Trit, Trit]:
    """5 gates: 2 half-adders + 1 carry-combine."""
    s1, c1 = t_half_adder(a, b)
    s2, c2 = t_half_adder(s1, cin)
    return s2, t_OR_combine(c1, c2)


# ---------------------------------------------------------------------------
# Multi-trit operations
# ---------------------------------------------------------------------------

def t_add(a: list[Trit], b: list[Trit], width: int) -> list[Trit]:
    """5 gates per trit. Same shape as binary ripple-carry."""
    result = [0] * width
    carry: Trit = 0
    for i in range(width):
        s, carry = t_full_adder(a[i], b[i], carry)
        result[i] = s
    return result


def t_negate(a: list[Trit]) -> list[Trit]:
    """Width NEG gates, in parallel.
    KEY DIFFERENCE FROM BINARY: no add-1 step needed."""
    return [t_NEG(t) for t in a]


def t_subtract(a: list[Trit], b: list[Trit], width: int) -> list[Trit]:
    """Width NEG + one ripple-add. No second add for two's complement."""
    return t_add(a, t_negate(b), width)


def t_multiply(a: list[Trit], b: list[Trit], width: int) -> list[Trit]:
    """Shift-and-add. On average 1/3 of trits are zero -> skipped.
    Of the nonzero trits, half are -1 (subtract via cheap NEG)."""
    result = [0] * width
    for i, b_i in enumerate(b):
        if b_i == 0:
            continue  # free skip - 1/3 of trits
        # Shift a left by i (no gates - it's wiring)
        partial = ([0] * i + a)[:width]
        if b_i == -1:
            partial = t_negate(partial)  # width NEG gates
        result = t_add(result, partial, width)
    return result


def t_sign_of(a: list[Trit]) -> Trit:
    """The sign IS the most-significant nonzero trit. Free in hardware
    (priority encoder), but we'll count one gate per inspected trit."""
    for t in reversed(a):
        COUNTER.ternary_gates += 1
        if t != 0:
            return t
    return 0


def t_compare(a: list[Trit], b: list[Trit], width: int) -> Trit:
    """A - B, then sign-of."""
    return t_sign_of(t_subtract(a, b, width))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)
    W = 10
    fails = 0
    for _ in range(1000):
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        a = int_to_trits(x, W)
        b = int_to_trits(y, W)
        if trits_to_int(t_add(a, b, W)) != x + y:
            fails += 1
        if trits_to_int(t_subtract(a, b, W)) != x - y:
            fails += 1
        if abs(x) < 50 and abs(y) < 50:
            if trits_to_int(t_multiply(a, b, W)) != x * y:
                fails += 1
        expected = (x > y) - (x < y)
        if t_compare(a, b, W) != expected:
            fails += 1
    print(f"Ternary self-test failures: {fails}")
    assert fails == 0
    print(f"Gates counted during testing: {COUNTER.ternary_gates}")
    print("Ternary arithmetic verified.")
