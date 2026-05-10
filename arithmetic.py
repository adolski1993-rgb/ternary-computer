"""
Arithmetic units built from the primitive gates in trit.py.

Everything here is implemented gate-by-gate, the way real hardware would be.
We never use Python's `+` on integer values of multi-trit numbers; we always
go through the gate functions so the simulation is faithful.
"""

from trit import (
    Trit, MIN, MAX, SUM, CARRY, NEG, IS_ZERO, IS_POS, IS_NEG,
    int_to_trits, trits_to_int,
)


# ---------------------------------------------------------------------------
# Half-adder and full-adder
# ---------------------------------------------------------------------------

def half_adder(a: Trit, b: Trit) -> tuple[Trit, Trit]:
    """Add two trits. Returns (sum_trit, carry_trit)."""
    return SUM(a, b), CARRY(a, b)


def full_adder(a: Trit, b: Trit, cin: Trit) -> tuple[Trit, Trit]:
    """Add three trits. Returns (sum_trit, carry_out)."""
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    # If c1 != 0, then a and b have the same sign and s1 has the opposite sign.
    # If c2 != 0, then s1 and cin have the same sign, so cin matches c1's
    # OPPOSITE.  Hence c1 and c2 always have opposite signs when both nonzero,
    # and at least one is zero.  Their algebraic sum is the correct combined
    # carry, and it's always a valid trit (-1, 0, or +1).
    cout = c1 + c2  # safe: never +/-2
    return s2, cout


# ---------------------------------------------------------------------------
# Multi-trit ripple-carry adder
# ---------------------------------------------------------------------------

def add(a: list[Trit], b: list[Trit], width: int) -> list[Trit]:
    """Add two trit-vectors of length `width`, returning a vector of `width`.
    Overflow is silently truncated (like real hardware without overflow flag).
    """
    assert len(a) == width and len(b) == width
    result = [0] * width
    carry: Trit = 0
    for i in range(width):
        s, carry = full_adder(a[i], b[i], carry)
        result[i] = s
    return result


def negate(a: list[Trit]) -> list[Trit]:
    """Negate a multi-trit number: NEG every digit in parallel."""
    return [NEG(t) for t in a]


def subtract(a: list[Trit], b: list[Trit], width: int) -> list[Trit]:
    """A - B = A + (-B).  Free in balanced ternary."""
    return add(a, negate(b), width)


# ---------------------------------------------------------------------------
# Shifts (multiply / divide by powers of 3)
# ---------------------------------------------------------------------------

def shift_left(a: list[Trit], n: int, width: int) -> list[Trit]:
    """Shift `n` trits left = multiply by 3^n."""
    return ([0] * n + a)[:width]


def shift_right(a: list[Trit], n: int, width: int) -> list[Trit]:
    """Shift `n` trits right = divide by 3^n (truncating toward zero,
    which in balanced ternary is the same as proper rounding)."""
    return (a[n:] + [0] * n)[:width]


# ---------------------------------------------------------------------------
# Multiplication: shift-and-add, exploiting that each trit is in {-1, 0, +1}
# ---------------------------------------------------------------------------

def multiply(a: list[Trit], b: list[Trit], width: int) -> list[Trit]:
    """Multiply two trit vectors. Returns a `width`-trit result (low part)."""
    result = [0] * width
    for i, b_i in enumerate(b):
        if b_i == 0:
            continue  # 1/3 of trits are zero on average -> free skip
        partial = shift_left(a, i, width)
        if b_i == -1:
            partial = negate(partial)
        result = add(result, partial, width)
    return result


# ---------------------------------------------------------------------------
# Comparison gates (built from subtraction + sign check)
# ---------------------------------------------------------------------------

def sign_of(a: list[Trit]) -> Trit:
    """Return -1, 0, or +1 depending on the sign of a multi-trit number.
    The sign IS the most-significant nonzero trit (a beautiful property
    of balanced ternary: no separate sign bit needed)."""
    for t in reversed(a):
        if t != 0:
            return t
    return 0


def compare(a: list[Trit], b: list[Trit], width: int) -> Trit:
    """Returns -1 if a<b, 0 if a==b, +1 if a>b."""
    return sign_of(subtract(a, b, width))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 10  # 10 trits: range -29524 .. +29524

    # Test addition and subtraction over a range of values
    import random
    random.seed(0)
    print("Testing arithmetic over 10-trit values...")
    fails = 0
    for _ in range(2000):
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        a = int_to_trits(x, W)
        b = int_to_trits(y, W)
        # add
        if trits_to_int(add(a, b, W)) != x + y:
            fails += 1
        # subtract
        if trits_to_int(subtract(a, b, W)) != x - y:
            fails += 1
        # multiply (smaller range to avoid overflow)
        if abs(x) < 100 and abs(y) < 100:
            if trits_to_int(multiply(a, b, W)) != x * y:
                fails += 1
        # compare
        expected = (x > y) - (x < y)
        if compare(a, b, W) != expected:
            fails += 1
    print(f"  Failures: {fails}")
    assert fails == 0
    print("  All arithmetic operations correct.")
