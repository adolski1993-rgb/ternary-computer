"""
Balanced ternary core: the Trit type and primitive gates.

Trits take values from {-1, 0, +1}. We use plain Python ints so arithmetic
stays readable, but we wrap operations in named gate functions to make the
ternary logic explicit (and to mirror what real hardware would do).
"""

from typing import Iterable

# Type alias for clarity. A Trit is always one of -1, 0, +1.
Trit = int

VALID_TRITS = (-1, 0, 1)


def _check(t: Trit) -> Trit:
    if t not in VALID_TRITS:
        raise ValueError(f"Not a valid trit: {t!r}")
    return t


# ---------------------------------------------------------------------------
# Unary gates
# ---------------------------------------------------------------------------

def NEG(a: Trit) -> Trit:
    """Negation: flips sign. -1 <-> +1, 0 -> 0. The workhorse gate."""
    return -_check(a)


def CYCLE(a: Trit) -> Trit:
    """Cycle: -1 -> 0 -> +1 -> -1.  Useful for state machines."""
    _check(a)
    return {-1: 0, 0: 1, 1: -1}[a]


# Literal-selector gates: output +1 iff input matches the target, else -1.
# Functionally complete when combined with MIN.
def IS_NEG(a: Trit) -> Trit:
    return 1 if _check(a) == -1 else -1


def IS_ZERO(a: Trit) -> Trit:
    return 1 if _check(a) == 0 else -1


def IS_POS(a: Trit) -> Trit:
    return 1 if _check(a) == 1 else -1


# ---------------------------------------------------------------------------
# Binary gates
# ---------------------------------------------------------------------------

def MIN(a: Trit, b: Trit) -> Trit:
    """Ternary AND: output the smaller input."""
    return min(_check(a), _check(b))


def MAX(a: Trit, b: Trit) -> Trit:
    """Ternary OR: output the larger input."""
    return max(_check(a), _check(b))


def SUM(a: Trit, b: Trit) -> Trit:
    """Modular sum (mod 3, mapped to balanced range).
    -1 + -1 = -2 -> +1
    +1 + +1 = +2 -> -1
    Otherwise: ordinary sum.
    The 'XOR-equivalent' of ternary; appears inside the half-adder.
    """
    _check(a); _check(b)
    s = a + b
    if s == -2: return 1
    if s == 2:  return -1
    return s


def CONSENSUS(a: Trit, b: Trit) -> Trit:
    """Output the majority value, 0 if no majority."""
    _check(a); _check(b)
    if a == b: return a
    return 0


def CARRY(a: Trit, b: Trit) -> Trit:
    """Carry-out for a single-trit addition.
    Nonzero only when both inputs share a sign and sum to +/- 2.
    """
    _check(a); _check(b)
    s = a + b
    if s == 2:  return 1
    if s == -2: return -1
    return 0


# ---------------------------------------------------------------------------
# Conversion helpers between Python ints and balanced-ternary digit lists.
# Convention: trits[0] is the LEAST-significant trit (3^0 place).
# ---------------------------------------------------------------------------

def int_to_trits(n: int, width: int | None = None) -> list[Trit]:
    """Convert int n to balanced-ternary digit list, LSB first.
    If width is given, pad/truncate to that many trits.
    """
    if n == 0:
        digits = [0]
    else:
        digits = []
        x = n
        while x != 0:
            r = x % 3
            x //= 3
            if r == 2:
                r = -1
                x += 1
            digits.append(r)
    if width is not None:
        if len(digits) > width:
            # Truncation drops high trits (overflow).
            digits = digits[:width]
        else:
            digits.extend([0] * (width - len(digits)))
    return digits


def trits_to_int(trits: Iterable[Trit]) -> int:
    """Convert LSB-first digit list back to an int."""
    n = 0
    p = 1
    for t in trits:
        _check(t)
        n += t * p
        p *= 3
    return n


def trits_to_str(trits: Iterable[Trit]) -> str:
    """Pretty-print: '+', '0', '-', written MSB-first like normal numbers."""
    sym = {-1: '-', 0: '0', 1: '+'}
    return ''.join(sym[t] for t in reversed(list(trits)))


def str_to_trits(s: str) -> list[Trit]:
    """Inverse of trits_to_str. Accepts '+', '0', '-'."""
    sym = {'-': -1, '0': 0, '+': 1, 'T': -1, '1': 1}
    return [sym[c] for c in reversed(s.strip())]
