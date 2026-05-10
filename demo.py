"""
Master demo: shows the whole ternary computer stack working together.
Run with: python3 demo.py
"""

from trit import (
    Trit, NEG, MIN, MAX, SUM, CARRY,
    int_to_trits, trits_to_int, trits_to_str,
)
from arithmetic import (
    half_adder, full_adder, add, subtract, multiply,
    negate, sign_of, compare,
)
from cpu import CPU, OP, i_reg, i_imm, WORD
from programs import fibonacci_program


def banner(s):
    print()
    print("=" * 70)
    print(f"  {s}")
    print("=" * 70)


# ---------------------------------------------------------------------------
banner("LAYER 1: Primitive gates")

print()
print("NEG:    NEG(-) =", NEG(-1), "  NEG(0) =", NEG(0), "  NEG(+) =", NEG(1))

print()
print("MIN truth table (ternary AND):")
print("       -   0   +")
for a in (-1, 0, 1):
    row = "  ".join(f"{MIN(a,b):>+d}" for b in (-1,0,1)).replace("+0", " 0")
    print(f"  {a:>+d}  {row}")

print()
print("SUM truth table (modular sum, the 'XOR' of ternary):")
print("       -   0   +")
for a in (-1, 0, 1):
    row = "  ".join(f"{SUM(a,b):>+d}" for b in (-1,0,1)).replace("+0", " 0")
    print(f"  {a:>+d}  {row}")

print()
print("Half-adder (a + b -> sum, carry):")
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        s, c = half_adder(a, b)
        print(f"  {a:>+d} + {b:>+d} = ({s:>+d}, carry {c:>+d})  "
              f"reconstructed: {s + 3*c}")


# ---------------------------------------------------------------------------
banner("LAYER 2: Multi-trit arithmetic")

W = 9
examples = [(127, 38), (-50, 100), (1234, -1234), (42, -7)]
print()
print(f"Working in {W}-trit balanced ternary (range +/- {(3**W-1)//2})")
print()
for x, y in examples:
    a = int_to_trits(x, W)
    b = int_to_trits(y, W)
    print(f"  x = {x:>5} = {trits_to_str(a)}")
    print(f"  y = {y:>5} = {trits_to_str(b)}")
    print(f"  x + y = {trits_to_int(add(a,b,W)):>5}  ({trits_to_str(add(a,b,W))})")
    print(f"  x - y = {trits_to_int(subtract(a,b,W)):>5}  ({trits_to_str(subtract(a,b,W))})")
    if abs(x) < 100 and abs(y) < 100:
        print(f"  x * y = {trits_to_int(multiply(a,b,W)):>5}  ({trits_to_str(multiply(a,b,W))})")
    print(f"  cmp   = {compare(a,b,W):>+d}")
    print()


# ---------------------------------------------------------------------------
banner("LAYER 3: The 'subtraction is free' demonstration")

print()
print("In binary:    A - B requires two's complement (flip bits, add 1).")
print("In ternary:   A - B = A + NEG(B), and NEG is per-trit sign flip.")
print()

x, y = 1234, 567
a = int_to_trits(x, W)
b = int_to_trits(y, W)
neg_b = negate(b)
print(f"  y     = {y:>5} = {trits_to_str(b)}")
print(f"  -y    = {-y:>5} = {trits_to_str(neg_b)}    <- single-cycle, parallel NEG")
result = add(a, neg_b, W)
print(f"  x + (-y) = {trits_to_int(result):>5} = {trits_to_str(result)}")
print(f"  Expected:   {x-y}")


# ---------------------------------------------------------------------------
banner("LAYER 4: CPU executing a real program (Fibonacci)")

print()
prog = fibonacci_program(15)
cpu = CPU()
cpu.load_program(prog)
cpu.run()
print(f"  First 15 Fibonacci numbers (computed by ternary CPU):")
print(f"    {cpu.outputs}")
print(f"  Cycles executed: {cpu.cycles}")
print(f"  Every arithmetic op went through balanced-ternary gates.")


# ---------------------------------------------------------------------------
banner("LAYER 4b: A different program -- factorial")

# n! using the multiply instruction
def factorial_program(n: int) -> list:
    # R1 = result (start at 1)
    # R2 = counter (start at n)
    # R3 = constant 1 (decrement)
    return [
        i_imm(OP.LDI, 1, 1),       # 0: R1 = 1
        i_imm(OP.LDI, 2, n),       # 1: R2 = n
        i_imm(OP.LDI, 3, 1),       # 2: R3 = 1
        # loop:
        i_imm(OP.JZ, 2, 3),        # 3: if R2==0 jump +3 -> PC=7 (PRINT)
        i_reg(OP.MUL, 1, 1, 2),    # 4: R1 = R1 * R2
        i_reg(OP.SUB, 2, 2, 3),    # 5: R2 = R2 - 1
        i_imm(OP.JMP, 0, -4),      # 6: PC was 7, -4 -> 3
        # done:
        i_reg(OP.PRINT, 1),        # 7: print R1
        i_reg(OP.HALT),            # 8
    ]

print()
for n in range(1, 8):
    cpu = CPU()
    cpu.load_program(factorial_program(n))
    cpu.run()
    expected = 1
    for i in range(1, n+1): expected *= i
    print(f"  {n}! = {cpu.outputs[0]:>5}  (expected {expected:>5}, "
          f"{cpu.cycles} cycles)")


# ---------------------------------------------------------------------------
banner("Stack summary")

print("""
  Layer 1: Trit type and primitive gates              [trit.py]
  Layer 2: Adders, multipliers, comparisons           [arithmetic.py]
  Layer 3: Registers, memory, ALU, fetch-decode loop  [cpu.py]
  Layer 4: Programs (Fibonacci, factorial)            [programs.py]
  Notes:   ISA design, ternary float, ternary NN, OS  [DESIGN_NOTES.md]

  Word width:    9 trits  (range +/-{0})
  Memory size:   {1} words
  Registers:     9 (R0 hardwired to zero)
  Opcodes:       17 implemented (out of 27 available with 3-trit field)

  The Setun ran balanced-ternary in 1958 with 18-trit words.
  This simulator runs the same logic in 2026, with 70 years of hindsight.
""".format((3**9-1)//2, 729))
