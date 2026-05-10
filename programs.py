"""
Programs running on the ternary CPU.
Each program is built as a list of encoded 9-trit instructions.
"""

from cpu import CPU, OP, i_reg, i_imm, encode, decode, WORD, NREGS
from trit import int_to_trits, trits_to_int, trits_to_str


def disassemble(program):
    """Pretty-print a program for inspection."""
    OP_NAMES = {
        OP.HALT: "HALT", OP.NOP: "NOP", OP.ADD: "ADD", OP.SUB: "SUB",
        OP.MUL: "MUL", OP.NEG: "NEG", OP.MIN_: "MIN", OP.MAX_: "MAX",
        OP.LDI: "LDI", OP.LD: "LD", OP.ST: "ST",
        OP.JMP: "JMP", OP.JZ: "JZ", OP.JN: "JN", OP.JP: "JP",
        OP.PRINT: "PRINT", OP.INPUT: "INPUT",
    }
    print(f"{'Addr':<5} {'Trits':<12} {'Mnemonic':<28} Decoded")
    print("-" * 70)
    for addr, instr in enumerate(program):
        op, rd, rs1, rs2, imm = decode(instr)
        name = OP_NAMES.get(op, f"OP{op}")
        if op in (OP.LDI, OP.JMP):
            mnem = f"{name} R{rd}, #{imm}"
        elif op in (OP.JZ, OP.JN, OP.JP):
            mnem = f"{name} R{rd}, #{imm}"
        elif op in (OP.HALT, OP.NOP):
            mnem = name
        elif op in (OP.PRINT, OP.INPUT):
            mnem = f"{name} R{rd}"
        elif op == OP.NEG:
            mnem = f"{name} R{rd}, R{rs1}"
        else:
            mnem = f"{name} R{rd}, R{rs1}, R{rs2}"
        print(f"{addr:<5} {trits_to_str(instr):<12} {mnem:<28} op={op} rd={rd} rs1={rs1} rs2={rs2} imm={imm}")
    print()


# ---------------------------------------------------------------------------
# Program: Fibonacci sequence
# Compute and print F(1)..F(N) using register-only arithmetic.
# ---------------------------------------------------------------------------

def fibonacci_program(n_terms: int) -> list:
    """Print the first n_terms Fibonacci numbers."""
    # Register plan:
    #   R1 = a, R2 = b, R3 = counter, R4 = scratch, R5 = constant 1
    program = []
    # Registers we reserve:
    #   R1: a, R2: b, R3: counter
    #   R4: scratch
    #   R5: +1  (decrement amount)
    program.append(i_imm(OP.LDI, 1, 1))      # 0
    program.append(i_imm(OP.LDI, 2, 1))      # 1
    program.append(i_imm(OP.LDI, 3, n_terms))# 2
    program.append(i_imm(OP.LDI, 5, 1))      # 3
    # loop_start (PC=4):
    program.append(i_imm(OP.JZ, 3, 6))       # 4: if R3==0, PC += 6 -> goes to PC=11 (HALT)
    program.append(i_reg(OP.PRINT, 1))       # 5: print R1
    program.append(i_reg(OP.ADD, 4, 1, 2))   # 6: R4 = R1+R2
    program.append(i_reg(OP.ADD, 1, 2, 0))   # 7: R1 = R2 (R2+R0)
    program.append(i_reg(OP.ADD, 2, 4, 0))   # 8: R2 = R4
    program.append(i_reg(OP.SUB, 3, 3, 5))   # 9: R3 = R3-1
    program.append(i_imm(OP.JMP, 0, -7))     #10: PC was incremented to 11, -7 -> back to 4
    program.append(i_reg(OP.HALT))           #11: HALT
    return program


if __name__ == "__main__":
    print("=" * 70)
    print("TERNARY CPU PROGRAM: Fibonacci(1..10)")
    print("=" * 70)
    prog = fibonacci_program(10)
    print()
    print("Disassembly:")
    print()
    disassemble(prog)

    cpu = CPU()
    cpu.load_program(prog)
    cpu.run()

    print(f"Outputs: {cpu.outputs}")
    print(f"Expected: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]")
    print(f"Cycles executed: {cpu.cycles}")

    expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert cpu.outputs == expected, f"Mismatch: got {cpu.outputs}"
    print()
    print("Fibonacci on a balanced-ternary CPU: SUCCESS.")
