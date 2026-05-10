"""
Setun-3: A small balanced-ternary CPU simulator.

Word width: 9 trits (range -9841 .. +9841, ~14 bits of binary equivalent)
Registers: 9 general-purpose registers R0..R8 (R0 is hardwired to zero, like RISC)
Memory:    729 words (3^6) of 9-trit words
Instruction width: 9 trits, decoded as:
    [opcode: 3 trits] [rd: 2 trits] [rs1: 2 trits] [rs2 or imm: 2 trits]

3 trits give us 27 opcodes (plenty).
2 trits address 9 registers via balanced-ternary representation
(though we map the 9 patterns onto R0..R8 via a small lookup).

This isn't a "real" hardware design, but it's a faithful model: every
arithmetic/logic operation goes through the gate-level functions in
arithmetic.py, so the CPU genuinely runs in balanced ternary.
"""

from dataclasses import dataclass, field
from trit import (
    Trit, NEG, MIN, MAX, SUM, IS_ZERO, IS_POS, IS_NEG,
    int_to_trits, trits_to_int, trits_to_str,
)
from arithmetic import (
    add, subtract, multiply, negate,
    shift_left, shift_right, sign_of, compare,
)


# ---------------------------------------------------------------------------
# Architecture parameters
# ---------------------------------------------------------------------------

WORD = 9          # trits per word
NREGS = 9         # registers R0..R8
MEMSIZE = 729     # 3^6 words

# 9-trit max magnitude:
MAX_VAL = (3 ** WORD - 1) // 2  # 9841


# ---------------------------------------------------------------------------
# Encoding registers as 2-trit fields
# ---------------------------------------------------------------------------
# A 2-trit field has 9 values: -4..+4 in balanced ternary.
# We map them onto register indices 0..8 in a fixed order.

_REG_ORDER = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def reg_field(reg_idx: int) -> list[Trit]:
    """Encode register index 0..8 as a 2-trit field (LSB first)."""
    return int_to_trits(_REG_ORDER[reg_idx], 2)

def decode_reg(field2: list[Trit]) -> int:
    """Decode a 2-trit field back to a register index 0..8."""
    return _REG_ORDER.index(trits_to_int(field2))


# ---------------------------------------------------------------------------
# Opcodes (3-trit values, written as ints in the balanced range -13..+13)
# ---------------------------------------------------------------------------

class OP:
    HALT = -13   # stop the CPU
    NOP  = 0     # do nothing
    # arithmetic / logic
    ADD  = 1     # rd = rs1 + rs2
    SUB  = 2     # rd = rs1 - rs2
    MUL  = 3     # rd = rs1 * rs2
    NEG  = 4     # rd = -rs1
    MIN_ = 5     # rd = MIN(rs1, rs2)  (trit-wise)
    MAX_ = 6     # rd = MAX(rs1, rs2)  (trit-wise)
    # immediate
    LDI  = 7     # rd = sign-extended immediate (rs1, rs2 form 4-trit signed imm)
    # memory
    LD   = 8     # rd = MEM[rs1 + imm(rs2 as small offset)]
    ST   = 9     # MEM[rs1 + imm(rs2)] = rd
    # control flow
    JMP  = 10    # PC = PC + (rs1 sign-extended; rd, rs2 unused as offset combine)
    JZ   = 11    # if rd == 0: PC += offset
    JN   = 12    # if rd <  0: PC += offset
    JP   = 13    # if rd >  0: PC += offset
    # special
    PRINT = -1   # print rd as integer (debugging convenience)
    INPUT = -2   # rd = next input value


OPCODES = {v: k for k, v in vars(OP).items() if not k.startswith('_') and isinstance(v, int)}


# ---------------------------------------------------------------------------
# Instruction encoding helpers
# ---------------------------------------------------------------------------

def encode(op: int, rd: int = 0, rs1: int = 0, rs2: int = 0,
           imm: int | None = None) -> list[Trit]:
    """Encode one 9-trit instruction. If `imm` is given, it occupies
    the rs1+rs2 fields as a 4-trit signed value (-40..+40)."""
    op_trits = int_to_trits(op, 3)
    rd_trits = reg_field(rd)
    if imm is not None:
        imm_trits = int_to_trits(imm, 4)
        return op_trits + rd_trits + imm_trits  # 3+2+4 = 9
    rs1_trits = reg_field(rs1)
    rs2_trits = reg_field(rs2)
    return op_trits + rd_trits + rs1_trits + rs2_trits  # 3+2+2+2 = 9


def decode(instr: list[Trit]):
    """Return (op, rd, rs1, rs2, imm)."""
    op = trits_to_int(instr[0:3])
    rd = decode_reg(instr[3:5])
    rs1_field = instr[5:7]
    rs2_field = instr[7:9]
    rs1 = decode_reg(rs1_field)
    rs2 = decode_reg(rs2_field)
    imm = trits_to_int(instr[5:9])  # 4-trit signed imm interpretation
    return op, rd, rs1, rs2, imm


# ---------------------------------------------------------------------------
# CPU state
# ---------------------------------------------------------------------------

@dataclass
class CPU:
    regs:   list[list[Trit]] = field(default_factory=lambda: [[0]*WORD for _ in range(NREGS)])
    memory: list[list[Trit]] = field(default_factory=lambda: [[0]*WORD for _ in range(MEMSIZE)])
    pc:     int = 0
    halted: bool = False
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    cycles: int = 0

    def load_program(self, program: list[list[Trit]], start: int = 0):
        for i, instr in enumerate(program):
            self.memory[start + i] = list(instr)

    def reg_int(self, idx: int) -> int:
        return trits_to_int(self.regs[idx])

    def set_reg(self, idx: int, value: list[Trit]):
        if idx == 0:
            return  # R0 is hardwired to zero
        self.regs[idx] = list(value)

    def step(self):
        if self.halted:
            return
        instr = self.memory[self.pc]
        op, rd, rs1, rs2, imm = decode(instr)
        self.pc += 1
        self.cycles += 1

        rs1_val = self.regs[rs1]
        rs2_val = self.regs[rs2]
        rd_val  = self.regs[rd]

        if op == OP.HALT:
            self.halted = True

        elif op == OP.NOP:
            pass

        elif op == OP.ADD:
            self.set_reg(rd, add(rs1_val, rs2_val, WORD))
        elif op == OP.SUB:
            self.set_reg(rd, subtract(rs1_val, rs2_val, WORD))
        elif op == OP.MUL:
            self.set_reg(rd, multiply(rs1_val, rs2_val, WORD))
        elif op == OP.NEG:
            self.set_reg(rd, negate(rs1_val))
        elif op == OP.MIN_:
            self.set_reg(rd, [MIN(a,b) for a,b in zip(rs1_val, rs2_val)])
        elif op == OP.MAX_:
            self.set_reg(rd, [MAX(a,b) for a,b in zip(rs1_val, rs2_val)])

        elif op == OP.LDI:
            self.set_reg(rd, int_to_trits(imm, WORD))

        elif op == OP.LD:
            addr = trits_to_int(rs1_val) + trits_to_int(rs2_val)
            addr %= MEMSIZE
            self.set_reg(rd, list(self.memory[addr]))

        elif op == OP.ST:
            addr = trits_to_int(rs1_val) + trits_to_int(rs2_val)
            addr %= MEMSIZE
            self.memory[addr] = list(rd_val)

        elif op == OP.JMP:
            self.pc = (self.pc + imm) % MEMSIZE
        elif op == OP.JZ:
            if sign_of(rd_val) == 0:
                self.pc = (self.pc + imm) % MEMSIZE
        elif op == OP.JN:
            if sign_of(rd_val) < 0:
                self.pc = (self.pc + imm) % MEMSIZE
        elif op == OP.JP:
            if sign_of(rd_val) > 0:
                self.pc = (self.pc + imm) % MEMSIZE

        elif op == OP.PRINT:
            self.outputs.append(self.reg_int(rd))
        elif op == OP.INPUT:
            v = self.inputs.pop(0) if self.inputs else 0
            self.set_reg(rd, int_to_trits(v, WORD))

        else:
            raise ValueError(f"Unknown opcode {op} at PC={self.pc-1}")

    def run(self, max_cycles: int = 100_000):
        while not self.halted and self.cycles < max_cycles:
            self.step()
        if not self.halted:
            raise RuntimeError(f"Did not halt within {max_cycles} cycles")


# ---------------------------------------------------------------------------
# Tiny assembler convenience: build programs from a list of tuples.
# ---------------------------------------------------------------------------

def asm(*instructions) -> list[list[Trit]]:
    """Each instruction is a tuple: (op, rd, rs1, rs2) or (op, rd, imm=N)
    or just (op,) for HALT/NOP.

    Use kwarg imm=... for immediate instructions:
        asm((OP.LDI, 1, 0, 0))            # bad, treats 0,0 as rs1,rs2
        asm((OP.LDI, 1), {'imm': 5})      # NO - use the helper below
    Better: use `i_imm` and `i_reg` builders.
    """
    out = []
    for entry in instructions:
        out.append(entry if isinstance(entry, list) else list(entry))
    return out


def i_reg(op, rd=0, rs1=0, rs2=0):
    return encode(op, rd=rd, rs1=rs1, rs2=rs2)

def i_imm(op, rd, imm):
    return encode(op, rd=rd, imm=imm)


# ---------------------------------------------------------------------------
# Self-test: simple program
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Compute 7 + 5 - 2 = 10, print it.
    program = [
        i_imm(OP.LDI, 1, 7),       # R1 = 7
        i_imm(OP.LDI, 2, 5),       # R2 = 5
        i_imm(OP.LDI, 3, 2),       # R3 = 2
        i_reg(OP.ADD, 4, 1, 2),    # R4 = R1 + R2 = 12
        i_reg(OP.SUB, 4, 4, 3),    # R4 = R4 - R3 = 10
        i_reg(OP.PRINT, 4),        # output R4
        i_reg(OP.HALT),
    ]
    cpu = CPU()
    cpu.load_program(program)
    cpu.run()
    print(f"Outputs: {cpu.outputs}")
    print(f"Cycles : {cpu.cycles}")
    assert cpu.outputs == [10]
    print("CPU smoke test passed.")
