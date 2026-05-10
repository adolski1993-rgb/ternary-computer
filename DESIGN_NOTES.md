# Ternary Computing: Design Notes Beyond the Simulator

You now have a working balanced-ternary CPU (`cpu.py`) that executes real
programs (`programs.py`) using gate-level primitives (`trit.py`,
`arithmetic.py`). This document sketches the four "what's next" directions.

---

## 1. Instruction-Set Architecture: What a Clean Ternary ISA Would Look Like

The toy ISA in `cpu.py` is deliberately small (17 opcodes). With 3 trits of
opcode you have 27 slots. A well-designed full-scale ternary ISA would
exploit the structure of trits in ways binary ISAs can't.

### The trit-as-condition trick

Every comparison naturally produces -1, 0, or +1. So conditional execution
becomes a single-trit predicate, not a flag register.

Instead of binary's `BEQ, BNE, BLT, BGT, BLE, BGE` (six branches), ternary
has:
- `JN  rd, off` — branch if rd < 0
- `JZ  rd, off` — branch if rd = 0
- `JP  rd, off` — branch if rd > 0

That's it. To branch "≤ 0", encode it as the SUM of JN and JZ targets, or
just use two instructions. To branch "≠ 0", use JN || JP.

Better still: a single instruction `BR3 rd, off_neg, off_zero, off_pos` that
takes three offsets and dispatches on the trit. One instruction, three-way
branch. Binary can't express this naturally; in binary you'd compile it into
a compare + conditional + unconditional, or a jump table.

### The compare-and-classify primitive

`CMP rd, rs1, rs2` returns -1/0/+1 directly into rd. No flag register, no
conditional move tax. Combined with BR3, this gives clean if/elif/else with
no branch chains:

```
CMP r4, r1, r2          # r4 = sign(r1 - r2)
BR3 r4, less, equal, greater
```

Three-way comparison is genuinely the natural primitive. Tony Hoare famously
regretted not putting three-way comparison into Algol; ternary hardware would
make it the default.

### Conditional execution everywhere

ARM's "predicate every instruction" idea works beautifully in ternary.
Add a 1-trit predicate field to every instruction:

```
[opcode 3t] [pred 1t] [rd 2t] [rs1 2t] [rs2 1t]
```

Predicate field meanings:
- `0` → always execute
- `+` → execute if last comparison was positive
- `-` → execute if last comparison was negative

This lets the compiler generate branchless code for short conditionals:
```
CMP r4, r1, r2
[+] ADD r5, r5, r6      # only if r1 > r2
[-] SUB r5, r5, r6      # only if r1 < r2
```

Two instructions, no branches, three-way if/else.

### Instruction format proposal (full design)

A clean 9-trit instruction:
```
[opcode 3t] [predicate 1t] [rd 2t] [rs1/imm 3t]
```
- 27 opcodes
- 3 predicate values (always / pos / neg)
- 9 registers
- 27 immediate values OR 9 source registers (mode bit folded into opcode)

For wider code, use 18-trit instructions (= one ternary "long"):
```
[opcode 3t] [predicate 1t] [rd 2t] [rs1 2t] [rs2 2t] [imm 8t]
```
- 8-trit immediate = -3280..+3280 range
- All three operands plus immediate together
- Closer to RISC-V's 32-bit instruction richness

### What disappears in a ternary ISA

- **Sign bit** — gone. Sign of a number is the sign of its most-significant
  nonzero trit.
- **Two's complement and overflow weirdness** — gone. Negation is per-trit.
- **Separate signed/unsigned ops** (binary needs `MUL/MULU`, `DIV/DIVU`,
  `SHR/SHRA`) — gone. Numbers are always signed; arithmetic shift is the
  only shift.
- **Carry/overflow flags** — gone in most cases. Branches dispatch on actual
  values, not stored flag state.
- **Half the comparison instructions** — gone, BR3 covers it.

A ternary RISC-V-class ISA would probably have ~40 instructions where binary
RISC-V has ~50. Not a huge win, but the instructions are *more orthogonal*.

---

## 2. Floating Point in Balanced Ternary

Binary IEEE 754 floats are ugly. They have:
- A separate sign bit
- A biased exponent (excess-127, excess-1023)
- An implicit leading 1 in the mantissa
- Two zeros (+0 and -0)
- Subnormals as a special case

Ternary makes most of this go away.

### The format

A ternary float is just `[exponent: E trits] [mantissa: M trits]`, both
balanced ternary. No sign bit. No bias. No implicit anything.

The value is `mantissa × 3^exponent`.

Sign comes for free because the mantissa is signed-balanced. -0 doesn't
exist (zero is just zero). The asymmetry of `[-127..+128]` exponents in IEEE
754 doesn't exist; the ternary exponent ranges from `-(3^E-1)/2` to
`+(3^E-1)/2`, perfectly symmetric.

### Normalization

A normalized ternary float has its most-significant mantissa trit nonzero.
That's it. No "implicit leading bit" trick. To normalize:
1. Find the leading nonzero trit.
2. Shift mantissa left until it's in position MSB-1 (or wherever the
   convention puts it).
3. Adjust exponent.

There's no "implicit 1" hack because the leading trit can be `+` OR `-`,
carrying real sign information. We can't drop it.

### Special values

- **Zero**: mantissa is all zeros, exponent ignored (or specified as -max).
- **Infinity**: encode as exponent = +max, mantissa = `+0...0` (positive
  infinity) or `-0...0` (negative infinity). Notice: still no separate sign
  bit, the mantissa's leading trit IS the sign of infinity.
- **NaN**: exponent = +max, mantissa nonzero non-leading. Plenty of bit
  patterns available.

### A concrete example: ternary32

Like binary's float32, but in ternary:
- 6 trits exponent (range -364..+364)
- 14 trits mantissa
- 20 trits total ≈ 32 bits of binary information

Range: roughly `3^364 ≈ 10^174`, comparable to binary's `2^127 ≈ 10^38`...
actually MUCH wider. To match float32's range with comparable precision,
ternary32 would have:
- 4-trit exponent (range -40..+40, so ~3^40 ≈ 10^19, similar to float32)
- 16-trit mantissa (precision ~3^16 ≈ 4×10^7, similar to float32's 24 bits)

Total: 20 trits. Same information density as 32 bits.

### The win

- No sign-bit special handling
- No bias arithmetic in the FPU
- Symmetric range and rounding
- Comparison of floats is *literally* compare-as-integers, no special cases
- Negation is a single per-trit gate

The ternary FPU is genuinely simpler than the binary one. Hoare would
approve.

---

## 3. Ternary Neural Networks

This is where ternary hardware might actually return to commercial relevance.

### The setup

Modern LLMs are gigantic, and their dominant operation is:
```
output = activations · weights
```
Matrix multiplication of huge matrices.

Standard practice: 16-bit floats (or 8-bit ints) for both. This is enormous
in memory and compute.

### BitNet b1.58

In 2024 Microsoft Research published BitNet b1.58, where every weight is
quantized to one of three values: -1, 0, or +1. Activations stay
higher-precision (8-bit), but weights become trits.

Stunning result: the model is essentially as accurate as full-precision at
3B+ parameters, while:
- Weight memory shrinks ~10x (1.58 bits vs 16 bits per weight)
- The matrix multiply becomes weight-conditional addition: no
  multiplications required
- Energy per operation drops dramatically

### Why ternary is the sweet spot

Why not binary `{-1, +1}`? Because zero matters enormously: it lets the
network express "this connection is irrelevant." Binarized networks
(`{-1, +1}` only) lose accuracy badly.

Why not 4-value or higher? Diminishing returns vs. complexity. Three values
turns out to be the smallest set that captures negative/zero/positive, which
matches what neural network weights actually need.

### What ternary hardware buys you

A trit-weighted MAC (multiply-accumulate) is:
```
if w == 0:    do nothing
if w == +1:   accumulator += activation
if w == -1:   accumulator -= activation
```

There's no multiplier. The "multiply" is a 3-way mux into add/sub/skip.

On binary hardware, you have to:
- Store the trit in 2 bits (wasting ~25%)
- Decode the 2 bits into a control signal
- Use a full multiplier or at least a mux + adder

On native ternary hardware:
- Store one trit in one trit cell
- The trit IS the control signal
- Adder accepts trit weights natively

Estimated speedup over a binary MAC of equivalent area: 3-5x. Estimated
energy improvement: 5-10x.

### The architecture: ternary tensor processor

```
+----------------+      +----------------+
| Weight memory  |      | Activation     |
| (trits, dense) |      | memory (8-bit) |
+--------+-------+      +--------+-------+
         |                       |
         v                       v
     +-------------------------------+
     | Ternary MAC array             |
     | (rows × cols of: trit-mux +   |
     |  signed adder + accumulator)  |
     +-------------------------------+
                  |
                  v
         +-----------------+
         | Activation      |
         | (quantize back  |
         | to 8-bit)       |
         +-----------------+
```

The MAC unit is dramatically simpler than a binary one. A 1024×1024 ternary
MAC array would fit in the same silicon as a much smaller binary array,
yielding higher throughput per watt for LLM inference.

This is the most realistic near-term application of ternary hardware, and
why the Setun's design might come back from the dead.

---

## 4. An Operating System in Base 3

Most OS concepts translate directly. A few become more elegant.

### Memory addressing

Pages of `3^N` words instead of `2^N`. For our 9-trit machine:
- Page size: 27 words (3^3)
- 3-trit page offset, 6-trit page number
- Page table is itself 3-deep (radix tree of three-way branches)

A radix-3 page table is a natural fit for the trit-aligned address space.
Each level of the table has 27 entries (one per 3-trit field). Walking the
table is a sequence of 3-way dispatches.

### Process states

Binary OSes have many process states, but the core distinction is binary
flags ("ready or not"). Ternary lets you encode three-state status
naturally:
- `+` running
- `0` ready (waiting for CPU)
- `-` blocked (waiting for I/O)

A scheduler can make decisions on a single trit per process. The bookkeeping
is more compact.

### Permissions

Read / write / execute permissions in a Unix file are 3 bits per actor
(owner, group, world). Sound familiar? In ternary, encode permissions as 3
trits per actor where each trit is:
- `-` denied
- `0` inherited from parent / default
- `+` granted

The `0` value gives you genuine inheritance semantics that POSIX has to
emulate with extended ACLs.

### Synchronization primitives

Locks are intrinsically binary (held/not-held), but **ternary semaphores**
have a natural meaning:
- `+1` available
- `0` busy but no waiters
- `-1` busy with waiters (priority inheritance hint)

Atomic compare-and-swap becomes "compare-and-update-by-one-trit" on this
state. The state machine for lock acquisition simplifies.

### Filesystems

Directory structures are trees. Most filesystems are heavily binary
(B-trees, etc.). A ternary B-tree (B3-tree?) has fanout 3 instead of 2,
which is mathematically the optimal fanout for search trees if comparison
cost is roughly constant. Real B-trees have much higher fanout for cache
reasons, so this is mostly aesthetic, but the search algorithms are cleaner.

### What stays the same

- Scheduling algorithms (round-robin, priority, CFS)
- Virtual memory concepts
- Process isolation
- IPC (pipes, sockets, signals)
- Filesystem semantics

The OS's job is mostly the same. The data structures change shape.

### What feels different

The kernel itself, written for ternary hardware, would have ~30% less code
for arithmetic-heavy parts (no special-casing for sign, overflow, signed
vs unsigned division), but the same complexity for everything else (drivers,
networking, scheduling).

---

## Summary: What You'd Build End-to-End

**Hardware:** Memristor or three-level voltage logic. ALU built from MIN,
MAX, NEG, SUM gates. 9-trit or 18-trit word size.

**ISA:** ~40 opcodes, three-way branches as primitive, predicated execution
on trit values, no sign-bit weirdness, three-way compare returns trit.

**Float format:** 20-trit float, no sign bit, no bias, symmetric range,
trivially comparable as integers.

**Neural net path:** Ternary weights are first-class. Inference uses
trit-mux MAC arrays. 3-5× throughput / 5-10× energy improvement for LLMs.

**OS:** Radix-3 page tables, three-state process and lock primitives,
ternary permission inheritance.

**Software stack:** Compiler targeting ternary RISC, with a calling
convention that exploits 3-way returns (not just success/failure but
success/partial/failure).

The Setun (1958) had the logic right. What it lacked was a workload that
genuinely benefited from ternary. In 2026, ternary neural network weights
might be that workload.
