# Getting this into Claude Code

A step-by-step guide for picking up this project in Claude Code on your
local machine. Two minutes total.

## Prerequisites

You'll need:
- Node.js 18+ (Claude Code requires it)
- Python 3.10+ (the simulator runs on stdlib only)
- An Anthropic API key OR a Claude Pro/Max subscription

If you don't have Claude Code installed yet:
```bash
npm install -g @anthropic-ai/claude-code
```

## Step 1: Create the project directory

```bash
mkdir -p ~/projects/ternary-computer
cd ~/projects/ternary-computer
```

Drop all the files from the download into that directory:
```
ternary-computer/
├── CLAUDE.md                       ← project memory (Claude Code reads this)
├── CONVERSATION_CONTEXT.md         ← distilled history of our discussion
├── DESIGN_NOTES.md                 ← deeper essays
├── README.md                       ← human-facing
├── SETUP.md                        ← this file
├── trit.py                         ← gates
├── arithmetic.py                   ← arithmetic from gates
├── cpu.py                          ← CPU
├── programs.py                     ← Fibonacci
├── demo.py                         ← master demo
├── binary_ops.py                   ← binary reference
├── ternary_ops.py                  ← instrumented ternary
├── benchmark.py                    ← head-to-head benchmark
├── benchmark_results.json          ← latest benchmark data
└── benchmark_visualization.html    ← interactive chart
```

## Step 2: Verify everything still works

Before involving Claude Code, confirm the simulator runs on your machine:

```bash
python3 demo.py
python3 benchmark.py
```

You should see Fibonacci output, factorial 1!..7!, and the gate-count
comparison table. If those work, you're good.

## Step 3: Launch Claude Code

From inside the project directory:
```bash
cd ~/projects/ternary-computer
claude
```

Claude Code will automatically read `CLAUDE.md` on launch — that's the
project memory file. It tells Claude what we built, what conventions we
used, and what the open directions are.

## Step 4: Bootstrap the conversation

For your first message, paste something like:

> I'm picking up a ternary computer simulator project we built together
> earlier. Read CLAUDE.md and CONVERSATION_CONTEXT.md first to get
> oriented, then let me know what you understand about the project's
> current state and what makes sense to work on next.

Claude Code will use the Read tool to ingest both files, then summarize
back to you and propose next steps. From there it has the same context
I have right now.

## Step 5: Pick a direction

The natural next moves, in rough order of teaching value (also listed
in CLAUDE.md):

1. **More benchmark workloads** — sorting, FFT, matrix multiply. Easy to
   add to `benchmark.py`, immediate visualization payoff.
2. **Wider word size** — bump WORD from 9 to 18 trits, re-run programs,
   see what gets cleaner.
3. **Ternary floating point** — implement the 20-trit ternary float from
   DESIGN_NOTES.md. Verify against Python floats.
4. **Tiny assembler** — read `.tasm` text files and emit encoded
   instructions. Programs become readable.
5. **Real BitNet inference** — pull a public BitNet b1.58 weight matrix,
   run inference on it via gate-counted ternary ops, compare to fp16.
   This would be the most defensible "ternary actually wins" result.
6. **Pipelining model** — add stage-level timing for throughput analysis.
7. **Live execution visualization** — trace a program trit-by-trit,
   show carries propagating through adders.

Just tell Claude Code which one interests you and it'll go.

## Tips for working in Claude Code

- **It can edit files directly.** No copy-paste; it'll modify `cpu.py`
  in place when you ask for changes. Use git to track diffs.
- **It runs your tests automatically.** When it changes `arithmetic.py`,
  it'll re-run `python3 arithmetic.py` to confirm self-tests still pass.
  Same for the other modules.
- **`/init` regenerates CLAUDE.md.** If the project grows significantly,
  ask Claude Code to update CLAUDE.md to reflect the new state. Or run
  `/init` to regenerate it from scratch.
- **Use `git init` early.** Initialize a git repo before significant
  changes so you can review diffs and roll back if needed:
  ```bash
  cd ~/projects/ternary-computer
  git init
  git add .
  git commit -m "Initial ternary computer simulator"
  ```
- **CLAUDE.md is project memory; ~/.claude/CLAUDE.md is global memory.**
  If you have personal coding preferences (snake_case vs camelCase,
  preferred test framework, etc.), put them in `~/.claude/CLAUDE.md`
  and Claude Code applies them across all your projects.

## If you want to skip Claude Code and use chat instead

You can also continue this conversation in any Claude chat by uploading
`CLAUDE.md` and `CONVERSATION_CONTEXT.md` as attachments and saying:

> Read both attached files. We're continuing a ternary computer simulator
> project. What's the current state and what's a sensible next step?

Same effect, no local install required. But Claude Code is much better
for actually editing the code, since it has direct file access and can
run tests itself.

## Common issues

**"Python version too old."**
The code uses `int | None` style union types (Python 3.10+). If you're
on 3.9, either upgrade or rewrite the type hints to use
`Optional[int]` from typing.

**"benchmark.py runs forever."**
It shouldn't. Each width pair takes ~3 seconds. If it hangs, you might
have hit a Python recursion or import issue — check that all `.py` files
are in the same directory.

**"Visualization doesn't render."**
The HTML file expects a modern browser. Open it directly with
`file://...` URLs; no server needed. If chart bars look broken, check
the browser console for JS errors.

**"CLAUDE.md says X but the code does Y."**
The code is authoritative. Tell Claude Code about the discrepancy and
ask it to update CLAUDE.md.
