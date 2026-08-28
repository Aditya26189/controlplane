# KICKOFF.md

Paste the block below as your first message to Claude Code, from the repo root.

---

```
Read CLAUDE.md, SPEC.md, TASKS.md, CONTRIBUTING.md, and DECISIONS.md in full
before writing any code. Then build this repo stage by stage per TASKS.md.

Context: this is a measurement experiment, not a product. It exists to produce
one defensible number — lift = R/f, the probe's recall divided by its flag rate
— for a competition submission with a deadline in days. A weak result honestly
reported is a valid outcome. A strong result produced by selecting on the test
set is worthless and worse than nothing.

Rules for how we work:

1. Stop at every stage gate in TASKS.md. Report using the protocol at the
   bottom of that file. Wait for my go-ahead before starting the next stage.
2. The invariants in CLAUDE.md are hard constraints. If something would require
   breaking one, stop and tell me instead of working around it.
3. Stage 3 is the expensive one (~1 hour of GPU). Do not start the full
   extraction until the left-padding equivalence check, the n=20 smoke run, and
   the base-rate check have all passed and you have shown me the results.
4. Assert aggressively at every boundary. The failure modes here are silent —
   wrong padding side, leaked splits, inverted label polarity all produce
   plausible-looking numbers with nothing raised. Crash instead.
5. Report surprises even when they look harmless.
6. Commit continuously per CONTRIBUTING.md — four to ten atomic commits per
   stage, never one dump at the end. Docs change in the same commit as the code
   that required them. A stage with an uncommitted working tree has not passed
   its gate, whatever its numbers say.
7. Log anything methodological in DECISIONS.md before the gate. That file is
   the answer to every "why did you do it that way?" a judge can ask, and this
   repo gets submitted publicly — the commit history is part of the deliverable.

Start with Stage 0. Before you write anything, tell me your understanding of
what the probe is measuring and why the test set can only be opened once — if
that's wrong, everything downstream is wrong and I'd rather find out now.
```

---

## Between-stage prompts

**Advancing:** `Gate looks good. Proceed to Stage N.`

**Before the expensive run:**
```
Show me the equivalence check max deviation, the 20 sample completions, and the
base rate on those 20 before you start the full extraction.
```

**If AUROC comes back weak:**
```
Don't tune. Work the checklist in Stage 4 in order: polarity, then the
equivalence check, then layer range, then n_examples. Report what each one
shows. If it's still at or below 0.55 after all four, we write it up as a
negative result — that's a legitimate outcome and I'd rather publish it than
manufacture a number.
```

**If it drifts into building things:**
```
Check the out-of-scope list in CLAUDE.md. We need the number, not a system.
```

**If it stops committing:**
```
Show me git log --oneline for this stage. I expect four to ten atomic commits,
not one. Split what's in the working tree into logical commits now, and keep
committing as you go for the rest of the build.
```

**If a fix changes a number:**
```
That moved a measured value. Record the before and after in the commit body,
and add a DECISIONS.md entry if the change was methodological rather than a
plain bug fix.
```

**Final check before you use anything in the deck:**
```
Audit every number in results/RESULTS.md against the artifact that produced it.
For each one tell me: which file, which script, which config value. Flag any
number you can't trace. Then confirm the test set was scored exactly once by
walking me through where in the code that happens.
```

---

## What to watch for

**The single highest-value check in the build is the left-padding equivalence test in Stage 3.** With right padding, position `-1` is a pad token and every activation is meaningless — but nothing errors, the pipeline completes, and you get an AUROC near 0.5 that reads as "the idea doesn't work." Do not let the agent skip it or downgrade it to a warning.

**Watch for test-set selection.** The most common way an agent produces an inflated number is choosing the layer or threshold on test because it's one fewer line of code. Ask it to walk you through where the test set is touched. There should be exactly one place.

**Watch for polarity.** Positive class is *incorrect*. Inverted, you get `1 - AUROC`, which looks like a strong negative result and can send you debugging in the wrong direction for an hour.

## What to do with the output

- `results/RESULTS.md` → the measured `R` and `f` go on Slide 3A. Replace the placeholder in the three-policy table with the real lift.
- The notebook → screen-record for the 2:02–2:28 segment of the video.
- The repo, public and MIT → most of the Round 2 prototype deliverable.

If lift comes in below about 5×, the honest pitch changes shape: lead with the architecture and the coverage argument, present the measurement as a first result with the limitations stated, and say generalisation is the Round 2 question. That is still a stronger submission than most teams will have, because it is measured. Do not round a weak number up.
