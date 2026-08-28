# KICKOFF.md

Paste the block below as your first message to Claude Code, from the repo root.

---

```
Read CLAUDE.md, SPEC.md, TASKS.md, CONTRIBUTING.md, DECISIONS.md and DEMO.md
in full before writing any code. Then build this repo phase by phase per
TASKS.md.

Context: this is a measurement system for a competition submission that will be
judged as a public repository, with live Q&A in front of technical judges. The
product's entire thesis is that unbacked claims are the problem. That means a
number in this repo that can't be traced to the run that produced it is
self-refuting, not just sloppy.

Rules for how we work:

1. Stop at every phase gate in TASKS.md. Report using the protocol at the
   bottom of that file. Wait for my go-ahead before the next phase.
2. The invariants in CLAUDE.md are hard constraints. If something would require
   breaking one, stop and tell me rather than working around it.
3. Commit continuously per CONTRIBUTING.md -- four to ten atomic commits per
   phase, never one dump at the end. Docs change in the same commit as the code
   that required them. A phase with an uncommitted working tree has not passed
   its gate, whatever its numbers say.
4. Log anything methodological in DECISIONS.md before the gate. Statistical
   decisions carry their derivation in the entry, not a pointer to it.
5. Assert aggressively. The failure modes here are silent: wrong padding side,
   leaked splits, inverted polarity, a subtly biased estimator, sizing a
   prevalence interval and calling it recall. All produce plausible numbers with
   nothing raised. Crash instead.
6. Never write a number into a document by hand. If it isn't computed by code
   in this repo and traceable to results/, it doesn't go in.
7. Report surprises even when they look harmless.

Start with Phase 0. Before you write anything, tell me in your own words:
  (a) what a warrant is and why it's keyed by (detector, operating_point,
      eval_set) rather than by detector alone;
  (b) the difference between yield and rate, and why one is free;
  (c) why sizing a sample for recall is not the same as sizing it for
      prevalence, and roughly what factor separates them here.

If any of those three is wrong, everything downstream is wrong, and I'd rather
find out now.
```

---

## Between-phase prompts

**Advancing:** `Gate looks good. Proceed to Phase N.`

**Before Phase 6 (the estimator):**
```
Before you write the sampling code, show me your derivation of dR/dq at the
declared workload and the two sample sizes it implies. I want to see the
propagation, not the result.
```

**If it stops committing:**
```
Show me git log --oneline for this phase. I expect four to ten atomic commits,
not one. Split the working tree into logical commits now and keep committing as
you go.
```

**If a fix changes a number:**
```
That moved a measured value. Record before and after in the commit body, and
add a DECISIONS.md entry if the change was methodological rather than a bug fix.
If it was statistical, include the derivation.
```

**If it drifts into building things:**
```
Check the out-of-scope list in CLAUDE.md. Does this make the warrant more
credible? If not, it doesn't ship.
```

**If a control fails and it starts tuning:**
```
Don't tune. A failed control refuses the warrant -- that's the designed
behaviour, not a bug to route around. Tell me which control, its measured
margin, and what you think caused it.
```

**Final audit before anything reaches a slide:**
```
Run the Phase 12 documentation audit and report it line by line. Then walk me
through: where in the code is the test set opened, where is refusal enforced,
and show me there's no override path. Then list every number in README.md with
the file, script and config value that produced it.
```

---

## What to watch for

**The three states.** The most likely design regression is `UNVALIDATED` quietly collapsing into `REFUSED` (system unusable) or into `VALID` (the failure we're arguing against). Check the routing code specifically.

**The override.** The most likely engineering regression is an emergency bypass on warrant refusal. It will look reasonable when proposed. It ends the product. `test_no_override` exists for this; make sure it's behavioural and not just a grep.

**The sizing.** If the agent produces a sample size without stating the target quantity, it has probably sized prevalence and labelled it recall. That is the error this spec was written around.

**Scenario mixing.** Any economic figure quoted alongside another from a different flag rate or base rate. `test_no_scenario_mixing` catches it; make sure it actually runs.

**Beat 4 depending on late work.** The demo's centrepiece needs the matrix (Phase 4) and drift (Phase 5). Phase 2.5 exists so the runner grows alongside — don't let it get deferred.

---

## If results come back weak

**If mean-pool doesn't collapse on long context:** report it as measured. A matrix where every cell is `VALID` is a less dramatic demo and a perfectly honest finding. Do not manufacture a failure to fill a beat.

**If the tier gap is small:** that's a finding — it tells enterprises when weight access isn't worth paying for. Report it either way.

**If the estimator coverage test fails:** stop. A biased estimator producing confident intervals is the worst possible outcome for a product whose thesis is measurement integrity. Fix it or report the bias.

**If Presidio at stock actually performs well on Hinglish:** verify the config is genuinely default, then report the real number and drop Beat 3. Losing a beat is much cheaper than being caught having tilted a comparison.

---

## What the output feeds

- `results/RESULTS.md` → the deck's evidence slides
- The warrant matrix → Beat 4
- The price list → Beat 5's close
- The public repo, README and decisions log → the Round 2 deliverable itself, and likely read directly during the live solution discussion
