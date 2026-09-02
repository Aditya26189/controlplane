# FAQ.md

<sub>[🏠 Project README](../README.md) · [📚 Documentation index](README.md) · [🗺️ Diagrams](DIAGRAMS.md) · [📖 Glossary](GLOSSARY.md)</sub>

The questions a technical reviewer actually asks, each answered with the
artifact that settles it. **No numbers are quoted here** — they live in
`README.md`'s claim table and in `results/`, which is where they should be
checked.

For the derivations, read [METHODS.md](METHODS.md). For what is missing, read
[LIMITATIONS.md](LIMITATIONS.md) before quoting anything.

---

## What this is

**In one sentence, what is the product?**

A detector produces a score; a warrant is a separate, time-bounded,
evidence-backed statement about what that score is worth on this distribution
right now. Everyone ships detectors. Almost nobody ships the second thing — so a
guardrail that has quietly stopped working looks exactly like one that works.

**Isn't this just an evaluation harness with extra steps?**

An evaluation harness tells you a number once. A warrant is keyed by
**(detector, operating point, envelope)**, expires, is revoked when the input
distribution moves, and is the thing policy actually reads. The difference shows
up in the cases a harness has no answer for: what happens when traffic drifts,
what happens when a detector was never measured on this distribution, what
happens when a bundle names an operating point nothing has warranted.

**Why does the analogy keep coming back to TLS certificates?**

Because the useful properties are the same ones: issued by something other than
the server, bounded in time, revocable when the facts change. And because nobody
has ever thought the certificate makes the server good — which is exactly the
misreading a warrant has to survive.

**Is this deployed anywhere?**

No. No auth, no rate limiting, no HA, no deployment manifests, no serving layer,
and adding one is out of scope. It is a measurement system with a demo.

---

## The evidence

**How do I know the numbers in the README are real?**

Run `make verify`. It parses the claim table, resolves every field in the
artifact each row names, and compares at the quoted precision — then recomputes
every metrics block from the frozen per-item scores in `results/scores/`. Both
run on a fresh clone with no GPU. A number edited by hand fails the build.

**What can `verify` not catch?**

Tier 1 alone cannot catch a README and its artifacts that went stale *together*
— which is why tier 2 exists. Tier 2 cannot prove those scores came from the
model and probe the artifact names — which is why tier 3 exists, and why it
reports **SKIPPED** rather than passing when the gitignored caches are absent.
The final line names any tier that did not run.

**Why commit scores rather than the activations?**

The caches are ~100 MB each; the frozen scores are a few hundred kilobytes. The
scores are what let a stranger recompute every published metric on a laptop.
Committing the caches would put the check behind a download most reviewers will
not do.

**Was the test set used to choose anything?**

No selection consults test. Thresholds and regularisation are chosen on
validation; test is scored once per validation run, afterwards. The artifact
records `test_scored`, and the splits are by question with zero overlap
asserted.

**How were the intervals computed?**

Bootstrap-percentile, resampled over `question_id` rather than over rows,
seeded. Row-level resampling would treat correlated items from one question as
independent and produce intervals that are too narrow. Percentile rather than
BCa because BCa's acceleration term is itself noisy at these sample sizes.
[METHODS.md](METHODS.md) §1.

---

## The uncomfortable questions

**Three of your own detectors are refused. Isn't that a bad result?**

It is the result. The machinery that refuses a detector is the product; a system
that never refuses anything has not demonstrated that it can. The refusals name
the version, the envelope and every failed criterion, and one of them
reproduces out of sample.

**Most of your warrant matrix is `UNVALIDATED`. Isn't that a gap?**

It is the expected shape. `UNVALIDATED` is the modal state in any real
deployment, and the alternative — collapsing "never measured here" into either
"fine" or "bad" — is the failure the whole project is about.

**Your headline detector's recall is low.**

Yes, and it is published with its base rate, its measured flag rate and the
ceiling that `1 / base_rate` imposes on any selector at all. A lift figure means
something different at a low ceiling than at a high one, which is why the
ceiling is never separated from the number.

**Some of your recall intervals were conditional on a threshold set by a handful
of items.**

Correct, and it is stated rather than discovered. The selection-aware widening
is computed and published as its own row; the claim table quotes the conditional
interval because that is what the artifact's metrics block holds, and names the
widening beside it. [LIMITATIONS.md](LIMITATIONS.md) §2.2.

**Some of your PII recognizers were fitted on the set they are measured on.**

Also stated. The out-of-sample holdout exists but is underpowered, and both
facts are in the limitations rather than in a footnote.

**Was anything published and later found wrong?**

Yes. One commit claimed a training-set reduction "cost nothing measurable". The
comparison was confounded — training size and evaluation sample both changed —
and the correct paired comparison found the opposite. **The history was not
rewritten.** A git note attached to that commit carries the correction; fetch
`refs/notes/*` after cloning to see it. `DECISIONS.md` 081.

**Your external-figure register once overclaimed its own provenance.**

It did, and the correction is at the top of
[EXTERNAL_FIGURES.md](EXTERNAL_FIGURES.md). A register built to stop numbers
arriving with their provenance stripped had a provenance column asserting a tier
nobody had reached. That is the same defect class the register exists to
prevent, occurring inside the guard — which is why it is written up rather than
quietly fixed.

---

## Method

**Why a linear probe and not a stronger model?**

Because the claim being tested is about what the representation already
contains, and because the probe has to be cheap enough to run on all traffic —
it is a trigger for spending an expensive check, not the check.

**Why read activations before generation?**

The signal is available before a token has been paid for, which makes the same
score usable for routing as well as monitoring. It is strictly less information
than an output-reading detector, and that trade is deliberate.

**Why does one probe aggregation collapse under long context?**

That is the measurement, not an accident: the same fitted probe, three
aggregations, one long-context shift, nothing retrained. Mean-pooling dilutes a
local signal across a long sequence. The alarming part is not that it collapses
— it is that it collapses to chance **while flagging almost nothing**, which a
conventional dashboard reads as clean traffic.

**Why is there no F1 anywhere?**

F1 blends two failure modes whose costs differ by orders of magnitude and hides
which one you have. `Metric.__post_init__` refuses the name, so it cannot be
added by accident.

**Why simulate the null bands instead of using the closed forms?**

Because the closed forms were wrong here, three separate times. Hanley–McNeil
assumes exchangeable scores under the null, and a fitted probe's are not; PSI's
0.10/0.25 bands are credit-scoring rules of thumb quoted without their sample
size and are not scale-free. Each control and the drift monitor simulate their
own null at construction and report the band applied. [METHODS.md](METHODS.md)
§3–4.

**How do you know the activations were not read off padding?**

Every validation runs a control that builds a **deliberately right-padded**
variant and requires the check to reject it. A check that only ever passes is
indistinguishable from a tolerance loosened until it passed.

---

## Scope and cost

**Can this work on a model I only reach through a vendor API?**

The activation-probe tier cannot — it needs the weights. The warrant machinery
itself does not care what the detector is: the PII detectors it warrants are
plain text matchers with no model access at all. Say which tier you mean before
someone else says it for you.

**Where do the cost and ROI numbers come from?**

Every cost, headcount or ROI figure in this repository is a **declared
estimate** and says so. The price list was specified and never built. The one
exception is the feasibility bound, which derives from measured rates and needs
no cost model — it is an impossibility result about any selector, not a claim
about ours.

**What is the feasibility bound, in one line?**

Given a measured base error rate, holding residual risk at a target requires
abstaining on at least some fraction of traffic — for *any* selector, however
good. It is why "just tighten the threshold" is not an available answer.

**What would you build next?**

The two specified-and-unbuilt phases, in order of what they unblock: the
on-traffic warrant with stratified estimation and its price list, then the
action gate. Both are described in [SPEC.md](SPEC.md) and their absence is
declared in [LIMITATIONS.md](LIMITATIONS.md) §3.

---

## Working on it

**Where do I start reading?**

[ONBOARDING.md](ONBOARDING.md) — one hour, in order. Then
[DIAGRAMS.md](DIAGRAMS.md) and [CODE_TOUR.md](CODE_TOUR.md).

**Something crashed. Is that a bug?**

Often it is a boundary refusing to let a misleading number through.
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) lists every exception type and what
each one defends.

**A number I need is wrong. Can I edit it?**

Not in a document. Every number is computed by code and traced to an artifact;
if a number is wrong, the pipeline is wrong. Fix the pipeline, regenerate, and
let `verify` confirm it.

**Why is there no `run_all.py` when three documents mention one?**

Because it does not exist. The four `make` targets are the real entry points and
`clean_clone_gate.py` is what exercises the clean-clone reproduction those
documents describe. The drift is recorded in [RUNBOOK.md](RUNBOOK.md) rather
than papered over.
