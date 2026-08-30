# GLOSSARY.md

Terms used precisely here and loosely almost everywhere else. Where a word can
be inflated into a stronger claim than the evidence supports, the wrong reading
is given too.

---

## The three objects

### Detector

Anything that reads a request or a response and emits a score. A linear probe
on question-time activations, a Presidio configuration, a regex-and-checksum
matcher. **A detector knows nothing about how good it is.**

Its **identity includes its configuration**. Two configurations are two
detectors with two ids, because a shared id would let a warrant measured on one
be quoted for the other.

### Envelope

A frozen, content-hashed evaluation set **plus a label definition**. Not just a
distribution: two sets can hold similar text and mean different things by
"positive", and the label category guard refuses to measure a detector against
labels that mean something else.

Changing an envelope creates a **different** envelope. It cannot inherit the
old one's warrants — that is the design, not an inconvenience.

### Warrant

A time-bounded, evidence-backed statement about what a detector's score is worth
on one distribution. Keyed by **(detector, operating point, envelope)** — never
by detector alone.

It carries measured metrics with intervals, the five control results, an issue
time and an expiry, and is either `VALID` or `REFUSED` with a reason naming
every failed criterion.

*Not* a quality badge. The analogy is a TLS certificate: issued by something
other than the server, bounded in time, revocable when the facts change — and
nobody has ever thought the certificate makes the server good.

---

## Warrant states

### `VALID`

Validated and cleared. Bounds hold on the named envelope until the expiry.

### `REFUSED`

Validated and failed. **A result, not an error**, and there is no override — no
flag, no environment variable, no admin bypass. If one existed the product would
be theatre, and a reviewer would look for exactly that.

### `UNVALIDATED`

Never tested on this distribution. **The modal state in production**, and it
must never collapse into either of the others. A system that cannot say "this
has never been measured here" will say something confident and wrong instead.

---

## Measurement vocabulary

### Yield vs rate — the single most important distinction here

- **Yield** — *"we surfaced N real errors this month"*. A **count** of reviewed,
  confirmed items. `EXACT`. Free. No sampling, no interval.
- **Rate** — *"we caught X% of errors"*. A claim about the traffic nobody
  reviewed. `ESTIMATED`. Costs labels. Always carries an interval.

Conflating them converts a free exact claim into an unbacked estimate and
nobody notices. Every metric is tagged with its kind, and construction fails if
the tag and the interval disagree.

### `EXACT` / `ESTIMATED`

The tag on a metric. `EXACT` refuses an interval; `ESTIMATED` requires one.
Both directions are enforced at construction, not by review.

### Free

A separate axis from `EXACT`/`ESTIMATED` — it is a statement about **label
cost**, not about certainty. Precision is free *and* estimated: free because the
flagged pool is reviewed anyway, estimated because that pool is a finite sample
and next month's precision is a forward claim.

### Estimand

The quantity a number is actually estimating. Naming it is what stops a
prevalence interval being reported as a recall interval — the two differ by a
derivative, and in this project's regime that factor is large.

### Operating point

A threshold plus the budget it was chosen to hit. Thresholds are selected on
**validation**; test is scored once, afterwards, and never consulted by a
selection.

### Flag rate — `f`

The fraction of traffic a detector flags. The **measured** test-set rate, never
the target the threshold aimed at. On test the realised rate differs from the
budget, and every downstream figure uses the realised one.

### Recall — `R`

Of the items that were actually positive, the fraction flagged. The positive
class is **incorrect** / **contains PII** depending on the envelope. Inverting
the polarity silently yields `1 − AUROC`, which reads as a strong negative
result and misdirects debugging for hours.

### Precision

Of the items flagged, the fraction that were actually positive. Reported
**separately** from recall, always. There is no blended F1 anywhere in this
codebase, and `Metric.__post_init__` refuses the name.

### AUROC

The probability that a randomly chosen positive is scored above a randomly
chosen negative. Base-rate independent, which makes it the **transferable**
quantity — a lift figure is specific to a workload's error rate; a ranking
quality is not.

### Lift, and its ceiling

`lift = R / f = precision / base_rate`. How many more positives you catch than
random sampling at the same budget.

Because precision ≤ 1, lift is **capped at `1 / base_rate`**. That ceiling is
reported beside every lift figure, because a given lift means something very
different at a low ceiling than at a high one. A warrant is refused when the
lift lower bound does not exceed 1.0 — when the evidence cannot show the
detector beats random sampling at the same budget.

### Conditional interval, and selection-aware widening

A published recall interval is **conditional on the threshold** — it treats as
fixed a quantity that was itself estimated, sometimes from very few validation
items. Propagating that selection noise widens the bound. Both numbers are
computed; the widening gets its own row rather than being buried.

### Bootstrap over questions

Resampling by `question_id`, not by row. Several items can share a question, so
a row-level bootstrap treats correlated items as independent and produces
intervals that are too narrow.

### Null band

The range within which a negative control's statistic is consistent with "no
signal". **Measured here, not looked up.** The closed form assumes exchangeable
scores under the null, and a fitted probe's scores are not — the fit induces
structure that survives label permutation, so each control simulates its own
null at construction and reports the band it applied.

### Power, against a declared tolerance

Whether the sample could have caught a deviation **that mattered** — not
whether it can separate the estimate from whatever happened to be observed. An
interval containing zero is not evidence of no effect; it is evidence the sample
could not resolve one.

### Single-class envelope

A set with only negatives (or only positives). AUROC, recall and precision are
**undefined** on it; the metric builder refuses to emit them rather than
emitting a plausible-looking 0.5, and the warrant claims a false-positive rate
only, with an exact binomial interval.

---

## The controls

### Control suite

Five checks run on every validation. Any failure refuses the warrant.

### `padding_fault`

Proves the activations came from real tokens, by building a **deliberately
right-padded** variant and requiring the check to *reject* it. Without that, a
tolerance loosened until it passed is indistinguishable from one that works.

### `label_shuffle`

AUROC must not survive permuting the labels.

### `null_feature`

A probe fitted on noise must score inside the measured null band.

### `canary`

A small, deliberately easy set built from the train split. A **regression
tripwire, not a measurement** — a high number on it is not evidence of anything;
a low one means something broke.

### `determinism`

Two runs at one seed must be bit-identical.

### Positive control

A check that is required to **fail** on a deliberately broken input. The
`padding_fault` control is one. It is what turns "the check passed" into "the
check still rejects the fault it exists for, on this hardware".

---

## Policy and decisions

### Profile

A named operating point with a declared floor — `customer_support`,
`internal_knowledge`, `decision_support`. The three are **three points on one
measured ROC**, not three invented thresholds: same detector, same envelope,
only the budget moves.

### Fail-closed

A policy bundle naming an operating point with no warrant behind it **does not
load**. Not a warning, not a default.

### Composition

Two warranted detectors, one decision. Bounds stay keyed per detector and are
**never merged** — two detectors agreeing is not a vote. A refusal is not
inherited. An unvalidated detector that fires triggers the profile default; one
that stays silent triggers nothing, because **silence from an unmeasured
detector is not evidence**.

### Trigger, not verdict

A detector's only output is a decision about where to spend an expensive check.
Nothing in this repository blocks, filters or gates a user-facing response.

### Certificate

The bounded, falsifiable assertion left behind by a decision — what was
checked, under which warrant, with which bounds.

### Ledger

An append-only, hash-chained SQLite store of every issuance, refusal,
revocation, override and decision. Generated and gitignored: the chain is only
meaningful for the run that built it.

### Override

A human decision on an escalated item. **Cannot be constructed without its
stratum and draw probability**, because a record missing those is unusable for
estimation and would silently bias any recall computed from the review queue.

---

## Drift

### Envelope violation

The live distribution has moved outside the one a warrant was measured on. It
is a property of the **input distribution**, so it invalidates every detector
measured on that distribution at once.

### PSI / MMD

The two distances the drift monitor computes over declared features. PSI's
0.10 / 0.25 bands are credit-scoring rules of thumb quoted without their sample
size and are **not scale-free** — the null grows with bin count and shrinks with
window size — so the monitor simulates its own null and refuses a configuration
whose false-alarm rate would exceed the declared limit.

### Revocation ladder

What happens after a violation: revoke → route to a detector that still holds a
warrant here, adopting **its** bounds and citing **its** warrant → if nothing
holds one at this floor, refuse and enqueue for re-measurement.

### Model-version invalidation

A changed model invalidates warrants measured on the old one. The weights are
part of what was measured.

---

## Reproduction

### Provenance block

Embedded in every artifact: UTC timestamp, git commit, branch, **dirty flag**,
library versions, device, seed, config hash, resolved config.

### Config hash

SHA-256 of the resolved configuration. Two artifacts carrying different hashes
do not describe one experiment.

### Claim table

The table in `README.md` naming, for every quantitative claim, the artifact and
the **field inside it**. Parsed and checked by `verify`; a number edited by hand
fails the build.

### Frozen scores

The committed per-item labels, scores and question ids behind each measured
block. Small enough to commit, which is what lets a fresh clone recompute every
metric with no GPU and no cache.

### The three verification tiers

Claim table → frozen scores → activations. Each proves something the previous
one cannot, and tier 3 reports SKIPPED rather than passing when the caches are
absent.

### Fixture

A synthetic artifact used to exercise the harness without a GPU. Fixture numbers
are **refused** by the results renderer rather than printed as measurements.

### External figure

A number about the world — a court award, a fine, a standard's clause number.
Not measured here, not derived here, not a declared estimate, so it needs a
register entry with a provenance tier before it may appear in the proposal.

---

**See also:** [ARCHITECTURE.md](ARCHITECTURE.md) · [METHODS.md](METHODS.md) for the derivations behind the statistical entries · [DIAGRAMS.md](DIAGRAMS.md) for the same vocabulary as pictures.
