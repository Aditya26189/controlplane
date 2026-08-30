# METHODS.md — estimators, bootstraps, bands, and where they came from

Every number in [README.md](../README.md) is produced by one of the estimators
below. This says which, why, and what each one assumes.

The recurring theme: **the closed forms were wrong here, and the fix was to
measure the null rather than to look it up.** Three separate times.

---

## 1. Resampling

**Bootstrap-percentile, 1000 resamples for validation runs, 2000 for the
paired comparison, seed 1729, resampled over `question_id` rather than over
rows.**

Resampling rows would break the group structure: TriviaQA ships several items
per question, so a row-level bootstrap treats correlated items as independent
and produces intervals that are too narrow. Every split in this project is by
question, deduplicated on the normalised question string first, with zero
overlap asserted between splits.

Percentile rather than BCa: BCa's acceleration term is estimated by jackknife,
which at n=600 over groups is itself noisy, and the correction was smaller than
its own uncertainty when checked.

`controlplane/validation/stats.py`

---

## 2. What carries an interval, and what does not

Two axes that are easy to conflate.

| | `EXACT` | `ESTIMATED` |
|---|---|---|
| means | a count of reviewed, confirmed items | a rate, i.e. a claim about a process |
| interval | **refused** — construction fails if one is supplied | **required** — construction fails without one |
| examples | `confirmed_errors` | `auroc`, `recall`, `precision`, `flag_rate` |

**Free** is a third and separate axis: it is a statement about *label cost* and
lives in the price list, not in the type. Precision is free *and* estimated —
free because the flagged pool is reviewed anyway, estimated because the flagged
pool is a finite sample of the traffic distribution and next month's precision
is a forward claim.

`DECISIONS.md` 022. `Metric.__post_init__` enforces both directions.

---

## 3. The negative-control band, measured three times

A negative control asserts *"AUROC is consistent with 0.5"*. Whether an
observed value is consistent with 0.5 depends on sampling noise, so a **fixed**
`[0.45, 0.55]` band is only a valid test at one holdout size.

**First attempt** — scale the band by the Hanley–McNeil null standard error.
That model said 5 repeats made the band a ±3.45 SE bar at n=600. Label-shuffle
then failed at 0.5546 on the second run. Three-sigma events do not happen on
the second run, so **the model was wrong.**

**Why it was wrong.** Hanley–McNeil assumes exchangeable scores under the null.
A *fitted* probe's scores are not: the fit itself induces structure that
survives label permutation. Measured over 200 permutations per variant, the
true spread exceeded the closed-form SE by **2.1x at d=32 and 8.3x at d=1**.

**What is used.** Each control simulates its own null at construction —
between 8 and 200 repeats, sized from the *measured* spread — and reports the
band it actually applied and why. A ±2 SE floor is retained as a backstop for
holdouts too small for repeats to rescue; it only ever widens, never tightens.

`DECISIONS.md` 029, 031.

---

## 4. The drift monitor's false-alarm rate, same story

PSI's 0.10 / 0.25 bands are credit-scoring rules of thumb quoted without their
sample size, and they are not scale-free: the null grows as `(k-1)/n`, so the
same band means different things at different bin counts and window sizes.

The monitor simulates its own null at construction and **refuses a
configuration** whose STABLE-window false-alarm rate would exceed 5%. A
threshold that fires on 30% of stable windows is not a monitor; it is noise
with a dashboard.

`DECISIONS.md` 070. `controlplane/drift/psi.py`, `null_band.py`

---

## 5. Lift, and its ceiling

```
lift = R / f = precision / base_rate
```

`f` is always the **measured** test-set flag rate, never the target the
threshold was aimed at. On test the realised rate differs from the budget, and
every downstream figure uses the realised one.

Because precision ≤ 1, lift is capped at `1 / base_rate`. That ceiling is
reported beside every lift figure, because a lift of 1.9 means something very
different at a ceiling of 2.17 (88% of attainable) than at a ceiling of 10.

A warrant is refused when the **lift lower bound does not exceed 1.0** — that
is, when the evidence cannot show the detector beats random sampling at the
same budget. `DECISIONS.md` 043.

---

## 6. Threshold selection, and the interval that ignores it

Thresholds are selected on **validation** to hit a declared flag-rate budget,
and reported on **test**. Test is never consulted by any selection.

The published recall interval is **conditional on that threshold** — it treats
as fixed a quantity that was itself estimated, sometimes from very few items.
At the `customer_support` budget the threshold is determined by **5 negatives**
in a 480-item validation split.

Propagating the selection noise widens the recall bound by **1.36x to 1.59x**.
Both are computed and both are in `results/paired_comparison.json`; the claim
table quotes the conditional interval, which is what the artifact's `metrics`
block holds, and gives the widening its own row rather than burying it.

`DECISIONS.md` 083. See [LIMITATIONS.md](LIMITATIONS.md) §2.2.

---

## 7. Comparing two models: pair on what both held out

The trap, and this project fell into it once and published the result.

Commit `67167ed` compared AUROC 0.8256 on a 600-item test set against 0.8232 on
a 960-item one and concluded the training reduction "cost nothing measurable".
**Training size and evaluation sample both changed between those numbers**, so
the difference is unattributable and the narrower interval is a test-n effect
that says nothing about training size.

The correct comparison pairs the two models on the **600 items both held out**
and bootstraps the *difference*, resampling the pairing. It found the opposite:
the reduction cost **0.0110 AUROC [0.0026, 0.0200]**, an interval excluding
zero.

The three warranted recalls were **underpowered** at that n — minimum
detectable difference 0.034 to 0.043 — and are reported as underpowered rather
than as no difference. An interval containing zero is not evidence of no
effect; it is evidence the sample could not resolve one.

`DECISIONS.md` 081. `controlplane/validation/paired.py`

---

## 8. Power, computed against what matters

Power is measured against the **declared tolerance**, not against whatever
deviation happened to be observed. An estimate landing near its target needs an
enormous n to be distinguished from it — which says nothing about whether the
sample could have caught a deviation that mattered.

At a 25% calibration tolerance, separating a real deviation from the budget
needs **n ≥ 1441**. Measured at n = 600 to 960, so every budget claim is
refused or unresolved, and each says which.

`DECISIONS.md` 069.

---

## 9. Single-class envelopes

`hard-negatives-200` is all negatives. AUROC, recall and precision are
**undefined** on it, and the warrant claims an FPR only — with an exact
binomial interval, since the bootstrap of a zero-variance statistic is
degenerate. `pii-reference` measures FPR 0.0000 [0.0000, 0.0183] there.

The metric builder detects the single-class case and refuses to emit the
undefined quantities rather than emitting a plausible-looking 0.5.

---

## 10. Labels

**Correctness**: TriviaQA alias matching against greedy generations. Aliases
shorter than 3 characters require an exact token match rather than substring
containment — a gold alias of `"US"` appears inside thousands of unrelated
generations, and substring matching would inflate the correct rate silently.

**PII**: construction-time. The identifiers are synthetic and the label is
known when the item is built, so there is no annotation noise — and equally, no
independent check that the labels mean what we think.

**No human labelling was done.** Cohen's κ, double-labelling and the blinded
review queue are specified in [SPEC.md](SPEC.md) and are not built; see
[LIMITATIONS.md](LIMITATIONS.md) §3.

---

## 11. Reproducibility

`random`, `numpy` and `torch` are seeded from `config.yaml` (seed 1729). Every
artifact records the resolved config, a SHA-256 config hash, the git commit, a
dirty flag, library versions and the device.

The `determinism` control asserts two runs at one seed are bit-identical, and
`make verify` re-derives the frozen-set validation from cached activations and
compares field by field — all three probe variants currently reproduce
bit-identically.

The **dirty flag** excludes `results/` and nothing else, because the pipeline
writes its own artifacts there and a bare `git status --porcelain` would mark
every stage after the first as dirty regardless of the code.
