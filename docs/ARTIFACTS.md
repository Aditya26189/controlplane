# ARTIFACTS.md — every output file, and what is inside it

`results/` and `evalsets/` are the evidence. This page says what each file is,
which stage wrote it, and which fields matter — so that following a number from
the claim table to the thing that produced it takes one lookup rather than a
search.

No values are quoted here. The claim table in [../README.md](../README.md) names
the artifact **and the field** for every published number, and `make verify`
resolves all of them.

**Contents:** [Provenance](#every-artifact-carries-provenance) · [Validation](#validation-artifacts) · [Matrix and policy](#matrix-and-policy) · [Comparisons](#comparisons-and-transfers) · [Detectors](#detector-artifacts) · [Pilot](#the-banking-pilot) · [Frozen scores](#frozen-scores--the-evidence-a-clean-clone-checks) · [Eval sets](#evalsets) · [Not committed](#what-is-deliberately-not-committed)

---

## Every artifact carries provenance

Each JSON file opens with a `provenance` block:

| Field | Why it is there |
|---|---|
| `timestamp_utc` | when |
| `git_commit`, `git_branch` | which code |
| `dirty`, `dirty_paths` | whether the tree was clean. A `dirty: true` artifact records a commit that does not describe the code that ran |
| `config_hash` | SHA-256 of the resolved config. **Two artifacts with different hashes do not describe one experiment** |
| `seed`, `config` | the full resolved configuration, inlined |
| `libraries`, `device`, `python` | the environment, without which several numbers are not interpretable |

That block is what turns a published number into something a stranger can check.
Compare `config_hash` across artifacts before quoting two of them in one
sentence.

---

## Validation artifacts

`results/validation-<detector>-<envelope>.json` — one per (detector, envelope)
measured. Written by `02_validate.py` and `09_detectors.py`.

| Field | Contains |
|---|---|
| `run_id`, `detector_id`, `variant`, `eval_set_id`, `envelope_id` | what was measured, and on what |
| `data_source` | where the items came from |
| `splits`, `base_rate`, `test_scored` | the split sizes, the envelope's base rate, and the record that test was scored |
| `probe_fit` | fitting details for probe detectors |
| `operating_point` | the threshold and the budget it was selected to hit |
| `metrics` | AUROC, recall, precision, flag rate — **each with `value`, its interval, and its `EXACT`/`ESTIMATED` kind** |
| `controls` | all five, each with pass/fail and the band actually applied |
| `warrant` | the issued warrant, or the refusal |
| `warrant_status`, `status_reason` | `VALID` / `REFUSED`, and the reason naming **every** failed criterion |

Present in the repository: the three probe aggregations on the TriviaQA
envelope, and the reference PII detector on the Hinglish, long-context and
hard-negative envelopes.

A single-class envelope — `hard-negatives-200` is all negatives — carries a
false-positive rate with an exact binomial interval, and the metrics builder
**refuses** to emit AUROC, recall or precision there rather than emitting a
plausible-looking 0.5.

---

## Matrix and policy

| File | Written by | Contains |
|---|---|---|
| `warrant_matrix.json` | `03_matrix.py` | `matrix` — every (detector × envelope) cell with its state, plus a `summary` counting VALID / REFUSED / UNVALIDATED; and `routing`, which detector to fall back to per envelope |
| `warrant_matrix.md` | `03_matrix.py` | the same, rendered |
| `RESULTS.md` | `03_matrix.py` | the results narrative, with **fixture numbers refused rather than printed** |
| `policy-<envelope>.json` | `07_policy.py` | `operating_points` — one block per profile, each with its own metrics and warrant; and `comparison` across the three |
| `tier_ladder.json` / `.png` | `02_validate.py` | what each level of model access buys, with intervals |
| `feasibility.json` | `11_feasibility.py` | `abstention_floor` per target risk, `profiles` with achieved risk and efficiency, `measured` and `declared` inputs kept separate, and a `not_derived_here` block naming what it deliberately does not claim |

`policy-*.json` fields are addressed in the claim table by a filter, e.g.
`operating_points[operating_point.operating_point_id=P-customer-support].metrics.recall.value`.
That syntax is resolved by `controlplane/report/claims.py`, so a row naming a
field that no longer exists fails the build rather than going stale.

---

## Comparisons and transfers

| File | Written by | Contains |
|---|---|---|
| `transfer-<detector>.json` | `04_transfer.py` | `source` and `target` blocks — the same fitted detector measured on the envelope it was fitted on and on the shifted one, nothing retrained |
| `paired_comparison.json` | `08_paired.py` | `split_relationship` (why the two are comparable at all), `thresholds`, `pinned_to_baseline_threshold`, `each_at_its_own_threshold`, `roc`, and `selection_aware_bounds` with the widening factor per operating point |
| `reconciliation.json` | `06_reconcile.py` | `round1` and `round2` numbers, the `branch` taken, `branch_meaning`, and `preregistered_in` — the decision entry that fixed the branches **before** the number was visible |

`preregistered_in` appears in several artifacts. It is the field that separates a
decision rule from a post-hoc rationalisation, and it is worth checking first
when reading any of them.

---

## Detector artifacts

| File | Contains |
|---|---|
| `detectors.json` | `runs` — one per (detector, eval set), each with metrics, controls and warrant status. This is where the Presidio configurations and the reference detector live |
| `holdout/detectors.json` | the same detectors on the held-out variant set — the out-of-sample check |
| `presidio_coverage.json` | which entity types the pinned Presidio version can recognise **at all**, produced by running the analyzer rather than by asserting it, plus `interpretation` and `uncovered` |
| `evalset_validation.json` | detector-versus-set validation runs used while building the sets |
| `evalset_build.json` | what was built, with each set's hash |

---

## The banking pilot

Four artifacts, each naming the decision entry that pre-registered it.

| File | Contains |
|---|---|
| `pilot_envelope.json` | the frozen prompts' distance from the fitted envelope, with `units` and `interpretation` |
| `pilot_run.json` | the pilot pass: `probe_fit`, `generation`, `acceptance_band`, `saturation`, `auroc`, the `branch` taken and what `decided_by` it |
| `pilot_null_band.json` | the null band and the power behind the routing decision, including whether the effective `n` is clusters rather than items |
| `pilot_seed_stability.json` | how often the gate clears across bootstrap seeds — `stability`, `verdict`, `why` |

The last one exists because **a gate that clears on the seed you happened to use
is not a gate**. Read `pilot_seed_stability.json` beside `pilot_run.json` or you
are reading half the result.

---

## Frozen scores — the evidence a clean clone checks

`results/scores/*.json`, written by `10_freeze_scores.py`, **committed**.

One file per measured (detector, envelope) block, holding the per-item labels,
scores and question ids. A few hundred kilobytes in total, which is the whole
point: tier 2 of `make verify` recomputes every metrics block from these using
the same builder, bootstrap count and seed, **on a fresh clone with no GPU and
no cache**.

If you want to check this project's arithmetic yourself, this directory is where
to start.

---

## `evalsets/`

Frozen, content-hashed evaluation sets plus `manifest.json` registering them.

| Set | Is |
|---|---|
| `triviaqa-600`, `triviaqa-2400-t960` | the TriviaQA envelopes |
| `hinglish-pii-200`, `hinglish-pii-200b` | hand-written Hinglish PII scenarios, and the held-out variant |
| `hard-negatives-200` | boundary cases that must be **allowed** — all negatives, so it measures false positives only |
| `canary-20-*` | the regression tripwires, built from train splits |
| `banking-dual-24` | the dual-labelled pilot set, with its `.draft` kept beside it |

The hash covers the items, the data source and the construction notes. **A
modified set is a different set** and cannot inherit the old one's warrants —
that is the design, and `DraftDivergedError` exists to enforce the pilot's
version of it.

---

## What is deliberately not committed

| Not committed | Why |
|---|---|
| `results/cache-*.npz` | ~100 MB each. Regenerable by `00_extract.py`; the frozen scores are the checkable evidence instead |
| the SQLite ledger | generated; the hash chain is only meaningful for the run that built it |
| `results/fixtures/*` values in prose | present as files, but the results renderer refuses to print fixture numbers as if they were measurements |

The rule behind all three: **never stage a file over 10 MB**, and never publish a
number whose provenance you cannot show.

---

**See also:** [RUNBOOK.md](RUNBOOK.md) for which stage writes each file · [METHODS.md](METHODS.md) for how the values inside them were computed · [TESTING.md](TESTING.md) for what stops them going stale.
