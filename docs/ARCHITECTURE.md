# ARCHITECTURE.md — what the system is and how the pieces fit

The one-sentence version: **a detector produces a score, a warrant says what
that score is worth on a named distribution, and policy reads the warrant
rather than the score.**

Everything below is a consequence of that separation.

---

## The three objects

### Detector

Anything that reads a request or a response and emits a score. A linear probe
on question-time activations, a Presidio configuration, a reference PII
matcher. A detector knows nothing about how good it is.

Its **identity includes its configuration**: `presidio-stock` and
`presidio-enabled` are different detectors, not one detector with a flag,
because they have different measured bounds and a shared id would let a warrant
measured on one be quoted for the other.

`controlplane/detectors/`

### Envelope

A frozen, content-hashed evaluation set **plus a label definition**. The hash
covers the items, the `data_source` and the construction notes, so a modified
set is a *different* set and cannot inherit the old one's warrants.

An envelope is not just a distribution. `hinglish-pii-200` and `triviaqa-600`
differ in what their labels *mean*, and the category guard refuses to measure a
PII detector against hallucination labels — recall against labels that mean
something else is not recall.

`controlplane/evalsets/`, `evalsets/`

### Warrant

A time-bounded, evidence-backed statement keyed by
**(detector, operating point, envelope)**. It carries measured metrics with
intervals, the five control results, an issue time and a 24-hour expiry, and
either `VALID` or `REFUSED` with a reason naming every failed criterion.

A warrant is issued by `issue_or_refuse`, which takes **no argument that could
promote a failing detector**. There is no `force`, no `override`, no
`min_confidence` to relax. A refusal cannot be relabelled downstream.

`controlplane/model/warrant.py`, `controlplane/validation/issuance.py`

---

## The flow

```
                  frozen eval set                 config.yaml
                  (envelope, hashed)              (one workload, one seed)
                         │                                │
                         ▼                                ▼
  model ──► extract ──► activations cache ──►  validation runner
  (GPU, once)           (results/cache-*.npz)        │
                                                     │  five controls
                                                     │  bootstrap intervals
                                                     ▼
                                            issue_or_refuse
                                                     │
                              ┌──────────────────────┴──────────────────────┐
                              ▼                                             ▼
                       VALID warrant                                  REFUSED
                       bounds + expiry                       reason names every
                              │                              failed criterion and
                              │                              the build it refused
                              ▼
                      warrant matrix  ◄──── drift monitor ────► revoke / downgrade
                    (detector × envelope)      PSI, MMD
                              │
                              ▼
                      policy bundle (Rego)
                      fails closed on a missing warrant
                              │
                              ▼
                        composed decision ──► certificate ──► ledger (hash chain)
```

### Stages

| stage | script | reads | writes |
|---|---|---|---|
| extract | `00_extract.py` | model, TriviaQA | `results/cache-*.npz` |
| build eval sets | `01_build_evalsets.py` | hand-written corpora | `evalsets/*.json`, manifest |
| validate | `02_validate.py` | a cache, an envelope | `results/validation-*.json` |
| matrix | `03_matrix.py` | every validation | `results/warrant_matrix.json`, `RESULTS.md` |
| transfer | `04_transfer.py` | two caches | `results/transfer-*.json` |
| canary | `05_canary.py` | canary cache | control evidence |
| reconcile | `06_reconcile.py` | Round 1 numbers | `results/reconciliation.json` |
| policy | `07_policy.py` | bundles, warrants | `results/policy-*.json` |
| paired | `08_paired.py` | two models, shared holdout | `results/paired_comparison.json` |
| detectors | `09_detectors.py` | PII envelopes | `results/detectors.json` |

Every stage writes to `results/` and every later stage reads from disk, so any
stage is independently re-runnable without repeating the GPU extraction.

---

## The five controls

Run on every validation. **Any failure refuses the warrant**, and nothing
promotes it back.

| control | what it proves | how it fails |
|---|---|---|
| `padding_fault` | activations came from real tokens | a right-padded variant is **accepted** rather than rejected |
| `label_shuffle` | the signal is not an artefact of the fitting | AUROC survives permutation |
| `null_feature` | the pipeline does not manufacture signal | a probe on noise scores outside the null band |
| `canary` | the detector detects the thing at all | recall below 1.0 on a deliberately easy set |
| `determinism` | the number is reproducible | two runs at one seed differ |

The null band is **measured, not assumed**: the Hanley–McNeil closed form
understates the true spread by 2.1x at d=32 because a fitted probe's scores are
not exchangeable, so the monitor simulates its own null at construction.
`DECISIONS.md` 029, 031, 070.

---

## Why policy reads the warrant and not the score

A threshold on a score is a number someone chose. A warrant is a number
someone measured, with an interval, on a named distribution, that expires.

The three profiles are **three points on one measured ROC**, not three invented
thresholds — same detector, same envelope, only `target_flag_rate` moves. Each
profile declares a floor, and the engine compares the floor against the
**interval bound**, not the point estimate. A bundle naming an operating point
with no warrant behind it **fails to load**; that is fail-closed, not a warning.

`controlplane/policy/`, `policies/*/bundle.yaml`

---

## Composition

Two warranted detectors, one decision. The rules were written before the code
(`DECISIONS.md` 088) and the four cases are enumerated in
[CASES.md](CASES.md) §2. The load-bearing ones:

- **bounds are keyed per detector and never merged.** Two detectors agreeing
  does not strengthen either bound; it is not a vote.
- **a refusal is not inherited.** One detector being refused does not
  invalidate the other's warrant; what was not checked is recorded instead.
- **an unvalidated detector that fires triggers the profile default**, not its
  own action — and one that stays silent triggers nothing, because silence from
  an unmeasured detector is not evidence.

`controlplane/policy/compose.py`

---

## Drift, and what happens after it

The monitor computes PSI and MMD over declared features (token length, script
mix, embedding centroid distance, reference perplexity, category mix) and
refuses a configuration whose false-alarm rate would exceed 5% — simulated at
construction, because PSI's 0.10/0.25 bands are credit-scoring rules of thumb
quoted without their sample size and are not scale-free.

Past the significant boundary, the ladder runs:

```
envelope violation → revoke → route to a detector that still holds a warrant
                            → adopt ITS bounds, cite ITS warrant
                            → if nothing holds one at this floor: REFUSE and enqueue
```

The measured version is the centrepiece. One long-context shift, three probe
aggregations, nothing retrained:

| detector | `triviaqa-600` | `triviaqa-longctx-600` |
|---|---|---|
| `T1-last_token` | 0.8256 [0.7934, 0.8567] | **VALID** 0.8135 [0.7797, 0.8447] |
| `T1-max_rolling_means` | 0.7853 | **REFUSED** 0.5553 [0.5105, 0.6015], flags 54% |
| `T1-mean_pool` | 0.7855 | **REFUSED** 0.5015 [0.4546, 0.5479], flags nothing |

Mean-pool at chance flags nothing, so a conventional dashboard reads it as
clean traffic. That is the failure this system exists to name.

`controlplane/drift/`

---

## The ledger

Every issuance, refusal, revocation, override and decision is appended to a
SQLite store with a hash chain over records. Retention is 400 days (DPDP Rule 6
minimum is one year). An override record **cannot be constructed without its
stratum and draw probability**, because a record missing those is unusable for
estimation and would silently bias any recall computed from the review queue.

The store is generated and gitignored: the chain is only meaningful for the run
that built it.

`controlplane/store/ledger.py`

---

## Where the invariants live

They are enforced at construction, not by convention.

| invariant | enforced by |
|---|---|
| no blended F1 anywhere | `Metric.__post_init__` refuses the name |
| every rate carries an interval | `Metric` refuses `ESTIMATED` without one |
| exact counts carry none | `Metric` refuses `EXACT` with one |
| a refusal states a reason | `Warrant.__post_init__` |
| the warrant key includes the envelope | `Warrant.compute_id` |
| left padding | asserted at load and again before every batched call |
| config invariants | `controlplane/config.py` `__post_init__`, naming each one |

A silent wrong answer is worse than a crash, so every boundary asserts.
