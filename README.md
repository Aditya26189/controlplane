# ControlPlane

**A detector produces a score. A warrant is a separate, time-bounded,
evidence-backed statement about what that score is worth on this distribution
right now.** Everyone ships detectors. Almost nobody ships the second thing —
so a guardrail that has quietly stopped working looks exactly like one that
works, and the dashboard stays green either way. ControlPlane measures each
detector on a named evaluation envelope, issues a warrant with bounds and an
expiry when the evidence supports it, and **refuses one when it does not**;
policy then reads the warrant rather than the score. It is the same idea as a
TLS certificate: issued by something other than the server, bounded in time,
revocable when the facts change — and nobody has ever thought the certificate
makes the server good.

---

## Quickstart

Under five minutes on a laptop. No GPU, no network after the clone.

```bash
git clone https://github.com/Aditya26189/controlplane.git
cd controlplane
git fetch origin "refs/notes/*:refs/notes/*"     # see "Reading the history" below
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt
make verify
```

`make verify` re-runs the frozen-set evaluation from cached activations and
prints every number in the claim table below beside the value measured on your
machine. It exits non-zero if any of them drift.

---

## What this is not

Stating the scope is worth more than having it inferred.

- **Not deployed.** No auth, no rate limiting, no HA, no deployment manifests.
  There is no serving layer and adding one is explicitly out of scope.
- **Not a verdict.** A detector is a trigger for spending an expensive check.
  Nothing here blocks, filters or gates a user-facing response.
- **Not a general result.** Measured on 2,400 TriviaQA items and hand-built PII
  sets, one model family (Qwen2.5-7B-Instruct, NF4), no real production traffic.
- **Not fully built.** Phase 6 economics and Phase 9's action gate were
  specified and never written; five contract documents still cite a
  `controlplane/economics/sizing.py` that does not exist. Logged as
  `DECISIONS.md` 096, declared in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
- **Not a claim about truthfulness.** The probe is a correlational classifier
  over activations. It does not measure what a model believes.

---

## The claim table

Every quantitative claim this repository makes, the artifact that contains it,
and the field inside that artifact. `make verify` parses this table, resolves
each field, and compares at the precision quoted. **A number edited by hand
here fails the build.**

All intervals are 95% bootstrap-percentile over questions, seed 1729.

| Claim | Value | Interval | Artifact | Field | Regenerate |
|---|---|---|---|---|---|
| Probe AUROC, `triviaqa-600`, n=600 | 0.8256 | [0.7934, 0.8567] | results/validation-T1-last_token.json | metrics.auroc.value | make verify |
| Probe recall at its operating point | 0.0794 | [0.0496, 0.1115] | results/validation-T1-last_token.json | metrics.recall.value | make verify |
| Probe precision there | 0.8800 | [0.7500, 1.0000] | results/validation-T1-last_token.json | metrics.precision.value | make verify |
| Measured flag rate there | 0.0417 | [0.0267, 0.0583] | results/validation-T1-last_token.json | metrics.flag_rate.value | make verify |
| Base error rate of that envelope | 0.4617 | - | results/validation-T1-last_token.json | base_rate | make verify |
| Last-token probe survives the long-context shift | 0.8135 | [0.7797, 0.8447] | results/transfer-T1-last_token.json | target.metrics.auroc.value | make verify |
| Mean-pool probe collapses to chance under it | 0.5015 | [0.4546, 0.5479] | results/transfer-T1-mean_pool.json | target.metrics.auroc.value | make verify |
| Max-rolling-means probe under it | 0.5553 | [0.5105, 0.6015] | results/transfer-T1-max_rolling_means.json | target.metrics.auroc.value | make verify |
| ...and it flags 54% of the stream | 0.5433 | [0.5050, 0.5833] | results/transfer-T1-max_rolling_means.json | target.metrics.flag_rate.value | make verify |
| `customer_support` recall, n=960 | 0.2171 | [0.1800, 0.2564] | results/policy-triviaqa-2400-t960.json | operating_points[operating_point.operating_point_id=P-customer-support].metrics.recall.value | make verify |
| `customer_support` measured flag rate | 0.1062 | [0.0885, 0.1260] | results/policy-triviaqa-2400-t960.json | operating_points[operating_point.operating_point_id=P-customer-support].metrics.flag_rate.value | make verify |
| `internal_knowledge` recall, n=960 | 0.3603 | [0.3173, 0.4063] | results/policy-triviaqa-2400-t960.json | operating_points[operating_point.operating_point_id=P-internal-knowledge].metrics.recall.value | make verify |
| `decision_support` recall, n=960 | 0.7367 | [0.6974, 0.7783] | results/policy-triviaqa-2400-t960.json | operating_points[operating_point.operating_point_id=P-decision-support].metrics.recall.value | make verify |
| Cost of the training reduction, paired on 600 held-out items | -0.0110 | [-0.0200, -0.0026] | results/paired_comparison.json | pinned_to_baseline_threshold[quantity=auroc].difference | make verify |
| Selection-aware widening of the recall bound | 1.5922 | - | results/paired_comparison.json | selection_aware_bounds[operating_point_id=P-customer-support].widening | make verify |
| Round 1 reproduced on Round 2's pipeline | 0.8256 | - | results/reconciliation.json | round2.T1-last_token.test_auroc | make verify |
| ...landing inside Round 1's published interval | A | - | results/reconciliation.json | branch | make verify |
| `presidio-stock` recall on Hinglish PII, REFUSED | 0.1176 | [0.0500, 0.2000] | results/detectors.json | runs[detector_id=presidio-stock,eval_set_id=hinglish-pii-200].metrics.recall.value | make verify |
| `presidio-enabled` recall, REFUSED | 0.2843 | [0.2075, 0.3810] | results/detectors.json | runs[detector_id=presidio-enabled,eval_set_id=hinglish-pii-200].metrics.recall.value | make verify |
| `presidio-enabled_plus_custom` recall, VALID | 0.6176 | [0.5185, 0.7128] | results/detectors.json | runs[detector_id=presidio-enabled_plus_custom,eval_set_id=hinglish-pii-200].metrics.recall.value | make verify |
| Our `pii-reference` recall, VALID | 0.7941 | [0.6981, 0.8846] | results/detectors.json | runs[detector_id=pii-reference,eval_set_id=hinglish-pii-200].metrics.recall.value | make verify |
| `presidio-stock` reproduces out of sample | 0.1471 | [0.0714, 0.2341] | results/holdout/detectors.json | runs[detector_id=presidio-stock,eval_set_id=hinglish-pii-200b].metrics.recall.value | make verify |
| `pii-reference` false-positive rate on benign traffic | 0.0000 | [0.0000, 0.0183] | results/validation-pii-reference-hard-negatives-200.json | metrics.flag_rate.value | make verify |
| Warrant matrix cells VALID | 13 | - | results/warrant_matrix.json | matrix.summary.VALID | make verify |
| Warrant matrix cells REFUSED | 4 | - | results/warrant_matrix.json | matrix.summary.REFUSED | make verify |
| Warrant matrix cells never measured | 39 | - | results/warrant_matrix.json | matrix.summary.UNVALIDATED | make verify |

**How to read the last three rows.** 39 of 56 cells are UNVALIDATED. That is
not a gap in the work — it is the expected shape of the thing. UNVALIDATED is
the modal state in any real deployment, and a system that cannot say *"this
detector has never been measured on this distribution"* will instead say
something confident and wrong.

**Read the probe recall with its base rate.** 0.0794 recall at a 4.2% flag rate
is a `lift` of roughly 1.9x over random sampling at the same budget, and the
ceiling at base rate 0.4617 is 2.17x. The headline is not "the probe is
accurate". It is that the number, its interval and its ceiling are all on the
record, and that the same machinery **refuses** three of the detectors it was
pointed at.

---

## Repo map

| path | what is in it |
|---|---|
| `controlplane/` | the package — model, store, validation, matrix, drift, policy, detectors, report |
| `evalsets/` `results/` | frozen content-hashed evaluation sets; every artifact behind the table above |
| `scripts/` `demo/` `notebooks/` | thin CLI wrappers, the two-pane demo runner, the Kaggle GPU notebook |
| `tests/` | 505 tests, including the ones that enforce this README |
| `docs/` | spec, methods, limitations, the case matrix, and the move mapping |
| `DECISIONS.md` `round1/` | 96 append-only decision entries; the Round 1 submission, moved whole |

---

## Reproduction

| target | without make | requires | time | proves |
|---|---|---|---|---|
| `make smoke` | `python scripts/smoke.py` | CPU, no network | < 60s | the clone works and the package imports |
| `make test` | `python -m pytest tests/ -q` | CPU | ~10 min | 505 tests green |
| `make verify` | `python scripts/verify.py` | CPU, cached activations | ~2 min | **every number in the claim table reproduces** |
| `make extract` | `python scripts/00_extract.py --config config.yaml` | GPU, 16 GB | ~1 h | activations regenerate from the source model |

Every recipe in the Makefile is a single command with no shell logic in it, so
the middle column is exact rather than approximate. If you are on Windows
without make, use it directly.

`make verify` is the "prove it" button as a command line, and it does two things:

1. **The claim table against the committed artifacts.** Resolves every field
   named above and compares at the quoted precision. Always runs.
2. **The committed artifacts against a re-run from cached activations.** This is
   the stronger check — step 1 would still pass if the artifacts and the README
   were stale *together*. The extraction cache is ~78 MB and gitignored, so on a
   fresh clone this step reports **SKIPPED**. It never reports a pass it did not
   earn.

Both exit non-zero on drift. Nothing in `results/` is written by either: the
re-run goes to a scratch directory, so a failed verification cannot damage the
evidence it was checking.

Nobody will run `make extract`. It is documented anyway, so the chain from raw
model to published number has no gap in it. The tested GPU path is
[notebooks/run_on_kaggle.ipynb](notebooks/run_on_kaggle.ipynb), which is
generated by `scripts/build_notebooks.py` and never hand-edited.

---

## Reading the history

Run this once after cloning:

```bash
git fetch origin "refs/notes/*:refs/notes/*"
```

Commit `67167ed` states that reducing the training set from 1200 to 960 items
"cost nothing measurable". **That claim is wrong**, and a git note attached to
that commit says so: the comparison was confounded, because the training size
and the evaluation sample both changed between the two numbers. The paired
comparison on the 600 items both models held out found the opposite — the
reduction cost 0.0110 AUROC [0.0026, 0.0200], an interval that excludes zero.

The history was not rewritten. A withdrawn claim that stays visible next to its
correction is worth more than a clean log, and the correction is only visible if
you fetch the notes ref — which is why this is in the setup section rather than
in a footnote. `DECISIONS.md` 081 carries the full argument.

---

## Limitations and open items

Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before quoting anything here.
The ones that would change how you read the table:

- **Calibration drift is detectable at 25%, not at 10%.** Separating a 25%
  deviation from the declared flag-rate budget needs n ≥ 1441; these envelopes
  were measured at n = 600 to 960. Every budget claim is refused or unresolved.
- **Every published recall interval was conditional on a threshold set by as
  few as five validation items.** The selection-aware bounds are 1.36x to 1.59x
  wider (`DECISIONS.md` 083). The table above quotes the conditional interval
  and names the widening in its own row.
- **The `enabled_plus_custom` recognizers were fitted on the set they are
  measured on.** 11 of 34 patterns are spec-derived; the rest are fitted, and
  the out-of-sample holdout is underpowered (`DECISIONS.md` 086, 087).
- **`DECISIONS.md` 080 is unresolved**, and the D.2 measured-pair gap is open.
- **8 of 17 populated matrix cells are synthetic fixtures.** Their numbers are
  refused rather than printed — see the warning at the top of
  [results/RESULTS.md](results/RESULTS.md).
- **The Presidio finding is a statement about `presidio-analyzer==2.2.364`.**
  It is pinned, the version is inside the refusal reason, and a test fails if
  the environment drifts from the pin.

---

## Documents

| document | what it answers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | what the system is and how the pieces fit |
| [docs/METHODS.md](docs/METHODS.md) | estimators, bootstraps, null bands and their derivations |
| [docs/CASES.md](docs/CASES.md) | every case, the test covering it, the artifact demonstrating it |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | scope, declared gaps, open items |
| [DECISIONS.md](DECISIONS.md) | 96 append-only entries — "why did you do it that way?" |
| [docs/PATHS.md](docs/PATHS.md) | what moved on 2026-08-29 and where it went |
| [docs/SPEC.md](docs/SPEC.md) | the technical specification |
| [docs/PROPOSAL.md](docs/PROPOSAL.md) | the business proposal |

MIT licensed. Every dependency is MIT, Apache-2.0 or BSD; the stack is open and
self-hostable, which is why Llama Guard and ShieldGemma are deliberately absent.
