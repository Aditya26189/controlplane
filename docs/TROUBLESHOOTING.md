# TROUBLESHOOTING.md — what a failure means

This codebase crashes on purpose. A silent wrong answer is worse than a stack
trace, so most of what looks like a bug here is a boundary refusing to let a
misleading number through.

Read the exception's own type first: each one exists so a caller can tell *"your
input is wrong"* from *"the code is wrong"*.

**Contents:** [Refusals are not errors](#first-a-refusal-is-not-an-error) · [Exceptions](#the-exceptions-and-what-each-one-is-defending) · [Control failures](#when-a-control-fails) · [Verification](#when-verify-fails) · [Environment](#environment-problems) · [Not a bug](#things-that-look-broken-and-are-not)

---

## First: a refusal is not an error

`REFUSED` is a **result**, not a failure. It means the machinery ran correctly
and the evidence did not support a warrant. Three of the detectors this project
pointed itself at are refused, and those refusals are among the most useful
outputs in the repository.

Nothing promotes a refusal. There is no `force`, no `--override`, no
`min_confidence` to relax; `issue_or_refuse` takes no argument that could. If
you are looking for one, the thing to change is the evidence — a bigger
envelope, a better detector, a different operating point — not the call.

The refusal reason names **every** failed criterion, not the first one.

---

## The exceptions, and what each one is defending

| Exception | Raised when | What it is protecting |
|---|---|---|
| `ConfigError` | the config is missing, malformed, or self-inconsistent | Its own type so you can tell "your config is wrong" from "the code is wrong". Config invariants are checked in `__post_init__`, each naming the invariant it enforces |
| `PaddingError` | a tokenizer is not padding on the left | The highest-value check in the extraction path. With right padding, position −1 is a pad token, activations are meaningless, nothing else raises, and AUROC lands near chance |
| `ProbeError` | a probe would be fitted or scored in a way that leaks or lies | Train-only fitting, and selection that never touches test |
| `AggregationError` | a sequence cannot be pooled as asked | Silent reshaping would change what the detector *is* while keeping its id |
| `MetricError` | a metric would misrepresent what is known about it | An `ESTIMATED` rate without an interval, an `EXACT` count with one, or a blended F1 — none can be constructed |
| `MeasurementError` | a statistic cannot be computed on the data given | e.g. quantities that are undefined on a single-class envelope, refused rather than emitted as a plausible-looking 0.5 |
| `WarrantError` | a warrant would claim more than its evidence supports | A refusal with no reason, or a key missing its envelope |
| `EvalSetError` | an eval set or its cache is inconsistent | A content hash that does not match, or a cache that is not the one the set names |
| `LabelCategoryError` | a detector is pointed at a set whose labels mean something else | Recall against labels that mean something else is not recall — a PII detector measured against hallucination labels |
| `BundleError` | a policy bundle is malformed or internally inconsistent | A deploy error rather than a rule silently falling through to its default |
| `WarrantResolutionError` | a well-formed bundle asks for a claim the matrix cannot back | **Never a warning.** The bundle does not load. This is the fail-closed path |
| `LedgerError` | an append would break the ledger's guarantees | The hash chain |
| `OverrideError` | an override record was written that the estimator could not use | A record missing its stratum or draw probability would silently bias any recall computed from the review queue |
| `UnclassifiedEntityError` | a detector emitted an entity the adapter has no classification for | Raised rather than filtered — quietly dropping an entity type changes measured recall |
| `DraftDivergedError` | the prompts about to be scored are not the prompts that were frozen | The pilot's pre-registration. Scoring a changed draft against a frozen decision rule is the thing pre-registration exists to prevent |
| `CertificateError` `FindingError` `SerdeError` | a record type would be built in violation of its own contract | Records are frozen dataclasses validated at construction |

---

## When a control fails

Any of the five failing refuses the warrant. What each failure actually tells
you:

| Control failed | Read it as |
|---|---|
| `padding_fault` | Either the padding really is wrong, **or** the check no longer discriminates. The control deliberately builds a right-padded variant and requires it to be *rejected*; if that variant is accepted, the tolerance is worthless and the run stops |
| `label_shuffle` | AUROC survived permuting the labels. The signal is an artefact of the fitting, not of the data |
| `null_feature` | A probe on noise scored outside the null band — the pipeline is manufacturing signal. Note the band is *simulated at construction*, not looked up; the closed form understates the spread here |
| `canary` | The detector missed items on a deliberately easy set. A regression tripwire, so treat it as "something broke", not as a measurement |
| `determinism` | Two runs at one seed differed. Look for unseeded randomness, dict ordering, or a device-dependent kernel before anything else |

---

## When `verify` fails

`verify.py` prints the tier, the artifact, the field, and both values. The fix
is never "edit the number until it matches".

| Tier | Failure means |
|---|---|
| 1 — claim table | A number in `README.md` no longer matches the artifact and field it names. Either the artifact was regenerated and the README was not, or someone typed a number by hand |
| 2 — frozen scores | A metrics block no longer follows from the per-item scores behind it. This is the case tier 1 cannot see: a README and its artifacts that went stale *together* |
| 3 — activations | The frozen scores no longer re-derive from the cached activations |
| 3 reports SKIPPED | **Not a failure.** The caches are gitignored, so a fresh clone cannot run this tier. The final line names any tier that did not run, so a skip is never mistaken for a pass |

If tier 1 fails after you regenerated an artifact on purpose: re-run the stage,
then update the README row, then re-run `verify`. The claim table is the
contract, and it is parsed rather than trusted.

---

## Environment problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on a controlplane import | not installed, or the wrong interpreter | `pip install -r requirements.lock.txt` inside the venv you are actually running |
| A Presidio result differs from the published one | `presidio-analyzer` is not at the pinned version | The finding is a statement *about that version*. Reinstall from `requirements.lock.txt`; a test fails when the environment drifts from the pin |
| Rego evaluation unavailable | `regopy` missing | It is a pip wheel, not an OPA binary. Install the requirements |
| CUDA out of memory during extraction | 7B plus a batch does not fit | Lower `--batch-size` (8 → 4). Do not disable quantisation on a 16 GB card |
| A stage cannot find a cache | extraction has not run here, or `--cache-dir` differs | Caches are gitignored by design. Either run `00_extract.py` or use the CPU tiers, which do not need them |
| No network on the machine that runs extraction | offline node | Pre-download into `HF_HOME`, then set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. [SETUP.md](SETUP.md) §3 |
| A test suite reads as green but something failed | you piped the command | `cmd \| tail` reports *tail's* exit status. Use `sh scripts/run.sh <cmd>` |
| `dirty: true` in a fresh artifact | the tree was not clean when the script ran | Commit, re-run, then commit the artifacts separately. The dirty flag excludes `results/` and nothing else |
| Two artifacts disagree about one number | they were produced under different config hashes | Compare the `provenance.config_hash` fields. Re-run the whole chain before quoting anything |

---

## Things that look broken and are not

- **Most of the warrant matrix is `UNVALIDATED`.** That is the expected shape.
  `UNVALIDATED` is the modal state in any real deployment, and a system that
  cannot say *"never measured here"* will say something confident and wrong
  instead.
- **`results/RESULTS.md` refuses to print some numbers.** Several matrix cells
  are synthetic fixtures. The renderer refuses them rather than printing a
  fixture as if it were a measurement.
- **A recall interval looks wide.** It is a bootstrap-percentile interval over
  questions, and several are additionally reported with a selection-aware
  widening. A narrow interval on a small holdout would be the suspicious thing.
- **`verify` tier 3 says SKIPPED on a clean clone.** Correct — never a pass it
  did not earn.
- **A detector is refused on one envelope and valid on another.** That is the
  entire point of keying warrants by envelope.
- **An eval set changed and its warrants vanished.** Sets are content-hashed; a
  modified set is a *different* set and cannot inherit the old one's evidence.

---

**See also:** [SETUP.md](SETUP.md) · [RUNBOOK.md](RUNBOOK.md) · [CASES.md](CASES.md), which names the test covering each of these behaviours.
