# RUNBOOK.md — every script, what it reads, what it writes

<sub>[🏠 Project README](../README.md) · [📚 Documentation index](README.md) · [🗺️ Diagrams](DIAGRAMS.md) · [📖 Glossary](GLOSSARY.md)</sub>

`scripts/` holds thin CLI wrappers. They parse arguments, call
`controlplane/`, and write files; **no logic lives in them**, which is why this
page can describe them by their inputs and outputs without lying.

Every script takes `--config config.yaml` unless noted. Every artifact it
writes carries a `provenance` block — config hash, git commit, dirty flag,
library versions, device, timestamp.

**Contents:** [Entry points](#the-entry-points) · [The measurement chain](#the-measurement-chain) · [The banking pilot](#the-banking-pilot) · [Maintenance](#maintenance-scripts) · [Order of operations](#order-of-operations) · [Rules](#rules-that-apply-to-every-run) · [Contract drift](#contract-drift-found-during-this-documentation-pass--now-reconciled)

---

## The entry points

Four commands cover almost everything a reader needs.

| Command | What it is for |
|---|---|
| `python scripts/smoke.py` | Did this clone come down intact? Under a minute, no network |
| `python scripts/verify.py` | Does every published number still reproduce? Three tiers, weakest first |
| `python -m pytest tests/ -q` | Is the suite green? |
| `python scripts/00_extract.py` | The GPU stage. Everything else runs from what it caches |

`scripts/verify.py` takes `--claims-only` (tier 1 alone, seconds), `--readme`
(check a different README path) and `--eval-set`.

`scripts/run.sh` is not a stage — it is the wrapper you should use when you
need a command's real exit status. `cmd | tail` reports *tail's* status, so a
failing suite reads as green; `sh scripts/run.sh <cmd>` writes the output to a
file, echoes `EXIT=<real status>`, and propagates it.

---

## The measurement chain

Run in this order on a fresh machine. Each one reads what the previous wrote.

### `00_extract.py` — the only GPU stage

Extracts TriviaQA question-time activations for both envelopes and self-checks
the output.

- **reads** the model and TriviaQA, per `config.yaml`
- **writes** `results/cache-*.npz` (gitignored, ~100 MB each)
- **flags** `--batch-size`, `--max-new-tokens`, `--cache-dir`, `--out`
- **when** only when the model, the prompt, the layer set or the aggregation
  changes. The caches are the expensive thing in this repository

### `01_build_evalsets.py` — freeze the evaluation sets

Builds the hand-written corpora into frozen, content-hashed sets and registers
them in a manifest.

- **writes** `evalsets/*.json`, `evalsets/manifest.json`,
  `results/evalset_build.json`
- **when** a corpus changes. Note that this produces a *different set*, with a
  different hash and therefore no inherited warrants — that is the design

### `02_validate.py` — one detector, one operating point, one envelope

Runs `/validate` and the tier ablation, then issues or refuses a warrant.

- **reads** a cache and a frozen envelope
- **writes** `results/validation-*.json`, `results/tier_ladder.json`
- **this is where** the five controls run, the threshold is frozen on
  validation, and test is scored once

### `03_matrix.py` — populate the warrant matrix

Runs the ablation across every activation-tier envelope plus the text
detectors, and renders the matrix.

- **writes** `results/warrant_matrix.json`, `results/warrant_matrix.md`,
  `results/RESULTS.md`
- **note** the renderer refuses to print fixture numbers as if they were
  measured

### `04_transfer.py` — the long-context shift

Fits on one envelope, evaluates on another, with nothing retrained.

- **writes** `results/transfer-*.json`, one per probe aggregation
- **flags** `--source-cache`, `--target-cache`, `--source-eval-set`,
  `--target-eval-set`, `--target-flag-rate`
- **why it matters** this is the measurement where one aggregation holds its
  warrant and another collapses to chance while flagging almost nothing — the
  failure a conventional dashboard reads as clean traffic

### `05_canary.py` — freeze the regression tripwire

Builds a small, deliberately easy set from the **train split only**.

- **writes** a canary eval set and its control evidence
- **flags** `--cache`, `--eval-set`, `--variant`, `--n-items`,
  `--target-flag-rate`, `--canary-id`, `--evalsets-out`
- **read this before using it** the canary is a tripwire, not a measurement.
  Recall below 1.0 on it fails a control; a high number on it is not evidence
  of anything

### `06_reconcile.py` — Round 1 against Round 2's pipeline

Classifies the result against branches that were **pre-registered before the
number was visible**.

- **writes** `results/reconciliation.json`, including which branch was taken
  and what that branch means

### `07_policy.py` — the three profiles

Issues the three operating points, loads the three bundles, and decides one
input under all three.

- **reads** `policies/*/bundle.yaml`, the warrant matrix
- **writes** `results/policy-*.json`
- **expect** a `BundleError` if a bundle names an operating point that nothing
  has warranted. That is fail-closed behaviour, not a bug

### `08_paired.py` — comparing two probes properly

Pairs two models on the items **both** held out and bootstraps the difference.

- **writes** `results/paired_comparison.json`, including the selection-aware
  widening of the recall bounds
- **flags** `--baseline-set`, `--variant-set`, `--variant`, `--bootstrap`
- **why** the naive comparison changed the training size *and* the evaluation
  sample at once, which makes the difference unattributable.
  [METHODS.md](METHODS.md) §7

### `09_detectors.py` — the PII detectors

Validates each Presidio configuration and our reference detector, then issues
or refuses each warrant.

- **writes** `results/detectors.json`, `results/holdout/detectors.json`
- **flags** `--eval-set`, `--hard-negatives`, `--skip-reference`
- **note** each configuration is a *separate detector id*, because a shared id
  would let a warrant measured on one be quoted for another

### `10_freeze_scores.py` — the evidence a stranger can check

Freezes the per-item labels, scores and question ids behind every measured
block.

- **writes** `results/scores/*.json` — small, and **committed**
- **why** this is what makes tier 2 of `verify` possible on a fresh clone with
  no GPU and no cache

### `11_feasibility.py` — the bound that is not about our detector

Derives the abstention floor and the review sizing from what was measured.

- **reads** a policy artifact (`--policy-artifact`)
- **writes** `results/feasibility.json`, including a `not_derived_here` block
  naming what it deliberately does not claim

---

## The banking pilot

Four scripts, run in order, each pre-registered in `DECISIONS.md` before it ran.

| Script | Does |
|---|---|
| `12_pilot_freeze.py` | Freezes the pilot prompts and measures their distance from the fitted envelope |
| `13_pilot_run.py` | The GPU pass: generate, judge, extract, score. Flags `--reference`, `--variant`, `--evalsets-dir`, `--batch-size` |
| `14_pilot_null_band.py` | Regenerates the IQR-ratio null band and its power — the numbers the decision routes on |
| `15_pilot_seed_stability.py` | Prices the margin: how often does the gate clear across seeds? Flags `--seeds`, `--resamples` |

The last one exists because a gate that clears on the seed you happened to use
is not a gate. Its artifact records the verdict *and* how much of the
bootstrap-seed distribution agrees with it.

`17_presidio_coverage.py` is separate: it records which entity types stock
Presidio can recognise at all, by running the analyzer rather than by asserting
it. A claim about a dependency is checkable by running the dependency.

---

## Maintenance scripts

| Script | Does |
|---|---|
| `build_notebooks.py` | Regenerates `notebooks/run_on_kaggle.ipynb`. **The script is the source of truth**; never hand-edit the notebook |
| `clean_clone_gate.py` | Clones this repository into a temporary directory and runs the gates inside it, so "it works on my machine" cannot be the reason it passes |
| `kaggle_run.py` | Pushes the extraction notebook to Kaggle as a batch kernel and collects the output. Flags `--yes`, `--accelerator`, `--timeout`, `--poll-seconds` |
| `run.sh` | Runs a command and reports its **real** exit status |

---

## Order of operations

```
00_extract ──► 01_build_evalsets ──► 02_validate ──► 03_matrix
                                          │
                                          ├──► 04_transfer
                                          ├──► 05_canary
                                          ├──► 06_reconcile
                                          ├──► 07_policy ──► 11_feasibility
                                          ├──► 08_paired
                                          └──► 09_detectors
                                                   │
                                                   ▼
                                            10_freeze_scores
                                                   │
                                                   ▼
                                              verify.py
```

`10_freeze_scores.py` runs last because it freezes what everything else
produced. Run `verify.py` after it, and again on a clean clone before
publishing anything.

---

## Rules that apply to every run

1. **Commit first.** A dirty tree makes the artifact's recorded commit false.
   `provenance()` sets `dirty: true` and it stays in the file.
2. **One declared workload.** Every economic figure derives from the single
   `workload` block in `config.yaml`. Mixing a flag rate from one scenario with
   a base rate from another is silently wrong.
3. **Never pipe a command whose status you need.** Use `sh scripts/run.sh`.
4. **A stage never recomputes an earlier stage's metric.** Later stages read
   the earlier artifact, so two documents cannot disagree about one number.
5. **Nothing promotes a refusal.** No script takes a flag that could.

---

## Contract drift found during this documentation pass — now reconciled

`CLAUDE.md` ("Definition of done"), `CONTRIBUTING.md` (pre-merge checklist) and
`TASKS.md` (Phase 12 gate) all referred to **`scripts/run_all.py`**, which has
never existed in this repository. The four `make` targets are the actual entry
points, and `clean_clone_gate.py` is what exercises the clean-clone
reproduction those three documents describe.

All three now name what ships. The drift is recorded here rather than silently
corrected, because a contract that quietly changed is a contract nobody can
trust to have meant something yesterday — and because the interesting part is
not the name, it is that three binding documents agreed with each other and
with nothing on disk. A cited path that no test resolves is a claim like any
other.

---

**See also:** [SETUP.md](SETUP.md) to get the environment right · [ARTIFACTS.md](ARTIFACTS.md) for what each output file contains · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for what a failure means.
