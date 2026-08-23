# Architecture

How this repository is put together, and where each rule in `CLAUDE.md` is actually enforced in code.

This document is **hand-written and contains no measured numbers** — those live only in generated files (`results/RESULTS.md`, `README.md`, `docs/HANDOVER.md`). If you want a number, follow the pointers at the end of each section.

**Contents:** [Two architectures](#two-architectures-dont-confuse-them) · [The pipeline](#the-pipeline) · [Artifact contracts](#artifact-contracts) · [Module map](#module-map) · [Where the invariants live](#where-the-invariants-live) · [Design rules](#design-rules-that-shape-the-code) · [Changing things](#changing-things) · [Deliberately absent](#deliberately-absent)

---

## Two architectures, don't confuse them

The competition concept and this repository are not the same system, and conflating them is the fastest way to overclaim.

**1. The product concept — a four-tier cascade** (Accenture Innovation Challenge, Problem Statement 1). Every response passes cheap checks; expensive ones are rationed:

| Tier | What it is | Traffic | Authority |
|---|---|---|---|
| 0 | Deterministic checks — PII and secret detection, schema and regex validation | 100% | can block |
| 1 | Cheap signals — logprob confidence, self-consistency, token-accounting anomalies | 100% | flags only |
| 2 | **Linear probe on question-time activations** | 100% | flags only, ~free |
| 3 | Expensive verdict — LLM-as-judge, semantic entropy, claim attribution | a few % | full authority |

**2. This repository — Tier 2 only, measured.** It answers one question: *is the Tier 2 signal good enough to decide who goes to Tier 3?* Nothing here serves traffic, gates a response, or implements Tiers 0, 1 or 3. There is no gateway, no API, no policy engine — those are explicitly out of scope in `CLAUDE.md`.

The output of the whole repository is one number and its confidence interval:

```
lift = R / f      recall divided by measured flag rate
                  = how many more errors you catch than random sampling
                    at the same Tier 3 budget
```

Everything in the repo either produces that number, validates it, or documents it.

---

## The pipeline

Five stages, each a **separate process**. A stage writes to `results/` and the next stage reads from disk — never from memory. That is what makes stage 02 onwards re-runnable in about a minute without repeating the GPU-hour extraction.

```
config.yaml ──────────── read by every stage, hashed into every artifact
     │
     ▼
┌─ 01_extract.py ────────────────────────────────────── GPU, ~2h at n=3000 ─┐
│  load TriviaQA rc.nocontext                                               │
│  normalise questions, deduplicate, drop empties                           │
│  split 60/20/20 BY question_id, assert disjoint                           │
│  load model NF4 + tokenizer, assert padding_side == "left"                │
│  left-padding equivalence check + right-padding positive control          │
│  for each batch (sorted by length, order restored):                       │
│      pass 1: forward(output_hidden_states=True)  -> activations           │
│      pass 2: generate(greedy)                    -> completions           │
│  label: normalised alias match (lenient) + strict EM                      │
│  base-rate gate: abort if outside config band                             │
└───────────────────────────────────────────────────────────────────────────┘
     │  splits.parquet · data_stats.json · activations.npz
     │  labels.parquet · extract_meta.json
     ▼
┌─ 02_train_probe.py ──────────────────────────────── CPU, seconds ─────────┐
│  assert polarity (y=1 means the answer was WRONG)                         │
│  sweep layer x C, fit on TRAIN, score on VALIDATION                       │
│  select layer and C on validation; flag if the winner is at a grid edge   │
│  freeze threshold on validation at the target flag rate                   │
│  ── open the test set (one place in the codebase) ──                      │
│  AUROC, precision, recall, confusion, bootstrap CIs, abstention, ROC      │
│  append the scoring to the append-only test log                           │
└───────────────────────────────────────────────────────────────────────────┘
     │  probe_sweep.json · probe.joblib · probe_test.json · test_scoring_log.json
     ▼
┌─ 03_economics.py ─── three policies, lift, ceiling, projections ──────────┐
│  reads probe_test.json only; recomputes no metric, so lift cannot drift   │
└───────────────────────────────────────────────────────────────────────────┘
     │  economics.json
     ▼
┌─ 04_latency.py ───── probe cost vs the generation it rides on ────────────┐
│  probe timed on a real activation vector, one at a time                   │
│  generation and prefill times come from extract_meta.json, not a rerun    │
└───────────────────────────────────────────────────────────────────────────┘
     │  latency.json
     ▼
┌─ 05_report.py ────── formatting only, no computation ─────────────────────┐
│  RESULTS.md (SPEC.md §13 order) · README.md from README_TEMPLATE.md       │
│  layer_sweep.png · roc_curve.png                                          │
└───────────────────────────────────────────────────────────────────────────┘
     │
     └─ 06_handover.py ── docs/HANDOVER.md, orientation for someone joining cold
```

`scripts/run_all.py` chains 01→05. `--smoke` runs the whole chain at `n_examples=100` into `results/smoke/`, so a smoke run can never overwrite a real one. `--from 02` skips extraction.

### Why two forward passes

Stage 01 runs prefill and generation as separate calls on purpose. Passing `output_hidden_states=True` into `generate()` retains hidden states for *every* decode step and exhausts a 16GB GPU. One extra prefill against ~32 decode steps is a rounding error in runtime (`SPEC.md` §4).

The prefill is also the honest denominator for the latency claim: the activation is a by-product of a forward pass the model was going to do anyway, so the probe adds **no additional pass**.

---

## Artifact contracts

Every artifact is JSON or parquet with a `provenance` block: UTC timestamp, git commit, branch, dirty flag and dirty paths, Python and library versions, device, config hash, seed, and the fully resolved config. That block is what makes a published number checkable rather than claimed.

| Artifact | Written by | Carries | Read by | Committed? |
|---|---|---|---|---|
| `splits.parquet` | 01 | `question_id` → train/val/test, the record of the split | nothing downstream — `labels.parquet` carries the `split` column | no |
| `data_stats.json` | 01 | rows loaded, duplicates dropped, split sizes | 05 | yes |
| `activations.npz` | 01 | fp16 activations keyed by layer (~150 MB) | 02, 04 | **no** — regenerable and large |
| `labels.parquet` | 01 | completions, lenient + strict correctness, labels, split, abstention | 02, 04 | no |
| `extract_meta.json` | 01 | model description, equivalence check both ways, timings, base rates | 04, 05 | yes |
| `probe_sweep.json` | 02 | every (layer, C) with validation AUROC, `winner_at_grid_boundary` | 05 | yes |
| `probe.joblib` | 02 | fitted scaler + classifier + frozen threshold | 04 | no |
| `probe_test.json` | 02 | the test scoring: AUROC, `f`, `R`, precision, confusion, bootstrap CIs, ROC, abstention | 03, 05 | yes |
| `test_scoring_log.json` | 02 | **append-only** record of every test scoring ever run | 05 | yes |
| `economics.json` | 03 | three policies, lift, ceiling, base-rate projections | 05 | yes |
| `latency.json` | 04 | probe timing, raw dot product, ratios, device | 05 | yes |
| `RESULTS.md`, `*.png` | 05 | the formal record and its plots | humans | yes |
| `README.md` | 05 | rendered from `README_TEMPLATE.md` | humans | yes |
| `docs/HANDOVER.md` | 06 | orientation and presentation guidance | humans | yes |

Rules that hold across all of them:

- **A stage never recomputes an earlier stage's metric.** Stage 03 reads the lift's inputs from `probe_test.json`; it cannot disagree with the stage that measured them.
- **`.gitignore` excludes the large regenerable ones.** Never stage a file over 10 MB (`CONTRIBUTING.md`).
- **`git_commit` in an artifact is `HEAD` when the script ran** — the commit of the *code*, which by construction is not the commit containing the artifact. Hence: commit code, run, then commit artifacts as an `exp:` commit.

---

## Module map

`src/` holds all logic. `scripts/` parse arguments and write files. Notebooks import from `src/` and display. Nothing in `src/` hardcodes a value that belongs in `config.yaml`.

| Module | Responsibility | Things worth knowing |
|---|---|---|
| `config.py` | Typed dataclasses per config block, YAML loading with overrides, seeding, SHA-256 config hash, `provenance()`, JSON artifact IO | `provenance()` shells out to `git status --porcelain` and records `dirty: true` — an artifact never silently claims a clean tree |
| `data.py` | TriviaQA loading, answer/question normalisation, correctness rules, dedup, question-level splitting, labelling | `is_correct` is lenient with a whole-token guard for aliases under 3 characters; `is_exact_match` is the strict audit column |
| `model.py` | Model + tokenizer loading, NF4 quantisation config, chat templating, layer-fraction resolution | `assert_left_padding` raises `PaddingSideError`; `resolve_layers` turns fractional depths into absolute indices, so one config transfers across model sizes |
| `extract.py` | Question-time activation extraction, batched generation, equivalence check, base-rate gate | Raises `EquivalenceCheckError` and `BaseRateError` rather than proceeding on a suspect run |
| `probe.py` | Standardisation, logistic regression, layer × C sweep, selection, threshold freezing | `assert_polarity` exists because inverting the label silently yields `1 − AUROC`, which reads as a catastrophic result rather than a bug |
| `evaluate.py` | AUROC, confusion, precision/recall, bootstrap CIs, abstention analysis, ROC points, test-scoring log | Precision and recall are always separate — there is no F1 anywhere in this repository, by rule |
| `economics.py` | Three-policy table, `lift`, `lift_ceiling`, base-rate projections, invariance check | `invariance_check` demonstrates numerically that base error rate and judge accuracy cancel from the ratio |
| `latency.py` | Probe cost measurement against generation and prefill | Reports the full scikit-learn call *and* the raw dot product; the slower figure is the one quoted |
| `report.py` | Renders every generated document and both plots | `render_readme` raises `KeyError` on an unfilled placeholder — a blank in a published README is worse than a crash |

`tests/` has one suite per invariant rather than per module: split integrity, padding side, left-padding equivalence, normalisation, polarity, no-test-leakage, determinism, economics identities, config, extraction, model loading, audit regressions, and an end-to-end smoke test.

---

## Where the invariants live

`CLAUDE.md` states seven invariants. Each one is enforced in code, not merely documented — this table is the map from rule to enforcement.

| # | Invariant | Enforced in |
|---|---|---|
| 1 | Activations are taken at question-time, last prompt token, before any generated token | `extract.py::last_token_activations`, `extract.py::extract_batch`; `model.py::build_prompt` uses `add_generation_prompt=True` so the final token is the assistant header |
| 2 | Test is never used for selection, and every scoring is disclosed | `probe.py::run_sweep` and `select_threshold` see validation only; `scripts/02_train_probe.py` opens test at one marked place; `evaluate.py::append_test_scoring` keeps the append-only log |
| 3 | Splits are by question, never by example | `data.py::deduplicate_questions`, `split_by_question`, `assert_split_integrity` (asserts disjointness on both `question_id` and normalised string) |
| 4 | Left padding for all batched inference | `model.py::assert_left_padding` at load time and again per batch in `extract.py::extract_batch`; `extract.py::check_left_padding_equivalence` compares batched against unbatched **and** requires the right-padded control to fail |
| 5 | Precision and recall reported separately, never blended | `evaluate.py::evaluate_at_threshold`; no F1 exists in `src/` or `tests/` |
| 6 | `f` is the **measured** test flag rate, not the target | `probe.py::select_threshold` docstring, `economics.py::compare_policies`, which takes the measured rate as its input |
| 7 | No README number that wasn't produced by a script, with seed and config hash beside it | `report.py::readme_values` maps every placeholder to an artifact value; `render_readme` refuses to publish an unfilled one; `report.py::config_hash_consistency` surfaces artifacts produced under different hashes |

The one worth understanding in depth is **invariant 4**. With right padding, position −1 of a batch is a pad token, every activation is meaningless, *nothing raises*, and the resulting AUROC near 0.5 reads as "the idea doesn't work". So the check is scale-invariant (relative L2 and cosine, because bfloat16 rounding makes an absolute tolerance useless) and it is paired with a **positive control**: the same comparison is repeated with the tokenizer deliberately right-padded and the run aborts unless that one is rejected. Without the control, a permissive limit is indistinguishable from a limit that was loosened until it passed. Both rows are published in `results/RESULTS.md` §1.

Invariant 2's history is also worth knowing: it once read "the test set is touched exactly once", which nobody could verify. It now forbids *selection* on test and requires every scoring to be logged — a weaker claim, but an auditable one (`DECISIONS.md` 016, 017).

---

## Design rules that shape the code

- **Fail loudly.** Custom exceptions (`PaddingSideError`, `EquivalenceCheckError`, `BaseRateError`, `PolarityError`) exist so that a suspect run stops instead of producing a plausible wrong number. A silent wrong answer is far worse than a crash.
- **No logic in scripts or notebooks.** Logic that lives only in a notebook is unreviewable and unrunnable in CI.
- **Everything through `config.yaml`.** Model, dataset, layer fractions, flag rate, seeds, tolerances. The config is hashed and embedded in every artifact.
- **Seed everything** — `random`, `numpy`, `torch`, `torch.cuda` — and decode greedily. Two runs at one seed produce identical probe coefficients; `tests/test_determinism.py` asserts it.
- **Docstrings say why, not what**, and name the invariant wherever code enforces one.

---

## Changing things

| You want to | Change | Then |
|---|---|---|
| Use a different model | `model.name` (and `layer_fractions` stay as-is — they are fractional) | full re-run; extraction is the expensive part |
| Probe different depths | `model.layer_fractions` | full re-run |
| Change the judge budget | `economics.target_flag_rate` | `run_all.py --from 02` |
| Widen the regularisation grid | `probe.C_grid` | `--from 02`, and pre-register in `DECISIONS.md` if it will re-score test |
| Change the labelling rule | `data.py::is_correct` + `labeling` block | full re-run; log the decision, and expect the base rate to move |
| Edit README prose | `README_TEMPLATE.md` | `python scripts/05_report.py` — never edit `README.md` directly |
| Change a notebook | `scripts/build_notebooks.py` | regenerate, then execute to repopulate outputs |

Anything methodological gets an append-only entry in `DECISIONS.md` before it runs. Re-opening the test set until a number improves is forbidden, and the append-only log is what makes that checkable by someone who was not here.

---

## Deliberately absent

Not oversights — scope decisions recorded in `CLAUDE.md`:

- No LiteLLM gateway, serving layer, or API.
- No policy engine, action ladder, or OPA integration.
- No second model family or second dataset (the GSM8K negative control is optional Stage 6 and is **not implemented** — the `negative_control` block in `config.yaml` reserves settings that no code reads yet).
- No UI or dashboard beyond the notebooks.
- No fine-tuning. The base model is frozen; the only thing trained is a logistic regression.

The honest boundary to state out loud: **the probe tier requires weight access.** An enterprise consuming a model through a vendor API cannot read its activations. The cascade still applies; Tier 2 specifically does not.

---

**Next:** [SETUP.md](SETUP.md) to run it · [FAQ.md](FAQ.md) for the questions reviewers ask · [../DECISIONS.md](../DECISIONS.md) for why each choice was made · [../SPEC.md](../SPEC.md) for the technical specification.
