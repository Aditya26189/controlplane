# CLAUDE.md

Persistent context for agents working in this repo. Read this fully before writing code. Read `SPEC.md` for the technical detail and `TASKS.md` for the build order.

---

## What this repo is

A reproducible experiment that measures whether a **linear probe on question-time activations** can select which LLM responses are worth sending to an expensive checker.

It exists to produce **one number** for a competition submission (Accenture Innovation Challenge 2026, problem statement 1):

> **lift = R / f** — the probe's recall divided by its flag rate. How many more errors we catch than random sampling at the same budget.

Everything in this repo either produces that number, validates it, or documents it. If a piece of work does none of those three, it does not belong here.

## The claim we are testing

An LLM's internal state, read **after it has finished reading the question but before it has generated a single token**, contains a readable signal about whether the answer it is about to produce will be wrong.

If true, a logistic regression on that state — one dot product, effectively free next to the generation you were already paying for — can run on 100% of traffic and select the small slice worth spending an expensive judge on.

**We are testing this, not assuming it.** A negative or weak result is a valid output of this repo. Report what we measure.

## Non-negotiable invariants

Violating any of these invalidates the result. If a change would break one, stop and flag it rather than working around it.

1. **Activations are taken at question-time.** Last token of the prompt, before any generated token exists. Never mid-generation, never from the answer.
2. **The test set is never used for selection, and every scoring of it is disclosed.** Layer selection, threshold selection, and hyperparameter choice all happen on validation. Test is opened at the end, for the final numbers.
   The original form of this invariant was "touched exactly once". It has in fact been opened three times (`DECISIONS.md` 016, 017), so the enforceable rule is the one that can be checked rather than the one that sounds strongest:
   - no selection may consult test — not the layer, not `C`, not the threshold;
   - every scoring is appended to `results/test_scoring_log.json` and disclosed in `RESULTS.md`;
   - any re-scoring is pre-registered in `DECISIONS.md`, with the prior numbers, **before** it runs.
   Re-opening test until a number improves is forbidden, and the append-only log is what makes that checkable by someone who was not here.
3. **Splits are by question, never by example.** Group by `question_id`, deduplicate near-identical question strings first, and assert zero overlap between splits.
4. **Left padding for all batched inference.** With right padding, position `-1` is a pad token and every activation is garbage. This fails silently — it produces plausible-looking AUROC near 0.5 and nothing errors out.
5. **Precision and recall are always reported separately.** Never a blended F1 anywhere in code, output, or README.
6. **`f` in every calculation is the measured test-set flag rate**, not the target flag rate you aimed for.
7. **No number gets written to the README that wasn't produced by a script in this repo**, with a seed and a config hash next to it.

## What the probe is and is not

- It is a **trigger**. Its only output is a decision about whether to spend a judge call.
- It is **not a verdict**. Nothing in this repo should ever be described as blocking, filtering, or gating a user-facing response.
- It is tuned for **recall**, and poor precision is acceptable by design. A false positive costs one wasted judge call. A false negative costs a customer acting on a wrong answer.
- Never describe it as measuring "truthfulness", "faithfulness", or "what the model believes". It is a correlational classifier over activations. Use that language in comments, docstrings, and README.

## Layout

```
CLAUDE.md            this file
SPEC.md              technical specification — read before implementing
TASKS.md             staged build order with acceptance gates
CONTRIBUTING.md      git workflow and documentation rules — binding
DECISIONS.md         append-only log of methodological choices
README_TEMPLATE.md   source for the generated README
config.yaml          all knobs; nothing hardcoded in src/
requirements.txt
src/
  config.py          config loading, dataclasses, config hashing
  data.py            TriviaQA loading, dedup, question-level splitting
  model.py           model + tokenizer loading, NF4, chat templating
  extract.py         activation extraction + greedy generation + labelling
  probe.py           standardisation, logistic regression, layer sweep
  evaluate.py        AUROC, precision, recall, bootstrap CIs
  economics.py       three-policy comparison, lift
  latency.py         wall-clock probe cost vs generation cost
  report.py          renders results/ into markdown + plots
scripts/
  01_extract.py      → results/activations.npz, results/labels.parquet
  02_train_probe.py  → results/probe_sweep.json, results/probe.joblib
  03_economics.py    → results/economics.json
  04_latency.py      → results/latency.json
  05_report.py       → results/RESULTS.md, results/*.png, README.md
  06_handover.py     → docs/HANDOVER.md
  run_all.py         orchestrates 01→05
  build_notebooks.py regenerates both notebooks from source
tests/
docs/
  README.md          index: which document answers which question
  HANDOVER.md        generated orientation doc for someone joining cold
  ARCHITECTURE.md    stage boundaries, artifact contracts, invariant enforcement map
  SETUP.md           running it: laptop, Kaggle T4, offline cluster, failure modes
  FAQ.md             reviewer questions, each answered with the artifact that settles it
  GLOSSARY.md        lift, f, R, polarity, positive control — defined once, precisely
notebooks/
  cascade_economics.ipynb   thin presentation wrapper over src/, no logic
  run_on_kaggle.ipynb       GPU runner: stage gates, pre-flight, full run
results/             all outputs; committed only as final JSON + RESULTS.md
```

Notebook JSON is not reviewable in a diff, so both notebooks are generated by
`scripts/build_notebooks.py` and that script is the source of truth for their
structure. Edit the script, regenerate, then execute to populate outputs.

```
```

## Coding standards

- Python 3.10+. Type hints on every public function. Google-style docstrings.
- **No logic in scripts or notebooks.** Scripts parse args, call `src/`, write files. The notebook imports from `src/` and displays. Any logic that lives only in a notebook is unreviewable and unrunnable.
- **No hardcoded constants.** Model name, dataset, layer list, flag rate, seeds, sample size all come from `config.yaml`.
- Every stage writes its output to `results/` and every subsequent stage reads from disk. Stages must be independently re-runnable without re-running the expensive extraction.
- Seed everything: `random`, `numpy`, `torch`. Log the resolved config, the config hash, and library versions into every output artifact.
- Structured logging to stdout with timestamps. Long-running loops get a `tqdm` bar with an ETA.
- Fail loudly. Assert shapes and invariants at every boundary. A silent wrong answer is far worse than a crash.

## Version control

Full rules in `CONTRIBUTING.md`. The parts you must not get wrong:

- **Commit continuously, not at the end.** Every stage should produce four to ten atomic commits. An uncommitted working tree at a stage gate is a failed gate.
- **Conventional Commits.** `feat` `fix` `docs` `test` `refactor` `chore` `exp`. `exp` is for runs that produce or change artifacts in `results/`.
- **One logical change per commit.** If the subject line needs "and", split it.
- **Any commit that moves a measured number states the before and after in its body.** This is the single most useful habit in the repo.
- **Never run a script against a dirty working tree.** Artifacts record the `HEAD` they were generated from; if the tree is dirty that record is false. `provenance()` must check `git status --porcelain` and flag `dirty: true`.
- **Stage branches, merged with `--no-ff` and tagged at each gate.** Tags are the rollback points, which matters because Stage 3 costs a GPU hour.
- **Never stage a file over 10 MB.** Stop and ask instead.

## Documentation

Three tiers, three rules.

**Contracts** (`CLAUDE.md`, `SPEC.md`, `TASKS.md`) — a doc change ships **in the same commit** as the code change that made it necessary. Never a trailing "update docs" commit. If code contradicts `SPEC.md`, stop: either the code is wrong or the spec needs updating deliberately, first, with the reasoning logged.

**Rationale** (`DECISIONS.md`) — append-only. Log every methodological choice a reviewer could challenge: dataset, split strategy, label rule, selection procedure, metric choice. Never edit or delete an entry; supersede it with a new one. This is the direct answer to "why did you do it that way?", which is most of what a technical judge asks.

**Generated** (`README.md`, `results/RESULTS.md`, `docs/HANDOVER.md`) — **never hand-edit a number.** If a number is wrong, the pipeline is wrong. Fix it and regenerate. Prose can be edited; numbers cannot.

**Orientation** (`docs/ARCHITECTURE.md`, `docs/SETUP.md`, `docs/FAQ.md`, `docs/GLOSSARY.md`, indexed by `docs/README.md`) — hand-written, and deliberately carrying **no measured numbers**: they point at the generated documents instead, so a re-run cannot leave them stating a stale figure. They describe structure, procedure and vocabulary, all of which change with the code rather than with the run.

Docstrings say *why*, not what. Where code enforces an invariant from this file, name the invariant in the docstring.

## Pitfalls that have burned this design before

Read these. Several of them fail silently, which is why they're listed rather than left to testing.

**Right padding.** See invariant 4. Assert `tokenizer.padding_side == "left"` at load time and again before every batched call.

**Selecting the layer on test data.** Tempting because it's one line. It inflates the headline number and it's the first thing a reviewer checks. Layer selection is a validation-set decision.

**Answer aliases leaking across splits.** TriviaQA ships multiple valid aliases per answer and occasionally near-duplicate questions. Deduplicate on the normalised question string before splitting.

**Substring matching on short aliases.** If a gold alias is `"US"`, it appears inside thousands of unrelated generations. Guard: aliases shorter than 3 characters require exact token match, not substring containment.

**Unbalanced labels read as signal.** If the model is right 85% of the time, a probe that always predicts "correct" scores 0.85 accuracy and 0.5 AUROC. Always report base rate alongside AUROC, and use `class_weight="balanced"`.

**Unstandardised features.** Residual stream vectors have large and layer-varying magnitudes. Fit `StandardScaler` on train only, then apply to val and test. Fitting on the full set leaks.

**`output_hidden_states=True` inside `generate()`.** It retains hidden states for every generated step and will exhaust GPU memory on a T4. Do a separate prefill forward pass for activations, then generate. See `SPEC.md` §4.

**Confusing target flag rate with measured flag rate.** You pick a threshold on validation to hit `f≈0.05`. On test the actual rate will differ. Every downstream calculation uses the measured one.

**Reporting lift without a confidence interval.** A single point estimate from ~600 test examples is not defensible. Bootstrap it.

## Environment notes

- Target hardware: a single 16GB GPU (Colab/Kaggle T4 is the reference environment). NF4 4-bit quantisation via `bitsandbytes`.
- If running on an HPC cluster where compute nodes are offline: pre-download the model and dataset on the login node into `HF_HOME`, then set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` on the compute node. Do not write code that assumes network access at runtime.
- Everything must run end-to-end on free-tier compute. If a change requires paid compute, stop and flag it.
- MIT licence throughout. Do not add a dependency under a restrictive or non-commercial licence — the project's public claim is that the stack is fully open and self-hostable. Notably: do not take a runtime dependency on Llama Guard or ShieldGemma.

## Definition of done

`python scripts/run_all.py --config config.yaml` on a clean checkout produces:

- `results/RESULTS.md` containing the layer sweep, the chosen layer with its validation justification, test AUROC with a bootstrap CI, measured `f`, measured `R`, precision, the three-policy table, measured lift with CI, the latency ratio, and the base rate.
- A populated `README.md` whose every number traces to `results/`.
- A passing test suite.
- Reproducibility: two runs at the same seed produce identical numbers.
- A clean working tree, a legible commit history with a tag per stage, and `DECISIONS.md` covering every methodological choice.
- A passing documentation audit (`CONTRIBUTING.md`, final section).

## Out of scope

Do not build these. They are Round 2 work and they will eat the timeline.

- The LiteLLM gateway, any serving layer, any API.
- The policy engine, action ladder, or OPA integration.
- A second model family or second dataset (except the optional GSM8K negative control in Stage 6).
- Any UI or dashboard beyond the notebook.
- Fine-tuning anything. The base model is frozen; only the logistic regression is trained.
