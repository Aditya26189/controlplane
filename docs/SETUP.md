# Setup and running

Three ways to run this repository, and what each one costs. Only **one stage needs a GPU**; everything else runs on a laptop in about a minute.

**Contents:** [What needs what](#what-needs-what) · [Laptop](#1-laptop-no-gpu) · [Kaggle or Colab](#2-kaggle-or-colab-t4--the-full-run) · [Offline cluster](#3-offline-cluster) · [Command reference](#command-reference) · [When it fails](#when-it-fails) · [Cost of a re-run](#cost-of-a-re-run)

---

## What needs what

| Stage | Needs | Roughly |
|---|---|---|
| `01_extract.py` | 16 GB GPU, network (model + dataset download) | 40–70 min at `n_examples: 3000` |
| `02_train_probe.py` | CPU, the saved `activations.npz` | seconds |
| `03_economics.py` | CPU, `probe_test.json` | instant |
| `04_latency.py` | CPU, `probe.joblib` + `activations.npz` | seconds |
| `05_report.py` | CPU, the JSON artifacts | seconds |
| `tests/` | CPU only, no network, no model download | ~10 s |

The measured wall clock for a specific run is recorded in `results/extract_meta.json` and reported in `results/RESULTS.md` §1 — those are the numbers to quote, not the estimates above.

---

## 1. Laptop (no GPU)

Enough to run the test suite, re-run stages 02–05 against the committed artifacts, and regenerate every document.

```bash
git clone <your-fork-url> controlplane
cd controlplane
python -m venv .venv && . .venv/Scripts/activate     # PowerShell: .venv\Scripts\Activate.ps1
                                                     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q                # the invariants
python scripts/05_report.py               # regenerate RESULTS.md, README.md, plots
```

Python 3.10 or newer.

**A partial install will look like a broken repo.** `requirements.txt` installs a CPU build of torch and `pyarrow`; without them, roughly two dozen tests error out with `ModuleNotFoundError: No module named 'torch'` or a parquet backend error. That is a missing dependency, not a failing invariant — install the full requirements before reading anything into a red test run. `bitsandbytes` is GPU-only and its import is optional on CPU, so it is expected to be inert on a laptop.

Re-running stages 02–05 needs `results/activations.npz` and `results/labels.parquet`, which are **not committed** (150 MB, regenerable). Without them you can still run the tests and regenerate the documents from the committed JSON, but not re-fit the probe. To get them, run a GPU extraction or copy the files from a teammate's run.

---

## 2. Kaggle or Colab T4 — the full run

The reference environment. `notebooks/run_on_kaggle.ipynb` walks the whole thing with a gate at each stage.

1. **Settings → Accelerator → GPU T4** (one is enough; NF4 keeps the 7B model under 16 GB).
2. **Settings → Internet → On** — the model and dataset are downloaded at runtime.
3. Get the repo into the session: set `REPO_URL` to your GitHub remote in the first code cell, or upload the repo as a Kaggle Dataset and set `INPUT_DIR`.
4. Run the cells top to bottom. The notebook checks for an existing `results/activations.npz` and tells you whether a GPU hour is actually required, then runs a pre-flight before the expensive loop.

Or drive it from a terminal:

```bash
pip install -r requirements.txt

python scripts/01_extract.py --data-only            # CPU: splits + data_stats, no model
python scripts/01_extract.py --limit 20 --dry-run   # pre-flight: loads the model, no artifacts
python scripts/run_all.py --config config.yaml      # the full pipeline
```

Do the pre-flight. It runs the padding equivalence check and its right-padding control before the loop, so a padding fault costs seconds instead of an hour.

**Before a real run, commit your working tree.** Every artifact records the `HEAD` it was generated from, and `provenance()` records `dirty: true` when the tree is not clean — a dirty run publishes numbers whose recorded commit does not describe the code that produced them.

---

## 3. Offline cluster

For an HPC setup where compute nodes have no network. Nothing in the code assumes network access at runtime.

On the login node:

```bash
export HF_HOME=/shared/path/hf_home
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
python -c "import datasets; datasets.load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')"
```

On the compute node:

```bash
export HF_HOME=/shared/path/hf_home
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python scripts/run_all.py --config config.yaml
```

Use the exact model name and dataset config from `config.yaml` — a mismatch fails at load time on a node that cannot fetch the difference.

---

## Command reference

```bash
# whole pipeline
python scripts/run_all.py --config config.yaml           # 01 -> 05
python scripts/run_all.py --config config.yaml --smoke   # n=100, into results/smoke/
python scripts/run_all.py --from 02                      # reuse saved activations
python scripts/run_all.py --from 02 --no-readme          # leave README.md alone

# stage 01, extraction
python scripts/01_extract.py --data-only                 # splits + stats, CPU, no model
python scripts/01_extract.py --limit 20 --dry-run        # pre-flight, writes nothing
python scripts/01_extract.py --limit 200                 # small real run
python scripts/01_extract.py --model <hf-id> --quantization none

# later stages, individually
python scripts/02_train_probe.py --config config.yaml
python scripts/03_economics.py  --config config.yaml
python scripts/04_latency.py    --config config.yaml
python scripts/05_report.py     --config config.yaml
python scripts/06_handover.py   --config config.yaml     # docs/HANDOVER.md

# documents and notebooks
python scripts/05_report.py --no-readme                  # RESULTS.md + plots only
python scripts/05_report.py --template README_TEMPLATE.md --readme README.md
python scripts/build_notebooks.py                        # regenerate both notebooks
```

`--smoke` writes everything into `results/smoke/` and redirects the rendered README there too, so it can never overwrite a real run's artifacts. Its base-rate sanity band is deliberately *not* relaxed — a smoke mode that disables its own safety checks tests nothing worth testing.

---

## When it fails

These are all deliberate stops. The pipeline is built to crash rather than publish a plausible wrong number.

| Symptom | What it means | What to do |
|---|---|---|
| `PaddingSideError` | The tokenizer is not left-padded. With right padding, position −1 is a pad token and every activation is meaningless | Do not work around it. `configure_tokenizer` sets the padding side; find what reset it |
| `EquivalenceCheckError` — left padding failed | Batched and unbatched last-token activations disagree beyond the scale-invariant limits | Real fault. Check the tokenizer, the batch construction, and that the prompts genuinely differ in length |
| `EquivalenceCheckError` — the right-padding **control passed** | The check can no longer discriminate a real padding fault; its limits are worthless as written | Stop. Tighten `equivalence_check` limits until the deliberately broken case is rejected again |
| `BaseRateError` | The measured accuracy is outside `labeling.base_rate_min/max`. Usually a malformed prompt, truncated generation, or a broken matching rule | Inspect a few completions in `labels.parquet` before touching the band |
| AUROC below `evaluation.min_auroc_to_proceed` | The probe found little signal on this data | Report it. `probe_test.json` carries an `auroc_floor` block with a checklist. A weak result is a valid output of this repo — do not tune until it improves |
| `PolarityError` | Labels are inverted somewhere. `y = 1` must mean the answer was **wrong** | Fix the label construction. Inverted polarity silently yields `1 − AUROC`, which reads as a catastrophe rather than a bug |
| `KeyError: README template has placeholder(s) with no value` | The template references something `readme_values` does not produce | Add the value in `src/report.py` from an artifact, or remove the placeholder. Never hand-fill it in `README.md` |
| CUDA OOM during extraction | 7B in NF4 plus a batch does not fit | Lower `generation.batch_size` to 4. Do not disable quantisation on a 16 GB card |
| `ModuleNotFoundError: torch` in the tests | Requirements not fully installed | `pip install -r requirements.txt` |
| `dirty: true` in an artifact's provenance | A script ran against an uncommitted working tree, so the recorded commit does not describe the code that produced the numbers | Commit, re-run, re-commit the artifacts as `exp:` |
| Report warns about differing config hashes | Stages ran under different configs, so the artifacts do not describe one experiment | Re-run the whole pipeline before quoting anything |

---

## Cost of a re-run

| Change | Re-run | Cost |
|---|---|---|
| README or docs prose | `scripts/05_report.py` | seconds |
| Flag rate, `C` grid, bootstrap count | `run_all.py --from 02` | ~1 min |
| Reference error rate, projections | `03_economics.py` then `05_report.py` | seconds |
| Layers, model, dataset, prompt, labelling rule | full run | a GPU hour |

Anything that moves a measured number needs an entry in `DECISIONS.md` and a commit body stating the before and after (`CONTRIBUTING.md`). If a re-run would re-score the test set, pre-register it in `DECISIONS.md` **before** it runs — that rule is what makes the selection discipline checkable by someone who was not here.

---

**Next:** [ARCHITECTURE.md](ARCHITECTURE.md) for what the stages actually do · [FAQ.md](FAQ.md) for the questions reviewers ask · [../CONTRIBUTING.md](../CONTRIBUTING.md) for the git workflow.
