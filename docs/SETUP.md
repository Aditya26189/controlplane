# SETUP.md — getting it running

Three environments, in order of how likely you are to need them: a laptop
(everything except extraction), a GPU notebook (extraction), and an offline
machine (both, with the downloads done in advance).

**Contents:** [Requirements](#requirements) · [Laptop](#1-laptop--the-normal-case) · [The four tiers](#the-four-reproduction-tiers) · [GPU](#2-gpu--the-extraction-stage) · [Offline](#3-offline-machine) · [Optional extras](#optional-extras) · [Verifying your install](#verifying-your-install)

---

## Requirements

| | |
|---|---|
| Python | 3.11 or newer |
| Everything except extraction | CPU only, no network after the clone |
| Extraction | a 16 GB GPU; NF4 quantisation keeps a 7B model inside it |
| Disk | the clone is small; activation caches are ~100 MB each and gitignored |

Two requirement files, and the difference matters:

- **`requirements.lock.txt`** — exact pins. Use this. Some findings in this
  repository are statements about a specific dependency version, and a test
  fails if the environment drifts from the pin.
- **`requirements.txt`** — lower bounds, for a from-scratch environment where
  you accept that a version-specific finding may not reproduce.

---

## 1. Laptop — the normal case

Under five minutes, and it exercises everything except the GPU stage.

```bash
git clone <your fork or the upstream>
cd controlplane
git fetch origin "refs/notes/*:refs/notes/*"

python -m venv .venv
. .venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
                                  # macOS/Linux: source .venv/bin/activate
pip install -r requirements.lock.txt

python scripts/smoke.py           # < 60s
python scripts/verify.py          # ~3 min
```

**Fetch the notes ref.** A commit in this history states a claim that was later
found to be wrong, and the correction lives in a git note attached to that
commit. The history was not rewritten — a withdrawn claim that stays visible
next to its correction is worth more than a clean log — but the correction is
only visible if you fetch the ref. `README.md`, "Reading the history".

---

## The four reproduction tiers

Every target is a single command with no shell logic in it, so the `make`
column and the plain column are exactly equivalent. Windows users without
`make` lose nothing.

| Target | Plain command | Needs | Time | Proves |
|---|---|---|---|---|
| `make smoke` | `python scripts/smoke.py` | CPU, no network | < 60s | the clone came down intact and the package imports |
| `make test` | `python -m pytest tests/ -q` | CPU | ~10 min | the suite is green |
| `make verify` | `python scripts/verify.py` | CPU | ~3 min | **every claim reproduces, and every metric recomputes from frozen scores** |
| `make verify` tier 3 | same, with caches present | CPU + caches | ~4 min | the frozen scores re-derive from the activations |
| `make extract` | `python scripts/00_extract.py --config config.yaml` | GPU, 16 GB | ~1 h | the activations regenerate from the source model |

`make verify-claims` (or `python scripts/verify.py --claims-only`) runs tier 1
alone in seconds — useful while editing prose in the README.

**What `verify` will not do:** write to `results/`. Its re-runs go to a scratch
directory, so a failed verification cannot damage the evidence it was checking.

---

## 2. GPU — the extraction stage

Nobody needs to run this to check the work; the frozen scores in
`results/scores/` are what a clean clone verifies against. It is documented so
the chain from raw model to published number has no gap in it.

**The tested path is the notebook**, `notebooks/run_on_kaggle.ipynb`:

1. Accelerator → **GPU T4** (one is enough; NF4 keeps the 7B under 16 GB).
2. Internet → **On**, for the model and dataset download.
3. Point the first cell at your fork, or upload the repo as a dataset.
4. Run top to bottom. The notebook gates each stage rather than running blind.

The notebook is **generated** by `scripts/build_notebooks.py` and never
hand-edited — notebook JSON is not reviewable in a diff. Edit the script,
regenerate, then execute to populate outputs.

From a terminal on a GPU box:

```bash
python scripts/00_extract.py --config config.yaml            # both envelopes
python scripts/00_extract.py --config config.yaml --batch-size 4   # if OOM
```

**Commit before you run.** Every artifact records the `HEAD` it was built from
and sets `dirty: true` when the tree is not clean, which makes the recorded
commit a lie about which code produced the numbers.

---

## 3. Offline machine

Nothing in the code assumes network access at runtime. Do the downloads on a
node that has it:

```bash
export HF_HOME=/shared/path/hf_home
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
python -c "import datasets; datasets.load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')"
```

Then, on the machine that will run it:

```bash
export HF_HOME=/shared/path/hf_home
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python scripts/00_extract.py --config config.yaml
```

Use the model name and dataset config exactly as they appear in `config.yaml` —
a mismatch fails at load time on a machine that cannot fetch the difference.

The CPU tiers (`smoke`, `test`, `verify`) need no network at all once the
dependencies are installed.

---

## Optional extras

Some paths need a dependency the core install does not:

| You want | You also need |
|---|---|
| the Presidio detector runs | `presidio-analyzer` at the pinned version — the finding is a statement about that version |
| the policy engine's Rego evaluation | `regopy` — a pip wheel, not a 50 MB OPA binary. It is in the requirements; `DECISIONS.md` 076 records what that substitution does and does not buy |
| to push extraction to Kaggle from a terminal | the `kaggle` CLI configured, then `scripts/kaggle_run.py` |
| plots regenerated | `matplotlib`, already in the requirements |

---

## Verifying your install

```bash
python scripts/smoke.py
```

Seven checks. What each one being green actually tells you:

| Check | Means |
|---|---|
| python version | you are on 3.11+ |
| package imports | `controlplane` and its model, policy and validation subpackages resolve |
| config loads | `config.yaml` parses, its invariants hold, and it hashes |
| provenance resolves | git metadata is reachable — the artifacts can record where they came from |
| artifacts readable | a real validation artifact parses and carries a warrant status |
| eval sets frozen | the frozen sets are present and content-hashed |
| claim table parses | every row of the README table names an artifact and a field |

`smoke` deliberately does not check whether the numbers are *right* — that is
`verify`'s job, and keeping them separate is what makes a smoke failure mean
something specific.

If any of it misbehaves, [TROUBLESHOOTING.md](TROUBLESHOOTING.md) has the
failure modes and what each one means.

---

**Next:** [RUNBOOK.md](RUNBOOK.md) for what every script does · [ONBOARDING.md](ONBOARDING.md) for what to read first · [ARTIFACTS.md](ARTIFACTS.md) for what the outputs contain.
