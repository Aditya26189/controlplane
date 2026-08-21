"""Generate the two notebooks from source, so they are reviewable as code.

Notebook JSON is not reviewable in a diff and is easy to corrupt by hand. This
script is the source of truth for both notebooks' *structure*; run it to
regenerate them, then execute the notebooks to populate their outputs.

    python scripts/build_notebooks.py

Regenerating discards any outputs already stored in the notebooks, so run it
before executing them, not after.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _cell_id(text: str) -> str:
    """Stable cell id derived from the cell text.

    Derived rather than random so that regenerating an unchanged notebook
    produces an unchanged file, and a diff shows only the cells that moved.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, text).hex[:8]


def markdown(text: str) -> dict:
    """A markdown cell."""
    body = text.strip()
    return {
        "cell_type": "markdown",
        "id": _cell_id(body),
        "metadata": {},
        "source": body.splitlines(True),
    }


def code(text: str) -> dict:
    """A code cell with no stored output."""
    body = text.strip()
    return {
        "cell_type": "code",
        "id": _cell_id(body),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": body.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    """Wrap cells in a minimal nbformat 4 document."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# notebooks/cascade_economics.ipynb -- the presentation wrapper
# --------------------------------------------------------------------------- #

PRESENTATION_CELLS = [
    markdown(
        """
# ControlPlane — cascade economics

Can you tell, **before** a language model writes a single token, whether the answer
it is about to give will be wrong?

If you can, monitoring gets much cheaper. Serious checkers — LLM-as-judge, semantic
entropy, claim attribution — cost 200–1000 ms per call, so nobody runs them on all
their traffic. They sample a few percent; the rest ships unchecked.

This notebook displays what `scripts/run_all.py` measured. **It contains no logic**:
every number is read from `results/*.json`, and every table is built by a function in
`src/report.py` that is covered by the test suite. Nothing here is computed for display.
"""
    ),
    code(
        """
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from IPython.display import Image, Markdown, display

from src.config import load_config
from src.report import (
    headline_markdown,
    latency_frame,
    load_artifacts,
    metadata_frame,
    policy_frame,
    sweep_frame,
    test_metrics_frame,
)

config = load_config(REPO_ROOT / "config.yaml")
artifacts = load_artifacts(config)
results_dir = REPO_ROOT / config.paths.results_dir
print(f"loaded {len(artifacts)} artifacts from {results_dir}")
"""
    ),
    markdown(
        """
## 1. What was run

The left-padding equivalence check is the load-bearing one. With right padding, position
`-1` of a padded batch is a pad token, every extracted activation is meaningless, and
nothing raises — the pipeline completes and returns an AUROC near 0.5 that reads as
"the idea doesn't work".
"""
    ),
    code("metadata_frame(artifacts)"),
    markdown(
        """
## 2. Where in the stack the signal lives

Validation AUROC for every layer and every regularisation strength tried. **The layer was
chosen here, on validation.** The test set was opened once, afterwards.

A smooth curve peaking mid-stack is itself evidence the signal is real rather than noise,
which is why the whole table is shown and not just the winner.
"""
    ),
    code("sweep_frame(artifacts)"),
    code('display(Image(filename=str(results_dir / "layer_sweep.png")))'),
    markdown(
        """
## 3. Test results — scored once

Precision and recall are reported separately and never blended into an F1. The two failure
modes differ in cost by orders of magnitude: a false positive wastes one judge call, a
false negative lets a user act on a wrong answer. The probe is tuned for recall and low
precision is accepted by design.
"""
    ),
    code("test_metrics_frame(artifacts)"),
    code('display(Image(filename=str(results_dir / "roc_curve.png")))'),
    markdown(
        """
## 4. The three policies

All at N = 1,000,000 responses. Rows two and three spend **the same judge budget**.

Coverage and verdict are different things. Every response is scored by the probe; only the
expensive verdict is rationed. Random sampling has a few percent coverage *and* a few
percent verdict. That gap is the whole result.
"""
    ),
    code("policy_frame(artifacts)"),
    markdown("## 5. The headline"),
    code("display(Markdown(headline_markdown(artifacts)))"),
    markdown(
        """
## 6. Does it slow the model down?

No, and this is measured rather than claimed. The probe adds **no additional forward
pass**: the activation it reads is a by-product of the prefill that generation already
performs, so its marginal cost is one scale-and-dot-product.
"""
    ),
    code("latency_frame(artifacts)"),
    markdown(
        """
## 7. What this does not show

The full limitations section is in [`results/RESULTS.md`](../results/RESULTS.md), written
from the artifacts rather than from boilerplate. The short version: one model, one dataset,
knowledge questions only, a single seed, and automatic alias matching standing in for human
judgment. This measures the probe, not an end-to-end system.
"""
    ),
    code('display(Markdown((results_dir / "RESULTS.md").read_text(encoding="utf-8")))'),
]


# --------------------------------------------------------------------------- #
# notebooks/run_on_kaggle.ipynb -- the GPU runner
# --------------------------------------------------------------------------- #

KAGGLE_CELLS = [
    markdown(
        """
# Run the experiment on a Kaggle T4

This notebook runs the whole pipeline on a GPU and produces every number in
`results/`. Everything else in this repo runs on a laptop; only the extraction
stage needs CUDA.

**Before you start**

1. Settings → Accelerator → **GPU T4 x2** (one is enough; NF4 keeps the 7B under 16 GB).
2. Settings → Internet → **On** (needed to fetch the model and the dataset).
3. Get this repo into the session — either set `REPO_URL` below to your GitHub
   remote, or upload the repo as a Kaggle Dataset and set `INPUT_DIR`.

Expected wall clock at `n_examples: 3000`: roughly 40–70 minutes for the
extraction, then a couple of minutes for everything else.
"""
    ),
    code(
        """
# Point at the repo. Set exactly one of these.
REPO_URL = ""                       # e.g. "https://github.com/<user>/controlplane-cascade.git"
INPUT_DIR = "/kaggle/input/controlplane-cascade"   # used when REPO_URL is empty

import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working/controlplane")
if REPO_URL:
    if WORK.exists():
        shutil.rmtree(WORK)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(WORK)], check=True)
else:
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(INPUT_DIR, WORK)

os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("working directory:", Path.cwd())
print(sorted(p.name for p in Path.cwd().iterdir()))
"""
    ),
    code(
        """
!pip install -q -r requirements.txt
"""
    ),
    markdown(
        """
## Stage 2 gate — does the model load, and does it answer?

TASKS.md Stage 2 asks for three things before the expensive stage: the model loads inside
the memory budget, one generated answer looks sane, and the resolved layer indices are
printed against the model's actual depth.
"""
    ),
    code(
        """
import torch

from src.config import load_config, set_seeds, setup_logging
from src.model import describe_model, load_model_and_tokenizer, peak_memory_gb, sanity_generate

setup_logging()
config = load_config("config.yaml")
set_seeds(config.seed)

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("config hash:", config.config_hash)

model, tokenizer = load_model_and_tokenizer(config)
info = describe_model(model, tokenizer, config)

print()
print("num_hidden_layers :", info["num_hidden_layers"])
print("hidden_size       :", info["hidden_size"])
print("probe layers      :", info["probe_layers"])
print("layer fractions   :", info["layer_fractions"])
print("padding side      :", info["padding_side"])
print("peak GPU memory   : %.2f GB" % peak_memory_gb())
print()
print("prompt template:")
print(info["example_prompt"])
"""
    ),
    code(
        """
for question in [
    "Who wrote the novel 'Nineteen Eighty-Four'?",
    "What is the capital of Australia?",
    "Which element has the chemical symbol 'Au'?",
]:
    print(f"Q: {question}")
    print(f"A: {sanity_generate(model, tokenizer, question, config)!r}")
    print()
"""
    ),
    markdown(
        """
## Padding diagnostic — measure before committing to the full run

Prints what the equivalence check actually sees on this model: relative L2 error and
cosine similarity under left padding, then the same under deliberate right padding. The
gap between the two columns is the check's discriminating power on *your* hardware.

Takes about 20 seconds and needs no full run.
"""
    ),
    code(
        """
import pandas as pd

from src.extract import compare_batched_unbatched, select_equivalence_prompts
from src.model import build_prompts, configure_tokenizer, resolve_layers
from src.data import prepare_dataset

frame, _ = prepare_dataset(config)
layers = resolve_layers(model, config)
prompts = select_equivalence_prompts(
    tokenizer,
    build_prompts(tokenizer, frame["question"].tolist(), config),
    config.equivalence_check.batch,
)
print("prompt token lengths:", [len(tokenizer(p)["input_ids"]) for p in prompts])

configure_tokenizer(tokenizer)
left = compare_batched_unbatched(model, tokenizer, prompts, layers)
tokenizer.padding_side = "right"
right = compare_batched_unbatched(model, tokenizer, prompts, layers)
configure_tokenizer(tokenizer)   # restore before anything else runs

rows = []
for layer in layers:
    key = str(layer)
    rows.append({
        "layer": layer,
        "LEFT rel L2": left["per_layer"][key]["max_relative_l2"],
        "LEFT cosine": left["per_layer"][key]["min_cosine"],
        "RIGHT rel L2": right["per_layer"][key]["max_relative_l2"],
        "RIGHT cosine": right["per_layer"][key]["min_cosine"],
        "activation norm": left["per_layer"][key]["reference_norm_median"],
    })
print()
print(pd.DataFrame(rows).to_string(index=False))
print()
print(f"limits: relative L2 <= {config.equivalence_check.relative_tolerance}, "
      f"cosine >= {config.equivalence_check.min_cosine}")
print("LEFT must pass both. RIGHT must fail both, by a wide margin.")
"""
    ),
    markdown(
        """
## Stage 3 pre-flight — do not skip this

Three checks before the GPU hour, per TASKS.md Stage 3:

1. the **left-padding equivalence check** on a batch of 4,
2. an **n=20 smoke run** whose completions you read by eye,
3. a **base-rate check** on those 20 — roughly half should be correct.

`--dry-run` writes nothing, so a bad result here costs a minute rather than an hour.
Free the notebook's model first so the subprocess gets the whole GPU.
"""
    ),
    code(
        """
import gc

del model
gc.collect()
torch.cuda.empty_cache()

!python scripts/01_extract.py --config config.yaml --limit 20 --dry-run
"""
    ),
    markdown(
        """
**Read the output above before continuing.**

- The equivalence check reports **relative L2 error** and **cosine similarity**, not an
  absolute deviation — in bfloat16 the absolute number is dominated by rounding and means
  nothing on its own (DECISIONS.md 014). Expect relative L2 well under `0.10` and cosine
  above `0.999`.
- It also runs a **positive control**: the same comparison with the tokenizer deliberately
  right-padded, which must be *rejected*. You should see that line in the log. If the
  control passes, the run stops — the limits would not be discriminating anything.
- If the check failed it raised, and the padding is wrong — stop, do not work around it.
- The completions should be short answers, not echoes of the prompt or empty strings.
- Roughly half should be marked `OK`. `0/20` or `20/20` means the prompt or the matching
  rule is broken, not that the model is unusually bad or good.

## The full run

Extraction, probe, economics, latency, report. Any stage can be re-run alone afterwards
with `--from`, so a failure at stage 04 does not cost the extraction again.
"""
    ),
    code(
        """
!python scripts/run_all.py --config config.yaml
"""
    ),
    markdown("## The result"),
    code(
        """
from IPython.display import Markdown, display

display(Markdown(Path("results/RESULTS.md").read_text(encoding="utf-8")))
"""
    ),
    markdown(
        """
## Take the artifacts home

Everything a reviewer needs is small — the JSON files, the two plots, `RESULTS.md` and the
rendered `README.md`. The activations (~150 MB) and the parquet files stay behind; they are
regenerable and are gitignored.

Download `results_bundle.zip` from the Kaggle output pane, unzip it over `results/` in your
local checkout, then commit with an `exp:` message recording the numbers that moved.
"""
    ),
    code(
        """
import zipfile

bundle = Path("/kaggle/working/results_bundle.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in ("results/*.json", "results/*.png", "results/RESULTS.md", "README.md"):
        for path in Path(".").glob(pattern):
            zf.write(path, path)
            print("added", path)
print()
print("wrote", bundle, f"({bundle.stat().st_size / 1024:.1f} KB)")
"""
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "notebooks"),
        help="where to write the notebooks",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, cells in [
        ("cascade_economics.ipynb", PRESENTATION_CELLS),
        ("run_on_kaggle.ipynb", KAGGLE_CELLS),
    ]:
        path = out_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(notebook(cells), fh, indent=1)
            fh.write("\n")
        print(f"wrote {path} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
