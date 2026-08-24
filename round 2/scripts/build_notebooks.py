"""Generate ``notebooks/run_on_kaggle.ipynb``. This script is its source of truth.

Notebook JSON is not reviewable in a diff — a one-character change to a cell
produces a diff full of execution counts and output blobs, and a reviewer cannot
tell a logic change from a re-run. So the notebook is generated from here, and
this file is what gets reviewed.

The notebook itself contains **no logic**. Every cell calls into ``src/`` or
runs a shell command. That is the same rule the demo runner follows and for the
same reason: logic that lives only in a notebook is unreviewable and unrunnable
by anything else.

Usage:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().split("\n"),
    }


def build_kaggle_notebook() -> dict:
    """The GPU extraction notebook, staged so a failure stops at the right place."""
    cells = [
        markdown(
            """
# ControlPlane — TriviaQA extraction

The **only** stage that needs a GPU. Everything downstream runs from the caches
this writes, which is what makes `/validate` fast enough to be a button a judge
can press.

This session extracts **both** envelopes:

| envelope | items | why |
|---|---|---|
| `triviaqa-600` | 2,400 questions → 1,200 train / 600 validation / 600 test | the tier ladder anchor |
| `triviaqa-longctx-600` | the 600 test questions, padded to 4–16k tokens | Beat 4's envelope shift |

Both in one run. A two-session plan is a plan where the second session does not
happen, and Beat 4 has no measured basis without the long-context pass.

**Runtime, from Round 1's measured throughput** rather than a guess. Round 1
extracted 3,000 examples in 7,744 s on the same model and card — 0.387
examples/s, of which generation is 2.37 s/item and prefill only 0.31 s.

| pass | items | dominated by | estimate |
|---|---|---|---|
| short context | 2,400 | generation | ~1.7 h |
| long context | 600 | prefill at 4–16k tokens | ~1–2 h |

Long context needs **no generation** — it reuses the short pass's answers and
labels, because they are the same questions — so it is prefill only, and prefill
at 10k tokens is far slower than at Round 1's 192. Budget 3–4 h total and start
early in a session.

### Before you start
- Accelerator: **GPU T4 ×2** (or P100). Internet: **on**, for the model and dataset.
- Nothing here writes outside `/kaggle/working`.
"""
        ),
        markdown("## 0 — Pre-flight\n\nStops here if the environment cannot do the job."),
        code(
            """
import subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
"""
        ),
        code(
            """
%pip install -q bitsandbytes accelerate datasets
"""
        ),
        code(
            """
import torch

# Round 1 measured 2.31 GB peak after load for this model in NF4. The long
# context pass adds a KV cache (~0.85 GB at 16k) and one captured layer
# (~0.11 GB). Capturing all 29 hidden states instead would add 3.1 GB, which is
# why extraction uses a forward hook on the single layer the probe is pinned to.
assert torch.cuda.is_available(), (
    "No GPU. Extraction needs one; every other stage in this project does not."
)
free, total = torch.cuda.mem_get_info()
print(f"{torch.cuda.get_device_name(0)}  {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")
assert total / 2**30 > 14, (
    f"{total/2**30:.1f} GiB total. Qwen2.5-7B in NF4 needs ~6 GiB for weights and "
    "the long-context pass needs headroom for a 16k-token KV cache."
)
"""
        ),
        markdown(
            """
## 1 — Get the repository, and prove it is the right half

**This repository contains two projects.** Round 1 sits at the root; Round 2
sits in `round 2/`. They have separate `src/`, separate `config.yaml`, and
incompatible schemas — Round 1's `ProbeConfig` has no `aggregations` field, so
pointing this notebook at the root fails several cells in, after the model has
loaded.

An earlier version of this cell fell back to the root when `round 2/` was
missing. That is the failure mode this project is about: an absence reading as
presence (`DECISIONS.md` 050). It now refuses.

The clone below is pre-filled and should need no editing. If it produces no
`round 2/` directory, the cell says what it searched and how to fix it rather
than proceeding with the wrong half.
"""
        ),
        code(
            """
import os, sys
from pathlib import Path

# Round 2 lives in "round 2/" on main. Clone it and the rest of the notebook
# finds it. Override only if you are running a fork or a work-in-progress
# branch; attaching the repo as a Kaggle Dataset also works, and CANDIDATES
# below searches /kaggle/input for it without any edit here.
CLONE_URL = "https://github.com/Aditya26189/controlplane.git"
CLONE_BRANCH = "main"

CANDIDATES = [
    Path("/kaggle/working/controlplane/round 2"),
    Path("/kaggle/working/controlplane"),
    *sorted(Path("/kaggle/input").glob("*/round 2")),
    *sorted(Path("/kaggle/input").glob("*")),
    Path.cwd() / "round 2",
    Path.cwd(),
]

if CLONE_URL and not Path("/kaggle/working/controlplane").exists():
    !git clone --depth 1 --branch {CLONE_BRANCH} {CLONE_URL} /kaggle/working/controlplane


def is_round_two(path: Path) -> bool:
    \"\"\"Round 2 has an extraction stage and a matrix; Round 1 has neither.\"\"\"
    return (
        (path / "config.yaml").is_file()
        and (path / "src" / "extract").is_dir()
        and (path / "src" / "matrix").is_dir()
    )


PROJECT = next((p for p in CANDIDATES if is_round_two(p)), None)
if PROJECT is None:
    raise SystemExit(
        "Round 2 not found.\n\n"
        "This repository contains two projects: Round 1 at the root and Round 2\n"
        "in 'round 2/'. They have separate src/ and config.yaml, and Round 1's\n"
        "ProbeConfig has no 'aggregations' field -- so running this notebook\n"
        "against the root fails several cells in, after the model has loaded.\n\n"
        "Refusing to fall back to the root, because a wrong config that loads is\n"
        "worse than one that does not (DECISIONS.md 050).\n\n"
        "Looked in:\n  " + "\n  ".join(str(p) for p in CANDIDATES) + "\n\n"
        "Fix, whichever is easiest:\n"
        "  - check CLONE_URL above is reachable and CLONE_BRANCH exists. The\n"
        "    clone is silent on failure in some Kaggle images, so re-run the\n"
        "    clone line on its own and read what it prints.\n"
        "  - or zip 'round 2/' locally, upload it as a Kaggle Dataset, and it\n"
        "    will be found automatically under /kaggle/input/ with no edit here.\n"
    )

# Drop any half-imported Round 1 modules before adding Round 2 to the path.
# Python caches by module name, so a stale 'src' would shadow the right one and
# the failure would look like a missing attribute rather than a wrong import.
for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
    del sys.modules[name]

sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
print("project root:", PROJECT)
"""
        ),
        code(
            """
!git -C "$(git -C . rev-parse --show-toplevel)" log --oneline -1
"""
        ),
        markdown(
            """
## 2 — Config and the padding assertion

The padding side is asserted at load, again before every batched forward pass,
and a third time at validation by the fault-injection control. With right
padding, position −1 of a batched sequence is a pad token, every activation is
read from nothing, and the probe lands near 0.5 AUROC — which reads as *"the
idea does not work"* rather than as a bug.
"""
        ),
        code(
            """
import logging
from src.config import load_config, set_seeds, setup_logging

setup_logging(logging.INFO)
config = load_config("config.yaml")
set_seeds(config.seed)

# Assert this is Round 2's config before anything expensive happens. Round 1's
# ProbeConfig has no `aggregations` and no `workload` block; loading it here
# would fail later, after the model is in memory, with an AttributeError that
# looks like a code bug rather than a wrong working directory.
missing = [
    name
    for name, present in (
        ("probe.aggregations", hasattr(config.probe, "aggregations")),
        ("probe.rolling_window", hasattr(config.probe, "rolling_window")),
        ("workload", hasattr(config, "workload")),
        ("validation.null_control_min_repeats",
         hasattr(config.validation, "null_control_min_repeats")),
    )
    if not present
]
if missing:
    raise SystemExit(
        f"This is not Round 2's config: {missing} absent.\n"
        f"Loaded from {Path('config.yaml').resolve()} (hash {config.config_hash}).\n"
        "Round 1's config hash is c429ce5e92da9a22; Round 2's is c89257bc4adc10c2.\n"
        "Go back to the previous cell and point PROJECT at 'round 2/'."
    )

print("config hash:", config.config_hash)
print("model:", config.model.name, "| quantization:", config.model.quantization)
print("aggregations:", list(config.probe.aggregations))
print("rolling window/stride:", config.probe.rolling_window, config.probe.rolling_stride)
"""
        ),
        code(
            """
from src.extract.model import load_model

loaded = load_model(config.model.name, quantization=config.model.quantization)
print(loaded.provenance())
print("layers resolved from fractional depths:", config.resolve_layers(loaded.num_hidden_layers))
"""
        ),
        markdown(
            """
## 3 — Extract

One call. It deduplicates TriviaQA, splits **by question**, captures the padding
evidence, generates greedily, labels by alias match with the short-alias guard,
and pools activations for every configured aggregation from the same forward
pass.

Set `--smoke` equivalent by passing `n_questions=120` first if you want to check
the wiring before spending the hour.
"""
        ),
        code(
            """
from src.extract.pipeline import extract_triviaqa

result = extract_triviaqa(
    config,
    loaded,
    n_questions=2400,      # 120 for a smoke run
    batch_size=8,
    long_batch_size=1,     # a 16k-token sequence does not batch on 16 GiB
    max_new_tokens=32,
    long_context=True,
)
result.report
"""
        ),
        markdown(
            """
### What to check before going further

- **base rate** should be somewhere near 0.15–0.35. Exactly 0 or 1 raises, but a
  base rate of 0.02 means almost every answer was judged correct and the alias
  matching is probably too loose.
- **match_rules** shows how labels were reached. A large `exact token match on
  short alias` count is expected; a large `empty generation` count is not.
- **token_length_max** for the long pass should land inside the configured
  4,000–16,000 band.
"""
        ),
        code(
            """
print("short :", len(result.short_evalset), "items,",
      f"base rate {result.short_evalset.base_rate:.4f}", result.short_evalset.envelope_id)
if result.long_evalset is not None:
    print("long  :", len(result.long_evalset), "items,",
          f"base rate {result.long_evalset.base_rate:.4f}", result.long_evalset.envelope_id)
    print("long token lengths:", result.long_cache.token_lengths.min(),
          "to", result.long_cache.token_lengths.max())
"""
        ),
        markdown(
            """
## 4 — Freeze and save

The eval sets are content-hashed; the hash **is** the envelope id and therefore
the third element of every warrant key measured on them. The caches are large
and gitignored — download them, do not commit them.
"""
        ),
        code(
            """
from src.evalsets import save_evalset

save_evalset(result.short_evalset, "evalsets")
short_path = result.short_cache.save("results/cache-triviaqa-600.npz")
paths = [short_path]
if result.long_evalset is not None:
    save_evalset(result.long_evalset, "evalsets")
    paths.append(result.long_cache.save("results/cache-triviaqa-longctx-600.npz"))
for path in paths:
    print(path, f"{path.stat().st_size / 2**20:.1f} MiB")
"""
        ),
        markdown(
            """
## 5 — Self-check, here rather than after downloading

Two checks, both cheap and both worth failing on the GPU rather than on a laptop
three hours later.

**Shape compatibility** asserts the measured path and the fixture path produce
the same metric *structure* — same metrics present, same kinds, same units,
intervals on both sides. Values must differ; shape must not. A mismatch means
the two are not measuring the same thing and every fixture-versus-measured
comparison in the repo is void.

**The transfer** scores the short-context probe on long-context inputs without
refitting. That is the drift question — *what is **this** probe worth here?* —
and it is what Beat 4 shows.
"""
        ),
        code(
            """
from src.validation.runner import validate

variant = f"T1-{config.probe.aggregations[0]}"
source = validate(
    config, result.short_evalset, result.short_cache,
    variant=variant,
    detector_id=f"probe-{variant}",
    detector_version=f"0.1.0+{loaded.name.split('/')[-1]}",
    target_flag_rate=0.05,
)
print(source.summary())
"""
        ),
        code(
            """
from src.validation.metrics_builder import assert_metric_shape_compatible, build_warrant_metrics
from src.validation.synthetic import synthetic_cache, synthetic_evalset

fixture_evalset = synthetic_evalset(
    eval_set_id="shape-check-synthetic",
    n_items=len(result.short_evalset),
    base_rate=result.short_evalset.base_rate,
    seed=config.seed, items_per_question=1, declare_splits=True,
)
fixture = synthetic_cache(
    fixture_evalset, seed=config.seed,
    window=config.probe.rolling_window, stride=config.probe.rolling_stride,
)
fixture_metrics = build_warrant_metrics(
    config, fixture.labels, fixture.matrix(variant)[:, 0], 0.5,
    groups=fixture.question_ids,
)
assert_metric_shape_compatible(
    fixture_metrics, source.metrics,
    first_name="fixture path", second_name="measured extraction",
)
print("shape check passed — both paths produce the same metric structure")
"""
        ),
        code(
            """
from src.detectors.probe import LinearProbe
from src.validation.evalsets import TRAIN, split_by_question
from src.validation.runner import validate_transferred

if result.long_evalset is not None:
    splits = split_by_question(result.short_evalset, seed=config.seed)
    probe = LinearProbe(
        source.probe_fit.C,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
    ).fit(result.short_cache.matrix(variant), result.short_cache.labels, splits[TRAIN])

    transferred = validate_transferred(
        config, result.long_evalset, result.long_cache,
        source=source, probe=probe, variant=variant,
    )
    print(transferred.summary())
"""
        ),
        markdown(
            """
## 6 — What to download

Take all four. The caches are the expensive part and everything else in the
project runs from them on a laptop.

```
results/cache-triviaqa-600.npz
results/cache-triviaqa-longctx-600.npz
evalsets/triviaqa-600.json
evalsets/triviaqa-longctx-600.json
results/extraction.json
```

Then, locally:

```bash
python scripts/03_matrix.py --config config.yaml
```

The `Outstanding measurement` section of `RESULTS.md` removes itself once these
two envelopes appear, and the matrix cells stop reading `FIXTURE — NOT MEASURED`.
"""
        ),
        code(
            """
import shutil
shutil.make_archive("/kaggle/working/controlplane-extraction", "zip", ".",
                    base_dir=None, root_dir=".",
                    )
print("bundle written to /kaggle/working/controlplane-extraction.zip")
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "notebooks"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_on_kaggle.ipynb"
    path.write_text(
        json.dumps(build_kaggle_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
