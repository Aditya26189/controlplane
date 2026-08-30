"""Generate ``notebooks/run_on_kaggle.ipynb``. This script is its source of truth.

Notebook JSON is not reviewable in a diff — a one-character change to a cell
produces a diff full of execution counts and output blobs, and a reviewer cannot
tell a logic change from a re-run. So the notebook is generated from here, and
this file is what gets reviewed.

The notebook itself contains **no logic**. Every cell calls into ``controlplane/`` or
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


def _source_lines(text: str) -> list[str]:
    """Split into nbformat's `source` list, keeping the trailing newlines.

    nbformat defines `source` as a "multiline string": a list whose entries
    **concatenate** to the cell body, so every line except the last must end in
    a newline. Splitting on "\n" and discarding them produces a list that some
    readers join with newlines anyway — Kaggle does — and others concatenate
    directly, running the whole cell onto one line.

    That difference is why a notebook can execute fine in one place and be a
    syntax error in another, which is the worst kind of bug to ship into a
    3-hour GPU session.
    """
    lines = text.strip().split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
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
import os

# Long-context extraction allocates and frees large tensors per item, and the
# default allocator fragments under that pattern -- a 16k-token forward can fail
# for want of a contiguous block while plenty of total memory is free. Must be
# set before the first CUDA allocation, so this cell comes before load_model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Refuse an unsupported card immediately. Kaggle's machine_shape is a free
# string the server does not validate: a typo silently falls back to a P100,
# which is sm_60. This PyTorch build floors at sm_70 and bitsandbytes ships no
# sm_60 NF4 kernel, so the run does not fail here -- it dies two minutes later
# inside ops.cu with "named symbol not found" and a dead Jupyter kernel, which
# names neither the cause nor the fix.
if torch.cuda.is_available():
    _major, _minor = torch.cuda.get_device_capability(0)
    if (_major, _minor) < (7, 0):
        raise SystemExit(
            chr(10).join([
                "unsupported GPU: %s, compute capability %d.%d"
                % (torch.cuda.get_device_name(0), _major, _minor),
                "",
                "NF4 via bitsandbytes needs sm_70 or newer; this is a P100 if it",
                "says 6.0. The accelerator was not applied.",
                "",
                "Fix: push with machine_shape NvidiaTeslaT4 (case-sensitive --",
                "the server ignores an unrecognised value rather than erroring),",
                "or pick GPU T4 x2 in the notebook settings.",
            ])
        )
    print("GPU ok: %s, sm_%d%d" % (torch.cuda.get_device_name(0), _major, _minor))
else:
    raise SystemExit(
        "no GPU. This notebook loads a 7B model in NF4 and cannot run on CPU."
    )

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
## 1 — Get the repository, and prove it is the right project

**This repository contains two projects.** Round 2 — this one — is at the
repository root, in the `controlplane/` package. Round 1 is the earlier probe
experiment and lives under `round1/`, in a package still called `src/`. They
have separate `config.yaml` files and incompatible schemas: Round 1's
`ProbeConfig` has no `aggregations` field, so pointing this notebook at Round 1
fails several cells in, after the model has loaded.

Before 2026-08-29 the layout was inverted — Round 1 at the root, Round 2 under
`round 2/`. A Kaggle Dataset zipped from that layout will **not** be accepted:
its package is `src/`, the notebook's cells import `controlplane.*`, and the
mismatch would surface as a missing attribute an hour into a GPU session.
Re-zip from the current tree.

An earlier version of this cell fell back to whatever it found when the
expected directory was missing. That is the failure mode this project is
about: an absence reading as presence (`DECISIONS.md` 050). It refuses.

The clone below is pre-filled and should need no editing. If it does not
produce a usable project, the cell says what it searched and how to fix it
rather than proceeding with the wrong one.
"""
        ),
        code(
            """
import os, sys
from pathlib import Path

# Round 2 is the repository root on main. Clone it and the rest of the notebook
# finds it. Override only if you are running a fork or a work-in-progress
# branch; attaching the repo as a Kaggle Dataset also works, and CANDIDATES
# below searches /kaggle/input for it without any edit here.
CLONE_URL = "https://github.com/Aditya26189/controlplane.git"
CLONE_BRANCH = "main"

# The "round 2" entries are the pre-2026-08-29 layout. They are still SEARCHED,
# so that a stale Kaggle Dataset is named in the failure message below rather
# than silently missed -- but is_round_two() rejects them, because that layout
# ships a src/ package and every cell here imports controlplane.*.
CANDIDATES = [
    Path("/kaggle/working/controlplane"),
    Path("/kaggle/working/controlplane/round 2"),
    *sorted(Path("/kaggle/input").glob("*")),
    *sorted(Path("/kaggle/input").glob("*/round 2")),
    Path.cwd(),
    Path.cwd() / "round 2",
]

if CLONE_URL and not Path("/kaggle/working/controlplane").exists():
    !git clone --depth 1 --branch {CLONE_BRANCH} {CLONE_URL} /kaggle/working/controlplane


def is_round_two(path: Path) -> bool:
    # Round 2 has an extraction stage and a matrix; Round 1 has neither.
    return (
        (path / "config.yaml").is_file()
        and (path / "controlplane" / "extract").is_dir()
        and (path / "controlplane" / "matrix").is_dir()
    )


PROJECT = next((p for p in CANDIDATES if is_round_two(p)), None)
if PROJECT is None:
    problem = [
        "Round 2 not found.",
        "",
        "This repository holds two projects: Round 2 at the root, in the",
        "controlplane/ package, and Round 1 under round1/, in a package still",
        "called src/. They have separate config.yaml files and Round 1's",
        "ProbeConfig has no 'aggregations' field -- so pointing this notebook",
        "at Round 1 fails several cells in, after the model has loaded.",
        "",
        "If you attached a Kaggle Dataset zipped before 2026-08-29, it has the",
        "old layout (Round 2 under 'round 2/', package named src/) and is",
        "correctly refused: the cells below import controlplane.*. Re-zip from",
        "the current tree.",
        "",
        "Refusing to fall back to whatever is there: a wrong config that loads",
        "is worse than one that does not (DECISIONS.md 050).",
        "",
        "Looked in:",
    ]
    problem += ["  " + str(p) for p in CANDIDATES]
    problem += [
        "",
        "Fix, whichever is easiest:",
        "  - check CLONE_URL is reachable and CLONE_BRANCH exists. The clone is",
        "    silent on failure in some Kaggle images, so re-run the clone line",
        "    on its own and read what it prints.",
        "  - or zip the repository root, upload it as a Kaggle Dataset, and it",
        "    is found automatically under /kaggle/input/ with no edit here.",
    ]
    raise SystemExit(chr(10).join(problem))

# Drop any half-imported modules before adding the project to the path. Python
# caches by module name, so a stale 'controlplane' would shadow the right one
# and the failure would look like a missing attribute rather than a wrong
# import.
for name in [n for n in sys.modules if n == "controlplane" or n.startswith("controlplane.")]:
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
from controlplane.config import load_config, set_seeds, setup_logging

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
    raise SystemExit(chr(10).join([
        "This is not Round 2's config: " + str(missing) + " absent.",
        "Loaded from " + str(Path("config.yaml").resolve()),
        "  hash " + config.config_hash,
        "Round 1's config hash is c429ce5e92da9a22; Round 2's is b4ca1ec022266551.",
        "Go back to the previous cell: PROJECT is pointing at round1/, or at a",
        "Kaggle Dataset zipped from the pre-2026-08-29 layout.",
    ]))

print("config hash:", config.config_hash)
print("model:", config.model.name, "| quantization:", config.model.quantization)
print("aggregations:", list(config.probe.aggregations))

# Pre-registered in DECISIONS 065: all three in ONE session. last_token is the
# Round 1 anchor and cannot be recovered from a cache of pooled features, so a
# run without it cannot settle the cross-round comparison -- and a later run of
# last_token alone would compare against a different sample, split and label
# set, replacing one cross-configuration comparison with another.
_missing = {"mean_pool", "max_rolling_means", "last_token"} - set(
    config.probe.aggregations
)
if _missing:
    raise SystemExit(
        "config declares %s; missing %s. An unlisted aggregation raises "
        "nothing -- a shorter list is a valid list -- which is how Round 2 "
        "measured a different detector from Round 1 for an entire phase "
        "(DECISIONS 050, config enumeration)."
        % (list(config.probe.aggregations), sorted(_missing))
    )
print("rolling window/stride:", config.probe.rolling_window, config.probe.rolling_stride)
"""
        ),
        code(
            """
from controlplane.extract.model import load_model

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

**`checkpoint_dir="."` saves the short pass before long context is attempted.**
The short pass is the expensive half — generation dominates at 2.37 s an item —
and the long pass is the one that can exhaust the card. On the first real run an
OOM in long context discarded 17 minutes of completed short-context work; it
will not again. If the long pass fails now, the short results are already on
disk and the cell below re-runs only the long half.
"""
        ),
        code(
            """
from controlplane.extract.pipeline import extract_triviaqa

result = extract_triviaqa(
    config,
    loaded,
    n_questions=2400,          # 120 for a smoke run
    batch_size=8,
    long_batch_size=1,         # a 16k-token sequence does not batch on 16 GiB
    max_new_tokens=32,
    long_context=True,
    checkpoint_dir=".",        # saves the short pass BEFORE long context runs
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
### If the long-context pass ran out of memory

The short pass is checkpointed, so re-run only the long half. Three levers, in
the order worth trying:

**Long context is now chunked, so this should not depend on the backend at
all.** Prefill runs in `model.prefill_chunk_tokens`-sized pieces carrying a KV
cache, which bounds the attention workspace to `heads × chunk × seq` instead of
`heads × seq²` — 2.55 GiB rather than 12.86 GiB at 11k tokens, on *any* kernel.
Causal attention makes it exact, not approximate.

The pre-flight tells you in seconds rather than in forty minutes, and now logs
the attention implementation and chunk size before it runs.

**If you re-ran only this cell, `loaded` is the model from the original load**
and predates the `attn_implementation="sdpa"` pin. Chunking covers that, but
re-run the load cell to clear the warning.

**The long pass now saves every 25 items** to
`results/cache-longctx-partial.npz` and resumes from it, so an interrupted run
costs at most 25 items rather than everything done so far.

**If it is running but too slow to finish**, raise `model.prefill_chunk_tokens`
in `config.yaml`. NF4 dequantises the full weight set on *every* forward, so the
chunk count multiplies that cost directly:

| chunk | forwards/item at 7k tok | attention peak at 11k tok |
|---|---|---|
| 2048 | 4 | 2.37 GiB |
| 4096 | 2 | 4.74 GiB |
| 6144 | 2 | 7.12 GiB |

4096 halves the dequant overhead and still fits comfortably in the ~10 GiB free
on a T4.

`_extract_long_context` now runs the **longest prompt alone** before the loop
and logs measured peak memory per card. The worst case is not item 0, which is
why every earlier failure surfaced deep into the run.

What was actually wrong, after measuring rather than arguing (DECISIONS 057):

| | verdict |
|---|---|
| logits `(seq × 152,064)` | **the cause.** The forward ran `Qwen2ForCausalLM`, applying `lm_head` at every position and discarding it. Now runs the trunk. |
| attention `seq²` | never paid. With sdpa no square tensor is materialised at all. |
| the attention mask | already skipped by transformers when it is all ones. That "fix" changed nothing. |

The one risk left is the attention implementation: on the **eager** path a
`heads × seq × seq` matrix is real and the softmax upcasts to float32 — 28.7 GiB
for a single op at 16k tokens. `load_model` now asks for sdpa explicitly and
refuses to load on anything else, so that cannot arrive silently.

If it still fails, read the peak figures in the error against the two terms it
prints, then narrow `evalsets.pad_tokens` in `config.yaml` (currently
`[4000, 16000]`) as a last resort, accepting a weaker envelope shift.

Flash-Attention-2 does not help: it needs sm_80 and a T4 is sm_75.
"""
        ),
        code(
            """
# Recovery only. Guarded, because running this after a successful pass repeats
# three hours of work -- which is exactly how one session was spent.
import torch
from controlplane.extract.pipeline import _extract_long_context

if result.long_cache is not None:
    print("long-context pass already complete:",
          result.long_evalset.eval_set_id, len(result.long_evalset),
          "-- nothing to do")
else:
    torch.cuda.empty_cache()
    print({i: f"{torch.cuda.mem_get_info(i)[0]/2**30:.1f} GiB free"
           for i in range(torch.cuda.device_count())})
    long_evalset, long_cache = _extract_long_context(
        config, loaded, result.short_evalset,
        layer=result.report["layer"], batch_size=1,
        checkpoint_dir=".",
    )
    # ExtractionResult is a plain mutable object, so assign directly. An
    # earlier version used _replace() behind a hasattr guard, which would have
    # silently left `result` stale and sent section 7 to validate nothing.
    result.long_evalset, result.long_cache = long_evalset, long_cache
    print(long_evalset.eval_set_id, len(long_evalset), long_evalset.envelope_id)
    print("token lengths:", long_cache.token_lengths.min(),
          "to", long_cache.token_lengths.max())
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
from controlplane.evalsets import save_evalset

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
from controlplane.validation.runner import validate

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
from controlplane.validation.metrics_builder import assert_metric_shape_compatible, build_warrant_metrics
from controlplane.validation.synthetic import synthetic_cache, synthetic_evalset

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
from controlplane.detectors.probe import LinearProbe
from controlplane.validation.evalsets import TRAIN, split_by_question
from controlplane.validation.runner import validate_transferred

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
## 6 — Validate, build the warrant matrix, and run the banking pilot

The measured numbers. Each script is a thin wrapper over `controlplane/`; they are run
as subprocesses with `check=True` so a failure stops the notebook rather than
leaving a run that looks successful and is missing artifacts.

Two eval sets, two envelopes. `triviaqa-600` is the anchor; `triviaqa-longctx-600`
is the same questions padded to 4–16k tokens, and the difference between them is
the envelope shift the drift story rests on.

**The last stage is the banking pilot** (`DECISIONS.md` 090 corrected, 101). It
is different in kind from everything above it: the others produce measured
artifacts, and this one produces a *decision* about whether a 240-item set is
worth authoring. It runs last so a failure there cannot cost the TriviaQA
numbers, and it reports its branch rather than taking one — one of the three
outcomes consumes the single retry `090` allows, and that is a person's call.

Its correctness labels are **measured here**, not authored: the 24 prompts were
frozen on CPU with gold answers and no labels, and this pass generates, judges,
and only then scores.
"""
        ),
        markdown(
            """
### First, give the GPU back

The extraction above holds a 7B model **in this kernel process**. Every stage
below runs as a *subprocess*, so they do not inherit that allocation -- they
compete with it. `13_pilot_run.py` is the only stage that loads a model of its
own, and on the 2026-08-30 run it died with `CUDA out of memory: GPU 1 has
14.56 GiB of which 46.81 MiB is free. Process 23 has 11.00 GiB in use` after
the other five stages had already succeeded. Process 23 was this notebook.

That is the second time the pilot has killed a three-hour run at the last
stage, the first being a wrong argument list. Both were cheap to check and
expensive to discover, so this cell frees the memory **and refuses to continue
without it** rather than letting the stages start and find out at the end.
"""
        ),
        code(
            """
import gc

import torch

# The extraction is finished and its caches are on disk; nothing below reads
# `loaded`. Dropping every reference is what actually frees the weights --
# empty_cache() alone only returns already-unreferenced blocks to the driver.
for name in ("loaded", "probe", "source", "transferred"):
    if name in dir():
        del globals()[name]
gc.collect()
torch.cuda.empty_cache()

free = {i: torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())}
print({i: f"{g:.1f} GiB free" for i, g in free.items()})

# The pilot loads Qwen2.5-7B in NF4 (~5 GiB) plus an activation workspace. If
# the largest card cannot offer that, the five cheap stages below would run for
# minutes and the expensive one would fail at the end -- which is exactly the
# failure this cell exists to convert into an immediate one.
NEEDED_GIB = 6.0
if free and max(free.values()) < NEEDED_GIB:
    raise SystemExit(
        f"only {max(free.values()):.1f} GiB free on the largest GPU; the pilot "
        f"needs about {NEEDED_GIB} GiB. Something still holds the weights -- "
        "restart the kernel and re-run rather than spending the stages below."
    )
"""
        ),
        code(
            """
import subprocess, sys

STAGES = [
    # The canary first: canary_control fails CLOSED without one, so every
    # warrant below would be refused on that control alone.
    ["scripts/05_canary.py", "--config", "config.yaml",
     "--cache", "results/cache-triviaqa-600.npz",
     "--eval-set", "triviaqa-600", "--variant", "T1-last_token"],
    # Validate ONLY on the envelope that has a train split. 02_validate FITS a
    # probe; triviaqa-longctx-600 is test-only by design (DECISIONS 051), so
    # running it there raises "train_index is empty" -- and had it not raised,
    # it would have refitted on long context and answered the wrong question.
    ["scripts/02_validate.py", "--config", "config.yaml",
     "--cache", "results/cache-triviaqa-600.npz",
     "--eval-set", "triviaqa-600"],
    # The long-context envelope is a TRANSFER: score the already-fitted probe.
    ["scripts/04_transfer.py", "--config", "config.yaml",
     "--source-cache", "results/cache-triviaqa-600.npz",
     "--target-cache", "results/cache-triviaqa-longctx-600.npz"],
    # The pre-registered branch rule (DECISIONS 065), executed not applied.
    ["scripts/06_reconcile.py", "--config", "config.yaml",
     "--cache", "results/cache-triviaqa-600.npz",
     "--eval-set", "triviaqa-600"],
    ["scripts/03_matrix.py", "--config", "config.yaml"],
    # The banking pilot. Runs LAST because it is the only stage whose outcome
    # is a decision rather than an artifact, and because a failure here must
    # not cost the measured TriviaQA numbers above.
    #
    # It generates 24 answers, judges them against gold aliases, checks the
    # acceptance band BEFORE scoring anything, extracts question-time
    # activations, and computes the IQR ratio against the envelope the probe
    # was fitted on. It reports a branch; it does not take one.
    ["scripts/13_pilot_run.py", "--config", "config.yaml",
     "--cache", "results/cache-triviaqa-600.npz"],
]
for stage in STAGES:
    print("=" * 70)
    print(">>>", " ".join(stage))
    done = subprocess.run([sys.executable, *stage], text=True,
                          capture_output=True)
    print(done.stdout[-4000:])
    if done.returncode != 0:
        print(done.stderr[-4000:])
        raise SystemExit("stage failed: " + " ".join(stage))
"""
        ),
        code(
            """
from pathlib import Path

matrix = Path("results/warrant_matrix.md")
print(matrix.read_text() if matrix.exists() else "no warrant matrix written")
"""
        ),
        markdown(
            """
## 7 — What to download

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
import shutil, os
from pathlib import Path

# results/ and evalsets/ only. Zipping "." would sweep in .git and the model
# cache, and Kaggle's output has a size cap.
staging = Path("/kaggle/working/bundle")
shutil.rmtree(staging, ignore_errors=True)
for name in ("results", "evalsets"):
    if Path(name).exists():
        shutil.copytree(name, staging / name)
archive = shutil.make_archive("/kaggle/working/controlplane-results", "zip",
                              root_dir=str(staging))
print("bundle:", archive, "%.1f MiB" % (os.path.getsize(archive) / 2**20))
for path in sorted(staging.rglob("*")):
    if path.is_file():
        print("  %-56s %8.2f MiB" % (path.relative_to(staging),
                                     path.stat().st_size / 2**20))
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


REFERENCE_DATASET = "controlplane-reference-caches"


def build_pilot_notebook() -> dict:
    """The banking pilot ALONE, against cached reference activations.

    **Why this exists.** The pilot sat at the end of a three-hour monolith with
    no checkpoint. Nine GPU-hours were spent across three attempts and it
    produced no pilot result: once on a wrong argument list, once on an OOM
    caused by the extraction never releasing the 7B model, and once before
    that. Each attempt re-ran a three-hour extraction whose output had not
    changed in order to reach a stage that takes twenty minutes.

    The activations are a pure function of (model, eval set, layer). They do
    not need recomputing to score a different set of prompts, so they are
    published as a Kaggle dataset and attached read-only. This turns the pilot
    into a twenty-minute job that can be retried.

    The manifest travels with the caches and is **checked**, not trusted: the
    eval-set content hashes in the dataset must match the ones this checkout
    builds, or the reference scores would come from a different envelope than
    the one the warrant names.
    """
    cells = [
        markdown(
            """
# ControlPlane.ai — the banking pilot, standalone

Runs `scripts/13_pilot_run.py` **only**, against reference activations attached
as a dataset instead of re-extracted. Twenty minutes rather than three hours.

**Attach the dataset before running:** `aditya103856/controlplane-reference-caches`
(Add Input → Datasets). It carries `cache-triviaqa-600.npz` and the eval sets
the reference probe is fitted on, under a manifest of SHA-256s and content
hashes.

Nothing here re-extracts. If the caches are wrong, the run refuses rather than
recomputing them — recomputing is the three-hour path this notebook exists to
avoid, and it should be a deliberate choice, not a fallback.
"""
        ),
        code(
            """
import subprocess, sys, os
from pathlib import Path

CLONE_URL = "https://github.com/Aditya26189/controlplane.git"
CLONE_BRANCH = "main"

os.chdir("/kaggle/working")
if not Path("controlplane").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", CLONE_BRANCH,
                    CLONE_URL, "controlplane"], check=True)
os.chdir("/kaggle/working/controlplane")
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True,
                     text=True).stdout.strip())
"""
        ),
        code(
            """
!pip install -q bitsandbytes accelerate 2>&1 | tail -2
"""
        ),
        markdown(
            """
## 1 — Attach the caches, and check them

The manifest is verified against this checkout. A cache built from a different
eval set would still load and still produce numbers; only the content hash
catches that, and it is the same argument as the pilot's own draft-hash guard.
"""
        ),
        code(
            """
import hashlib, json, shutil, sys
from pathlib import Path

sys.path.insert(0, "/kaggle/working/controlplane")

# Do not assume the mount path. Kaggle derives the input directory from the
# dataset's slug, and version 1 of this kernel died on a hardcoded
# /kaggle/input/controlplane-reference-caches that did not exist -- with the
# metadata correctly listing the dataset. Find the manifest instead, and print
# what IS mounted so the next failure diagnoses itself.
INPUT = Path("/kaggle/input")
# rglob, not glob: Kaggle mounted this dataset at
# /kaggle/input/datasets/<owner>/<slug>/, not the /kaggle/input/<slug>/
# the docs imply. The depth is not worth predicting -- search for the
# manifest wherever it landed.
candidates = sorted(INPUT.rglob("MANIFEST.json")) if INPUT.exists() else []
if not candidates:
    print("nothing mounted under /kaggle/input carries a MANIFEST.json.")
    print("what IS mounted:")
    if INPUT.exists():
        for entry in sorted(INPUT.rglob("*"))[:40]:
            print("   ", entry.relative_to(INPUT))
    else:
        print("    (/kaggle/input does not exist)")
    raise SystemExit(
        "Add Input -> Datasets -> "
        "aditya103856/controlplane-reference-caches, then re-run. Refusing to "
        "re-extract: that is the three-hour path this notebook replaces."
    )
SOURCE = candidates[0].parent
print("reference caches mounted at", SOURCE)

manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
print("caches built from commit", manifest["built_from_commit"][:8])

Path("results").mkdir(exist_ok=True)
Path("evalsets").mkdir(exist_ok=True)
for item in sorted(SOURCE.iterdir()):
    if item.suffix == ".npz":
        shutil.copy2(item, Path("results") / item.name)
    elif item.suffix == ".json" and item.name != "MANIFEST.json":
        shutil.copy2(item, Path("evalsets") / item.name)

# SHA-256 of every cache, against the manifest. Cheap, and the alternative is
# discovering a truncated upload after the model has loaded.
for name, meta in manifest["files"].items():
    digest = hashlib.sha256((Path("results") / name).read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        raise SystemExit(f"{name} sha256 {digest} != manifest {meta['sha256']}")
    print(f"  {name:<34} {meta['bytes'] / 2**20:7.1f} MiB  sha256 ok")

# The eval sets must be the ones THIS checkout believes in, or the reference
# scores describe a different envelope than the warrant names.
from controlplane.evalsets.registry import load_evalset

for eval_set_id, meta in manifest["evalsets"].items():
    local = Path("evalsets") / f"{eval_set_id}.json"
    if not local.exists():
        continue
    got = load_evalset(local).content_hash
    if got != meta["content_hash"]:
        raise SystemExit(
            f"{eval_set_id}: content hash {got[:16]} != manifest "
            f"{meta['content_hash'][:16]}. The cached activations were "
            "extracted from a different eval set."
        )
    print(f"  {eval_set_id:<34} content hash ok ({meta['n_items']} items)")
"""
        ),
        markdown(
            """
## 2 — The GPU is untouched

No extraction ran in this kernel, so the card should be empty. Checked anyway,
because the assumption that it was free is what cost the 2026-08-30 run: the
full notebook held 11.00 GiB of a 14.56 GiB card in the kernel process while
the pilot tried to load its own copy in a subprocess.
"""
        ),
        code(
            """
import torch

free = {i: torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())}
print({i: f"{g:.1f} GiB free" for i, g in free.items()})

NEEDED_GIB = 6.0
if not free:
    raise SystemExit("no GPU visible; the pilot generates and extracts and needs one")
if max(free.values()) < NEEDED_GIB:
    raise SystemExit(
        f"only {max(free.values()):.1f} GiB free; the pilot needs about "
        f"{NEEDED_GIB} GiB. Restart the kernel."
    )
"""
        ),
        markdown(
            """
## 3 — The pilot

Generates 24 answers, judges them against gold aliases, checks `101`'s
acceptance band **before** scoring anything, extracts question-time
activations, and computes the IQR ratio against the reference envelope.

It reports a branch; it does not take one. Its first act is to verify that the
draft it rebuilt from `BANKING_PILOT_QUESTIONS` matches the committed freeze —
that guard passed on the 2026-08-30 run at hash `312516ded744e4fe`, before the
OOM killed the stage after it.
"""
        ),
        code(
            """
import subprocess, sys

stage = ["scripts/13_pilot_run.py", "--config", "config.yaml",
         "--cache", "results/cache-triviaqa-600.npz"]
print(">>>", " ".join(stage))
done = subprocess.run([sys.executable, *stage], text=True, capture_output=True)
print(done.stdout[-8000:])
if done.returncode != 0:
    print(done.stderr[-8000:])
    raise SystemExit("pilot failed")
"""
        ),
        code(
            """
from pathlib import Path

for name in ("pilot_run.json", "pilot_envelope.json"):
    path = Path("results") / name
    print("=" * 70)
    print(name, "--", "present" if path.exists() else "ABSENT")
    if path.exists():
        print(path.read_text()[:3000])
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _strip_magics(lines: list[str]) -> list[str]:
    """Replace IPython magics with ``pass``, without mangling Python.

    Two traps, both found by running this guard on notebooks that were already
    correct:

    ``pass`` rather than a blank line, at the **same indent**. A magic is often
    the only statement in an ``if`` body, and blanking it turns a valid cell
    into "expected an indented block".

    A leading ``%`` is only a magic at the start of a *logical* line. Inside an
    open bracket it is the modulo/format operator, and
    ``% (name, major, minor),`` continuing a ``"..." % (...)`` expression is
    not a magic however much it looks like one. Bracket depth is tracked so
    continuations are left alone.

    Args:
        lines: The cell's source, already split on newlines.

    Returns:
        The same lines with top-level magics replaced.
    """
    out: list[str] = []
    depth = 0
    for line in lines:
        stripped = line.lstrip()
        if depth == 0 and stripped.startswith(("!", "%")):
            out.append(line[: len(line) - len(stripped)] + "pass")
        else:
            out.append(line)
        # Crude but sufficient: string literals in these cells do not carry
        # unbalanced brackets, and a miscount only costs a false positive that
        # the numbered traceback makes obvious.
        depth += line.count("(") + line.count("[") + line.count("{")
        depth -= line.count(")") + line.count("]") + line.count("}")
        depth = max(depth, 0)
    return out


def _assert_cells_compile(notebook: dict, name: str) -> None:
    """Every code cell must parse before the notebook is written.

    A generated cell is assembled from nested string literals, and escaping
    survives one layer and not the next: a backslash-n meant to be two
    characters inside the generated source arrives as a real newline, splitting
    a string literal across lines. That shipped twice. The second time it cost
    a Kaggle round trip to discover an IndentationError that ``compile()``
    reports in microseconds -- and then it happened a third time, in the first
    draft of *this function*, which is the argument for it existing.

    Cells are compiled with IPython magics stripped, since ``!pip`` and ``%``
    are notebook syntax rather than Python.

    Args:
        notebook: The nbformat dict about to be written.
        name: Filename, for the error message.

    Raises:
        SyntaxError: With the cell index and the offending source, numbered.
    """
    newline = chr(10)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        cleaned = newline.join(_strip_magics(source.split(newline)))
        try:
            compile(cleaned, f"{name}:cell{index}", "exec")
        except SyntaxError as exc:
            numbered = newline.join(
                f"{i:>4}| {line}"
                for i, line in enumerate(source.split(newline), 1)
            )
            raise SyntaxError(
                f"{name} cell {index} does not compile: {exc}{newline}{numbered}"
            ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "notebooks"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("run_on_kaggle.ipynb", build_kaggle_notebook),
        ("run_pilot_on_kaggle.ipynb", build_pilot_notebook),
    ):
        notebook = builder()
        _assert_cells_compile(notebook, name)
        path = out / name
        path.write_text(
            json.dumps(builder(), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
