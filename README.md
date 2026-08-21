# ControlPlane — cascade economics

Can you tell, before a language model writes a single token, whether the answer it is about to give will be wrong?

If you can, monitoring gets much cheaper. Serious checkers — LLM-as-judge, semantic entropy, claim attribution — cost 200 to 1000 milliseconds per call, so nobody runs them on all their traffic. They sample a few percent and the rest ships unchecked. This repo measures whether a linear probe on the model's internal state can pick which few percent are worth checking.

The number it exists to produce is **lift = R / f** — the probe's recall divided by its measured flag rate. How many more errors it catches than random sampling at the same judge budget.

## Status: pipeline complete, awaiting the GPU run

Every stage is implemented and tested. The extraction stage needs a CUDA GPU, which the development machine does not have, so **no measured result has been produced yet**.

This file is replaced wholesale by `scripts/05_report.py` once the pipeline runs. It contains no measured numbers, deliberately: nothing gets written here that was not produced by a script in this repo, with a seed and a config hash beside it.

To produce the result:

```bash
pip install -r requirements.txt
python scripts/run_all.py --config config.yaml
```

Or open [`notebooks/run_on_kaggle.ipynb`](notebooks/run_on_kaggle.ipynb) on a free Kaggle T4, which walks through the stage gates, the pre-flight checks, and the full run.

## What runs where

| Stage | Script | Needs a GPU |
|---|---|---|
| Data — load, deduplicate, split by question | `01_extract.py --data-only` | no |
| Extraction — activations and greedy generations | `01_extract.py` | **yes** |
| Probe — layer sweep, threshold, one test scoring | `02_train_probe.py` | no |
| Economics — three policies, lift | `03_economics.py` | no |
| Latency — probe cost vs generation cost | `04_latency.py` | no |
| Report — RESULTS.md, README.md, plots | `05_report.py` | no |

Everything except extraction runs on a laptop from the artifacts on disk, so a stage can be re-run without repeating the expensive one:

```bash
python scripts/run_all.py --from 02
```

## Method, in short

1. The model reads a question. It has generated nothing yet.
2. At one mid-stack layer, take the residual-stream vector at the final prompt token — a by-product of the prefill pass the model performs anyway.
3. A logistic regression scores it. One dot product.
4. Responses above the threshold go to the expensive checker. The rest don't.

The probe is a **trigger**, not a verdict. It never blocks anything, which is why it is tuned for recall and why low precision is acceptable: a false positive costs one wasted judge call, a false negative costs a user acting on a wrong answer.

## Reading the repo

- [`CLAUDE.md`](CLAUDE.md) — the invariants. Start here.
- [`SPEC.md`](SPEC.md) — the technical specification.
- [`DECISIONS.md`](DECISIONS.md) — every methodological choice and why, including the ones a reviewer would challenge.
- [`TASKS.md`](TASKS.md) — the staged build order and its gates.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — git and documentation rules.

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs offline with no model download: the padding-equivalence test uses a 41k-parameter Qwen2 built in the test fixture, and the pipeline smoke test runs stages 02–05 against a synthetic extraction.

## Licence

MIT.
