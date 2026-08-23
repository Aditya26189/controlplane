# Glossary

Terms used precisely in this repository. Where a term is easy to slide into a stronger claim than the evidence supports, the wrong reading is given too.

---

### Question-time activation

The residual-stream vector at the **last token of the prompt**, taken after the model has read the question and before it has generated anything. In practice: `hidden_states[layer][:, -1, :]` from a prefill-only forward pass, with the batch left-padded so position −1 is the true final prompt token for every row.

*Not*: an activation taken mid-generation, or from the answer. `CLAUDE.md` invariant 1.

### Probe

A logistic regression over one layer's question-time activations, with features standardised by a scaler fit on **train only**. It is the only thing trained in this repository; the language model is frozen throughout.

*Not* a truth detector, a hallucination detector, or a measure of what the model believes. It is a correlational classifier over activations.

### Trigger vs verdict

A **trigger** decides where to spend an expensive check. A **verdict** decides what happens to a response. The probe is only ever a trigger — it never blocks, edits, or gates anything. That distinction is what makes low precision acceptable by design.

### Cascade

The tiered checking architecture the competition concept proposes: deterministic checks and cheap signals on 100% of traffic, the probe on 100% of traffic at negligible cost, and an expensive verdict on the small slice the probe selects. **This repository implements and measures the probe tier only.**

### Flag rate — `f`

The fraction of responses the probe flags for an expensive check. Two versions, never interchangeable:

- **target** flag rate — `economics.target_flag_rate` in the config, what the threshold is tuned on validation to hit;
- **measured** flag rate — what the frozen threshold actually produces on test.

Every downstream calculation uses the **measured** one (`CLAUDE.md` invariant 6). Confusing the two is one of the pitfalls the repo explicitly guards against.

### Recall — `R`

Of all responses that were actually wrong, the fraction the probe flagged. The positive class is **incorrect**, so recall is "how many of the errors did we catch". Reported with a bootstrap interval, never alone.

### Precision

Of all responses the probe flagged, the fraction that were actually wrong. Reported separately from recall, always. There is no F1 anywhere in this codebase — blending the two hides which failure mode you have, and the two differ in cost by orders of magnitude.

### Base rate / base error rate

The fraction of responses that were **wrong**. Reported alongside every headline number, because an unbalanced label distribution makes accuracy meaningless: if a model is right 85% of the time, a probe that always predicts "correct" scores 0.85 accuracy and 0.5 AUROC.

Note that `results/RESULTS.md` §2 reports it over the whole sampled dataset, while the README's result table reports it for the **test split** — different sets of examples, so the two differ slightly. Each document says which it is showing.

### Lift

```
lift = R / f
```

How many more errors the probe catches than random sampling at the **same budget** of expensive checks. Equivalently `precision / base_rate`. Both the base error rate and the judge's accuracy cancel from this ratio, which is why the multiplier does not rest on an assumption about production error rates.

### Lift ceiling

`1 / base_rate`. Since `lift = precision / base_rate` and precision cannot exceed 1, no probe can beat this bound on a given workload. A hard benchmark with a high error rate has a low ceiling, so a measured lift must always be read against it — "89% of the maximum attainable here" and "2.3×" are the same fact stated with and without its scale.

### AUROC

Area under the ROC curve: the probability that a randomly chosen wrong response is scored higher than a randomly chosen correct one. Base-rate independent, which makes it the **transferable** quantity in this repo — lift is specific to a workload's error rate, AUROC is not.

### Polarity

The convention that `y = 1` means **the generated answer was wrong**. Fixed once and held everywhere, because inverting it silently produces `1 − AUROC` — a number that reads as a catastrophic negative result rather than as a bug. `src/probe.py::assert_polarity` guards it; `DECISIONS.md` 004.

### Left padding

Padding batched sequences on the left so that position −1 is the real final token for every row regardless of length. With right padding, position −1 is a pad token, every activation is garbage, and **nothing raises** — the AUROC simply lands near chance. `CLAUDE.md` invariant 4, and the reason for the equivalence check below.

### Equivalence check and positive control

Before every extraction, the same batch is compared batched-vs-unbatched under left padding (must **pass**, on relative L2 and cosine criteria) and again under deliberate right padding (must **fail**). The second half is the positive control: without it, a permissive limit is indistinguishable from a limit that was loosened until it passed. Both rows are published. `DECISIONS.md` 014.

### Selection discipline

The rule that no choice — layer, regularisation strength `C`, threshold — may consult the test set, that every scoring of test is appended to `results/test_scoring_log.json` and disclosed, and that any re-scoring is pre-registered in `DECISIONS.md` with the prior numbers before it runs. `CLAUDE.md` invariant 2; history in `DECISIONS.md` 006, 016, 017.

### Layer sweep

Training the probe at each candidate depth and comparing **validation** AUROC. The full grid is published rather than just the winner: a smooth curve peaking mid-stack is itself evidence the signal is real rather than noise.

### Grid boundary

When the best `C` is the smallest or largest value in the grid, the search hit a wall rather than found an optimum. `probe_sweep.json` records `winner_at_grid_boundary` for exactly this reason; a boundary winner means widening the grid, not reporting the result as final. `DECISIONS.md` 016, 017.

### Provenance

The block embedded in every artifact: UTC timestamp, git commit and branch, whether the working tree was **dirty**, Python and library versions, device, seed, config hash, and the fully resolved config. It is what turns a published number into something a stranger can check.

### Config hash

SHA-256 of the resolved configuration, embedded in every artifact. If two artifacts carry different hashes, they do not describe one experiment, and `src/report.py::config_hash_consistency` says so in the report rather than leaving it to be noticed.

### Smoke run

`run_all.py --smoke`: the full chain at `n_examples=100`, writing into `results/smoke/` so it can never overwrite a real run. Its safety gates are **not** relaxed — a smoke mode that disables its own checks tests nothing worth testing.

### Judge

The expensive check the probe is triaging for: LLM-as-judge, semantic entropy, claim attribution — anything in the 200–1000 ms range. Its accuracy `a` is a config parameter, kept explicit precisely to show that it cancels out of the lift ratio.

### Abstention

A generation containing a pattern like "I don't know" or "I'm not sure", detected on the normalised string. Used as secondary validation: if the probe direction also tracks the model's *expressed* uncertainty, that is independent evidence it reads something real rather than a dataset artifact. Below the configured rate, the comparison is marked underpowered rather than reported as a number.

### Negative control

A run deliberately expected to produce a **weak** result — here, GSM8K, since the published result reports that this method's generalisation falters on mathematical reasoning. Optional Stage 6, **not implemented**: `config.yaml` reserves the settings but no code reads them yet.

---

**See also:** [ARCHITECTURE.md](ARCHITECTURE.md) · [FAQ.md](FAQ.md) · [../SPEC.md](../SPEC.md) for the formal definitions these summarise.
