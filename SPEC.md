# SPEC.md — technical specification

Companion to `CLAUDE.md`. This document pins down the parts that are easy to get subtly wrong. Reference snippets are illustrative, not drop-in — adapt to the module structure, but preserve the semantics exactly.

---

## 1. Data

**Source:** `mandarjoshi/trivia_qa`, config `rc.nocontext`, split `validation`.

Use `rc.nocontext` — the no-context variant. The model must answer from parametric knowledge, which is precisely the regime where "does it know this?" is a meaningful question. If you pass the supporting document, the task becomes reading comprehension and the probe is measuring something else entirely.

**Fields used:** `question_id`, `question`, `answer.value`, `answer.aliases`, `answer.normalized_aliases`.

**Pipeline:**

1. Load the split.
2. Normalise each question string (lowercase, collapse whitespace, strip punctuation) and drop duplicates, keeping the first occurrence. Log how many were dropped.
3. Drop rows with an empty question or an empty alias list.
4. Shuffle with the configured seed.
5. Take `config.data.n_examples` (default 3000).
6. Split **by `question_id`** into train/val/test at 60/20/20.
7. Assert the three `question_id` sets are pairwise disjoint. Assert no normalised question string appears in more than one split.

Persist the split assignment to `results/splits.parquet` so every downstream stage uses the same one.

## 2. Answer normalisation and labelling

Follow the SQuAD/TriviaQA convention:

```python
def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
```

**Matching rule.** The model generates a sentence; the gold answer is a short span. Pure exact match would label almost everything incorrect. So:

```python
def is_correct(prediction: str, aliases: list[str]) -> bool:
    pred = normalize_answer(prediction)
    if not pred:
        return False
    for alias in aliases:
        a = normalize_answer(alias)
        if not a:
            continue
        if len(a) < 3:
            # Short aliases ("US", "UK") match spuriously inside longer text.
            # Require a whole-token match instead of substring containment.
            if a in pred.split():
                return True
        elif a in pred:
            return True
    return False
```

**Also compute and record strict exact match** (`normalize_answer(prediction) == normalize_answer(alias)`) as a secondary column. The README reports the lenient rule as primary and states the strict number alongside, so a reviewer can see the labelling choice didn't manufacture the result. If the two disagree by more than ~10 percentage points in base rate, say so explicitly in `RESULTS.md`.

**Sanity gate.** After labelling, print the base rate (fraction correct). Expect roughly 0.45–0.70 for Qwen2.5-7B on TriviaQA no-context. If it lands outside 0.25–0.85, stop and investigate — either the prompt is malformed, generation is truncated, or the matching rule is broken. Do not proceed to probe training on a degenerate label distribution.

## 3. Prompting

Use the tokenizer's chat template. Keep the system prompt minimal and fixed; it is part of the experimental condition and must not vary.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer the question concisely."},
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

`add_generation_prompt=True` appends the assistant turn header. The final prompt token is therefore the last token before the model begins its answer — which is exactly the position we probe. Record the exact template string in the output artifacts.

## 4. Activation extraction

**This is the core of the experiment. Get it exactly right.**

Two passes per batch:

**Pass 1 — prefill only, for activations.** A plain forward call with `output_hidden_states=True`. This gives every layer in one shot, which makes the layer sweep essentially free.

**Pass 2 — generation, for labels.** A standard `generate()` call.

Do **not** combine them by passing `output_hidden_states=True` into `generate()`. That retains hidden states for every decoding step and will exhaust a T4's memory. The extra prefill is one forward pass against ~32 decode steps — a rounding error in total runtime.

```python
tokenizer.padding_side = "left"          # invariant 4 — assert this
assert tokenizer.padding_side == "left"

enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

# Pass 1: activations
with torch.no_grad():
    out = model(**enc, output_hidden_states=True, use_cache=False)

# out.hidden_states is a tuple of length (n_layers + 1):
#   index 0   = embedding output
#   index L   = output of transformer block L  (1-indexed)
# each element has shape (batch, seq_len, hidden_size)

for layer in config.model.layers:
    # Left padding guarantees position -1 is the true final prompt token
    # for every sequence in the batch, regardless of its length.
    acts[layer].append(out.hidden_states[layer][:, -1, :].float().cpu().numpy())

# Pass 2: labels
with torch.no_grad():
    gen = model.generate(
        **enc,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=False,                 # greedy — deterministic
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
prompt_len = enc["input_ids"].shape[1]
completions = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
```

**Assertions to write into the extraction loop:**

- `tokenizer.padding_side == "left"`
- `len(out.hidden_states) == model.config.num_hidden_layers + 1`
- every requested layer index is within range
- `acts[layer].shape == (batch, model.config.hidden_size)`
- the activation tensor contains no NaN or Inf
- **left-padding correctness check:** for one batch, run each sequence individually with no padding and confirm the layer-`L` last-token activation matches the batched version to within `1e-2`. Attention masking makes this exact in principle and near-exact in fp16. If it doesn't match, padding is wrong. Run this check once at startup on a batch of 4 and fail hard if it doesn't pass.

**Layers to extract.** Predictive power saturates in the middle of the stack. Default for a 28-layer model: `[8, 11, 14, 17, 20, 23, 26]`. Express in config as fractional depths so the same config works across model sizes.

**Storage.** `results/activations.npz`, fp16, keyed by layer. 3000 examples × 7 layers × 3584 dims × 2 bytes ≈ 150 MB. Add `results/activations.npz` to `.gitignore`; commit only the derived JSON.

**Batching.** Default batch size 8 on a T4. Sort by prompt length before batching to cut padding waste, then restore original order before saving — and assert the restoration by carrying `question_id` through and checking it round-trips.

## 5. The probe

**Features:** the activation vector at one layer. **Label:** `1` if the generated answer was incorrect (so the positive class is "this is going to be wrong" — the thing we want to catch).

Be consistent about this polarity everywhere. Getting it backwards silently produces `AUROC = 1 - true_auroc`, which looks like a catastrophic result rather than an error. Write it in the docstring and assert on a sanity example.

```python
scaler = StandardScaler().fit(X_train)          # train only — never fit on val/test
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=config.probe.C,
    random_state=config.seed,
)
clf.fit(scaler.transform(X_train), y_train)
```

**Layer sweep.** For each candidate layer, train on train, evaluate AUROC on **validation**. Pick the winner on validation. Record the full sweep in `results/probe_sweep.json` — a table of all layers, not just the winner, because a reviewer wants to see the shape of the curve, and a smooth curve peaking mid-stack is itself evidence the signal is real rather than noise.

**Regularisation.** Small grid over `C ∈ {0.001, 0.01, 0.1, 1.0}` on validation. With ~1800 training examples and ~3584 features, this is a high-dimensional, low-sample regime — regularisation matters and the best `C` will likely be small.

**Threshold selection.** On validation, find the threshold whose flag rate is closest to `config.economics.target_flag_rate` (default 0.05). Freeze it. Apply it unchanged to test.

**Then open the test set, once.** Report: AUROC, measured flag rate `f`, recall `R`, precision, base rate.

## 6. Metrics and uncertainty

**Bootstrap.** 1000 resamples of the test set with replacement, percentile CIs at 95%, for AUROC, `R`, `f`, and lift. With a ~600-example test set a point estimate is not defensible on its own, and a reviewer asking "how confident are you in 14×?" should get a real interval.

**Report separately, never blended:** precision and recall. No F1 in the codebase.

**Also report the base rate** (fraction of test responses that are incorrect) next to every headline number.

## 7. Economics

```
lift = R / f
```

where `R` is measured test recall and `f` is the **measured** test flag rate.

Three policies over `N` responses with base error rate `e`:

| Policy | Judge calls | Coverage | Errors caught | Relative cost |
|---|---|---|---|---|
| Judge everything | `N` | 100% | `e·N·a` | `1/f` |
| Random sample at rate `f` | `f·N` | `f` | `f·e·N·a` | `1×` |
| Probe-triggered | `f·N` | 100% | `R·e·N·a` | `1×` |

`a` is judge accuracy. It appears in every row and cancels out of the ratio, which is why **lift is independent of both the base error rate and the judge's own accuracy**. Implement `a` as a config parameter defaulting to `1.0`, and state the cancellation explicitly in `RESULTS.md` — it converts the softest-looking assumption in the analysis into a non-issue.

Emit `results/economics.json` with the worked table at `N = 1_000_000`, plus lift with its CI.

## 8. Latency

Measure, don't assert. The competition brief asks directly how the system avoids slowing the model down, and a measured ratio is a far stronger answer than a claim.

- **Probe cost:** time `scaler.transform` + `clf.decision_function` on a single already-computed activation vector. 1000 repetitions, report median and p95 in microseconds.
- **Generation cost:** median wall-clock time of the `generate()` calls recorded during extraction, per response.
- **Prefill cost:** median time of the pass-1 forward, per response — this is the honest denominator, since the activation comes from a forward pass the model was going to do anyway.
- **Report the ratio.** Probe cost as a fraction of generation cost. Expect four to six orders of magnitude.

State clearly that the probe adds no additional forward pass: the activation is a by-product of the prefill that generation already performs.

Write `results/latency.json`, including device name, `torch.__version__`, and quantisation setting, since the numbers are meaningless without them.

## 9. Secondary validation — abstention correlation

The source result reports that the probe direction also tracks the model's own uncertainty. Cheap to check, and it's independent evidence the probe reads something real rather than a dataset artifact.

Flag generations containing abstention patterns (`"i don't know"`, `"i'm not sure"`, `"i cannot"`, `"unable to"`, `"no information"` — case-insensitive, on the normalised string). Then report the mean probe score for abstaining vs non-abstaining responses, and the AUROC of the probe score for predicting abstention.

If the abstention rate is under 2% of test examples, say so and mark the comparison underpowered rather than reporting a noisy number.

## 10. Optional Stage 6 — GSM8K negative control

Run only if the main pipeline is complete and time remains.

The published result says probe generalisation **falters on mathematical reasoning**. Run the identical pipeline on `gsm8k` (`main` config), with the label rule swapped for final-numeric-answer matching, and expect a materially lower AUROC.

**Frame the outcome as a reproduction, not a failure.** A submission that says "it works on knowledge questions, here is the evidence, and here is the evidence it does not work on mathematical reasoning, exactly as the literature reports" is substantially more credible than one claiming universal performance. Put the result in `RESULTS.md` under a heading that says so.

## 11. Reproducibility

- Seed `random`, `numpy`, `torch`, and `torch.cuda` from `config.seed`.
- Greedy decoding only — no sampling anywhere in the labelling path.
- Every artifact embeds: resolved config, SHA-256 config hash, git commit, library versions, device name, UTC timestamp.
- `scripts/run_all.py --smoke` runs the whole pipeline at `n_examples=100` in under five minutes for CI and for verifying a change didn't break the chain.

## 12. Tests

| Test | Asserts |
|---|---|
| `test_normalization` | articles/punctuation/case handled; short-alias guard rejects `"US"` inside `"just us"`; whole-token match accepts it in `"the us"` |
| `test_split_integrity` | zero `question_id` overlap; zero normalised-question overlap; 60/20/20 within one example |
| `test_padding_side` | loader raises if padding side is not left |
| `test_left_padding_equivalence` | batched last-token activations match unbatched, tolerance `1e-2` |
| `test_economics` | `lift == R/f` exactly; random-sample policy yields lift `1.0`; lift invariant to base error rate and judge accuracy |
| `test_polarity` | positive class is "incorrect"; a probe trained on inverted labels produces `1 - auroc` |
| `test_no_test_leakage` | scaler and classifier are fit only on train indices |
| `test_determinism` | two runs at one seed produce identical probe coefficients |
| `test_smoke` | full pipeline at `n=100` completes and writes every expected artifact |

## 13. What `RESULTS.md` must contain

In this order:

1. Run metadata — model, quantisation, device, seed, config hash, git commit, timestamp.
2. Dataset — source, split sizes, dedup count, base error rate.
3. Layer sweep table — validation AUROC per layer, chosen layer, why.
4. Test results — AUROC with CI, measured `f`, `R`, precision, base rate.
5. Three-policy table at `N = 1,000,000`.
6. **Headline: lift with 95% CI**, plus the sentence that `R/f` is independent of base error rate and judge accuracy.
7. Latency table and the ratio.
8. Abstention correlation.
9. GSM8K negative control, if run.
10. Limitations — one model, one dataset, single seed unless swept, lenient-vs-strict labelling gap, and the fact that this measures probe quality, not end-to-end system performance.

Section 10 is not optional and not boilerplate. Write it honestly and specifically.
