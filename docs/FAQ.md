# FAQ

The questions a technical reviewer actually asks, each answered with the artifact that settles it. No numbers are quoted here — numbers live in generated files, and every answer points at the one that carries it.

For *how to present* the work, read [HANDOVER.md](HANDOVER.md) instead; this page is about the method.

---

## About the claim

**What is actually being claimed?**

That a model's internal state, read after it has finished reading the question but **before it generates a single token**, carries a readable signal about whether the answer it is about to produce will be wrong — strong enough that a logistic regression on it can select which responses are worth an expensive checker.

Not claimed: that it detects hallucinations, measures truthfulness, or knows what the model believes. It is a correlational classifier over activations. That language is used deliberately throughout the repo.

**Is the probe blocking or filtering responses?**

No, and nothing in this repo should ever be described that way. It is a **trigger**: its only output is a decision about whether to spend a judge call. That is why it is tuned for recall and why weak precision would be acceptable — a false positive wastes one judge call, a false negative lets a user act on a wrong answer.

**Why is the headline number a ratio and not an accuracy?**

Because the question is economic, not statistical: at a fixed budget of expensive checks, how many more errors do you catch than by sampling at random? That is `lift = R / f`. Both the assumed base error rate and the judge's own accuracy appear in every row of the policy table and cancel from the ratio — `src/economics.py::invariance_check` demonstrates it numerically and `results/RESULTS.md` states it.

**Then why isn't the lift larger?**

Because `lift = R/f = precision / base_rate`, and precision cannot exceed 1. So lift is capped at `1/base_rate` **for any probe whatsoever**. TriviaQA no-context is a deliberately hard benchmark where the model is wrong on a large fraction of questions, which puts a low ceiling on any triage method. `results/RESULTS.md` reports the measured lift, the ceiling, and the fraction of the ceiling achieved, side by side.

The transferable quantity is therefore the **AUROC** — a ranking quality, independent of base rate. The lift is what that ranking is worth on a workload where the model fails this often. `DECISIONS.md` 015 records the reasoning.

**The README shows much larger lift at lower error rates. Is that a result?**

No, and it is labelled as such wherever it appears. A ROC curve is base-rate independent, so the measured curve can be re-read at rarer error rates — but that assumes the probe ranks equally well on that workload, which is exactly the cross-domain generalisation this repo has **not** tested. It is an illustration of headroom, never a result.

---

## About the method

**Why TriviaQA with no context?**

Automatic correctness labels, short answers, and — critically — the no-context variant forces the model to answer from parametric knowledge. Pass the supporting document and the task becomes reading comprehension, at which point "does the model know this?" is no longer the question being asked. Alternatives considered and rejected: `DECISIONS.md` 001.

**Why read activations at the question, not during or after generation?**

More information is available later, but by then you have already paid for generation, which forecloses the routing use case. Question-time probing is strictly less informative and deliberately so: the same signal can route a hard question to a stronger model *before* any tokens are generated. Expect lower AUROC than a post-hoc output-reading detector. `DECISIONS.md` 002.

**How do I know the test set wasn't used to pick the layer?**

Three things, in increasing order of how much they'd survive an adversarial read:

1. The layer, the regularisation strength and the threshold are all selected in `src/probe.py`, which is only ever handed the train and validation masks.
2. `scripts/02_train_probe.py` opens the test set at exactly one place, marked in a comment box.
3. Every scoring of the test set is appended to `results/test_scoring_log.json`, which is committed. The count is disclosed in `results/RESULTS.md`, and the one re-scoring was pre-registered in `DECISIONS.md` 016 *before* it ran, with 017 recording its outcome.

The invariant once read "the test set is touched exactly once". Nobody could check that, so it was replaced with a rule that can be checked: no selection may consult test, and every scoring is logged. That change is itself recorded rather than quietly made.

**Why deduplicate questions before splitting?**

TriviaQA's validation split contains many near-duplicate question strings. An example-level split puts the same question in train and test and inflates AUROC by an unknown amount. Splits are by `question_id` after normalising and deduplicating question strings, and `src/data.py::assert_split_integrity` asserts pairwise disjointness on both keys. The dedup count is published in `results/RESULTS.md` §2. `DECISIONS.md` 003, 011.

**Isn't lenient alias matching generous?**

Yes, and that is why strict exact match is computed alongside and both are published. The model generates sentences; gold answers are short spans, so strict exact match would label nearly everything incorrect and produce a degenerate label distribution. Aliases under 3 characters require a whole-token match rather than substring containment — without that guard, a gold alias like `US` matches inside thousands of unrelated generations. Where the two rules disagree by more than about ten points, `RESULTS.md` says so explicitly. `DECISIONS.md` 007, 010.

**Why is there no F1 anywhere?**

By rule. F1 blends two failure modes whose costs here differ by orders of magnitude, and it hides which one you have. Precision and recall are always reported separately, in code and in every document. `DECISIONS.md` 005.

**Why is precision high and recall low?**

That is what a threshold set to a small flag rate does: only the most confident flags fire. The threshold is chosen on validation to hit the configured budget, and the *measured* test flag rate — not the target — is what every downstream calculation uses. Change `economics.target_flag_rate` and re-run from stage 02 to see the whole trade-off curve; `results/roc_curve.png` shows it directly.

**How do you know the activations aren't garbage?**

This is the failure the repository is most defended against. With right padding, position −1 of a batch is a pad token, every activation is meaningless, nothing raises an error, and the AUROC lands near chance — reading as "the idea doesn't work" rather than "the code is broken".

So every run measures one batch twice: once left-padded, once **deliberately right-padded**, and aborts unless the deliberately broken one is *rejected*. Both rows are published in `results/RESULTS.md` §1. The criteria are relative L2 and cosine rather than absolute tolerances, because in bfloat16 batched and unbatched matmuls legitimately disagree in the last few bits. `DECISIONS.md` 014.

**Is the reported cost of the probe honest?**

`results/latency.json` carries two figures: the full scikit-learn call and the raw dot product. The **slower** one is quoted everywhere, deliberately. Generation and prefill times come from the same calls that produced the labels, not from a fresh favourable benchmark. And the probe adds **no additional forward pass** — the activation is a by-product of the prefill the model already performs.

**Is it reproducible?**

`random`, `numpy`, `torch` and `torch.cuda` are all seeded from `config.seed`; decoding is greedy. Two runs at one seed produce identical probe coefficients, asserted by `tests/test_determinism.py`. Every artifact embeds the resolved config, its SHA-256 hash, the git commit, library versions, device and a UTC timestamp — and records `dirty: true` if the working tree was not clean when it ran.

---

## About the scope

**Does this work on a model I access through an API?**

No — and this is the sharpest question the work faces, so answer it first rather than being caught by it. The probe tier requires access to the weights. The *cascade* still applies where you don't own them: the cheap tier degrades to output-layer signals, and the expensive tier is unchanged. But Tier 2 as measured here needs an open-weights model you host. Do not claim universality.

**Does it work on maths, code, multi-turn, other models?**

Untested here. The published result this reproduces reports that probe generalisation falters on mathematical reasoning; the GSM8K negative control is optional Stage 6 and has **not been implemented** — the `negative_control` block in `config.yaml` reserves the settings, but no code reads them yet. Until it is run, that boundary is cited rather than measured, and it is listed as a limitation in every generated document. `DECISIONS.md` 001, 008.

**Where is the gateway, the policy engine, the dashboard?**

Out of scope by design (`CLAUDE.md`, "Out of scope"). This repository measures whether the Tier 2 signal is good enough to be worth building the rest around. Building the serving layer first would have produced a demo with no evidence behind it.

**What would make this result stronger?**

In rough order of value per hour: the API-access design note (what the cascade becomes without weight access); a second dataset or model to test transfer; the GSM8K control run deliberately as a reproduction of a documented limitation; a seed sweep, so intervals reflect split and fitting variation rather than test-set sampling alone.

**A number in the README looks wrong. Can I fix it?**

Not by editing it. Every number in `README.md`, `results/RESULTS.md` and `docs/HANDOVER.md` is rendered from an artifact in `results/` — if a number is wrong, the pipeline is wrong. Fix the pipeline and regenerate with `scripts/05_report.py`. Prose can be edited by hand, in `README_TEMPLATE.md`; numbers cannot.

---

**See also:** [ARCHITECTURE.md](ARCHITECTURE.md) · [SETUP.md](SETUP.md) · [GLOSSARY.md](GLOSSARY.md) · [../DECISIONS.md](../DECISIONS.md), which is the long-form version of most answers on this page.
