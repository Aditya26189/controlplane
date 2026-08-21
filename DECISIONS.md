# DECISIONS.md

Append-only log of methodological choices. Newest at the bottom. **Never edit or delete an entry** — a reversed decision gets a new entry that supersedes the old one by number.

This is the answer sheet for "why did you do it that way?" Every entry should be readable aloud to a technical reviewer.

**Format:**

```
## NNN — <decision, in one line>
**Date:** YYYY-MM-DD · **Status:** accepted | superseded by NNN
**Context:** what forced a choice
**Decision:** what we chose
**Alternatives:** what we rejected, and why
**Consequences:** what this costs us, and what a reviewer could fairly object to
```

Log a decision when a reviewer could reasonably challenge it. Not for variable names or file layout.

---

## 001 — TriviaQA, no-context, as the only primary dataset
**Date:** 2026-08-20 · **Status:** accepted

**Context:** The probe needs a dataset with automatic correctness labels and enough wrong answers to learn from.

**Decision:** `mandarjoshi/trivia_qa`, config `rc.nocontext`, validation split. No second primary dataset.

**Alternatives:**
- *GSM8K / MATH* — rejected as primary. The source paper (arXiv 2509.10625) reports that probe generalisation falters on mathematical reasoning. Using them would produce a weak result attributable to a documented property of the method, not to our implementation. Retained instead as an optional negative control (see 008).
- *Natural Questions* — viable, but longer free-form answers make automatic exact-match labelling less reliable.
- *With supporting context* — rejected. Passing the document turns the task into reading comprehension, and "does the model know this?" stops being the question being asked.

**Consequences:** Results are specific to short-form factual recall. Cross-domain generalisation is untested and must be stated as a limitation. A reviewer can fairly say the result is narrow — the honest answer is that it is, and that breadth is Round 2 work.

---

## 002 — Probe question-time activations, not mid-generation
**Date:** 2026-08-20 · **Status:** accepted

**Context:** The activation could be read at several points: after the prompt, during generation, or after the answer completes.

**Decision:** Final prompt token, before any generated token exists.

**Alternatives:**
- *Mid-generation* — more information available, but you have already paid for generation, which forecloses the routing use case.
- *Post-generation* — equivalent to an output-reading checker, which is what we are trying to avoid.

**Consequences:** Strictly less information than post-hoc methods, so we should expect lower AUROC than an output-reading detector — that trade is deliberate. In exchange the same signal supports pre-generation routing, and the trigger fires before any generation cost is incurred.

---

## 003 — Split by question_id, after deduplicating question strings
**Date:** 2026-08-20 · **Status:** accepted

**Context:** TriviaQA ships multiple answer aliases per question and some near-duplicate question phrasings. A naive random split at example level leaks near-identical items across train and test.

**Decision:** Normalise question strings, deduplicate keeping the first occurrence, then split by `question_id` at 60/20/20. Assert pairwise disjointness on both `question_id` and normalised question string.

**Alternatives:** *Random example-level split* — one line simpler, and inflates AUROC by an unknown amount. Not defensible under questioning.

**Consequences:** Slightly smaller effective dataset. Dedup count must be logged; if it is large, the raw dataset was noisier than assumed and that is worth knowing.

---

## 004 — Positive class is "incorrect"
**Date:** 2026-08-20 · **Status:** accepted

**Context:** Polarity has to be fixed once and held everywhere.

**Decision:** `y = 1` means the generated answer was wrong. The probe predicts wrongness; recall is the fraction of wrong answers flagged.

**Alternatives:** *Positive = correct* — equally valid, but it makes "recall" mean the opposite of what the economics needs, and every downstream formula would need mental inversion.

**Consequences:** Inverting this silently produces `1 - AUROC`, which reads as a strong negative result and misdirects debugging. Asserted in tests and stated in docstrings.

---

## 005 — Tune for recall; accept low precision
**Date:** 2026-08-20 · **Status:** accepted

**Context:** The threshold trades recall against precision. Which failure do we prefer?

**Decision:** Maximise recall at a fixed flag-rate budget. Precision is reported but not optimised.

**Alternatives:** *Balanced F1* — rejected outright. F1 blends the two failure modes into one number that hides which one you have, and the two modes here differ in cost by orders of magnitude.

**Consequences:** Most flagged responses will be fine, and that is by design: a false positive costs one wasted judge call, a false negative costs a user acting on a wrong answer. The framing depends entirely on the probe never having blocking authority — if that ever changes, this decision must be revisited.

---

## 006 — Test set opened exactly once
**Date:** 2026-08-20 · **Status:** accepted

**Context:** Layer, regularisation strength, and threshold all need choosing.

**Decision:** All three chosen on validation. Test scored once, at the end, for the reported numbers only.

**Alternatives:** *Select layer on test* — one fewer split to manage, and it inflates the headline number. It is also the first thing a technical reviewer checks.

**Consequences:** Roughly 600 test examples, so intervals are wide — addressed with a 1000-sample bootstrap rather than by borrowing from validation. If the reported AUROC is weak, it is weak; we do not go back.

---

## 007 — Lenient alias matching, with strict exact match reported alongside
**Date:** 2026-08-20 · **Status:** accepted

**Context:** The model generates sentences; gold answers are short spans. Strict exact match would label nearly everything incorrect.

**Decision:** A normalised gold alias appearing in the normalised prediction counts as correct, with aliases under 3 characters requiring a whole-token match. Strict exact match computed and reported as a secondary column.

**Alternatives:**
- *Strict EM only* — degenerate label distribution.
- *LLM-as-judge for labelling* — more accurate, but it introduces a second model's errors into the ground truth and costs compute we do not have.

**Consequences:** Lenient matching over-counts correctness where a gold alias appears incidentally. Reporting both base rates lets a reviewer see the size of that effect. If they diverge by more than ~10 points, `RESULTS.md` must say so explicitly.

---

## 008 — GSM8K retained as an optional negative control
**Date:** 2026-08-20 · **Status:** accepted

**Context:** The method is documented to fail on mathematical reasoning (see 001).

**Decision:** Run it deliberately, if time allows, and report the weak result as a reproduction of a published limitation.

**Alternatives:** *Omit it* — simpler, and forfeits the credibility of demonstrating we validated against the literature rather than only citing it.

**Consequences:** Costs roughly an hour. Requires framing discipline in `RESULTS.md` — presented as a reproduction, never as a shortcoming to be explained away.

---

## 009 — Lift reported as R/f, with base error rate and judge accuracy shown to cancel
**Date:** 2026-08-20 · **Status:** accepted

**Context:** The headline number needs to survive the question "but you assumed a 3% error rate."

**Decision:** Report `lift = R/f`. Implement judge accuracy `a` and base error rate `e` explicitly in the economics module, and demonstrate in both tests and `RESULTS.md` that neither affects the ratio.

**Alternatives:** *Report absolute errors caught* — more concrete for a business audience, and it depends on an assumed base rate, which is the softest input in the analysis.

**Consequences:** Turns the most attackable assumption into a non-issue. Absolute counts remain in the three-policy table as illustration, clearly marked as depending on the assumed rate.

---

<!-- New entries below this line. Do not edit anything above it. -->

## 010 — Short-alias guard is whole-token matching; SPEC.md §12's example corrected
**Date:** 2026-08-21 · **Status:** accepted

**Context:** `SPEC.md` §2 specifies that a normalised alias shorter than 3 characters must match as a whole token (`a in pred.split()`) rather than as a substring. `SPEC.md` §12 then gave `"US"` inside `"just us"` as the case the guard should *reject*. The two contradict each other: `"just us"` splits to `["just", "us"]`, so the whole-token rule accepts it, and the reference implementation in §2 returns `True`.

**Decision:** §2 is authoritative — the guard is whole-token matching. §12's example was wrong and has been corrected to `"US"` inside `"Augustus"`, which is a genuine substring match that the guard does reject. The rule itself is unchanged.

**Alternatives:**
- *Make short aliases require full-string equality* — would have made §12's example pass, but it silently discards correct answers: a model that replies `"the US"` to a country question is right, and this rule would label it wrong. It also contradicts §2's stated semantics, which the spec instructs implementers to preserve exactly.
- *Leave the spec inconsistent and skip the test* — the guard is on CLAUDE.md's pitfall list precisely because its failure is silent, so it needs a test that can actually fail.

**Consequences:** `"just us"` is labelled correct for gold answer `"US"`. That is the intended behaviour of token matching and it is a rare surface form; the alternative loses many more true positives than it gains. A reviewer can check the boundary directly in `tests/test_normalization.py`, which now covers both the substring rejection and the token acceptance.

---

## 011 — Deduplication removes 44% of TriviaQA rc.nocontext validation, as designed
**Date:** 2026-08-21 · **Status:** accepted · **Refines:** 003

**Context:** Running the 003 dedup rule on the real split dropped 7,983 of 17,944 rows — 44.5%. A drop that large needs explaining before anyone reads a number produced downstream of it.

**Decision:** Keep the rule. The drop is a property of the dataset, not a bug. Measured: `rc.nocontext` validation has 17,944 rows over 9,960 unique `question_id`s — mean 1.80 rows per id, max 2. The `rc` configs carry one row per (question, evidence document) pair, and the `nocontext` variant strips the documents while keeping the rows, so ~80% of questions appear twice with identical text and identical id.

**Alternatives:** *Keep the duplicates and split at example level* — would have put the same question, byte-for-byte, in both train and test for roughly 80% of questions. This is the exact leak CLAUDE.md invariant 3 exists to prevent, and it would have inflated test AUROC by an amount we could not bound.

**Consequences:** The effective pool is 9,961 unique questions, which still comfortably exceeds the 3,000 sampled. The number is reported in `results/RESULTS.md` rather than buried, because "you discarded 44% of your data" is a fair question with a good answer.

---

## 012 — Threshold by rank, and a fixed tie-break in layer selection
**Date:** 2026-08-21 · **Status:** accepted · **Refines:** 006

**Context:** Two selection steps admit several defensible rules, and both change which numbers get reported.

**Decision:**
1. *Threshold.* On validation, take the k-th largest probe score for `k = round(target_flag_rate · n_val)`. This hits the target rate on validation exactly rather than approximately.
2. *Tie-break.* When two (layer, C) pairs tie on validation AUROC, prefer the shallower layer, then the smaller C.

**Alternatives:**
- *Threshold by score quantile* — equivalent in the limit, but with ties in the score distribution the realised rate can drift from the target, and the drift is invisible.
- *Tie-break by dict or iteration order* — whatever `max()` happens to return. Two runs of the same sweep could then disagree about the winner, which would break the reproducibility claim for no benefit.

**Consequences:** The realised flag rate on **test** still differs from the target — that is expected, and CLAUDE.md invariant 6 requires every downstream calculation to use the measured test rate. The tie-break favours the cheaper and more strongly regularised option, which is the conservative direction; it is recorded because a reviewer comparing two runs should know the rule rather than infer it.

---

## 013 — Bootstrap recall and flag rate jointly, not independently
**Date:** 2026-08-21 · **Status:** accepted · **Refines:** 009

**Context:** The headline is `lift = R/f`, and both inputs come from the same test set. The confidence interval has to reflect that.

**Decision:** Each of the 1000 bootstrap resamples draws test rows with replacement once, then recomputes AUROC, `f`, `R`, precision and `lift` on that single resample. The interval on lift is the percentile interval of the resampled lifts.

**Alternatives:** *Bootstrap `R` and `f` separately and combine their intervals* — simpler, and wrong. `R` and `f` are positively correlated through the same threshold on the same rows; treating them as independent misstates the interval on their ratio, and the ratio is the number being defended.

**Consequences:** Resamples where a metric is undefined (no positives, nothing flagged, a single label class) are dropped for that metric and the surviving count is recorded beside the interval, rather than being coerced to zero — which would drag the interval toward a value the run never produced. At small `f` and small test `n`, a visible fraction of resamples flag nothing, so that count is worth reading before quoting the interval.

---

## 014 — The left-padding check uses scale-invariant criteria, plus a positive control
**Date:** 2026-08-21 · **Status:** accepted · **Supersedes the tolerance in** `SPEC.md` §4

**Context:** The first full run on a Kaggle T4 aborted at the equivalence check. Qwen2.5-7B, NF4, bfloat16, `padding_side` confirmed `left` by the Stage 2 gate. Per-layer absolute deviations were `{8: 0.125, 11: 0.25, 14: 0.375, 17: 0.375, 20: 0.75, 23: 0.625, 26: 3.0}` against SPEC's absolute tolerance of `1e-2`.

Every one of those values is an exact multiple of 0.125, and bfloat16 has an 8-bit mantissa: at residual-stream magnitudes of 16–256 its ULP is 0.125–1.0. The deviations are 1–6 ULP, and they grow with depth exactly as residual-stream magnitude grows. Batched and unbatched forwards use different GEMM shapes, so cuBLAS picks different tilings and accumulates in a different order. This is arithmetic noise, not a padding fault.

**Decision:** Judge the check on two **scale-invariant** quantities instead of an absolute deviation: per-row relative L2 error (limit `0.10`) and per-row cosine similarity (limit `0.999`), both in `config.yaml` under `equivalence_check`. Additionally, **every run repeats the comparison with the tokenizer deliberately right-padded and requires that it fail.** If the control passes, the run stops.

**Alternatives:**
- *Raise the absolute tolerance until the run passes* — rejected outright. It is unfalsifiable: no absolute number distinguishes rounding from a wrong read position, because the activations' magnitude varies by an order of magnitude across depth and by dtype. It is also precisely the move `KICKOFF.md` warns against, and a reviewer would be right to discount every number downstream of it.
- *Run the check in float32* — would make the comparison exact, but it measures a forward pass the experiment does not perform. The check must exercise the arithmetic the run actually uses.
- *Cosine alone* — nearly sufficient (right padding drops cosine to 0.036 on the fixture model) but blind to a uniform rescaling. Relative L2 catches that, so both are required.

**Consequences:** The limits look permissive next to `1e-2`, and on their own they would be. The positive control is what makes them defensible: measured separation on the fp32 fixture is relative L2 `1.8e-07` (left) against `1.36` (right), a factor of 7.7 million, and cosine `1.000000000` against `0.036`. Both limits sit far from either regime. Every `extract_meta.json` now records both sides, so a reviewer can check the margin rather than trust the threshold. If a future model or dtype narrows that margin, the control fails and the run stops rather than proceeding on a check that no longer discriminates.
