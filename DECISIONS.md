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
