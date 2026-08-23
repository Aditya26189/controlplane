# Documentation

Everything written about this repository, and the order to read it in.

## By what you need

**"I have ten minutes and I may have to present this."**
→ [HANDOVER.md](HANDOVER.md). What was built, what the headline number means, what bounds it, and which framings to avoid. Generated from `results/`, so it is never out of date with the numbers.

**"I am reviewing this and I want to find the flaw."**
→ [FAQ.md](FAQ.md) for the short answers, [../DECISIONS.md](../DECISIONS.md) for the long ones, [../results/RESULTS.md](../results/RESULTS.md) for every table and interval, and [../results/test_scoring_log.json](../results/test_scoring_log.json) for the audit trail on the test set.

**"I am going to change some code."**
→ [ARCHITECTURE.md](ARCHITECTURE.md) first, then [../SPEC.md](../SPEC.md) and [../CLAUDE.md](../CLAUDE.md) for the invariants, then [../CONTRIBUTING.md](../CONTRIBUTING.md) for the git and documentation rules — which are binding, not advisory.

**"I just want to run it."**
→ [SETUP.md](SETUP.md). Laptop, Kaggle T4, or an offline cluster, plus what each failure mode means.

**"What does this word mean here?"**
→ [GLOSSARY.md](GLOSSARY.md). Lift, flag rate, polarity, positive control, selection discipline — defined once, precisely, with the wrong readings called out.

## Every document

| Document | Tier | Written by |
|---|---|---|
| [HANDOVER.md](HANDOVER.md) | generated | `scripts/06_handover.py` |
| [ARCHITECTURE.md](ARCHITECTURE.md) | hand-written | contributors |
| [SETUP.md](SETUP.md) | hand-written | contributors |
| [FAQ.md](FAQ.md) | hand-written | contributors |
| [GLOSSARY.md](GLOSSARY.md) | hand-written | contributors |
| [../README.md](../README.md) | generated | `scripts/05_report.py` from `README_TEMPLATE.md` |
| [../results/RESULTS.md](../results/RESULTS.md) | generated | `scripts/05_report.py` |
| [../CLAUDE.md](../CLAUDE.md) | contract | — |
| [../SPEC.md](../SPEC.md) | contract | — |
| [../TASKS.md](../TASKS.md) | contract | — |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | contract | — |
| [../DECISIONS.md](../DECISIONS.md) | rationale, append-only | — |

## The three rules that govern all of it

1. **Generated documents are never hand-edited for numbers.** `README.md`, `results/RESULTS.md` and `HANDOVER.md` are rendered from artifacts in `results/`. If a number is wrong, the pipeline is wrong — fix it and regenerate. README prose is edited in `README_TEMPLATE.md`.
2. **Contract documents change in the same commit as the code that made the change necessary.** Never a trailing "update docs" commit. If code contradicts `SPEC.md`, stop: either the code is wrong, or the spec needs updating deliberately, first, with the reasoning logged.
3. **`DECISIONS.md` is append-only.** A reversed decision is superseded by a new entry that links back, never edited away. It is the direct answer to "why did you do it that way?", which is most of what a technical reviewer asks.

The hand-written pages on this index deliberately carry **no measured numbers**. They point at the generated documents instead, so they cannot drift out of date with a re-run.
