<div align="center">

# 📚 Documentation

**Every document in this repository, and which question it answers.**

[![Orientation](https://img.shields.io/badge/Orientation-11_pages-6366f1?style=flat-square)](#-orientation)
[![Contracts](https://img.shields.io/badge/Contracts-5_binding-d1242f?style=flat-square)](#-contracts)
[![Evidence](https://img.shields.io/badge/Evidence-append_only-22c55e?style=flat-square)](#-evidence-and-rationale)
[![No stale numbers](https://img.shields.io/badge/Orientation_pages-no_measured_numbers-8b949e?style=flat-square)](#the-rules-these-documents-follow)

</div>

> [!TIP]
> **In a hurry?** Pick the row below that matches why you are here. Every path is
> ordered — the documents build on each other in the sequence given.

---

## 🧭 Start with why you are here

| You are… | Read, in this order |
|:---|:---|
| ⏱️ **Reviewing this in twenty minutes** | [../README.md](../README.md) claim table → [DIAGRAMS.md](DIAGRAMS.md) → [LIMITATIONS.md](LIMITATIONS.md) → [FAQ.md](FAQ.md) |
| 🌱 **New to the codebase** | [ONBOARDING.md](ONBOARDING.md) → [DIAGRAMS.md](DIAGRAMS.md) → [CODE_TOUR.md](CODE_TOUR.md) → [ARCHITECTURE.md](ARCHITECTURE.md) |
| ▶️ **Trying to run it** | [SETUP.md](SETUP.md) → [RUNBOOK.md](RUNBOOK.md) → [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 🔍 **Checking whether a number is real** | [../README.md](../README.md) claim table → [ARTIFACTS.md](ARTIFACTS.md) → [METHODS.md](METHODS.md) → `make verify` |
| 🕵️ **Looking for the flaw** | [LIMITATIONS.md](LIMITATIONS.md) → [../DECISIONS.md](../DECISIONS.md) → [CASES.md](CASES.md) → [TESTING.md](TESTING.md) |
| 🎤 **Presenting it** | [DIAGRAMS.md](DIAGRAMS.md) → [JOURNEY.md](JOURNEY.md) → [DEMO.md](DEMO.md) → [DEMO_SCRIPT.md](DEMO_SCRIPT.md) |
| 💬 **Confused by a word** | [GLOSSARY.md](GLOSSARY.md) |

---

## 🧩 Orientation

> Hand-written. **No measured numbers** — they point at the claim table and
> `results/` instead, so a re-run cannot leave them stale.

| Document | Answers |
|:---|:---|
| 🗺️ [DIAGRAMS.md](DIAGRAMS.md) | What does the system look like? Eleven diagrams: the three objects, the pipeline, the warrant lifecycle, the controls, policy, composition, drift, verification, the Round 1 → Round 2 journey, and the repository end to end |
| 🌱 [ONBOARDING.md](ONBOARDING.md) | I have one hour. What do I read, run, and look at, in what order? |
| 🧭 [CODE_TOUR.md](CODE_TOUR.md) | What is in each package under `controlplane/`, and which file do I open for X? |
| ⚙️ [SETUP.md](SETUP.md) | How do I get this running — laptop, GPU notebook, offline machine? |
| 📓 [RUNBOOK.md](RUNBOOK.md) | What does each script do, what does it read, what does it write, and when would I run it? |
| 🚑 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | It crashed, or it refused. What does this mean and what do I do? |
| 📦 [ARTIFACTS.md](ARTIFACTS.md) | What is every file in `results/` and `evalsets/`, and what is inside it? |
| 🧪 [TESTING.md](TESTING.md) | What does the test suite defend, and which suite covers what? |
| 📖 [GLOSSARY.md](GLOSSARY.md) | Warrant, envelope, operating point, estimand, null band, yield vs rate — defined once, precisely |
| ❓ [FAQ.md](FAQ.md) | The questions a technical reviewer actually asks, each answered with the artifact that settles it |
| 🧾 [JOURNEY.md](JOURNEY.md) | What did this project actually do, phase by phase, and what changed our mind? |

## 📐 Contracts

> The design. **Binding on contributors**, and read before writing code.

| Document | Answers |
|:---|:---|
| 📜 [SPEC.md](SPEC.md) | The technical specification: data model, validation harness, matrix, drift, policy, gate |
| 🧱 [../CLAUDE.md](../CLAUDE.md) | The invariants, the silent failures, the coding standards, what is out of scope |
| 🗂️ [TASKS.md](TASKS.md) | The phased build order and the gate at the end of each phase |
| 🔀 [CONTRIBUTING.md](CONTRIBUTING.md) | Git workflow, commit rules, what is committed, the documentation audit |
| 🎬 [KICKOFF.md](KICKOFF.md) | The original brief |

## 🔬 Evidence and rationale

> Where the numbers and the reasoning actually live.

| Document | Answers |
|:---|:---|
| 🧠 [../DECISIONS.md](../DECISIONS.md) | Append-only entries. *"Why did you do it that way?"* — including the reversals |
| 📊 [METHODS.md](METHODS.md) | Estimators, bootstraps, null bands, and where each came from |
| ⚠️ [LIMITATIONS.md](LIMITATIONS.md) | Scope, declared gaps, open items. **Read before quoting anything** |
| ✅ [CASES.md](CASES.md) | Every case, the test covering it, the artifact demonstrating it |
| 🚚 [PATHS.md](PATHS.md) | What moved on 2026-08-29 and where it went |
| 📈 [../results/RESULTS.md](../results/RESULTS.md) | The rendered results, with fixture numbers refused rather than printed |

## 🎥 Presentation

| Document | Answers |
|:---|:---|
| 🖥️ [DEMO.md](DEMO.md) | The two-pane demo this build exists to produce |
| 🗣️ [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | The beat-by-beat script and what is on screen |
| 💼 [PROPOSAL.md](PROPOSAL.md) | The business proposal |
| 🔖 [EXTERNAL_FIGURES.md](EXTERNAL_FIGURES.md) | The register that gates every figure about the world, and its provenance tiers |

---

## The rules these documents follow

> [!IMPORTANT]
> These five are why the documentation can be trusted at all. They are enforced
> by tests, not by discipline.

**1 · Every number is computed by code.**
If it cannot be traced to an artifact in `results/`, it does not go in a
document. The README's claim table names the artifact **and the field** for each
of its numbers, and `make verify` resolves all of them. A number edited by hand
fails the build.

**2 · The orientation pages carry no measured numbers at all.**
They point at the generated documents instead, so a re-run cannot leave them
stale. Structure, procedure and vocabulary change with the code, not with the
run.

**3 · `DECISIONS.md` is append-only.**
A reversed decision is superseded by a new entry that links back — never edited
away. That is why entries 108/109 and 116 read as corrections rather than as
history that never happened.

**4 · Prose documentation is written at the end, not during the build.**
`CLAUDE.md` sets this. `DECISIONS.md` is the sole exception, because it is the
one document that cannot be reconstructed later.

**5 · A figure about the world needs a register entry.**
Court awards, fines, standard clause numbers — see
[EXTERNAL_FIGURES.md](EXTERNAL_FIGURES.md). A test enforces it against the
proposal.

<div align="center">

---

**[⬆ Back to the project README](../README.md)**

</div>
