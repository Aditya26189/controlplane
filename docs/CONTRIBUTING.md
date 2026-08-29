# CONTRIBUTING.md

Git and documentation workflow. Binding on agents and humans. Read alongside `CLAUDE.md`.

---

## Why this matters here

1. **This repo is a judged public artifact.** Round 2 is submitted as a public GitHub repository. A history of one commit called `final` reads as a student project; a legible history of atomic commits reads as engineers who have shipped. The history is part of the deliverable.
2. **Phases are expensive and interdependent.** When something downstream is wrong, you return to a tagged known-good state rather than debugging forward.
3. **Numbers here are claims.** This project's entire thesis is that unbacked claims are the problem. A repo whose numbers can't be traced to the run that produced them would be self-refuting.

Not doing: release branches, semver, CI pipelines, coverage gates, changelogs. Wrong tools for this timeline.

---

## Branching

`main` is always green — tests pass, pipeline runs.

```bash
git checkout -b phase/6-on-traffic-warrant
# ... atomic commits ...
git checkout main
git merge --no-ff phase/6-on-traffic-warrant -m "Phase 6: on-traffic warrant"
git tag -a phase-6 -m "Gate passed: <one-line summary>"
git branch -d phase/6-on-traffic-warrant
```

`--no-ff` preserves the phase boundary. Tags are the rollback points. **Never merge a branch whose gate has not passed.**

---

## Commits

Conventional Commits plus one addition.

**Types:** `feat` · `fix` · `docs` · `test` · `refactor` · `chore` · `exp`

`exp` marks a run that produces or changes artifacts in `results/`. It exists so a reviewer can filter to "when did the numbers change, and why."

**Rules**

- Imperative subject, under 72 chars, no trailing period.
- **One logical change per commit.** If the subject needs "and", it's two commits.
- Commit at every working checkpoint. Four to ten per phase.
- Never commit code failing its own tests.
- **Any commit that moves a measured number states before and after in the body.**
- **Any commit touching a statistical claim states the quantity estimated and the propagation used.** This is the specific discipline that catches sizing errors.

**Examples**

```
fix(sampling): size recall interval through dR/dq, not prevalence

Sizing used n = 1.96^2 q(1-q)/m^2 with m in recall units. That formula
sizes a PREVALENCE estimate. Recall = TP/(TP + q*N_u), and at the declared
workload dR/dq = -4.65, so the recall margin must be divided by that before
sizing. Sample size scales as 1/m^2, so the old figure understated labels
by ~22x.

Quantity: recall. Propagation: m_q = m_R / |dR/dq|.
Before: 456 labels claimed for +/-2pp recall.
After:  5,292 labels for +/-2pp; 847 for +/-5pp.

Refs: SPEC.md §6.2, DECISIONS.md 014
```

```
exp(matrix): populate warrant matrix across four envelopes

T1 mean-pool REFUSED on triviaqa-longctx-600 (recall 0.02, below min 0.10).
T1 max-rolling-means VALID, AUROC 0.79 [0.74, 0.83], wider than on
triviaqa-600 as expected.

config_hash: 8b21f04c
```

---

## What is and isn't committed

**Committed — the evidence:**
- all source, tests, config, policies, documentation
- `evalsets/` manifests and hashes (and the sets themselves if under size)
- `results/*.json`, `results/RESULTS.md`, `results/*.png`
- the demo notebook **with outputs intact** — it renders on GitHub for judges who will never run it. Do not add `nbstripout`.

**Not committed — regenerable and large:**
- `results/activations.npz`, `results/*.parquet`, model weights, caches, venvs

**If a `git add` would stage a file over 10 MB, stop and ask.** Do not extend `.gitignore` unilaterally.

**Never committed:** review labels containing real personal data. The label store is separate, gitignored, and referenced by hash only. Synthetic evaluation PII is fine and must be clearly marked synthetic in the manifest.

---

## The provenance ordering problem

Every artifact embeds the git commit it was generated from. Committing the artifact creates a *new* commit, so the hash inside a file can never be the hash of the commit containing it.

> **`git_commit` in an artifact is `HEAD` at the moment the script ran** — the commit of the *code* that produced it.

Sequence: commit code → run script → commit artifacts as a separate `exp:` commit. **Never run a script against a dirty tree** — the recorded hash would be a lie. `provenance()` runs `git status --porcelain` and records `dirty: true` when the tree isn't clean.

If an artifact does get generated from a dirty tree, regenerate it on a clean
one and commit that as its own `exp:` commit stating that only the provenance
block moved — verified field by field, not asserted. That happened on
2026-08-29 to the score sets and `results/feasibility.json`, and it is what
the dirty flag is for: it was visible, so it got fixed.

---

## Do not use `sed -i` on tracked files

The markdown in this repository is stored with **CRLF** line endings. `sed -i`
under Git Bash rewrites the whole file with LF, so a two-line edit lands as a
several-hundred-line diff and the actual change becomes unreviewable — in
`README.md` and `docs/LIMITATIONS.md`, which are the two files a reader is
most likely to inspect. It happened twice in one session; `e440d55` carries
the result, and `git show -w e440d55` is what recovers the real change there.

Edit through Python (`pathlib.Path.write_text`, preserving the existing
endings) or an editor that respects them. No `.gitattributes` is added,
deliberately: `text=auto` would renormalise every tracked file on the next
checkout, which trades one large diff for a repository-wide one days before
submission.

---

## Documentation

### Documentation is deferred to a closing phase

**Do not write or update narrative documents during the build.** See
`CLAUDE.md`, "Documentation — written at the end, not during the build".

This reverses the earlier rule that a doc change shipped in the same commit as
the code that caused it. That rule produced good documents at a large cost in
build time.

**`DECISIONS.md` is the exception and stays live**, because it is the one
document that cannot be written later — it records why at the moment a
measurement forced a choice. Keep entries short: the decision, the numbers, the
alternative rejected, the objection a reviewer would raise. Derivation where
statistical. Nothing else.

**Commit messages carry the rest.** A commit that makes a methodological choice
states the choice and the numbers behind it, in the body, while the reason is
still known. A one-line commit message is a loss of information rather than a
stylistic lapse.

Where shipped behaviour diverges from `SPEC.md`, say so in the commit message
and leave the spec alone. Reconciling the two is the documentation phase's job,
and the Phase 12 audit still checks that every section matches shipped
behaviour.

### Tier 3 — Generated: `README.md`, `results/RESULTS.md`

Derived from `results/`. **Never hand-edit a number.** If a number is wrong, the pipeline is wrong. Prose editable; numbers not.

### Docstrings

Google style, saying *why*. Where code enforces an invariant from `CLAUDE.md`, name it. Where a function computes a statistical quantity, state the quantity and the propagation:

```python
def size_for_recall_margin(m_R: float, workload: Workload) -> SampleSize:
    """Sample size for stratum B to bound recall to +/- m_R at 95%.

    Sizes the PREVALENCE estimate q, then propagates:
        R = TP / (TP + q*N_u)
        dR/dq = -TP*N_u / (TP + q*N_u)^2
        m_q = m_R / |dR/dq|
        n = 1.96^2 * q(1-q) / m_q^2

    Sizing directly in recall units is wrong by a factor of (dR/dq)^2 --
    ~22x at the declared workload, because the unflagged pool is ~232x the
    flagged one and any error in q is multiplied by that ratio.

    Enforces CLAUDE.md invariant 6: all inputs come from one declared workload.
    """
```

### Keeping docs honest

Audit at the Phase 12 gate, reported line by line:

- every invariant in `CLAUDE.md` enforced somewhere in code — name file and line
- every `SPEC.md` section matches shipped behaviour; flag drift
- every number in `README.md` traces to a file in `results/`
- **every statistical claim states the quantity estimated and the propagation used**
- every `DECISIONS.md` entry still accurate
- no TODO or placeholder in any committed document

---

## Pre-merge checklist

- [ ] gate criteria in `TASKS.md` met and reported
- [ ] tests pass, including the estimator coverage test
- [ ] `DECISIONS.md` entry for anything methodological — short, with the numbers and the derivation where statistical
- [ ] no narrative documentation written during the build
- [ ] no file over 10 MB, no real personal data, no secrets, no absolute local paths
- [ ] any commit that moved a measured number records before and after
- [ ] `main` still runs `scripts/run_all.py --smoke` end to end
