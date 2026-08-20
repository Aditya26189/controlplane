# CONTRIBUTING.md

Git and documentation workflow. Binding on agents and humans alike. Read alongside `CLAUDE.md`.

---

## Why this matters here

Two reasons, both specific to this project:

1. **This repo is a judged public artifact.** In Round 2 it is submitted as a public GitHub repository with a README. A repo whose history is one commit called `final` reads as a student project. A legible history of atomic commits reads as engineers who have shipped before. The history is part of the deliverable.
2. **Stage 3 costs a GPU hour.** When something downstream turns out to be wrong, you want to return to a known-good tagged state, not debug forward through uncommitted changes.

What we are *not* doing: release branches, semantic versioning, CI pipelines, coverage gates, changelogs. Wrong tool for a one-week deadline. Discipline where it pays, no ceremony where it doesn't.

---

## Branching

`main` is always green — the test suite passes and the pipeline runs.

Work happens on short-lived stage branches:

```
main
 ├── stage/0-scaffold
 ├── stage/1-data
 ├── stage/2-model
 ├── stage/3-extraction
 ├── stage/4-probe
 ├── stage/5-economics
 ├── stage/6-negative-control
 └── stage/7-packaging
```

At each gate in `TASKS.md`:

```bash
git checkout -b stage/3-extraction        # at stage start
# ... atomic commits during the stage ...
# gate passes:
git checkout main
git merge --no-ff stage/3-extraction -m "Stage 3: activation extraction"
git tag -a stage-3 -m "Gate passed: <one-line gate summary>"
git branch -d stage/3-extraction
```

`--no-ff` preserves the stage boundary in the history. The tags are what make rollback trivial: `git checkout stage-2` returns you to a state before extraction ran.

**Never merge a branch whose gate has not passed.** A failing stage stays on its branch until it's fixed.

---

## Commits

Conventional Commits, with one addition for this repo.

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat` · `fix` · `docs` · `test` · `refactor` · `chore` · `exp`

`exp` is ours: an experiment run that produces or updates artifacts in `results/`. It exists so a reviewer can filter the history down to "when did the numbers change, and why."

**Rules**

- Subject in the imperative, under 72 characters, no trailing period.
- **One logical change per commit.** Do not mix a bug fix with a refactor. If the message needs the word "and", it's two commits.
- Commit at every working checkpoint, not once per stage. A stage should produce four to ten commits.
- Never commit code that fails its own tests. Fix it, or commit it explicitly as `wip:` on a stage branch and squash before merging.

**Examples**

```
feat(extract): add left-padding equivalence check

Compares batched last-token activations against unbatched for a batch
of 4, tolerance 1e-2. Fails hard on mismatch.

Right-padding is the highest-risk silent failure in this pipeline:
position -1 becomes a pad token, every activation is meaningless, and
nothing raises. AUROC lands near 0.5 and reads as a negative result.

Refs: CLAUDE.md invariant 4, SPEC.md §4
```

```
exp(probe): full run, n=3000, seed 1729

Layer 17 selected on validation (AUROC 0.71).
Test AUROC 0.694 [0.658, 0.729], f=0.052, R=0.61, lift 11.7x.

config_hash: a3f8c21e
```

```
fix(data): guard short aliases against substring match

Alias "US" matched inside "just us", labelling wrong answers correct.
Aliases under 3 chars now require whole-token match. Base rate moved
0.71 -> 0.63.

Refs: SPEC.md §2
```

Note the third one: it records that a fix **moved a headline number**. Any commit that changes a measured value says so in the body, with before and after. This is the single most useful habit in the repo — when a judge asks "why is the base rate 0.63?", the answer is one `git log -S` away.

---

## What is and isn't committed

**Committed — these are the evidence:**
- all source, tests, config, documentation
- `results/*.json` — the measured numbers
- `results/RESULTS.md`
- `results/*.png` — plots
- `notebooks/cascade_economics.ipynb` **with its outputs intact**

That last one is a deliberate exception to the usual rule about stripping notebook output. This notebook renders on GitHub for judges who will never run it, so its outputs are the point. Do not add `nbstripout`.

**Not committed — regenerable, and large:**
- `results/activations.npz` (~150 MB)
- `results/*.parquet`
- `results/probe.joblib`
- caches, virtualenvs, `__pycache__`

Already covered in `.gitignore`. **If a `git add` would stage a file over 10 MB, stop and ask** rather than committing it or extending `.gitignore` unilaterally.

---

## The provenance ordering problem

Every artifact embeds the git commit it was generated from. But committing the artifact creates a *new* commit — so the hash inside the file can never be the hash of the commit containing it.

The convention, and it must be applied consistently:

> **`git_commit` in an artifact is `HEAD` at the moment the script ran** — that is, the commit of the *code* that produced it.

So the sequence is: commit the code, run the script, then commit the artifacts in a separate `exp:` commit. Never run a script against a dirty working tree — the recorded hash would be a lie. `provenance()` must call `git status --porcelain` and record `dirty: true` if the tree isn't clean, so a reviewer can see it.

---

## Documentation

Three tiers, with different rules.

### Tier 1 — Contracts: `CLAUDE.md`, `SPEC.md`, `TASKS.md`

The design. Read by the agent before every work session.

**Rule: a doc change ships in the same commit as the code change that made it necessary.** Never a trailing "update docs" commit — by then the reason is forgotten and the doc is written from the code instead of from the intent.

If you change an invariant, a default, a method, or a file layout, the contract changes in that same commit. If you find yourself writing code that contradicts `SPEC.md`, stop: either the code is wrong, or the spec is out of date and needs updating *first*, deliberately, with the reasoning recorded in `DECISIONS.md`.

### Tier 2 — Rationale: `DECISIONS.md`

Append-only log of every methodological choice. **The most important document in the repo for competition purposes**, because it is the direct answer to every "why did you do it that way?" a judge can ask.

One entry per decision, newest at the bottom, never edited or deleted — a decision that gets reversed is superseded by a new entry that links back to it.

Log a decision when it is one a reviewer could reasonably challenge. Dataset choice, split strategy, label rule, layer selection procedure, metric choice, threshold procedure. Not: variable names, file layout, formatting.

### Tier 3 — Generated: `README.md`, `results/RESULTS.md`

Derived from `results/`. **Never hand-edit a number in either.** If a number is wrong, the pipeline is wrong; fix the pipeline and regenerate.

Prose in the README can be edited by hand. Numbers cannot. `README_TEMPLATE.md` is the source; `README.md` is its rendered output.

### Docstrings

Google style, on every public function. The docstring says **why**, not what — the signature already says what. Anywhere an invariant from `CLAUDE.md` is being enforced, name it:

```python
def extract_activations(prompts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    """Extract question-time activations at the final prompt token.

    Runs a prefill-only forward pass so hidden states for every layer come
    from one call, making the layer sweep free. Deliberately does not pass
    output_hidden_states into generate(), which retains states for every
    decode step and exhausts a 16GB GPU.

    Requires left padding (CLAUDE.md invariant 4) — with right padding,
    position -1 is a pad token and every returned vector is meaningless,
    with nothing raised.

    Returns:
        Layer index -> (n_prompts, hidden_size) float32 array, in input order.
    """
```

### Keeping docs honest

At the Stage 7 gate, run a documentation audit and report it:

- every invariant in `CLAUDE.md` is enforced somewhere in code — name the file and line
- every `SPEC.md` section corresponds to shipped behaviour; flag drift
- every number in `README.md` traces to a file in `results/`
- every `DECISIONS.md` entry is still accurate
- no TODO or placeholder text survives in a committed document

---

## Pre-merge checklist

Before merging any stage branch to `main`:

- [ ] gate criteria in `TASKS.md` met and reported
- [ ] tests pass
- [ ] docs updated in the same commits as the code
- [ ] `DECISIONS.md` has an entry for anything methodological
- [ ] no file over 10 MB staged
- [ ] no secrets, tokens, or absolute local paths committed
- [ ] any commit that moved a measured number records before and after
- [ ] `main` still runs `scripts/run_all.py --smoke` end to end
