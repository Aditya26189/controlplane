# PATHS.md — what moved on 2026-08-29, and where it went

Every artifact in this repository records the commit it was generated from.
Moving a file that an artifact points at does not error; it produces an
artifact referencing a path that no longer exists, silently. That is the
failure class this project is about, so the reorganisation was audited before
anything moved and the mapping is recorded here.

Rationale: `DECISIONS.md` 095. Reproduction: `make verify`.

---

## Why anything moved

The repository is named `controlplane`. Until this date, cloning it landed a
reader on **Round 1** — the probe/cascade experiment — while the Round 2
control plane sat one level down in a directory whose name contains a space.
Round 2 is now the repository root; Round 1 is under `round1/`, moved whole
and unmodified.

## The four move commits

| commit | what moved | files |
|---|---|---|
| `c4bb31f` | Round 1 → `round1/` | 81 |
| `a90625a` | `round 2/` → repository root | 172 |
| `04a566e` | `src/` → `controlplane/` | 74 |
| `a0680f8` | five contracts → `docs/` | 5 |

Every path is a `git mv`. Nothing was deleted, nothing was recreated, and
`git log --follow` reaches the original commits through every rename —
verified on `round1/src/probe.py`, `round1/README.md`,
`round1/results/economics.json`, `controlplane/policy/compose.py`,
`controlplane/model/warrant.py`, `evalsets/hinglish-pii-200.json` and
`results/detectors.json`.

---

## Mapping

### Round 1 — `c4bb31f`

| old | new |
|---|---|
| `README.md` | `round1/README.md` |
| `CLAUDE.md`, `SPEC.md`, `TASKS.md`, `CONTRIBUTING.md`, `DECISIONS.md`, `KICKOFF.md`, `README_TEMPLATE.md` | `round1/<same>` |
| `config.yaml`, `requirements.txt`, `LICENSE`, `.gitignore` | `round1/<same>` |
| `src/`, `tests/`, `scripts/`, `results/`, `notebooks/`, `docs/` | `round1/<same>/` |
| `results_bundle/`, `results_bundle (1)/`, `results_bundle (2)/` | `round1/<same>/` |

Round 1's `results/` travelled with its own `config.yaml`, `src/` and
`scripts/`, so every artifact in it still resolves against the tree that
produced it. Its `DECISIONS.md` is a separate file from this project's and is
not merged into it.

### Round 2 — `a90625a`

| old | new |
|---|---|
| `round 2/<anything>` | `<anything>` |

Directory names are unchanged. `results/`, `evalsets/`, `policies/`,
`scripts/`, `tests/`, `demo/`, `notebooks/`, `kaggle/` keep the names their
artifacts, decisions, commit messages and the notes ref already use.

The ignored working files travelled with their directories: the extraction
caches (`results/cache-*.npz`), the audit databases (`results/*.db`) and the
long-context eval sets (`evalsets/*-longctx.json`, `evalsets/triviaqa-longctx-600.json`).

### The package — `04a566e`

| old | new |
|---|---|
| `round 2/src/` | `controlplane/` |
| `from src.…` (248 sites) | `from controlplane.…` |

### The contracts — `a0680f8`

| old | new |
|---|---|
| `SPEC.md` | `docs/SPEC.md` |
| `TASKS.md` | `docs/TASKS.md` |
| `CONTRIBUTING.md` | `docs/CONTRIBUTING.md` |
| `DEMO.md` | `docs/DEMO.md` |
| `KICKOFF.md` | `docs/KICKOFF.md` |

`DECISIONS.md` and `CLAUDE.md` **did not move**. `DECISIONS.md` is cited 180
times across the tree and 8 times inside itself, where the citations cannot be
corrected because the file is append-only. `CLAUDE.md` is read from the
project root by the agent harness. E.2's rule applies to both: root clutter
costs less than a broken reference.

---

## What was deliberately NOT updated

Two classes of string still say `round 2` or `src`. Both are correct as they
stand and changing either would destroy information.

### `provenance.dirty_paths` in every results artifact

```
provenance/dirty_paths[1] = round 2/CLAUDE.md
provenance/dirty_paths[5] = round 2/src/config.py
```

This is a record of **which files were uncommitted when a number was
produced** — a fact about a past tree, not a pointer to an input. Rewriting it
would falsify provenance. Read these as historical; the file they name is at
its new path in the table above.

### The synthetic generator literals

`controlplane/validation/synthetic.py:148` and `:317` still read
`"generator": "src.validation.synthetic.…"`. Both sit inside `construction`,
`construction` feeds `EvalSet.content_hash`, the content hash **is** the
envelope id, and the envelope id is the third element of the warrant key.
Verified before the rename: the same construction dict hashes to `6b7654bb…`
with `src.` and `a682d25f…` with `controlplane.`. Renaming them would
re-issue every synthetic fixture under a new envelope and orphan the warrants
in `results/fixtures/`. They are frozen identities, not import paths, and both
are commented in place.

`canary-src`, an eval-set id in `tests/test_smoke.py`, is untouched for the
same reason.

---

## What made the move safe

The audit found that **no artifact references an eval set by path.** Every one
carries `eval_set_id` and a content hash, and `evalsets/manifest.json` stores a
bare filename resolved against `paths.evalsets_dir`. The only path-shaped
string inside any artifact is `results/controlplane.db`, an echo of
`store.path`, and `results/` kept its name.

That is why 172 files moved without a single artifact being regenerated and
without a single number changing. It is a property of how the eval-set
registry was built, not of care taken during the move.

---

## Layout freeze

The layout is frozen as of `DECISIONS.md` 097 and the `repo-v1` tag. Changes
after that point are additive only: new files, new docs, new tests. No moves,
no renames — the demo harness hardcodes paths.
