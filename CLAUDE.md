# CLAUDE.md

Persistent context. Read fully before writing code. `SPEC.md` has the technical detail, `TASKS.md` the build order, `CONTRIBUTING.md` the git and documentation rules, `DECISIONS.md` the rationale log, `DEMO.md` the thing this all has to produce.

---

## What we are building

**A warrant layer on top of AI detection.**

A detector produces a score. A **warrant** is a separate, time-bounded, evidence-backed statement about what that score is worth *right now, on this input distribution*. The product is not detection. It is the ability to say, with intervals:

> *We tell you what your error rate is on your traffic, we tell you when that number stops being true, and we tell you what it costs to keep it true.*

Three clauses, three mechanisms:

| Clause | Mechanism |
|---|---|
| what your error rate is on your traffic | On-traffic warrant via stratified estimation (Phase 6) |
| when that number stops being true | Envelope matrix + drift revocation (Phases 4–5) |
| what it costs to keep it true | The price list — computed, not typed (Phase 6) |

Every piece of work either implements one of those, validates one, or documents one. Anything else does not belong in this repo.

## The single most important distinction in this codebase

**Yield is exact. Rate is estimated.**

- *"We surfaced 850 real errors this month"* — a **count** of reviewed, confirmed items. Exact. Free. No sampling, no interval.
- *"We caught 14% of errors"* — a **rate**. Requires estimating how many errors sit in the traffic nobody reviewed. Costs labels. Always carries an interval.

Conflating these is the most damaging error available in this project, because it converts a free exact claim into an unbacked estimate and nobody notices. Every metric in the codebase is tagged `EXACT` or `ESTIMATED` in its type. User-facing output must show which.

## Non-negotiable invariants

Breaking any of these invalidates the result. If a change would require it, stop and flag rather than working around.

1. **Warrants are keyed by (detector, operating_point, eval_set).** Never by detector alone. An envelope violation is a property of the *input distribution*, so it invalidates every detector measured on that distribution simultaneously.
2. **Three warrant states, three behaviours.** `VALID` (validated, cleared), `REFUSED` (validated, failed), `UNVALIDATED` (never tested here). `UNVALIDATED` is the modal state in production and must never collapse into either of the others.
3. **Refusal has no override.** No flag, no env var, no admin bypass. If one exists, the entire product is theatre and a reviewer will look for exactly that.
4. **No point estimate reaches a user.** Every rate carries a 95% interval. Every interval names its `n`.
5. **Precision and recall travel together.** Never lift alone, never recall alone, never a blended F1 anywhere in the codebase.
6. **One declared workload.** All economics derive from a single parameter block in `config.yaml`. Mixing a flag rate from one scenario with a base rate from another is silently wrong and produces numbers that don't survive a reviewer with a calculator.
7. **Reviewers are blind.** The label queue never exposes flag status, score, stratum, or ordering signal. Anchoring across strata biases the estimator in the direction that flatters us.
8. **Every number in a document is computed by code.** If it cannot be traced to an artifact in `results/`, it does not go in the README, the deck, or the demo.
9. **Eval sets are frozen and content-hashed.** Changing one creates a new ID. Selection happens on validation; test is scored once per validation run.

## Silent failures — the list

These produce plausible output with nothing raised. They are why the control suite exists.

**Sizing prevalence and calling it recall.** `n = 1.96²·q(1−q)/m²` sizes a *prevalence* estimate. Recall is `TP/(TP+q·N_u)`, and in our regime `∂R/∂q ≈ −4.65` because the unflagged pool is ~232× the flagged one. Sample size scales as `1/margin²`, so mis-sizing this understates labels needed by ~22×. **Always propagate through the derivative. Never quote a sample size in recall units without it.**

**Scenario mixing.** Precision at n=10,000 comes from a 5%-flag-rate workload; recall at ~850 labels comes from a 1.48%-flag-rate workload. Quoting both in one table is wrong. All figures derive from one declared workload, computed by `controlplane/economics/sizing.py`.

**Right padding.** Position `−1` becomes a pad token, activations are meaningless, AUROC lands near 0.5 and reads as "the idea doesn't work."

**Selecting on test.** Layer, threshold, regularisation are validation decisions.

**Mean-pooling on long context.** The documented failure mode — mean-pooled linear probes collapse under long-context shift. Build max-of-rolling-means alongside and report both.

**Neyman allocation on day one.** Neyman needs per-band prevalence `q_h` you don't have yet. Month one runs proportional allocation at full SRS cost. Design for it; say it in the README.

**Label noise swallowing the interval.** A ±5pp recall interval is meaningless at inter-rater κ = 0.5. Double-label ~10% of both strata and publish κ next to every warrant.

**Fitting the scaler on the full set.** Train indices only.

**Label polarity.** Positive class is *incorrect*. Inverted gives `1 − AUROC`, which reads as a strong negative result and misdirects debugging for hours.

## What the probe is and is not

- A **trigger**. Its only output is a decision about where to spend an expensive check.
- **Never a verdict.** It does not block, filter, or gate a user-facing response. The action gate does that, and the gate's two most important rules do not consult any detector score.
- Tuned for recall; poor precision is acceptable by design and must be stated, not hidden.
- Never described as measuring truthfulness, faithfulness, or belief. It is a correlational classifier over activations. Use that language in docstrings, comments, and output.

## Layout

```
README.md          the deliverable — claim table, quickstart, scope
CLAUDE.md          this file
DECISIONS.md       append-only rationale log — stays at the root, heavily cited
Makefile           the four reproduction tiers
config.yaml        all knobs, including the single declared workload
docs/
  SPEC.md          technical specification — read before implementing
  TASKS.md         phased build order with gates
  CONTRIBUTING.md  git workflow and documentation rules — binding
  DEMO.md          the demo this must produce
  KICKOFF.md       the original brief
  ARCHITECTURE.md  what the system is and how the pieces fit
  METHODS.md       estimators, bootstraps, bands and their derivations
  LIMITATIONS.md   scope, declared gaps, open items
  CASES.md         the case matrix — every case, its test, its artifact
  PATHS.md         the 2026-08-29 move mapping, old path to new
  PROPOSAL.md      the business proposal
  README.md        index: which document answers which question
  DIAGRAMS.md      the system in diagrams, including the Round 1 to Round 2 journey
  ONBOARDING.md    the first hour, in order
  SETUP.md         laptop, GPU notebook, offline machine
  RUNBOOK.md       every script: what it reads, what it writes, when to run it
  TROUBLESHOOTING.md  what each exception and each refusal actually means
  ARTIFACTS.md     every file in results/ and evalsets/, and its fields
  TESTING.md       what the suite defends, suite by suite
  CODE_TOUR.md     what is in each package under controlplane/
  GLOSSARY.md      warrant, envelope, estimand, null band — defined once
  FAQ.md           reviewer questions, each answered with the artifact that settles it
  JOURNEY.md       what the project did, phase by phase, and what changed our mind
controlplane/
  model/           Finding, OperatingPoint, Warrant, Certificate, enums
  store/           SQLite, hash chain, retention, queries
  extract/         the only GPU stage: TriviaQA, activations, padding evidence
  evalsets/        hand-written corpora, builders, frozen registry
  validation/      control suite, /validate, warrant issuance and refusal
  matrix/          the (detector × envelope) warrant matrix and routing
  drift/           envelope computation, PSI, MMD, revocation ladder
  sampling/        NOT BUILT — Phase 6; DECISIONS 096
  economics/       feasibility.py, review.py. sizing.py — the price list —
                   is NOT BUILT; DECISIONS 096, amended by 099
  policy/          bundle loading, load-time warrant resolution
  detectors/       adapters: probe, presidio (qwen3guard/judge not built)
  gate/            NOT BUILT — Phase 9
  report/          renders results/ into markdown and plots
  gateway/         LiteLLM adapter — certificates on an OpenAI-format response
scripts/           thin CLI wrappers — no logic
tests/
evalsets/          frozen, content-hashed
policies/          versioned Rego/Cedar bundles
demo/              two-pane runner, stream player
notebooks/         run_on_kaggle.ipynb -- generated, never hand-edited
results/           all outputs
round1/            the Round 1 submission, moved whole and unmodified
```

Notebook JSON is not reviewable in a diff, so `notebooks/run_on_kaggle.ipynb` is
generated by `scripts/build_notebooks.py` and that script is what gets reviewed.
Edit the script, regenerate, then execute to populate outputs.

## Coding standards

- Python 3.11+. Type hints on every public function. Google-style docstrings that say *why*.
- **No logic in scripts, notebooks, or the demo runner.** They call `controlplane/`. Logic that lives only in a notebook is unreviewable.
- **No hardcoded constants.** Everything from `config.yaml`.
- Frozen dataclasses for the core model objects. These are records, not mutable state.
- Every stage writes to `results/` and reads from disk. Stages independently re-runnable without repeating expensive extraction.
- Seed everything. Log resolved config, config hash, git commit, library versions, device into every artifact.
- Assert at every boundary. Crash loudly. A silent wrong answer is far worse than a stack trace.
- **Never pipe a command whose exit status you need.** `cmd | tail` reports
  *tail's* status, so a failing test suite reads as green and a killed script
  reads as finished. Use `sh scripts/run.sh <cmd>`, which writes output to a
  file, echoes `EXIT=<real status>`, and propagates it. This trap fired twice in
  one session, the second time two messages after it was documented — reading is
  not a control, so the correct form has to be the default rather than the
  remembered one.

## Version control

Full rules in `CONTRIBUTING.md`. The parts you must not get wrong:

- **Commit continuously.** Four to ten atomic commits per phase, never one dump. An uncommitted tree at a phase gate is a failed gate.
- **Conventional Commits.** `feat` `fix` `docs` `test` `refactor` `chore` `exp`. `exp` for runs that produce or change artifacts in `results/`.
- **Any commit that moves a measured number records before and after in the body.**
- **Never run a script against a dirty tree.** Artifacts record the `HEAD` they were built from; `provenance()` must check `git status --porcelain` and set `dirty: true`.
- Phase branches, merged `--no-ff`, tagged at each gate. Tags are the rollback points.
- Never stage a file over 10 MB. Stop and ask.

## Documentation — written at the end, not during the build

**Prose documentation is a closing phase, not a running task.** During the
build, ship code and tests. Do not write or update `SPEC.md`, `TASKS.md`,
`README.md`, architecture notes, handover docs, or any other narrative document
as you go. **`DECISIONS.md` is the single exception** — see below.

This is a deliberate reversal of the original rule, which required a doc change
in the same commit as the code that caused it. That rule produced good
documents and cost a large fraction of the build time, and the judgement is
that the time is better spent on code while the code is what is uncertain.

### What is deferred

- **Contracts** (`CLAUDE.md`, `SPEC.md`, `TASKS.md`). Where shipped behaviour
  diverges from a contract, leave the contract alone and note the divergence in
  the commit message. Reconcile in the documentation phase.
- **Narrative documents** — architecture write-ups, handover docs, design
  notes, anything explaining the system to a reader. None of these during the
  build.
- **Generated** (`README.md`, `results/RESULTS.md`). Rendered from `results/`
  by code and never hand-written; the final render happens once, at the end.

### What still happens during the build

Four things, because each is either irreplaceable or nearly free:

1. **`DECISIONS.md` stays live.** It is the exception, and the reason is that it
   cannot be written later: it records *why* at the moment a measurement forced
   a choice, and six weeks on the numbers are recoverable but the reasoning is
   a reconstruction. It is also the document a technical judge actually reads.

   **Keep entries short.** One decision, the numbers that forced it, the
   alternative rejected, and the objection a reviewer would raise. Statistical
   decisions carry their derivation. Everything else — restating context the
   reader has, motivating the project, explaining the same idea twice — belongs
   in the documentation phase or nowhere.

2. **Commit messages carry the reasoning.** A commit that moves a measured
   number states before and after. A commit that makes a methodological choice
   states the choice and the numbers behind it. `git log` is the second half of
   the record.

3. **Docstrings.** These are code, not documentation — covered by the coding
   standards above. A function whose contract lives only in a separate document
   is a function nobody can call correctly.

4. **Statistical claims carry their derivation**, wherever they appear. A sample
   size, interval or projection states the quantity being estimated and the
   propagation used. This is the discipline that catches a prevalence interval
   labelled as a recall interval, and it is worth its cost at the point the
   number is produced rather than at the point it is written up.

### The documentation phase

Reconciles the contracts against shipped behaviour, writes the architecture
and handover material, and renders the generated documents. `DECISIONS.md` is
already written by then and only needs tidying. The Phase 12
audit is unchanged: every invariant enforced somewhere in code, every number in
`README.md` traceable to `results/`, every statistical claim carrying its
propagation.

**Orientation pages carry no measured numbers.** `docs/DIAGRAMS.md`,
`ONBOARDING.md`, `SETUP.md`, `RUNBOOK.md`, `TROUBLESHOOTING.md`,
`ARTIFACTS.md`, `TESTING.md`, `CODE_TOUR.md`, `GLOSSARY.md`, `FAQ.md` and
`JOURNEY.md` describe structure, procedure and vocabulary, and point at the
claim table and `results/` for every value. They change with the code rather
than with the run, so a re-run cannot leave them stating a stale figure — and
nothing in them needs to be re-verified when a number moves.

## Definition of done

`make verify` on a clean checkout, with `scripts/clean_clone_gate.py` for the
clone-and-run path, produces:

- `results/RESULTS.md` with the warrant matrix, the tier ladder, drift/revocation traces, the computed price list, and κ.
- A populated `README.md` whose every number traces to `results/`.
- Passing test suite, including the five controls and the estimator tests.
- Two runs at one seed produce identical numbers.
- Clean tree, phase tags, `DECISIONS.md` covering every methodological choice.
- A passing documentation audit (`CONTRIBUTING.md`, final section).

## Out of scope

- A gateway, proxy, or router. Sit behind LiteLLM; write an adapter, late, and cut it if pressed.
- A fine-tuned guard model. Wrap and warrant Qwen3Guard instead.
- A per-response bias score. Unsound. Either build the async cohort path or document why it can't be per-response.
- Any claim of adversarial robustness.
- Formal verification, multi-tenancy, RBAC, SSO, k8s.
- Runtime dependency on Llama Guard or ShieldGemma — neither licence is OSI-permissive and our public claim is a fully open self-hostable stack.
