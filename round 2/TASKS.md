# TASKS.md — build order

Phases are ordered so **you have a submittable system at the end of every phase**. Each has a gate. Do not begin a phase until the previous gate passes and is reported.

## Applies to every phase

Read `CONTRIBUTING.md` before Phase 0. These are part of every gate, not extras:

- Work on `phase/N-name`. Four to ten atomic commits per phase, never one.
- **Do not write narrative documentation during the build.** Architecture notes,
  handover docs and contract updates are a closing phase; see `CLAUDE.md`.
- `DECISIONS.md` is the exception and stays live. Short entries: the decision,
  the numbers, the alternative rejected.
- At the gate: clean tree, tests pass, `git merge --no-ff` to `main`, `git tag -a phase-N`.
- **A phase with uncommitted changes has not passed its gate**, whatever its numbers.

---

## Phase 0 — Scaffold

`git init` if needed. First commit is the existing documents as `docs: add project contracts and build plan`. Then branch.

Build `src/config.py`: dataclasses mirroring `config.yaml`, YAML load with validation, fractional-depth → layer-index resolution, SHA-256 config hashing, and `provenance()` returning config hash, git commit, `dirty` flag from `git status --porcelain`, library versions, device, UTC timestamp.

**Gate:** config hash stable across runs. `provenance()` correctly reports `dirty` after touching a file. Docs commit plus ≥3 scaffold commits. Tagged `phase-0`.

---

## Phase 1 — Data model and store

All frozen dataclasses per `SPEC.md` §1. SQLite store with hash chaining, retention config, and the required query paths.

**Gate:** a certificate round-trips. `test_hash_chain` passes — mutating a row breaks the chain, demonstrably. `test_warrant_key` passes. `test_yield_vs_rate` passes: an `EXACT` metric carries no interval, an `ESTIMATED` one always does.

---

## Phase 2 — Validation harness

The five controls. `/validate`. Warrant issuance and refusal. Then the tier ablation on `triviaqa-600`: T1 mean-pool, T1 max-of-rolling-means, T2 logprob family, T3 judge — all from one cached extraction, all with bootstrap CIs.

**Gate:** `/validate` completes in under a minute from cache. All five controls report with measured margins. The deliberately broken padding case is **rejected**. `test_no_override` passes. Tier ladder plotted with intervals.

**You are now submittable.** If everything after this failed, the validation harness plus tier ladder is already a differentiated entry.

---

## Phase 2.5 — Minimal demo runner

Two panes, a stream player, and the `Prove it` button wired to `/validate`. Ugly is fine. **Grow this every subsequent phase rather than building it at the end** — that is what keeps the submittable-at-every-phase invariant true.

**Gate:** a recorded stream plays through both panes; pressing `Prove it` runs a real validation and renders real control results.

---

## Phase 3 — Evaluation sets

Build `triviaqa-longctx-600`, `hinglish-pii-200`, `hard-negatives-200`, `canary-20`. Hand-build the middle two — do not generate them with an LLM and call them ground truth.

**Gate:** every set content-hashed and registered. `/validate` runs against each. FPR on `hard-negatives-200` measured and reported. Dedup and construction notes in `DECISIONS.md`.

---

## Phase 4 — The warrant matrix

Run the Phase 2 ablation once per eval set. Populate every (detector × envelope) cell with `VALID`, `REFUSED`, or `UNVALIDATED`. Implement matrix lookup and `route()`.

**Gate:** the matrix renders with every cell populated. Mean-pool is **REFUSED** on `triviaqa-longctx-600` while max-of-rolling-means holds a valid warrant with wider bounds — or, if not, the actual result is reported without adjustment. `test_three_states` passes.

---

## Phase 5 — Drift and revocation

Envelope computation stored in the warrant. PSI per feature, MMD on embeddings, sliding window with a 200-request minimum. The four-state ladder. Matrix routing on revocation. Model-version invalidation.

**Gate:** feed `triviaqa-longctx-600` as live traffic. The system detects the shift, revokes, **consults the matrix, routes to a detector holding a valid warrant on that envelope, adopts its bounds**, and writes a certificate explaining all of it — with no manual trigger. If no valid warrant exists, it refuses and enqueues.

---

## Phase 6 — On-traffic warrant

`src/economics/sizing.py` with the derivative propagation. Stratified estimator. Sampling scheduler with proportional allocation for month one and Neyman thereafter. Blinded interleaved label queue. Double-labelling at ~10% and Cohen's κ. The computed price list.

**Gate:**
- `test_sizing_derivative` passes — sizing for recall differs from sizing for prevalence by `(∂R/∂q)²`.
- `test_sizing_units` passes — the function refuses to return a number without a declared target quantity.
- `test_no_scenario_mixing` passes.
- `test_stratified_unbiased` passes: on synthetic data with known ground truth, nominal coverage over 1000 trials.
- `test_blinding` passes — queue payload carries no flag status, score, stratum, or ordering signal.
- The price list renders from the declared workload. **No figure typed by hand anywhere.**
- Measured Neyman design effect reported against the ~1.5× expectation.

---

## Phase 7 — Policy engine

OPA or Cedar. Versioned bundles. Load-time warrant resolution with hard failure. Three profiles. Weighted-error objective in the bundle.

**Gate:** a bundle referencing an unwarranted operating point fails to load with an error naming the missing warrant. Three profiles produce three different actions on one input, at three points on one measured curve.

---

## Phase 8 — Detector adapters

Probe, Qwen3Guard, LettuceDetect, Presidio, LLM judge. Then the Presidio three-configuration sequence end to end.

**Gate:** stock-configuration Presidio is **refused** a warrant on `hinglish-pii-200`, with measured recall in the refusal reason. All three configurations measured and reported. What `InAadhaarRecognizer` actually validates has been verified and recorded in `DECISIONS.md`.

---

## Phase 9 — Action gate and agent

Reversibility registry. Sticky session Rule-of-Two. The gate. Four-tool banking agent with an indirect-injection scenario.

**Gate:** an injected instruction passes every text-level check — visibly — and `transfer_funds` is blocked on reversibility plus session state. `test_gate_no_detector` passes: the first two rules produce correct decisions with all detector scores nulled.

---

## Phase 10 — Full demo harness

Grow Phase 2.5 into the five beats in `DEMO.md`. Conventional stack in the left pane at documented defaults. All three Presidio configurations shown on the left.

**Gate:** the full demo runs end to end twice without intervention. Backup recording captured. Beat 4 shows matrix routing, not just an alarm.

---

## Phase 11 — Gateway adapter *(lowest value — cut first if pressed)*

Sit behind LiteLLM. An adapter, not a gateway. Async path for deep checks on all traffic; inline path for the warranted fast tier.

**Gate:** an unmodified OpenAI-format client receives certificates with no application-code change.

---

## Phase 12 — Submission

Public repo, generated README, decisions log, architecture doc, demo video.

**Documentation audit** — run and report each line:
- every invariant in `CLAUDE.md` enforced somewhere in code; name file and line
- every `SPEC.md` section matches shipped behaviour; flag drift
- every number in `README.md` traces to a file in `results/`
- every statistical claim states the quantity estimated and the propagation used
- every `DECISIONS.md` entry still accurate; add entries decided during the build
- no TODO or placeholder text in any committed document

**History review** — `git log --oneline --graph` reads as a legible account to someone who wasn't here. Twelve phase tags, atomic commits, every commit that moved a measured number recording before and after.

**Gate:** clean-clone reproduction. Delete `results/`, run `run_all.py --smoke`, everything regenerates. Both audits reported.

---

## Reporting protocol

At every gate:

```
PHASE N — PASS / FAIL
Gate checks:   <each check, with its value>
Artifacts:     <files written>
Commits:       <count, and the tag applied>
Decisions:     <DECISIONS.md entries added>
Runtime:       <wall clock>
Surprises:     <anything unexpected, however minor>
Next:          <what phase N+1 will do>
```

**Stop and ask rather than working around, if:** any control fails and the cause isn't obvious; a warrant would need issuing with a failed control; the estimator's coverage test fails; a fix would require violating an invariant in `CLAUDE.md`; or a number you're about to write cannot be traced to an artifact.
