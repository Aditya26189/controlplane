# TASKS.md — build order

Build in stages. **Each stage has an acceptance gate. Do not begin a stage until the previous gate passes.** Report gate results before moving on.

The expensive stage is 3. Everything before it is cheap and exists to make sure stage 3 isn't run twice.

## Applies to every stage

Read `CONTRIBUTING.md` before Stage 0. These are part of every gate, not extras:

- Work on `stage/N-name`. Commit atomically as you go — four to ten commits per stage, never one.
- Documentation changes ship in the same commit as the code that required them.
- Log anything methodological in `DECISIONS.md` before the gate, not after.
- At the gate: clean working tree, tests pass, then `git merge --no-ff` to `main` and `git tag -a stage-N`.
- **A stage with uncommitted changes has not passed its gate**, regardless of its numbers.

---

## Stage 0 — Scaffold *(~30 min)*

`git init` if the repo isn't already initialised. First commit is the documents that already exist (`CLAUDE.md`, `SPEC.md`, `TASKS.md`, `CONTRIBUTING.md`, `DECISIONS.md`, `README_TEMPLATE.md`, `config.yaml`, `LICENSE`, `.gitignore`) as `docs: add project contracts and build plan`. Then branch to `stage/0-scaffold`.

Create `requirements.txt`, `src/` package with `__init__.py`, empty modules per the layout in `CLAUDE.md`, and `src/config.py` implementing:

- dataclasses mirroring `config.yaml`
- YAML loading with validation and clear errors on unknown or missing keys
- fractional-depth → absolute-layer-index resolution using the loaded model's layer count
- SHA-256 hashing of the resolved config
- a `provenance()` helper returning config hash, git commit, library versions, device name, UTC timestamp

- `provenance()` must additionally run `git status --porcelain` and record `dirty: true` when the tree is not clean, so no artifact can silently claim a commit it wasn't built from.

**Gate:**
- `python -c "from src.config import load_config; c = load_config('config.yaml'); print(c.config_hash)"` prints a stable hash. Re-running gives the same hash.
- `provenance()` returns a real git commit and correctly reports `dirty` — verify by touching a file and calling it again.
- History shows the docs commit and at least three scaffold commits. Merged to `main`, tagged `stage-0`.

---

## Stage 1 — Data *(~45 min)*

Implement `src/data.py`: load, normalise questions, deduplicate, filter empties, shuffle, subsample, split by `question_id` at 60/20/20, persist to `results/splits.parquet`.

Implement `normalize_answer` and `is_correct` per `SPEC.md` §2, including the short-alias guard.

Write `tests/test_normalization.py` and `tests/test_split_integrity.py`.

**Gate:**
- Both test files pass.
- Splits are pairwise disjoint on `question_id` **and** on normalised question string — assert in code, not just in tests.
- Print and report: rows loaded, duplicates dropped, final `n`, split sizes.

---

## Stage 2 — Model loading *(~45 min)*

Implement `src/model.py`: load Qwen2.5-7B-Instruct with NF4 via `BitsAndBytesConfig`, load the tokenizer, **force `padding_side="left"`**, set `pad_token` to `eos_token` if unset, and expose a `build_prompt(question)` using the chat template with `add_generation_prompt=True`.

Write `tests/test_padding_side.py`.

**Gate:**
- Model loads on the target GPU within memory budget; print peak allocated memory.
- Generate one answer to a fixed test question and print it — verify it looks like a sane short answer, not an empty string or a repetition loop.
- Print the resolved absolute layer indices from the configured fractional depths, and `model.config.num_hidden_layers`.

---

## Stage 3 — Extraction *(~30 min to write, 40–70 min to run)*

Implement `src/extract.py` and `scripts/01_extract.py` per `SPEC.md` §4.

**Before the full run, do all three of these:**

1. **Left-padding equivalence check** on a batch of 4 — batched vs unbatched last-token activations must match to `1e-2`. Fail hard if not. This is the single highest-value check in the repo; the failure it catches is silent.
2. **Smoke run** at `n=20`. Inspect the generated completions by eye. Confirm they are answers, not echoes of the prompt or empty strings.
3. **Base-rate check** on those 20 — roughly half should be correct. If it's 0/20 or 20/20, something is wrong with prompting or matching.

Then run the full extraction. Write `results/activations.npz` (fp16, keyed by layer), `results/labels.parquet`, `results/extract_meta.json`.

**Gate:**
- Equivalence check passed and its max deviation is logged.
- No NaN or Inf in any saved activation array.
- `question_id` round-trips through the length-sorted batching in the original order — assert it.
- Base rate on the full set is within 0.25–0.85. **If it is outside this range, stop and report rather than continuing.**
- Print: total runtime, examples/sec, median generation time per response, peak GPU memory.

---

## Stage 4 — Probe *(~1 h)*

Implement `src/probe.py` and `src/evaluate.py` per `SPEC.md` §5–6.

Order of operations, strictly:
1. Sweep layers × `C` grid, training on train, scoring AUROC on **validation**.
2. Pick the best (layer, `C`) on validation. Log the full sweep.
3. Pick the threshold on validation to hit the target flag rate.
4. **Only now** score the test set, once.
5. Bootstrap CIs, 1000 resamples.

Write `tests/test_polarity.py`, `tests/test_no_test_leakage.py`, `tests/test_determinism.py`.

Output `results/probe_sweep.json`, `results/probe.joblib`, `results/probe_test.json`.

**Gate:**
- Tests pass.
- The sweep table is printed in full — all layers, not just the winner.
- Positive class is confirmed to be "incorrect".
- The scaler is verifiably fit on train indices only.
- Report test AUROC with CI, measured `f`, `R`, precision, base rate.

**If test AUROC is at or below ~0.55:** stop and report it rather than tuning. Try, in this order: (a) confirm polarity, (b) confirm the equivalence check from Stage 3 still passes, (c) widen the layer range to include earlier and later layers, (d) increase `n_examples`. Do not fix a weak result by selecting on test. A weak result honestly reported is a valid outcome and belongs in `RESULTS.md`.

---

## Stage 5 — Economics, latency, report *(~1.5 h)*

Implement `src/economics.py`, `src/latency.py`, `src/report.py` and scripts `03`–`05` per `SPEC.md` §7–9.

Write `tests/test_economics.py` — including the invariance checks: lift must not change when base error rate or judge accuracy changes.

Generate `results/RESULTS.md` per `SPEC.md` §13, plus two plots: the layer sweep curve, and an ROC curve for the chosen layer with the operating threshold marked.

**Gate:**
- `lift == R/f` to floating-point tolerance.
- Random-sample policy yields lift exactly `1.0`.
- Invariance tests pass.
- `RESULTS.md` renders and contains all thirteen required elements, including an honest, specific limitations section.
- Latency ratio computed and reported with device metadata.

---

## Stage 6 — Optional: GSM8K negative control *(~1 h)*

Only if stages 0–5 are complete. Per `SPEC.md` §10. Expect a materially lower AUROC and write it up as a successful reproduction of a documented limitation.

**Gate:** result appears in `RESULTS.md` under a heading framing it as a reproduction.

---

## Stage 7 — Packaging *(~1 h)*

- `scripts/run_all.py` orchestrating 01→05, with `--smoke` completing at `n=100` in under five minutes.
- `notebooks/cascade_economics.ipynb` — imports from `src/`, loads `results/*.json`, displays the sweep, the ROC, the three-policy table, and the latency ratio. **No logic in the notebook.** This is what gets screen-recorded for the video, so make the three-policy table the visual centrepiece and readable at screen-capture resolution.
- `README.md` from `README_TEMPLATE.md`, every number pulled from `results/`.
- Full test suite green from a clean checkout.

**Documentation audit** — run it and report each line:
- every invariant in `CLAUDE.md` is enforced somewhere in code; name file and line for each
- every `SPEC.md` section matches shipped behaviour; flag any drift
- every number in `README.md` traces to a file in `results/`
- every `DECISIONS.md` entry is still accurate; add entries for anything decided during the build and not yet logged
- no TODO or placeholder text survives in any committed document

**History review** — `git log --oneline --graph` should read as a legible account of the build to someone who wasn't here. Seven stage tags, atomic commits, and every commit that moved a measured number recording before and after. This history is part of the Round 2 deliverable; a reviewer will scroll it.

**Gate:**
- Clean-clone reproduction: delete `results/`, run `run_all.py --smoke`, confirm every artifact regenerates.
- Documentation audit passes, reported line by line.
- History review done and reported.
- Working tree clean, merged to `main`, tagged `stage-7`.

---

## Reporting protocol

At every gate, report:

```
STAGE N — PASS / FAIL
Gate checks:   <each check, with its value>
Artifacts:     <files written>
Runtime:       <wall clock>
Surprises:     <anything unexpected, however minor>
Next:          <what stage N+1 will do>
```

**Stop and ask rather than working around, if:** the base rate lands outside 0.25–0.85; the equivalence check fails; test AUROC is at or below 0.55; anything would require paid compute; or a fix would require violating an invariant in `CLAUDE.md`.

Report surprises even when they seem harmless. A dedup count of 40% or a generation time three times the estimate are both signals worth surfacing early.
