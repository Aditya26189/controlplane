# CODE_TOUR.md — what is in each package

<sub>[🏠 Project README](../README.md) · [📚 Documentation index](README.md) · [🗺️ Diagrams](DIAGRAMS.md) · [📖 Glossary](GLOSSARY.md)</sub>

A map of `controlplane/`, package by package, with the file you want for each
job. Descriptions follow the modules' own docstrings; where a module exists to
prevent a specific failure, that failure is named.

The rule that shapes all of it: **logic lives in `controlplane/`.** `scripts/`,
`demo/` and `notebooks/` call it and render. Logic that lives only in a script
or a notebook is unreviewable.

**Contents:** [Where do I go for…](#where-do-i-go-for) · [model](#controlplanemodel--the-records) · [validation](#controlplanevalidation--measuring-and-issuing) · [evalsets](#controlplaneevalsets--the-envelopes) · [detectors](#controlplanedetectors--the-things-being-measured) · [extract](#controlplaneextract--the-gpu-stage) · [matrix](#controlplanematrix--detector--envelope) · [drift](#controlplanedrift--when-a-warrant-stops-being-true) · [policy](#controlplanepolicy--reading-the-warrant) · [store](#controlplanestore--the-ledger) · [economics](#controlplaneeconomics--what-can-and-cannot-be-derived) · [report](#controlplanereport--rendering-and-proving) · [demo, gateway, config](#controlplanedemo-gateway-and-config)

---

## Where do I go for…

| I want to | Open |
|---|---|
| understand the central record | `model/warrant.py` |
| see why a rate cannot exist without an interval | `model/metrics.py` |
| find the code that refuses a warrant | `validation/issuance.py` |
| read the five controls | `validation/controls.py` |
| see where test is scored | `validation/runner.py` |
| add a detector | `detectors/`, then `validation/text_runner.py` if it is stateless |
| add an evaluation set | `evalsets/builders.py` and `evalsets/registry.py` |
| understand a bootstrap interval | `validation/stats.py` |
| see how policy fails closed | `policy/resolution.py` |
| see how two detectors combine | `policy/compose.py` |
| understand drift bands | `drift/psi.py`, `drift/null_band.py` |
| find what the demo shows | `demo/session.py`, `report/beats.py` |
| find what checks the README | `report/claims.py` |

---

## `controlplane/model` — the records

Frozen dataclasses validated at construction. You cannot build an illegal record
and fix it later, which is the difference between an invariant and a convention.

| Module | Holds |
|---|---|
| `warrant.py` | **The central type.** What a detector's score is worth, on one distribution, until when. Enforces that a refusal states a reason and that the key includes the envelope |
| `metrics.py` | Metrics that cannot be constructed in violation of their own kind — `EXACT` refuses an interval, `ESTIMATED` requires one, and a blended F1 is refused by name |
| `findings.py` | Findings, operating points, distribution envelopes |
| `certificate.py` | The bounded, falsifiable assertion a decision leaves behind |
| `calibration.py` | The calibration claim, kept beside the warrant that makes it |
| `override.py` | Human overrides on escalated items — **cannot be constructed without stratum and draw probability**, because a record missing those would silently bias any recall computed from the review queue |
| `enums.py` | The enumerations every other record is built from |
| `serde.py` | Typed JSON round-tripping and canonical hashing |

---

## `controlplane/validation` — measuring and issuing

The largest package, and the one to read first.

| Module | Does |
|---|---|
| `runner.py` | `/validate`: one detector, one operating point, one envelope |
| `text_runner.py` | `/validate` for stateless text detectors — regex, checksum, rule-based |
| `controls.py` | The five controls, each reporting the band it applied |
| `issuance.py` | `issue_or_refuse` — **takes no argument that could promote a failing detector** |
| `metrics_builder.py` | The single place a metrics block is built from scores and labels |
| `stats.py` | AUROC, recall, precision, flag rate, bootstrap intervals |
| `scores.py` | Frozen per-item scores — the evidence a clean clone can actually check |
| `selection.py` | Threshold-selection uncertainty in a warranted recall |
| `paired.py` | Paired comparison of two probes on nested splits |
| `roc.py` | The ROC curve and the local slope at an operating point |
| `calibration.py` | Does the threshold still spend what it said it would? |
| `ablation.py` | The tier ladder: what each level of model access actually buys |
| `evalsets.py` | Eval sets and the cached extraction `/validate` runs from |
| `synthetic.py` | Synthetic sets and caches, for exercising the harness without a GPU |
| `warrant_stats.py` | Statistics specific to warrants |

**Read `runner.py` once, top to bottom.** The order is the argument: fit on
train, select on validation, run the controls, score test once, build metrics,
then issue or refuse.

---

## `controlplane/evalsets` — the envelopes

Hand-written corpora, builders, and the frozen registry. The PII sets are
**hand-built, not LLM-generated and called ground truth**.

| Module | Does |
|---|---|
| `builders.py` | Assembles corpora into frozen, content-hashed sets |
| `registry.py` | Freezing to disk, and the manifest that registers each set |
| `categories.py` | What each set's labels *mean* — the guard that refuses to measure a detector against labels for something else |
| `hinglish.py` | Hand-written Hinglish scenarios |
| `identifiers.py` | Synthetic Indian identifiers, and the forms they get disclosed in |
| `hard_negatives.py` | Boundary cases that **must be allowed** — the set that measures false positives |
| `banking.py` | The dual-labelled banking pilot |
| `resplit.py` | Reallocating a frozen set's declared splits without touching its data |

---

## `controlplane/detectors` — the things being measured

| Module | Is |
|---|---|
| `probe.py` | The linear probe: standardise, fit logistic regression, score. Raises rather than fitting or scoring in a way that leaks |
| `aggregation.py` | Pooling a sequence of residual-stream vectors into one feature vector — last-token, mean-pool, max-of-rolling-means |
| `presidio_adapter.py` | Presidio behind the warrant machinery, **unmodified**. Raises on an entity type it has no classification for rather than filtering it away |
| `presidio_custom.py` | The custom recognizers for the third configuration |
| `pii_reference.py` | Our own reference detector: patterns plus checksums |
| `identifiers_patterns.py` | Regular expressions for Indian identifiers |

**Each configuration is a separate detector id.** `presidio-stock` and
`presidio-enabled` are not one detector with a flag.

---

## `controlplane/extract` — the GPU stage

The only part that needs a GPU, and the only part most readers never run.

| Module | Does |
|---|---|
| `pipeline.py` | TriviaQA in, frozen eval sets and caches out |
| `activations.py` | Question-time activation extraction, and the padding evidence that proves it |
| `model.py` | Loading the model and tokenizer, with the left-padding assertion that has to hold |
| `triviaqa.py` | Loading, deduplication, question-level splitting, labelling |

---

## `controlplane/matrix` — detector × envelope

| Module | Does |
|---|---|
| `matrix.py` | The matrix itself, with its three states |
| `routing.py` | `route()` — given an envelope, which detector still holds a warrant here |

---

## `controlplane/drift` — when a warrant stops being true

| Module | Does |
|---|---|
| `monitor.py` | The sliding window and the revocation ladder |
| `psi.py` | Population Stability Index against a warrant's stored reference bins |
| `null_band.py` | The null distribution of PSI, so a band can be **checked rather than assumed** |
| `ladder.py` | A drift verdict decides what a warrant is still worth |
| `response.py` | Detect, transition, route — with **no manual trigger** |
| `certify.py` | The certificate a drift response leaves behind |
| `model_version.py` | Model-version invalidation: changed weights invalidate what was measured on the old ones |

---

## `controlplane/policy` — reading the warrant

| Module | Does |
|---|---|
| `bundle.py` | Policy as versioned, content-hashed data |
| `resolution.py` | **Load-time** warrant resolution — the fail-closed path |
| `engine.py` | The Rego adapter, via `regopy`; the only thing here that evaluates a rule. An engine cannot be built from an unresolved bundle, so no ordering exists in which a rule runs before its warrants were checked |
| `compose.py` | Composing two warranted detectors into one decision |
| `objective.py` | The weighted-error objective |
| `runner.py` | Issues the three operating points and decides one input under all three profiles |
| `errors.py` | `BundleError`, `WarrantResolutionError` — the second is **never a warning** |

Two traps `engine.py` documents and defends against, both silent: a Rego binding
that accepts a JSON *string* as input and then resolves nothing, returning
`ALLOW` for reasons no log would show; and an entrypoint typo that surfaces as a
native exception through the FFI boundary rather than a Python error.

---

## `controlplane/store` — the ledger

`ledger.py` is an append-only, hash-chained SQLite store of every issuance,
refusal, revocation, override and decision. Mutating a row breaks the chain,
demonstrably — that is a test, not a claim. It is generated and gitignored,
because the chain is only meaningful for the run that built it.

---

## `controlplane/economics` — what can and cannot be derived

| Module | Does |
|---|---|
| `feasibility.py` | The abstention floor: what **no** detector can do, however good. An impossibility result, derived from measured rates, needing no cost model |
| `review.py` | Review volume, and how many reviewed items a recall claim needs |

The price list — the module every cost figure was supposed to come from — was
specified and never built. That is why every cost, headcount and ROI figure in
this repository is a declared estimate and says so.

---

## `controlplane/report` — rendering and proving

| Module | Does |
|---|---|
| `claims.py` | Parses the README claim table and checks every number against its artifact |
| `results.py` | Renders `results/RESULTS.md`, with a hard refusal to print fixture numbers |
| `reproduce.py` | Re-derives the frozen-set evaluation from cached activations and diffs it |
| `clean_clone.py` | Clones the repository into a temporary directory and runs the gates inside it |
| `beats.py` | Assembles the demo's beats from committed artifacts. Reads; decides nothing |
| `plots.py` | Plots, each carrying the provenance of the numbers in it |

---

## `controlplane/demo`, `gateway`, and `config`

- **`demo/session.py`** decides; `demo/stream.py` replays a recorded request
  stream so the demo does not depend on live traffic. The runner in `demo/`
  renders and holds no logic.
- **`gateway/adapter.py`** is an OpenAI-format adapter that returns
  certificates — an adapter, not a gateway. Explicitly the lowest-value piece.
- **`config.py`** is the single source of every knob: typed dataclasses
  mirroring `config.yaml`, validation in `__post_init__` naming each invariant
  it enforces, SHA-256 config hashing, and `provenance()` — which shells out to
  `git status --porcelain` so an artifact can never silently claim a clean tree.

---

## Not built, and deliberately so

`controlplane/sampling/` (stratified estimation) and `controlplane/gate/` (the
action gate) are named in the contracts and do not exist. See
[LIMITATIONS.md](LIMITATIONS.md) §3 — the absence is declared rather than
discovered.

---

**See also:** [DIAGRAMS.md](DIAGRAMS.md) for how these fit together · [ARTIFACTS.md](ARTIFACTS.md) for what they write · [TESTING.md](TESTING.md) for what defends them.
