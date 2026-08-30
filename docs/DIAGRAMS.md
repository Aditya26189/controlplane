# DIAGRAMS.md — the system in pictures

Eleven diagrams, in the order a reader needs them: what the system *is*, how
a number gets made, what happens to it afterwards, how the repository proves any
of it, and where every directory sits. Every diagram names the code it describes.

These render natively on GitHub. Nothing here carries a measured number — the
numbers live in [README.md](../README.md)'s claim table and in `results/`,
which is where a reader should check them.

**Contents:** [1 The three objects](#1-the-three-objects) · [2 The pipeline](#2-the-pipeline-stage-by-stage) · [3 Warrant lifecycle](#3-the-warrant-lifecycle) · [4 A validation run](#4-what-happens-inside-one-validation-run) · [5 The five controls](#5-the-five-controls) · [6 Policy](#6-policy-reads-the-warrant-not-the-score) · [7 Composition](#7-composing-two-detectors) · [8 Drift](#8-drift-and-the-revocation-ladder) · [9 Verification](#9-how-the-repository-proves-itself) · [10 The journey](#10-what-we-built-round-1-to-round-2) · [11 The repository](#11-the-repository-end-to-end)

---

## 1. The three objects

The whole system is one sentence: **a detector produces a score, a warrant says
what that score is worth on a named distribution, and policy reads the warrant
rather than the score.**

```mermaid
flowchart LR
    subgraph D["DETECTOR — controlplane/detectors/"]
        D1["produces a score<br/>knows nothing about how good it is<br/>identity includes its configuration"]
    end
    subgraph E["ENVELOPE — controlplane/evalsets/"]
        E1["frozen, content-hashed eval set<br/>PLUS a label definition<br/>change it and it is a different set"]
    end
    subgraph W["WARRANT — controlplane/model/warrant.py"]
        W1["keyed by detector x operating point x envelope<br/>bounds with intervals, five control results<br/>issued at, expires in 24h<br/>VALID or REFUSED with a reason"]
    end

    D -->|measured on| E
    E -->|evidence for| W
    W -->|read by| P["POLICY — controlplane/policy/"]
    D -. "never read directly" .-> P

    style W fill:#1f6feb22,stroke:#1f6feb
    style P fill:#2da44e22,stroke:#2da44e
```

The dotted line is the point of the project. A conventional stack draws it
solid: policy thresholds a raw score, and a detector that quietly stopped
working looks exactly like one that works.

**Why the detector's identity includes its config:** `presidio-stock` and
`presidio-enabled` are two detectors, not one detector with a flag. A shared id
would let a warrant measured on one be quoted for the other.

---

## 2. The pipeline, stage by stage

Each stage is a separate process that writes to `results/` and reads the
previous stage's output from disk. Only the first needs a GPU.

```mermaid
flowchart TD
    CFG["config.yaml<br/>one workload, one seed<br/>hashed into every artifact"]

    S0["00_extract.py<br/>GPU, ~1h"] --> A0["results/cache-*.npz<br/>gitignored, ~100 MB"]
    S1["01_build_evalsets.py"] --> A1["evalsets/*.json + manifest.json<br/>frozen, content-hashed"]
    A0 --> S2["02_validate.py"]
    A1 --> S2
    S2 --> A2["results/validation-*.json"]
    A2 --> S3["03_matrix.py"] --> A3["results/warrant_matrix.json<br/>results/RESULTS.md"]
    A0 --> S4["04_transfer.py"] --> A4["results/transfer-*.json"]
    A0 --> S5["05_canary.py"] --> A5["canary eval sets + control evidence"]
    A2 --> S6["06_reconcile.py"] --> A6["results/reconciliation.json"]
    A2 --> S7["07_policy.py"] --> A7["results/policy-*.json"]
    A0 --> S8["08_paired.py"] --> A8["results/paired_comparison.json"]
    A1 --> S9["09_detectors.py"] --> A9["results/detectors.json"]
    A2 --> S10["10_freeze_scores.py"] --> A10["results/scores/*.json<br/>committed, ~200 KB"]
    A7 --> S11["11_feasibility.py"] --> A11["results/feasibility.json"]

    CFG -.-> S0 & S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8 & S9 & S10 & S11

    style S0 fill:#d1242f22,stroke:#d1242f
    style A10 fill:#2da44e22,stroke:#2da44e
```

Stages `12`–`15` and `17` are the banking pilot and the Presidio coverage
claim; they hang off the same config and write their own artifacts. Full list
with flags in [RUNBOOK.md](RUNBOOK.md).

**The green box is why a stranger can check this repository.** `results/scores/`
holds the per-item labels, scores and question ids behind every measured block.
It is small enough to commit, so `make verify` can recompute every metric on a
fresh clone with no GPU and no cache.

---

## 3. The warrant lifecycle

Three states, three behaviours, and `UNVALIDATED` never collapses into either
of the others.

```mermaid
stateDiagram-v2
    [*] --> UNVALIDATED: a detector exists,<br/>this envelope never measured

    UNVALIDATED --> VALID: validation run,<br/>all five controls pass,<br/>bounds clear the floor
    UNVALIDATED --> REFUSED: validation run,<br/>any control fails or<br/>a bound misses the floor

    VALID --> EXPIRED: 24 hours
    VALID --> REVOKED: drift past the<br/>envelope boundary
    VALID --> INVALIDATED: model version changed

    EXPIRED --> UNVALIDATED: re-measure
    REVOKED --> UNVALIDATED: re-measure
    INVALIDATED --> UNVALIDATED: re-measure

    REFUSED --> [*]: no override exists

    note right of REFUSED
        There is no force, no admin bypass,
        no min_confidence to relax.
        issue_or_refuse takes no argument
        that could promote a failing detector.
    end note

    note left of UNVALIDATED
        The modal state in any real deployment.
        Most cells of the matrix are here, and
        that is the expected shape, not a gap.
    end note
```

A system that cannot say *"this detector has never been measured on this
distribution"* will instead say something confident and wrong.

`controlplane/model/warrant.py`, `controlplane/validation/issuance.py`

---

## 4. What happens inside one validation run

```mermaid
sequenceDiagram
    autonumber
    participant R as validation/runner.py
    participant E as evalsets (frozen)
    participant D as detector
    participant C as validation/controls.py
    participant M as metrics_builder.py
    participant I as validation/issuance.py

    R->>E: load envelope, assert content hash
    R->>E: check label category matches the detector
    R->>D: fit on TRAIN split only
    R->>D: select threshold on VALIDATION, to the declared budget
    Note over R,D: test is never consulted by any selection
    R->>D: score TEST once
    R->>C: run the five controls
    C-->>R: pass / fail, each with the band it applied
    R->>M: build metrics from scores and labels
    M-->>R: AUROC, recall, precision, flag rate,<br/>each with a bootstrap interval
    R->>I: issue_or_refuse(controls, metrics, floors)
    alt every control passed and every bound clears
        I-->>R: VALID warrant, bounds + 24h expiry
    else anything failed
        I-->>R: REFUSED, reason naming every failed criterion
    end
    R->>R: write results/validation-*.json
```

The scaler is fit on train indices only, and the bootstrap resamples over
`question_id` rather than rows — TriviaQA ships several items per question, and
a row-level bootstrap would treat correlated items as independent and produce
intervals that are too narrow. [METHODS.md](METHODS.md) §1.

---

## 5. The five controls

Every validation runs all five. **Any failure refuses the warrant**, and
nothing downstream can promote it back.

```mermaid
flowchart TD
    V["a validation run"] --> P1 & P2 & P3 & P4 & P5

    P1["padding_fault<br/>a right-padded variant<br/>must be REJECTED"]
    P2["label_shuffle<br/>AUROC must NOT survive<br/>permuting the labels"]
    P3["null_feature<br/>a probe on noise must<br/>score inside the null band"]
    P4["canary<br/>recall must be 1.0 on a<br/>deliberately easy set"]
    P5["determinism<br/>two runs at one seed<br/>must be bit-identical"]

    P1 & P2 & P3 & P4 & P5 --> G{"all five pass?"}
    G -->|yes| OK["continue to the bound checks"]
    G -->|no| NO["REFUSED<br/>reason names every failed criterion"]

    style NO fill:#d1242f22,stroke:#d1242f
    style OK fill:#2da44e22,stroke:#2da44e
```

**The control that must fail to pass.** `padding_fault` deliberately builds a
right-padded variant and requires the check to *reject* it. Without that, a
tolerance loosened until the check passed is indistinguishable from a check
that works. With right padding, position −1 of a batch is a pad token, every
activation is meaningless, nothing raises, and AUROC lands near chance — which
reads as "the idea doesn't work" rather than "the code is broken".

**The null band is measured, not looked up.** The Hanley–McNeil closed form
understates the true spread here, because a fitted probe's scores are not
exchangeable under label permutation, so each control simulates its own null at
construction and reports the band it applied. `DECISIONS.md` 029, 031, 070.

---

## 6. Policy reads the warrant, not the score

```mermaid
flowchart TD
    B["policies/*/bundle.yaml<br/>versioned, content-hashed"] --> L{"load-time resolution<br/>policy/resolution.py"}
    W[("warrant matrix<br/>results/warrant_matrix.json")] --> L

    L -->|"every named operating point<br/>has a VALID warrant"| LOADED["bundle loads"]
    L -->|"any operating point has<br/>no warrant behind it"| FAIL["BUNDLE FAILS TO LOAD<br/>fail-closed, not a warning"]

    LOADED --> CHK{"warrant's interval BOUND<br/>vs the profile's declared floor"}
    CHK -->|"bound clears the floor"| ACT["the profile's action"]
    CHK -->|"bound misses"| DEF["the profile default"]

    style FAIL fill:#d1242f22,stroke:#d1242f
```

Two things in that diagram are load-bearing:

- **The comparison uses the interval bound, not the point estimate.** A point
  estimate that clears a floor with an interval straddling it has not cleared
  anything.
- **A bundle naming an operating point with no warrant does not load.** Not a
  warning, not a default — a load failure. `controlplane/policy/errors.py`

The three profiles — `customer_support`, `internal_knowledge`,
`decision_support` — are **three points on one measured ROC**, not three
invented thresholds. Same detector, same envelope; only the flag-rate budget
moves.

---

## 7. Composing two detectors

Two warranted detectors, one decision. The rules were written before the code
(`DECISIONS.md` 088).

```mermaid
flowchart TD
    A["detector A<br/>VALID warrant"] --> C{"compose<br/>policy/compose.py"}
    B["detector B<br/>state varies"] --> C

    C --> R1["both VALID<br/>bounds stay keyed per detector<br/>NEVER merged"]
    C --> R2["one REFUSED<br/>the other's warrant still stands<br/>what was not checked is recorded"]
    C --> R3["one UNVALIDATED and it FIRES<br/>profile default, not its own action"]
    C --> R4["one UNVALIDATED and it is SILENT<br/>nothing triggers"]

    style R4 fill:#8250df22,stroke:#8250df
```

Case 4 is the one people get wrong. **Silence from an unmeasured detector is
not evidence.** Two detectors agreeing does not strengthen either bound either —
it is not a vote, and the bounds are never merged. All four cases are
enumerated with their tests in [CASES.md](CASES.md) §2.

---

## 8. Drift and the revocation ladder

```mermaid
flowchart TD
    T["live traffic window"] --> F["envelope features<br/>token length, script mix,<br/>embedding centroid distance,<br/>reference perplexity, category mix"]
    F --> M["PSI + MMD vs the warrant's<br/>stored reference bins<br/>drift/psi.py, drift/monitor.py"]
    M --> Q{"past the significant<br/>boundary?"}
    Q -->|no| K["warrant stands"]
    Q -->|yes| REV["REVOKE the warrant"]
    REV --> RT{"another detector holding<br/>a warrant on this envelope?"}
    RT -->|yes| AD["route to it<br/>adopt ITS bounds, cite ITS warrant"]
    RT -->|no| RF["REFUSE and enqueue<br/>for re-measurement"]

    NB["null_band.py simulates PSI's null<br/>at construction and REFUSES a config<br/>whose false-alarm rate would exceed 5%"] -.-> M

    style RF fill:#d1242f22,stroke:#d1242f
```

PSI's 0.10 / 0.25 bands are credit-scoring rules of thumb quoted without their
sample size, and they are not scale-free — the null grows as `(k−1)/n`. A
threshold that fires on 30% of stable windows is not a monitor; it is noise with
a dashboard. `DECISIONS.md` 070.

The measured demonstration is one long-context shift across three probe
aggregations with nothing retrained: last-token holds its warrant, mean-pool
collapses to chance and **flags nothing** — which a conventional dashboard reads
as clean traffic. That is the failure this system exists to name.
[ARCHITECTURE.md](ARCHITECTURE.md), "Drift".

---

## 9. How the repository proves itself

`make verify` is the "prove it" button. Three checks, weakest first, each
proving something the one before it cannot.

```mermaid
flowchart TD
    T1["TIER 1 — the claim table<br/>every number in README.md resolved<br/>against the artifact and field it names"]
    T2["TIER 2 — frozen scores<br/>every metrics block recomputed from<br/>results/scores/, same builder and seed"]
    T3["TIER 3 — activations<br/>the frozen scores re-derived from<br/>the cached activations"]

    T1 --> W1["cannot catch a README and its<br/>artifacts that went stale TOGETHER"]
    W1 --> T2
    T2 --> W2["cannot prove those scores came from<br/>the model and probe the artifact names"]
    W2 --> T3
    T3 --> W3["caches are gitignored, so on a fresh<br/>clone this reports SKIPPED —<br/>never a pass it did not earn"]

    style T1 fill:#2da44e22,stroke:#2da44e
    style T2 fill:#2da44e22,stroke:#2da44e
    style T3 fill:#bf871122,stroke:#bf8711
```

All three exit non-zero on drift, and **nothing in `results/` is written by any
of them** — the re-run goes to a scratch directory, so a failed verification
cannot damage the evidence it was checking.

Running order for a newcomer: `make smoke` (under a minute), `make verify`
(a few minutes), `make test`. See [SETUP.md](SETUP.md).

---

## 10. What we built, Round 1 to Round 2

The step-by-step of the project itself. Round 1 measured one detector; Round 2
built the layer that says what any detector's number is worth.

```mermaid
flowchart TD
    subgraph R1["ROUND 1 — now round1/, moved whole and unmodified"]
        r1a["question-time linear probe<br/>on Qwen2.5-7B activations"]
        r1b["cascade economics: lift = R/f<br/>with its ceiling 1/base_rate"]
        r1c["left-padding equivalence check<br/>+ its positive control"]
        r1a --> r1b --> r1c
    end

    R1 ==>|"the question Round 1 could not answer:<br/>what is that number worth on<br/>traffic nobody measured it on?"| R2

    subgraph R2["ROUND 2 — the repository root"]
        p0["Phase 0-1 — scaffold, data model,<br/>hash-chained ledger"]
        p2["Phase 2 — validation harness,<br/>the five controls, issue_or_refuse"]
        p3["Phase 3 — frozen, content-hashed<br/>eval sets and label categories"]
        p4["Phase 4 — the detector x envelope<br/>warrant matrix"]
        p5["Phase 5 — drift, PSI/MMD,<br/>the revocation ladder"]
        p7["Phase 7 — policy as data,<br/>load-time warrant resolution"]
        p8["Phase 8 — detector adapters:<br/>Presidio, our PII reference"]
        p10["Phase 10-12 — demo, pilot,<br/>claim table, clean-clone gate"]
        p0 --> p2 --> p3 --> p4 --> p5 --> p7 --> p8 --> p10
    end

    R2 -.->|"specified, NOT built"| NB["Phase 6 price list<br/>Phase 9 action gate<br/>docs/LIMITATIONS.md §3"]

    style NB fill:#d1242f22,stroke:#d1242f
```

Two things that diagram is deliberately honest about:

- **Phase 6 and Phase 9 were specified and never written.** Every cost,
  headcount or ROI figure in this repository is therefore a declared estimate
  and says so. The feasibility bound is the exception — it derives from measured
  rates and needs no cost model.
- **The move is audited.** Round 1 became `round1/` and Round 2 became the root
  via `git mv`, so `git log --follow` reaches the original commits through every
  rename. The mapping is in [PATHS.md](PATHS.md).

The narrative version, with what each phase cost and what it changed, is in
[JOURNEY.md](JOURNEY.md).

---

## 11. The repository, end to end

Every directory, what is in it, and which way the arrows point. This is the map
to open beside a first clone.

```mermaid
flowchart TB
    subgraph CONTRACT["CONTRACTS — read before writing code"]
        C1["CLAUDE.md<br/>invariants, silent failures, scope"]
        C2["docs/SPEC.md<br/>technical specification"]
        C3["docs/TASKS.md<br/>phases and their gates"]
        C4["docs/CONTRIBUTING.md<br/>git and documentation rules"]
    end

    subgraph INPUT["DECLARED INPUTS"]
        I1["config.yaml<br/>one workload, one seed, hashed"]
        I2["evalsets/<br/>frozen, content-hashed sets<br/>+ manifest.json"]
        I3["policies/<br/>customer_support<br/>internal_knowledge<br/>decision_support"]
    end

    subgraph CODE["controlplane/ — 84 modules, all the logic"]
        M1["model/ · store/<br/>records and the hash-chained ledger"]
        M2["extract/ · detectors/<br/>the GPU stage and the things measured"]
        M3["evalsets/ · validation/<br/>envelopes, controls, issue_or_refuse"]
        M4["matrix/ · drift/<br/>warrant matrix, PSI, revocation ladder"]
        M5["policy/ · economics/<br/>fail-closed resolution, feasibility bound"]
        M6["report/ · demo/ · gateway/<br/>rendering, the two panes, the adapter"]
    end

    subgraph THIN["THIN WRAPPERS — no logic"]
        W1["scripts/<br/>00_extract … 17_presidio_coverage<br/>smoke.py · verify.py"]
        W2["demo/<br/>run_demo.py · show_beats.py"]
        W3["notebooks/<br/>generated, never hand-edited"]
    end

    subgraph EVID["EVIDENCE — committed"]
        E1["results/*.json<br/>warrants, metrics, controls, provenance"]
        E2["results/scores/<br/>per-item labels and scores"]
        E3["results/RESULTS.md<br/>README claim table"]
        E4["DECISIONS.md<br/>append-only, including the reversals"]
    end

    subgraph GATE["GATES — what stops drift"]
        G1["tests/<br/>construction guards, negative tests,<br/>phase gates, document gates"]
        G2["make smoke · make test<br/>make verify · make extract"]
    end

    subgraph HIST["HISTORY"]
        H1["round1/<br/>the Round 1 submission, moved whole"]
        H2["docs/PATHS.md<br/>the move mapping, old path to new"]
    end

    CONTRACT -.->|"binding on"| CODE
    INPUT --> CODE --> EVID
    THIN -->|"call"| CODE
    EVID --> GATE
    GATE -->|"exit non-zero on drift"| EVID
    H1 -.->|"the question it could not answer"| CODE

    style CODE fill:#1f6feb18,stroke:#1f6feb
    style EVID fill:#2da44e18,stroke:#2da44e
    style GATE fill:#bf871118,stroke:#bf8711
```

Three things that diagram is saying, and they are the shape of the whole
project:

- **Logic only lives in the middle box.** Scripts, notebooks and the demo runner
  call into it. Logic that lives only in a script is unreviewable, and logic that
  lives only in a notebook cannot be run in CI.
- **Evidence and gates point at each other.** The artifacts are checked by the
  gates, and the gates exit non-zero rather than warning. That loop is why a
  number in this repository is checkable rather than asserted.
- **Round 1 is history, not a dependency.** It is preserved whole because the
  question it could not answer is the reason Round 2 exists.

---

**See also:** [ARCHITECTURE.md](ARCHITECTURE.md) for the prose version ·
[CODE_TOUR.md](CODE_TOUR.md) for the module behind each box ·
[GLOSSARY.md](GLOSSARY.md) for any term above that is doing more work than it
looks like.
