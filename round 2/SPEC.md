# SPEC.md — technical specification

Companion to `CLAUDE.md`. Reference snippets are illustrative; preserve the semantics exactly.

---

## 1. Core data model

Frozen dataclasses. Build these first; everything else consumes them.

### 1.1 Enums

```python
class Severity(IntEnum):        INFO=0; LOW=1; MEDIUM=2; HIGH=3; CRITICAL=4
class AccessTier(IntEnum):      T3_TEXT=3; T2_LOGPROBS=2; T1_ACTIVATIONS=1
class WarrantStatus(Enum):      VALID; STALE; REVOKED; REFUSED; UNVALIDATED
class MetricKind(Enum):         EXACT; ESTIMATED      # see CLAUDE.md, yield vs rate
class Reversibility(IntEnum):   READ_ONLY=0; REVERSIBLE_WRITE=1; IRREVERSIBLE_WRITE=2; EXTERNAL_COMM=3
```

`MetricKind` is not decorative. Every metric object carries it, and the renderer refuses to print an `ESTIMATED` value without an interval.

### 1.2 Finding

```python
@dataclass(frozen=True)
class Finding:
    finding_id: str
    detector_id: str            # "probe-qwen2.5-7b-L23", "presidio-stock", "qwen3guard-0.6b"
    detector_version: str       # semver + weights hash
    category: str               # PII | HALLUCINATION | INJECTION | UNSAFE | COST | BIAS_SIGNAL
    severity: Severity
    confidence: float           # raw detector score [0,1]
    evidence_spans: list[Span]  # character offsets — required, drives explainability
    access_tier: AccessTier
    latency_ms: float
    warrant_id: str | None      # None == UNWARRANTED, an honest state, not an error
```

Categories overlap by design. A fabricated detail about a person emits both `HALLUCINATION` and `PII`. Detectors never resolve conflicts; policy does.

### 1.3 Warrant

```python
@dataclass(frozen=True)
class Warrant:
    warrant_id: str
    detector_id: str
    detector_version: str
    operating_point_id: str
    eval_set_id: str            # content hash — this IS the envelope key
    validation_run_id: str
    issued_at: datetime
    expires_at: datetime
    metrics: WarrantMetrics     # each field: value, ci_low, ci_high, n, MetricKind
    n_test: int
    base_rate: float
    envelope: DistributionEnvelope
    controls: list[ControlResult]
    kappa: float | None         # inter-rater agreement, where labels are human
    status: WarrantStatus
    status_reason: str | None
```

**The key is `(detector_id, operating_point_id, eval_set_id)`.** Not detector alone. This is invariant 1 and the entire matrix depends on it.

`status` on a *record* is one of `VALID`, `STALE`, `REVOKED`, `REFUSED`. **`UNVALIDATED` is a property of a matrix cell, not of a warrant**, and a `Warrant` cannot be constructed carrying it: an unvalidated cell has no metrics, no envelope and no controls, so building a record for one would mean inventing all three. The matrix reports `UNVALIDATED` for a cell holding no warrant. See `DECISIONS.md` 024.

### 1.4 Certificate

```python
@dataclass(frozen=True)
class Certificate:
    certificate_id: str
    request_id: str
    session_id: str
    timestamp: datetime
    findings: list[Finding]
    resolution: Resolution              # action, policy_version, triggering_finding_ids
    warrants_relied_upon: list[str]
    weakest_warrant_status: WarrantStatus
    claimed_bounds: dict                # the bounded, falsifiable assertion
    envelope_match: EnvelopeMatchResult  # which envelope this input landed in
    access_tier_available: AccessTier
    prev_certificate_hash: str
    self_hash: str
```

`claimed_bounds` is what makes liability bounded: the system asserts *"checked at measured recall 0.14 [0.10, 0.19] on envelope E"*, not *"this is safe."*

### 1.5 Store

SQLite, append-only, hash-chained: `self_hash = SHA256(prev_hash || canonical_json(record))`.

Requirements from DPDP Rule 6: ≥1 year retention; queryable by session, time range, policy version, detector version, warrant status, and personal-data category accessed. Every record self-describing enough to interpret a year later without the current codebase.

Demonstrate tamper-evidence by mutating a row and showing the chain break. This is a two-line test and it is worth a slide.

---

## 2. Validation harness

### 2.1 The control suite

All five run on every validation. **A failed control fails the run and refuses the warrant.**

| Control | Method | Pass condition |
|---|---|---|
| **Padding fault injection** | Score one batch with correct left padding and with deliberately broken right padding | Correct: rel-L2 ≤ 0.1, cosine ≥ 0.999. Broken: **must be rejected** |
| **Label shuffle** | Retrain on permuted labels, score held-out. Repeated until the null is resolved | **Mean** AUROC ∈ [0.45, 0.55] **and** SE(mean) ≤ ¼ of the band width |
| **Null feature** | Replace activations with Gaussian noise matched in mean/variance. Same repeat rule | **Mean** AUROC ∈ [0.45, 0.55], same power condition |
| **Canary recall** | Score `canary-20` | Recall == 1.0 |
| **Determinism** | Re-run scoring twice at fixed seed | Bit-identical |

Label-shuffle and null-feature are **negative controls**: they assert the pipeline can produce a null result when there is no signal. A pipeline that cannot fail cannot be trusted when it succeeds. That sentence is the thesis applied to our own code and it belongs in the README.

Both are repeated rather than run once, and **each sizes its own repeat count from its own measured null spread**. A single draw is unusable: at a holdout of 600 the null spread is 0.068 for a 32-feature probe and 0.271 for a single-feature one, because a fitted probe's scores are not exchangeable and a one-dimensional probe's shuffle merely picks a sign. No closed form predicts that — the Hanley–McNeil SE understates it by 2.1× and 8.3× respectively — so the spread is measured and `k ≥ (2·spread / band_half)²` repeats are run, capped by `validation.null_control_max_repeats`. Measured counts: 8 repeats at 32 features, 22 at 4, 125 at 1.

A control that reaches the cap without resolving its null **fails as underpowered** rather than passing. Its job is to show the pipeline *can* produce a null result; without the power to show it, it has shown nothing. Derivation, the measured null distributions and the rejected alternatives are in `DECISIONS.md` 029 and 031.

### 2.2 `/validate`

```
POST /validate { detector_id, operating_point_id, eval_set_id }
→ ValidationRun {
    run_id, started_at, completed_at,
    metrics with 95% bootstrap CIs (1000 resamples),
    control_results: [5, each pass/fail with measured margin],
    envelope: reference distribution from this eval set,
    warrant: issued | refused,
    provenance: git commit, config hash, versions, device, seed
  }
```

Under one minute from cached activations. Streams progress. **Displays the deliberately-broken padding case being rejected** — that is the moment in the demo.

### 2.3 Refusal criteria

Refuse if any control fails; or AUROC lower CI bound ≤ 0.55; or recall below the policy's declared minimum at the operating point; or FPR on the hard-negative set above the declared maximum; or `n_test` < 200.

Recall and hard-negative FPR are compared at the **interval bound**, not the point estimate — a profile declaring "at least 10% recall" is asking for a guarantee, and a point estimate of 0.11 whose lower bound is 0.06 does not supply one. `n_test` counts the **test split**, so an eval set is sized by that split rather than by its total; see `DECISIONS.md` 030.

**No override path exists.** Not a flag, not an env var.

---

## 3. The warrant matrix

### 3.1 Why a matrix and not a curve

An envelope violation is a property of the input distribution, not of a detector. Long-context traffic invalidates the score-to-error-rate mapping for T1, T2 and T3 *simultaneously*. "Downgrade to the next tier" without re-checking the envelope replaces one unwarranted claim with another.

So build:

|  | `triviaqa-600` | `triviaqa-longctx-600` | `hinglish-pii-200` | `hard-negatives-200` |
|---|---|---|---|---|
| T1 probe, mean-pool | VALID | **REFUSED** | UNVALIDATED | VALID |
| T1 probe, max-rolling-means | VALID | VALID (wider) | UNVALIDATED | VALID |
| T2 logprob | VALID | ? | ? | ? |
| T3 judge | VALID | ? | ? | ? |
| presidio-stock | n/a | n/a | **REFUSED** | VALID |

Cheap to build: Phase 2's ablation extracts all tiers in one pass; run it once per eval set.

### 3.2 Routing

Drift fires → *"which detector holds a `VALID` warrant on the envelope I am actually in?"* → route there, adopt **that warrant's bounds**, and if the row is empty, refuse everything and fall back to the profile's most conservative action.

```python
def route(envelope_id: str, profile: Profile) -> RoutingDecision:
    candidates = matrix.valid_warrants(envelope_id)
    eligible  = [w for w in candidates if profile.accepts(w.metrics)]
    if eligible:
        return RoutingDecision(warrant=best(eligible), claimed_bounds=...)
    unvalidated = matrix.unvalidated_cells(envelope_id)
    enqueue_for_validation(unvalidated)          # matrix self-populates from live traffic
    return RoutingDecision(warrant=None, action=profile.conservative_default,
                           reason=f"no valid warrant on envelope {envelope_id}")
```

### 3.3 The three states, precisely

| State | Meaning | Behaviour |
|---|---|---|
| `VALID` | Validated here, cleared the bar | Normal operation, full claimed bounds |
| `REFUSED` | Validated here, failed the bar | Detector removed from service on this envelope until re-validated by a human |
| `UNVALIDATED` | Never tested here | **Modal state in production.** No claim available → conservative action, enqueue for validation, log |

Collapsing `UNVALIDATED` into `REFUSED` makes the system unusable on day one. Collapsing it into `VALID` is the failure the whole product argues against.

These are **cell** states. `VALID`, `STALE`, `REVOKED` and `REFUSED` are carried on the warrant record occupying the cell; `UNVALIDATED` is the state of a cell with no record in it. Representing it as a record would require fabricating the metrics that make a record a claim, which is the failure mode the state exists to prevent.

---

## 4. Evaluation sets

Content-hashed, immutable.

| Set | Contents | Validates |
|---|---|---|
| `triviaqa-600` | Existing held-out set | Round 1 anchor |
| `triviaqa-longctx-600` | Same questions padded with distractors to 4–16k tokens | The drift trigger; mean-pool collapse |
| `hinglish-pii-200` | Hand-built code-switched Hindi-English with Aadhaar/PAN/UPI/phone in verbatim, spaced, and obfuscated forms | India differentiator; Presidio refusal |
| `hard-negatives-200` | Boundary cases that **must be allowed**: security analyst summarising malware, clinician discussing overdose thresholds, HR describing a discrimination complaint, customer quoting abuse in a grievance | FPR — the number a skeptic accepts |
| `canary-20` | Known positives the detector must always catch | Regression tripwire |
| `agent-actions-N` | Tool-call traces labelled by reversibility class and safety | Phase 9 |

Hand-build the middle three. Do not generate them with an LLM and call them ground truth.

`hard-negatives-200` is the most under-appreciated item here. FPR measured on easy benign traffic makes a damaging filter look excellent.

---

## 5. Drift and revocation

### 5.1 Envelope features

Computed at validation time, stored **inside the warrant**.

1. **Input token length** — the highest-priority feature; long-context is the documented probe killer
2. **Script/language mix** — Latin / Devanagari / mixed; code-switching density
3. **Embedding centroid distance** — mean cosine from the eval set centroid
4. **Reference perplexity** — small-model perplexity as a distributional proxy
5. **Category mix** — proportion of requests triggering each finding category

### 5.2 Detection

- **Per-feature:** Population Stability Index over a sliding window. `<0.10` stable, `0.10–0.25` moderate, `>0.25` significant. PSI is native vocabulary in Indian banking risk teams.
- **Multivariate:** MMD on embeddings with a permutation test.
- **Window:** sliding, minimum 200 requests before any verdict. Do not revoke on noise.

### 5.3 Revocation ladder

| Envelope state | Status | Behaviour |
|---|---|---|
| Inside | `VALID` | Normal, full bounds |
| PSI 0.10–0.25 | `STALE` | Widen reported CIs, trigger background revalidation, log |
| PSI > 0.25 | `REVOKED` | **Refuse to certify.** Consult the matrix for a valid warrant on the new envelope. Route or refuse |
| Control failure | `REFUSED` | Out of service until human re-validation |

### 5.4 Model-version invalidation

A probe is pinned to a model version. On model change, every activation-tier warrant is invalidated until revalidation. State this as an operational cost; the mitigation is that T2 and T3 are model-agnostic and keep working through the gap.

---

## 6. On-traffic warrant — stratified estimation

**This is what makes the product more than a staleness detector.** Escalated items are already reviewed. Those reviews are labels we currently discard.

### 6.1 The two strata

- **Stratum A — flagged.** 100% reviewed already. Free.
- **Stratum B — random sample of unflagged traffic.** The expensive half.

Importance-weight the two for an unbiased on-traffic estimate.

### 6.2 Sizing — get this right, it is the error the spec exists to prevent

Let `N` = monthly volume, `f` = flag rate, `N_u = N(1−f)` unflagged, `TP` = confirmed true positives in stratum A, `q` = prevalence of errors in the unflagged pool.

```
R = TP / (TP + q·N_u)
∂R/∂q = −TP·N_u / (TP + q·N_u)²
```

To hit a recall margin `m_R`, size the *prevalence* estimate to `m_q = m_R / |∂R/∂q|`, then:

```
n = 1.96² · q(1−q) / m_q²
```

**Worked, at the declared workload** (N=200,000, base error rate 0.03, f=0.0148, TP=850, N_u=197,036, q=0.0261):

```
∂R/∂q = −850 × 197,036 / 6,000²  =  −4.65
m_R = 0.05  →  m_q = 0.01075  →  n ≈  847
m_R = 0.02  →  m_q = 0.00430  →  n ≈ 5,292
```

**The trap this encodes.** `1.96²·(0.05)(0.95)/0.02² = 456` sizes a *prevalence* estimate to ±2pp. Calling that "recall to ±2pp" understates the labels needed by roughly 22×, because sample size scales as `1/margin²` and the derivative is 4.65. The amplification is structural: the unflagged pool is ~232× the flagged one, so any error in `q` is multiplied by that ratio on its way into `R`.

**Assert this in code.** `sizing.py` must expose the derivative and a test must confirm that sizing for recall differs from sizing for prevalence by `(∂R/∂q)²`.

### 6.3 Allocation

Neyman allocation across probe-score bands concentrates labels where variance is:

```
n_h ∝ W_h · √(q_h(1−q_h))
```

**Expect ~1.5× reduction, not more.** Neyman variance goes as `[Σ W_h √(q_h(1−q_h))]²`, and √ is concave, so a large band with small-but-nonzero prevalence dominates the sum precisely *because* it is large. On plausible band structure for a detector at AUROC 0.855 the design effect lands near 0.67. Treat 2× as a good outcome. **Size it empirically from the observed score distribution; do not assume a factor.**

**Month one runs proportional allocation** because Neyman needs `q_h` you do not have yet. The first warrant costs full SRS price. Say this in the README rather than letting someone discover it.

### 6.4 The price list — computed, never typed

`src/economics/sizing.py` derives all of this from the single declared workload in `config.yaml`. The README pulls the rendered table. **No figure here is written by hand**, which is also how invariant 6 is enforced structurally rather than by discipline.

| Claim | Extra labels | Cost | Why |
|---|---|---|---|
| Precision | 0 | **free** | Stratum A already reviewed |
| FPR | 0 | **free** | FP known exactly, denominator known |
| **Yield** — *"we surfaced N real errors"* | 0 | **free** | Exact count, `MetricKind.EXACT` |
| Recall ±5pp | ~847 SRS / ~570 Neyman | computed | Requires estimating `q` over the unflagged pool |
| Recall ±2pp | ~5,292 SRS / ~3,500 Neyman | computed | Same, at 6× precision |

Three of the four are free. The pitch line falls out:

> *"Three of the four numbers you need are free — you're already reviewing the flags, you're just throwing the labels away. Only recall costs money, because recall means estimating what you missed across two hundred thousand responses nobody looked at. Here's what each precision level costs. Pick one."*

### 6.5 Label quality

**Blinding.** Stratum A reviewers see items the system flagged; stratum B reviewers see items it didn't. If flag status, score, stratum or ordering is visible, the two strata get systematically different label distributions and the importance-weighted estimate is biased *in the direction that flatters us*. Interleave both strata into **one blinded queue with identical UI**. Nearly free, and it is the difference between a warrant and a self-report.

**Noise.** Published work measures human agreement near 48% on contextual redaction, and label noise materially degrades measured detector performance. A ±5pp interval is meaningless at κ = 0.5. **Double-label ~10% of both strata; publish Cohen's κ alongside every warrant.** At the ±5pp tier that is ~85 extra reviews.

A warrant carrying its own label-agreement statistic is doing to itself what it does to detectors — refusing to claim more precision than its evidence supports.

---

## 7. Policy engine

### 7.1 Policy is data

OPA/Rego or Cedar. **Do not write a DSL.** Versioned, content-hashed, hot-reloadable bundles. Every certificate stamps the policy version.

### 7.2 Load-time warrant resolution

On bundle load, resolve every referenced operating point against the matrix. **Missing, expired, refused, unvalidated, or below declared minimums → the bundle fails to load**, with an error naming the missing warrant. Not a warning.

```yaml
profile: customer_support
version: 3.1
requires_warrant:
  - detector: probe-qwen2.5-7b-L23
    operating_point: P-conservative
    envelope: triviaqa-600
    min_recall: 0.10
    max_fpr_hard_negatives: 0.02
    max_age: 24h
rules:
  - when: warrant.weakest_status != VALID and action.reversibility >= IRREVERSIBLE_WRITE
    then: BLOCK
    reason: "no valid warrant backs an irreversible action"
  - when: finding.category == PII and finding.severity >= HIGH
    then: REDACT
  - when: severity >= HIGH and confidence_band == UNCERTAIN
    then: ESCALATE
```

The first rule is the entire product in four lines.

### 7.3 Profiles

Three, each at a **different validated operating point on the same measured curve** — not three invented thresholds.

| Profile | Inline budget | Operating point | Rationale |
|---|---|---|---|
| Customer support | ≤200ms | High precision, low flag rate | Over-blocking destroys the channel; volume high |
| Internal knowledge | ≤500ms | Balanced, retrieval verification on | Corpus exists; internal users tolerate friction |
| Decision support | ≤2s | High recall, escalation-heavy | Low volume, high consequence, review affordable |

### 7.4 Weighted-error objective

```
τ* = argmin_τ  Σ wᵢ·ERRORᵢ(τ) / Σ wᵢ
  w_fpr_benign = 50 ; w_fnr = 5 ; w_fpr_hard_negative = 2
```

Weights live in the bundle, are versioned, and appear on screen. Complexity 4 answered by declaring the tradeoff, not solving it.

---

## 8. Detector adapters

| Detector | Licence | Role |
|---|---|---|
| Probe (T1) | ours | Correctness trigger; tier-ladder anchor |
| Qwen3Guard-0.6B / Stream | Apache-2.0 | Content safety; its safe/controversial/unsafe output already models deferral |
| LettuceDetect / TinyLettuce | MIT | RAG grounding for the knowledge profile |
| Presidio | MIT | PII, and the warrant-refusal demo |
| LLM judge | — | T3 tier, cascade fallback |

### 8.1 The Presidio sequence

Presidio ships `IN_AADHAAR`, `IN_PAN`, `IN_PASSPORT`, `IN_VEHICLE_REGISTRATION`, `IN_VOTER`, `IN_GSTIN` — **all English-only and disabled by default.**

Measure three configurations on `hinglish-pii-200`, all three reported:

1. **Stock** — recognisers off. Expect near-zero recall on Indian identifiers → **warrant REFUSED** automatically.
2. **Enabled** — recognisers on. Better; still imperfect on obfuscated and code-switched forms.
3. **Enabled + custom** — checksum-validated Aadhaar (Verhoeff), PAN structural, UPI VPA, IFSC.

**Verify what the shipped `InAadhaarRecognizer` actually validates before claiming novelty on the checksum.** If it already does Verhoeff, the contribution is the measurement and the Hinglish evaluation, not the algorithm. Claim the right thing.

When citing the published Presidio HIGH-sensitivity recall of 0.07, state that the benchmark is synthetic and deliberately includes obfuscated and non-verbatim disclosure forms. Describing its construction is what stops it being turned against you.

---

## 9. Action gate

### 9.1 Reversibility

Registered per tool at definition time — a **static property of the tool**, not an inference about the request. That is what makes it robust to an attacker who has defeated every detector.

### 9.2 Session Rule-of-Two

Three sticky booleans per session: `has_ingested_untrusted_input`, `has_sensitive_access`, `can_change_state_or_communicate`. **Two of three is the limit.** Sticky for the session — once untrusted content has entered, it has entered. This is complexity 5 answered.

### 9.3 The gate

```
BLOCK    if reversibility >= IRREVERSIBLE_WRITE and warrant.weakest_status != VALID
BLOCK    if rule_of_two_violated(session) and reversibility >= IRREVERSIBLE_WRITE
CONFIRM  if reversibility >= IRREVERSIBLE_WRITE and severity >= MEDIUM
ESCALATE if severity >= HIGH and confidence_band == UNCERTAIN
ALLOW    otherwise
```

The first two rules consult no detector score. Deliberate: published work broke twelve injection defences at >90% attack success, so any control depending on detection has a measured ceiling. A control keyed to static tool class plus session state holds against an attacker who has completely defeated the probe.

### 9.4 The agent

One domain, four tools, clean reversibility gradient. Banking: `get_balance` (READ_ONLY), `search_transactions` (READ_ONLY), `draft_dispute` (REVERSIBLE_WRITE), `transfer_funds` (IRREVERSIBLE_WRITE). Gate at **tool-call time**, not text-generation time.

---

## 10. Tests

| Test | Asserts |
|---|---|
| `test_hash_chain` | Mutating a row breaks the chain |
| `test_warrant_key` | Warrants keyed by (detector, op, eval_set); same detector on two envelopes yields two warrants |
| `test_three_states` | `UNVALIDATED` routes conservatively and enqueues; `REFUSED` removes from service; they are not interchangeable |
| `test_no_override` | No code path issues a warrant with a failed control. Grep-level assertion plus behavioural test |
| `test_controls` | All five controls; each fails when it should |
| `test_padding_equivalence` | Batched == unbatched to 1e-2; right-padded batch is rejected |
| `test_sizing_derivative` | `∂R/∂q` matches analytic form; **sizing for recall differs from sizing for prevalence by `(∂R/∂q)²`** |
| `test_sizing_units` | Sizing function refuses to return a number without a declared target quantity |
| `test_no_scenario_mixing` | All economics figures derive from one workload block; test fails if two are mixed |
| `test_yield_vs_rate` | `MetricKind.EXACT` values carry no interval; `ESTIMATED` values always do |
| `test_blinding` | Label queue payload contains no flag status, score, stratum, or ordering signal |
| `test_stratified_unbiased` | On synthetic data with known ground truth, the estimator recovers true recall within CI at nominal coverage over 1000 trials |
| `test_policy_refusal` | Bundle referencing an unwarranted operating point fails to load |
| `test_gate_no_detector` | Gate's first two rules produce correct decisions with all detector scores nulled |
| `test_polarity` | Positive class is "incorrect" |
| `test_no_test_leakage` | Scaler and classifier fit on train indices only |
| `test_determinism` | Two runs at one seed, identical coefficients |
| `test_smoke` | Full pipeline at n=100 completes, all artifacts written |

`test_stratified_unbiased` is the most important new test. An estimator that is subtly biased produces confident wrong intervals, and nothing raises.

---

## 11. What `results/RESULTS.md` must contain

1. Run metadata — models, quantisation, device, seed, config hash, git commit, timestamp
2. Eval set registry — IDs, hashes, sizes, base rates
3. **The warrant matrix** — every (detector × envelope) cell with status and metrics
4. Tier ladder on each envelope, with CIs
5. Control results for every run, including the rejected fault case
6. Drift trace — envelope violation detected, revocation, routing decision
7. **The computed price list**, with the declared workload stated above it
8. Stratified estimator validation — coverage over synthetic trials
9. Cohen's κ
10. Presidio three-configuration comparison
11. Action-gate results, including the detector-nulled case
12. Limitations — written honestly and specifically, not boilerplate
