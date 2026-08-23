# DECISIONS.md

Append-only log of methodological choices. Newest at the bottom. **Never edit or delete** — supersede by number.

The answer sheet for "why did you do it that way?" Every entry should be readable aloud to a technical reviewer. **Statistical decisions carry their derivation in the entry, not a pointer to it.**

**Format:** decision · date · status · context · decision · alternatives rejected and why · consequences and what a reviewer could fairly object to.

---

## 001 — Warrants are keyed by (detector, operating_point, eval_set)
**Status:** accepted

**Context:** An earlier design keyed warrants by detector alone and had drift "downgrade to the next tier."

**Decision:** The eval set is the envelope key and is part of the warrant's identity.

**Alternatives:** *Key by detector alone* — simpler, and wrong. An envelope violation is a property of the input distribution, so long-context traffic invalidates T1, T2 and T3 simultaneously. Downgrading would replace one unwarranted claim with another.

**Consequences:** Warrants become a matrix rather than a list, and validation runs once per (detector × envelope). Cheap, since the ablation already extracts all tiers in one pass. A reviewer can fairly ask why we don't have every cell filled — the answer is `UNVALIDATED`, see 002.

---

## 002 — `UNVALIDATED` is a first-class state distinct from `REFUSED`
**Status:** accepted

**Context:** Real traffic constantly lands in envelopes never validated on.

**Decision:** Three states. `VALID` (tested, cleared), `REFUSED` (tested, failed), `UNVALIDATED` (never tested here). `UNVALIDATED` routes to the profile's conservative default and enqueues that cell for validation.

**Alternatives:** *Collapse into `REFUSED`* — makes the system unusable on day one, since the modal production state is "never tested here." *Collapse into `VALID`* — precisely the failure the product argues against.

**Consequences:** The matrix self-populates from live traffic, which is a nice closed loop. Costs a third code path everywhere status is consumed.

---

## 003 — Refusal has no override path
**Status:** accepted

**Decision:** No flag, env var, or admin bypass can issue a warrant when a control failed or a threshold wasn't met.

**Alternatives:** *An override for operational emergencies* — the realistic engineering choice, and it would make the entire product theatre. A reviewer will look for exactly this.

**Consequences:** Operationally rigid by design. The escape valve is the matrix: route to a detector that does hold a valid warrant, or degrade to the conservative default.

---

## 004 — Yield is exact; rate is estimated; they never mix
**Status:** accepted

**Context:** "We surfaced 850 real errors" and "we caught 14% of errors" look like the same kind of claim and are not.

**Decision:** Every metric carries `MetricKind.EXACT` or `ESTIMATED`. Exact values are counts of reviewed confirmed items and carry no interval. Estimated values always carry one. The renderer refuses to print an estimated value without an interval.

**Consequences:** Yield becomes a free claim and the basis of the price list. A reviewer could object that yield alone is uninformative without a denominator — correct, and that is exactly what recall costs money to buy.

---

## 005 — Tuned for recall at the trigger tier; precision reported, not optimised
**Status:** accepted

**Decision:** At the trigger operating point, maximise recall at a fixed flag-rate budget. Never report a blended F1.

**Consequences:** Most flagged responses are fine, by design: a false positive costs one review, a false negative costs a user acting on a wrong answer. **Depends entirely on the probe never having blocking authority.** If that changes, revisit.

---

## 006 — Test set opened once per validation run
**Status:** accepted

**Decision:** Layer, regularisation and threshold chosen on validation. Test scored once per run, and every run is published.

**Consequences:** Intervals are wide at n≈600; addressed with bootstrap rather than by borrowing from validation.

---

## 007 — Hand-built evaluation sets for Hinglish and hard negatives
**Status:** accepted

**Decision:** `hinglish-pii-200` and `hard-negatives-200` are hand-constructed.

**Alternatives:** *LLM-generated* — faster, and it would make a model's judgment the ground truth for measuring models. Circular.

**Consequences:** Small sets, wide intervals, and a labelling process we can describe. `hard-negatives-200` is the source of the FPR number; FPR measured on easy benign traffic makes a damaging filter look excellent.

---

## 008 — Presidio evaluated at stock configuration, and at two more
**Status:** accepted

**Context:** Presidio ships Indian recognisers English-only and disabled by default.

**Decision:** Measure and report three configurations: stock, recognisers enabled, enabled plus custom checksum-validated. Stock is a fair test because stock is what teams ship.

**Alternatives:** *Only stock* — the strongest single result, and open to "you crippled it." *Only best-configured* — loses the finding entirely.

**Consequences:** The point survives at all three: even best-configured there is a residual, and only we report the number. Must verify what `InAadhaarRecognizer` actually validates before claiming novelty on the checksum.

---

## 009 — Reported detector benchmarks are labelled by construction
**Status:** accepted

**Decision:** When citing published detector numbers, state the benchmark's construction — synthetic, injected-versus-organic, vendor-run.

**Context:** A synthetic benchmark deliberately containing obfuscated forms produces a low regex recall that is real and important, and describing its construction is what stops it being turned against us.

---

## 010 — The action gate's first two rules consult no detector score
**Status:** accepted

**Decision:** Irreversible actions require a valid warrant and a session that hasn't violated Rule-of-Two. Neither condition reads a detector output.

**Context:** Published work broke twelve injection defences at >90% attack success. Any control depending on detection has a measured ceiling.

**Consequences:** We claim no adversarial robustness anywhere. The gate holds against an attacker who has completely defeated the probe, because it doesn't depend on it.

---

## 011 — Session-level Rule-of-Two flags are sticky
**Status:** accepted

**Decision:** Once untrusted input has entered a session, the flag stays set for that session.

**Consequences:** Long sessions accumulate constraint and eventually can't take irreversible actions without confirmation. That is the intended behaviour and is how compounding multi-turn risk is handled. A reviewer could object that this degrades usability in long sessions — true, and it is the tradeoff being made deliberately.

---

## 012 — Policy is versioned data; bundles fail to load without warrants
**Status:** accepted

**Decision:** OPA/Rego or Cedar. A bundle referencing an operating point with no valid warrant **fails to load**, with an error naming the missing warrant.

**Alternatives:** *Warn and continue* — the ordinary engineering choice, and it silently reintroduces exactly the unbacked-claim problem the product exists to solve.

**Consequences:** Deploying a new profile requires validating it first. That is the point.

---

## 013 — All economics derive from one declared workload
**Status:** accepted

**Context:** An earlier draft quoted precision at n=10,000 (from a 5%-flag-rate scenario) alongside recall sizing at ~850 labels (from a 1.48%-flag-rate scenario). Mixing scenarios in one table is silently wrong.

**Decision:** A single workload block in `config.yaml`. `src/economics/sizing.py` derives every figure from it. `test_no_scenario_mixing` fails the build if two are mixed. No economic figure is typed by hand anywhere.

**Consequences:** Changing the assumed workload changes every downstream number consistently, which is the correct behaviour and makes the sensitivity visible.

---

## 014 — Recall intervals are sized through ∂R/∂q, not in recall units
**Status:** accepted

**Context:** An earlier estimate claimed ~460 labels bought recall to ±2pp. That figure sizes a *prevalence* estimate, not a recall estimate.

**Decision:** Size the prevalence estimate, then propagate.

```
R      = TP / (TP + q·N_u)
∂R/∂q  = −TP·N_u / (TP + q·N_u)²
m_q    = m_R / |∂R/∂q|
n      = 1.96² · q(1−q) / m_q²
```

At the declared workload (N=200,000, base error rate 0.03, f=0.0148, TP=850, N_u=197,036, q=0.0261):

```
∂R/∂q = −850 × 197,036 / 6,000² = −4.65
m_R = 0.05 → m_q = 0.01075 → n ≈   847
m_R = 0.02 → m_q = 0.00430 → n ≈ 5,292
```

**Why the error was large:** sample size scales as `1/m²`, so a factor of 4.65 in the margin is a factor of ~22 in `n`. The amplification is structural — the unflagged pool is ~232× the flagged one, so any error in `q` is multiplied by that ratio on its way into `R`.

**Consequences:** The honest price is ~847 labels for ±5pp, not ~460 for ±2pp. `test_sizing_derivative` asserts the relationship; `test_sizing_units` refuses to return a sample size without a declared target quantity.

---

## 015 — Neyman allocation expected at ~1.5×, with month one proportional
**Status:** accepted

**Context:** Score-band stratification concentrates labels where variance is, but the gain is bounded.

**Decision:** Expect a design effect near 0.67 — about 1.5×. Treat 2× as a good outcome. Measure empirically from the observed score distribution rather than assuming.

**Why it caps out:** Neyman variance goes as `[Σ W_h √(q_h(1−q_h))]²`, and √ is concave. A large band with small-but-nonzero prevalence dominates the sum *because* it is large. You cannot allocate your way out of a huge pool with low-but-nonzero prevalence.

**Month one runs proportional allocation**, because Neyman needs per-band `q_h` we don't have yet. The first warrant costs full SRS price. Stated in the README rather than discovered.

---

## 016 — Label queue is blinded and interleaved across strata
**Status:** accepted

**Context:** Stratum A reviewers see flagged items; stratum B reviewers see unflagged ones. Visible flag status, score, stratum or ordering gives the two strata systematically different label distributions.

**Decision:** One queue, identical UI, no flag status, no score, no stratum marker, no ordering signal. `test_blinding` asserts the payload carries none of them.

**Consequences:** Nearly free, and it is the difference between a warrant and a self-report. The bias it prevents runs in the direction that flatters us, which is the direction nobody catches.

---

## 017 — Cohen's κ published alongside every human-labelled warrant
**Status:** accepted

**Context:** Published work measures human agreement near 48% on contextual redaction. A ±5pp interval is meaningless at κ = 0.5.

**Decision:** Double-label ~10% of both strata; report κ on the warrant. At the ±5pp tier that is roughly 85 extra reviews.

**Consequences:** A warrant that carries its own label-agreement statistic refuses to claim more precision than its evidence supports — the thesis applied one level up.

---

## 018 — Lift is never quoted without precision
**Status:** accepted

**Context:** Lift `R/f` and precision are the same fact in different clothes. At a 3% base rate the projected lift is ~9.5× and precision ~0.285 — roughly seven in ten flags are false alarms.

**Decision:** Both, in one breath, with the comparison: *"about seven in ten flags would be false alarms, which is still nine times better than random sampling, where ninety-seven in a hundred would be."*

**Consequences:** Quoting the flattering half of a real number is the exact failure pattern that cost us in Round 1. Paired, the claim is unattackable.

---

## 019 — Cross-workload projections are labelled as sizing estimates, not claims
**Status:** accepted

**Context:** Projecting the measured ROC to a 3% base-rate workload crosses a distributional boundary — which is precisely what the envelope check exists to refuse.

**Decision:** State the projection, then state that our own envelope check would refuse to warrant it without revalidation on that traffic. It is a sizing estimate; the claim is what comes back from `/validate` on the customer's data.

**Consequences:** Turning our own caveat machinery on our own favourite number is the most credible move available in a Q&A, and it costs one sentence.

---

## 020 — Invariants that can be expressed as config assertions are enforced at load time
**Status:** accepted

**Context:** Several `CLAUDE.md` invariants are one config edit away from being violated silently. `probe.positive_class` flipped to `"correct"` yields `1 − AUROC`, which reads as a strong negative result rather than as a bug. `sampling.blind_queue: false` biases the stratified estimate in the direction that flatters us. `policy.fail_closed_on_missing_warrant: false` turns the product into theatre. None of these raises anything at the point of use.

**Decision:** Where an invariant is a property of the configuration, `src/config.py` asserts it in the relevant dataclass's `__post_init__`, and the error message names the invariant and the failure it prevents. Currently pinned: `probe.positive_class == "incorrect"`, `probe.standardize`, `sampling.blind_queue`, `sampling.allocation_month_one == "proportional"`, `policy.fail_closed_on_missing_warrant`, `store.hash_chain`, `store.retention_days >= 365`, the presence of all five controls in `validation.controls`, a `null_control_band` straddling 0.5, `psi_stable < psi_significant`, `token_length` among the drift features, and a licence denylist over every configured model string.

**Alternatives:** *Assert at the point of use* — each check then lives in the module that could be harmed by it, which is later, more scattered, and silent for any code path not yet written. *Document and rely on review* — the failure modes listed above are exactly the ones review does not catch, because the resulting numbers look reasonable.

**Consequences:** These knobs are no longer knobs; the config file documents them as fixed and the loader enforces it. Changing one requires editing code and reading why it was pinned, which is the intended friction. A reviewer could fairly object that a config that refuses most of its own values is really a constant table — partly true, and the answer is that the values are still declared in one readable place with their reasons attached.

---

## 021 — Round 2 is built under `round 2/`, reusing Round 1 only through published numbers
**Status:** accepted

**Context:** Round 1 is a shipped, tagged result occupying the repository root — its own `CLAUDE.md`, `SPEC.md`, `config.yaml`, `src/` and `results/`. Round 2's contracts describe an identically-named layout.

**Decision:** Round 2 lives in `round 2/` in the same repository. Round 1 is not modified, and Round 2 depends on it only through two published numbers — the measured operating point (`tpr 0.1416`, `fpr 0.0110`), transcribed into `round 2/config.yaml` as declared inputs to the workload block.

**Alternatives:** *Overlay Round 2 on the root* — destroys the Round 1 result and its history. *A separate repository* — loses the ability to trace the Round 2 operating point back to the run that measured it, which is the property this project exists to defend.

**Consequences:** `provenance()` distinguishes the project root (holding `config.yaml`) from the repository root (holding `.git`); git commands run against the latter, artifact paths resolve against the former. The dirty flag covers the whole repository, so an uncommitted Round 1 edit correctly marks a Round 2 artifact dirty — the flag describes the tree the recorded commit came from.

**A reviewer could fairly object** that the two carried-forward numbers are hand-transcribed, which is exactly the practice this project argues against. They are labelled as declared inputs rather than results, they are never re-derived, and every figure computed from them is computed by code. Round 1's own `results/` is the trace.

---

## 022 — "Free" and "exact" are different axes; precision is free but estimated
**Status:** accepted

**Context:** The price list (`SPEC.md` §6.4) marks precision, FPR and yield as **free** — zero extra labels, because stratum A is reviewed already. Invariant 4 says every *rate* carries a 95% interval. Read carelessly, those two sentences contradict: if precision is free, why does it have an interval?

**Decision:** They are separate axes and the type system encodes only one of them.

* `MetricKind.EXACT` means *a count of reviewed, confirmed items* — no inference about anything unreviewed. Only yield-type quantities qualify: confirmed errors surfaced, false positives seen. These carry no interval.
* `MetricKind.ESTIMATED` covers every rate, including precision and FPR, and always carries an interval and an `n`.
* **Free** is a statement about *label cost*, and it lives in the price list, not in `MetricKind`.

So precision is free *and* estimated. The 850 confirmed errors are a fact about the month that happened; the precision figure is a claim about the process, and the flagged pool is a finite sample of the traffic distribution. A reviewer asking "what will precision be next month?" is asking for an estimate, and that is the question a warrant answers.

**Alternatives rejected:** *Tag precision `EXACT` because it costs nothing* — conflates the two axes and would let a point estimate reach a user with no interval, violating invariant 4 for the sake of a word in a table. *Give exact counts intervals too, for consistency* — destroys the free claim, which is the basis of the whole price list, and is the exact error `CLAUDE.md` names as the most damaging available here.

**Consequences:** `WarrantMetrics` requires `confirmed_errors` to be `EXACT` with unit `count`, and `auroc`/`recall`/`precision`/`flag_rate` to be `ESTIMATED`. `Metric.__post_init__` refuses an `EXACT` metric carrying an interval and an `ESTIMATED` one without. `test_yield_vs_rate` asserts both directions.

**A reviewer could fairly object** that this month's precision *is* an exact fact about this month. Correct — and that fact is reported as the yield pair (confirmed errors, false positives seen), both `EXACT`. The ratio is labelled estimated because it is only interesting as a forward claim.

---

## 023 — Invariant 5 is enforced by refusing the name, not by convention
**Status:** accepted

**Context:** "Never a blended F1 anywhere in the codebase" is the kind of rule that holds until someone adds a convenience property, or a report label, or a column header.

**Decision:** `Metric.__post_init__` rejects any name matching `f1`, `f_1`, `fbeta`, `f2`, `fscore`, `f_measure` and their variants, with an error stating why. Since every reported number is a `Metric`, a blended score cannot reach a document without deleting the check.

**Alternatives rejected:** *A grep in the test suite* — catches the literal string in source, not a label built at runtime from config or from a detector's own output. Kept as well, but not as the only guard.

**Consequences:** `WarrantMetrics` also requires precision and recall as separate mandatory fields, so a warrant claiming one without the other is unconstructible. A reviewer could fairly object that a project could legitimately want an F-score for an internal comparison — it could, and the answer is that the exchange rate between a wasted review and a user acting on a wrong answer is the decision this product exists to expose, so hiding it inside a single number is not a tradeoff we are willing to make silently.

---

## 024 — `UNVALIDATED` is a matrix-cell state, never a warrant record's status
**Status:** accepted · **refines** 002, and `SPEC.md` §1.3 and §3.3 were updated in the same commit

**Context:** `WarrantStatus` lists five members and `Warrant.status` is typed as one of them. Implementing the record made a contradiction visible: a warrant carries `metrics`, `envelope`, `controls`, `n_test`, `base_rate` and `kappa`, and an `UNVALIDATED` cell has none of those, because nobody ever ran the validation. Constructing such a record means inventing the numbers that make a record a claim — which is the precise failure the state exists to prevent.

**Decision:** `UNVALIDATED` stays in the enum, because the matrix and the routing code need to name it. `Warrant.__post_init__` refuses to construct a record carrying it. A cell is `UNVALIDATED` **because it holds no record**, and `matrix.status(key)` returns `UNVALIDATED` for a missing cell rather than reading it off an object.

**Alternatives rejected:** *Allow a record with `None` metrics* — every consumer then has to check for `None` before reading bounds, and the one that forgets reads `None` as a number or crashes at render time in front of a judge. The absence of a record cannot be dereferenced by accident. *Drop `UNVALIDATED` from the enum and use `Optional[Warrant]` everywhere* — loses the name, and the name is what keeps the three states distinct in the routing code (invariant 2).

**Consequences:** Code asking "what do we know here?" calls the matrix, never a warrant. That is the correct direction anyway: the question is about a cell, and only the matrix knows which cells exist. `test_three_states` asserts that a missing cell reports `UNVALIDATED` and routes conservatively while a `REFUSED` cell removes the detector from service, and that the two are not interchangeable.

**A reviewer could fairly object** that this makes the enum's five members mean two different kinds of thing. True — and the spec now says so explicitly in §1.3 and §3.3 rather than leaving it to be discovered from the code.

---

<!-- New entries below. Do not edit anything above this line. -->
