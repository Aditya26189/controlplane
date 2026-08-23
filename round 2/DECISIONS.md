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

## 025 — The store claims tamper-evidence, not tamper-proofing, and the limit is tested
**Status:** accepted

**Context:** `SPEC.md` §1.5 asks for an append-only hash-chained store and says to demonstrate tamper-evidence by mutating a row and showing the chain break. Building it surfaced three attacks with three different outcomes, and only two of them are caught.

**Decision:** Verification checks three things, and the limit of the third is stated rather than glossed.

1. **Edit a body.** That row's `self_hash` no longer matches `SHA256(prev_hash ‖ body)`. Caught at that row.
2. **Edit a body and recompute its hash.** The body check now passes, but the next row's `prev_hash` still records the old value. Caught at the following row. The two checks cover each other, which is why both exist — either alone leaves a hole.
3. **Delete rows from the head.** The remainder re-anchors and would verify cleanly, so `verify_chain` also checks the anchor: a chain not starting at genesis must be explained by a surviving `retention_event` naming the hash it removed up to. An undeclared truncation is reported as a break.

**The limit:** an attacker who deletes *every* row also deletes the retention event that would have declared it, and an empty ledger verifies. No log that is its own only witness can do better — detecting total erasure requires an anchor outside the file: a published head hash, a replicated store, or a notary. **We have not built one.** So the claim is tamper-*evidence* against edits and partial deletions, not tamper-proofing, and `test_total_erasure_is_undetectable_and_we_say_so` asserts the limitation so it cannot be quietly claimed away later.

**Alternatives rejected:** *Store the head hash in a `meta` table and check it* — raises the bar by one `UPDATE` and reads as a defence while providing none, which is worse than the stated limitation. *Claim tamper-proofing and hope nobody asks* — the failure mode this entire project argues against, applied to our own log.

**Consequences:** The limitation goes in `RESULTS.md` §12 and in the README. If an external anchor is ever added, this entry is superseded rather than edited.

---

## 026 — Purging is an explicit, logged act, and the retention floor is a hard refusal
**Status:** accepted

**Context:** DPDP Rule 6 sets a *minimum* retention of one year. Deleting from a hash chain breaks it, so retention and tamper-evidence pull against each other.

**Decision:** Nothing is deleted automatically. `purge_older_than` defaults to a dry run, **refuses** any cutoff newer than `now − retention_days` regardless of arguments, and when it does delete, appends a `retention_event` recording the count, the seq range and the hash of the last record removed. The surviving chain re-anchors at that hash, and verification checks the two against each other.

**Alternatives rejected:** *Delete silently on a schedule* — the chain would verify over a log that no longer contains what it claims to, which is a worse property than not purging at all. *Never delete* — a data-minimisation obligation is not satisfied by keeping everything forever, and a store that cannot forget cannot honour an erasure request.

**Consequences:** The ledger takes an injectable clock, because the alternative way to test a 400-day floor is to wait 400 days. A reviewer could fairly object that an injectable clock is also a way to forge timestamps — true of any system clock, and the mitigation is that the timestamp is inside the hashed body, so a backdated record cannot be inserted into an existing chain without breaking it.

---

## 027 — Synthetic fixtures cannot masquerade as measured eval sets, structurally
**Status:** accepted

**Context:** The harness has to be buildable and testable on a laptop, but the numbers it produces must come from a real extraction on a real model. The obvious risk is a synthetic fixture's numbers ending up in `RESULTS.md` — either by a script defaulting to the fixture, or by someone reading a plot six weeks later and forgetting which run produced it. Labelling by convention fails exactly when it matters.

**Decision:** `data_source` (`"measured"` | `"synthetic"`) is part of an eval set's **hashed identity**, alongside its items and construction notes. Since the content hash *is* the envelope id, and the envelope id is the third element of the warrant key (invariant 1), a synthetic set occupies a **different cell in the warrant matrix** from the real set of the same name. Numbers measured on a fixture are filed under `sha256:…` for the fixture and can never be read as numbers for `triviaqa-600`.

Three further guards fall out of the same mechanism:
- `synthetic_cache()` **refuses** to attach generated features to a set marked `measured`, because the cache would then carry the real set's hash.
- `ExtractionCache.load()` takes the eval set's current hash and refuses a cache whose hash disagrees — a set edited after extraction makes its cache stale, and validating against it files numbers under an envelope that no longer describes the data.
- Both the set's `construction` block and the cache's `extra` block carry an explicit warning string, and both are hashed, so the warning cannot be stripped without changing the identity.

**Alternatives rejected:** *A boolean flag checked by the report writer* — one `if` between a fixture and a published number, and the failure is silent when someone adds a second code path. *Keep fixtures only in `tests/`* — attractive, but the smoke test, the CPU development path and the demo's rehearsal fallback all need them, and copies of a generator drift.

**Consequences:** The synthetic signal strengths in `synthetic_cache` are **parameters, not findings**, and a tier ladder computed from them is a picture of that function. This is stated in the module docstring, in the cache's `extra` block, and on any plot generated from a synthetic run. The real ladder requires the real extraction.

**What the fixtures do reproduce honestly** is mechanism rather than magnitude: sequences carry a localised signal and are pooled through the real `aggregate()` code, so mean pooling's collapse on long context is arithmetic we can observe, not a value we typed. Measured on the fixture generator: a 32-position signal in 96 tokens versus the same signal in 1,536 tokens.

**A reviewer could fairly object** that a repo containing a synthetic-data generator invites suspicion about which numbers are real. Fair — the answer is that every artifact carries `data_source` in its provenance, every warrant is keyed by a hash that differs between the two, and `test_synthetic_cannot_masquerade` asserts the separation rather than describing it.

---

## 028 — Max-of-rolling-means carries an extreme-value bias that grows with context
**Status:** accepted · **flagged for the real ablation, not resolved here**

**Context:** `SPEC.md` §3.1 and the Phase 4 gate anticipate that mean pooling is **REFUSED** on `triviaqa-longctx-600` while max-of-rolling-means holds a valid warrant at wider bounds. Building the synthetic fixture produced the opposite ordering, and the reason is structural enough to write down before the real run rather than after.

**What happened:** on the fixture, short context gives mean-pool 0.703 and max-rolling 0.770 — the expected direction. At 16× the context length, mean-pool reads 0.600 and max-rolling **0.545**, i.e. max-rolling degrades *further and faster*.

**Why, and why it is not a bug in either implementation:**

Max-of-rolling-means takes an element-wise maximum over `W` window means. Under the null — a window containing only noise — each window mean is approximately `N(0, σ²/w)`, and the maximum of `W` such draws has expectation growing like `σ/√w · √(2 ln W)`. That is a **positive bias that depends only on how many windows there are**, not on whether any signal is present. As context grows, `W` grows, the bias grows for *every* item regardless of label, and it grows with variance attached. Meanwhile the true signal contributes to exactly one window and does not grow at all.

So the strategy has two competing effects as context lengthens: it protects the signal from dilution (the reason it exists), and it accumulates an extreme-value pedestal that adds label-independent variance (the reason it can lose anyway). Which effect wins depends on the ratio of signal span to window length and on `W`. On the fixture, with a 32-position signal inside a 64-token window, the signal is already halved within its own window before the pedestal is added, and the pedestal wins.

**Decision:** change nothing to make the fixture produce the anticipated ordering. Two reasons. First, `KICKOFF.md` is explicit — *"do not manufacture a failure to fill a beat"* — and manufacturing the *success* half of the same beat is the same offence. Second, this is a real property of the estimator and the real ablation needs to be read with it in mind.

**What this changes for Phase 4:** the matrix cell for max-of-rolling-means on long context is genuinely open, and both outcomes are reportable. If it holds a warrant, Beat 4 runs as scripted. If it is refused alongside mean-pool, the honest finding is that **no activation-tier aggregation we tested survives the envelope shift**, the matrix routes to T2 or T3, and Beat 4 is *stronger*, not weaker: the system refuses to certify at the tier it prefers and says why.

**Worth testing at the real run, cheaply:** the pedestal scales with `√(2 ln W)`, so a window sized to the signal span rather than fixed at 64 tokens would reduce both the in-window dilution and `W`. `probe.rolling_window` is a config knob and the sweep is a validation-set decision, so it can be done without touching test.

---

<!-- New entries below. Do not edit anything above this line. -->
