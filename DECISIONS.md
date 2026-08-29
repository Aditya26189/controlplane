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

## 029 — Negative controls average over repeats, and their band is noise-aware
**Status:** accepted · **changes a pass condition in `SPEC.md` §2.1; spec updated in the same commit**

**Context:** `SPEC.md` §2.1 specifies label-shuffle and null-feature as passing when AUROC lands in `[0.45, 0.55]`. Implemented literally — one permutation, one fixed band — the control failed on the first clean run at 0.4375, and again at 0.5684 after an unrelated fix. Neither failure indicated a fault. That is the worst possible behaviour for a control: it refuses warrants at random, and a suite that cries wolf gets switched off.

**The derivation.** A negative control asserts *"AUROC is consistent with 0.5"*. Whether an observed value is consistent with 0.5 depends on sampling noise, which depends on `n` — so a **fixed** band is only a valid test at one particular holdout size. Under H₀ the Hanley–McNeil standard error is

```
SE = sqrt((n_pos + n_neg + 1) / (12 · n_pos · n_neg))
```

At base rate 0.152, the configured ±0.05 band measures:

| holdout n | null SE | band width | P(fails with no fault) |
|---|---|---|---|
| 150 | 0.0656 | ±0.76 SE | **44.6%** |
| 300 | 0.0463 | ±1.08 SE | 28.1% |
| 600 | 0.0329 | ±1.52 SE | **12.8%** |
| 1200 | 0.0232 | ±2.15 SE | 3.1% |
| 2400 | 0.0164 | ±3.05 SE | 0.2% |

At the sizes this project works with, the spec's control as written refuses roughly **one warrant in eight for no reason but noise**.

**Decision — two changes, in order of importance.**

1. **Average over repeats.** Both negative controls now run `validation.null_control_repeats` (5) independent draws — permutations for label shuffle, noise draws for null feature — and test the **mean**. The SE of the mean falls as `1/√repeats`, so at n=600 the effective SE drops from 0.0329 to 0.0147 and the configured ±0.05 band becomes ±3.45 SE: a real bar. Observed per-permutation values on one run were `[0.5684, 0.5954, 0.4169, 0.5336, 0.5997]`, spanning 0.18 — which is precisely why a single draw could never carry this.

2. **Floor the band at ±2 SE of the mean.** Retained as a backstop for holdouts small enough that repeats cannot rescue them. It only ever *widens*, never tightens, so the declared bar is honoured wherever it is statistically meaningful. Each control reports the band it actually applied and why.

**Power against real faults is essentially unchanged.** The faults these controls exist to catch — split leakage, index misalignment, a feature encoding the label — do not produce an AUROC of 0.56. They produce one far outside any band under discussion. What was lost is the ability to detect a leak worth ~0.01 AUROC, which was never detectable at this `n` anyway; the previous configuration only appeared to detect it.

**Alternatives rejected:** *Widen the configured band to ±0.10* — hides the `n`-dependence instead of addressing it, and at n=2400 it would be a needlessly weak bar. *Report the failure and refuse* — correct in the literal reading of the spec and operationally useless, since the refusal carries no information. *Drop the negative controls* — they are two of the five and the reason the suite means anything.

**Consequences:** `config.yaml` gains `validation.null_control_repeats`. Both controls report the mean, the standard deviation, every per-run value, and the applied band, so a reader can audit the decision rather than take the pass on trust. Cost is 5× the probe fits for two controls, which is a few seconds.

**A reviewer could fairly object** that averaging permutations makes the control easier to pass, and that we changed a pass condition after seeing it fail. Both are true and neither is hidden: the numbers that prompted the change are in the table above, the change was made on the *distribution* of the statistic rather than on the threshold, and it was made before any measured result existed to be flattered by it — every run so far is synthetic fixture data, which cannot reach `RESULTS.md` by construction (`DECISIONS.md` 027).

---

## 030 — An eval set is sized by its test split, not by its total
**Status:** accepted

**Context:** `SPEC.md` §4 names `triviaqa-600` and §2.3 refuses any warrant with `n_test < 200`. Splitting 600 items three ways at (0.5, 0.25, 0.25) yields a test split of 150 — below the refusal bar, so the set named in the spec could never produce a warrant.

**Decision:** An eval set may **declare** a split per item, and a declared split is honoured rather than re-derived. `triviaqa-600` means *600 held-out test items* — the Round 1 anchor, which was a held-out set — with train and validation rows supplied from a separate extraction. Sets that declare no splits still get the derived question-level split, which is what the hand-built Phase 3 sets will use.

Two guards: a partial declaration (some items only) is an error rather than a silent mix of declared and derived; and declared splits are checked for question overlap exactly as derived ones are, since a hand-written split is at least as likely to put one question on both sides.

**Consequences, and this is the operationally important part:** the real extraction must cover roughly **2,400 TriviaQA items**, not 600 — 1,200 train, 600 validation, 600 test — for the anchor set to clear both `min_n_test` and the negative-control power floor from `DECISIONS.md` 029. That is a four-fold increase in GPU time over the naive reading of the spec, and it is better discovered now than during the run.

---

## 031 — Negative controls size their repeats from their own measured null
**Status:** accepted · **supersedes the sizing half of 029**, whose reference distribution was wrong

**Context:** Entry 029 fixed a real problem — a fixed `[0.45, 0.55]` band is only valid at one holdout size — but fixed it against the **wrong reference distribution**. It used the Hanley–McNeil null SE and concluded that 5 repeats made the band a ±3.45 SE bar at n=600. Running the ablation, label-shuffle then failed at 0.5546, which that model says is a 3.8 SE event. Three-sigma events do not happen on the second run, so the model was wrong.

**What we measured.** 200 permutations per variant on the fixture, holdout n=600, base rate 0.152:

| variant | features | empirical null mean | empirical null sd | Hanley–McNeil SE | ratio |
|---|---|---|---|---|---|
| `T1-max_rolling_means` | 32 | 0.4982 | 0.0684 | 0.0324 | **2.11×** |
| `T3-judge` | 1 | 0.4754 | 0.2706 | 0.0324 | **8.34×** |

The null is centred correctly — there is no leak — but it is far wider than the closed form, and the width depends on the **feature dimensionality**.

**Why.** Hanley–McNeil gives the variance of AUROC when the score vector is an exchangeable random permutation of ranks. A fitted probe's scores are not exchangeable: they are a smooth function of the features, so similar items receive similar scores and the effective number of independent draws is far below `n`. In the limit that makes the failure obvious, a **one-dimensional** feature gives a probe that is essentially ±(that feature), so a label shuffle picks a *sign* and AUROC lands near `A` or `1 − A`. That is a two-point distribution with sd ≈ 0.27, and no amount of reasoning about `n` predicts it.

**Decision:** stop using a closed form. Each negative control **measures its own null** and sizes its repeat count from it:

```
SE(mean) = spread / sqrt(k)          band is meaningful when band_half >= 2 * SE(mean)
                                     => k >= (2 * spread / band_half)^2
```

Run `null_control_min_repeats` (8) draws, estimate the spread, continue to the implied `k`, capped at `null_control_max_repeats` (200). Passing now requires **two** things: the mean lies inside the configured band, *and* the SE of that mean is at most half the band's half-width. Failing the second is reported as a **failure**, not a pass — a control whose job is to demonstrate that the pipeline can produce a null result, and which lacks the power to demonstrate it, has demonstrated nothing.

**Measured repeat counts on the fixture**, sized automatically: 8 at 32 features, 22 at 4 features, **125 at 1 feature**. The last matches the analytic estimate of ≈117 from the observed spread, which is the check that the sizing rule is doing what it claims.

**What survives from 029:** the diagnosis that a fixed band is `n`-dependent and the false-failure table, both correct. **What does not:** the claim that 5 repeats suffice, and the ±2 SE floor computed from the closed form. Both are replaced by the empirical rule above.

**Alternatives rejected:** *Use the closed form and widen the band* — the direction of the error varies by 4× between variants, so any fixed widening is simultaneously too loose for T1 and far too tight for T3. *Drop the control on low-dimensional features* — T3 is the tier the whole ladder is compared against, and having no negative control on it is precisely where a fault would hide. *Fix repeats at 200 for everything* — wasteful at T1 and still arbitrary; the measured spread is available for free and is the honest input.

**A reviewer could fairly object** that we have now revised a control's pass condition twice. Correct, and both revisions are recorded with the numbers that forced them. The first was right about the problem and wrong about the reference; the second measured the reference instead of assuming it. No measured result existed at either point — every run so far is synthetic fixture data, which cannot reach `RESULTS.md` by construction (`DECISIONS.md` 027) — so nothing was tuned against a number we wanted.

---

## 032 — A single-class envelope supports an FPR claim and nothing else
**Status:** accepted

**Context:** `hard-negatives-200` contains **no positives** by construction — every item is a benign boundary case that must be allowed. That makes AUROC, recall and precision undefined on it, while `SPEC.md` §3.1 shows it as a column in the warrant matrix with `VALID`/`REFUSED` cells, and §2.3's refusal criteria are keyed on an AUROC lower bound that cannot exist there.

**Decision:** `WarrantMetrics.auroc`, `.recall` and `.precision` are optional, **all-or-nothing**. Either all three are present or none are; a warrant carrying AUROC without recall, or recall without precision, is unconstructible. Invariant 5 therefore survives intact — precision and recall are absent *together*, so there is still no way to claim one without the other.

On a single-class envelope the refusal path substitutes the criterion: the AUROC bar is skipped, and a **declared `max_fpr_hard_negatives` becomes mandatory**. Without it there would be no bar at all, which is worse than refusing. `lift` raises rather than returning a number, because lift is `recall / flag_rate` and inventing the missing half is exactly the failure this project is about.

**Alternatives rejected:** *Add synthetic positives so every metric is defined* — destroys what the set is. Its value comes from every item being one that must be allowed; a set with positives in it measures something else. *Refuse to warrant single-class envelopes* — then the FPR number a skeptic actually cares about carries no warrant, which inverts the priority.

**Consequences:** the matrix has cells whose claim is narrower than others', and that narrowness is visible in the record rather than inferred. A reviewer could fairly object that `VALID` means different things in different columns — true, and the certificate names which metrics the warrant actually carries.

---

## 033 — `hinglish-pii-200` is balanced, and its precision is not a production precision
**Status:** accepted

**Context:** 51 hand-written scenarios × 3 disclosure forms gives 153 positives against 47 near-miss negatives — a base rate of 0.77. Precision measured on a set that enriched says nothing about precision on real traffic, and 47 negatives leave FPR with an interval too wide to report.

**Decision:** two of the three disclosure forms per scenario, rotating, giving **102 positives and 98 negatives** at a base rate of 0.51, with every scenario present and each form covered 34 times. Per-form recall — the finding this set exists to produce — retains n=34 per form.

**And the caveat is recorded in the set itself.** The `construction` block states that the set is enriched relative to real traffic, so precision measured here is not a production precision, and that FPR for a PII detector comes from the near-miss negatives while FPR for a *content* detector comes from `hard-negatives-200`. Both are hashed with the contents, so the caveat cannot be separated from the number.

**Measured on our reference detector, which is the point of building it:** recall by disclosure form at threshold 0.35 — verbatim **1.000**, spaced **0.676**, obfuscated **0.706** (n=34 each). A detector purpose-built for these formats still loses roughly 30% on non-verbatim disclosure. That is the structure Presidio's published 0.07 comes from, reproduced on our own floor.

---

## 034 — A control may be inapplicable, and that is fenced three ways
**Status:** accepted

**Context:** Three of the five controls exist to catch faults in *fitting* — a padding side that makes activations meaningless, a split that leaks, features that carry the label. A stateless rule-based detector fits nothing, so those failure modes cannot occur for it. Reporting them as "passed" would claim checks that never ran; reporting them as "failed" would refuse every rule-based detector forever.

**Decision:** `ControlResult.applicable`. An inapplicable control is recorded, with its reason, and does not refuse the warrant. The warrant then carries `controls_run: 2` where a probe's carries five, so a reader can see that a rule-based detector's warrant rests on less evidence.

This is the one escape hatch the design could grow, so it is fenced:

1. **Applicability is declared per detector class in code**, in `_STATELESS_INAPPLICABLE`, never as a runtime argument. A per-run flag is exactly how "not applicable" becomes "not checked, and nobody noticed".
2. An inapplicable control must carry **no verdict and no margin** — `passed=True, margin=0.0` enforced in `__post_init__` — so it cannot masquerade as a pass with room to spare.
3. It must **state why the mechanism cannot exist**. "Not applicable" without a reason is an override with better manners.

**Alternatives rejected:** *Run the three controls anyway and let them trivially pass* — a control that cannot fail is worse than no control, and this repo's own README argues exactly that. *Refuse warrants to stateless detectors* — Presidio is a stateless detector and warranting it is a Phase 8 deliverable.

**A reviewer could fairly object** that this is a hole in invariant 3. It is a *narrowing* of it: refusal still has no override, and what changed is which controls are in scope for which detector class — declared in code, visible in every warrant, and asserted by `test_inapplicable_controls_are_fenced`.

---

## 035 — Zero events do not mean zero rate: exact intervals at the boundary
**Status:** accepted

**Context:** The reference detector fired on **0 of 200** hard negatives. The percentile bootstrap returned `[0.0000, 0.0000]`, because every resample of a set with no events also has no events. A zero-width interval is a claim of *perfect certainty from 200 observations* — the loudest possible false claim in a project whose thesis is that unbacked claims are the problem, and it was produced by our own estimator without raising anything.

**Decision:** when the bootstrap collapses to zero width on a proportion, fall back to the **Clopper–Pearson exact binomial interval**, computed from the event count and the trial count.

```
0 events in 200 trials, 95%:  [0, 0.0183]
```

which is the familiar rule of three — with no events in `n` trials the upper bound is about `3/n`. *"At most 1.8%, from 200 observations"* is both true and useful. *"0.0%"* is neither.

Quantity: a binomial proportion. Propagation: Clopper–Pearson, inverting the binomial CDF through the Beta distribution, which has guaranteed coverage precisely where normal approximations and resampling both fail.

**Alternatives rejected:** *Add a pseudo-count* — arbitrary, and shifts the point estimate away from the observed value. *Report the zero-width interval and let the reader interpret it* — the reader is a judge, and the interval says something false. *Use Wilson everywhere* — Wilson is better behaved than the normal approximation but still degenerate at exactly zero, which is the case that occurred.

**Consequences:** `estimated()` accepts the event and trial counts for proportions and uses them only at the boundary; away from it the bootstrap is unchanged, so no existing number moves. The estimator string records which was used, so a reader can tell them apart.

**Before:** `fpr_hard_negatives 0.0000 [0.0000, 0.0000]`.
**After:** `fpr_hard_negatives 0.0000 [0.0000, 0.0183]`.

---

## 036 — Within-set FPR and hard-negative FPR are different claims
**Status:** accepted

**Context:** The first run of the Phase 3 script **refused** the reference detector a warrant on `hinglish-pii-200`, citing `fpr_hard_negatives_upper_ci 0.6339, required <= 0.02`. The detector had never been scored on the hard-negative set. What had been measured was its FPR against that set's *own* near-miss negatives — order ids and transaction references — and the number had been written into the field a profile declares its hard-negative maximum against.

**Decision:** two fields, two meanings. `fpr` is the FPR within whatever set is being scored. `fpr_hard_negatives` is populated **only** when the eval set under test *is* the hard-negative set, and only that field faces a profile's declared maximum.

**Why it matters beyond a naming tidy-up:** the refusal was *plausible*. A detector refused for a bar it failed is exactly what this system is supposed to produce, and nothing about the output looked wrong — the number was real, the criterion was real, and they were about different things. That is the shape of every failure mode in `CLAUDE.md`'s silent-failure list, arriving in our own refusal path.

**Consequences:** `validate_text_detector` takes an explicit `is_hard_negative_set` flag rather than inferring it from the set's name, because a name is a string and someone will rename a set.

---

## 037 — `hinglish-pii-200-longctx` exists to show that a regex detector is length-invariant
**Status:** accepted

**Context:** `SPEC.md` §4 defines `triviaqa-longctx-600` as the drift trigger, and it cannot be built without model generations. Building the long-context transform against the Hinglish set instead gave a result worth keeping.

**Measured:** the reference PII detector scores **identically** on `hinglish-pii-200` and `hinglish-pii-200-longctx` — AUROC 0.7734 on both, every metric equal to the last digit — despite mean prompt length rising from 71 characters to 39,402.

**Why this is a finding and not a triviality:** it is the control case for Beat 4. When the mean-pooled probe collapses on long context and the matrix routes away from it, the obvious question is *"is this just what happens to everything on long inputs?"* The answer is no, and here is a detector on the same transform whose numbers do not move at all. A stateless pattern matcher has no pooling step to dilute, so context length is invisible to it. That makes the probe's collapse a property of **pooling**, not of long inputs in general — which is a much sharper claim.

**Consequences:** the long-context transform is validated end to end before the TriviaQA sets exist, and the matrix gains a row whose length-invariance is measured rather than assumed. `build_longctx` preserves `question_id` from the base set so a split derived on either lands identically, and the padded set gets its own content hash and therefore its own envelope — it inherits no warrant from the base set, which is invariant 1 doing its job.

---

## 038 — Superseding 028: the max-of-rolling-means pedestal was a fixture artefact
**Status:** accepted · **supersedes 028**

**Context:** Entry 028 recorded that max-of-rolling-means degraded *faster* than mean pooling on long context (0.545 against 0.600 on the fixture), attributed it to an extreme-value pedestal growing like `√(2 ln W)` in the window count, and flagged the Phase 4 cell as genuinely open.

**What changed:** the measurement in 028 was taken before the synthetic generator was corrected in Phase 2. At that point every cache drew its own readable direction from its own seed, and the signal was spread across all 32 dimensions rather than living in a subspace. Both are unrealistic, and both distorted the comparison — a signal spread over every dimension survives dilution trivially, which flatters mean pooling.

**Measured now**, same fixture, corrected generator, n_test 600:

| aggregation | short context | long context |
|---|---|---|
| `T1-mean_pool` | AUROC 0.797 [0.749, 0.847], recall 0.216 [0.129, 0.306] | AUROC 0.629 [0.560, 0.689], recall **0.034 [0.000, 0.077]** |
| `T1-max_rolling_means` | AUROC 0.801 [0.754, 0.846], recall 0.250 [0.163, 0.345] | AUROC 0.724 [0.662, 0.781], recall 0.159 [0.088, 0.240] |

Mean pooling loses **84% of its recall** and its lower bound falls to zero — it can support no recall claim at all. Max-of-rolling-means loses 36% and keeps a usable interval. The direction 028 questioned is the direction the corrected fixture produces.

**What survives from 028:** the analysis of the pedestal is still correct as *mechanism* — the maximum of `W` window means does carry a label-independent positive bias growing like `√(2 ln W)`, and it does compete against the protection from dilution. What was wrong was the conclusion that the pedestal wins, which was an artefact of a generator that made dilution too easy to survive.

**What this does not establish.** These are fixture numbers and cannot reach `RESULTS.md` (`DECISIONS.md` 027). The real cells are `UNVALIDATED` and stay that way until the GPU run. The value here is that the *machinery* demonstrably distinguishes the two aggregations, and the suggestion in 028 — sweep `probe.rolling_window` against the signal span on validation — is still worth doing on real activations.

---

## 039 — The AUROC bar does not catch a recall collapse; the profile minimum does
**Status:** accepted

**Context:** Populating the matrix produced a result the refusal criteria in `SPEC.md` §2.3 do not catch. On long context, `T1-mean_pool` holds recall **0.034 [0.000, 0.077]** — a lower bound of exactly zero, meaning no recall claim is supportable at all — and yet its warrant is issued **VALID**, because its AUROC lower bound is 0.560 and the declared bar is 0.55.

A warrant that says *"I rank slightly better than chance, and I will catch between none and 7.7% of your errors"* is technically valid and operationally worthless.

**Decision:** do not add a refusal criterion. Report it as measured, and let the **profile minimum** be what catches it — which it does, correctly and visibly:

```
customer_support (recall >= 0.10)  short context  -> ROUTED to a probe
customer_support (recall >= 0.10)  long context   -> SUSPENDED
    probe-T1-mean_pool: requires recall >= 0.1; the warrant's lower bound is
    0.0000 (point estimate 0.0340)
```

**Why not tighten the refusal bar:** three reasons, in order of weight.

1. **It would invent a threshold the spec does not declare.** `min_auroc_lower_ci` is a config value with a stated meaning; a second implicit bar on recall would be a number we chose because we did not like a result, which is the failure pattern this repo exists to argue against.
2. **The two-layer structure is the correct design and this demonstrates it.** Validation asks *"is this measurement sound?"* — controls passed, interval honest, `n` adequate: yes. A profile asks *"is it good enough for what I do?"* — separately, and per profile. `customer_support` at 0.10 and `decision_support` at 0.50 draw different lines on the same warrant, which is impossible if the bar lives in issuance.
3. **A universal recall floor would be wrong for some detectors by design.** The probe is tuned for recall at a fixed flag-rate budget and its precision is deliberately poor; a hard-negative FPR warrant carries no recall at all. One number cannot serve them.

**What this changes for Beat 4:** the narration is *sharper* than the script anticipated. The scripted beat says the warrant is revoked. What actually happens is more interesting and harder to argue with — the warrant stays valid, the measurement is still sound, and **the profile suspends itself because the honest number is no longer enough for what it does**. "This detector is broken" is a weaker claim than "this detector is still working exactly as measured, and what it can now prove is below what this profile requires."

**A reviewer could fairly object** that a `VALID` cell showing recall `[0.000, 0.077]` looks like a system claiming something useless is fine. The matrix renders the interval next to the status precisely so that reading is available, and the routing decision names the number that suspended the profile.

---

## 040 — The within-set FPR conflation recurred on the second code path
**Status:** accepted

**Context:** `DECISIONS.md` 036 recorded a defect on the text-validation path: the FPR measured *within* whatever set was under test was being written into `fpr_hard_negatives`, the field a profile declares its maximum against. It was fixed there. The probe path had the same bug and was not touched, because the fix was made where the symptom appeared rather than where the cause lived.

**How it surfaced:** every profile suspended on every activation envelope, including short context where the probe is at its best. The probe's within-set FPR upper bound of 0.029 was being judged against `customer_support`'s declared hard-negative maximum of 0.02 — a bar that had never been measured for that detector on any hard-negative set.

**Decision:** both runners now take an explicit `is_hard_negative_set` flag and populate `fpr_hard_negatives` only when the set under test *is* that set; otherwise the within-set FPR is reported as `fpr`.

**The lesson worth recording**, which is why this is an entry rather than a commit note: the second occurrence was invisible for exactly as long as the first, and for the same reason — the output was plausible. "Every profile is suspended" reads like a conservative system doing its job, and it is what a genuinely weak detector would produce. It was only caught because the suspension persisted on the envelope where the probe was known to be strong, which was a hunch rather than a check.

**Consequences:** a fix applied at one call site when the cause is a shared concept should be followed by a search for the other call sites. Recorded here so the next occurrence of this shape gets found by reading rather than by luck.

---

## 041 — The matrix lists declared-but-unbuilt envelopes so their cells show UNVALIDATED
**Status:** accepted

**Context:** `triviaqa-600` and `triviaqa-longctx-600` cannot be built without model generations, so no warrant references them. A matrix built only from the warrants that exist would not contain those columns at all.

**Decision:** `WarrantMatrix` takes the detector and envelope axes explicitly rather than deriving them from the warrants present, and the script passes every declared eval set including the unbuilt ones. Their cells render `UNVALIDATED`.

**Why this matters more than it looks:** *"we have not measured this"* and *"we have not thought about this"* are different, and only the first is acceptable in a system whose product is knowing what it does not know. An absent column is indistinguishable from an oversight; an `UNVALIDATED` column is a stated gap. The current matrix is 24 `UNVALIDATED` cells out of 35, which is the expected shape rather than an embarrassment — `UNVALIDATED` is the modal state in production, and a matrix that did not look like this would mean we were hiding cells.

---

## 042 — Routing ranks by recall lower bound, not interval width
**Status:** accepted · **corrects the ranking rule introduced in Phase 4**

**Context:** The first ranking rule ordered eligible warrants by interval width, tightest first, on the reasoning that a narrow claim is more useful than a wide one. The Phase 4 report described the resulting choice as the rule working as designed. It was not.

**What it actually did:** on short context it preferred mean-pool (recall 0.216 [0.129, 0.306], width 0.177) over max-of-rolling-means (0.250 [0.163, 0.345], width 0.182). Width differed by 3%; the midpoint differed by 16%, and the lower bound by 26%. Generalised, width-first prefers recall `[0.10, 0.11]` over `[0.30, 0.45]`, which is plainly wrong.

**Decision:** rank by the **recall lower confidence bound**, descending. Ties break on width (between two equally provable claims, prefer the better-measured one) and then on detector id, so routing stays deterministic.

**Why the lower bound specifically:** it is what the detector can be *shown* to deliver, and claiming what you can prove is the entire product thesis. Width-first optimises for confidence in a claim rather than the size of it — a subtly different objective that happens to coincide with the right answer often enough to look correct.

**Why this mattered more than a cosmetic ordering:** Phase 5's route-on-revocation calls the same function. A wrong ranking rule would have propagated directly into Beat 4's fallback selection, where the system picks the detector it falls back *to* in front of an audience.

**Consequences:** re-running Phase 4 with the corrected rule routes `max_rolling_means` on short context, which is the answer the thesis implies. `test_routing_ranks_by_lower_bound_not_width` pins the generalised case that made the error obvious.

---

## 043 — Refuse a warrant whose lift lower bound does not exceed 1.0
**Status:** accepted · **found by Phase 4 measurement, not designed in**

**Context, and the provenance matters more than the criterion:** this was not in the spec and was not anticipated. Phase 4 populated the matrix and produced a warrant issued **VALID** on recall `0.034 [0.000, 0.077]`. Its AUROC lower bound was 0.560 against a declared bar of 0.55, so every stated refusal criterion passed. The first response was to report it as measured and let the profile minimum catch it — which was right as far as it went, and left a real gap.

**The gap:** the refusal bar is AUROC-based, and AUROC is a **ranking-quality** measure. The product's claim is about **usefulness at a budget**. Those are different quantities and they come apart exactly where this was found. A detector can rank slightly better than chance and still be worthless at any operating point you would actually run.

**The derivation.** Lift is `R / f` — how many more errors the detector surfaces than random sampling of the same number of items. `f` is the *measured* flag rate (invariant 6). Random sampling at budget `f` yields recall `f` in expectation, so lift 1.0 is the null: *no better than spending the same money at random*. Measured on the offending cell:

```
recall 0.034 [0.000, 0.077]  at  f = 0.0250
lift    1.36 [ 0.00,  3.08]
```

The point estimate is above 1, and the lower bound is **zero**. The system could not show the detector beat random sampling, and issued a warrant saying it was fine.

**Decision:** refuse when the **lower bound** on lift does not exceed 1.0. `MIN_LIFT_LOWER_BOUND = 1.0` is a constant in code, not a config value, because 1.0 is the *definition* of "no better than chance at this cost" rather than a threshold anyone chose. Wanting a higher bar is a policy judgement and belongs in a profile's declared minimum.

**Measured effect across the whole matrix** — it refuses two cells and touches nothing else:

| cell | lift | verdict |
|---|---|---|
| `T1-max_rolling_means` short | 5.00 [3.27, 6.90] | VALID |
| `T1-max_rolling_means` long | 3.41 [1.88, 5.14] | VALID |
| `T1-mean_pool` short | 4.63 [2.76, 6.56] | VALID |
| `T1-mean_pool` long | 1.36 [**0.00**, 3.08] | **REFUSED** |
| `T2-logprob` long | 2.56 [1.35, 3.95] | VALID |
| `T3-judge` long | 1.18 [**0.56**, 1.96] | **REFUSED** |

**Both mechanisms are kept.** Refusal and profile suspension answer different questions and both fire here for different reasons. Refusal says *the warrant is worthless to anyone*. Suspension says *the warrant is sound and insufficient for this profile* — `customer_support` at recall ≥ 0.10 suspends on long context even where warrants remain valid. Replacing one with the other would lose a distinction the demo turns on.

**A reviewer could fairly object** that the Phase 4 gate now reads as the spec scripted it — mean-pool REFUSED on long context — after a report that made a point of not engineering that outcome. The order of events is the answer, and it is in the git history: the gate was reported with mean-pool VALID and the discrepancy stated; the criterion was derived afterwards from the product's own definition of usefulness; and it was checked against every other cell to confirm it refuses two and spares six. A criterion chosen to produce a demo would not have refused `T3-judge` as well.

**Caveat worth stating:** on an enriched eval set, lift is compressed toward 1 because random sampling does well when positives are common. `hinglish-pii-200` has a base rate of 0.51, so its lift is 1.28 [1.13, 1.43] — genuinely modest, correctly reported, and a reminder that lift is only interpretable against the base rate it was measured at.

---

## 044 — A metric computed twice is a metric that will drift; there is now one implementation
**Status:** accepted · **replaces the mitigation proposed in 040**

**Context:** Entry 040 recorded that the `fpr_hard_negatives` conflation recurred on a second code path, and proposed that the next instance "gets found by reading". That is not a control. It is what every team has, and this repo's entire claim is to be different from that.

**What the failure class actually is:** *the same quantity, two implementations, one drifts*. `WarrantMetrics` was constructed in two places with five `estimated()` calls each. The bug shipped in one, was fixed in one, and survived in the other through 193 passing tests. It was caught by a hunch about an unrelated symptom.

**Decision:** eliminate the duplication rather than test that two copies agree. `src/validation/metrics_builder.py` is the single place a `WarrantMetrics` is constructed; both runners call it and neither computes a metric itself. Two controls guard it:

1. **`test_warrant_metrics_has_exactly_one_construction_site`** parses every module with `ast` and asserts exactly one `WarrantMetrics(...)` call exists. Parsed rather than grepped so a construction inside a docstring cannot register. `test_estimated_is_called_from_one_module` does the same one level down, for the function that decides bootstrap count, coverage, resampling unit and boundary fallback.
2. **`test_metric_paths_agree`** asserts identical input produces identical metrics. One construction site is not sufficient on its own: two callers could still pass different seeds or different resampling units.

**And a routing positive control**, for the second shape the bug took. The symptom was *universal suspension* — every profile suspended on every envelope — which is indistinguishable on screen from a conservative system working correctly. `test_routing_positive_control` asserts that a detector with recall 0.40 [0.30, 0.50] **routes** rather than suspending. This is the null-feature control's reasoning applied to routing: a system that cannot produce the non-null outcome cannot be trusted when it produces the null one.

**Grep pass for other dual-path metrics**, done and recorded: `WarrantMetrics` had two construction sites (now one), `estimated()` had two calling modules (now one), and no other quantity in `src/` is computed in more than one place.

---

## 045 — Contamination scope from the fixture generator fix, measured rather than assumed
**Status:** accepted

**Context:** Entry 038 superseded 028 on the grounds that its numbers predated a fix to the synthetic generator. That raises a scope question the report treated as settled: **which other artifacts are contaminated?**

**Two things changed between 028's measurement and Phase 4's**, and 038 attributed the reversal to only one of them without testing:

* (a) the generator fix — `direction_seed` shared across caches, and sorted variant iteration, which changes the whole RNG stream;
* (b) sample size — 600 items with derived splits (train 300, test 150) became 2,400 with declared splits (train 1,200, test 600).

**Measured, under the current generator, at both sizes:**

| condition | mean-pool long | max-rolling long | winner |
|---|---|---|---|
| 028's size (n=600, derived splits) | 0.565 | 0.706 | max-rolling |
| Phase 4's size (n=2400, declared) | 0.629 | 0.724 | max-rolling |

At 028's *own* sample size the current generator still reverses its result. So the reversal is attributable to the generator change, and 038's causal story holds — now tested rather than asserted.

**Scope, established from `provenance()` git stamps rather than by inspection.** The generator fix landed in `db11799`. Every committed `exp:` commit in Round 2 postdates it:

```
31310c3  exp(validation): tier ladder ...          after
589e6e6  exp(evalsets): register four sets ...     after
2ed2a34  exp(evalsets): regenerate registry ...    after
55237f0  exp(evalsets): regenerate from clean ...  after
6133ef7  exp(matrix): populate every cell ...      after
```

**Conclusion: zero committed artifacts are contaminated.** The contamination is confined to prose — 028's inline numbers, recorded in a `feat:` commit and already superseded. Phase 3's results use no synthetic generator at all (the eval sets are hand-built), so they are unaffected by construction.

**What made this answerable in one command** is that every artifact carries the commit it was generated from. That is the provenance discipline paying for itself, and it is worth noting that the question *"what else is wrong?"* is normally unanswerable.

---

## 046 — `RESULTS.md` refuses to print a number from a synthetic envelope
**Status:** accepted

**Context:** A consequence of the Phase 2 disclosure that the report did not draw. There are no measured TriviaQA numbers anywhere in this repo, so **the entire warrant matrix is currently fixture data** — the tier ladder, the lift criterion's two refusals and six sparings, and 038's reversal are all properties of a generator we wrote. Each is internally valid and none is evidence about a language model.

Until that changes, the only thing standing between a fixture number and a slide is somebody remembering. That is precisely the control this project argues is insufficient, applied to itself.

**Decision:** `DistributionEnvelope` carries `data_source`, and the `RESULTS.md` renderer **refuses to print any number** from an envelope that is not explicitly `measured`. The cell renders `FIXTURE — NOT MEASURED` and the value is absent — not greyed, not footnoted. A number that is not on the page cannot be read off it.

Carried on the envelope rather than looked up in a registry because `SPEC.md` §1.5 requires records be self-describing enough to interpret a year later without this codebase, and *"was this a real distribution or a generator?"* is the first question a reader of an old warrant needs answered.

**Fail-closed in both directions.** An envelope that does not declare a `data_source` at all is refused too, because the realistic failure is a new code path that forgets the field rather than one that sets it wrong. The first implementation defaulted the matrix table to `"measured"` when the attribute was absent, which meant a legacy record printed its numbers — caught by `test_an_envelope_without_a_declared_source_is_refused`, which is the test existing precisely because the guard's own default was the hole.

**Also structural:** `REQUIRED_EXTRACTIONS` lists `triviaqa-600` and `triviaqa-longctx-600` with what each blocks, and the renderer prints an **Outstanding measurement** section until they land. A hard dependency tracked only in a plan is a dependency that slips; this one is in the artifact a reader opens first, and the section disappears by itself once the extraction exists.

**Consequences:** `RESULTS.md` currently opens with a warning that 8 of 11 populated cells are fixtures and prints three real rows — the `pii-reference` measurements on the hand-built sets. That is an accurate description of where the project is, and it will improve by measurement rather than by editing.

---

## 047 — Lift is reported with its ceiling, because 1.0 is a base-rate-dependent bar
**Status:** accepted · **refines 043**

**Context:** Entry 043 introduced `MIN_LIFT_LOWER_BOUND = 1.0` and noted in passing that lift compresses toward 1 on enriched eval sets. That caveat is more consequential than a note: it means the criterion is stated as absolute and behaves as **base-rate-dependent**, so a genuinely strong detector on an enriched envelope can sit near the floor.

**The derivation.** Flagging a fraction `f` of items when a fraction `b` are positive caps true positives at `min(f, b)`, so

```
R    <= min(f, b) / b
lift  = R / f  <=  min(1/b, 1/f)  =  1 / max(b, f)
```

On `hinglish-pii-200`: base rate 0.51, measured flag rate 0.62, so the ceiling is **1.61**. The measured lift of 1.28 is therefore **79% of everything achievable**, not "barely better than chance" — which is how 1.28 reads with no ceiling beside it.

**Decision:** `WarrantMetrics` carries `base_rate`, and `lift` reports its ceiling and its fraction of that ceiling. The renderer prints them together, the same way precision and recall travel together (invariant 5) and for the same reason: each is misleading alone.

**Alternatives rejected:** *Make the bar base-rate-relative — refuse below some fraction of the ceiling* — attractive, and it would replace a criterion derived from a definition with one derived from a taste. 1.0 remains the definition of "no better than chance at this cost"; how far above it you require is a policy judgement and belongs in a profile's declared minimum. *Leave it as a caveat in prose* — the number that needs the caveat is printed in a table, and prose three sections away is not attached to it.

**Consequences:** every rendered lift now carries `— N% of the X ceiling at base rate Y`. A reviewer comparing lift across envelopes with different base rates gets the comparable number rather than the raw one.

---

## 048 — Shape compatibility between the fixture path and the real extraction
**Status:** accepted · **the check is written; it runs when the GPU extraction lands**

**Context:** Removing the duplicate metrics implementation (044) closed the "two implementations of one quantity" risk. A different one remains and the grep pass could not have found it: the fixture path and the real extraction path will produce metrics that *should* be comparable and might silently not be, because they differ in normalisation, split derivation, or label polarity. Values must differ. **Shape must not.**

**Decision:** `assert_metric_shape_compatible(first, second)` asserts the same metrics are present and the same ones absent, that each shared metric has identical `kind` and `unit`, and that every estimated metric carries an interval on both sides. It says nothing about values. Run it on the same eval set id through both paths when the extraction lands; a mismatch means the two are not measuring the same thing and every fixture-versus-measured comparison is void.

**Exercised now, on the two paths that exist**, so it is known to work before it is needed.

**One finding from writing it:** the kind-flip axis — an `EXACT` count on one path and an `ESTIMATED` one on the other, which is the yield/rate confusion arriving through the back door — turns out to be **unconstructible** through the real types. `WarrantMetrics.__post_init__` already refuses it. The assertion's kind check is belt-and-braces, and the test records that the failure cannot be built rather than merely that it would be caught. That is the stronger position and worth knowing about explicitly.

---

## 049 — The routing tiebreak is stated in code, not left to the sort
**Status:** accepted

**Context:** Ranking by recall lower bound (042) leaves ties, and Phase 5 calls `route()` under revocation to choose the detector the system falls back **to**, live, in front of an audience.

**Decision:** the sort key is `(has_recall, -lower_bound, width, detector_id)`. The final key is explicit. Python's sort is stable, so without it the winner would depend on the order the matrix yielded cells, which depends on dictionary insertion order, which depends on the order warrants were appended to the ledger. "Whatever the sort does" is not a specification.

`test_routing_tiebreak_is_stated_not_incidental` builds the same two warrants in both orders and asserts the same winner.

**And `UNVALIDATED` cells never reach the ranking at all.** They hold no warrant, so `valid_warrants()` cannot return one, and a cell with no measurement can never outrank one with a measurement however promising it looks. That is invariant 2 at the ranking layer: an absence is not a weak positive. `test_an_unvalidated_cell_never_outranks_a_measured_one` pins it, including that the unmeasured cell is still enqueued for validation rather than ignored.

---

## 050 — The failure class: absence reading as presence
**Status:** accepted · **names the class behind 024, 032, 035, 046, 053-057, 062, 064, 066 and others**

**Context:** The same bug has now appeared four times in four costumes, and each time it was caught by a different accident. Naming the instances is not enough, because the fifth occurrence will look like none of them. Naming the *class* is the only thing that generalises.

**The fifth arrived and the prediction held**: it was not in the code at all, but in how a cause was diagnosed. See the row and the rule added below.

**The class:** *a missing value silently rendering as a positive one.* Not a wrong value — a **missing** one, converted into a claim by a default, a fallback, or a collapsed state. It is the exact inverse of what this product sells, which is why it keeps appearing here specifically: every layer of this system exists to distinguish "we measured this" from "we did not", and every layer therefore has a place where that distinction can be quietly lost.

**The four instances so far:**

| where | the absence | what it silently became |
|---|---|---|
| `WarrantStatus` (024) | never validated on this envelope | `VALID`, if `UNVALIDATED` collapsed into it |
| single-class envelopes (032) | no positives, so no recall | a recall number computed from an empty denominator |
| boundary intervals (035) | zero events observed | `[0.0000, 0.0000]` — perfect certainty |
| `data_source` (046) | field not declared | `"measured"`, via a `getattr` default |
| **debugging method** (053, 055, 056, 057) | the alternative causes were never computed | the one candidate that was computed, read as identified |
| **config enumeration** (062) | `last_token` implemented but not listed in `aggregations` | the two listed options, read as the complete set of options |
| **shell exit codes** (064) | `pytest ... \| tail` reports *tail's* status | a check that could not fail, read as a check that passed |

**A sixth surface, and it is the inverse shape — worth separating.** The five
above are all *absence reading as presence*. This one is a **guard whose scope
is narrower than the thing it gates**, which fails open on everything outside
that scope, silently.

The canary (066) was built from one aggregation's scores. It bound
`last_token` — caught 20/20 — and did not bind `mean_pool`, which caught 15/20
and was refused on a control that had nothing to do with its quality. The
control ran, reported a number, and passed or failed; nothing was absent. It was
simply *measuring a different thing from the one it was gating*.

That is the surface most likely to recur as detectors are added: every new
detector arrives outside the scope of every control written before it, and a
control that binds narrowly does not announce what it is not covering. The
canary now records `variants_required_to_catch` and
`variants_excluded_below_auroc_floor` in its construction, so its scope is
readable rather than implied.

The rule: **a guard must declare what it binds, and the declaration must be
checkable against what it gates.** "It passed" is not information until you know
what it was looking at.

**Three of these are in the process column, not the code column**, and that is
why the first three rules did not catch them. Two deserve separate mention.

*Config enumeration* is its own surface. `last_token` existed in
`aggregation.py`, with a comment naming it as the anchor Round 1 was measured
on, and was absent from `config.yaml`'s `aggregations`. **An unlisted option is
not an error** — nothing can raise, because a shorter list is a valid list. So
every cross-round comparison in this repo ran between two different detectors
for an entire phase, and the code that would have made it valid was sitting
there named. The absence of a declaration read as a complete declaration.

*Shell exit codes* is the cheapest instance and the easiest to repeat:
`python -m pytest ... | tail -6` exits with **tail's** status, not pytest's, so a
red suite reports success. A gate that cannot fail is not a gate.

On the debugging-method row: Four GPU sessions were lost to the same
move: compute one candidate quantity, find it the right order of magnitude
against the observed failure, and stop. The attention term matched a 10.93 GiB
OOM at 10.97 GiB; the logits at the same length were 12.32 GiB and matched at
least as well, and were never computed. *Absence of a check on the alternatives
read as presence of a verdict.*

The rule that generalises it: **a quantity that matches the observation is not
the cause until the alternatives have been computed too.** And where the code
path is knowable by execution, execute it — intercepting
`scaled_dot_product_attention` settled in under a minute a question that two
weeks of arithmetic had answered wrongly twice (057).

The fourth is the instructive one: **it appeared inside the guard built to prevent the class.** `getattr(envelope, "data_source", "measured")` was written while implementing the refusal, and it made an undeclared envelope print its numbers. It was caught only because the test was written for the legacy-record case rather than the happy path.

**Decision — four rules, applied wherever a value can be missing:**

1. **Default to the refusing value, never the permissive one.** `getattr(x, "field", None) != "measured"` rather than `getattr(x, "field", "measured") != "measured"`. If the answer is unknown, the answer is no.
2. **Represent absence as absence, not as a record with empty fields.** `UNVALIDATED` is a cell with no warrant, not a warrant with `None` metrics; a single-class envelope has `recall=None`, not `recall=0`. An absence cannot be dereferenced into a number by accident; a zero can.
3. **Compute the alternatives before naming a cause.** A plausible match is a
   hypothesis, not a finding, and one computed quantity next to several
   uncomputed ones is an absence dressed as a verdict. Where the behaviour can
   be executed rather than reasoned about, execute it.
4. **Write the test for the absent case first.** The happy path is what gets written by default and the absent path is what gets forgotten, so the test for "what happens when this is missing?" is the one that finds anything.

**A checklist for review**, since this is the shape to look for rather than a specific line:

* any `getattr(..., default)` or `dict.get(..., default)` where the default is the permissive branch;
* any `or` fallback on a value that could legitimately be zero, empty or `False`;
* any place a `None` is coerced to a number before being rendered;
* any enum where one member means "no information" and is compared with `!=` against a specific other member rather than `is` against itself;
* any interval, count or rate produced from an empty or single-class sample.

**A reviewer could fairly object** that three of the four instances were caught, so the existing controls work. Two were caught by tests written for them; one was caught by a hunch about an unrelated symptom; one was caught by a test written for a case that seemed unlikely. That is not a control, it is a hit rate, and the point of naming the class is to convert the hit rate into a thing to look for.

---

## 051 — The long-context envelope covers the test split only
**Status:** accepted

**Context:** `triviaqa-longctx-600` is what Beat 4 turns on. The obvious implementation extracts long-context activations for all 2,400 items so `run_ablation` can fit a fresh probe per envelope, exactly as it does for every other cell.

**Decision:** extract long-context activations for the **600 test questions only**, and score them with the probe fitted on short context via `validate_transferred`.

**This is a measurement decision, not a saving.** Refitting on long context answers *"how well can a probe do on long-context data?"*. Beat 4 asks *"what happens to **this** probe when the traffic changes underneath it?"* Those are different experiments, and only the second is the drift story — nobody retrains between one request and the next, so a refitted number describes a system that does not exist.

The cost difference is real and secondary: 2,400 sequences at 4–16k tokens against 600, roughly four hours against one on a T4, on a card where a 16k-token sequence at 7B NF4 does not batch at all. Had the cheaper option also been the wrong measurement, we would have paid for the right one.

**What the matrix cell means as a result.** `(probe-T1-mean_pool, triviaqa-longctx-600)` is now *"the probe fitted on short context, evaluated on long-context traffic"*, which is what a production envelope violation looks like. That is a narrower claim than the other cells in the same row and the warrant records it: `validate_transferred` carries the source run's controls with their detail amended to say they were established on the source extraction, and the split counts read `train: 0, validation: 0`.

**Alternatives rejected:** *Extract both and report both* — defensible, twice the GPU cost, and it invites the reader to compare a transferred number with a refitted one as though they were the same quantity. *Refit only* — measures the wrong thing at four times the price.

**A reviewer could fairly object** that a transferred warrant rests on controls that were not re-run on the new envelope. Correct, and it is stated in the record rather than glossed: the negative controls and the padding evidence describe the *fitted probe and the extraction that produced it*, both of which are unchanged by the transfer. What is not carried is anything describing the new distribution — the envelope, the metrics and the base rate are all measured on long context.

---

## 052 — Round 1's measured bundle, and what it changed in the Round 2 extraction
**Status:** accepted

**Context:** A Round 1 results bundle was produced on 2026-08-23, on commit `1284c8b`. Comparing it against the committed Round 1 results: **0 of 11 measured quantities moved** — AUROC 0.855141, base rate 0.388333, flag rate 0.061667, recall 0.141631, precision 0.891892, lift 2.296717, and the confusion matrix (33, 4, 200, 363) are all bit-identical. Only provenance differs. That is the reproducibility property demonstrated across sessions rather than asserted, and it is worth recording as a result in itself.

**What it does not contain:** activations. The caches are gitignored and were not bundled, and Round 1 extracted only the last prompt token, so they could not have supplied Round 2's `mean_pool` and `max_rolling_means` anyway — those need the full sequence. The Round 2 extraction is still required.

**Four things it changed**, each derived from a measured number rather than an estimate:

**1. The single-layer forward hook.** Round 1 records 28 layers and hidden size 3584. `output_hidden_states=True` returns all `L+1` tensors and the probe uses exactly one, so at a 16k-token sequence that is **3.10 GB materialised to use 0.11 GB**. On a 16 GB card, weights (4.5) + KV cache (0.85) + hidden states (3.10) is 8.45 GB before attention workspace, at the length where workspace is largest. Extraction now attaches a forward hook to the single decoder block instead: 5.46 GB. The `hidden_states` indexing convention is preserved, because the layer was chosen by fractional depth against it.

**2. Runtime estimates replaced with measured throughput.** The notebook claimed 1–1.5 h short and 45–60 min long, which was a guess. Round 1 measured 3,000 examples in 7,744 s — 0.387 examples/s, generation 2.37 s/item against prefill 0.31 s. So the short pass is generation-bound at roughly 1.7 h for 2,400, and the long pass is prefill-only (it reuses the short pass's answers, being the same questions) at 4–16k tokens, which is far slower per item than Round 1's 192. Budget 3–4 h.

**3. Strict exact match recorded alongside lenient.** Round 1 measured lenient accuracy 0.594 against strict EM 0.106 on the *same generations* — a gap of **0.488**, larger than any effect this project reports. Lenient is what labels the probe, because "Homer wrote the Iliad" is a correct answer to "Who wrote the Iliad?" and strict EM calls it wrong. But a base rate quoted without saying which rule produced it is uninterpretable, and comparing one against the other is meaningless. Round 2 now records both, as Round 1 did.

**4. The fixture's base rate is optimistic and the real ceiling is tight.** The synthetic fixture uses 0.152; the real test split is **0.388**. Since the lift ceiling is `1 / max(base_rate, flag_rate)` (`DECISIONS.md` 047), the fixture implies a ceiling near 6.6 while the real one is **2.575**. Round 1's measured lift of 2.297 is therefore **89% of everything attainable**, not a modest result — and `MIN_LIFT_LOWER_BOUND = 1.0` has correspondingly less room above it than the fixture suggests. Round 1's lift lower bound of ~2.0 clears the bar comfortably, so the criterion does not threaten the real anchor; but any tier that loses half its recall on the real base rate lands near the floor, which is a live possibility for the long-context cells.

**A reviewer could fairly object** that Round 1's ceiling formula (`1 / base_rate`) and Round 2's (`1 / max(base_rate, flag_rate)`) differ. They agree wherever the base rate binds, which is Round 1's regime (0.388 > 0.062). Round 2's form is the general one and reduces to Round 1's whenever the flag budget is slack.

---

## 053 — Long context needs the memory-efficient attention backend, not a smaller band
**Status:** accepted

**Context:** The first real extraction completed the short pass (2,400 items, base rate 0.4667 lenient / 0.5342 strict, layer 23) and then died: `OutOfMemoryError: Tried to allocate 10.93 GiB` inside `scaled_dot_product_attention`.

**Diagnosis from the number.** `10.93 GiB` at 28 heads and bfloat16 implies a sequence near 14,500 — it is the full `heads × seq × seq` attention matrix. PyTorch's SDPA falls back to the **math** backend whenever an explicit attention mask is present, and math materialises that matrix. The memory-efficient backend is O(seq) and is supported on Turing (sm_75), which a T4 is; Flash-Attention-2 needs sm_80 and does not apply.

So the fix is the backend, not the band. Narrowing `pad_tokens` to 8,000 would also have worked (3.34 GB) and would have weakened the envelope shift Beat 4 depends on, to work around a two-line configuration problem.

**Decision:** `efficient_attention()` wraps every extraction forward pass, selecting `EFFICIENT_ATTENTION` with a `MATH` fallback, across both the torch ≥ 2.3 and 2.0–2.2 API spellings. `pad_tokens` stays at `[4000, 16000]`.

**Three consequences, one of which is a process fix:**

- **Checkpoint before the fragile step.** `extract_triviaqa` now saves the short-context eval set and cache *before* long context runs. The short pass is the expensive half — generation at 2.37 s/item — and the OOM discarded 17 minutes of completed work that had no reason to be at risk. A failure in the long pass is now recoverable without re-running the short one.
- **Adaptive retry.** On OOM the batch halves to 1, and at 1 it raises naming the sequence lengths that failed and the three levers, because "CUDA out of memory" alone does not say whether to change the backend, shorten the band, or free the card.
- **Spread the weights.** With two T4s, `device_map="auto"` put all 5 GB of NF4 weights on card 0 and left card 1 idle, since the model fits. `load_model` now caps per-GPU memory so weights split and each card keeps headroom for the attention workspace.

**A reviewer could fairly object** that the checkpointing is scaffolding rather than method. It is, and it is recorded because the failure it prevents already happened once and cost real GPU time.

---

## 054 — Measured base rate on the real extraction: 0.4667 lenient, 0.5342 strict
**Status:** accepted

**Context:** First real Round 2 extraction, 2,400 TriviaQA questions through Qwen2.5-7B-Instruct NF4 at layer 23.

**Measured:**

```
base rate  0.4667 lenient   0.5342 strict EM   gap 0.0675
match rules: substring 1240, exact-token-on-short-alias 40,
             no alias matched 1119, empty generation 1
7,200 distinct questions after dedup (0 near-duplicates dropped)
```

**Two things worth recording.**

The **lenient/strict gap is 0.0675**, against Round 1's 0.488 on the same dataset and model. Round 1's strict rule was stricter than this one — it required the generation to *be* the answer, where this one normalises first — so the two gaps are not comparable, and the useful reading is that Round 2's lenient rule is close to its strict rule rather than that the model improved.

The **base rate is 0.4667**, against Round 1's 0.388 and the synthetic fixture's 0.152. Since the lift ceiling is `1 / max(base_rate, flag_rate)` (`DECISIONS.md` 047), the real ceiling here is **2.14** — tighter than Round 1's 2.575 and far tighter than the fixture's 6.6. `MIN_LIFT_LOWER_BOUND = 1.0` therefore has less room above it than any fixture run suggested, and a tier that loses recall on long context may land near the floor. That is a real possibility for the long-context cells and is not a reason to move the bar.

**Also noted:** the dedup dropped 0 near-duplicates from 7,200 questions, where Round 1 dropped 7,983 from 17,944. Different pass — Round 1 deduplicated the whole split, this one stops once it has three times the questions it needs — so the counts are not comparable and 0 is not evidence that near-duplicates are absent.

---

## 055 — Selecting the efficient attention backend is not enough; the mask has to go
**Status:** accepted · **corrects 053, which fixed half the problem**

**Context:** Entry 053 wrapped extraction in `efficient_attention()` to select SDPA's memory-efficient backend. The long-context pass failed again with the *same* allocation, 10.93 GiB, plus a second error on top:

```
OutOfMemoryError: Tried to allocate 10.93 GiB
During handling of the above exception, another exception occurred:
RuntimeError: generator didn't stop after throw()
```

**Two bugs, and the second hid the first.**

**The context manager swallowed the OOM.** It wrapped `yield` in `except (ImportError, AttributeError, RuntimeError)` so it could fall back across torch API versions. `torch.cuda.OutOfMemoryError` **subclasses `RuntimeError`**, so an OOM in the body was caught by the fallback handler, which then yielded a second time. That produced `generator didn't stop after throw()`, replaced the real error with a confusing one, and defeated the retry logic that was watching for `OutOfMemoryError`. Fixed by *constructing* the manager inside the try and running the body outside it.

**Selecting the backend does not use it.** The memory-efficient kernel **declines float attention masks** and falls back to math silently. Transformers builds a 4D float mask whenever `attention_mask` is passed — and at batch size 1 that mask is all ones and carries no information at all, because a single sequence is never padded. Dropping it lets transformers take the `is_causal` path, which the efficient kernel handles natively.

```
seq 14,500:  math 10.97 GB   efficient 0.39 GB
seq 16,000:  math 13.35 GB   efficient 0.43 GB
```

**Decision:** at batch size 1, `attention_mask` is removed from the forward inputs and the all-ones mask is reconstructed locally for pooling. `pad_tokens` stays `[4000, 16000]`.

**Also set** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the notebook's pre-flight, before the first CUDA allocation. Long-context extraction allocates and frees large tensors per item, and the default allocator fragments under that pattern — a forward can fail for want of a contiguous block while plenty of total memory is free.

**What this cost:** two failed GPU sessions. The first was a real diagnosis with an incomplete fix; the second failed for the same reason wearing a different error. The lesson is narrow and worth stating: *requesting* a backend is not *using* one, and a fallback that is silent by design needs a check that it did not fire.

**A reviewer could fairly object** that none of this is verifiable without a GPU. True — the context-manager bug is now pinned by `test_efficient_attention_propagates_body_exceptions` with `RuntimeError` explicitly in the list, because that is the one that bit. The mask behaviour cannot be tested on CPU and is verified only by the next session completing.

---

<!-- New entries below. Do not edit anything above this line. -->

## 056 — The logits were the larger allocation, and 053 and 055 both missed them

Third failed GPU session on the same stage. 055's fix worked as far as it went:
the OOM propagated as an OOM rather than as `generator didn't stop after
throw()`, and the retry ran. It still ran out of memory at batch size 1.

The forward pass called `Qwen2ForCausalLM`, which applies `lm_head` at **every**
position and then upcasts to float32 with both copies alive. At Qwen2.5-7B's
vocabulary of 152,064 that is `seq x 152064 x 6` bytes:

| seq | logits | attention (math backend) |
|---|---|---|
| 8,000 | 6.80 GiB | 3.34 GiB |
| 11,103 | **9.43 GiB** | 6.43 GiB |
| 14,500 | **12.32 GiB** | 10.97 GiB |

The session reported 10.4 GiB free on card 1. Logits alone exceed it at 11,103
tokens regardless of what attention does.

The forward now runs the transformer trunk (`model.model`), which produces no
logits. The probe reads one hidden state from layer 23; the vocabulary
projection was computed and discarded in full.

**Why two diagnoses missed it.** 053 computed the attention term as 10.97 GiB at
seq 14,500, compared it against the reported 10.93 GiB, and treated the match as
identification. The two terms are the same order of magnitude across the whole
band, so at 14,500 the logits (12.32 GiB) fit the observation at least as well.
A quantity that matches the observed value is not the cause unless the
alternatives were also computed. Both were needed; neither alone sufficed.

The OOM message carried the same error and propagated it to the next reader: it
named only attention and listed "confirm the efficient backend" as the first
remedy. It now computes both terms from the model's own config and prints peak
allocated memory per card, so the next failure is read off a measurement rather
than inferred.

Rejected: narrowing `evalsets.pad_tokens` to `[4000, 8000]`. It would have made
the pass fit and left both real causes in place, and the band is what produces
the envelope shift Beat 4 rests on.

Not verifiable on CPU: `test_forward_runs_the_trunk_not_the_causal_lm` asserts
the wrapper never runs, which pins the code path. Whether the pass now fits in
16 GiB is confirmed only by the next session completing.

## 057 — Supersedes 055: the mask fix was a no-op, and 053 and 056 were argued rather than measured

Three GPU sessions were lost on this stage. All three diagnoses were produced by
computing one candidate quantity, finding it the right order of magnitude, and
stopping. This entry records what measurement showed and retires the claims.

**055 was a no-op.** It asserted that passing an attention mask forces a 4D float
mask, which SDPA's memory-efficient kernel declines, falling back to math. That
is true of a *padded* batch and false of an all-ones one:
`masking_utils._ignore_causal_mask_sdpa` returns True when `padding_mask.all()`.
Intercepting `scaled_dot_product_attention` on a real Qwen2:

| call | reaches SDPA as |
|---|---|
| all-ones mask passed | `attn_mask=None, is_causal=True` |
| mask omitted | `attn_mask=None, is_causal=True` |
| batch 2, real padding | `attn_mask=(2,1,32,32), is_causal=False` |

Identical. The mask-drop is kept as version insurance, demoted in the comment,
and pinned by `test_all_ones_mask_is_already_skipped`.

**053 was aimed at a term that was never paid.** With sdpa there is no `seq x seq`
tensor at all — confirmed by enumerating every tensor in a forward pass and
finding none square. The attention arithmetic in 053 described the eager path,
which was not in use.

**056 identified the right term but by the same faulty method.** The logits are
real: counting bytes through `TorchDispatchMode`, the causal LM allocates 5.4x
the trunk at seq 1024, and its largest tensor is `(seq, vocab)` against
`(seq, intermediate)` — an 8x ratio at the real model's 152,064 vocabulary,
before the float32 upcast. The fix stands. The reasoning that produced it did
not, and would have been wrong again next time.

**What actually changed, beyond the trunk fix:**

- `load_model` now passes `attn_implementation="sdpa"` explicitly and **refuses
  to load** on anything else. This was the one live risk left: the default is
  right on the machines checked, and eager would cost `28 x 16000^2 x 4` = 28.7
  GiB for a single op — the failure being chased, arriving by a path nobody had
  ruled out. A default that happens to be right is not a guarantee.
- `_preflight_longest` runs the **longest prompt alone** before the 600-item
  loop and logs measured peak memory per card. The worst case is not item 0, so
  every previous failure cost 40 minutes before revealing anything. It now costs
  seconds and reports a number instead of requiring one to be inferred.
- The float32 upcast in the capture hook moved off the GPU (229 MB at 16k).

**The rule this stage earns:** a quantity that matches the observed value is not
the cause until the alternatives are computed too. Where the code path is
knowable by execution, execute it — three of the four claims above were settled
in under a minute by intercepting a function, after two weeks of arguing them.

## 058 — Chunked prefill: stop depending on which attention kernel gets selected

Fourth failed session. The pre-flight from 057 worked and produced the first
real evidence of the run:

```
cards {0: '12.7 free of 14.6 GiB, peak 2.9', 1: '10.4 free of 14.6 GiB, peak 3.9'}
```

**Peak allocation never exceeded 3.9 GiB while 10.4 GiB was free.** This is not
gradual exhaustion; a single request failed outright, and was large enough to
fail on a card with 12.7 GiB free. The fp32 softmax over a `seq^2` score matrix
is `28 x 11103^2 x 4` = 12.86 GiB, which fits that shape and never enters the
peak because it never succeeded.

**The message discarded the evidence.** `raise ... from None` threw away torch's
own text, which states the requested size and device — the one number that would
have identified the allocation without inference. Now chained, with the first
lines of the original quoted above the reference figures.

**The fix is to stop needing the answer.** Prefill now runs in chunks carrying a
KV cache, bounding the attention workspace to `heads x chunk x seq`:

| | unchunked | chunked at 2048 |
|---|---|---|
| attention workspace at seq 11,103 | 12.86 GiB | 2.55 GiB |
| KV cache | — | 637 MB |

That bound holds on the eager path, the math backend and the efficient backend
alike. Four sessions were spent reasoning about which kernel would be selected;
this makes the answer not matter.

Exact, not approximate: causal attention means a cached prefix produces the same
hidden states as one pass. Measured at 1.1e-08 max absolute difference across
two chunk sizes, and pinned by `test_chunked_prefill_matches_a_single_pass`.

**What chunking does not do.** It does not beat the best backend. Against sdpa
it slightly *raises* the largest tensor, by materialising a `heads x chunk x seq`
block the efficient kernel avoids entirely. The first version of
`test_chunking_bounds_the_attention_workspace` asserted a reduction against sdpa
and failed, correctly. It now measures against eager, where the `seq^2` matrix
is real, and asserts the bound rather than a win.

Cost: `model.prefill_chunk_tokens` added to config, moving the config hash from
`c89257bc4adc10c2` to `bfb92a89ceacd678`. Committed artifacts keep the old hash,
which is what provenance is for — they were produced under that config.

Also: the pre-flight now logs the attention implementation, because re-running a
single notebook cell against a live kernel keeps the model object from the
original load. A fix applied in `load_model` is not present in a model that was
loaded before it.

## 059 — Beat 4 measured: both aggregations fail the envelope shift, in opposite directions

The first measured extraction (2400 questions, 600 long-context, 3.2 h on 2x T4)
transferred to the long-context envelope. Probe fitted on `triviaqa-600`, scored
unchanged on `triviaqa-longctx-600`; nothing refitted, nothing reselected.

| variant | AUROC source | AUROC target | flag rate | lift |
|---|---|---|---|---|
| `T1-mean_pool` | 0.785 [0.750, 0.821] | 0.502 [0.455, 0.548] | 0.040 -> 0.000 | 1.99 -> n/a |
| `T1-max_rolling_means` | 0.785 [0.750, 0.821] | 0.555 [0.511, 0.602] | 0.042 -> 0.543 | 1.99 -> 1.08 |

Base rate 0.4617, lift ceiling 2.166. On the source envelope both reach 92% of
that ceiling. On the shifted envelope both fail, and the failures are not alike:

- `mean_pool` **fails silent**. Scores drop below the frozen threshold, nothing
  is flagged, and the system reports clean traffic.
- `max_rolling_means` **fails expensive**. Scores inflate, 54.3% of traffic is
  flagged (13x the budget) at precision 0.497 against a base rate of 0.462 —
  random sampling with a detector's name on it.

`max_rolling_means` was built specifically to survive long-context shift
(CLAUDE.md's silent-failure list). It does not. It degrades differently, not
better, and its lower CI of 0.511 still falls short of the 0.55 floor. Reporting
only the aggregation that was supposed to work would have left the claim
untested exactly where it matters.

Both warrants REFUSED, which is the layer doing its job.

**Caveat that must travel with these numbers.** Every measured warrant is also
refused on `canary`, which fails *closed* on an absent set. There is no TriviaQA
canary — `evalsets/canary-20-pii.json` is Presidio's. That refusal is correct
behaviour and says nothing about the probe; only `auroc_lower_ci` on the
long-context envelope is a measurement. Building the canary is outstanding.

## 060 — Three ways the deliverable hid its own measurements

Found while wiring the measured extraction through validation. Each let a
measured result read as something else.

**`02_validate` refused measured eval sets** with "the registry that turns it
back into an EvalSet does not exist yet". `load_evalset` has existed at
`src/evalsets/registry.py:82` since the eval sets were first frozen; the guard
was stale. This is what stopped the Kaggle run after 2.6 h of successful
extraction. Now wired, with a hash check refusing a cache/eval-set mismatch —
pairing a cache with the wrong eval set would attach real activations to someone
else's labels.

**The matrix rendered rows that could not match the measured warrants.**
`03_matrix` built its detector list as `probe-{variant}` from the fixture
ladders; `02_validate` writes `probe-{model}-{variant}`. So `triviaqa-600`
displayed UNVALIDATED while a REFUSED warrant sat in the ledger — a measured
refusal shown as "never tested", which is the one reading the matrix exists to
prevent. Now `declared UNION ledger`.

The first attempt at that fix read `detector_id` off the `LedgerRecord`. The
column exists in SQL; the attribute does not. The script raised, and left the
previous `warrant_matrix.md` in place — so the output looked unchanged rather
than broken. Read off the warrant now.

**Transfer warrants were never written to the ledger**, so Beat 4 could not
appear in the matrix at all. Also mine: `04_transfer` used a bare
`probe-{variant}` id, putting the same measurement in the matrix twice under two
names. Unified on the model-qualified id, because a probe reads one specific
model's residual stream and a model change invalidates the warrant.

## 061 — A canary for the activation probe, and the first VALID measured warrants

`canary_control` fails **closed** on an absent set, and the only canary in the
repo was Presidio's `canary-20-pii`. So every measured activation-tier warrant
was refused on that control alone — a refusal that said nothing about the probe
and buried the one that did (059).

`scripts/05_canary.py` freezes `canary-20-triviaqa`: 20 known-incorrect items
from the **train split only**, chosen by the probe's own score, with their
activations sliced out of the existing cache. No GPU.

**What it proves and does not.** Items are chosen by the current probe, so the
current probe catches them by construction. Circular as a measurement, correct
as a tripwire: it detects *change* — a moved threshold, a reordered dataset, a
swapped model — not quality. It is not independent evidence the probe works, and
the construction record says so.

Train only: selecting on validation or test would be selection on the splits
those exist to protect.

Two bugs found while building it, both mine:

- The first version fitted with `C_grid[0]` while the validated run selects
  `C=0.001` on validation. A canary chosen by a probe nobody runs is a tripwire
  for the wrong thing. Now selects C the same way `validate()` does.
- It froze without checking the canary could pass. A pre-tripped canary refuses
  every warrant forever and says nothing — the exact failure it exists to end.
  It now refuses to freeze one whose items do not clear the operating threshold.

I also raised a false alarm in between: I compared scores from a C=0.0001 probe
against the threshold belonging to the C=0.001 probe and concluded the canary
could not pass. Measured properly, all 20 clear both thresholds.

**Result.** All five controls pass and the first VALID measured warrants are
issued:

| detector | triviaqa-600 | triviaqa-longctx-600 |
|---|---|---|
| `probe-...-T1-max_rolling_means` | VALID R=0.08 [0.05, 0.12] | REFUSED |
| `probe-...-T1-mean_pool` | VALID R=0.08 [0.05, 0.11] | REFUSED |

Same detector, same operating point, two envelopes, opposite verdicts. The
refusals are now on merit:

- `mean_pool`: `auroc_lower_ci 0.4546, required > 0.55`
- `max_rolling_means`: `auroc_lower_ci 0.5105` **and** `lift_lower_ci 1.076
  [0.969, 1.182] at flag rate 0.5433, required > 1.0`

The lift interval straddles 1.0. That criterion caught a detector flagging 54%
of traffic that is not demonstrably better than random sampling at the same
budget — which is what it was added for, and it fired on the first real
measurement rather than on a fixture.

## 062 — Reconciling R=0.08 against Round 1's published 0.1416

Round 2's first measured recall reads 0.0794 [0.050, 0.113] against a Round 1
handover, in a public repo, claiming 0.1416. Reported alone that is a regression
notice. Reported with its companions (invariant 5) it is mostly not one.

| | Round 1 (last_token) | Round 2 @ f=0.05 | Round 2 @ Round 1's f |
|---|---|---|---|
| measured flag rate | 0.0617 | 0.0400 | 0.0550 |
| recall | 0.1416 | 0.0794 | 0.1119 |
| precision | 0.8919 | 0.9167 | 0.9394 |
| base rate | 0.3883 | 0.4617 | 0.4617 |
| lift | 2.297 | 1.986 | 2.035 |
| ceiling `1/base` | 2.575 | 2.166 | 2.166 |
| % of ceiling | 89.2% | 91.7% | **94.0%** |
| test AUROC | 0.8551 | 0.7853 | 0.7853 |

Three separable causes, largest first.

**1. The operating point moved.** Round 1's threshold landed at a measured
f=0.0617 against a 0.05 target; Round 2's lands at 0.0400. Recall scales with
budget, so most of the apparent drop is a smaller budget. Re-thresholded to
Round 1's flag rate, Round 2 recalls 0.1119 — about 71% of the distance from
0.0794 to 0.1416 closes on this alone.

**2. The base rate rose, so the ceiling fell.** `lift = precision / base_rate`,
so a base rate of 0.4617 against 0.3883 caps lift at 2.166 rather than 2.575.
Measured against what was attainable, Round 2 reaches **94.0% of its ceiling
against Round 1's 89.2%** — better, not worse. Raw recall moves with the
benchmark's difficulty; the fraction of the ceiling does not.

**3. A genuine ranking gap remains: AUROC 0.7853 against 0.8551.** AUROC is
base-rate independent, so this one is real and neither of the above explains it.

**The most likely cause of (3) is that these are different detectors.** Round 1
read the **last token of the prompt** — invariant 1, question-time, before any
generated token. Round 2's `config.yaml` declared only `mean_pool` and
`max_rolling_means`, both pooled over every position. `last_token` was
implemented in `aggregation.py` all along, with a comment naming it "the anchor
the Round 1 number was measured with", and was simply never listed. So Round 2
has never measured the configuration Round 1 published, and the comparison above
is between two different features.

`last_token` is now in the aggregation list. It **cannot be recovered from the
existing cache** — the cache stores pooled features, not per-position hidden
states — so closing this needs one re-extraction. Until that runs, the AUROC gap
is attributed but not demonstrated, and no claim should be made that Round 2
reproduces or fails to reproduce Round 1.

Config hash moves to `b4ca1ec022266551`. Committed artifacts keep the hash they
were measured under.

**What must travel with R going forward.** Never the recall alone. Flag rate,
precision, base rate, lift and the ceiling, or the number invites exactly the
reading this entry exists to correct.

**RELEASED by 066 (branch A).** Was suspended pending 065: The 94.0%-vs-89.2%-of-ceiling framing above compares
Round 2's best *pooled* aggregation against Round 1's `last_token`. That is the
cross-detector comparison this entry itself establishes was never valid, so the
framing is not usable until `last_token` is measured on Round 2's pipeline. It
is left in place rather than deleted because the arithmetic is correct and only
the comparison is not.

## 063 — results/ is the deliverable and is measured-only

Two results directories — `results/` holding fixture artifacts and
`results/measured/` holding the real ones — is two answers to the same question
with nothing declaring which is authoritative. That is the failure this product
argues against, reproduced in the filesystem.

The `data_source` guard cannot close it. It refuses fixture *numbers* at the
field level; this ambiguity is at the directory level, one layer above anything
a field can see.

`results/` is now the deliverable: measured artifacts, one `RESULTS.md`, one
ledger. Fixture artifacts moved to `results/fixtures/`. They are regenerable
from a seeded generator and nothing downstream reads them.

`results/` was never purely fixture, which is why "pick one" would not have
worked: `validation-pii-reference-*` and `evalset_*` are measured, on
hand-written eval sets, and belonged in the deliverable all along. Only the
`*-fixture*` artifacts moved.

`test_results_is_measured_only` fails if a fixture artifact appears at the top
level or if `results/measured/` returns.

## 064 — Universal refusal is a pipeline-bug signature, not a finding

Four cells read REFUSED in one session for reasons unrelated to any detector: a
stale guard in `02_validate`, a naming mismatch rendering REFUSED as
UNVALIDATED, transfer warrants never reaching the ledger, and a missing canary
refusing everything. Every one produced **conservative-looking output that reads
as the system working**, which is why looking at it found none of them.

Same signature as the `fpr_hard_negatives` recurrence (040): a fix aimed at one
instance when the cause was shared. The routing positive control was aimed at
one instance too.

`test_universal_refusal_is_treated_as_a_pipeline_bug` generalises it, in two
parts:

1. across the whole matrix, at least one populated cell must reach VALID;
2. **on the measured envelopes specifically**, at least one cell must be
   populated at all.

The second part is not redundant. Fault-injecting the naming mismatch — every
measured cell UNVALIDATED, fixture cells untouched — passes part 1 and fails
part 2, which is exactly how that bug presented. Both parts were confirmed by
injecting the fault and watching the test fail, because a check that passes
whatever it is fed proves nothing (the padding control's own argument).

## 065 — PRE-REGISTERED: the last_token re-extraction, and what each outcome means

**Written before the run.** The interpretation is fixed here so it cannot be
chosen after seeing the number. This is the case that most rewards the habit:
there is a public handover asserting 0.1416 and a Round 2 measurement reading
0.0794, and any post-hoc reading of the gap will be the flattering one.

### What runs

One extraction, `n_questions=2400` plus the 600-item long-context envelope, with
**all three aggregations in the same session**: `mean_pool`,
`max_rolling_means`, `last_token`. Same cache, same splits, same labels, same
model, same seed.

Running `last_token` alone in a later session would replace one
cross-configuration comparison with another — different sample, different
labels, different split derivation — and prove nothing about the gap. Same
reasoning that put long context in the same session as short.

Config hash `b4ca1ec022266551`.

### The two outcomes, and what each licenses

**A. `last_token` test AUROC lands near 0.855** (say within the Round 1 bootstrap
CI, [0.8217, 0.8878]).

The aggregation attribution in 062 is **confirmed**. Round 1 reproduces on Round
2's pipeline, and the gap between the rounds was that Round 2 had been measuring
pooled features against a last-token result. The reconciliation table in 062
becomes a like-for-like comparison and the 94.0%-vs-89.2% framing becomes
sayable. Nothing further is owed.

**B. `last_token` lands near 0.785**, with the pooled variants.

The aggregation attribution is **refuted**, and the gap is something else. It
does not get explained away — it becomes the next investigation, ahead of Phase
5, with these candidates and no others until they are eliminated:

- labelling: 0.4667 lenient base rate here against 0.406 in Round 1, and the
  alias-matching rules may have moved;
- split derivation: 1200/600/600 by question here against 1800/600/600;
- sample: 2400 questions against 3000, drawn by a different seed path;
- model version or quantisation: the same id can resolve to a different
  revision, and NF4 compute dtype differs between the rounds' configs.

**C. Anything else** — materially above 0.878 or below 0.75 — is neither branch
and gets its own entry before anything is claimed.

### What the declared aggregation is, and why it is declared now

The tier ladder reports all three; that is what an ablation is for. But **Beat 4
and the headline use `last_token`**, declared here, before the numbers exist.

The reason is independent of any score: `last_token` is what this method's own
invariant describes — *"activations are taken at question-time, last token of
the prompt, before any generated token exists"*. The pooled variants were added
later to probe long-context robustness. Choosing the method's stated definition
is a reason that existed before the run; choosing whichever scores highest
afterwards is selection on the test set at the level of detector architecture,
and it would be undetectable in any artifact.

**If `last_token` performs worst of the three, it is still what Beat 4 reports.**
That is the whole point of writing this down now.

### Held until this lands

The 94.0%-of-ceiling framing in 062 is **suspended**. It currently compares
Round 2's best pooled aggregation against Round 1's `last_token`, which is the
comparison this entry exists to establish was never valid.

## 066 — RESULT: branch A. Round 1 reproduces, and last_token survives the shift

The re-extraction pre-registered in 065 ran: 2400 questions plus the 600-item
long-context envelope, all three aggregations in one session, config
`b4ca1ec022266551`.

### The branch, decided by the classifier that predates the data

```
T1-last_token          test AUROC 0.825552
Round 1 CI             [0.821680, 0.887818]
margin above floor     +0.003872
```

**BRANCH A.** The aggregation attribution in 062 is confirmed: Round 2 had been
measuring pooled features against a last-token result, and every cross-round
comparison before this was between two different detectors.

**It clears the interval by 0.0039**, and sits 0.0296 below Round 1's point
estimate of 0.8551. Inside is inside and the rule was fixed in advance, but a
reviewer will compute that margin and it should not be found rather than
offered. What the result licenses is "Round 1 reproduces within its published
interval", not "Round 1 reproduces exactly".

At Round 1's measured flag rate, on Round 2's test split:

| variant | AUROC | f | recall | precision | lift |
|---|---|---|---|---|---|
| `last_token` | 0.8256 | 0.0567 | 0.1119 | 0.9118 | 1.975 |
| `max_rolling_means` | 0.7853 | 0.0550 | 0.1119 | 0.9394 | 2.035 |
| `mean_pool` | 0.7855 | 0.0550 | 0.1119 | 0.9394 | 2.035 |

**The declared variant does not win on lift.** `last_token` ranks best and lifts
worst of the three at this operating point — it flagged 34 items against 33 for
the same 31 true positives. It is still what Beat 4 reports, because that was
declared in 065 before any of these numbers existed. Identical recall across all
three is a small-integer collision at n=600, not an error: the AUROCs differ, so
the scores differ.

The ceiling framing suspended in 062 is **released**: the comparison is now
like-for-like.

### The transfer result, which changes Beat 4

| variant | source | target | target warrant |
|---|---|---|---|
| `last_token` | 0.826 [0.793, 0.857] | **0.813 [0.780, 0.845]** | **VALID** R=0.13 |
| `max_rolling_means` | 0.785 | 0.555 [0.511, 0.602] | REFUSED |
| `mean_pool` | 0.785 | 0.502 [0.455, 0.548] | REFUSED |

**The last-token aggregation survives the long-context shift almost intact**,
and holds a valid warrant on the shifted envelope at a *higher* recall than on
the source. Both pooled aggregations collapse — including `max_rolling_means`,
which was built specifically to survive it.

This supersedes the reading in 059 that "both aggregations fail the envelope
shift". That was true of the two measured then, and the one that does not fail
was absent from the config at the time — the same omission 062 records. The
correct statement is: **pooled aggregations collapse under long-context shift;
the question-time last-token activation does not.**

Beat 4 becomes the routing story with measured numbers rather than a universal
failure: three detectors, one shift, two refused, one holding, and the matrix
naming which.

### The canary had to be rebuilt, and the first version was wrong

Built on `last_token` alone, it was caught 20/20 by `last_token` and 15/20 by
`mean_pool`, refusing two of three ladder rungs on a control unrelated to their
quality — the ablation made meaningless by its own tripwire.

A canary is a property of the pipeline, not of one aggregation. Items are now
selected to clear **every** variant's threshold, ranked by the worst margin
across variants, and 05_canary refuses to freeze one any rung cannot catch.
Worst margin on the frozen set: 0.0131. All three rungs now issue.

## 067 — Recall never travels without its budget, and last_token did not improve

`last_token`'s row read recall 0.079 -> 0.126 and lift 1.906 -> 1.944 across the
envelope shift: a detector apparently performing *better* on the harder
envelope, with nothing in the table explaining why. That reading is wrong, and
the table was what made it available.

The base rate did **not** move. Both envelopes are the same 600 test questions —
the long-context set is those questions padded — so labels, positives and the
ceiling are identical:

| | base | flag | recall | precision | lift | ceiling | % of ceiling |
|---|---|---|---|---|---|---|---|
| `triviaqa-600` | 0.4617 | 0.0417 | 0.0794 | 0.880 | 1.906 | 2.166 | 88.0% |
| `triviaqa-longctx-600` | 0.4617 | **0.0650** | 0.1264 | 0.897 | 1.944 | 2.166 | 89.7% |

277 positives on both. Flagged 25 -> 39; TP 22 -> 35.

**What moved is the flag rate.** The frozen threshold spends 56% more budget on
the shifted envelope, because the score distribution shifted up. Recall rose in
proportion. Lift — the budget-normalised quantity, which is what the product
claims in — is flat.

So the honest statement is not "it improved". It is: **the ranking survived the
shift and the calibration did not.** The threshold no longer delivers the
operating point it was chosen for: 4.2% on source, 6.5% on target, against a 5%
target. A warrant that pins recall at a stated budget is claiming something the
detector no longer does on this envelope, even though its AUROC held at
0.826 -> 0.813.

That is a drift signal on the *operating point* rather than the ranking, and it
is the sharpest input Phase 5 has: the surviving detector still needs
re-thresholding on the new envelope, and nothing currently detects that.

**The rendering fix.** `flag rate` and `base rate` are now columns in
`RESULTS.md`, beside recall rather than in the lift footnote, and the matrix
cell renders `R=0.13 [0.09, 0.17] @f=0.065`. Recall alone is not a claim
(invariant 5), and the compact matrix cell was the place the omission actively
misled — the one row a reader stops on.

## 068 — The declaration ordering, as hashes rather than prose

065 is only worth anything if a reader can check it predates the data without
taking the claim on trust:

| commit | UTC+5:30 | what it fixed |
|---|---|---|
| `d99a63a` | 2026-08-27 12:28:22 | DECISIONS 065: three branches, four fallback candidates, `last_token` declared for Beat 4 |
| `598cf85` | 2026-08-27 12:28:22 | merge of the above to `main` |
| `efa1291` | 2026-08-27 12:36:03 | the branch rule as code, with Round 1's CI as unreachable module constants |
| `fb0dd5e` | 2026-08-27 17:39:52 | the result: branch A, `last_token` 0.825552 |

The Kaggle run that produced the data started at **12:52 UTC+5:30** and finished
at **15:31**, between `efa1291` and `fb0dd5e`. Verify with:

```
git log --format='%h %ad %s' --date=iso-strict d99a63a..fb0dd5e
```

The declared aggregation and the acceptance bounds were both committed before
the extraction started. `test_reconciliation_branches_are_fixed_constants`
asserts no argparse flag can reach either, so the run could not have widened its
own acceptance region even in principle.

## 069 — A warrant makes two claims: ranking and calibration, separably

`T1-last_token` transferred to the long-context envelope with AUROC 0.826 →
0.813 — ranking essentially intact — while the frozen threshold went from
flagging 4.2% of traffic to 6.5% against a declared 5% target. Those are
separable properties of a detector and nothing in the system distinguished them.

**The blind spot is shaped like the thesis.** Every warrant measures ranking
quality (AUROC, and lift derived from a threshold measured *on that envelope*).
A warrant can therefore be perfectly sound about ranking and still assert a
recall-at-budget the detector no longer delivers — a warrant making an unbacked
claim, in the exact sense this product exists to refuse. Collapsing both into one
status would have to call `last_token` on long context either sound, hiding the
budget question, or refused, discarding a ranking that demonstrably survived.

So `Warrant.calibration` carries a second claim, and the states are deliberately
not symmetric with `WarrantStatus`:

- **`DRIFTED`** — the declared target lies outside the realised flag rate's
  interval. The operating point is demonstrably not the operating point.
- **`CALIBRATED`** — the target lies inside, so drift is **not shown**. Weaker
  than "calibration held".
- **`UNKNOWN`** — no target declared, or no interval. Never a pass.

**There is no state meaning "calibration verified", and the reason is the
measurement.** At n=600, a realised 6.5% has interval [0.048, 0.085], which
covers the 5% target. So the honest reading of `last_token` on long context is
*not* "the calibration drifted" — I said that first and it overstated. It is:
the point estimate is 1.30× the target and this sample cannot tell us either
way.

**Power is measured against a deviation worth acting on, never the observed
one.** My first implementation compared the `n` needed to resolve the *observed*
gap against the actual `n`. That is backwards, and a test caught it: as an
estimate approaches its target the gap shrinks and the required `n` diverges, so
`0.0501` against a `0.0500` target reported needing n=18,282,171 and flagged
near-perfect agreement as underpowered.

`config.validation.calibration_tolerance` (0.25) declares the deviation worth
catching. A claim is underpowered when the realised interval extends outside the
band `target ± tolerance` — the sample cannot rule out a departure large enough
to act on. Detecting a 25% deviation from a 5% budget needs **n ≥ 1441**; a 10%
deviation needs 7987.

| | band | realised CI | status | underpowered |
|---|---|---|---|---|
| `last_token` @ short | [0.0375, 0.0625] | [0.0267, 0.0583] | CALIBRATED | **yes** |
| `last_token` @ long | [0.0375, 0.0625] | [0.0483, 0.0850] | CALIBRATED | **yes** |
| `max_rolling_means` @ long | [0.0375, 0.0625] | [0.5050, 0.5833] | DRIFTED | — |
| `mean_pool` @ long | [0.0375, 0.0625] | [0.0000, 0.0061] | DRIFTED | — |

**Not one measured warrant is a clean calibration pass.** Every budget claim in
the repo is either unresolved at n=600 or refuted, including on the source
envelope. `pii-reference` reads `UNKNOWN`: it declares no flag-rate budget, and
that is an absence rather than a pass. The ranking claims are supported — AUROC intervals are tight
enough to separate 0.826 from 0.785 — but the "at flag rate f" half of every
warrant is weakly evidenced, and now says so rather than being read off a point
estimate.

The matrix annotates only the exception: `CAL:DRIFTED` where the budget claim is
refuted, `CAL:n/a` where drift was not shown at an `n` too small to show it. An
unannotated cell means the claim was testable and passed.

**Consequence for Phase 5, and it is cheap.** Realised-versus-target flag rate is
a drift signal requiring no embedding computation at all — a counter and a
target. It catches *that the operating point is no longer the operating point*,
where PSI on token length catches *why the input distribution moved*. The second
is the diagnosis; the first is the alarm, and it is one division.

## 070 — PSI's 0.10/0.25 bands are not scale-free, and the monitor checks its own null

PSI is the right measure for this audience — Indian banking risk teams read it
natively (`SPEC.md` §5.2). Its bands are a credit-scoring rule of thumb, and
they are always quoted without a sample size. They are not scale-free: under the
null, a window drawn from the very distribution the reference was built from has
PSI of roughly `(k-1)/n`.

Measured, by drawing both the window **and** the reference:

| bins | window | null p95 | P(PSI > 0.10) |
|---|---|---|---|
| 5 | 200 | 0.062 | 0.004 |
| 10 | 200 | 0.114 | **0.101** |
| 20 | 200 | 0.215 | **0.760** |
| 20 | 2000 | 0.066 | 0.001 |

At twenty bins and the configured 200-request window, **three windows in four
report `MODERATE_SHIFT` on traffic that has not moved.** `SPEC.md` §5.2 says *do
not revoke on noise*; a monitor alarming at that rate gets switched off, which
is the same outcome as never building it.

**The shipped configuration is fine, and I overstated the alarm first.** The real
envelope uses 8 bins against a 2400-item reference: false-alarm rate 0.005 at a
200-window. My initial measurement used 10 bins from 600 items — a harsher
configuration than the repo actually runs — and I reported 8% before checking
what was deployed. The finding stands; the panic did not.

So `DriftMonitor` simulates its own null at construction and **refuses a
configuration whose false-alarm rate exceeds `drift.max_false_alarm_rate`**
(0.05). The band is checked against the envelope it will actually be used with,
rather than trusted because it is conventional.

**Both sides are resampled.** Holding the stored bin probabilities exact put the
false-alarm rate at 0.02 where resampling the underlying values measured 0.08 —
the reference's own estimation error is part of the noise a live window is
scored against, and a guard built on the optimistic number would pass a
configuration alarming four times as often as promised. Drawing the reference at
`n_reference` as well reproduces the empirical rate (0.101 against 0.08 measured,
0.760 against 0.79).

Same shape as 029 and 031, where a fixed negative-control band was wrong at the
sample size actually used and had to be sized from the measured null. Third
instance of the same lesson: **a threshold quoted without its sample size is a
threshold that has not been checked.**

## 071 — A PSI driven by empty bins is a claim about missing data

`ln(0)` diverges, so every PSI implementation floors empty bins at an epsilon,
and that constant silently sets the magnitude. Measured on the real envelopes —
long-context traffic against short-context bins, 9 of 10 bins empty:

| epsilon | long context (9/10 floored) | 50/50 mixed (0 floored) |
|---|---|---|
| 1e-3 | 6.14 | 1.0140 |
| 1e-6 | 12.37 | 1.0140 |
| 1e-8 | 16.50 | 1.0140 |

**The floored number moves 2.7× across plausible epsilons; the measured one does
not move at all.** Both revoke, so the *decision* is unaffected — but only one
magnitude is quotable, and "PSI 12.37" would be quoting the smoothing constant.

`PsiResult` therefore reports `bins_smoothed`, `smoothed_share` and
`driven_by_smoothing`, and the verdict's reason says so in words: the finding is
that traffic left the reference support, not that the shift measured that size.

Out-of-range values are clipped into the edge bins rather than dropped.
Discarding them would compute PSI over the surviving subset and report stability
for traffic that had left the distribution entirely — which is the failure mode
this whole module exists to catch.


## 072 - MMD is scoped out of Phase 5

`SPEC.md` 5.2 declares two drift checks. Only per-feature PSI is implemented;
MMD on embeddings is not built and is not planned before submission.

It would not have changed a decision on any envelope measured here. The shift
that matters -- short-context TriviaQA against its long-context variant -- is a
univariate token-length move so large that 9 of 10 reference bins are empty
(entry 071), and PSI revokes on it at every epsilon tried. A multivariate test
earns its place when the marginals look stable and the joint has moved; nothing
measured has that shape.

Rejected: a cheap MMD over the same univariate features PSI already uses. That
is a multivariate test in name only, and reporting it as the SPEC's check would
overstate coverage.

The gap is disclosed rather than left implicit. `config.drift.mmd_permutations`
stays declared and unused, so the trace of a specified-then-dropped check
survives. `EnvelopeMatchResult.mmd_p_value` stays `None` and is never defaulted
to a passing p-value -- a 1.0 reads as a test that ran and found nothing, which
is a stronger claim than "no test ran". Every drift certificate names the
omission in `unchecked`.

## 073 - An action may name a warrant-level trigger, not only a finding

`Resolution` required `triggering_finding_ids` for any non-`ALLOW` action. The
reasoning -- an untraceable action is an unappealable one -- is right; the
requirement was too narrow.

When a revoked warrant escalates a request, nothing was found *in* that
request. Satisfying the old rule meant constructing a `Finding`, which requires
a `Category` from an enum of PII, HALLUCINATION, INJECTION, UNSAFE, COST and
BIAS_SIGNAL -- none of which describes a distribution shift. That is a
fabricated content claim in the record built to prevent them.

So: a non-`ALLOW` action must name *a* trigger, of which there are two kinds --
content triggers (`triggering_finding_ids`) and warrant-level triggers
(`triggered_by`, e.g. `"envelope:SIGNIFICANT_SHIFT"`). Traceability is unchanged
in strength.

Rejected: a `DRIFT` member on `Category`, which would put a property of the
traffic on an axis classifying properties of a response; and exempting
`ESCALATE`, which would make the commonest non-`ALLOW` action the only
untraceable one.

## 074 - "T2 and T3 keep working through a model change" is a claim about detectors, not numbers

`SPEC.md` 5.4's mitigation, stated precisely, is weaker than it sounds.

True: T2 and T3 carry no parameters fitted to a model's residual stream, so
nothing inside them breaks. A probe's weights do break -- against a different
model they are an unrelated function, not a degraded one, which is why the
response is unconditional revocation rather than a widened interval.

Not true and not claimed: that a surviving T2/T3 warrant's *measured bounds*
still hold. Recall was measured against one model's outputs; a different model
gets different things wrong. The honest description is "the detector still runs
and its numbers were never re-measured".

So `invalidate_for_model_change` returns a record for every warrant including
the untouched ones, carrying that sentence. Returning only the invalidations
would make survival invisible and so indistinguishable from endorsement.

Same distinction as 029, 031 and 070: a number quoted outside the conditions it
was measured under is a different number, not a weaker one.


## 075 - The demo reports INSUFFICIENT_DATA, and Beat 4 has to be long enough

Wiring the real drift monitor into the demo session replaced a hardcoded
`INSIDE` / `max_psi=0.0` / `n_window=1` with a scored window, and immediately
exposed a conflict between two things already decided.

`config.drift.window_size` is 200 (SPEC.md 5.2: do not revoke on noise). Demo
streams are 2-40 events. So every demo certificate now reports
`INSUFFICIENT_DATA`.

That is the correct answer, not a degraded one. The stream is drawn from the
warrant's own test rows, so the traffic genuinely is inside the envelope -- but
at n=10 the system has no evidence of that, and certifying `INSIDE` would
assert a stability it never measured. This is the same rule as entry 070's null
band and the ladder's fourth rung: absence of evidence is not a verdict.

**Consequence for Phase 10:** Beat 4 shows a revocation, so its long-context
segment must carry at least `window_size` events. A 40-event Beat 4 would show
`INSUFFICIENT_DATA` for its whole length and prove nothing. The alternative --
lowering `window_size` for the demo -- was rejected: a threshold tuned to make a
demo work is the exact failure this project argues against, and the false-alarm
guard would refuse the configuration anyway.

The right pane says `"10 of 200 requests -- no envelope verdict yet"` rather
than printing `max PSI 0.000`, which would read as a measured stability.


## 076 - Rego via rego-cpp, not OPA, and what that costs

`SPEC.md` 7.1 says OPA/Rego or Cedar, and `config.policy.engine` is `opa`. OPA
ships as a ~50 MB Go binary. `CLAUDE.md` makes free-tier compute a hard
constraint, `results/` has a 10 MB staging limit, and the demo machine cannot be
assumed to have network at run time -- so an engine that arrives as a binary
download is an engine the demo does not reliably have.

We evaluate Rego through `regopy` (MIT, Python bindings to Microsoft's
`rego-cpp`): a pip-installable wheel, no Go toolchain, no separate process.

**What this does not buy:** `rego-cpp` is an independent implementation, not
OPA. Its coverage of the language is close but not identical, and a policy that
evaluates one way here is not thereby proven to evaluate the same way under
OPA. The mitigation is that the shipped bundles use a deliberately small subset
-- `default`, `contains ... if`, comparison, one comprehension, `max`, `count`,
`sprintf` -- and that the engine validates its own output rather than trusting
it: an entrypoint that does not resolve, a decision missing `rule_id`, or an
action outside the enum all fail at construction.

Rejected: shelling out to `opa eval` when the binary happens to be present and
falling back otherwise. Two evaluators with subtly different semantics, chosen
by what is installed, is worse than either one alone.

Rejected: parsing the `when`/`then` pairs out of the YAML ourselves. That is the
DSL 7.1 forbids, and it does not announce itself -- it arrives as "just enough
matching to avoid the dependency".

**Two silent traps found in the binding, both pinned by tests.** `set_input`
accepts a JSON *string* without complaint, sets the input document to that
string, and then every `input.foo` reference fails to resolve and every rule
falls through to its default -- a permissive decision with nothing in the output
to show for it. And querying an undefined entrypoint raises a native access
violation through the FFI rather than a Python error. The engine refuses a
non-mapping payload and probes its entrypoint at construction.

## 077 - The customer_support profile cannot be backed at n=600

Phase 7's power check refuses the profile the product leads with, and the
arithmetic is not close enough to argue about.

A profile declaring `calibration.sensitivity: 0.25` asserts it can detect a 25%
deviation from its flag-rate budget. `customer_support` runs at a budget of
0.10, and separating 0.125 from 0.100 needs **n >= 673**. The warrant is
measured on the 600-item test split. Short by 73 items.

| profile | budget | n needed at 25% | n measured | loads |
|---|---|---|---|---|
| customer_support | 0.10 | 673 | 600 | **no** |
| internal_knowledge | 0.20 | 288 | 600 | yes |
| decision_support | 0.50 | 58 | 600 | yes |

The direction is worth stating because it is counter-intuitive: **a smaller
budget is harder to warrant**, since a fixed *relative* tolerance is a smaller
absolute gap and the sample size scales as `1/gap^2`. The tier under the most
cost pressure is the one whose budget is hardest to stand behind.

**Rejected: lowering `calibration_sensitivity` until the bundle loads.** The
sensitivity is a declared statistical claim, and re-deriving it from the sample
that has to support it is the same move as selecting a threshold on test. The
number would then mean "whatever n we happened to have".

**Rejected: exempting the profile.** An exemption is invariant 3 with extra
steps.

The fix is more test items, and nothing else. Until then `customer_support` is
refused at load, which is the mechanism behaving correctly -- the refusal is the
product working, not the product failing.

## 078 - A declared ceiling with no measurement behind it is refused

`max_fpr_hard_negatives` was defaulting to 1.0 when a manifest omitted it, and
the check was skipped when the warrant carried no hard-negative measurement. Two
different absences, both silently passing.

Hard-negative FPR is measured on `hard-negatives-200`. The probe holds no
warrant there (Phase 8), so on `triviaqa-600` the metric is `None`. A profile
declaring `max_fpr_hard_negatives: 0.02` against that warrant was having its
ceiling quietly ignored -- an unbacked guarantee on every certificate it issues.

Two changes. The field is **required** in the manifest and may be `null`: "no
ceiling declared on this envelope" is a claim and has to be written down, since
a defaulted 1.0 and an explicit null are the same arithmetic and different
statements. And a ceiling declared against a warrant with no such measurement is
a **load failure**, on the same rule that stops `UNVALIDATED` counting as
`VALID`: an absence is not a pass.

The three shipped bundles therefore declare `null` with the reason inline, and
the resolution notes carry it onto the record. That makes the Phase 8 gap
visible in the artifact rather than inferable from a missing field.


## 079 - Pre-registration: a larger test split so customer_support can be warranted

**Written before the re-split runs and before any number on the new set exists.**
Entry 077 records the refusal this responds to.

### The problem, in numbers

`customer_support` declares a 0.10 flag-rate budget and 25% calibration
sensitivity. Separating 0.125 from 0.100 needs n >= 673. `triviaqa-600` holds
2400 items split 1200 / 600 / 600 (50/25/25), so the warrant is measured at
n = 600. Short by 73.

### What is being done

A **new frozen eval set**, `triviaqa-2400-t960`: the same 2400 items in the same
order, with the declared splits reallocated 40/20/40 -> train 960, validation
480, test 960.

**This is not a re-scoring of `triviaqa-600`.** `split` is inside the content
hash, so a reallocation produces a different hash and therefore a different
identity (invariant 9). The old set keeps its id, its envelope, its warrants and
its published numbers, all untouched. The new set is a new envelope, scored
once. Nothing already reported is reopened.

### Why the split size is not a peek

n = 960 is chosen from a power calculation, not from an outcome. 960 clears the
673 that a 0.10 budget needs, with enough margin to also clear the 865 an 0.08
budget would need. No number on the new set exists yet; the only input to the
choice is arithmetic that was already in entry 077.

What *was* learned from the old set is that 600 is insufficient. Acting on that
by building a larger held-out sample is the intended response to an underpowered
result. Acting on it by lowering the declared sensitivity until 600 suffices
would not be, which is why 077 rejected it.

### The activations are reused, and that loosens a check

The extraction cache is keyed to the eval set's full content hash, which
includes `split`. The new set would therefore be refused a cache whose
activations are, item for item, exactly correct -- `split` is not an input to
extraction. A GPU re-extraction to reproduce identical activations would be
waste dressed as rigour.

So `EvalSet` gains an **extraction identity**: a hash over item ids, question
ids, prompts, responses, labels and their order, excluding `split` and the set's
name. The cache is checked against that instead.

This is a genuine weakening and is recorded as one. What it still catches: any
edit to a prompt, a response, a label, the item set, or their order -- every way
the activations could stop describing the data. What it no longer catches: a set
renamed or re-split. Both are cases where the activations are unchanged by
construction.

### Declared success criterion

The re-split succeeds if **all** of these hold on the new envelope:

1. All three operating points issue `VALID` with all five controls passing.
2. Each profile's recall lower bound clears its `min_recall` -- 0.10 / 0.25 /
   0.50 for customer_support / internal_knowledge / decision_support.
3. `customer_support` loads: n_test >= 673 at a 0.10 budget.
4. The three profiles produce **three distinct actions** on one input.

### What we commit to before seeing the result

- **Report whatever comes out, including if it is worse.** Train shrinks
  1200 -> 960 and validation 600 -> 480. AUROC on the old set was 0.8256
  [0.7934, 0.8567]; recall lower bounds were 0.126 / 0.279 / 0.686 at
  thresholds 0.9045 / 0.8169 / 0.4673. If the new numbers are lower, they are
  the numbers, and both sets stay published side by side.
- **If a profile's recall lower bound falls below its floor, that is the
  result.** It gets reported as a profile that cannot be backed, exactly as
  customer_support was. It does not get fixed by lowering the floor.
- **One re-split, not a search.** If 40/20/40 fails the criterion above, we do
  not try 35/20/45. A sequence of splits tried until one passes is selection on
  test wearing a different hat. The next move after a failure would be more
  items, and it would be pre-registered on its own terms.
- The refusal is already shipped and tagged (`phase-7`). Whatever happens here,
  the history retains the run where the governance layer refused the flagship
  profile.


## 080 - Outcome of the pre-registered re-split: all four criteria met

Reporting against `079`, which was written before the re-split ran.

### The four declared criteria

1. **All three operating points issue `VALID` with all five controls passing.**
   Met. 5/5 on each.
2. **Each profile's recall lower bound clears its `min_recall`.** Met.
3. **`customer_support` loads.** Met: n_test 960 against the 673 a 0.10 budget
   needs at 25% sensitivity.
4. **Three distinct actions on one input.** Met: ALLOW / REDACT / ESCALATE.

### The numbers, both sets side by side

| operating point | triviaqa-600, n=600 | triviaqa-2400-t960, n=960 | floor |
|---|---|---|---|
| `P-customer-support` | 0.1661 [0.1255, 0.2128] | 0.2171 [0.1800, 0.2564] | 0.10 |
| `P-internal-knowledge` | 0.3321 [0.2789, 0.3960] | 0.3603 [0.3173, 0.4063] | 0.25 |
| `P-decision-support` | 0.7329 [0.6858, 0.7865] | 0.7367 [0.6974, 0.7783] | 0.50 |

AUROC 0.8256 [0.7934, 0.8567] -> 0.8232 [0.7985, 0.8507].

**Training on 960 instead of 1200 cost nothing measurable.** The AUROC point
estimate moved -0.0024, far inside either interval, and the interval itself
*narrowed* from 0.0633 wide to 0.0522 because n went 600 -> 960. That was not
guaranteed and 079 committed to reporting it either way; it is worth stating
plainly that the trade was close to free rather than implying it was expected.

The recall figures are not directly comparable between columns and should not be
read as improvement. Each threshold is re-selected on a different validation
split, so the operating points are near neighbours rather than the same point
measured twice -- `P-customer-support` moved 0.9045 -> 0.8909 and its measured
flag rate 0.0850 -> 0.1062, closer to the 0.10 it was aiming at.

### One thing worth naming

`triviaqa-600` and `triviaqa-2400-t960` are the same 2400 items and therefore
the same input distribution. They are two measurement partitions of one traffic
distribution, not two traffics. The matrix nonetheless treats them as separate
envelopes and the old warrants did not transfer -- the bundles went
`UNVALIDATED` on the new envelope until they were re-pointed at it, which is
invariant 1 doing exactly what it should even in the case where a human can see
the distributions are identical.

That is the right default. But it means "envelope" is currently carrying two
distinct ideas: *what distribution was this measured on* and *which partition of
it produced the number*. They coincide everywhere else in this repo. Recorded
here rather than resolved, because resolving it would change the warrant key.

### What this does not change

Entry 077 stands. The refusal happened, is tagged at `phase-7`, and
`results/policy-triviaqa-600.json` keeps its numbers. The fix was more evidence,
not a weaker claim -- `calibration.sensitivity` is still 0.25 in all three
bundles and was never touched.

---

### AMENDMENT (added with 081 and 082; the text above is left as written)

This log is append-only, so nothing above has been edited. Two things in it are
wrong and are corrected here at the point of the error.

**1. The training-cost claim is withdrawn.** "Training on 960 instead of 1200
cost nothing measurable" compared an AUROC measured on the 600-item test set
against one measured on the 960-item test set. Training size and evaluation
sample both moved, so the 0.0024 gap confounds them, and the interval narrowing
from 0.0633 to 0.0522 is a test-`n` effect that says nothing about training size
at all. The claim was not supported by any comparison that had been run.

The paired comparison in **081** finds the opposite: the training reduction cost
**0.0110 AUROC [0.0026, 0.0200]**, an interval that excludes zero. The sentence
"the trade was close to free" was wrong, and wrong in the flattering direction.

**2. The envelope framing is wrong, and so is the trigger.** The two ideas
"envelope" is carrying are not *distribution* and *partition*. They are
**distribution** and **sample**. The re-split left the distribution identical --
the same 2400 items -- and drew a different sample from it. A warrant is a point
estimate plus an interval, which is by construction a statement about a sample
estimating a property of a distribution, and the system has no vocabulary for
"same envelope, new measurement".

The trigger is **Phase 6, not Phase 8**. On-traffic warrant renewal *is*
same-distribution-new-sample, every renewal, indefinitely. If each renewal reads
as a new envelope: the matrix accumulates rows that are all the same envelope,
validation age resets on every renewal, and "this warrant has held for six
weeks" becomes unsayable. Validation age is a banner field in demo Beat 1 and
the continuity claim the whole staleness story rests on.

**Shape of the likely resolution**, recorded so Phase 6 does not start cold: an
envelope identity separate from the eval-set content hash, with warrants
carrying a sample reference underneath it. A renewal then produces a new sample
reference against an unchanged envelope identity, and age accrues against the
envelope.

**Not implemented now.** It changes the warrant key and every ledger row. Left
to Phase 6, where the migration can be done once.


## 081 - The paired comparison: the training reduction did cost AUROC

Supersedes the training-cost claim in `080`, which was withdrawn before this ran.

### What was wrong

`080` compared AUROC 0.8256 on the 600-item test set against 0.8232 on the
960-item test set and attributed the 0.0024 gap to the training reduction.
**Training size and evaluation sample both changed between those numbers.** The
gap confounds them. The interval narrowing 0.0633 -> 0.0522 is a test-`n`
effect and is not evidence about training size in either direction.

### The split relationship, measured rather than assumed (B.1)

| check | expected | observed |
|---|---|---|
| `old_test & new_test` | 600 | **600** |
| `new_train subset of old_train` | True | **True** |
| `new_test superset of old_test` | True | **True** |

It is a **promotion**, not a reshuffle: 240 items left train and 120 left
validation, all landing in test, and nothing else moved. So the paired set is
the entire old test split, n=600, and neither model trained on any of it.
`leaked_from_old_train` and `leaked_from_new_train` are both empty.

**This was luck resting on a reused seed, and it is worth naming as such.**
`triviaqa-600`'s declared splits came from `rng(1729).permutation` at 50/25/25;
`resplit_by_question` used the same generator and seed at 40/20/40. Same
permutation, different cut points, therefore strict nesting. A re-split at a
different seed would have reshuffled, cut the paired set to roughly 240 items,
and the comparison below would have been badly underpowered. Recorded because
the next person to re-split needs to know that reusing the source's seed is what
buys a paired comparison.

Sanity check that the pairing is real: the baseline model on the paired set
reproduces the published 600-set numbers exactly -- AUROC 0.8255524136312324 to
sixteen digits, and all three recalls to four.

### The method (B.3)

Paired bootstrap, 2000 resamples: draw items with replacement, recompute
**both** models on that same resample, record the difference. Not two
independent intervals compared for overlap -- both models score the same items,
so most variance is shared and cancels in the difference; comparing overlap
discards that and is underpowered in the way that looks like evidence of no
difference.

MDD is the smallest true difference this sample would detect 80% of the time at
a two-sided 5% level, `(1.96 + 0.84) x SE`. It travels with every result,
because a CI containing zero cannot be read without it.

### The result, thresholds pinned to the 1200-run (this is what 079 asked)

| quantity | 1200-trained | 960-trained | difference | 95% CI | MDD |
|---|---|---|---|---|---|
| AUROC | 0.8256 | 0.8145 | **-0.0110** | **[-0.0200, -0.0026]** | 0.0125 |
| recall @ P-customer-support | 0.1661 | 0.1697 | +0.0036 | [-0.0201, +0.0281] | 0.0340 |
| recall @ P-internal-knowledge | 0.3321 | 0.3249 | -0.0072 | [-0.0353, +0.0187] | 0.0387 |
| recall @ P-decision-support | 0.7329 | 0.7256 | -0.0072 | [-0.0366, +0.0226] | 0.0428 |

**AUROC: the interval excludes zero.** Training on 960 instead of 1200 cost
0.0110 AUROC, between 0.0026 and 0.0200. That is a measured cost, and the
opposite of what `080` reported.

**The three warranted recalls: underpowered, and reported as such.** Every
interval contains zero, and every MDD (0.034-0.043) is large relative to the
differences observed. This sample **cannot** distinguish the two models at any
operating point. It is not evidence that they perform alike, and it is not
reported as such.

That the AUROC difference resolves while the recall differences do not is
expected rather than surprising: AUROC pools every threshold and so uses all 600
items, while recall at a threshold is estimated from the positives above it --
at `P-customer-support` roughly 47 of 277 positives.

### Why the direction still favours the re-split

The cost is real and it is small, and the decision it was taken for is
unaffected: on the 960-item envelope every profile's recall lower bound clears
its floor with margin (0.180 vs 0.10, 0.317 vs 0.25, 0.697 vs 0.50), and
`customer_support` can now be warranted at all. A detector 0.011 AUROC weaker
that can be warranted beats a stronger one that cannot.

But that is a judgement about the trade, not a reason to have called the cost
zero.

---

### AMENDMENT to the paragraph above — the trade argument was overclaimed

The sentence *"a detector 0.011 AUROC weaker that can be warranted beats a
stronger one that cannot"* is wrong, and wrong in the same flattering direction
as the error this entry was written to correct. Recorded here rather than edited
out, because catching a flattering number and then defending the decision with a
flattering framing is the same failure one level up.

**The 1200-trained detector was already warranted.** All three profiles' recall
lower bounds cleared their floors on the 600-item envelope too -- 0.1255 against
0.10, 0.2789 against 0.25, 0.6858 against 0.50. Nothing was unwarrantable
before. What `077` refused was `customer_support`'s **calibration** claim, on
sample size, and that is a different claim from the ranking one this trade is
about.

What the re-split actually bought is **interval width on the warranted
quantities**:

| operating point | 600-envelope width | 960-envelope width | narrower by |
|---|---|---|---|
| `P-customer-support` | 0.0873 | 0.0764 | 12.5% |
| `P-internal-knowledge` | 0.1171 | 0.0891 | 23.9% |
| `P-decision-support` | 0.1007 | 0.0809 | 19.7% |

So the honest statement is: **0.0110 AUROC was paid for recall intervals 12-24%
narrower, plus a calibration claim that became supportable at all.** That is a
good trade and it does not need the stronger claim.

`083` qualifies both columns further: neither carries threshold-selection noise,
so both are narrower than the truth by roughly 1.4-1.6x. The *comparison*
between them stands, since the omission applies to both.


## 082 - The thresholds moved, and the recall asymmetry is ROC geometry

### C.1: the third confound was real

Thresholds are selected on validation, and validation went 600 -> 480. Read from
the run artifacts at full precision, not from config and not re-derived:

| operating point | n=600 threshold | n=960 threshold | move |
|---|---|---|---|
| `P-customer-support` | 0.90454704755232651 | 0.89094813278895302 | -0.01359891 |
| `P-internal-knowledge` | 0.81690146673133424 | 0.79828110607636127 | -0.01862036 |
| `P-decision-support` | 0.46730787294702475 | 0.46556394163042264 | -0.00174393 |

All three moved. So part of the published recall change is re-calibration, and
the pinned comparison in `081` is what separates it out.

`calibration.sensitivity` remaining 0.25 in all three bundles does not speak to
this. That is an input to selection, not the selected value -- a distinction
worth keeping, because "the config did not change" is exactly the kind of
statement that gets mistaken for "the threshold did not change".

### The three-way decomposition

The published move from the 600-envelope to the 960-envelope splits into the
model, the re-calibration, and the larger evaluation sample:

| operating point | pub n=600 | model | re-calibration | test sample | pub n=960 | total |
|---|---|---|---|---|---|---|
| `P-customer-support` | 0.1661 | +0.0036 | **+0.0289** | +0.0185 | 0.2171 | +0.0510 |
| `P-internal-knowledge` | 0.3321 | -0.0072 | **+0.0217** | +0.0137 | 0.3603 | +0.0281 |
| `P-decision-support` | 0.7329 | -0.0072 | +0.0000 | +0.0111 | 0.7367 | +0.0039 |

**Re-calibration is the largest single component of the customer_support move,
and the model contributes almost none of it.** The +0.051 that `080` reported
was not the detector getting better.

The model column is the pinned paired result from `081` and is underpowered at
every point; it is shown for completeness, not as three measured effects.

### C.2: the geometry, and it confirms

Measured ROC on the 960-item test split, local slope fitted by least squares
over a window in FPR (a two-point derivative on an empirical step function is
zero or infinite and describes nothing):

| operating point | FPR | recall | flag rate | local slope | window |
|---|---|---|---|---|---|
| `P-customer-support` | 0.0152 | 0.2171 | 0.1062 | **6.70** | 0.065 |
| `P-internal-knowledge` | 0.0436 | 0.3603 | 0.1865 | **5.27** | 0.093 |
| `P-decision-support` | 0.2467 | 0.7367 | 0.4677 | **0.85** | 0.099 |

**A 7.9x difference in local slope across three points on one curve.** The
hypothesis holds: `P-customer-support` sits on the steep segment, which is why a
threshold move of 0.0136 bought it +0.0289 recall while a move of 0.0017 bought
`P-decision-support` nothing measurable.

`results/roc_operating_points.png` draws it with a tangent at each point.

### What this licenses saying

The profile with the **highest traffic and the tightest latency budget** is also
the one whose warranted recall is **most sensitive to threshold placement** --
6.7 units of recall per unit of FPR against 0.85 at the escalation-heavy tier.

That is a direct argument for warranting operating points individually rather
than warranting a detector, and it is now demonstrated on our own measured curve
rather than asserted. It also explains why `P-customer-support` was the profile
whose calibration claim failed first in `077`: it is the point where the budget
is smallest and the curve is steepest, so it is the most expensive place on the
curve to make a promise and the hardest to keep one.

The claim is about **local sensitivity**, not about which operating point is
better. Nothing here says the steep point is a worse choice; it says its
warranted number moves more for a given threshold change, which is a reason to
warrant it separately rather than a reason to avoid it.


## 083 - Every published recall interval was conditional on a threshold set by five items

### The finding

Thresholds are selected on validation to hit a flag-rate budget, then frozen and
applied to test. The bootstrap that produces the reported recall interval
resamples **test only**. Selection noise is not in it, so every recall interval
this repo has published is conditional on the threshold being correct.

`082` is what makes the size of that omission calculable. The threshold's
position on the ROC's FPR axis is set by the validation negatives above it:

| operating point | budget | val negatives above | slope (082) |
|---|---|---|---|
| `P-customer-support` | f=0.10 | **5** of 255 | 6.70 |
| `P-internal-knowledge` | f=0.20 | 14 of 255 | 5.27 |
| `P-decision-support` | f=0.50 | 82 of 255 | 0.85 |

Five items place the customer_support point. Resampling validation moves it, and
the local slope decides what the move costs in recall.

### One correction to how this was put to me

It is **not** a budget violation. The declared budget is a *flag rate*, the
operating point aims at 0.10, and test realises **0.1062** -- so the certificate
meets the budget it claims, and the calibration machinery from Phase 4 is
checking the right quantity. Realised FPR (0.0152 on test against 0.0196 on
validation) is a derived quantity, not a declared one, and the gap sits inside
the validation FPR's own SE of 0.0087.

The failure is narrower and still serious: the interval understates uncertainty
at the point where the product's flagship profile runs. Stating it as a budget
breach would be the same overclaiming in the opposite direction.

### The nested bootstrap

Resample validation, reselect the threshold on the draw, resample test
independently, recompute recall. The probe fit is **held fixed** -- what comes
out is recall uncertainty given this probe including where its threshold lands,
which is what the warrant needs, since a warrant is issued for a specific fitted
probe. A fully nested version refitting on resampled training data is a larger
and much more expensive statement.

Validation and test are resampled independently, because they are disjoint
samples and coupling the draws would understate the combined variance.

### Result, 2000 draws, on `triviaqa-2400-t960`

| operating point | recall | conditional 95% CI | selection-aware 95% CI | width | widening |
|---|---|---|---|---|---|
| `P-customer-support` | 0.2171 | [0.1774, 0.2551] | **[0.1468, 0.2706]** | 0.0777 -> 0.1238 | **1.59x** |
| `P-internal-knowledge` | 0.3603 | [0.3157, 0.4048] | [0.3009, 0.4269] | 0.0891 -> 0.1261 | 1.42x |
| `P-decision-support` | 0.7367 | [0.6935, 0.7747] | [0.6734, 0.7840] | 0.0811 -> 0.1106 | 1.36x |

**The widening is monotone in the slope and in the negative count.** Up to 37%
of the reported precision at customer_support was an artefact of treating a
selected threshold as a fixed one.

**No load decision changes.** All three selection-aware lower bounds still clear
their floors -- 0.1468 against 0.10, 0.3009 against 0.25, 0.6734 against 0.50.
The bounds are wider; the conclusions are the same.

### A better version of the 082 demo beat

`082` argued for warranting operating points individually from a spread in point
sensitivity. This says the same thing in the units a warrant is written in: the
profile with the most traffic and the tightest budget has an interval **1.59x**
wider than reported, against **1.36x** at the escalation-heavy tier. Interval
width is what a warrant claims, so this is the version that lands.

### Not yet folded into issuance

`validate()` still computes and the certificates still publish the **conditional**
interval. Switching would move every published recall bound in the repo, which
wants explicit sanction rather than arriving as a side effect of an analysis.
Recommended for Phase 6, alongside the envelope/sample separation in the `080`
amendment -- both change what issuance writes, and doing them in one migration
means re-scoring once rather than twice.

Until then the honest reading of any published recall interval in this repo is:
**conditional on the threshold, and roughly 1.4-1.6x too narrow.**


## 084 - Stock Presidio is refused a warrant on Hinglish PII

Phase 8, D.1. Presidio was wrapped and measured, not tuned. The refusal is the
deliverable.

### What "stock" means, verified rather than assumed

Presidio 2.2.364 ships six Indian recognizers -- `InPanRecognizer`,
`InAadhaarRecognizer`, `InGstinRecognizer`, `InVehicleRegistrationRecognizer`,
`InVoterRecognizer`, `InPassportRecognizer` -- and **registers none of them by
default**. A default `AnalyzerEngine()` loads 17 recognizers, zero of them
Indian.

So "stock Presidio misses Indian identifiers" is not a performance claim to
argue about. The recognizers are not loaded. A team that pip-installs Presidio
and points it at Indian traffic gets nothing for Aadhaar, PAN, UPI or IFSC until
somebody knows to go looking.

**Presidio ships no recognizer for UPI VPA or IFSC at all**, in any
configuration. Those are the two identifiers most specific to Indian retail
banking, and they are the gap that no amount of enabling closes.

### What `InAadhaarRecognizer` actually validates

Read from source, as `TASKS.md` Phase 8 requires. It is **not** a naive regex:

- two patterns, `[0-9]{12}` and `[0-9]{4}[- :][0-9]{4}[- :][0-9]{4}`,
  both scored **0.01** and labelled "Very Weak";
- `validate_result` sanitises by removing `-`, space and `:` **only**, then
  requires twelve digits, numeric, **first digit >= 2**, a valid **Verhoeff**
  check digit, and not a palindrome.

Two consequences. The Verhoeff check is the real UIDAI algorithm, so this is a
genuinely strong validator and a correct rejection of a made-up number is not a
miss. And the sanitiser does not strip `.`, so `99.99.48.54.32.83` -- one of the
three disclosure forms in `hinglish-pii-200` -- matches neither pattern and is
never offered to the validator. **The miss is at the pattern stage, not the
validation stage.**

That distinction decided the custom recognizers: they widen the separator set
and inherit `validate_result` unchanged, so nothing accepts an identifier the
built-in would have rejected on its checksum.

### The fixture is not the confound

`hinglish-pii-200` draws Aadhaar values from the UIDAI 9999 test range and
records `checksum_valid` per item. That flag agrees with Verhoeff on **34 of 34**
Aadhaar items, so a low recall here is not made-up numbers correctly failing a
checksum. Checked before measuring, because it would have produced a completely
wrong conclusion about Presidio.

### Measured, `hinglish-pii-200`, n=200, base rate 0.51

| detector | recall | status | canary |
|---|---|---|---|
| `presidio-stock` | 0.1176 [0.0500, 0.2000] | **REFUSED** | 4/20 |
| `presidio-enabled` | 0.2843 [0.2075, 0.3810] | **REFUSED** | 12/20 |
| `presidio-enabled_plus_custom` | 0.6176 [0.5185, 0.7128] | VALID | 20/20 |
| `pii-reference` | 0.7941 [0.6981, 0.8846] | VALID | 20/20 |

Refusal reasons carry the numbers: stock fails the canary at 0.2000 against a
required 1.0, and its AUROC lower bound is 0.4456 against a required 0.55.
Enabled fails the canary at 0.6000 and misses the AUROC bar at 0.5409 -- by
0.009.

By disclosure form, recall on the positives:

| configuration | verbatim | spaced | obfuscated |
|---|---|---|---|
| `stock` | 0.1765 | 0.1765 | **0.0000** |
| `enabled` | 0.6471 | 0.2059 | **0.0000** |
| `enabled_plus_custom` | 0.6471 | 0.4706 | **0.0000** |

**No configuration detects a single obfuscated disclosure.** Those are forms a
real customer uses -- Devanagari framing, a number split across a sentence
("999936 ka 090910"), a masked phone ("XXXXXXXXX4685"), "at the rate" for `@`.

### The canary is the sharpest number here

`canary-20-pii` is twenty **verbatim, checksum-valid identifiers in plain
English frames**, built to be trivially easy on the principle that a canary a
detector can plausibly miss is a tripwire that fires on noise. Stock Presidio
catches **4 of 20**. Fully enabled it catches 12, missing every UPI VPA and
every IFSC.

### The machinery needed no changes

`TASKS.md` asks that a change to the certificate schema, the validation harness
or the drift monitor be logged as a finding before it is made. **None was
needed.** `validate_text_detector` took the adapter unmodified, the same
`PiiMatch` type the reference detector emits carried through, and the warrants
were issued and refused by the existing `issue_or_refuse`.

### One bug, in our adapter, not in Presidio

The adapter filters Presidio's output to an allowlist of identifier entities so
a `DATE_TIME` hit does not count as PII. `UPI_VPA` and `IN_IFSC` were missing
from that allowlist -- entities only the custom recognizers emit -- so
`enabled_plus_custom` was measured at 0.3725 and refused, when it actually
scores 0.6176 and is valid. Found by asking why the custom UPI recognizer had
not moved the canary. An allowlist that silently drops a detector's output is
the adapter misrepresenting the tool, and it happened to misrepresent it
*downwards*, which is the direction least likely to be questioned.


## 085 - Pre-registration: an out-of-sample set for the fitted recognizers

**Written before `hinglish-pii-200b` is built and before any number on it
exists.** Corrects a contamination in `084`.

### What is wrong with 084

Two of its four rows are in-sample and were not labelled as such.

`presidio-enabled_plus_custom` at recall **0.6176**: the custom recognizers in
`src/detectors/presidio_custom.py` were written by reading this set's failures.
The sequence was measure stock -> diagnose the miss at the pattern stage ->
widen the separator set -> re-measure. That is a detector fitted to the
evaluation data by a human rather than by gradient descent, and the number is
in-sample.

`pii-reference` at recall **0.7941**: `git log --diff-filter=A` shows
`src/detectors/pii_reference.py` and `evalsets/hinglish-pii-200.json` were added
in the **same commit**, `e271be2`. The detector is ours and was co-developed
with the set it is measured on. Same problem, and it was never disclosed.

The canary is worse than either. `enabled_plus_custom` scores 20/20 on a 20-item
gate whose pass condition is recall exactly 1.0 -- and the recognizers were
extended until it passed. A gate that could not have failed is not evidence.

`presidio-stock` (0.1176) and `presidio-enabled` (0.2843) are clean. Neither was
ever adapted to this set, so the refusals the demo rests on are honest.

### Why a fresh seed alone would not fix it

`_apply_form` in `src/evalsets/identifiers.py` renders the `spaced` form by
drawing `separator` from a **declared inventory of six** -- `" "`, `"-"`,
`" - "`, `"."`, `" . "`, `"  "` -- and `chunk` from `{2, 3, 4}`. The custom
recognizers use `_SEPARATOR = r"[\s.\-:]"` with a `{0,3}` repeat and a
digit-pairs alternative, which covers **that entire inventory**.

So the recognizers were fitted to the generator's declared space, readable from
source, not to observed instances. Redrawing at a new seed produces different
instances of forms already covered and tests nothing. This has to be said
plainly because a new-seed holdout would have looked rigorous and measured
nothing.

### What is being built

`hinglish-pii-200b`: the same 51 scenario templates and 26 near-miss templates,
fresh identifier values at seed **20260828**, and an **extended form inventory**
the recognizers were not written against:

- separators added: `"/"`, `"_"`, `"|"`, `","`, and `""` (no separator);
- chunk sizes added: 5 and 6.

None of `/`, `_`, `|`, `,` is in `_SEPARATOR`. Items drawn from the extension
should therefore fail, and how many do is the measurement.

### What this holds out, and what it does not

**Held out:** identifier values, the separator/chunk combinations outside the
original inventory, and the pairing of scenario to form.

**Not held out:** the 51 scenario templates and the three form *families*.
Those are the population definition rather than a tuning artifact -- the
recognizers were fitted to separators, not to scenario text. Reusing them keeps
`200b` a sample from the same population plus a declared extension, rather than
a different population.

This is therefore **not a clean holdout**, and it is not claimed as one. It is a
partial one that isolates the axis the fitting actually happened on. A clean
holdout needs scenarios written by someone who has not read the recognizers.

### Declared before measuring

- **Report whatever comes out**, including a large drop.
- The recognizers are **frozen** at commit `7ae7ac1` and are not touched
  again in response to what `200b` shows. If they are ever extended to cover
  `/` or `_`, that is a new detector version measured on a further set.
- **The comparison that matters** is `enabled_plus_custom` on `200` versus on
  `200b`. The gap is the fitting.
- `stock` and `enabled` are measured on `200b` too. They were never fitted, so
  their numbers should move only by sampling; a large move would mean `200b` is
  a harder set rather than a fair one, and would be reported as such.
- If `enabled_plus_custom` falls below the issuance bar on `200b`, its warrant
  is **refused there**, and `084`'s VALID row stands only for `200` with the
  contamination stated.


## 086 - The holdout was built, and it is underpowered. Reporting it anyway.

Executes the pre-registration in `085`. Two of its findings are useful and the
third is a failure of my own design, reported because `085` committed to
reporting whatever came out.

### Result on `hinglish-pii-200b`

| detector | `200` (in-sample for rows 3-4) | `200b` | change |
|---|---|---|---|
| `presidio-stock` | 0.1176 [0.0500, 0.2000] | 0.1471 [0.0714, 0.2341] | +0.0295 |
| `presidio-enabled` | 0.2843 [0.2075, 0.3810] | 0.3137 [0.2340, 0.3977] | +0.0294 |
| `presidio-enabled_plus_custom` | 0.6176 [0.5185, 0.7128] | 0.6471 [0.5667, 0.7359] | +0.0295 |
| `pii-reference` | 0.7941 [0.6981, 0.8846] | 0.8333 [0.7592, 0.9035] | +0.0392 |

**Nothing dropped.** The fitted detectors did not fall on the holdout.

### Why that is not the vindication it looks like

All four moved up by almost exactly the same amount, including the two that were
never fitted to anything. A uniform shift across fitted and unfitted detectors
alike is a property of the *set*, not of the detectors: `200b` is marginally
easier. It says nothing about whether the fitting inflated row 3.

The reason is a defect in how I built the extension, and it is the `MDD` lesson
from `081` in a different costume: **I did not check whether the holdout had the
power to detect the thing it was built to detect.**

- The extended inventory applies only to the `spaced` disclosure form, which is
  **34 of 102** positives. The other two thirds were never affected.
- Of the five separators added, one was the **empty string** — which renders an
  identifier contiguously, i.e. in its *canonical* form. That is the easiest
  case, not a novel one. I added it as an extension and it works as a
  simplification.
- Measured directly: only **5 of 102** positives in `200b` carry a separator
  outside `presidio_custom._SEPARATOR`. At n=5 nothing is measurable.

So `085`'s central comparison — `enabled_plus_custom` on `200` versus `200b` —
**was not actually testable by the set I built to test it.** The number is
reported and the claim it was meant to support is not made.

### What is now established, and what is not

**Established:** row 3's recall does not collapse out of sample, on a set whose
identifier values, form pairings and a small fraction of formats it had not
seen. That is weak positive evidence.

**Not established:** that the custom recognizers generalise to formatting they
were not written against. The holdout contains almost none of it.

`084`'s row 3 and row 4 therefore remain **in-sample numbers**, now with an
out-of-sample companion that does not contradict them and does not confirm the
thing at issue.

### What a real test needs

A set where the `spaced` and `obfuscated` forms are generated from an inventory
disjoint from the fitted one, across all three forms rather than one, sized so
that a drop of the size worth caring about would be visible. Roughly: if row 3's
true out-of-sample recall were 0.45 rather than 0.65, detecting that at 80% power
needs on the order of 90-100 affected positives, not 5.

The recognizers stay frozen at `7ae7ac1` regardless. They are not being adjusted
in response to any of this.

### Two corrections to 084 that are established

**The AUROC criterion is not the recall criterion restated.** It was put to me
that with FPR = 0 the two are algebraically linked, `AUROC = 0.5 + 0.5 x recall`.
The premise does not hold, and the fault is in how `084` reported it: the
FPR = 0.0000 quoted there is on **`hard-negatives-200`**, a different set. On
`hinglish-pii-200`'s own 98 negatives, stock's FPR is **0.1020**. The identity
for a binary detector is `0.5 + 0.5 x (recall - FPR)` = 0.5078, which matches the
measured AUROC of 0.5079.

They remain highly correlated for a near-binary detector, so they are not two
independent hurdles and `084` should not be read as though four separate
criteria were cleared. But they are not the same number: AUROC nets off the
false positives on the set's own negatives, and for `pii-reference` — twelve
distinct score levels and FPR 0.4388 on this set — the binary identity gives
0.6777 against a measured 0.7734, because the ranking above the threshold
carries real information.

**`pii-reference` is ours and was co-developed with the set.**
`git log --diff-filter=A` puts `src/detectors/pii_reference.py` and
`evalsets/hinglish-pii-200.json` in the same commit, `e271be2`. `084` presented
its 0.7941 as a reference point without disclosing that. It is disclosed now,
and it is why the honest comparison in the demo is stock-versus-enabled, both of
which are genuinely out-of-sample.


## 087 - Decomposing row 3, and correcting a stale table in 084

Three additions to `086`, all of which make `084` more usable rather than more
caveated.

### 1. 084's per-disclosure-form table was computed before the allowlist fix

`084` reported that **no configuration detects a single obfuscated
disclosure**. That table was produced with the buggy entity allowlist, which
dropped `UPI_VPA` and `IN_IFSC`. Corrected, on `hinglish-pii-200`:

| configuration | verbatim | spaced | obfuscated | overall |
|---|---|---|---|---|
| `presidio-stock` | 0.1765 | 0.1765 | **0.0000** | 0.1176 |
| `presidio-enabled` | 0.6471 | 0.2059 | **0.0000** | 0.2843 |
| `presidio-enabled_plus_custom` | 0.9706 | 0.6471 | 0.2353 | 0.6176 |
| `pii-reference` | 1.0000 | 0.6765 | 0.7059 | 0.7941 |

The claim survives **exactly where it is clean**: `stock` and `enabled`, the two
configurations never fitted to this set, score **0.0000** on obfuscated
disclosures. `enabled_plus_custom` reaches 0.2353, and that row is the fitted
one. So the corrected statement is narrower and better sourced than the original:
*off-the-shelf Presidio, in either configuration a team can obtain without
writing code, detects none of them.*

### 2. Row 3 decomposed into a clean part and a fitted part

`enabled_plus_custom` newly catches **34** positives that `enabled` misses. They
split by where the recognizer's pattern came from:

| origin of the pattern | gained on `200` | gained on `200b` |
|---|---|---|
| **spec-derived** — UPI VPA and IFSC in verbatim form | **11** | **11** |
| fitted — UPI/IFSC in spaced form | 6 | 6 |
| fitted — UPI/IFSC in obfuscated form | 8 | 8 |
| fitted — Aadhaar/PAN separator widening | 9 | 8 |
| Aadhaar/PAN verbatim and obfuscated | 0 | 0 |

**11 of 34, or 32.4%, is clean.** The UPI VPA pattern
`[A-Za-z0-9._-]{3,}\s?@\s?[A-Za-z]{2,}` and the IFSC pattern
`[A-Za-z]{4}0[A-Za-z0-9]{6}` are transcriptions of the published identifier
formats. Someone who had never seen this eval set would write them the same way,
because Presidio ships no recognizer for either and there was nothing to fit
*to* — the gap is categorical, not a formatting gap.

The other 23 are fitted: the `"at the rate"` spelling and every
separator-tolerant variant were written after reading this set's failures.

In recall terms on `200`, `enabled` 0.2843 -> **0.3922 from spec-derived work
alone** -> 0.6176 with the fitted work. The middle number is the one that
carries out of sample, and it is stable at 11 gained on both sets.

An earlier cut of this split by identifier *category* and put 73.5% in the clean
column. That was wrong: the spaced and obfuscated UPI/IFSC patterns are fitted
too, and grouping by category hid it. Splitting by pattern origin rather than by
category is what makes the number honest.

### 3. `200b` is compositionally identical, so the +0.03 is sampling

`086` left the uniform shift unexplained. It is not composition drift:

| | `200` | `200b` |
|---|---|---|
| items / positives / prevalence | 200 / 102 / 0.5100 | 200 / 102 / 0.5100 |
| kinds | 34 / 22 / 20 / 16 / 10 | 34 / 22 / 20 / 16 / 10 |
| forms | 34 / 34 / 34 | 34 / 34 / 34 |

Identical on every axis the builder controls. The +0.03 is roughly **3 items in
102** and comes from redrawn identifier values and a different scenario-to-form
pairing. Mundane, and better stated than left as an unexplained property.


## 088 - Composition rules for two warranted detectors, written before the code

Phase 8, D.2. The rules below are the design; `src/policy/compose.py` implements
them. Written first because a composition rule inferred from an implementation
is a rule nobody chose.

### The claim this answers

The brief observes that a fabricated detail about a person is simultaneously a
hallucination and a privacy concern, which makes clean categorisation hard.

**Our answer is that no categorisation is required.** A warrant certifies a
detector's operating point, not a taxonomy bucket. Two detectors score the same
input, each carries its own warrant with its own measured bounds on its own
envelope, and the policy layer composes the two decisions. Nothing has to decide
which category the input "really" is.

**We are deliberately not building a taxonomy classifier** that assigns an input
to both categories. That would concede the point while appearing to answer it:
it reintroduces exactly the categorisation step the argument says is
unnecessary, and it would need its own warrant on its own eval set, which
nobody has measured.

### What composes, and what does not

**Actions compose. Bounds do not.**

An action is a decision about one request, and the composed action is a function
of the two detectors' actions. A bound is a measured claim about a detector on
an envelope, and there is no arithmetic that turns two detectors' bounds into a
joint bound without a measurement of the pair. Their errors are not independent
-- both read the same text -- and assuming independence to multiply them would
manufacture a number nobody measured.

So a composed certificate carries **both bounds, side by side, each labelled
with its detector and envelope**, and claims no joint recall. A reader can see
what each detector was worth; nobody can read off what the pair was worth,
because that was never measured.

### The four cases

**1. Both VALID, both flag.** Action: the **more restrictive** of the two, by the
ladder `ALLOW < REDACT < CONFIRM < ESCALATE < BLOCK`. Bounds: both cited.

The reason for most-restrictive rather than a vote: the two detectors are
looking for different things, so agreement that *something* is wrong is not two
opinions on one question. A PII finding and a hallucination finding on one
response are two separate true statements, and the response has to satisfy both.

**2. Both VALID, they disagree** — one flags, one does not. Action: the
**flagging detector's action**. Bounds: both cited, and the certificate records
that the other detector did not fire.

Explicitly **not** a vote, and not the conservative default either. A
disagreement between detectors looking for different things is not evidence of
uncertainty -- the PII detector not firing on a hallucination is the PII
detector working correctly. Treating it as a dissenting vote would let a
correct silence cancel a correct finding.

**3. One VALID, one REFUSED.** Action: the valid detector's action, taken alone.
Bounds: the valid detector's only. The refusal is recorded in `unchecked`.

The composed decision **does not** inherit the refusal. A refused detector is
out of service on that envelope; it contributes no finding and no bound, and
treating its absence as a veto would take a working detector out of service
because an unrelated one failed its controls. But the certificate must say what
was not checked, because "we checked for PII" and "our PII detector is out of
service" produce identical-looking `ALLOW`s otherwise.

Nor does it degrade the tier. Tier is a property of the access a detector has,
not of how many detectors ran.

**4. One VALID, one UNVALIDATED.** Action: the valid detector's action,
**unless** the unvalidated detector fires, in which case the profile's
conservative default applies. Bounds: the valid detector's only. The unvalidated
detector's finding is recorded with `warrant_id: null`.

This is the case that must stay distinct from 3, and the distinction is
`CLAUDE.md` invariant 2. `REFUSED` means *measured here and failed*: its output
is known to be unreliable on this envelope and is ignored. `UNVALIDATED` means
*never measured here*: its output is information of unknown quality, which is
not the same as no information and not the same as bad information.

So an unvalidated detector's finding cannot be quoted with a bound -- there is
no bound -- but it can trigger the conservative default, which is what a
profile's `conservative_default` is for. Enqueuing the cell for validation is
how the matrix fills itself in.

### The rule that overrides all four

If **no** detector holds a valid warrant on this envelope, the composed
certificate claims no bounds at all and the profile's conservative default
applies. Two unvalidated detectors do not add up to one validated one.

### What a composed certificate must carry

- every finding, including those from detectors with no warrant;
- `warrants_relied_upon` naming only the detectors whose bounds are quoted;
- `claimed_bounds` keyed **by detector**, never merged;
- `weakest_warrant_status` across the detectors actually relied upon;
- `unchecked` naming every detector that did not contribute and why -- refused,
  unvalidated, or not run.


## 089 - An envelope is a distribution plus a label definition

Found while building D.2. The composition demo needs two detector categories
holding valid warrants on **one** envelope, and the matrix has none: the probe is
warranted only on `triviaqa-2400-t960`, the PII detectors only on the PII sets.

The obvious fix was to warrant `pii-reference` — a text detector, no GPU needed —
on the probe's envelope. It worked. It produced:

    pii-reference on triviaqa-2400-t960 — recall 0.0063 [0.0018, 0.0111]

**That is not a PII recall.** TriviaQA's positive class means *"the model's
answer was incorrect"*. What was measured is the fraction of wrong answers that
happen to contain a personal identifier — a quantity nobody wants, with a
correct interval, filed under a warrant key that reads as a PII claim.

Nothing errored. Every part was doing its job: the set has labels, the detector
produces scores, the metrics are arithmetically correct. Only the *meaning* was
mismatched, and meaning was the one thing not represented anywhere a check could
reach.

### The rule

**A detector can only be warranted on an eval set whose labels mean what that
detector detects.** An envelope has been treated throughout this repo as an
input distribution; it is a distribution *and* a label definition, and the
second half was implicit until it broke.

`src/evalsets/categories.py` declares the mapping and
`validate_text_detector` refuses a mismatch before scoring — before, because
everything after that point is arithmetically correct either way.

**Single-class sets are the exception, and a principled one.**
`hard-negatives-200` has no positives, so it makes no category claim and any
detector may be measured there. "How often does this fire on traffic that should
never be flagged" is worth asking of any detector, whatever it detects.

**An unmapped set is refused, not defaulted.** A set whose label meaning nobody
has declared is exactly the case that produced the bad warrant.

### Why a registry and not a field on the set

`construction["label_meaning"]` already records it, in prose, and prose cannot be
checked. A structured field on `EvalSet` is the obvious fix and is unavailable:
construction notes are inside the content hash, so adding one would change the
identity of every frozen set and orphan every warrant keyed on it. Third time
this constraint has bitten — see also `resplit_by_question`'s nesting flag and
`build_hinglish_pii`'s `extended_forms`. The pattern is now established: facts
*about* a frozen set that were not known when it was frozen live beside it, not
inside it.

### What this blocks

**D.2's gate cannot be met with measured warrants.** No eval set carries labels
for both a hallucination and a PII positive class, and the guard now correctly
prevents manufacturing one. The composition rules in `088` are implemented and
exhaustively tested against fixtures; what is missing is a measured pair.

Three ways forward, none of which should be chosen quietly:

1. **A dual-labelled eval set** — items carrying both an "answer incorrect" and
   a "contains identifier" label. This is the honest fix and it is the only one
   that makes the brief's actual claim measurable, since the claim is precisely
   that one item can be both. Needs a GPU pass for the probe side.
2. **Both detectors on `hard-negatives-200`**, which is single-class and admits
   any detector. That yields two FPR-only warrants and a real composed decision,
   but demonstrates composition on an envelope where neither detector can claim
   recall. Needs a GPU pass for the probe's activations on that set.
3. **Report the composition mechanism as built and tested, and the measured pair
   as an open gap.** Costs nothing and claims nothing false.

The finding itself is worth more than the demo it blocked. A system that will
happily warrant a PII detector against hallucination labels is a system whose
warrants mean less than they appear to, and nothing else in the build would have
surfaced it.


## 090 - Pre-registration: the dual-labelled set and the powered holdout

**Written before either set is built and before any number on either exists.**
One GPU pass covers both; `085` failed by not checking its own power, so both
have their power stated here and verified after generation but **before**
scoring.

### Why a dual-labelled set rather than a shared single-class envelope

The brief's overlap bullet says a fabricated detail about a person can
simultaneously be a hallucination and a privacy concern. **An item carrying both
labels is that sentence made measurable.** The alternative considered in `089` —
composing on `hard-negatives-200`, which is single-class and admits any detector
— demonstrates the mechanism on an envelope where the co-occurrence *cannot
exist*, and so answers a different question than the one asked.

### The set: `banking-dual-240`

240 hand-written items in a 2x2 over two independent labels:

| cell | hallucination | PII | n |
|---|---|---|---|
| both | 1 | 1 | 60 |
| hallucination only | 1 | 0 | 60 |
| PII only | 0 | 1 | 60 |
| neither | 0 | 0 | 60 |

Giving 120 PII positives and 120 hallucination positives — each enough for a
recall interval comparable to the 102 in `hinglish-pii-200`.

**The set carries two label columns, not one.** `EvalItem.label` cannot express
this, so the second lives in `meta` and the set is registered in
`EVAL_SET_CATEGORY` under whichever column a given validation run reads. A run
declares which label it is measuring against; a run that does not is refused, by
the same guard `089` added.

### The co-occurrence rate is constructed, and the certificate must say so

Measured on real traffic, wrong answers containing an identifier ran **0.0063**
on `triviaqa-2400-t960` — about 6 items in 960. At that rate a 240-item set
would hold **one or two** co-occurring items and the composed VALID/VALID case
would have nothing to run on.

So co-occurrence is oversampled to **25%**, roughly **40x** its measured rate.
That is a deliberate construction and it has a consequence: **any composed bound
measured here describes this constructed distribution and not production
traffic.** That statement goes in the certificate's `claimed_bounds` as a field,
not in a footnote — a reader who sees a composed recall must see, in the same
object, that its envelope was enriched.

Per-detector bounds are unaffected: each detector's recall is measured against
its own label column, and enrichment changes prevalence rather than recall.
Precision and any lift figure **are** affected and will not be quoted from this
set.

### Target counts per composition case, not just overall

`085`'s failure was aggregate power with nothing in the cell that mattered. The
four cases in `088` split into two kinds:

**Content-dependent** — need items:

| case | condition | target items |
|---|---|---|
| 1. both VALID, both flag | probe fires and PII fires | **>= 25** |
| 2a. disagree | probe fires, PII silent | **>= 25** |
| 2b. disagree | PII fires, probe silent | **>= 25** |
| 0. neither fires | — | >= 25 |

**Status-dependent** — need no items, only a detector in that state on this
envelope: case 3 uses `presidio-stock`, which `084` shows is REFUSED wherever it
is measured; case 4 uses any detector never validated here.

At the operating points now warranted — probe flag rate ~0.106, `pii-reference`
recall 0.79 — the 2x2 above should land roughly 25-45 items in each of the four
content cells. **Verified after generation and before scoring.** If any cell
falls below 25, the set is rebuilt with adjusted cell sizes and that is recorded;
it is not scored and reported from a cell of three.

### The powered holdout: `hinglish-pii-300c`

`086` found the holdout could not detect what it was built to detect: the
extension touched only the `spaced` form, one of five added separators was the
**empty string** (which renders identifiers canonically and is therefore a
simplification, not an extension), and only 5 of 102 positives ended up outside
the fitted class.

Replacing it, with two checks `085` should have had:

1. **The extension applies to all three disclosure forms**, not one.
2. **Every added separator is mechanically asserted to change the rendered
   string** — `rendered != canonical` — which would have caught the empty
   string at build time.
3. **Target: >= 90 positives carrying formatting outside
   `presidio_custom._SEPARATOR`.** Counted after generation, before scoring. At
   90 affected positives, a drop from 0.65 to 0.45 is detectable at 80% power;
   at 5 it was not.

### Committed before the run

- Report whatever comes out, both sets, including a large drop on the holdout.
- The custom recognizers stay frozen at `7ae7ac1`.
- If the dual set's composed VALID/VALID cell is under 25 after generation, say
  so and rebuild rather than scoring it.
- The composed certificate states its enriched co-occurrence rate inline.
- `banking-dual-240` needs a GPU pass for the probe's activations. This machine
  has none (`cuda_available: false`), so the set and its labels are built and
  frozen on CPU now and the extraction runs on Kaggle. **Nothing is scored until
  it does** — no placeholder numbers, no synthetic stand-in reported as measured.

---

### AMENDMENT, before authoring and before the run

Four things settled after the arithmetic was run rather than assumed. The
original text above stands; this is added, not edited.

#### 1. Factorial construction, and the clustering it forces

The 240 items are **60 base scenarios crossed with (correct / incorrect) x
(identifier / no identifier)**, holding the surrounding text fixed across the
four cells.

The reason is not economy of effort. Authoring 240 items freely would let
writing style correlate with the labels - wrong answers reading differently from
right ones in ways a probe can pick up - and the measurement would be of
authorship rather than of incorrectness. Fixing the frame makes the two axes
orthogonal by construction, so no stylistic confound can attach to either label.

**Consequence: items within a scenario are not independent, so every interval on
this set must be a cluster bootstrap resampling scenarios, not items.**
Item-level resampling would understate every interval by roughly the cluster
factor and produce correct-looking numbers that are wrong. Pre-registered here
because it is exactly the class of error `081` was written about, and because a
bootstrap that resamples the wrong unit does not announce itself.

#### 2. The set supports one profile, not three. Declared now.

The probe's TriviaQA warrants do not transfer to this envelope - invariant 1,
which already fired on the re-split when identical items under a different
partition went `UNVALIDATED`. So the probe must be **validated** on
`banking-dual-240`, and at 240 items with 120 hallucination-negatives:

| operating point | measured FPR | negatives above threshold at N=120 | N_neg for >= 20 | implied set size |
|---|---|---|---|---|
| `P-customer-support` | 0.0152 | **1.8** | 1318 | ~2635 |
| `P-internal-knowledge` | 0.0436 | 5.2 | 458 | ~917 |
| `P-decision-support` | 0.2467 | 29.6 | 81 | ~162 |

A threshold positioned by 1.8 negative items is not a threshold, and `083`
showed the selection-aware interval widens worst exactly where fewest negatives
position it - 1.59x at five. At 1.8 it would be unusable.

**So `banking-dual-240` warrants the probe at `P-decision-support` only, and the
composed demo runs there.** Declared before authoring rather than discovered
after. Sizing up to reach `internal_knowledge` would need ~917 hand-written
items and is not proposed.

This costs nothing elsewhere: the three-profiles-on-one-curve result lives on
`triviaqa-2400-t960` and is unaffected. And decision_support - low volume, high
consequence, escalation-heavy - is the right register for a fabricated detail
about a person in a banking context anyway.

#### 3. The probe may simply not work on this envelope, and that is a result

The probe was fitted on TriviaQA activations: English trivia questions. This set
is Hinglish banking-support text. Nothing guarantees the signal transfers, and
if AUROC's lower bound misses the 0.55 issuance bar the probe is **REFUSED**
here.

That would give a `VALID / REFUSED` composition - case 3 of `088`, demonstrated
on measured warrants - which is a legitimate and interesting beat, but it is
**not the overlap demonstration** and must not be presented as one. Committed in
advance: if the probe is refused on this envelope, we report the refusal, the
composed pair stays unmeasured, and `phase-8`'s gate stays open.

**Cheap de-risking, before authoring all 60:** author a **12-scenario pilot**
(48 items), tokenise on CPU - no GPU needed - and compute the token-length and
script-mix envelope distance against the `triviaqa-2400-t960` reference. That
does not predict whether the probe's *signal* transfers, but a very large input
distance is an early warning worth having for the price of a pilot.

#### 4. What happens if the run does not land

A declared rule rather than a deadline, decided now so it is policy rather than
a concession made under pressure:

**If the Kaggle extraction has not landed by the time the submission is
assembled, `phase-8` is tagged with the composed-pair gap named in the tag
annotation, and the gap goes in the README's open-items list.** No placeholder
numbers, no fixture result presented as measured, and no softening of the gate
clause to make it fit.

#### 5. The pilot's saturation criterion, derived rather than described

`3` above said to check whether the probe "collapses" or "ranks weakly but
monotonically". Those are the right two failures and they are useless as stated:
a post-hoc reading of a 24-point histogram finds whichever one the reader is
hoping for. So the criterion is numeric and fixed **now**.

**Measure:** the interquartile range of the probe's scores on the pilot's 24
hallucination-labelled items, as a fraction of its IQR on `triviaqa-2400-t960`
test, where the probe demonstrably works.

**Threshold, derived from a null band rather than invented.** Drawing 24 items
at random from TriviaQA test — data the probe *does* rank — 5000 times:

| percentile of the IQR ratio | value |
|---|---|
| p1 | 0.486 |
| p2.5 | 0.553 |
| **p5** | **0.605** |
| p10 | 0.671 |
| p50 | 0.921 |

Reference IQR is 0.4992 on n=960, range [0.0190, 0.9844].

**A pilot IQR ratio below 0.605 is narrower than sampling noise explains at this
n, and is declared saturation.** Above it, the spread is consistent with a probe
that is ranking — weakly or well — and a low AUROC is then a statement about
discriminative power rather than about activations being off-distribution.

Same construction as `070`'s PSI null band and `029`/`031`'s negative-control
bands: a threshold quoted without its sample size is a threshold that has not
been checked.

#### 6. One retry, and the first pilot is reported either way

If the pilot saturates, the scenarios may be re-authored **once**, closer to the
register the probe was fitted on, and the pilot re-run.

**One retry, not a search.** Two attempts at moving text toward a distribution
where the probe performs is tuning the eval set to the detector, which is the
same species of error as tuning Presidio until it passed — and it would be
harder to see, because nothing about it looks like tuning.

The first pilot's IQR ratio and AUROC are reported whatever the second shows. If
the second also saturates, the probe is reported as not transferring to this
envelope, the composed pair stays unmeasured, and `4` above applies.

#### 7. The decision_support interval on this set carries its selection-aware widening

Recorded here rather than in a working note, because it modifies how an interval
is computed on a set that does not exist yet, and notes like that get
rediscovered as bugs.

At ~30 hallucination-negatives above the `P-decision-support` threshold, `083`'s
threshold-selection noise applies to this envelope as it does to every other.
The widening factor will be smaller than the 1.59x measured at five negatives
and larger than nothing.

**It goes inside the reported interval, not beside it as a caveat.** A
conditional interval with a footnote is read as the interval; the whole point of
`083` is that the conditional number understates by a factor nobody sees.

#### 8. Sequencing: the pilot runs before the proposal work

The pilot gates everything downstream of it — the remaining 48 scenarios cannot
be authored until it lands, and their content depends on its result. The
proposal has no such dependency.

So the pilot goes first even though the proposal has the higher marginal return,
because a late pilot means authoring 48 items under time pressure with the
fallback in `4` already in force. The fallback exists for that case; the
ordering exists so it is not needed.

---

### CORRECTION, before authoring: the factorial design cannot measure the probe

Found by checking the clustering assumption behind the null band in `5`. The
band was wrong, and underneath it the construction in `1` was wrong in a way
that would have produced a confident, well-measured, meaningless result.

#### The break

`1` said to cross 60 scenarios with (correct / incorrect) x (identifier / none),
**holding the surrounding text fixed**. The correctness axis was to be authored
by varying the assistant's response.

**The probe reads only the prompt.** `build_prompt(tokenizer, item.prompt,
SYSTEM_PROMPT)` — question-time, before any generated token exists, which is the
entire thesis. So within one scenario the `correct` and `incorrect` cells
present the probe with **identical input carrying opposite labels**, and its
AUROC on that axis is exactly 0.5 by construction.

It would not have looked like a bug. Twenty-four distinct prompts across twelve
scenarios and two identifier states give a perfectly normal score spread; the
IQR-ratio check in `5` would have passed; AUROC would have come out at 0.5; and
the conclusion would have been *"the probe ranks on this envelope but does not
discriminate"* — plausible, correctly computed, and about an artifact of how the
set was built.

Third instance of this shape in the build, after the `pii-reference`-on-TriviaQA
warrant (`089`) and the empty-string separator (`086`). All three produce
arithmetically correct output about the wrong question.

#### The corrected construction

**Correctness cannot be authored. It is a property of the question and it is
measured.** `_label_items` is the existing path: generate an answer, judge it
against gold aliases, `label = 0 if correct else 1`.

So the axes are no longer symmetric, and that is the fix rather than a
compromise:

| axis | how it is set |
|---|---|
| identifier present / absent | **authored**, frame held fixed within a question |
| answer correct / incorrect | **measured**, by generating and judging |

The stylistic confound `1` existed to prevent is now *impossible on the
correctness axis*, because nothing about that label is written by us. It remains
handled on the identifier axis by holding the frame fixed. Strictly better than
the original.

**Consequences, all of which need declaring:**

1. **The questions must have checkable gold answers.** *"What is my balance"*
   has none. So the set is banking **factual lookup** — IFSC codes, branch
   details, product terms, fee schedules — with alias lists, not open-ended
   support chat. That narrows the register and makes it less like real traffic;
   declared rather than glossed.
2. **Cell sizes are not controllable by construction.** Co-occurrence is
   (identifier-present fraction) x (model error rate on these questions), and
   only the first is ours. At 50% authored identifier presence and a plausible
   error rate this lands near the 20-25% the original design targeted, but it is
   an outcome rather than a setting. The `>= 25 per cell` check in `2` therefore
   stays exactly where it is — after generation, before scoring.
3. **The run needs generation, not just extraction.** A larger GPU job than
   pre-registered. The pipeline already supports it.
4. **Clustering is now two items per question, not four**, since the correctness
   axis no longer multiplies the frame.

#### The null band was also wrong, and by more than it looked

`5` drew 24 **independent** items from TriviaQA test. The pilot's items are
clustered, so their effective sample size is the number of questions, not the
number of items, and their IQR runs narrower for reasons unrelated to
saturation.

Redrawn at the effective n:

| draw | p5 of IQR ratio | p50 |
|---|---|---|
| 24 independent items (as pre-registered) | 0.605 | 0.921 |
| **12 clusters (effective n = questions)** | **0.439** | 0.837 |

The pre-registered threshold was **38% too high**. A pilot landing between 0.439
and 0.605 would have been declared saturated when it was only clustered, and the
scenarios would have been re-authored to fix a problem that did not exist —
spending the one retry `6` allows on nothing.

**The criterion becomes: IQR ratio below 0.439 is saturation.** Drawn at the
number of clusters rather than the number of items, which is the same correction
the cluster bootstrap in `1` makes to every interval on this set.

#### What does not change

`2` (decision_support only), `3` (a refusal is a result, reported as such), `4`
(the fallback rule), `6` (one retry), `7` (`083` widening inside the interval)
and `8` (pilot before proposal) all stand as written.


## 091 - What an envelope is, arrived at by three near-misses

Three parts, each discovered separately, each by a number that read wrong rather
than by a test that failed. Stated together because the pieces only make sense
as one definition.

**An envelope is a distribution, a label definition, and a sample.**

| part | discovered by | what went wrong |
|---|---|---|
| distribution | the design, from the start | — |
| **sample** | the re-split (`080` amendment) | the same 2400 items re-partitioned read as a *new envelope*, so warrants did not transfer and validation age would reset on every renewal |
| **label definition** | `pii-reference` on TriviaQA (`089`) | a PII detector was warranted against hallucination labels and produced a correct interval about the wrong question |

Only the first was designed. The other two were each present all along and
invisible until a specific number looked wrong to a human.

### Why the omissions were invisible

Both near-misses produced output that was **correct in every checkable respect**.
The re-split's warrants were properly keyed, properly hashed, properly refused
when they should have been. The PII-on-TriviaQA warrant had a valid Clopper-
Pearson interval, a passing control suite, and an accurate `n`. Nothing was
approximated and nothing raised.

What was wrong in both cases was a *meaning* that the system had no
representation for, and a system cannot check what it cannot represent. That is
the general lesson and it is worth more than either fix: the failure mode this
product exists to prevent — a claim that is well-formed, well-measured, and
about a different question than it appears to be — occurred **twice inside the
system built to prevent it**, and both times it was caught by a person reading a
number and finding it implausible, not by a control.

A reviewer is entitled to ask what else is unrepresented. The honest answer is
that we do not know, and that the three parts above are the ones two months of
building surfaced.

**But "a human noticed" is not the same as "we got lucky", and the difference is
the argument for the whole design.** A PII detector reporting recall 0.0063 is
absurd on its face - and it is only absurd on its face because the warrant
surfaces the operating point, the sample size, the interval and the envelope
together, in a form a person can read and find implausible. The same detector
behind a green dashboard reports that it ran. There is nothing there to
disbelieve.

So the answer to "what else is unrepresented" is not a promise that nothing is.
It is that the system reports in a shape where a person can notice, which is
what made both catches possible. That is the case for the warrant, made against
our own worst outcome rather than a hypothetical one.

### The consequence for Phase 6

Renewal is same-distribution, same-labels, **new sample**. The `080` amendment
records the shape: an envelope identity separate from the eval-set content hash,
with warrants carrying a sample reference underneath it, so age accrues against
the envelope rather than resetting. That migration is where all three parts get
represented properly, and it should be done once.

---

## 092 - Facts about a frozen set that were not known at freeze time live beside it

Three occurrences is a pattern, so it is named once here rather than defended
three times.

`resplit_by_question`'s nesting relationship, `build_hinglish_pii`'s
`extended_forms`, and `EVAL_SET_CATEGORY`'s label meaning were each, at first,
going to be written into an eval set's `construction` notes. Each time the same
thing stopped it: **construction notes are inside the content hash**, so adding
a field changes the set's identity and orphans every warrant keyed on it. Each
time the fix was a declaration living beside the set rather than inside it.

**The principle: the content hash's job is immutability, not completeness.**

A frozen set's hash answers exactly one question — *is this the same data as when
the warrant was issued?* Facts learned afterwards are not part of that question.
Writing them in would mean every new insight about a set silently invalidates
every measurement made on it, which inverts the purpose: the hash exists so
numbers stay attached to the data they were measured on, and a hash that moves
whenever anyone learns something is a hash that guarantees nothing.

So:

- **Inside the hash:** the items, their order, their labels, the data source, and
  the construction parameters known at build time.
- **Beside the hash:** anything discovered later — how a set relates to another
  set, what its labels mean, which extension it was built with. Declared in code,
  versioned with the code, and refusing to default when unmapped.

The test that keeps this honest is the same each time: rebuild the frozen set and
assert it still hashes to the committed value. `test_the_extended_inventory_does_not_change_the_frozen_set`
and `test_the_nesting_check_does_not_change_the_frozen_identity` both exist for
that reason.

The cost is real and worth stating: three side registries are three places a
reader must look, and none of them is discoverable from the set file itself. A
`registry` module that gathers them would be an improvement and is not urgent.


## 093 - Override records carry their stratum and draw probability, or they are refused

Phase 8, D.3. The minimum viable feedback loop: capture what a reviewer decided
about an escalated item, store it against the certificate that escalated it,
expose count and direction. **Not retraining** — it is the label-capture path
Phase 6's estimator needs anyway, shaped so Phase 6 consumes it without a
migration.

### The bias this schema exists to prevent

Overrides exist **only on escalated items**, so the label pool is conditioned on
the detector having scored above threshold. Fed to a stratified estimator
unweighted, that biases recall **upward** — the flagged stratum is enriched for
true positives, so recall measured on reviewed items is recall among the items
the detector already liked.

It would arrive as a number that looks better than the truth, with nothing in
the record to show why. Each record therefore carries the stratum it was drawn
from and the probability it was drawn with, and the estimator weights by
`1/selection_probability`.

### Why stored and not derived, which is the whole design decision

The obvious alternative is to reconstruct the stratum at read time from the
score and the threshold. It fails, and it fails silently.

An item's stratum depends on **the threshold and envelope in force when it was
captured**, and both move. `082` measured all three thresholds moving between
two runs of the same detector; an envelope is re-drawn on every renewal. A
record read six months later would be reconstructed against a threshold that did
not exist when it was written — silently reassigning items between strata and
reweighting the whole estimate.

There is no error path. The reconstruction always succeeds and always yields a
plausible stratum. So the fields are captured at write time and a record without
them **cannot be constructed**, which means it cannot reach the ledger: there is
no route from a malformed record to a stored one, because `OverrideRecord`
validates in `__post_init__` and `append_override` only accepts records.

A hard failure rather than a warning, because records written without these are
not repairable later. A warning would produce exactly the artifact the schema is
meant to prevent, and would produce it in bulk before anyone read the log.

### Two error kinds, never one counter

`ESCALATE_TO_ALLOW` is a false positive and costs one wasted review.
`ALLOW_TO_ESCALATE` is a false negative and costs a customer acting on a wrong
answer. `human_decision` and `direction` are cross-checked against each other,
because letting them drift apart makes the false-negative count unreadable.

### Weighted counts travel beside raw ones

`override_summary` reports both. A raw count of false negatives from a reviewed
sample is not an estimate of anything, and quoting it as a rate is the
upward-biased number above. One unflagged item drawn at 1-in-200 and found to be
a miss stands for 200; the summary says so.

### What is deliberately not stored

Message content. The store is queried by session under DPDP Rule 6 and is not a
place to accumulate text; records carry an opaque `item_ref`. Reviewer identity
likewise — `reviewer_ref` is opaque, and exists for inter-rater agreement rather
than for attribution.


## 094 - The LiteLLM adapter, and the line that keeps it an adapter

Phase 8, D.4. `CLAUDE.md` rules a gateway out of scope, and the distinction is
easy to lose by accretion: a gateway owns the request path, terminates
connections, holds credentials and routes — it becomes something an enterprise
has to operate and trust. This sits behind LiteLLM, which already does that job,
and adds one thing: the certificate.

The upstream call is **injected**, not imported, so the adapter never holds a
key. `test_the_adapter_owns_no_credentials_and_no_routing` greps its own source
for `api_key`, `base_url`, `requests.`, `httpx.` and `retry`, so the boundary is
enforced rather than remembered.

### How a certificate reaches an unmodified client

The gate is *no application code change*, which rules out a second endpoint or a
wrapper the caller has to invoke. Two places remain:

- **inline**, on an additive namespaced key `response["control_plane"]`. An
  OpenAI-format client ignores keys it does not know, so the certificate reaches
  a caller who wants it and is invisible to one who does not;
- **out of band**, in the ledger, addressable by the request id the caller
  already has.

Both, because a field in a response the caller may discard is not an audit
trail, and a ledger the caller cannot see does not make the demo work.

### The inline path claims less than the async path, and says so

They are not the same check at different speeds. An inline certificate claims
what the fast tier supports and lists the deep checks in `unchecked`, with
`deep_checks_pending` on the object. Presenting an inline result as though the
deep checks had passed would be the unbacked claim this project refuses: a
caller would read an `ALLOW` meaning *nothing fast found anything* as *nothing
found anything*.

### A budget overrun is recorded, not enforced

The tempting behaviour is to drop the certificate when the checks exceed the
profile's inline budget, so the latency promise holds. That is wrong in the
direction that matters: the response goes out uncertified and **looks identical
to one that passed**. A slow check is a fact about the deployment; it belongs on
the certificate where somebody can see it, not disappeared to protect a number.

So the certificate is always issued and the overrun goes in `unchecked` with
both figures.

### A malformed upstream response still gets a certificate

Tolerant of `message.content` and the older `text`, and returns empty rather
than raising when neither is present. A broken upstream is not a reason to fail
the certificate — it is a reason for the certificate to say nothing was checked,
which is a statement. A missing certificate is an absence nobody notices.

---

## 095 - Round 2 moved to the repository root; 021 is superseded on location, not on reasoning

Block E, E.1-E.2. `021` decided Round 2 would be built under `round 2/`, and
rejected the alternative it called *"overlay Round 2 on the root — destroys the
Round 1 result and its history."* That rejection was correct and still is.

**The clause that needed superseding is the location, not the reasoning.** An
overlay writes Round 2 over Round 1's files and loses them. What happened here
is the opposite operation: Round 1 was moved *whole* into `round1/` with
`git mv`, in its own commit, before anything else moved. Nothing was deleted,
no file was recreated, and `git log --follow` reaches the original commits
through both renames — verified on `round1/src/probe.py` (5 commits, back to
the package skeleton), `round1/README.md` (4, back to the pre-run version) and
`round1/results/economics.json`. 021's reason for refusing the overlay is
honoured by the method, not violated by the outcome.

**Why it had to move at all.** The repository is named `controlplane`. Cloning
it landed a reader on Round 1's README — the probe experiment, headline
`lift 2.3x` — while the warrant system sat one level down in a directory whose
name contains a space. Every `make` target, demo path and command a judge would
type had to quote `"round 2"`. Block E's whole premise is that a judge clones
the repo and within ten minutes understands what it claims and runs something
that works. No README written inside `round 2/` fixes that, because the root one
is what renders.

### What made the move safe was built in 2024 hindsight, not decided here

The audit that preceded the move (E.1, report-and-stop) found that **no
artifact references an eval set by path.** Every one carries `eval_set_id` and
a content hash; `evalsets/manifest.json` stores a bare filename resolved against
`paths.evalsets_dir`. The only path-shaped string inside any artifact is
`results/controlplane.db`, an echo of `store.path`, and `results/` kept its
name. That property is why 172 files moved without a single artifact being
regenerated, and it was a consequence of `003` and `009` rather than of anything
done in this block.

### Two classes of string were deliberately left stale

**`provenance.dirty_paths`.** Artifacts record `round 2/CLAUDE.md`,
`round 2/src/config.py` and similar. These are a record of which files were
uncommitted when a number was produced — facts about a past tree, not pointers
to inputs. They are not updated. Rewriting them would falsify provenance, which
is the one edit this project cannot make.

**The synthetic generator literals.** `controlplane/validation/synthetic.py`
still says `"generator": "src.validation.synthetic.synthetic_evalset"` and
`...synthetic_cache`. They sit inside `construction`, `construction` feeds
`EvalSet.content_hash`, the content hash **is** the envelope id, and the
envelope id is the third element of the warrant key (invariant 1). Verified
directly before the rename: the same construction dict hashes to `6b7654bb…`
with `src.` and `a682d25f…` with `controlplane.`. Renaming them would silently
re-issue every synthetic fixture under a new envelope and orphan the warrants in
`results/fixtures/`. They are frozen identities, not import paths, and both are
commented in place saying so.

`canary-src`, an eval-set id in `tests/test_smoke.py`, is untouched for the same
reason.

### The package was renamed in the same operation, on purpose

`src/` became `controlplane/`: 74 files, 248 import sites. `src` is a directory
name, not a package name; `make smoke` is meant to prove the package imports,
and `import src` proves nothing about a project called controlplane. It also
blocks installing the tree from a lockfile, which E.7 needs.

Done now rather than later because E.9 freezes the layout once the demo harness
lands, and the demo will hardcode paths. A rename afterwards is the thing E.9
exists to forbid. 470 tests passed before the rename and after it.

**Past entries in this file still say `src/`.** They are not edited. This file
is append-only and those references were true when written; this entry is where
a reader learns the package moved.

### A reviewer could fairly object

That the repository now mixes two projects at different depths, and that
`round1/` reads as demoted. It is superseded, not abandoned — its result
reproduced within its published interval, `round2` carries its operating point
forward as a declared input, and both its README and the root README say so in
their first lines. The alternative, a second repository, was rejected by 021 for
a reason that has not changed: it would break the trace from the carried-forward
operating point back to the run that measured it.

---

## 096 - Phase 6 economics was specified and never built, and five contracts still cite it

Found by the E.1 path audit, which was looking for something else.

`src/economics/sizing.py` — now `controlplane/economics/sizing.py` — is cited as
load-bearing in **five** places: `config.yaml`, `CLAUDE.md` (the scenario-mixing
pitfall), `SPEC.md` §6.4, `TASKS.md` Phase 6, and `KICKOFF.md`. `SPEC.md` §12
lists `test_no_scenario_mixing` in the test table. **Neither the module nor the
test exists.** Phase 6 was specified and the timeline went elsewhere.

**The claim those five citations make is that no economic figure is typed by
hand, because a module derives all of them from the single `workload` block.**
That enforcement does not exist. What does exist:

- `config.yaml` still declares exactly one `workload` block, so there is one
  scenario to mix figures *from*, and no second one anywhere in the tree;
- no economic figure appears in `results/RESULTS.md`, the warrant matrix, or any
  committed artifact. The measured outputs are AUROC, recall, precision, flag
  rate, lift and their intervals. Nothing downstream consumes `workload`.

So the invariant is currently satisfied by absence rather than by construction —
there are no economics figures to mix. That is a much weaker guarantee than the
one the contracts describe, and it stops holding the moment anyone writes a cost
number into the proposal or the deck.

**Decision: log it and declare it, do not build it.** Building a sizing module
now means new numbers with no measurement behind them, days before submission,
in a block whose purpose is presentation. The honest form is a declared gap.

**Consequences.** `docs/LIMITATIONS.md` carries it. Any cost or headcount figure
in `docs/PROPOSAL.md` or the deck is hand-derived and must be labelled as a
declared estimate, not a measured result — the same treatment as the two
carried-forward Round 1 numbers in `021`. The contracts keep their references
rather than having them quietly deleted: a spec that describes an unbuilt module
and says so is scoped; one edited to hide it is not.

**A reviewer could fairly object** that a spec citing a module that does not
exist is exactly the failure this project is about — an artifact pointing at
nothing. That is the right objection, it is why this entry exists, and the gap
was found by the repository's own audit discipline rather than by a reader.


---

## 097 - The layout is frozen. Additive changes only from here.

Block E, E.9. Declared now, before the demo harness begins, because the demo
will hardcode paths and a move afterwards breaks it in ways that surface during
rehearsal or during the presentation.

**Frozen as of this entry and the `repo-v1` tag.** No moves, no renames. New
files, new documents and new tests are fine and expected; relocating an
existing one is not, whatever it would tidy.

### What the freeze covers

Directory names — `controlplane/`, `results/`, `evalsets/`, `policies/`,
`scripts/`, `tests/`, `demo/`, `docs/`, `notebooks/`, `round1/`. Eval-set ids,
detector ids, operating-point ids, decision numbers, test names, tag names.
Everything an artifact, a commit message or the notes ref might refer to.

The relocation that preceded it is `095`, and its mapping is `docs/PATHS.md`.
That work is the reason the freeze can be declared with confidence: the audit
that came first established that no artifact references an eval set by path, so
there is nothing left to move that would be worth the risk.

### What Block E found that nothing else would have

Worth recording, because the argument for doing an audit before a
reorganisation is usually made in the abstract and it paid twice here in
concrete terms.

1. **`controlplane/economics/sizing.py` does not exist**, and five contract
   documents cite it as load-bearing. Found by the E.1 path audit, which was
   looking for something else entirely. Logged as `096`; a judge grepping
   `SPEC.md` would have found it in seconds.

2. **The synthetic generator literals are inside a content hash.** Renaming
   `src.validation.synthetic.synthetic_evalset` to `controlplane....` would
   have re-issued every synthetic fixture under a new envelope id and orphaned
   the warrants in `results/fixtures/` — silently, since nothing errors when a
   hash changes, the numbers simply stop belonging to anything. Caught by
   checking rather than by assuming: the same construction dict hashes to
   `6b7654bb…` with `src.` and `a682d25f…` without.

Two more surfaced while building the enforcement rather than the code:

3. **Five of the test names in the first draft of `docs/CASES.md` were wrong** —
   four renamed at some point, one never written. `test_every_case_names_a_real_test`
   caught all five on its first run. A case matrix citing a test that does not
   exist is worse than no matrix, because it reads as coverage.

4. **`test_the_readme_test_count_is_the_real_one` failed on itself.** Adding it
   moved the count from 500 to 501 and it refused to pass until the README said
   so. That is the intended behaviour of a check that has no exceptions.

### The enforcement that replaces discipline

Three things that used to be somebody's job to remember are now build failures:

- **A hand-edited README number.** `make verify` parses the claim table,
  resolves each field in its artifact and compares at the quoted precision. The
  negative tests feed it an edited value, an edited interval, a missing artifact
  and a dead field, and assert each is caught; an empty table is an error rather
  than a vacuous pass.
- **A case matrix drifting from the suite.** Every test it names must exist.
- **A dependency claim expiring.** The Presidio refusal now reads
  `[presidio-stock==2.2.364] …`, and a test fails if the installed version
  leaves the pin. The claim is about a release, which stays true, rather than
  about a library, which does not.

### A reviewer could fairly object

That freezing the layout days before submission optimises for the demo rather
than for the repository, and that `docs/` still holds build contracts a
finished project would not ship. Both fair. The answer to the first is that
this is exactly when a freeze is worth most; to the second, that `SPEC.md` and
`TASKS.md` are the record of what was planned against what was built, and
`096` is only legible because both are still there.


---

## 098 - `construction` records the inputs to generation, never the identity of the generator

The mirror image of `092`, found from the other side, and the fourth instance of
one boundary.

`092` established where the content hash *ends*: facts learned after a set was
frozen live beside it, because the hash's job is immutability rather than
completeness. This is the same boundary violated in the opposite direction —
something that was never a property of the data got written *into* the identity.

### What was found

`controlplane/validation/synthetic.py` writes, into the `construction` dict that
feeds `EvalSet.content_hash`:

```python
"generator": "src.validation.synthetic.synthetic_evalset"
```

That string is a **module path**. The envelope id — and therefore the third
element of every warrant key issued on a synthetic fixture — was coupled to the
package layout. Renaming `src/` to `controlplane/` in `095` would have re-issued
every synthetic fixture under a new envelope and orphaned the warrants in
`results/fixtures/`, and nothing would have raised: a changed hash does not
error, the numbers simply stop belonging to anything.

Verified rather than assumed before the rename: the same construction dict
hashes to `6b7654bb…` with `src.` and `a682d25f…` with `controlplane.`.

### The principle

**`construction` records the *inputs* to generation. It never records the code
identity of the generator.**

- **Inside:** seed, `n_items`, requested base rate, items per question, whether
  the set is long-context, declared splits, the warning text. Every one is a
  parameter someone chose, and changing any of them genuinely produces different
  data that genuinely deserves a different identity.
- **Never inside:** module paths, class names, function references, package
  names, file paths, version strings of the generating code. None of them
  changes the data. All of them change when the code is reorganised, which is
  the one thing that must not move an envelope id.

The generator's identity is not lost — it belongs in `provenance`, which every
artifact already carries, alongside the git commit and the config hash. That is
where "which code produced this" is answered, and provenance is deliberately
outside the hash for exactly this reason.

### Why this is an entry and not two comments

The two known sites are frozen with comments in place, because correcting them
now would do the damage the rule exists to prevent — the strings are stale, and
that is strictly better than an orphaned fixture. `docs/PATHS.md` records both.

But freezing two sites is a patch, and the failure is silent and general: any
future `construction` dict capturing a dotted path has it, and nobody would
notice until a rename. So `test_construction_records_inputs_not_code_identity`
scans every `construction` and `extra` dict the builders produce for a string
that looks like a dotted path and resolves to a real module in this package, and
fails naming the key. The two frozen literals are allowlisted **by exact value**,
with the reason, so the allowlist cannot silently absorb a third.

That converts "we caught two" into "this class cannot recur".

### A reviewer could fairly object

That the frozen literals now lie — they name a module that no longer exists. They
do, and the comment beside each says so. The alternative was to re-issue every
synthetic fixture to fix a cosmetic inaccuracy in a string nothing reads at
runtime. `092` made the same trade and for the same reason: an identity that
moves when someone tidies is worse than an identity that is slightly ugly.


---

## 099 - Two of the proposal's numbers are derived, and the price list is still not built

**Amends `096`**, which declared Phase 6 economics unbuilt and every cost figure
a declared estimate. That remains true of every *money* figure. It was too
strong about two quantities that need no new measurement.

### What is now derived rather than declared

**1. The abstention floor.** For traffic with measured base error rate `mu` and
a declared risk ceiling `alpha` on what is kept, the minimum fraction any
selector must abstain on is

```
a  >=  (mu - alpha) / (1 - alpha)
```

assuming a **perfect** selector — one that abstains only on errors and never on
a correct response. Derivation in `controlplane/economics/feasibility.py`;
checked against an exhaustively-searched optimal selector on a 1000-item
population rather than against itself, because a closed form asserted by the
test that computes it proves nothing.

On `triviaqa-2400-t960`, base error rate **0.4510**:

| declared target risk | minimum abstention | most that can be served |
|---|---|---|
| 0.20 | 0.3138 | 68.6% |
| 0.10 | 0.3900 | 61.0% |
| 0.05 | **0.4221** | 57.8% |
| 0.02 | 0.4398 | 56.0% |
| 0.01 | 0.4455 | 55.5% |

This answers *"why not just tighten the threshold until the error rate is
acceptable?"* with an impossibility result instead of an opinion. At this base
rate, a 5% residual risk target requires abstaining on at least 42% of traffic
**however good the detector is**. No threshold, ensemble or amount of tuning
gets under it.

**2. How far each operating point is from that floor.** Fully measured — base
rate, recall and flag rate all from one envelope, no declared input anywhere:

| profile | abstains on | ships residual risk | perfect selector would abstain on | cost |
|---|---|---|---|---|
| `customer_support` | 0.1062 | 0.3951 | 0.0925 | **1.15x** |
| `internal_knowledge` | 0.1865 | 0.3547 | 0.1493 | **1.25x** |
| `decision_support` | 0.4677 | 0.2231 | 0.2934 | **1.59x** |

The right way to read this: the detector buys most of what is theoretically
available at the loose end and less at the tight end. `decision_support` costs
59% more review than its own residual risk strictly requires — which is a real
number about a real gap, and far more useful than a claim of optimality.

### A consistency check the sweep forced

`test_efficiency_is_never_below_one_for_an_achievable_point` swept the
(recall, flag rate) grid and produced an efficiency of **0.65** — an operating
point apparently beating a bound no selector can beat.

The combination was impossible. `mu * recall` is the share of all traffic that
is a *caught error*, and it cannot exceed the share flagged. At `mu=0.4510`,
`recall=0.3` implies catching 13.5% of traffic as errors while flagging 10%.
`achieved_risk` now refuses that, naming it as rates that cannot describe one
envelope — which is the scenario-mixing error `CLAUDE.md` names, surfacing in a
new place. The three real operating points all satisfy it.

### What is still not built

`sizing.py`, the computed price list, the stratified estimator, the Neyman
allocation schedule, the blinded label queue, Cohen's κ, and
`test_no_scenario_mixing`. **Every cost, headcount and saving figure remains a
hand-derived declared estimate and must be labelled one.**

`controlplane/economics/` now exists, which means the `NOT BUILT` marker in
`CLAUDE.md`'s layout block is no longer accurate as written and has been
corrected to name the two modules that exist and the one that does not. The
package docstring says the same thing in the place someone importing it will
read.

`review_volume` derives items per month from the measured flag rate and one
declared workload, and labels every input by origin — so a figure computed from
it carries "this half was measured, this half was declared" rather than losing
it in a spreadsheet.

**A reviewer could fairly object** that a floor computed at TriviaQA's 0.4510
base error rate says nothing about production traffic, where the declared rate
is 0.03. Correct, and the artifact carries the envelope id for exactly that
reason. The floor is a statement about *this* distribution; what transfers is
the shape of the argument, not the number. At `mu=0.03` and `alpha=0.01` the
floor is 0.0202 — the same inequality, a far smaller consequence, and the
honest thing is to say which one is being quoted.


---

## 100 - Absence is not a pass: a check that did not report is a check that did not run

Named once here because it has now appeared four times in four different
components, and the fix each time was local while the principle was not.

**The rule.** When a verifier cannot see the outcome of a check, it reports
that check as *not having run*. Never as passed. This applies whether the
check was skipped deliberately, could not run for want of an input, crashed,
or simply did not appear in whatever the verifier was reading.

The four instances, in the order they surfaced:

| where | what absence looked like | what it would have meant |
|---|---|---|
| `081` | an interval containing zero | "no difference", when the sample could not resolve one |
| `086` | a holdout with 5 affected positives | "the fitting was harmless", when the set could not detect it |
| `089` | a detector with no category declared | a warrant against labels meaning something else |
| **this** | a tier missing from the gate's view | **a check reported green that never ran** |

The fourth is the sharpest, because the component that failed is the one whose
entire job is reporting accurately.

### What happened

The clean-clone gate detected verify's outcome by matching prose in the last
25 lines of captured output: `if "SKIPPED" in verify.tail`. One string, one
row, and a note reading *"the claim table was still checked against the
committed artifacts."*

That was true when verify had two tiers. When it grew a third (`results/scores/`,
the score tier that lets a clean clone re-derive anything at all), the sentence
became false in both directions at once: the score tier had reproduced 24
comparisons **inside the clone** and was filed as not having run, while the
note claimed the claim table was all that had been checked.

Then the corrected version -- one row per tier, still matched on prose --
immediately reported the *claim table* as skipped, because at three tiers its
success line had scrolled off the 25-line tail. Conservative, and still wrong.

Neither version ever reported a failing check as green. Both understated
coverage, which is the safe direction. It is still a gate that cannot be
trusted to say what it verified, and "wrong in the safe direction" is the
excuse this project refuses everywhere else.

### The fix, and why prose was the wrong mechanism

`scripts/verify.py` now emits a machine-readable line per tier, last in its
output so truncation cannot reach it:

```
TIER|1|claim_table|OK|31/31
TIER|2|frozen_scores|OK|24/24
TIER|3|activations|SKIPPED|0/0
```

The gate parses that. Prose is for people and moves whenever the wording
improves; a contract does not. Three behaviours are fixed by it rather than
by discipline:

- a tier reporting `DRIFT` **fails** the gate;
- a tier absent from the summary is reported **skipped**, never passed;
- **no tier lines at all is a failure**, not an empty pass. That case would
  otherwise be the worst one: a verifier whose output was lost entirely,
  producing a gate with nothing to object to.

`test_prose_alone_is_not_enough` feeds the gate output that says all the right
things in words and asserts it is **rejected**. Without that test the fix
decays the next time someone improves the wording.

### A reviewer could fairly object

That parsing a pipe-delimited line is fragile in its own way, and that the
gate should call verify as a library rather than scraping a subprocess. Fair,
and it is the better design. The gate runs verify **inside a fresh clone under
that clone's interpreter**, which is the whole point -- it is testing the
tree a judge receives, not this process's imports. Crossing that boundary
needs a serialised contract of some kind; this is the smallest one that
truncation cannot break, and it is versioned with the code that emits it.


