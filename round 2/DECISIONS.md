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

<!-- New entries below. Do not edit anything above this line. -->
