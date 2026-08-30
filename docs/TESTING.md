# TESTING.md — what the suite defends

The tests here are not coverage. Each suite exists because a specific failure
either happened or would be invisible if it did — several of them were written
after the thing they now catch had already shipped once.

```bash
python -m pytest tests/ -q                    # the whole suite
python -m pytest tests/test_validation.py -q  # one suite
sh scripts/run.sh python -m pytest tests/ -q  # when you need the real exit status
```

**Never pipe a test run whose result you care about.** `pytest | tail` reports
*tail's* status, so a failing suite reads as green. That trap fired twice here,
the second time shortly after it was documented — which is why the wrapper
exists rather than the rule.

**Contents:** [The four kinds](#the-four-kinds-of-test-here) · [Suite by suite](#suite-by-suite) · [Tests that guard documents](#the-tests-that-guard-documents) · [Adding a test](#adding-a-test)

---

## The four kinds of test here

1. **Construction guards.** A record that would misrepresent what is known about
   it cannot be built. A rate without an interval, a count with one, a refusal
   without a reason, an override without its stratum. These run at
   `__post_init__`, so the invariant holds everywhere rather than everywhere it
   was remembered.
2. **Negative tests.** A checker that passes on the real input proves very
   little — it would also pass if it silently found nothing. So the tampering
   cases are tested too: a wrong value, a dead field, a missing artifact, a
   mutated ledger row.
3. **Gate tests.** Named in [TASKS.md](TASKS.md) before the phase was built.
   `test_hash_chain`, `test_warrant_key`, `test_yield_vs_rate`,
   `test_no_override`, `test_three_states`.
4. **Document gates.** Tests that read the repository's own prose and fail when
   it drifts from the artifacts or from the register that governs it.

---

## Suite by suite

### The records and the store

| Suite | Defends |
|---|---|
| `test_model.py` | Record-model invariants, including two of the three Phase 1 gate tests — the warrant key, and yield-vs-rate |
| `test_store.py` | Ledger behaviour, including `test_hash_chain`: mutating a row **demonstrably** breaks the chain |
| `test_override.py` | An override record the estimator could not use cannot be written |
| `test_calibration.py` | The calibration claim as a warrant's second, separable assertion |
| `test_construction_identity.py` | `construction` records the inputs to generation, never the generator's code |

### Measurement

| Suite | Defends |
|---|---|
| `test_validation.py` | The controls, refusal, leakage, polarity and determinism — the core of the harness |
| `test_warrant_stats.py` | Certification at issuance and the anytime-valid revocation trigger |
| `test_paired.py` | The paired comparison and the ROC geometry — the fix for the confounded comparison that once shipped |
| `test_scores.py` | Frozen scores, and the verification tier they make possible |
| `test_extraction.py` | Everything about extraction that can be checked without a GPU |
| `test_dual_path.py` | The failure class that bit twice: two code paths computing the same thing, one drifting |

### Envelopes and detectors

| Suite | Defends |
|---|---|
| `test_evalsets.py` | Construction, freezing, and which claims a set can support |
| `test_categories.py` | An envelope is a distribution **plus a label definition** — a detector cannot be scored against labels that mean something else |
| `test_resplit.py` | Re-splitting a frozen set, and the cache reuse that licenses |
| `test_presidio.py` | The Presidio adapter, including that the finding is a statement about a pinned version |
| `test_banking_pilot.py` | The pilot, its pre-registration, and its divergence guard |

### Matrix, drift, policy

| Suite | Defends |
|---|---|
| `test_matrix.py` | The matrix and routing on it |
| `test_drift.py` | PSI against a warrant's stored bins, and the honesty flag on top of it |
| `test_drift_response.py` | The revocation ladder, the routing it triggers, and the record it leaves |
| `test_model_version.py` | Changed weights invalidate warrants measured on the old ones |
| `test_policy.py` | The policy engine, including that an unresolvable bundle does not load |
| `test_compose.py` | The four composition cases, written before the code |

### Economics and the demo

| Suite | Defends |
|---|---|
| `test_economics.py` | The abstention floor and the review sizing |
| `test_demo.py` | The demo runs a **real** validation, with real certificates and a real chain |
| `test_demo_beats.py` | Each beat assembles from committed artifacts, **or says why it cannot** |
| `test_gateway.py` | The adapter returns certificates on an unmodified OpenAI-format response |

### The repository itself

| Suite | Defends |
|---|---|
| `test_smoke.py` | Every script actually runs |
| `test_script_wiring.py` | Every call a script makes into `controlplane/` binds to a **real signature** — the largest suite, because a thin wrapper that calls a renamed function fails only when someone runs it |
| `test_config.py` | Config loading, hashing, layer resolution, provenance |
| `test_clean_clone.py` | The clean-clone gate's reporting |
| `test_fixture_guards.py` | A fixture number can never be read as a measurement |

---

## The tests that guard documents

Five suites read the prose and fail when it drifts. They exist because **reading
is not a control** — a register nobody checks is a register nobody consults.

| Suite | Enforces |
|---|---|
| `test_claims.py` | Every number in `README.md` resolves to a field in `results/`, at the precision quoted. The negative cases matter more than the positive one: a wrong value, a wrong interval, a missing artifact and a dead field are all tested to fail |
| `test_cases_matrix.py` | [CASES.md](CASES.md) cannot drift from the suite — every row names a test that exists and an artifact that exists, and every row is covered |
| `test_docs_index.py` | `docs/README.md` cannot drift from `docs/` — every docs page is indexed and every relative link resolves |
| `test_external_figures.py` | The external-figure register gates the proposal: figures marked dropped or unverified must not appear there, and the register may not claim a provenance tier nobody reached |
| `test_fixture_guards.py` | Synthetic fixtures cannot be rendered as measurements |

If you add a number to a document, one of these is what will tell you.

---

## Adding a test

Match the kind to the failure:

- **Something could be constructed that misrepresents itself** → enforce it in
  `__post_init__` and test both directions. The illegal case must raise.
- **A checker could pass vacuously** → write the tampering case. Assert that a
  deliberately wrong input **fails**, not only that the right one passes. This is
  the same argument as the `padding_fault` control: a check that only ever
  passes is indistinguishable from a check that does nothing.
- **A script calls into the package** → `test_script_wiring.py` already binds
  every call to a real signature; keep it that way rather than discovering a
  rename at run time.
- **A document makes a claim** → give it a gate, in the suites above.

And follow the repository rule that produced most of these: when a bug is found,
the commit that fixes it carries the test that would have caught it, and the
commit body records what moved.

---

**See also:** [CASES.md](CASES.md) — every case, its test and its artifact · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for what a failure means · [CONTRIBUTING.md](CONTRIBUTING.md) for the pre-merge checklist.
