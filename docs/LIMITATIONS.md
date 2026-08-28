# LIMITATIONS.md — scope, declared gaps, open items

Read this before quoting any number from [README.md](../README.md).

Nothing here was found by a reviewer. Every item was found by this project's
own controls, pre-registrations or audits, and each names the decision entry
that records it. That is the point of the list: a system whose value rests on
saying what it does not know has to be willing to say it about itself.

---

## 1. What was measured, and on what

| | |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct, NF4 4-bit. **One family, one size.** |
| Hallucination envelopes | TriviaQA no-context: 2,400 items, split by question into 960 / 480 / 960 |
| PII envelopes | `hinglish-pii-200`, `hinglish-pii-200b`, `hard-negatives-200` — **hand-built by us**, synthetic identifiers |
| Real traffic | **None.** No production data of any kind touched this. |
| Human labelling | **None.** Correctness labels come from TriviaQA alias matching; PII labels are construction-time. |

No result here transfers to another model family, another task, or real
traffic without being re-measured. That is not a caveat bolted on — it is the
argument. A detector's bounds are a property of a detector *and an envelope*,
and this repository exists to say so.

---

## 2. The four that change how you read the claim table

### 2.1 Calibration drift is detectable at 25%, not at 10%

Separating a 25% deviation from a declared flag-rate budget needs **n ≥ 1441**.
These envelopes were measured at n = 600 to 960. Every budget claim in the
matrix is therefore refused outright or marked **unresolved** — the interval
extends past the band and this sample could not have narrowed it.

This is a limitation of sample size, not of method. The *ranking* claims on the
same warrants are supported: AUROC intervals at this n are tight enough to
separate the detectors from each other, which is what the drift beat rests on.
Closing it needs more test items, not more code.

`DECISIONS.md` 069, and the sensitivity table in
[results/RESULTS.md](../results/RESULTS.md).

### 2.2 Every published recall interval is conditional on a threshold set by as few as five items

Thresholds are selected on validation to hit a flag-rate budget. At the
`customer_support` budget, the threshold is determined by **5 negatives** in a
480-item validation split. The published recall interval treats that threshold
as fixed, which it is not.

Propagating the selection noise widens the recall bound by **1.36x to 1.59x**:

| operating point | conditional | selection-aware | widening |
|---|---|---|---|
| `P-customer-support` | [0.1774, 0.2551] | [0.1468, 0.2706] | 1.59x |
| `P-internal-knowledge` | [0.3157, 0.4048] | [0.3009, 0.4269] | 1.42x |
| `P-decision-support` | [0.6935, 0.7747] | [0.6734, 0.7840] | 1.36x |

The claim table quotes the **conditional** interval, because that is what the
artifact's `metrics` block holds, and names the widening in its own row rather
than burying it. `DECISIONS.md` 083, `results/paired_comparison.json`.

### 2.3 Two of the four PII recalls are in-sample

`presidio-enabled_plus_custom` at 0.6176 and `pii-reference` at 0.7941 were
both measured on the set their patterns were written against. The sequence was:
measure stock → read the failures → widen the recognisers → re-measure. That is
a detector fitted to its evaluation data by a human rather than by gradient
descent, and the number is in-sample.

Decomposed by pattern origin, **11 of 34 patterns (32.4%) are spec-derived**;
the rest are fitted, including the spaced and obfuscated UPI/IFSC forms.

`hinglish-pii-200b` was pre-registered and built as an out-of-sample companion.
Every refusal reproduces on it, and every recall moved *up* by 0.03–0.04 —
including for the two detectors that were never fitted, which means the holdout
is marginally easier rather than that the fitting was harmless. **It is
underpowered**: only 5 of 102 positives carry formatting outside what the
recognisers were written against.

The half of the finding the demo rests on — that **stock and fully-enabled
Presidio are refused** — is not affected: neither was fitted to anything.

`DECISIONS.md` 084, 085, 086, 087.

### 2.4 8 of 17 populated matrix cells are synthetic fixtures

They exercise the harness. They are not measurements of a language model. The
renderer **refuses to print their numbers** rather than relying on a reader
noticing a footnote, and their eval sets hash differently so they occupy
different matrix cells and can never be read as measured ones.

The T2 (logprob) and T3 (judge) rungs of the tier ladder are fixture-only. The
tier curve as a *measured* object covers T1 activations and T3 text detectors.

`DECISIONS.md` 034, and the warning at the top of
[results/RESULTS.md](../results/RESULTS.md).

---

## 3. Specified and not built

### 3.1 Phase 6 economics — and five contracts still cite it

`controlplane/economics/sizing.py` and `test_no_scenario_mixing` are cited as
load-bearing in **five** documents — `config.yaml`, `CLAUDE.md`,
[SPEC.md](SPEC.md) §6.4 and §12, [TASKS.md](TASKS.md) Phase 6, and
[KICKOFF.md](KICKOFF.md). **Neither exists.**

The claim those citations make is that no economic figure is typed by hand,
because a module derives all of them from the single `workload` block. That
enforcement does not exist. What does:

- `config.yaml` declares exactly one `workload` block, so there is one scenario
  to mix figures *from*, and no second one anywhere in the tree;
- **no economic figure appears in any committed artifact.** The measured
  outputs are AUROC, recall, precision, flag rate, lift and their intervals.
  Nothing downstream consumes `workload`.

So the invariant currently holds by *absence* rather than by construction —
there are no economics to mix. That is much weaker than what the contracts
describe, and it stops holding the moment anyone writes a cost number into the
proposal or the deck.

**Any cost, headcount or ROI figure in [PROPOSAL.md](PROPOSAL.md) or the deck
is hand-derived and must be labelled a declared estimate, not a measured
result** — the same treatment as the two carried-forward Round 1 numbers.

The contracts keep their references rather than having them quietly deleted. A
spec that describes an unbuilt module and says so is scoped; one edited to hide
it is not. `DECISIONS.md` 096. Found by the E.1 path audit, which was looking
for something else.

### 3.2 Phase 9 action gate

`controlplane/gate/` — the reversibility registry, session Rule-of-Two and
action ladder — was specified and not built. Composition produces an action;
nothing enforces one. Marked NOT BUILT in the `CLAUDE.md` layout block.

### 3.3 Other declared absences

- **No serving layer.** No auth, no rate limiting, no HA, no deployment
  manifests. Ruled out of scope in `CLAUDE.md`; the LiteLLM adapter is an
  adapter and holds no credentials and no routing, enforced by
  `test_the_adapter_owns_no_credentials_and_no_routing`.
- **No second model family and no second task.** Every cross-envelope claim
  here is one model under a distribution shift, not two models.
- **`qwen3guard` and the LLM-judge adapter are not built.** The judge rung is a
  fixture.
- **OPA is not used.** Rego is evaluated through `regopy` (rego-cpp), which is
  not OPA and does not share its evaluation semantics in every corner.
  `DECISIONS.md` 076.

---

## 4. Open items at submission

### 4.1 The D.2 measured pair — **open**

The composition rules of `DECISIONS.md` 088 are implemented and exhaustively
tested against fixtures. What is missing is a **measured pair**: no eval set
carries labels for both a hallucination positive class and a PII positive
class, and the category guard now correctly refuses to manufacture one by
warranting a detector against labels that mean something else.

Construction notes are inside the content hash, so adding a second label to a
frozen set would change its identity and orphan every warrant keyed on it —
the third time that constraint has bitten.

Three ways forward, all requiring a GPU pass, none taken:

1. a dual-labelled eval set — the honest fix, and the only one that makes the
   brief's actual claim measurable;
2. both detectors on `hard-negatives-200`, giving two FPR-only warrants and a
   real composed decision on an envelope where neither can claim recall;
3. report the mechanism as built and tested and the measured pair as an open
   gap — **this is what was done.**

The finding is worth more than the demo it blocked. A system that will happily
warrant a PII detector against hallucination labels is one whose warrants mean
less than they appear to, and nothing else in the build would have surfaced it.
`DECISIONS.md` 089, 090.

### 4.2 `DECISIONS.md` 080 — **unresolved**

080 reports all four pre-registered re-split criteria met. Its criterion 3
turns on `customer_support` loading at n_test 960 against the 673 a 0.10 budget
needs at 25% sensitivity — and §2.1 above says every budget claim at these
sample sizes is unresolved. The two statements are in tension and 080 has not
been superseded with the resolution.

Quote 080's recall numbers, which are measured and reproduce. Do not quote it
as evidence that the calibration question is settled.

### 4.3 The withdrawn claim at `67167ed`

That commit message states the training reduction "cost nothing measurable".
**It is wrong**, the comparison was confounded, and the correction lives in a
git note rather than a rewritten history:

```bash
git fetch origin "refs/notes/*:refs/notes/*"
```

Without that fetch a reader sees the withdrawn claim and not its correction.
The paired comparison found the reduction cost **0.0110 AUROC [0.0026,
0.0200]**, an interval excluding zero. `DECISIONS.md` 081.

---

## 5. Things that are limitations of the *result*, not of the work

Worth separating, because they read as weaknesses and are not.

- **Recall is low.** 0.0794 at a 4.2% flag rate. That is a lift of ~1.9x at a
  ceiling of 2.17x — about 88% of everything attainable at this base rate.
  TriviaQA no-context is deliberately hard; the model is wrong on 46% of it,
  which caps what any triage can beat random sampling by.
- **39 of 56 matrix cells are UNVALIDATED.** This is the expected shape, not a
  gap. UNVALIDATED is the modal state in any real deployment, and the value of
  the matrix is that it can say so.
- **Four warrants are REFUSED.** Also the point. A component correctly refused
  is a stronger artifact than one tuned until it passed.
