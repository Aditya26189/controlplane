# PROPOSAL.md

> **Scaffold plus two derived results.** The prose argument is a content
> deliverable, written separately. What is here is the quantitative spine it
> should be built on, so that when it is written the numbers are already
> checked rather than typed.

---

## The constraint this document is written under

`controlplane/economics/sizing.py` — the price list — **does not exist**. It is
cited as load-bearing in five contract documents and was never built
(`DECISIONS.md` 096). So:

> **Every cost, headcount, saving and ROI figure in this document is a
> hand-derived declared estimate and must be labelled one.**

The same treatment the two carried-forward Round 1 numbers get in
`DECISIONS.md` 021. Writing an unlabelled money figure here is the single
fastest way to undo what the rest of the repository is for.

**Two things are exempt**, because they need no new measurement and no cost
model. Both are derived by `scripts/11_feasibility.py` into
[`results/feasibility.json`](../results/feasibility.json) and checked by
`tests/test_economics.py`. They are below.

---

## 1. The abstention floor — an impossibility result

Every threshold conversation reaches the same question: *why not just tighten
the threshold until the error rate is acceptable?* This answers it with a bound
rather than an opinion.

For traffic with base error rate `mu`, holding residual risk at `alpha` on the
responses you keep requires abstaining on at least

```
(mu - alpha) / (1 - alpha)
```

of it — assuming a **perfect** selector, one that abstains only on errors and
never on a correct response. No detector does better. No threshold, ensemble or
amount of tuning gets under it.

On `triviaqa-2400-t960`, measured base error rate **0.4510**:

| target residual risk | minimum abstention | most that can be served |
|---|---|---|
| 20% | 31.4% | 68.6% |
| 10% | 39.0% | 61.0% |
| **5%** | **42.2%** | **57.8%** |
| 2% | 44.0% | 56.0% |
| 1% | 44.5% | 55.5% |

**The line for a deck:** on this distribution, a 5% residual-risk target means
abstaining on at least 42% of traffic — before any question of how good anyone's
detector is. The choice is not *whether* to build a review path. It is whether
you can say how large it has to be.

**Quote this with its envelope.** `mu = 0.4510` is TriviaQA no-context, a
deliberately hard benchmark. At the declared production rate of `mu = 0.03`, a
1% target has a floor of **2.0%** — the same inequality, a far smaller
consequence. Say which one you are quoting. The artifact carries the envelope
id so the two cannot be confused.

## 2. How far the three profiles are from that floor

Fully measured. Base rate, recall and flag rate all from one envelope; no
declared input anywhere in this table.

| profile | abstains on | ships residual risk | a perfect selector would abstain on | cost of ours |
|---|---|---|---|---|
| `customer_support` | 10.6% | 39.5% | 9.3% | **1.15×** |
| `internal_knowledge` | 18.6% | 35.5% | 14.9% | **1.25×** |
| `decision_support` | 46.8% | 22.3% | 29.3% | **1.59×** |

**How to read it.** The detector captures most of what is theoretically
available at the loose end and less at the tight end. `decision_support` costs
59% more review than its own residual risk strictly requires.

That is a real number about a real gap, and it is worth more than a claim of
optimality. It is also the honest form of a roadmap: the 1.59× is what a better
detector would recover, and it is bounded — nothing gets below 1.0.

## 3. Review volume, per profile

Measured flag rate against **one declared workload** — 200,000 interactions a
month at a declared 3% production error rate. The volume and the production
error rate are declared; the flag rate and recall are measured.

| profile | flagged/month | true positives | false positives | errors missed |
|---|---|---|---|---|
| `customer_support` | 21,250 | 1,303 | 19,947 | 4,697 |
| `internal_knowledge` | 37,292 | 2,162 | 35,130 | 3,838 |
| `decision_support` | 93,542 | 4,420 | 89,121 | 1,580 |

**This is not a cost.** Converting items to money needs the price list, which
is not built. It is the volume a cost model would take as input, and it is
labelled measured-vs-declared throughout so that conversion cannot quietly lose
which half was which.

**The row that matters is the last one.** At `customer_support`, 4,697 errors a
month reach users unflagged. That is the number a warrant makes sayable, and it
is the reason the profile declares `on_calibration_drift: REFUSE`.

## 4. What a recall claim costs to maintain

Reviewed items needed for the recall interval to reach a declared half-width,
sized at the measured recall rather than at the conservative `p = 0.5`:

| profile | ±0.05 | ±0.02 |
|---|---|---|
| `customer_support` | 175 | 1,094 |
| `internal_knowledge` | 238 | 1,483 |
| `decision_support` | 200 | 1,249 |

The design effect (0.67) is **declared, not measured** — `config.yaml` says
"measure, don't assume", and it has not been measured. Treat these as the right
order of magnitude, not as commitments.

---

## Intended prose sections

- The problem, in one paragraph, for someone who does not know what a probe is.
- Why "we have guardrails" is not an answer, and what a warrant adds.
- The measured evidence — drawn from the [README claim table](../README.md),
  nothing new.
- **The feasibility bound**, as the answer to "just tighten the threshold".
- What is refused, and why that is the strongest part of the submission.
- Scope and what deployment would need, bounded by
  [LIMITATIONS.md](LIMITATIONS.md).
- Costs, if any, each labelled a declared estimate with its assumptions.
- **The regulatory argument**, led by the Digital Omnibus rather than by the
  Uber fine: a right generates a fine when breached, a permission list
  generates a standing evidence obligation, and a demonstrated, time-bounded,
  revocable condition is what a warrant is. Every figure from
  [EXTERNAL_FIGURES.md](EXTERNAL_FIGURES.md).
- **The risk section, built on the self-catch log.** Fifteen errors found
  internally before external review, several flattering. The strongest is not a
  code bug: a hypothetical design effect crossed a document boundary, lost the
  context that made it hypothetical, and reached a sentence about this
  project's costs. That is the failure this product exists to prevent,
  occurring in the medium a customer reads.

## Rules for anything added here

1. A number that is measured goes in the README claim table with its artifact
   and field, where `make verify` checks it.
2. A number that is derived cites the script that derives it and the artifact
   it lands in.
3. A number that is neither is a **declared estimate** and says so on the page,
   not in a footnote.
4. A number **about the world** — a court award, a regulatory fine, a standard's
   clause number — is an **external figure**, and none of the three rules above
   fits it. It is not measured here, not derived here, and calling it a declared
   estimate would be false. It goes in
   [EXTERNAL_FIGURES.md](EXTERNAL_FIGURES.md) with its source, its verification
   state, who verified it and when, and any caveat that must travel with it.

   **A figure not in that register does not go in this document**, and no
   section is written until every external figure in it is verified. A figure
   still unverified when its section is written is **dropped, not softened** —
   softening is how a hypothetical `DEFF 1.60` from a planning document reached
   a sentence about this project's certification cost (`DECISIONS.md` 113, 115).
