# EXTERNAL_FIGURES.md

> **The register that gates the proposal.** A figure about the world — a court
> award, a regulatory fine, a standard's clause number — is not measured here,
> not derived here, and not a declared estimate. `PROPOSAL.md`'s three classes
> do not cover it, so without a fourth it enters as ordinary prose. That is the
> unguarded import path in `DECISIONS.md` 113, and the route that brought a
> hypothetical `DEFF 1.60` and a wrong standard number `prEN 18284`.
>
> **A figure not in this register does not go in the proposal.**

## Provenance of this register

Every entry below was **verified by the project author against the cited
sources on 2026-08-30**. It was *not* verified by this repository, and not by
the assistant that wrote this file. Recording that distinction is the point:
a register claiming repo authority for facts the repo cannot check would be
the same defect one level up (`DECISIONS.md` 115).

`verified+primary` means the author reached the issuing body's own text —
a tribunal decision, a regulator's announcement, a regulation in the OJ.
`verified+secondary` means corroborated only through vendor or trade press,
and it is cited **as a claim by that party**, never as fact.

---

## The Article 22 anchor

### Uber — Autoriteit Persoonsgegevens, 21 August 2026 — €824,990,000

**`verified+primary`** (the AP's own announcement; CNIL co-operation under
one-stop-shop corroborated via LexisNexis).

Fully automated decisions about drivers — accounts deactivated on suspicion of
fraud or on persistently low ratings — with a separate finding that Uber did
not sufficiently inform drivers the decisions were automated. Conduct period
2018–2022. Second-largest GDPR penalty ever, behind Meta's €1.2bn.

Roughly **1.85% of 2025 turnover, about 46% of the 4% statutory ceiling**. The
realised percentage is the number to quote; the ceiling is what everyone else
quotes and it is weaker because it is hypothetical.

**Two caveats that must travel with it.** Omitting either invites a reader who
follows the citation to supply it themselves, which is worse than saying it:

1. **Uber has appealed** and disputes the amount as disproportionate. An
   analysis from May 2026 found nearly **40% of the €7.1bn in announced GDPR
   fines** had been annulled or remained under legal challenge. Cite it as an
   **opening position, not a transfer of funds.**
2. **Article 22 itself is moving** — see the Digital Omnibus entry below.

**Why this anchor and not Foodinho.** It is the decision-support profile almost
exactly: a score crossing a threshold, an action taken, no human in the loop,
no route to contest.

### Foodinho S.r.l. (Glovo group) — Garante, 22 November 2024 — €5,000,000

**`verified+primary`.** 35,000+ couriers; a prior €2.6M in July 2021 (EDPB
newsroom). Article 22 element corroborated.

**Accurate, and demoted.** Superseded as the anchor by Uber on recency and
scale. Retained here because a demoted figure that stays in the register cannot
be quietly re-promoted without passing this gate again.

---

## The Digital Omnibus — and why it is an argument, not a caveat

Published **19 November 2025**. Would convert Article 22 from a **right held by
the individual** into a **list of conditions under which automated processing
is permitted**.

**`verified+primary`** (Commission publication).

Stated *first*, before the Uber figure, so the anchor's legal instability is
disclosed rather than discovered. Then landed as the second and stronger
argument for the product:

> A right generates a **fine** when it is breached. A permission list generates
> a **standing evidence obligation** — the controller must demonstrate it meets
> the conditions, continuously, for as long as it processes.
>
> A demonstrated, time-bounded, revocable condition is what a warrant *is*.

So the ground shifting under the anchor does not weaken the pitch. It moves the
market from *"pay when you are caught"* to *"show your working, continuously"*,
which is a larger and more durable one.

---

## Insurance

### Armilla / Lloyd's — April 2025 standalone AI liability policy

**`verified+secondary` — vendor and trade press only. Cite as a vendor claim.**

Chaucer leading the binder; coverage naming hallucinations, model drift,
mechanical failures and deviations from expected behaviour. Limits expanded to
**$25M per insured** by early 2026. Vanguard AI launched 10 February 2026
($25M+ AI aggregate, $10M cyber).

Never write "the market prices this at $25M". Write "Armilla and Chaucer state
that…".

### Verisk generative-AI exclusion endorsements — effective 1 January 2026

**`verified+secondary`.** The counterweight, and it strengthens the argument
rather than undermining it: general coverage is **narrowing** at the same time
specialist capacity is **pricing AI performance**. Both movements point at the
same gap — nobody can price what nobody can measure.

---

## Standards and supervisory guidance

### EU AI Act timing — Regulation (EU) 2026/1744

**`verified+primary`.** Published in the OJ **24 July 2026**, in force
**27 July 2026**. Defers Annex III high-risk obligations to **2 December 2027**
and Annex I to **2 August 2028**. **Article 50 transparency did not move.**

**Recital 40 gives TWO reasons** for the deferral, and quoting one is
catchable: delayed availability of **standards, common specifications and
guidance**, *and* delayed establishment of **national competent authorities**.

The first is the load-bearing half — enacted EU law formally recording that the
measurement tooling was not ready — but it is quoted with the second present,
not instead of it.

### Article 15 accuracy and robustness — `prEN 18229-2`

**`verified+primary`** (CEN/CENELEC JTC 21 work programme).

- Accuracy/robustness: **`prEN 18229-2`**
- NLP evaluation methodology: **`prEN ISO/IEC 23282`**
- `prEN 18284` is **dataset quality and governance** — **not** Article 15
- `prEN 18285` is **conformity assessment**

**`prEN 18284` was cited for Article 15 in an earlier draft and is wrong.** It
would discredit the citation in front of anyone who knows the work programme.
Neither of the correct standards is finished or OJ-cited, which is the point
being made.

### ISO/IEC 42001 — clause numbers

**`verified+secondary`, and the standard is paywalled.**

A.6.2.4, A.6.2.5, A.6.2.6, A.6.2.7, A.6.2.8 and Clauses 9.1 / 8.4 / 10.2
corroborate across independent secondary sources. But those sources give the
Annex A control count as **38, 39 and 42** — three different numbers for a
countable property of the same document, which is the tell that none of them
is reading the text.

**Cite by function, not by number**: *"the deployment control requiring
documented release criteria"*. That preserves the mapping argument and is
unfalsifiable by phone. If a clause number is load-bearing, buy the standard.

### "350+ certified organisations" — **`dropped`**

ISO/IEC 42001 is not yet in the ISO Survey and **no official worldwide
certificate count exists**. The ~350 figure traces to press-release
compilations.

Reword to *"a few hundred organisations by public count, with no official
registry."* Adoption skewing to cloud/AI platforms and professional services
**strengthens** the argument that the certificate travels with the
**organisation** rather than with the detector.

### SR 11-7 — superseded, not rescinded

**`verified+primary`.** **SR 26-2 superseded SR 11-7**, and also rescinded
SR 21-8 and several OCC bulletins. **"Superseded" is the safer verb**; an
earlier draft said "rescinded 17 April 2026".

---

## Dropped and unverified

### IBM / Ponemon breach cost — **`dropped`**

Vendor-sponsored, and the headline is a **mean, not a median**. Nothing in the
argument rests on it. If used at all, label it vendor-sponsored, say it is a
mean, and prefer the component decomposition.

### The "1.000 false-positive rate" claim — **`unverified`, do not ship as written**

An earlier draft said *"one shipped product posting a 1.000 false-positive rate
on long-context inputs."* The clearest published 100% FPR result is **CAPTURE
(arXiv 2505.12368)**, and the sentence is wrong on two counts:

- **PromptGuard is Meta's open-weights model, not a shipped commercial
  product.**
- The condition was **context-aware over-defense benchmarking**, not
  long-context inputs.

Either trace it to its actual source and restate the condition precisely, or
replace with **BELLS-O (arXiv 2606.20668, June 2026)** — independent, 28
systems across 17 providers, reporting FPR alongside latency and cost.

### EU Platform Work Directive — human decision-making for account
suspension/termination from 2 December 2026 — **`unverified`. Do not use.**

Single non-primary source. If true it is a hard date worth its own check,
because it would be a second Article 22-shaped obligation with a near horizon.

---

## Rule

`PROPOSAL.md` admits three classes of number: **measured**, **derived**,
**declared estimate**. This file defines the fourth, **external**, and the rule
that governs it:

> **No section of the proposal is written until every external figure in it is
> `verified+primary` or `verified+secondary`.** A figure still `unverified`
> when its section is written is **dropped**, not softened.

Softening is how `DEFF 1.60` survived: a number that could not carry its claim
was given a hedge instead of a decision.
