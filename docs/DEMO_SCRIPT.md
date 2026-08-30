# DEMO_SCRIPT.md — the five-minute video

> **One rule.** Every number on screen is read from a committed artifact by a
> command you run live. Nothing is a slide. If a command fails on camera, say
> so — this is a project about not hiding the failure mode.

**714 narration words.** At demo pace (150 wpm) that is **4:46** spoken, and
the commands run underneath rather than after. At a slower 130 wpm it is 5:30 —
so if you narrate deliberately, drop the "Point at the note" line in Beat 1 and
the last sentence of Beat 4.

Do not add sentences. If something must go in, take something out.

---

## Before you record

```bash
git status --porcelain          # empty
make verify                     # VERIFIED, 3 tiers
python -m pytest tests/ -q      # green
```

Terminal **110 columns**, large font. `DECISIONS.md` open in a second tab.
Dry-run once.

---

## 0:00 – 0:30 — The claim

**Over a still terminal:**

> Round 1 built a detector. Round 2 asks what happens when an enterprise runs
> several AI use cases at once, each with a different risk signature.
>
> Our answer isn't a better detector. It's a **warrant** — a time-bounded,
> evidence-backed statement about what a score is worth right now, on this
> traffic.

```bash
make verify
```

> Thirty-one documented claims checked against their artifacts, twenty-four
> metric blocks recomputed from frozen scores, three re-derived from raw
> activations. If our documentation drifts from our data, this fails.

---

## 0:30 – 1:15 — BEAT 1: it refuses its own flagship

```bash
python demo/show_beats.py --beat 1
```

> This is the part nobody demos.
>
> Customer support declares a false-positive ceiling of 2%. Certifying that at
> 95% confidence needs **149** clean held-out negatives — **203** across three
> profiles with the multiplicity correction. We don't have them.
>
> So the system **refuses to certify**, and prints the price of lifting the
> refusal. That refusal has no override — no flag, no environment variable, no
> admin bypass. If one existed, the product is theatre.

**Point at the note:**

> And it's uninflated — a clustering correction needs a measured correlation we
> don't have, so we don't assume one.

---

## 1:15 – 2:00 — BEAT 2 + 3: one score, three actions

```bash
python demo/show_beats.py --beat 2
```

> One input. One detector. **One score — 0.8446.**
>
> Customer support **allows**. The internal assistant **redacts**. Decision
> support **escalates** to a human.
>
> The detector didn't change. The policy did. And each row runs under a warrant
> certified at its **measured** operating point — not at the target it wants.
> That target is the one we just refused.

```bash
python demo/show_beats.py --beat 3
```

> Enterprises consume models by API and can't inspect internals. So we measured
> what survives as access degrades — activations, log-probabilities, text-only.
> **Each tier gets its own warrant**, and one that can't clear the bar is
> refused rather than shipped with an asterisk.

---

## 2:00 – 2:45 — BEAT 4: the overlap

```bash
python demo/show_beats.py --beat 4
```

> A fabricated detail about a person is simultaneously a hallucination and a
> privacy concern. Both labels live on the same items — correctness
> **measured**, identifier presence **authored** so nothing co-varies with it.
>
> Every recall here is **held-out**. Stock Presidio: **0.147**, refused.
> Recognisers enabled: 0.314, still refused. Our custom ones: **0.647**, passes.
> Our reference implementation: 0.833.
>
> Two of four **REFUSED** — the refusal is the system working. Most of those
> patterns were fitted on a different set, which is why you're looking at the
> held-out number.

---

## 2:45 – 3:25 — BEAT 5: drift, and the modal state

```bash
python demo/show_beats.py --beat 5
```

> Fifty-six cells. **Thirteen valid, four refused, thirty-nine unvalidated.**
>
> That last one is what most systems hide. A warrant is keyed by detector,
> operating point *and* eval set — most pairs were never tested here.
> **Unvalidated is the modal state**, and it never collapses into valid or
> refused.
>
> Two triggers. **Envelope drift** watches the input distribution move — no
> labels, and no guarantee; it's a heuristic gate. **The revocation bound** is
> anytime-valid over the whole deployment — that needs labelled negatives, and
> it's built and tested but not wired to live traffic.

---

## 3:25 – 4:00 — The live loop

```bash
python demo/run_demo.py --fixture --events 4 --auto-prove
```

> Left, a conventional stack: a score. Right, the same request with a warrant —
> recall and precision with intervals and their n, the envelope hash, an expiry,
> and a certificate chained into a tamper-evident ledger.
>
> Then it revalidates live and runs five negative controls. Label-shuffle is the
> one I'd check: it proves the pipeline returns a null when there's no signal.
> This run is a **synthetic fixture** and the banner says so.

---

## 4:00 – 4:45 — What we got wrong

**Switch to `DECISIONS.md`.**

> Our pilot cleared its issuance bar at 0.5554 against 0.55 — by **0.0054**. So
> we swept the bootstrap seed 400 times. The mean lower bound is **0.5478**,
> below the bar. **The gate's expected verdict is fail.** We drew a pass at the
> 66th percentile — and **didn't author** the 240-item set it would have
> justified.
>
> Seventeen entries like that, most of them flattering. The most recent was
> yesterday: a budget in this demo came from a planning document instead of our
> config, and put a number a third too large on the screen you just watched.
>
> That's the failure this product exists to catch, happening to us, in the
> medium a customer reads. So a plausible number answering the wrong question
> isn't an assertion for us. It's a base rate.

---

## 4:45 – 5:00 — Close

> We tell you your error rate on your traffic. We tell you when it stops being
> true. We tell you what it costs to keep it true.
>
> And when we can't tell you, we refuse — and we print the price.

---

## Two answers to rehearse

**"Where's the feedback loop?"**

> Deliberately not built, and the reason is the interesting part. The naive
> version — retrain on reviewer overrides — makes the detector *worse*: you only
> get labels for items the detector flagged, so you train on a biased sample and
> the bias compounds each cycle. Doing it right needs inverse-propensity
> weighting with the selection probability recorded per item. Our override
> schema hard-fails without that field, so the data is being collected correctly
> today. The estimator is specified, unbuilt, and in our open items.

**"22% recall is low."**

> It is, and it's a choice. That's one point on a measured ROC — the same
> detector runs at 74% recall for decision support. Customer support is
> FPR-constrained: at tens of thousands of interactions a week the false-positive
> rate is the binding cost, so we buy precision with recall. It clears its
> declared floor of 10% on the interval's lower bound, not its point estimate.
> And the local ROC slope differs 7.9× across the three profiles — which is
> exactly why we warrant operating points individually rather than warranting a
> detector.

---

## Command reference

| beat | command |
|---|---|
| verification | `make verify` |
| 1–6 | `python demo/show_beats.py --beat N` |
| all | `python demo/show_beats.py` |
| live loop | `python demo/run_demo.py --fixture --events 4 --auto-prove` |

**If a command fails on camera:** say what failed, run
`python demo/show_beats.py` (committed artifacts only), continue. A beat whose
artifact is missing prints *why* rather than rendering blank.
