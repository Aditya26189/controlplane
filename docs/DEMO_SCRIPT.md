# DEMO_SCRIPT.md — the five-minute video

> **One rule for the recording.** Every number on screen is read from a
> committed artifact by a command you run live. Nothing is a slide. If a
> command fails on camera, that is information and you say so — this is a
> project about not hiding the failure mode.

**Total: 5:00.** Timings are generous; the beats run in seconds.

---

## Before you record

```bash
git status --porcelain          # must be empty
make verify                     # must print VERIFIED, 3 tiers
python -m pytest tests/ -q      # must be green
```

Terminal at **110 columns**, font large enough to read at 1080p. Have two
things open in tabs you can switch to: `README.md` and `DECISIONS.md`.

Dry-run once. The whole script is four commands.

---

## 0:00 – 0:35 — The claim, and why it is different

**Say, over a still terminal:**

> Round 1 built a detector that flags hallucination, bias and privacy leaks.
> Round 2 asked what happens when an enterprise runs several AI use cases at
> once, each with a different risk signature.
>
> Our answer is not a better detector. It's a **warrant** — a separate,
> time-bounded, evidence-backed statement about what a detector's score is
> worth *right now, on this traffic*.
>
> Three clauses. We tell you what your error rate is on your traffic. We tell
> you when that number stops being true. And we tell you what it costs to keep
> it true.
>
> Everything you're about to see is computed by a command, from an artifact in
> the repository. Nothing here is a slide.

**Run:**

```bash
make verify
```

**Point at:** the three tier lines at the end.

> Thirty-one claims in the README, twenty-four metric blocks recomputed from
> frozen scores, three re-derived from raw activations. If a number in our
> documentation stops matching its artifact, this fails.

---

## 0:35 – 1:20 — BEAT 1: it refuses its own flagship, and prices the refusal

```bash
python demo/show_beats.py --beat 1
```

**Say:**

> This is the part I'd lead with, because it's the part nobody demos.
>
> Our flagship customer-support profile wants a false-positive budget of 1.5%.
> To certify that with 95% confidence you need **199** clean held-out negatives.
> Across three profiles, with the multiplicity correction, **271**.
>
> We don't have them. So the system **refuses to certify**, and it prints the
> price of lifting the refusal. That refusal has no override — no flag, no
> environment variable, no admin bypass. If one existed, the whole product is
> theatre and a reviewer would look for exactly that.

**Point at the note:**

> And these numbers are deliberately *not* inflated by a clustering correction,
> because we'd need a measured intra-cluster correlation we don't have. An
> earlier draft of this had a bigger, better-looking number in it. It came from
> a design effect in a planning document that was hypothetical. We caught it,
> retracted it, and logged it.

---

## 1:20 – 2:10 — BEAT 2 + 3: one score, three actions; and the API problem

```bash
python demo/show_beats.py --beat 2
```

**Say:**

> The problem statement says a one-size-fits-all checker rarely works. Here is
> one input, one detector, **one score — 0.8446**.
>
> Customer support **allows** it. The internal knowledge assistant **redacts**.
> Decision support **escalates** to a human.
>
> The detector did not change. The **policy** changed. Each profile declares its
> own recall floor and latency budget, and the warrant is checked against those
> declarations at load time — a profile whose evidence doesn't meet its own
> declaration **fails to load** rather than running degraded.

```bash
python demo/show_beats.py --beat 3
```

> The problem statement also says enterprises consume a model via API and can't
> inspect internals. So we measured what survives as access degrades:
> activations, log-probabilities, text-only. **Each tier gets its own warrant.**
> A tier that can't clear the issuance bar is refused, not shipped with an
> asterisk.

---

## 2:10 – 2:55 — BEAT 4: the overlap, and PII measured out of sample

```bash
python demo/show_beats.py --beat 4
```

**Say:**

> The brief says a fabricated detail about a person is *simultaneously* a
> hallucination and a privacy concern. We built a set where both labels exist on
> the same items — correctness **measured** by generating and judging, identifier
> presence **authored** so nothing co-varies with it.
>
> On the privacy side: stock Presidio scores **0.1176 recall** on Hinglish PII
> and is **REFUSED**. With recognisers enabled, 0.2843 — still refused. With our
> custom recognisers, 0.6176, and it passes.
>
> The honest part is the middle column: two of those three are **REFUSED**, and
> the refusal is the system working.

**Point at the note:**

> And we state the scope limit on screen: our pilot's privacy axis is
> identifier-presence-in-the-prompt, which is not the output-side quantity a
> production path measures. That's declared, not glossed.

---

## 2:55 – 3:40 — BEAT 5: drift, revocation, and the modal state

```bash
python demo/show_beats.py --beat 5
```

**Say, pointing at the cell counts:**

> Fifty-six detector-envelope cells. **Thirteen valid. Four refused.
> Thirty-nine unvalidated.**
>
> That last number is the one most systems would hide. A warrant is keyed by
> detector, operating point *and* eval set — so most pairs were simply never
> tested here, and **unvalidated is the modal state in production**. It is a
> distinct state with its own behaviour. It never collapses into "valid" and it
> never collapses into "refused."
>
> The brief says there's often no reliable real-time ground truth. Revocation
> doesn't need it: it fires on the **input distribution** moving, which you can
> observe without labels. And the revocation trigger is anytime-valid — you can
> check it every request forever and the false-revocation probability is still
> bounded over the whole deployment.

**Point at the note:**

> That bound was validated on independent negatives and is **not** validated
> under session correlation. It's on screen because a limitation you have to be
> asked about is a limitation you were hiding.

---

## 3:40 – 4:15 — The live loop and the audit trail

```bash
python demo/run_demo.py --fixture --events 4 --auto-prove
```

**Say, while it plays:**

> Left pane, a conventional stack: a score. Right pane, the same request with a
> warrant — the claimed recall and precision with intervals and their n, the
> envelope hash, the warrant's expiry, and a certificate chained into a
> tamper-evident ledger.
>
> Then it revalidates **live**, on camera, and runs five negative controls:
> padding, label shuffle, null features, canary, determinism. The label-shuffle
> control is the one I'd check if I were you — it proves the pipeline produces a
> null result when there's no signal.

> This run is on a **synthetic fixture** and the banner says so. The measured
> numbers are the ones in the beats.

---

## 4:15 – 4:50 — What we got wrong, and why that's the pitch

**Switch to `DECISIONS.md`. Scroll the entry headings.**

> One hundred and sixteen decisions, append-only. Sixteen of them are errors we
> found in our own work before anyone external saw them, several of which
> flattered us.
>
> The strongest isn't a code bug. A design effect from a planning document — a
> hypothetical — crossed into a sentence about our certification cost and made a
> number look 17% better. Correctly computed, for a different question, in its
> original home.
>
> **That is exactly the failure this product exists to catch**, and it happened
> to us, in prose, on submission week. So when we say a plausible number
> answering the wrong question is a real and frequent failure mode, that isn't
> an assertion. It's a measured base rate on the most careful team we had access
> to.

---

## 4:50 – 5:00 — Close

> We tell you what your error rate is on your traffic. We tell you when it stops
> being true. We tell you what it costs to keep it true.
>
> And when we can't tell you, we refuse, and we print the price.

---

## If you have 30 seconds more — the strongest single number

```bash
python demo/show_beats.py --beat 1
cat results/pilot_seed_stability.json | head -30
```

> Our banking pilot cleared its issuance bar at 0.5554 against a 0.55
> requirement — by **0.0054**. So we swept the bootstrap seed 400 times. The
> mean lower bound is **0.5478**, *below* the bar, and it clears in **38% of
> seeds**.
>
> The gate passed on the draw we ran. We could have shipped that. Instead we
> measured how much of it was the draw, and the honest sentence is "clears in
> 38% of seeds" — which is why we did **not** author the 240-item set that
> result would have justified.

---

## Command reference

| beat | command | runtime |
|---|---|---|
| verification | `make verify` | ~40 s |
| 1 refusal | `python demo/show_beats.py --beat 1` | instant |
| 2 profiles | `python demo/show_beats.py --beat 2` | instant |
| 3 tiers | `python demo/show_beats.py --beat 3` | instant |
| 4 overlap | `python demo/show_beats.py --beat 4` | instant |
| 5 drift | `python demo/show_beats.py --beat 5` | instant |
| 6 audit | `python demo/show_beats.py --beat 6` | instant |
| all | `python demo/show_beats.py` | instant |
| live loop | `python demo/run_demo.py --fixture --events 4 --auto-prove` | ~20 s |

**Fallback if anything fails on camera:** say what failed, run
`python demo/show_beats.py` which reads only committed artifacts, and continue.
A beat whose artifact is missing prints *why* rather than rendering blank.
