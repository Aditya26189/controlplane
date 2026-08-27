# DEMO.md — what this must produce

The build exists to make this run. Read it before Phase 2.5 and grow the runner toward it every phase.

**Format:** two panes, side by side, one input stream.

- **Left — the conventional stack.** Presidio, a guard model, an LLM judge. Deliberately the modal Round 2 submission. **Everything on the left runs at its documented defaults; we change nothing.**
- **Right — ControlPlane.** Same stream, same detectors, plus warrants.

---

## Beat 1 — Normal traffic (30s)

Both panes handle it. Both look healthy.

The right pane additionally shows a **warrant banner**: detector, operating point, envelope, measured bounds with interval, validation age, envelope status.

> *"Both of these are working. Only one of them can tell you what 'working' means."*

---

## Beat 2 — Three profiles, one curve (45s)

Same input through customer support, internal knowledge, decision support. Three different actions.

Show the three operating points on **one measured ROC** with the weighted-error vector visible and labelled as policy.

> *"These aren't three thresholds we picked. They're three points on one curve we measured, and each one is separately warranted."*

---

## Beat 3 — The Presidio refusal (45s)

A Hinglish message containing an Aadhaar number.

The left pane passes it silently. The right pane has already refused Presidio a warrant on that envelope and says so, with the measured recall in the refusal reason.

**Inoculate immediately — do not wait to be asked:**

> *"Everything on the left is at its documented default. We changed nothing. That's the finding — Presidio ships Indian recognisers, they're English-only, and they're off unless you know to turn them on."*

Then show all three configurations on the left — stock, enabled, enabled-plus-custom — with the measured recall for each.

> *"Even at its best there's a residual on obfuscated and code-switched forms. The difference isn't that ours catches more. It's that ours tells you the number."*

Showing the best configuration is what makes the beat unattackable. The point survives it.

---

## Beat 4 — The shift, and what stops being provable (90s)

**The centrepiece, and it is now measured rather than scripted.**

The scripted version had a revocation: a warrant going stale, an alarm, a
fallback. What the extraction actually produced is stronger, because no
revocation is needed to make the point. **The same detector, at the same
operating point, holds a valid warrant on one envelope and is refused on the
other.** Two cells in the same row of the matrix, opposite verdicts, both
measured on the same 3.2-hour run.

Switch the stream to long-context inputs, live.

**Left:** keeps returning confident verdicts. Never stops looking healthy. Let it
run long enough that the audience notices nothing is happening.

**Right:** the matrix row for `probe-qwen2.5-7b-instruct-T1-mean_pool`:

| envelope | verdict | |
|---|---|---|
| `triviaqa-600` | **VALID** | R = 0.08 [0.05, 0.11], AUROC 0.785 [0.750, 0.821] |
| `triviaqa-longctx-600` | **REFUSED** | `auroc_lower_ci 0.4546, required > 0.55` |

And the row beneath it, for the aggregation built to survive exactly this shift:

| envelope | verdict | |
|---|---|---|
| `triviaqa-600` | **VALID** | R = 0.08 [0.05, 0.12] |
| `triviaqa-longctx-600` | **REFUSED** | `auroc_lower_ci 0.5105` **and** `lift_lower_ci 1.076 [0.969, 1.182]` at flag rate 0.5433 |

Narration, tracking the screen:

> *"Same probe. Same threshold. Nothing retrained. On short context it holds a
> warrant. On long context it is refused — and not because it went quiet. It
> flags fifty-four per cent of the traffic. Thirteen times the budget. Its lift
> interval is 1.08, and it crosses 1.0, which means we cannot show it beats
> picking at random at the same cost."*

Then, pointing left:

> *"Both of these systems just stopped working. Only one of them knows — and it
> knows in the units you buy: how much you would spend, and what you would get
> for it."*

**The sentence this beat exists for:**

> **"This detector is sound here, and cannot be shown to beat random sampling
> there."**

**Why the failures are worth showing separately.** The two aggregations fail in
opposite directions, and both are invisible without the warrant:

- `mean_pool` **fails silent**: its scores fall below the frozen threshold, it
  flags nothing, and the dashboard reads as clean traffic.
- `max_rolling_means` **fails expensive**: its scores inflate, it flags 54.3% at
  precision 0.497 against a base rate of 0.462 — random sampling wearing a
  detector's name.

`max_rolling_means` was built specifically to survive long-context shift. It does
not. Say so on stage: a demo that shows the mitigation working is a demo nobody
believes, and the honest version is the one where the layer catches the thing its
own authors built to prevent.

**Why this and not an alarm.** An alarm says something broke. This says *here is
what I can still prove, here is what I can no longer prove, and here is the cost
of the difference.* That is the product.

**If the revocation ladder is built by demo day**, it layers on top: PSI on token
length crosses its threshold, the warrant moves through the four-state ladder,
routing consults the matrix, and `decision_support` suspends because no warrant
on this envelope meets its declared minimum. That is a better *mechanism* story.
It is not a better *evidence* story, and the table above is evidence.

---

## Beat 5 — Prove it (60s)

Invite a judge to press the button.

The frozen set re-runs live. On screen: metrics with intervals, and all five controls — including the **deliberately broken padding case being rejected**.

> *"That last line is us breaking our own test on purpose, every run, to check it still catches the fault it exists for. Two of those five controls are negative controls — they assert this pipeline can produce a null result when there's no signal. A pipeline that can't fail can't be trusted when it succeeds."*

Then the price list:

> *"Three of the four numbers you need are free — you're already reviewing your flags, you're just throwing the labels away. Only recall costs money, because recall means estimating what you missed across two hundred thousand responses nobody looked at. Here's what each precision level costs. Pick one."*

**Close:**

> *"We tell you what your error rate is on your traffic. We tell you when that number stops being true. And we tell you what it costs to keep it true."*

Cut to the Round 1 green dashboard — now honest.

---

## Rules

**Record a backup.** If the live run fails: *"that's precisely the failure mode we log"* — then cut to the recording. Delivered calmly it reads as poise.

**Never say a number without its interval.** Not once, not in passing.

**Never say lift without precision.** If the projection comes up: *"about seven in ten flags would be false alarms at that base rate — still nine times better than random sampling, where ninety-seven in a hundred would be."*

**If asked whether the projection transfers:** *"Our own envelope check would refuse to warrant it without revalidation on your traffic. It's a sizing estimate. The claim is what comes back from `/validate` on your data."*

**Beat 4 must survive being run twice in a row.** Test it that way.

**One voice throughout.** Handing off between team members breaks a demo the way it breaks a story.
