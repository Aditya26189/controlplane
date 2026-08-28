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

## Beat 4 — The shift, and which detector still holds (90s)

**The centrepiece, measured. Three detectors, one shift, and the matrix names
the one that survives.**

Switch the stream to long-context inputs, live.

**Left:** keeps returning confident verdicts. Never stops looking healthy. Let it
run long enough that the audience notices nothing is happening.

**Right:** one column of the matrix moves. Same probe family, same operating
point, nothing retrained:

| detector | `triviaqa-600` | `triviaqa-longctx-600` |
|---|---|---|
| `T1-last_token` | VALID · AUROC 0.826 [0.793, 0.857] | **VALID** · AUROC 0.813 [0.780, 0.845] |
| `T1-max_rolling_means` | VALID · AUROC 0.785 [0.750, 0.821] | **REFUSED** · 0.555 [0.511, 0.602] |
| `T1-mean_pool` | VALID · AUROC 0.785 [0.750, 0.821] | **REFUSED** · 0.502 [0.455, 0.548] |

Narration, tracking the screen:

> *"Same shift, three detectors. Mean-pool drops to 0.50 — chance — and stops
> flagging anything at all, so the dashboard reads as clean traffic.
> Max-of-rolling-means goes the other way: it flags fifty-four per cent of the
> stream, thirteen times the budget, and its lift interval crosses 1.0, which
> means we cannot show it beats picking at random at that cost. Both are
> refused. The last-token probe holds — 0.826 to 0.813 — and keeps its warrant.
> The system routes there, and tells you the bounds it is now claiming."*

Then, pointing left:

> *"Both systems just met traffic they were not measured on. Only one of them
> knows which of its detectors still works."*

**The sentence this beat exists for:**

> **"Two of these cannot be shown to beat random sampling here. This one still
> holds, and here are its bounds."**

**Why the two failures are worth naming separately.** They fail in opposite
directions and both are invisible without the warrant:

- `mean_pool` **fails silent** — scores fall below the frozen threshold, nothing
  is flagged, the dashboard looks clean.
- `max_rolling_means` **fails expensive** — 54.3% flagged at precision 0.497
  against a base rate of 0.462. Random sampling wearing a detector's name.

`max_rolling_means` was built specifically to survive long-context shift. It does
not. Say that on stage: the mitigation this project built for this exact problem
is one of the two that gets refused, and the layer caught it. A demo where the
authors' own fix works is a demo nobody believes.

**Why `last_token` and not whichever scored best.** It was declared as the
headline aggregation in `DECISIONS.md` 065 **before the extraction ran**, on the
grounds that it is what invariant 1 describes — question-time, last token of the
prompt, before any generated token. It then turned out to be both the best
ranker and the only survivor. If asked, show the commit timestamp: the
declaration predates the data. Choosing it afterwards would have been selection
on the test set at the level of detector architecture, and undetectable in any
artifact.

Note it does **not** win on every measure — at Round 1's flag rate it lifts 1.975
against the pooled variants' 2.035. Say so if pressed. Reporting the declared
detector rather than the flattering one is the point.

**Why this and not an alarm.** An alarm says something broke. This says *here is
what I can still prove, here is what I can no longer prove, and here is which
detector to send this traffic to.* That is the product.

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
