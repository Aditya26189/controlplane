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

## Beat 4 — The shift, and provable graceful degradation (90s)

**The centrepiece. This is not an alarm — it is the matrix doing visible work.**

Switch the stream to long-context inputs, live.

**Left:** keeps returning confident verdicts. Never stops looking healthy. Let it run long enough that the audience notices nothing is happening.

**Right, in sequence, all automatic:**

1. Drift monitor fires — PSI on token length crosses 0.25
2. `T1 mean-pool` warrant **REVOKED** on this envelope
3. System consults the **matrix**: which detector holds a valid warrant on `triviaqa-longctx-600`?
4. Routes to whichever cell is `VALID` — adopts **that warrant's bounds**, wider
5. `decision_support` profile **suspended** — the new bounds fall below its declared minimum
6. Certificate written with all of it: revocation reason, new envelope, new warrant, new claimed bounds, policy version

Narration, tracking the screen:

> *"T1 just revoked — this input is outside the distribution those numbers came from. The matrix says T3 holds a valid warrant on long-context at lower recall, wider interval. Falling back, claimed bounds updated. Decision-support is suspended, because the number it requires isn't available on this traffic."*

Then, pointing left:

> *"Both of these systems just stopped working. Only one of them knows."*

**Why this beat and not an alarm:** an alarm says something broke. This says *here is what I can still prove, here is what I can no longer prove, and here is what I've stopped claiming as a result.* That is the product.

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
