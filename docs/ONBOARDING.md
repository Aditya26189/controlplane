# ONBOARDING.md — your first hour

<sub>[🏠 Project README](../README.md) · [📚 Documentation index](README.md) · [🗺️ Diagrams](DIAGRAMS.md) · [📖 Glossary](GLOSSARY.md)</sub>

Written for someone who has just cloned this and has to be useful today. Read
the parts in order; each one is sized to what it costs.

---

## Minute 0 — the one sentence

**A detector produces a score. A warrant is a separate, time-bounded,
evidence-backed statement about what that score is worth on this distribution
right now.**

Everyone ships detectors. Almost nobody ships the second thing — so a guardrail
that has quietly stopped working looks exactly like one that works, and the
dashboard stays green either way.

If you remember nothing else: **policy reads the warrant, not the score.**

---

## Minutes 1–5 — make it run

```bash
python -m venv .venv && . .venv/Scripts/activate    # macOS/Linux: . .venv/bin/activate
pip install -r requirements.lock.txt
python scripts/smoke.py
```

Seven checks, under a minute, no network. It tells you the package imports, the
config loads and hashes, an artifact is readable, the eval sets are frozen, and
the claim table parses. If that passes, your clone is sound.

Then the real one:

```bash
python scripts/verify.py
```

It resolves every number in the README against the artifact and field that
README names, then recomputes every metrics block from the frozen per-item
scores. On a fresh clone the third tier reports SKIPPED, because the activation
caches are gitignored — that is correct behaviour, not a failure.

Details and the other two entry points: [SETUP.md](SETUP.md).

---

## Minutes 5–15 — see the shape

Open [DIAGRAMS.md](DIAGRAMS.md) and read the first three diagrams:

1. **The three objects** — detector, envelope, warrant.
2. **The pipeline** — which script writes which artifact.
3. **The warrant lifecycle** — and why `UNVALIDATED` is the modal state.

Then look at one real artifact, because the diagrams are only a claim about it:

```bash
python -c "import json;d=json.load(open('results/validation-T1-last_token.json'));print(list(d))"
```

You will see `controls`, `metrics`, `warrant`, `warrant_status`,
`status_reason`, and a `provenance` block carrying the config hash, the git
commit, a dirty flag and the library versions. Every artifact in `results/` has
that block. It is what makes a published number checkable rather than claimed.

---

## Minutes 15–30 — read the two documents that matter most

**[../README.md](../README.md)'s claim table.** Every quantitative claim the
project makes, the artifact that contains it, and the field inside that
artifact. This is the spine. If you are ever unsure whether something is
measured or asserted, look for it here — if it is not in the table, it is not a
measured claim.

**[LIMITATIONS.md](LIMITATIONS.md).** Read before quoting anything. The four
items in §2 change how the claim table should be read; §3 lists what was
specified and never built. Knowing these before a reviewer raises them is the
difference between a defensible answer and a scramble.

---

## Minutes 30–45 — the code

[CODE_TOUR.md](CODE_TOUR.md) maps every package. The four files that carry the
most weight, in reading order:

| File | Why start here |
|---|---|
| `controlplane/model/warrant.py` | The central record. Its `__post_init__` enforces that a refusal states a reason and that the key includes the envelope |
| `controlplane/model/metrics.py` | Where `EXACT` vs `ESTIMATED` is enforced at construction — a rate without an interval cannot be built |
| `controlplane/validation/issuance.py` | `issue_or_refuse`, which takes **no argument** that could promote a failing detector |
| `controlplane/policy/resolution.py` | Load-time warrant resolution: a bundle naming an operating point with no warrant does not load |

Notice the pattern: the invariants are enforced at construction, not by
convention. You cannot build an illegal record and fix it later.

---

## Minutes 45–60 — pick up a thread

Whichever fits what you were asked to do:

- **Add or change a detector** → `controlplane/detectors/`, then
  [RUNBOOK.md](RUNBOOK.md) `09_detectors.py`. Remember that a detector's
  identity includes its configuration.
- **Add an evaluation set** → `controlplane/evalsets/`, then
  `01_build_evalsets.py`. Sets are frozen and content-hashed; changing one
  creates a different set, deliberately.
- **Change a policy profile** → `policies/*/bundle.yaml`, then
  `07_policy.py`. Expect the bundle to refuse to load if you name an operating
  point nothing has warranted.
- **Understand a statistical choice** → [METHODS.md](METHODS.md), then the
  `DECISIONS.md` entry it cites.

---

## The five things that will bite you

1. **Never pipe a command whose exit status you need.** `cmd | tail` reports
   *tail's* status, so a failing test suite reads as green. Use
   `sh scripts/run.sh <cmd>`. This trap fired twice in one session here.
2. **Never run a script against a dirty tree.** Artifacts record the `HEAD` they
   were built from; `provenance()` sets `dirty: true` and the record becomes a
   lie about which code produced the numbers.
3. **Test is never consulted by a selection.** Layer, threshold and
   regularisation are validation decisions. Selecting on test inflates the
   headline and is the first thing a reviewer checks.
4. **Positive class is *incorrect*.** Inverting it silently yields `1 − AUROC`,
   which reads as a strong negative result and misdirects debugging for hours.
5. **Yield is exact; rate is estimated.** *"We surfaced N real errors this
   month"* is a count of reviewed, confirmed items — free, no interval.
   *"We caught X% of errors"* is a claim about traffic nobody reviewed — it
   costs labels and always carries an interval.
   Conflating them converts a free exact claim into an unbacked estimate and
   nobody notices.

---

## Where to ask the repository instead of a person

| Question | Where the answer already is |
|---|---|
| Why was this done this way? | [../DECISIONS.md](../DECISIONS.md) — search the topic |
| Is this case handled? | [CASES.md](CASES.md) — every case names its test and its artifact |
| Where did this number come from? | The README claim table, then [ARTIFACTS.md](ARTIFACTS.md) |
| What does this word mean here? | [GLOSSARY.md](GLOSSARY.md) |
| What broke? | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| What is deliberately missing? | [LIMITATIONS.md](LIMITATIONS.md) §3 |
