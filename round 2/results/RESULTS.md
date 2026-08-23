# RESULTS

> [!WARNING]
> **8 of 11 populated cells are synthetic fixtures and their numbers are not printed below.**
>
> A fixture number is internally valid and is not evidence about a language model. The renderer refuses to print them rather than relying on a reader noticing a footnote. Cells affected:
>
> - `probe-T1-max_rolling_means` on `triviaqa-600-synthetic` — triviaqa-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T1-mean_pool` on `triviaqa-600-synthetic` — triviaqa-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T2-logprob` on `triviaqa-600-synthetic` — triviaqa-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T3-judge` on `triviaqa-600-synthetic` — triviaqa-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T1-max_rolling_means` on `triviaqa-longctx-600-synthetic` — triviaqa-longctx-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T1-mean_pool` on `triviaqa-longctx-600-synthetic` — triviaqa-longctx-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T2-logprob` on `triviaqa-longctx-600-synthetic` — triviaqa-longctx-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model
> - `probe-T3-judge` on `triviaqa-longctx-600-synthetic` — triviaqa-longctx-600-synthetic is a synthetic fixture; its numbers describe a generator we wrote, not a language model

## Outstanding measurement

These extractions are hard dependencies of the submission. Until they land, the sections below are structurally complete and empirically unbacked.

| eval set | needed for | blocks |
|---|---|---|
| `triviaqa-600` | the tier ladder — what T1 access actually buys over T2 and T3 | Phase 10 (demo shows real numbers), Phase 12 (README numbers must trace to results/) |
| `triviaqa-longctx-600` | Beat 4 — the envelope shift that drives revocation and routing | Phase 10; the drift demo has no measured basis without it |

## Warrant matrix

Cells on synthetic envelopes show their status but not their numbers.

| detector | triviaqa-600-synthetic | triviaqa-longctx-600-synthetic | hinglish-pii-200 | hinglish-pii-200-longctx | hard-negatives-200 | triviaqa-600 | triviaqa-longctx-600 |
|---|---|---|---|---|---|---|---|
| pii-reference | UNVALIDATED | UNVALIDATED | VALID R=0.79 [0.70, 0.88] | VALID R=0.79 [0.70, 0.88] | VALID FPR=0.000 [0.000, 0.018] | UNVALIDATED | UNVALIDATED |
| probe-T1-max_rolling_means | VALID · FIXTURE — NOT MEASURED | VALID · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T1-mean_pool | VALID · FIXTURE — NOT MEASURED | REFUSED · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T2-logprob | VALID · FIXTURE — NOT MEASURED | VALID · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T3-judge | VALID · FIXTURE — NOT MEASURED | REFUSED · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |

Cell states: VALID 9, REFUSED 2, UNVALIDATED 24. 24 of 35 cells have never been measured, which is the expected shape: UNVALIDATED is the modal state in production.

## Measured results

| detector | envelope | status | AUROC | recall | precision | lift | n |
|---|---|---|---|---|---|---|---|
| `pii-reference` | `hinglish-pii-200` | VALID | 0.773 [0.690, 0.856] (95% CI, n=200) | 0.794 [0.698, 0.885] (95% CI, n=200) | 0.653 [0.511, 0.810] (95% CI, n=200) | 1.281 [1.126, 1.427] (95% CI, n=200) — 79% of the 1.61 ceiling at base rate 0.510 | 200 |
| `pii-reference` | `hinglish-pii-200-longctx` | VALID | 0.773 [0.690, 0.856] (95% CI, n=200) | 0.794 [0.698, 0.885] (95% CI, n=200) | 0.653 [0.511, 0.810] (95% CI, n=200) | 1.281 [1.126, 1.427] (95% CI, n=200) — 79% of the 1.61 ceiling at base rate 0.510 | 200 |
| `pii-reference` | `hard-negatives-200` | VALID | n/a | n/a | n/a | n/a | 200 |

## Provenance

- config hash `c89257bc4adc10c2`
- git commit `a5a40e1bd77a337a8981d8ecf7697d0070c731ad`
- dirty tree: `False`
- seed `1729`
- generated `2026-08-23T15:27:08+00:00`
