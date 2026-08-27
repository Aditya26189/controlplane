# RESULTS

> [!WARNING]
> **8 of 15 populated cells are synthetic fixtures and their numbers are not printed below.**
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

## Warrant matrix

Cells on synthetic envelopes show their status but not their numbers.

| detector | triviaqa-600-synthetic | triviaqa-longctx-600-synthetic | hinglish-pii-200 | hinglish-pii-200-longctx | hard-negatives-200 | triviaqa-600 | triviaqa-longctx-600 |
|---|---|---|---|---|---|---|---|
| pii-reference | UNVALIDATED | UNVALIDATED | VALID R=0.79 [0.70, 0.88] | VALID R=0.79 [0.70, 0.88] | VALID FPR=0.000 [0.000, 0.018] | UNVALIDATED | UNVALIDATED |
| probe-T1-max_rolling_means | VALID · FIXTURE — NOT MEASURED | VALID · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T1-mean_pool | VALID · FIXTURE — NOT MEASURED | REFUSED · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T2-logprob | VALID · FIXTURE — NOT MEASURED | VALID · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T3-judge | VALID · FIXTURE — NOT MEASURED | REFUSED · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-qwen2.5-7b-instruct-T1-max_rolling_means | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.12] | **REFUSED** |
| probe-qwen2.5-7b-instruct-T1-mean_pool | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.11] | **REFUSED** |

Cell states: VALID 11, REFUSED 4, UNVALIDATED 34. 34 of 49 cells have never been measured, which is the expected shape: UNVALIDATED is the modal state in production.

## Measured results

| detector | envelope | status | AUROC | recall | precision | lift | n |
|---|---|---|---|---|---|---|---|
| `pii-reference` | `hinglish-pii-200` | VALID | 0.773 [0.690, 0.856] (95% CI, n=200) | 0.794 [0.698, 0.885] (95% CI, n=200) | 0.653 [0.511, 0.810] (95% CI, n=200) | 1.281 [1.126, 1.427] (95% CI, n=200) — 79% of the 1.61 ceiling at base rate 0.510 | 200 |
| `pii-reference` | `hinglish-pii-200-longctx` | VALID | 0.773 [0.690, 0.856] (95% CI, n=200) | 0.794 [0.698, 0.885] (95% CI, n=200) | 0.653 [0.511, 0.810] (95% CI, n=200) | 1.281 [1.126, 1.427] (95% CI, n=200) — 79% of the 1.61 ceiling at base rate 0.510 | 200 |
| `pii-reference` | `hard-negatives-200` | VALID | n/a | n/a | n/a | n/a | 200 |
| `probe-qwen2.5-7b-instruct-T1-max_rolling_means` | `triviaqa-600` | VALID | 0.785 [0.750, 0.821] (95% CI, n=600) | 0.083 [0.054, 0.117] (95% CI, n=600) | 0.920 [0.800, 1.000] (95% CI, n=600) | 1.993 [1.286, 2.819] (95% CI, n=600) — 92% of the 2.17 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-mean_pool` | `triviaqa-600` | VALID | 0.785 [0.750, 0.821] (95% CI, n=600) | 0.079 [0.050, 0.113] (95% CI, n=600) | 0.917 [0.786, 1.000] (95% CI, n=600) | 1.986 [1.254, 2.835] (95% CI, n=600) — 92% of the 2.17 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-max_rolling_means` | `triviaqa-longctx-600` | REFUSED | 0.555 [0.511, 0.602] (95% CI, n=600) | 0.585 [0.527, 0.642] (95% CI, n=600) | 0.497 [0.447, 0.550] (95% CI, n=600) | 1.076 [0.969, 1.182] (95% CI, n=600) — 58% of the 1.84 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-mean_pool` | `triviaqa-longctx-600` | REFUSED | 0.502 [0.455, 0.548] (95% CI, n=600) | 0.000 [0.000, 0.013] (95% CI, n=600) | 0.000 [0.000, 0.000] (95% CI, n=600) | n/a | 600 |

## Provenance

- config hash `bfb92a89ceacd678`
- git commit `29c80a065464b66533888518b016a4d8e43873d2`
- dirty tree: `True`
- seed `1729`
- generated `2026-08-26T20:27:08+00:00`
