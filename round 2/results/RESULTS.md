# RESULTS

> [!WARNING]
> **8 of 17 populated cells are synthetic fixtures and their numbers are not printed below.**
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
| pii-reference | UNVALIDATED | UNVALIDATED | VALID R=0.79 [0.70, 0.88] @f=0.620 | VALID R=0.79 [0.70, 0.88] @f=0.620 | VALID FPR=0.000 [0.000, 0.018] | UNVALIDATED | UNVALIDATED |
| probe-T1-max_rolling_means | VALID · FIXTURE — NOT MEASURED | VALID · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T1-mean_pool | VALID · FIXTURE — NOT MEASURED | REFUSED · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T2-logprob | VALID · FIXTURE — NOT MEASURED | VALID · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T3-judge | VALID · FIXTURE — NOT MEASURED | REFUSED · FIXTURE — NOT MEASURED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-qwen2.5-7b-instruct-T1-last_token | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.11] @f=0.042 · CAL:n/a | VALID R=0.13 [0.09, 0.17] @f=0.065 · CAL:n/a |
| probe-qwen2.5-7b-instruct-T1-max_rolling_means | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.12] @f=0.042 · CAL:n/a | **REFUSED** |
| probe-qwen2.5-7b-instruct-T1-mean_pool | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.11] @f=0.040 · CAL:n/a | **REFUSED** |

Cell states: VALID 13, REFUSED 4, UNVALIDATED 39. 39 of 56 cells have never been measured, which is the expected shape: UNVALIDATED is the modal state in production.

## Measured results

| detector | envelope | status | AUROC | recall | precision | flag rate | base rate | lift | n |
|---|---|---|---|---|---|---|---|---|---|
| `pii-reference` | `hinglish-pii-200` | VALID | 0.773 [0.690, 0.856] (95% CI, n=200) | 0.794 [0.698, 0.885] (95% CI, n=200) | 0.653 [0.511, 0.810] (95% CI, n=200) | 0.6200 [0.5143, 0.7350] (95% CI, n=200) | 0.5100 | 1.281 [1.126, 1.427] (95% CI, n=200) — 79% of the 1.61 ceiling at base rate 0.510 | 200 |
| `pii-reference` | `hinglish-pii-200-longctx` | VALID | 0.773 [0.690, 0.856] (95% CI, n=200) | 0.794 [0.698, 0.885] (95% CI, n=200) | 0.653 [0.511, 0.810] (95% CI, n=200) | 0.6200 [0.5143, 0.7350] (95% CI, n=200) | 0.5100 | 1.281 [1.126, 1.427] (95% CI, n=200) — 79% of the 1.61 ceiling at base rate 0.510 | 200 |
| `pii-reference` | `hard-negatives-200` | VALID | n/a | n/a | n/a | 0.0000 [0.0000, 0.0183] (95% CI, n=200) | 0.0000 | n/a | 200 |
| `probe-qwen2.5-7b-instruct-T1-last_token` | `triviaqa-600` | VALID | 0.826 [0.793, 0.857] (95% CI, n=600) | 0.079 [0.050, 0.112] (95% CI, n=600) | 0.880 [0.750, 1.000] (95% CI, n=600) | 0.0417 [0.0267, 0.0583] (95% CI, n=600) | 0.4617 | 1.906 [1.190, 2.676] (95% CI, n=600) — 88% of the 2.17 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-max_rolling_means` | `triviaqa-600` | VALID | 0.785 [0.750, 0.821] (95% CI, n=600) | 0.083 [0.054, 0.117] (95% CI, n=600) | 0.920 [0.800, 1.000] (95% CI, n=600) | 0.0417 [0.0283, 0.0583] (95% CI, n=600) | 0.4617 | 1.993 [1.286, 2.819] (95% CI, n=600) — 92% of the 2.17 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-mean_pool` | `triviaqa-600` | VALID | 0.785 [0.750, 0.821] (95% CI, n=600) | 0.079 [0.050, 0.113] (95% CI, n=600) | 0.917 [0.786, 1.000] (95% CI, n=600) | 0.0400 [0.0250, 0.0567] (95% CI, n=600) | 0.4617 | 1.986 [1.254, 2.835] (95% CI, n=600) — 92% of the 2.17 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-last_token` | `triviaqa-longctx-600` | VALID | 0.813 [0.780, 0.845] (95% CI, n=600) | 0.126 [0.090, 0.170] (95% CI, n=600) | 0.897 [0.794, 0.978] (95% CI, n=600) | 0.0650 [0.0483, 0.0850] (95% CI, n=600) | 0.4617 | 1.944 [1.388, 2.613] (95% CI, n=600) — 90% of the 2.17 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-max_rolling_means` | `triviaqa-longctx-600` | REFUSED | 0.555 [0.511, 0.602] (95% CI, n=600) | 0.585 [0.527, 0.642] (95% CI, n=600) | 0.497 [0.447, 0.550] (95% CI, n=600) | 0.5433 [0.5050, 0.5833] (95% CI, n=600) | 0.4617 | 1.076 [0.969, 1.182] (95% CI, n=600) — 58% of the 1.84 ceiling at base rate 0.462 | 600 |
| `probe-qwen2.5-7b-instruct-T1-mean_pool` | `triviaqa-longctx-600` | REFUSED | 0.502 [0.455, 0.548] (95% CI, n=600) | 0.000 [0.000, 0.013] (95% CI, n=600) | 0.000 [0.000, 0.000] (95% CI, n=600) | 0.0000 [0.0000, 0.0061] (95% CI, n=600) | 0.4617 | n/a | 600 |

## Provenance

- config hash `4eb9f8bd410cb192`
- git commit `dd10bf375082fdea65391512bdf3b7c15987708e`
- dirty tree: `True`
- seed `1729`
- generated `2026-08-27T13:01:41+00:00`
