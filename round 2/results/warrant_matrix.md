| detector | triviaqa-600-synthetic | triviaqa-longctx-600-synthetic | hinglish-pii-200 | hinglish-pii-200-longctx | hard-negatives-200 | triviaqa-600 | triviaqa-longctx-600 |
|---|---|---|---|---|---|---|---|
| pii-reference | UNVALIDATED | UNVALIDATED | VALID R=0.79 [0.70, 0.88] @f=0.620 | VALID R=0.79 [0.70, 0.88] @f=0.620 | VALID FPR=0.000 [0.000, 0.018] | UNVALIDATED | UNVALIDATED |
| probe-T1-max_rolling_means | VALID R=0.25 [0.16, 0.35] @f=0.050 · CAL:n/a | VALID R=0.16 [0.09, 0.24] @f=0.047 · CAL:n/a | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T1-mean_pool | VALID R=0.22 [0.13, 0.31] @f=0.047 · CAL:n/a | **REFUSED** | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T2-logprob | VALID R=0.17 [0.10, 0.25] @f=0.052 · CAL:n/a | VALID R=0.14 [0.07, 0.21] @f=0.053 · CAL:n/a | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T3-judge | VALID R=0.17 [0.10, 0.25] @f=0.050 · CAL:n/a | **REFUSED** | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-qwen2.5-7b-instruct-T1-last_token | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.11] @f=0.042 · CAL:n/a | VALID R=0.13 [0.09, 0.17] @f=0.065 · CAL:n/a |
| probe-qwen2.5-7b-instruct-T1-max_rolling_means | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.12] @f=0.042 · CAL:n/a | **REFUSED** |
| probe-qwen2.5-7b-instruct-T1-mean_pool | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.11] @f=0.040 · CAL:n/a | **REFUSED** |

Cell states: VALID 13, REFUSED 4, UNVALIDATED 39. 39 of 56 cells have never been measured, which is the expected shape: UNVALIDATED is the modal state in production.
