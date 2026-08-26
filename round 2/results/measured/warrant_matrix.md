| detector | triviaqa-600-synthetic | triviaqa-longctx-600-synthetic | hinglish-pii-200 | hinglish-pii-200-longctx | hard-negatives-200 | triviaqa-600 | triviaqa-longctx-600 |
|---|---|---|---|---|---|---|---|
| pii-reference | UNVALIDATED | UNVALIDATED | VALID R=0.79 [0.70, 0.88] | VALID R=0.79 [0.70, 0.88] | VALID FPR=0.000 [0.000, 0.018] | UNVALIDATED | UNVALIDATED |
| probe-T1-max_rolling_means | VALID R=0.25 [0.16, 0.35] | VALID R=0.16 [0.09, 0.24] | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T1-mean_pool | VALID R=0.22 [0.13, 0.31] | **REFUSED** | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T2-logprob | VALID R=0.17 [0.10, 0.25] | VALID R=0.14 [0.07, 0.21] | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-T3-judge | VALID R=0.17 [0.10, 0.25] | **REFUSED** | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| probe-qwen2.5-7b-instruct-T1-max_rolling_means | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.12] | **REFUSED** |
| probe-qwen2.5-7b-instruct-T1-mean_pool | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | VALID R=0.08 [0.05, 0.11] | **REFUSED** |

Cell states: VALID 11, REFUSED 4, UNVALIDATED 34. 34 of 49 cells have never been measured, which is the expected shape: UNVALIDATED is the modal state in production.
