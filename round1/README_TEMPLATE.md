# README_TEMPLATE.md

Fill this in at Stage 7 and save as `README.md`. Every `{{placeholder}}` is replaced by a value read from `results/`. **If a number cannot be traced to a file in `results/`, it does not go in.** Delete this header line and the note below before publishing.

> Note to the agent: keep the writing plain. No marketing language, no adjectives that aren't doing work. A reviewer should be able to check every claim against an artifact in this repo within two minutes.

---

# ControlPlane — cascade economics

Can you tell, before a language model writes a single token, whether the answer it is about to give will be wrong?

If you can, monitoring gets much cheaper. Serious checkers — LLM-as-judge, semantic entropy, claim attribution — cost 200 to 1000 milliseconds per call, so nobody runs them on all their traffic. They sample a few percent and the rest ships unchecked. This repo measures whether a linear probe on the model's internal state can pick which few percent are worth checking.

## Result

| Metric | Value |
|---|---|
| Model | {{model_name}} ({{quantization}}) |
| Dataset | {{dataset}}, n={{n_test}} held-out questions |
| Base error rate | {{base_rate}} |
| Probe layer | {{layer}} of {{n_layers}} (selected on validation) |
| Test AUROC | {{auroc}} [{{auroc_ci_low}}, {{auroc_ci_high}}] |
| Measured flag rate `f` | {{flag_rate}} |
| Measured recall `R` | {{recall}} [{{recall_ci_low}}, {{recall_ci_high}}] |
| Precision | {{precision}} |
| **Lift (`R/f`)** | **{{lift}}×** [{{lift_ci_low}}, {{lift_ci_high}}] |

**Reading:** at the same judge budget as random sampling, the probe surfaces **{{lift}}×** as many wrong answers.

**Read that number with its ceiling.** `lift = R/f = precision / base_rate`, so precision ≤ 1 caps lift at `1/base_rate`. This test set has a base error rate of {{measured_base_rate}}, which caps lift at **{{lift_ceiling}}×** — the measured {{lift}}× is **{{lift_pct_of_ceiling}} of everything attainable here**. TriviaQA no-context is a deliberately hard benchmark; the model is wrong on nearly 40% of it, which leaves little room for any triage to beat random sampling by a wide margin.

The base error rate *assumed in the policy table below*, and the judge's accuracy, do both cancel from the ratio — the multiplier does not rest on an assumption about production error rates. That is a separate point from the ceiling above, and the two are easy to conflate.

### Headroom — projection, not measurement

A ROC curve is base-rate independent, so the measured curve can be re-read at rarer error rates at the same budget:

{{projection_table}}

> {{projection_caveat}}

## How it works

1. The model reads a question. It has generated nothing yet.
2. At layer {{layer}}, take the residual-stream vector at the final prompt token — {{hidden_size}} numbers, a by-product of the prefill pass the model performs anyway.
3. A logistic regression scores it. One dot product: **{{probe_latency_us}} µs**, against **{{generation_latency_ms}} ms** to generate the response — a factor of {{latency_ratio}}.
4. Responses above the threshold go to the expensive checker. The rest don't.

The probe never blocks anything. Its only output is a decision about where to spend a judge call, which is why it is tuned for recall and why low precision is acceptable: a false positive costs one wasted judge call, a false negative costs a user acting on a wrong answer.

## The three policies

At N = 1,000,000 responses, base error rate {{reference_error_rate}}:

| Policy | Judge calls | Responses seen by any check | Errors caught | Relative cost |
|---|---|---|---|---|
| Judge everything | 1,000,000 | 100% | {{policy_a_caught}} | {{policy_a_cost}}× |
| Random {{flag_rate_pct}}% sample | {{policy_b_calls}} | {{flag_rate_pct}}% | {{policy_b_caught}} | 1× |
| **Probe-triggered** | **{{policy_c_calls}}** | **100%** | **{{policy_c_caught}}** | **1×** |

Coverage and verdict are different things. Every response passes the cheap layers; the expensive verdict is what's rationed. Random sampling has {{flag_rate_pct}}% coverage *and* {{flag_rate_pct}}% verdict. That gap is the whole result.

## Layer sweep

{{layer_sweep_table}}

![Layer sweep](results/layer_sweep.png)

Validation AUROC by depth. The layer was chosen here, on validation. Test played no part in the choice.

## Start here

New to this work? **[docs/HANDOVER.md](docs/HANDOVER.md)** is the orientation doc: what was built, what the numbers mean, what bounds them, and which framings to avoid when presenting.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/run_all.py --config config.yaml --smoke   # ~5 min, n=100
python scripts/run_all.py --config config.yaml           # full run
```

Runs on a single 16GB GPU with NF4 quantisation. Free Colab or Kaggle T4 is sufficient. Full run: {{total_runtime}} on {{device_name}}.

Offline clusters: pre-download the model and dataset into `HF_HOME` on a login node, then set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.

## Method notes

**Question-time probing.** Activations come from the last prompt token before generation. Not mid-generation, not from the answer. This is what makes the signal usable as a pre-generation router as well as a monitor.

**Splitting.** By `question_id`, after deduplicating normalised question strings. TriviaQA ships answer aliases and some near-duplicate questions; splitting at example level leaks them across train and test and inflates AUROC.

**Selection discipline.** Layer, regularisation strength, and threshold were all chosen on validation; test was never consulted for any of them. Every scoring of the test set is logged and published — see `results/test_scoring_log.json` and `DECISIONS.md` 016-017, which pre-registers the one re-scoring and reports its outcome.

**Labelling.** Greedy decoding, then normalised alias matching (lowercase, strip punctuation, drop articles). Aliases under 3 characters require a whole-token match rather than substring containment. Strict exact match is reported alongside as an audit: {{strict_em_base_rate}} vs {{base_rate}} lenient.

**Uncertainty.** 1000-sample bootstrap, 95% percentile intervals.

## Secondary validation

When the model abstains ("I don't know"), mean probe score is {{abstain_score}} against {{non_abstain_score}} for non-abstaining responses (AUROC {{abstain_auroc}}, abstention rate {{abstain_rate}}). The direction tracks the model's own expressed uncertainty as well as its correctness — independent evidence the probe reads something real rather than a dataset artifact.

{{negative_control_section}}

## Limitations

- **One model, one dataset.** {{model_name}} on TriviaQA. Cross-model and cross-dataset generalisation is untested here.
- **Knowledge questions only.** The published result reports that this method's generalisation falters on mathematical reasoning. {{negative_control_note}}
- **This measures the probe, not a system.** No gateway, no serving path, no end-to-end latency under load. The latency figures are component measurements.
- **`f` depends on workload.** The flag rate reported here is for this dataset at this threshold. Real traffic has a different difficulty distribution.
- **Judge accuracy is assumed to cancel.** It does cancel from the ratio, but a real judge misses errors the probe correctly flagged, so absolute errors-caught counts are upper bounds.
- **Single seed** unless a seed sweep is reported above.
- **Labelling is automatic.** Normalised alias matching is a proxy for correctness, not human judgment.

## Licence

MIT.
