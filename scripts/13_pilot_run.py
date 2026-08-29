"""The banking pilot's GPU pass: generate, judge, extract, score.

``DECISIONS.md`` 090 (corrected), 101. Thin wrapper: parses arguments, calls
``controlplane/``, writes files. No logic (``CLAUDE.md``).

Needs a GPU. Runs the four things that cannot happen on CPU, in the order the
pre-registration fixes:

1. **Generate** an answer for each of the 24 frozen prompts, greedily, and
   **judge** it against the gold aliases. This is where correctness becomes a
   label -- it is measured, never authored (correction to ``090``).
2. **Check the acceptance band before scoring anything.** ``101``: 3 to 9 of
   the 12 questions wrong is the two-sided 5% band under the fitted regime.
   Outside it the set is a construction defect, and no probe number from it
   means anything.
3. **Extract** question-time activations for the same 24 prompts.
4. **Score** them with the probe fitted on ``triviaqa-2400-t960``'s train
   split, and compute the **IQR ratio** against that set's test split.

**This script reports; it does not branch.** ``101`` defines three outcomes
with three different responses, and one of them consumes the single retry
``090`` allows. Choosing between them is a decision a person makes after
reading the numbers, not something a script does while nobody is watching.

Usage:
    python scripts/13_pilot_run.py --config config.yaml \\
        --cache results/cache-triviaqa-600.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import (  # noqa: E402
    load_config,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from controlplane.detectors.probe import select_regularisation  # noqa: E402
from controlplane.evalsets import save_evalset  # noqa: E402
from controlplane.evalsets.banking import (  # noqa: E402
    BAND_HIGH,
    BAND_LOW,
    MIN_AUROC_LOWER_CI,
    SATURATION_IQR_RATIO,
    build_banking_dual_pilot,
    decide_branch,
    evalset_from_draft,
    wrong_count_by_question,
)
from controlplane.evalsets.registry import load_evalset  # noqa: E402
from controlplane.extract.activations import (  # noqa: E402
    extract_activations,
    generate_answers,
)
from controlplane.extract.model import build_prompt, load_model  # noqa: E402
from controlplane.extract.pipeline import SYSTEM_PROMPT  # noqa: E402
from controlplane.validation.evalsets import (  # noqa: E402
    TEST,
    TRAIN,
    VALIDATION,
    ExtractionCache,
)

_LOG = logging.getLogger("scripts.13_pilot_run")

def _iqr(values: np.ndarray) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--cache", default=str(PROJECT_ROOT / "results" / "cache-triviaqa-600.npz"),
        help="extraction cache holding the reference envelope's activations",
    )
    parser.add_argument("--reference", default="triviaqa-2400-t960")
    parser.add_argument("--variant", default="T1-last_token")
    parser.add_argument("--evalsets-dir", default=str(PROJECT_ROOT / "evalsets"))
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "pilot_run.json")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    draft = build_banking_dual_pilot(seed=config.seed)
    _LOG.info(
        "pilot draft %s: %d items over %d questions, hash %s",
        draft.eval_set_id, len(draft.items),
        len({i.question_id for i in draft.items}), draft.content_hash[:16],
    )

    # ---------------------------------------------------------------- model
    loaded = load_model(config)
    prompts = [
        build_prompt(loaded.tokenizer, item.prompt, SYSTEM_PROMPT)
        for item in draft.items
    ]

    # ------------------------------------------------- 1. generate and judge
    _LOG.info("generating %d answers (greedy, so the label is reproducible)", len(prompts))
    answers = generate_answers(
        loaded, prompts, max_new_tokens=48, batch_size=args.batch_size
    )
    evalset = evalset_from_draft(draft, answers)
    labels = np.array([i.label for i in evalset.items])
    wrong_questions = wrong_count_by_question(evalset)
    _LOG.info(
        "labelled: %d/%d items incorrect, %d/12 questions wrong",
        int(labels.sum()), len(labels), wrong_questions,
    )
    for item in evalset.items:
        _LOG.info(
            "  %-28s pii=%d label=%d  %r",
            item.question_id, item.meta["pii"], item.label, item.response[:60],
        )

    # ------------------------------------- 2. the band, BEFORE anything else
    in_band = BAND_LOW <= wrong_questions <= BAND_HIGH
    _LOG.info(
        "acceptance band [%d, %d]: %d wrong questions -- %s",
        BAND_LOW, BAND_HIGH, wrong_questions,
        "IN BAND" if in_band else "CONSTRUCTION DEFECT",
    )

    # Identifier-stratum error rates. DECISIONS 101 records this as a
    # full-scale check and says it is unmeasurable at n=12; reported here so
    # the number exists, explicitly marked as underpowered.
    strata = {}
    for state in (0, 1):
        rows = [i for i in evalset.items if i.meta["pii"] == state]
        strata[str(state)] = {
            "n": len(rows),
            "error_rate": float(np.mean([i.label for i in rows])),
        }
    _LOG.info(
        "identifier strata (UNDERPOWERED at n=12, reported not interpreted): %s",
        {k: round(v["error_rate"], 4) for k, v in strata.items()},
    )

    # ------------------------------------------------------ 3. extract, 4. score
    layer = ExtractionCache.load(args.cache).layer
    _LOG.info("extracting question-time activations at layer %d", layer)
    pooled = extract_activations(
        loaded, prompts,
        layer=layer,
        aggregations=[args.variant.split("-", 1)[1]],
        window=config.probe.rolling_window,
        stride=config.probe.rolling_stride,
        batch_size=args.batch_size,
        chunk_tokens=config.model.prefill_chunk_tokens,
    )
    pilot_features = pooled[args.variant.split("-", 1)[1]]

    reference_set = load_evalset(Path(args.evalsets_dir) / f"{args.reference}.json")
    cache = ExtractionCache.load(args.cache)
    splits = {
        name: np.array(
            [i for i, it in enumerate(reference_set.items) if it.split == name]
        )
        for name in (TRAIN, VALIDATION, TEST)
    }
    features, ref_labels = cache.matrix(args.variant), cache.labels
    probe, fit = select_regularisation(
        features, ref_labels, splits[TRAIN], splits[VALIDATION],
        C_grid=config.probe.C_grid,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
        split_name=VALIDATION,
    )
    reference_scores = probe.score(features[splits[TEST]])
    pilot_scores = probe.score(pilot_features)

    reference_iqr = _iqr(reference_scores)
    pilot_iqr = _iqr(pilot_scores)
    ratio = pilot_iqr / reference_iqr if reference_iqr else float("nan")
    saturated = ratio < SATURATION_IQR_RATIO
    _LOG.info(
        "IQR: pilot %.4f / reference %.4f = %.4f (threshold %.3f) -- %s",
        pilot_iqr, reference_iqr, ratio, SATURATION_IQR_RATIO,
        "SATURATED" if saturated else "spread is consistent with ranking",
    )

    from controlplane.validation.stats import auroc as auroc_of
    from controlplane.validation.stats import bootstrap_interval

    auroc_point = auroc_low = auroc_high = None
    if len(set(labels.tolist())) > 1:
        auroc_point = auroc_of(labels, pilot_scores)
        # Cluster bootstrap, resampling QUESTIONS. The two identifier states of
        # a question are one cluster, so resampling items would treat correlated
        # rows as independent and narrow every interval below what the data
        # supports -- pre-registered in the amendment to DECISIONS 090 precisely
        # because a bootstrap on the wrong unit does not announce itself.
        auroc_low, auroc_high, _ = bootstrap_interval(
            auroc_of,
            labels,
            pilot_scores,
            n_resamples=config.validation.bootstrap_samples,
            ci=config.validation.ci,
            seed=config.seed,
            groups=np.array([i.question_id for i in evalset.items]),
        )
        _LOG.info(
            "pilot AUROC %.4f [%.4f, %.4f] -- cluster bootstrap over questions",
            auroc_point, auroc_low, auroc_high,
        )

    # ------------------------------------------------------------- 5. report
    verdict = decide_branch(
        wrong_questions=wrong_questions,
        iqr_ratio=ratio,
        auroc_lower_ci=auroc_low,
    )
    _LOG.info("BRANCH: %s (consumes retry: %s)", verdict.branch, verdict.consumes_retry)
    _LOG.info("%s", verdict.response)

    save_evalset(evalset, args.evalsets_dir)
    _LOG.info("wrote the labelled set to %s", args.evalsets_dir)

    payload = {
        "eval_set_id": evalset.eval_set_id,
        "envelope_id": evalset.envelope_id,
        "draft_content_hash": draft.content_hash,
        "model_name": cache.model_name,
        "layer": layer,
        "variant": args.variant,
        "probe_fit": {"C": fit.C, "selected_on": VALIDATION, "fitted_on": args.reference},
        "generation": {
            "n_items": len(evalset.items),
            "n_questions": len({i.question_id for i in evalset.items}),
            "items_incorrect": int(labels.sum()),
            "questions_wrong": wrong_questions,
            "base_rate_items": float(labels.mean()),
            "match_rules": evalset.construction["match_rules"],
        },
        "acceptance_band": {
            "low": BAND_LOW,
            "high": BAND_HIGH,
            "observed": wrong_questions,
            "in_band": in_band,
            "derivation": (
                "two-sided 5% band under Binomial(12, 0.4510), the measured base "
                "error rate of triviaqa-2400-t960. DECISIONS 101."
            ),
        },
        "identifier_strata": {
            "note": (
                "UNDERPOWERED at 12 questions and reported rather than "
                "interpreted. DECISIONS 101 defers this to full scale, where a "
                "difference would mean the composed decision is partly "
                "measuring identifier presence rather than difficulty."
            ),
            "by_state": strata,
        },
        "saturation": {
            "pilot_iqr": pilot_iqr,
            "reference_iqr": reference_iqr,
            "reference_eval_set_id": args.reference,
            "reference_n": int(splits[TEST].size),
            "ratio": ratio,
            "threshold": SATURATION_IQR_RATIO,
            "saturated": saturated,
            "derivation": (
                "drawn at 12 clusters, not 24 items. The pre-registered 0.605 "
                "was drawn at independent items and was 38% too high for a "
                "clustered pilot; see the correction to DECISIONS 090."
            ),
        },
        "auroc": (
            None if auroc_point is None
            else {
                "value": auroc_point,
                "ci_low": auroc_low,
                "ci_high": auroc_high,
                "ci_level": config.validation.ci,
                "n_clusters": len({i.question_id for i in evalset.items}),
                "estimator": (
                    "percentile bootstrap resampling question_id, not items -- "
                    "the two identifier states of a question are one cluster"
                ),
                "min_lower_ci_for_issuance": MIN_AUROC_LOWER_CI,
            }
        ),
        "branch": verdict.branch,
        "response": verdict.response,
        "consumes_retry": verdict.consumes_retry,
        "decided_by": (
            "This script reports. Choosing the response is a person's decision "
            "after reading the numbers, because one branch consumes the single "
            "retry DECISIONS 090 allows."
        ),
        "preregistered_in": "DECISIONS.md 090 (corrected), 101",
    }
    write_json_artifact(Path(args.out), payload, config)
    _LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
