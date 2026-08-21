"""Cascade economics: the three-policy comparison and the headline lift.

``lift = R / f`` -- the probe's recall divided by its **measured** flag rate.
It is the one number this repo exists to produce, and its value is that both
the base error rate and the judge's own accuracy appear in every policy and
cancel from the ratio (SPEC.md §7, DECISIONS.md 009).

The three policies compared, over ``N`` responses with base error rate ``e``
and judge accuracy ``a``:

===================  =============  ==========  ===============  =============
Policy               Judge calls    Coverage    Errors caught    Relative cost
===================  =============  ==========  ===============  =============
Judge everything     ``N``          100%        ``e·N·a``        ``1/f``
Random sample at f   ``f·N``        ``f``       ``f·e·N·a``      ``1x``
Probe-triggered      ``f·N``        100%        ``R·e·N·a``      ``1x``
===================  =============  ==========  ===============  =============

Coverage and verdict are different things, and the gap between them in the
middle row is the whole result: random sampling has ``f`` coverage *and* ``f``
verdict, while the probe reads every response cheaply and rations only the
expensive verdict.
"""

import logging
from typing import Any, Optional, Sequence

LOGGER = logging.getLogger(__name__)


def lift(recall: float, flag_rate: float) -> float:
    """Errors caught per unit of judge budget, against random sampling.

    Both policies spend the same number of judge calls (``f·N``). Random
    sampling catches ``f·e·N·a`` errors; the probe catches ``R·e·N·a``. The base
    error rate ``e`` and judge accuracy ``a`` appear in both and cancel, so the
    ratio is ``R / f`` and depends on neither -- which is what makes the number
    defensible against "but you assumed a 3% error rate".

    Args:
        recall: Fraction of incorrect responses the probe flags, measured on test.
        flag_rate: Fraction of all responses the probe flags, **measured** on
            test -- never the target rate aimed at during threshold selection
            (CLAUDE.md invariant 6).

    Returns:
        The multiplier. 1.0 means no better than random sampling.

    Raises:
        ValueError: if the flag rate is zero, where the ratio is undefined.
    """
    if flag_rate <= 0:
        raise ValueError(
            "flag rate is zero, so lift is undefined: the probe flagged nothing "
            "and there is no budget to compare against"
        )
    return recall / flag_rate


def lift_ceiling(base_rate: float) -> float:
    """The largest lift any probe could achieve at this base error rate.

    Algebraically, ``lift = R/f = precision / base_rate`` -- flag rate cancels.
    Precision cannot exceed 1, so no probe, however good, can beat
    ``1 / base_rate`` on a workload where that fraction of responses is wrong.

    This is the piece of context the headline number is meaningless without. A
    lift of 2.3 on a benchmark with a 39% error rate is 89% of everything that
    was available; the same probe on rarer errors has far more headroom
    (DECISIONS.md 015).

    Args:
        base_rate: Fraction of responses that are incorrect.

    Returns:
        The maximum attainable lift.
    """
    if not 0.0 < base_rate <= 1.0:
        raise ValueError(f"base_rate must be in (0, 1], got {base_rate}")
    return 1.0 / base_rate


def lift_from_precision(precision: float, base_rate: float) -> float:
    """``lift = precision / base_rate``, the identity behind the ceiling.

    Computed independently of ``R/f`` so the two can be checked against each
    other; they are the same quantity written two ways.
    """
    if not 0.0 < base_rate <= 1.0:
        raise ValueError(f"base_rate must be in (0, 1], got {base_rate}")
    return precision / base_rate


def project_lift_at_base_rate(
    fpr: Sequence[float],
    tpr: Sequence[float],
    base_rate: float,
    budget: float,
) -> Optional[dict[str, Any]]:
    """Re-evaluate the measured ROC at a different base error rate.

    A ROC curve is base-rate independent -- it describes how well the probe
    *ranks*, which is a property of the probe, not of how often the model is
    wrong. So the same curve can be read at another operating regime: at base
    rate ``p``, a threshold with false-positive rate ``FPR`` and true-positive
    rate ``TPR`` flags ``f = (1-p)·FPR + p·TPR`` of all traffic and achieves
    ``lift = TPR / f``.

    **This is a projection, not a measurement**, and everything that renders it
    must say so. It assumes the probe ranks equally well on the other workload,
    which is exactly the cross-domain generalisation this repo has not tested.

    Args:
        fpr: False-positive rates from the measured ROC, ascending.
        tpr: True-positive rates from the measured ROC, ascending.
        base_rate: The base error rate to project onto.
        budget: Maximum flag rate allowed.

    Returns:
        The best operating point within budget, or None if the curve has none.
    """
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    best: Optional[tuple[float, float, float]] = None
    for false_positive, true_positive in zip(fpr, tpr):
        flag_rate = (1.0 - base_rate) * false_positive + base_rate * true_positive
        if flag_rate <= budget:
            best = (flag_rate, true_positive, false_positive)
    if best is None or best[0] <= 0.0:
        return None
    flag_rate, true_positive, false_positive = best
    return {
        "base_rate": float(base_rate),
        "flag_rate": float(flag_rate),
        "recall": float(true_positive),
        "fpr": float(false_positive),
        "lift": float(true_positive / flag_rate),
        "ceiling": lift_ceiling(base_rate),
        "projected": True,
    }


def policy_table(
    n_responses: int,
    error_rate: float,
    flag_rate: float,
    recall: float,
    judge_accuracy: float = 1.0,
) -> list[dict[str, Any]]:
    """Work the three policies at a concrete traffic volume.

    ``judge_accuracy`` multiplies every "errors caught" figure and therefore
    cancels from any ratio between two rows. It is implemented as an explicit
    parameter rather than assumed away so that a reviewer can vary it and watch
    the lift stay put (SPEC.md §7).

    Args:
        n_responses: Traffic volume ``N``.
        error_rate: Assumed base error rate ``e``. Illustrative only -- it
            scales every row identically and cancels from the lift.
        flag_rate: Measured test flag rate ``f``, which sets the shared budget.
        recall: Measured test recall ``R``.
        judge_accuracy: Fraction of true errors the judge itself catches.

    Returns:
        One dict per policy: judge calls, coverage, errors caught, relative cost.
    """
    if not 0.0 < flag_rate <= 1.0:
        raise ValueError(f"flag_rate must be in (0, 1], got {flag_rate}")
    errors = n_responses * error_rate
    return [
        {
            "policy": "judge_everything",
            "label": "Judge everything",
            "judge_calls": float(n_responses),
            "coverage": 1.0,
            "errors_caught": errors * judge_accuracy,
            "relative_cost": 1.0 / flag_rate,
        },
        {
            "policy": "random_sample",
            "label": f"Random {flag_rate:.1%} sample",
            "judge_calls": float(n_responses) * flag_rate,
            "coverage": flag_rate,
            "errors_caught": errors * flag_rate * judge_accuracy,
            "relative_cost": 1.0,
        },
        {
            "policy": "probe_triggered",
            "label": "Probe-triggered",
            "judge_calls": float(n_responses) * flag_rate,
            # Every response is scored by the probe; only the verdict is rationed.
            "coverage": 1.0,
            "errors_caught": errors * recall * judge_accuracy,
            "relative_cost": 1.0,
        },
    ]


def compare_policies(
    n_responses: int,
    error_rate: float,
    flag_rate: float,
    recall: float,
    judge_accuracy: float = 1.0,
    recall_ci: Optional[tuple[Optional[float], Optional[float]]] = None,
    flag_rate_ci: Optional[tuple[Optional[float], Optional[float]]] = None,
    measured_base_rate: Optional[float] = None,
    precision: Optional[float] = None,
    roc: Optional[dict[str, Any]] = None,
    projection_base_rates: Sequence[float] = (),
) -> dict[str, Any]:
    """Build the full economics result: the table, the lift, and the invariance.

    ``lift_from_table`` recomputes the multiplier as the ratio of the
    probe-triggered row's errors caught to the random-sample row's, and it must
    equal ``R/f``. That equality is the claim the whole submission rests on, so
    it is computed two ways and checked rather than asserted in prose.

    Args:
        n_responses: Traffic volume for the worked table.
        error_rate: Assumed base error rate, illustrative only.
        flag_rate: Measured test flag rate.
        recall: Measured test recall.
        judge_accuracy: Judge accuracy, which cancels from the ratio.
        recall_ci: Optional bootstrap bounds on recall.
        flag_rate_ci: Optional bootstrap bounds on the flag rate.

    Returns:
        A JSON-serialisable economics block.
    """
    table = policy_table(n_responses, error_rate, flag_rate, recall, judge_accuracy)
    by_policy = {row["policy"]: row for row in table}
    headline = lift(recall, flag_rate)

    random_caught = by_policy["random_sample"]["errors_caught"]
    lift_from_table = (
        by_policy["probe_triggered"]["errors_caught"] / random_caught
        if random_caught
        else None
    )
    if lift_from_table is not None and abs(lift_from_table - headline) > 1e-9:
        raise AssertionError(
            f"lift from the policy table ({lift_from_table}) disagrees with R/f "
            f"({headline}); the table and the formula must be the same claim"
        )

    result: dict[str, Any] = {
        "n_responses": int(n_responses),
        "reference_error_rate": float(error_rate),
        "judge_accuracy": float(judge_accuracy),
        "measured_flag_rate": float(flag_rate),
        "measured_recall": float(recall),
        "lift": float(headline),
        "lift_from_policy_table": lift_from_table,
        "policies": table,
        "invariance": invariance_check(flag_rate, recall, judge_accuracy),
        "note": (
            "lift = R/f. The base error rate ASSUMED in the policy table, and "
            "the judge's accuracy, appear in every policy's errors-caught figure "
            "and cancel from the ratio, so the multiplier does not depend on "
            "either. That is a different statement from the ceiling below: the "
            "MEASURED lift equals precision/base_rate and is therefore bounded "
            "by the base rate of the set it was measured on (DECISIONS.md 015)."
        ),
    }
    if measured_base_rate is not None:
        ceiling = lift_ceiling(measured_base_rate)
        result["ceiling"] = {
            "measured_base_rate": float(measured_base_rate),
            "max_attainable_lift": ceiling,
            "fraction_of_ceiling_achieved": float(headline / ceiling),
            "explanation": (
                "lift = R/f = precision/base_rate, so precision <= 1 caps lift at "
                "1/base_rate. A probe cannot beat this however well it ranks."
            ),
        }
        if precision is not None:
            result["ceiling"]["lift_from_precision"] = lift_from_precision(
                precision, measured_base_rate
            )
    if roc is not None and projection_base_rates:
        projections = [
            project_lift_at_base_rate(
                roc.get("fpr", []), roc.get("tpr", []), rate, flag_rate
            )
            for rate in projection_base_rates
        ]
        result["projection"] = {
            "budget": float(flag_rate),
            "rows": [row for row in projections if row is not None],
            "caveat": (
                "PROJECTION, NOT MEASUREMENT. A ROC is base-rate independent, so "
                "the measured curve can be read at another base error rate. This "
                "assumes the probe ranks equally well on that workload -- the "
                "cross-domain generalisation this repo has NOT tested. Treat as "
                "an illustration of headroom, never as a result."
            ),
        }
    if recall_ci is not None and flag_rate_ci is not None:
        result["lift_inputs_ci"] = {
            "recall": list(recall_ci),
            "flag_rate": list(flag_rate_ci),
        }
    return result


def invariance_check(
    flag_rate: float,
    recall: float,
    judge_accuracy: float = 1.0,
    error_rates: tuple[float, ...] = (0.01, 0.03, 0.10, 0.50),
    judge_accuracies: tuple[float, ...] = (0.5, 0.8, 1.0),
) -> dict[str, Any]:
    """Demonstrate numerically that lift moves with neither ``e`` nor ``a``.

    "But you assumed a 3% error rate" is the most natural attack on this
    analysis, and the answer is that the assumption cancels. Rather than saying
    so, this recomputes the table across a wide range of both parameters and
    records that every resulting lift is the same number.

    Returns:
        The recomputed lifts and a boolean confirming they are all equal.
    """
    lifts: list[dict[str, float]] = []
    for e in error_rates:
        for a in judge_accuracies:
            table = {
                row["policy"]: row
                for row in policy_table(1_000_000, e, flag_rate, recall, a)
            }
            caught_random = table["random_sample"]["errors_caught"]
            lifts.append(
                {
                    "error_rate": e,
                    "judge_accuracy": a,
                    "lift": (
                        table["probe_triggered"]["errors_caught"] / caught_random
                        if caught_random
                        else float("nan")
                    ),
                }
            )
    values = [row["lift"] for row in lifts]
    invariant = bool(max(values) - min(values) < 1e-9) if values else False
    return {
        "error_rates_tested": list(error_rates),
        "judge_accuracies_tested": list(judge_accuracies),
        "lifts": lifts,
        "all_equal": invariant,
        "spread": float(max(values) - min(values)) if values else None,
    }
