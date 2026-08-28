"""The validation harness: controls, /validate, and warrant issuance or refusal.

The five controls of ``SPEC.md`` §2.1 run on every validation and any failure
refuses the warrant. Two of them are *negative* controls — label shuffle and
null feature — which assert that this pipeline can produce a null result when
there is no signal. A pipeline that cannot fail cannot be trusted when it
succeeds, and that sentence is the project's own thesis turned on its own code.
"""

from .stats import (
    MeasurementError,
    assert_polarity,
    auroc,
    bootstrap_interval,
    confusion_at,
    estimated,
    exact_count,
    false_positive_rate_at,
    flag_rate_at,
    precision_at,
    recall_at,
    threshold_for_flag_rate,
    weighted_error,
)

__all__ = [
    "MeasurementError",
    "assert_polarity",
    "auroc",
    "bootstrap_interval",
    "confusion_at",
    "estimated",
    "exact_count",
    "false_positive_rate_at",
    "flag_rate_at",
    "precision_at",
    "recall_at",
    "threshold_for_flag_rate",
    "weighted_error",
]
