"""Metrics that cannot be constructed in violation of their own kind.

This module is where three ``CLAUDE.md`` invariants stop being prose:

* **Invariant 4 — no point estimate reaches a user.** An ``ESTIMATED`` metric
  without an interval is not a metric this codebase can represent.
* **Invariant 5 — precision and recall travel together, and there is no F1.**
  :class:`WarrantMetrics` requires both, and :class:`Metric` refuses a name that
  denotes a blended score.
* **Yield is exact, rate is estimated.** An ``EXACT`` metric carrying an
  interval is a rate mislabelled as a count, which is the most damaging error
  available here: it converts a free exact claim into an unbacked estimate and
  nothing raises.

``test_yield_vs_rate`` asserts all of it. The point of putting the rules in
``__post_init__`` rather than in a renderer is that a record read back from the
store a year later is checked by the same code that checked it on the way in.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Optional

__all__ = ["Metric", "MetricError", "WarrantMetrics"]

from .enums import MetricKind


class MetricError(ValueError):
    """Raised when a metric would misrepresent what is known about it."""


# Names denoting a blended precision/recall score. Invariant 5 forbids one
# anywhere in code, output or documentation: a single number that trades a false
# positive (one wasted review) against a false negative (a user acting on a
# wrong answer) at a fixed exchange rate hides the only choice that matters
# here. Refused by name so it cannot arrive through a config or a report label.
_BLENDED_NAME = re.compile(
    r"(^|[^a-z0-9])(f1|f_1|fbeta|f_beta|f2|f_2|fscore|f_score|f_measure|fmeasure)"
    r"([^a-z0-9]|$)",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class Metric:
    """One measured quantity, carrying what is and is not known about it.

    Args:
        name: What was measured, e.g. ``"recall"`` or ``"confirmed_errors"``.
        value: The measurement.
        kind: ``EXACT`` for a count of reviewed, confirmed items; ``ESTIMATED``
            for anything requiring inference about unreviewed items.
        n: The sample or population size the value came from. Required for both
            kinds — invariant 4 says every interval names its ``n``, and an
            exact count is uninterpretable without its denominator either.
        ci_low: Lower bound. Required for ``ESTIMATED``, forbidden for ``EXACT``.
        ci_high: Upper bound. Same rule.
        ci_level: Coverage of the interval, e.g. ``0.95``.
        unit: ``"rate"``, ``"count"``, ``"ratio"`` or similar. Rendered beside
            the value so a count is never read as a proportion.
        estimator: How the interval was produced, e.g.
            ``"bootstrap-percentile-1000"``. An interval whose construction is
            unstated is not reproducible.

    Raises:
        MetricError: If the metric would misrepresent its own kind.
    """

    name: str
    value: float
    kind: MetricKind
    n: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    ci_level: Optional[float] = None
    unit: str = "rate"
    estimator: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise MetricError("a metric must be named; an unnamed number is not a claim")
        if _BLENDED_NAME.search(self.name):
            raise MetricError(
                f"{self.name!r} names a blended precision/recall score. CLAUDE.md "
                "invariant 5 forbids one anywhere: a false positive costs one "
                "wasted review and a false negative costs a user acting on a wrong "
                "answer, and a blended score fixes that exchange rate silently. "
                "Report precision and recall separately."
            )
        if not math.isfinite(self.value):
            raise MetricError(f"{self.name}: value must be finite, got {self.value}")
        if self.n < 0:
            raise MetricError(f"{self.name}: n must be non-negative, got {self.n}")

        has_interval = self.ci_low is not None or self.ci_high is not None

        if self.kind is MetricKind.EXACT:
            if has_interval:
                raise MetricError(
                    f"{self.name}: an EXACT metric carries no interval. It is a "
                    "count of reviewed, confirmed items — there is nothing to "
                    "estimate. Attaching an interval converts a free exact claim "
                    "into an unbacked estimate (CLAUDE.md, yield versus rate)."
                )
            if self.ci_level is not None:
                raise MetricError(
                    f"{self.name}: an EXACT metric has no interval, so it has no "
                    "coverage level"
                )
            if self.estimator is not None:
                raise MetricError(
                    f"{self.name}: an EXACT metric is counted, not estimated, so "
                    f"it has no estimator; got {self.estimator!r}"
                )
            return

        # ESTIMATED from here down.
        if self.ci_low is None or self.ci_high is None:
            raise MetricError(
                f"{self.name}: an ESTIMATED metric must carry both bounds. "
                "CLAUDE.md invariant 4 — no point estimate reaches a user."
            )
        if not math.isfinite(self.ci_low) or not math.isfinite(self.ci_high):
            raise MetricError(f"{self.name}: interval bounds must be finite")
        if self.ci_low > self.ci_high:
            raise MetricError(
                f"{self.name}: interval is inverted, [{self.ci_low}, {self.ci_high}]"
            )
        if not self.ci_low <= self.value <= self.ci_high:
            raise MetricError(
                f"{self.name}: value {self.value} lies outside its own interval "
                f"[{self.ci_low}, {self.ci_high}]. Usually a sign the point "
                "estimate and the interval were computed from different data."
            )
        if self.n <= 0:
            raise MetricError(
                f"{self.name}: an ESTIMATED metric must name its n. "
                "CLAUDE.md invariant 4 — every interval names its n."
            )
        if self.ci_level is None or not 0.0 < self.ci_level < 1.0:
            raise MetricError(
                f"{self.name}: an ESTIMATED metric must state its coverage level "
                f"in (0, 1), got {self.ci_level!r}"
            )
        if not self.estimator:
            raise MetricError(
                f"{self.name}: an ESTIMATED metric must state how its interval was "
                "produced. An interval whose construction is unstated is not "
                "reproducible."
            )

    # -- rendering ---------------------------------------------------------- #

    @property
    def has_interval(self) -> bool:
        """True when the metric carries bounds — equivalently, when it is estimated."""
        return self.ci_low is not None and self.ci_high is not None

    @property
    def width(self) -> Optional[float]:
        """Interval width, or None for an exact count.

        The number that answers "how much precision did those labels buy?", and
        the one the price list is denominated in.
        """
        if not self.has_interval:
            return None
        return self.ci_high - self.ci_low  # type: ignore[operator]

    def render(self, digits: int = 4) -> str:
        """Format for display, never dropping the interval or the ``n``.

        There is no code path that renders an estimated value alone: invariant 4
        is enforced at the point of formatting as well as at construction,
        because a formatter is where a bound usually goes missing.
        """
        if self.kind is MetricKind.EXACT:
            shown = f"{int(self.value)}" if self.unit == "count" else f"{self.value:.{digits}f}"
            return f"{shown} (exact, n={self.n})"
        return (
            f"{self.value:.{digits}f} "
            f"[{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}] "
            f"({int(self.ci_level * 100)}% CI, n={self.n})"  # type: ignore[operator]
        )

    def widened(self, factor: float, reason: str) -> "Metric":
        """Return the same estimate with a wider interval and a named reason.

        Used by the ``STALE`` rung of the revocation ladder (``SPEC.md`` §5.3):
        when the envelope has moved but not far enough to revoke, the honest
        response is to keep claiming the value with less precision, not to keep
        claiming it with the same precision.

        Widening only ever loses information, so it is safe in the direction
        that matters. The reason is appended to the estimator string so a widened
        interval can never be mistaken for a measured one.

        Args:
            factor: Multiplier on the half-width. Must be >= 1.
            reason: Why the interval was widened, recorded in the estimator.

        Raises:
            MetricError: If applied to an exact metric, or if ``factor`` < 1.
        """
        if self.kind is MetricKind.EXACT:
            raise MetricError(
                f"{self.name}: an exact count has no interval to widen. If it "
                "needs one, it was never exact."
            )
        if factor < 1.0:
            raise MetricError(
                f"{self.name}: widening factor must be >= 1, got {factor}. "
                "Narrowing an interval without new evidence is the failure this "
                "project exists to prevent."
            )
        half = (self.ci_high - self.ci_low) / 2.0  # type: ignore[operator]
        centre = self.value
        return dataclasses.replace(
            self,
            ci_low=centre - half * factor,
            ci_high=centre + half * factor,
            estimator=f"{self.estimator}; widened x{factor:g} ({reason})",
        )


@dataclasses.dataclass(frozen=True)
class WarrantMetrics:
    """The measured bounds a warrant asserts.

    Precision and recall are both required fields, which is invariant 5 made
    structural: there is no way to construct a warrant that claims one without
    the other, so there is no way for a report to quote recall alone.

    ``flag_rate`` is here because lift is ``recall / flag_rate`` and quoting
    lift without the flag rate it was computed at is how a lift number becomes
    unfalsifiable. ``confirmed_errors`` is the exact yield — the free claim —
    and sits beside the estimated rates so the difference is visible at a
    glance rather than explained in a footnote.

    Args:
        auroc: Threshold-free ranking quality. Estimated.
        recall: Estimated. Sensitive to the unreviewed pool, so this is the one
            that costs money (``SPEC.md`` §6.2).
        precision: Estimated. Free in labels, because the flagged stratum is
            already reviewed — free is a statement about label cost, not about
            certainty (``DECISIONS.md`` 022).
        flag_rate: The measured rate at which this operating point fires.
        confirmed_errors: Count of reviewed, confirmed true positives. Exact.
        fpr_hard_negatives: FPR on the hard-negative set, where it means
            something. Optional because not every envelope has one.
        base_rate: Positive-class prevalence on the set these were measured on.
            Carried here, beside the metrics rather than only on the warrant,
            because **lift is not interpretable without it**: the maximum
            achievable lift is ``1 / max(base_rate, flag_rate)``, so a detector
            on an enriched set sits near a low ceiling however good it is
            (``DECISIONS.md`` 047).
        ranking_absent_reason: Why ``auroc``/``recall``/``precision`` are
            absent, when they are. Carried because **the metrics alone cannot
            distinguish the two causes**: a single-class envelope and a
            threshold that flagged nothing both leave the ranking triple empty
            with a present ``flag_rate`` and ``fpr``. Issuance used to infer
            "single class" from an absent AUROC and wrote "measured no
            positives in this eval set" into a refusal for a 600-item set that
            had plenty (``DECISIONS.md`` 108).
        extra: Any additional metrics, each carrying its own kind.
    """

    auroc: Optional[Metric]
    recall: Optional[Metric]
    precision: Optional[Metric]
    flag_rate: Metric
    confirmed_errors: Metric
    fpr_hard_negatives: Optional[Metric] = None
    base_rate: Optional[float] = None
    ranking_absent_reason: Optional[str] = None
    extra: tuple[Metric, ...] = ()

    def __post_init__(self) -> None:
        # Ranking metrics are all-or-nothing. On a single-class envelope --
        # hard-negatives-200 contains no positives by construction -- AUROC,
        # recall and precision are undefined, and the honest record says so
        # rather than reporting a number computed from an empty denominator.
        # Invariant 5 survives: precision and recall are absent *together*, so
        # there is still no way to claim one without the other.
        present = [
            name
            for name in ("auroc", "recall", "precision")
            if getattr(self, name) is not None
        ]
        if present and len(present) != 3:
            missing = sorted({"auroc", "recall", "precision"} - set(present))
            raise MetricError(
                f"ranking metrics must be present together or absent together; "
                f"got {present} without {missing}. Precision and recall travel "
                "together (CLAUDE.md invariant 5), and an AUROC without them "
                "describes a curve nobody is operating on."
            )
        for name in ("auroc", "recall", "precision", "flag_rate"):
            metric = getattr(self, name)
            if metric is None:
                continue
            if metric.kind is not MetricKind.ESTIMATED:
                raise MetricError(
                    f"{name} must be ESTIMATED. It is a rate measured on a finite "
                    "sample of a distribution; calling it exact claims certainty "
                    "the sample does not supply."
                )
        if self.confirmed_errors.kind is not MetricKind.EXACT:
            raise MetricError(
                "confirmed_errors must be EXACT. It is a count of reviewed, "
                "confirmed items — the free claim that the price list is built "
                "on (CLAUDE.md, yield versus rate)."
            )
        if self.confirmed_errors.unit != "count":
            raise MetricError(
                f"confirmed_errors must have unit 'count', got "
                f"{self.confirmed_errors.unit!r}; a yield read as a rate is the "
                "exact confusion this type exists to prevent"
            )
        if self.fpr_hard_negatives is not None:
            if self.fpr_hard_negatives.kind is not MetricKind.ESTIMATED:
                raise MetricError("fpr_hard_negatives must be ESTIMATED")

    def all_metrics(self) -> tuple[Metric, ...]:
        """Every metric on this warrant, in report order."""
        ordered = [
            metric
            for metric in (self.auroc, self.recall, self.precision)
            if metric is not None
        ]
        ordered += [self.flag_rate, self.confirmed_errors]
        if self.fpr_hard_negatives is not None:
            ordered.append(self.fpr_hard_negatives)
        ordered.extend(self.extra)
        return tuple(ordered)

    @property
    def lift(self) -> Metric:
        """Recall over measured flag rate, as an estimated ratio.

        Carried as a metric rather than a float so it cannot be quoted without
        its interval, and so the flag rate it was computed at travels with it.
        Interval bounds come from the recall interval at the measured flag rate:
        the flag rate is measured on the same run and its own uncertainty is an
        order of magnitude smaller, so treating it as fixed widens nothing.

        ``DECISIONS.md`` 018: lift is never quoted without precision. That is a
        reporting rule the renderer enforces, and it is why precision is a
        required field on this record rather than an optional one.
        """
        if self.recall is None:
            raise MetricError(
                "lift is recall / flag_rate and recall is undefined on this "
                "envelope, which contains only one class. There is no lift to "
                "report and reporting one would invent the missing half."
            )
        f = self.flag_rate.value
        if f <= 0:
            raise MetricError(
                "lift is recall / flag_rate and the measured flag rate is zero; "
                "nothing was flagged, so there is no lift to report"
            )
        ceiling = self.lift_ceiling
        context = f"recall interval at measured flag rate f={f:.4f}"
        if ceiling is not None:
            context += (
                f"; ceiling {ceiling:.3f} at base rate {self.base_rate:.4f} "
                f"(max achievable lift is 1/max(base_rate, flag_rate))"
            )
        return Metric(
            name="lift",
            value=self.recall.value / f,
            kind=MetricKind.ESTIMATED,
            n=self.recall.n,
            ci_low=self.recall.ci_low / f,  # type: ignore[operator]
            ci_high=self.recall.ci_high / f,  # type: ignore[operator]
            ci_level=self.recall.ci_level,
            unit="ratio",
            estimator=context,
        )

    @property
    def lift_ceiling(self) -> Optional[float]:
        """The largest lift physically achievable at this base rate and budget.

        Flagging a fraction ``f`` of items when a fraction ``b`` are positive
        caps true positives at ``min(f, b)``, so::

            R    <= min(f, b) / b
            lift  = R / f  <=  min(1/b, 1/f)  =  1 / max(b, f)

        This is why ``MIN_LIFT_LOWER_BOUND = 1.0`` is a **base-rate-dependent**
        bar despite being stated as an absolute one. On an enriched set the
        ceiling is low and a genuinely strong detector sits near the floor: at
        base rate 0.51 and flag rate 0.62 the ceiling is 1.61, so a measured
        lift of 1.28 is 79% of everything available — not "barely useful", which
        is how 1.28 reads without this number beside it.

        Returns:
            The ceiling, or None when the base rate was not recorded.
        """
        if self.base_rate is None or self.recall is None:
            return None
        denominator = max(self.base_rate, self.flag_rate.value)
        if denominator <= 0:
            return None
        return 1.0 / denominator

    @property
    def lift_fraction_of_ceiling(self) -> Optional[float]:
        """Measured lift as a fraction of the achievable maximum.

        The number that makes lift comparable across envelopes with different
        base rates, which raw lift is not.
        """
        ceiling = self.lift_ceiling
        if ceiling is None or ceiling <= 0:
            return None
        return self.lift.value / ceiling
