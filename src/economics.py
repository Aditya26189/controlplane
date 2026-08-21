"""Cascade economics: the three-policy comparison and the headline lift.

``lift = R / f`` -- the probe's recall divided by its **measured** flag rate.
It is the one number this repo exists to produce, and its value is that both
the base error rate and the judge's own accuracy appear in every policy and
cancel from the ratio (SPEC.md §7, DECISIONS.md 009).

The rest of this module is built in Stage 5; ``lift`` lives here from Stage 4
because ``evaluate.py`` bootstraps it and there must be exactly one definition.
"""

import logging

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
