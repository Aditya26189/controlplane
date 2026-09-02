"""The reproduction comparison must survive a rebuild, and nothing more.

Bitwise float equality was the original rule for tier 2 and 3 of ``make
verify``. It fails in the direction that costs the most: on a **correct**
re-derivation. Bootstrap percentiles reduce over arrays whose summation order
depends on the BLAS build and the CPU, so the same code on the same data
differs in the last unit in the last place on a second machine -- observed as
``0.006129271330669258`` against ``0.006129271330669257``, which the verifier
reported as DRIFT.

A reviewer who cannot run the "prove it" button on their own laptop has been
given nothing, and "your numbers drifted" is the worst possible thing to tell
them when they did not. So the comparison now carries a tolerance.

The risk of a tolerance is that it hides the drift it exists to catch, which is
why the negative cases below matter more than the positive one: a difference at
the published precision, and one only slightly above the tolerance, must both
still fail.
"""

from __future__ import annotations

import math

import pytest

from controlplane.report.reproduce import (
    REPRODUCTION_ABS_TOL,
    REPRODUCTION_REL_TOL,
    values_agree,
)


# --------------------------------------------------------------------------- #
# What it must accept
# --------------------------------------------------------------------------- #


def test_the_observed_last_ulp_difference_is_accepted() -> None:
    """The exact pair that failed on a second machine, kept as the case."""
    assert values_agree(0.006129271330669258, 0.006129271330669257)


@pytest.mark.parametrize("value", [0.8256, 0.0794, 0.4617, 1.0, 0.5])
def test_one_ulp_either_way_is_accepted(value: float) -> None:
    """Summation order moves a double by an ulp, not by anything meaningful."""
    assert values_agree(value, math.nextafter(value, math.inf))
    assert values_agree(value, math.nextafter(value, -math.inf))


def test_identical_values_are_accepted() -> None:
    assert values_agree(0.8256, 0.8256)
    assert values_agree(0, 0)
    assert values_agree(None, None)


def test_zero_is_handled_by_the_absolute_floor() -> None:
    """A relative tolerance degenerates at zero, and a measured FPR can be zero."""
    assert values_agree(0.0, 0.0)
    assert values_agree(0.0, REPRODUCTION_ABS_TOL / 2)
    assert not values_agree(0.0, 1e-9)


# --------------------------------------------------------------------------- #
# What it must still reject -- the half that makes the tolerance safe
# --------------------------------------------------------------------------- #


def test_a_difference_at_the_published_precision_is_rejected() -> None:
    """Numbers are published to four decimals; that scale must never pass."""
    assert not values_agree(0.8256, 0.8257)
    assert not values_agree(0.0794, 0.0795)


def test_a_difference_just_above_the_tolerance_is_rejected() -> None:
    """The boundary is where a too-generous tolerance would be discovered."""
    base = 0.5
    assert not values_agree(base, base * (1 + REPRODUCTION_REL_TOL * 100))


def test_the_tolerance_is_far_tighter_than_the_published_precision() -> None:
    """A guard on the constant itself.

    The argument for the tolerance is that it is orders of magnitude below any
    number this project publishes. If someone widens it later, this is what
    says so.
    """
    assert REPRODUCTION_REL_TOL <= 1e-10, (
        "the reproduction tolerance is approaching the precision of published "
        "numbers; at that point it stops being float noise and starts hiding "
        "drift"
    )
    assert REPRODUCTION_ABS_TOL <= 1e-12


# --------------------------------------------------------------------------- #
# Non-numeric values are compared exactly
# --------------------------------------------------------------------------- #


def test_a_missing_bound_never_matches_a_present_one() -> None:
    """None means the metric does not carry that bound. It is not zero."""
    assert not values_agree(None, 0.0)
    assert not values_agree(0.0, None)


def test_booleans_are_not_read_as_numbers() -> None:
    """``bool`` is an ``int`` in Python; True must not compare equal to 1.0."""
    assert not values_agree(True, 1.0)
    assert not values_agree(1.0, True)
    assert values_agree(True, True)


def test_strings_are_compared_exactly() -> None:
    assert values_agree("VALID", "VALID")
    assert not values_agree("VALID", "REFUSED")
