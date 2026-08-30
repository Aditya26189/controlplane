"""Rendering results into markdown and plots.

Nothing here computes a number. Everything it draws or prints comes from an
artifact in ``results/``, which is what makes "every number in a document is
computed by code" (invariant 8) checkable rather than aspirational.
"""

from .plots import plot_tier_ladder
from .results import FIXTURE_MARKER, ResultsRefusal, render_results

__all__ = ["FIXTURE_MARKER", "ResultsRefusal", "plot_tier_ladder", "render_results"]
