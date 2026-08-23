"""Rendering results into markdown and plots.

Nothing here computes a number. Everything it draws or prints comes from an
artifact in ``results/``, which is what makes "every number in a document is
computed by code" (invariant 8) checkable rather than aspirational.
"""

from .plots import plot_tier_ladder

__all__ = ["plot_tier_ladder"]
