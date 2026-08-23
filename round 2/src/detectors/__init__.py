"""Detector adapters and the shared pooling/probe machinery.

Every detector reduces to the same contract: given items, produce a score per
item, higher meaning more likely to be a problem. What differs is the access
tier each one needs — activations, logprobs, or text alone — and that is exactly
what the tier ladder measures, because access is what an enterprise is deciding
whether to pay for.
"""

from .aggregation import AggregationError, aggregate, max_rolling_means, mean_pool
from .probe import LinearProbe, ProbeError, ProbeFit, select_regularisation

__all__ = [
    "AggregationError",
    "LinearProbe",
    "ProbeError",
    "ProbeFit",
    "aggregate",
    "max_rolling_means",
    "mean_pool",
    "select_regularisation",
]
