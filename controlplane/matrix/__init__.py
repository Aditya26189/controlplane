"""The (detector x envelope) warrant matrix, and routing on it.

A list of warrants answers "is this detector any good?". A matrix answers
"what do we know about this detector on the traffic we are seeing right now?",
and those have different answers -- which is invariant 1, and the reason drift
cannot be handled by downgrading a tier.
"""

from .matrix import MatrixCell, WarrantMatrix
from .routing import Profile, RoutingDecision, route

__all__ = [
    "MatrixCell",
    "Profile",
    "RoutingDecision",
    "WarrantMatrix",
    "route",
]
