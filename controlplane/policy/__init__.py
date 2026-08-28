"""The policy engine. ``SPEC.md`` §7.

Policy is versioned, content-hashed data, not code (§7.1). Every operating
point a bundle relies on is resolved against the warrant matrix **at load
time**, and anything missing, expired, refused, unvalidated or below a declared
minimum stops the bundle loading (§7.2). Only then may a rule run.

The ordering is enforced by the types: :class:`RegoEngine` takes a
:class:`ResolvedBundle`, and the only way to obtain one is
:func:`resolve_bundle`.
"""

from .bundle import (
    CalibrationRequirement,
    OnCalibrationDrift,
    PolicyBundle,
    WarrantRequirement,
    parse_duration,
)
from .engine import PolicyDecision, RegoEngine, build_engine
from .errors import BundleError, PolicyError, WarrantResolutionError
from .objective import (
    ErrorRates,
    ThresholdChoice,
    choose_threshold,
    error_rates_at,
    weighted_error,
)
from .resolution import ResolvedBundle, ResolvedRequirement, resolve_bundle
from .runner import (
    ProfileComparison,
    ProfileOutcome,
    issue_operating_points,
    run_profile_comparison,
)

__all__ = [
    "BundleError",
    "CalibrationRequirement",
    "ErrorRates",
    "OnCalibrationDrift",
    "PolicyBundle",
    "PolicyDecision",
    "PolicyError",
    "ProfileComparison",
    "ProfileOutcome",
    "RegoEngine",
    "ResolvedBundle",
    "ResolvedRequirement",
    "ThresholdChoice",
    "WarrantRequirement",
    "WarrantResolutionError",
    "build_engine",
    "choose_threshold",
    "error_rates_at",
    "issue_operating_points",
    "parse_duration",
    "resolve_bundle",
    "run_profile_comparison",
    "weighted_error",
]
