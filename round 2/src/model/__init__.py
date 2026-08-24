"""Core record types: findings, warrants, certificates and the enums behind them.

These are records, not mutable state — every one is a frozen dataclass whose
``__post_init__`` refuses to build something that would misstate what is known.
That placement is deliberate: a record read back from the store a year from now
is checked by the same code that checked it on the way in, so an invariant
cannot be enforced only at the boundary where it was convenient.

Where each ``CLAUDE.md`` invariant is enforced:

===========  ==========================================================
Invariant    Enforced by
===========  ==========================================================
1            :class:`WarrantKey` — all three elements required
2            :class:`Warrant` refuses ``UNVALIDATED``; absence is the state
3            :class:`Warrant` — a failed control forces ``REFUSED``
4            :class:`Metric` — ``ESTIMATED`` needs bounds, level and ``n``
5            :class:`Metric` refuses blended names; :class:`WarrantMetrics`
             requires precision and recall together
8            :func:`content_hash` and the store's chain
===========  ==========================================================
"""

from .certificate import UNSEALED, Certificate, CertificateError, Resolution
from .enums import (
    AccessTier,
    Action,
    Category,
    ConfidenceBand,
    EnvelopeState,
    MetricKind,
    Reversibility,
    Severity,
    WarrantStatus,
)
from .findings import (
    DistributionEnvelope,
    EnvelopeFeature,
    EnvelopeMatchResult,
    Finding,
    FindingError,
    OperatingPoint,
    Span,
)
from .metrics import Metric, MetricError, WarrantMetrics
from .serde import (
    SerdeError,
    canonical_json,
    chain_hash,
    content_hash,
    from_jsonable,
    parse_utc,
    to_jsonable,
    utc_now,
)
from .warrant import ControlResult, Warrant, WarrantError, WarrantKey

__all__ = [
    "UNSEALED",
    "AccessTier",
    "Action",
    "Category",
    "Certificate",
    "CertificateError",
    "ConfidenceBand",
    "ControlResult",
    "DistributionEnvelope",
    "EnvelopeFeature",
    "EnvelopeMatchResult",
    "EnvelopeState",
    "Finding",
    "FindingError",
    "Metric",
    "MetricError",
    "MetricKind",
    "OperatingPoint",
    "Resolution",
    "Reversibility",
    "SerdeError",
    "Severity",
    "Span",
    "Warrant",
    "WarrantError",
    "WarrantKey",
    "WarrantMetrics",
    "WarrantStatus",
    "canonical_json",
    "chain_hash",
    "content_hash",
    "from_jsonable",
    "parse_utc",
    "to_jsonable",
    "utc_now",
]
