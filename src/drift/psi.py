"""Population Stability Index against a warrant's stored reference bins.

PSI is the drift measure Indian banking risk teams already read, which is why
``SPEC.md`` §5.2 picks it over anything more fashionable: a number whose
vocabulary the audience owns needs no defending.

``PSI = sum_i (a_i - e_i) * ln(a_i / e_i)`` over bins, where ``e`` is the
reference proportion stored in the warrant's envelope and ``a`` is the live
window's. Bands from ``config.drift``: below ``psi_stable`` is inside, up to
``psi_significant`` is moderate, above it is significant.

**The empty-bin trap.** ``ln(0)`` diverges, so a live window with no mass in a
bin the reference used makes PSI infinite. Every implementation smooths it, and
the smoothing constant silently sets the number: at eps=1e-6 a single empty bin
contributes ~13, at eps=1e-4 it contributes ~9, and both are "PSI is huge"
dressed as a precise quantity. So the epsilon is declared in config rather than
buried, and :class:`PsiResult` reports **how many bins were smoothed**. A PSI
driven by smoothed bins is a statement about absence of data in a region, not
about a measured shift there, and the caller can tell the difference.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence

import numpy as np

from ..model.enums import EnvelopeState
from ..model.findings import EnvelopeFeature

__all__ = ["PsiResult", "population_stability_index", "state_for_psi"]

#: Floor applied to a bin proportion before the logarithm. Declared rather than
#: chosen inline: it sets the magnitude of every PSI driven by an empty bin.
DEFAULT_EPSILON = 1e-6


@dataclasses.dataclass(frozen=True)
class PsiResult:
    """A PSI value and the facts needed to decide whether to believe it.

    Args:
        feature: Which envelope feature this measures.
        psi: The index. Larger means more shift; the bands are in config.
        n_live: Live window size. A verdict below the configured minimum is the
            caller's to refuse — this type reports, it does not gate.
        n_reference: Items the reference bins were built from.
        bins_smoothed: How many live bins were empty against a non-empty
            reference bin and therefore floored at ``epsilon``. **Read this
            before quoting the PSI**: a value driven by smoothed bins says the
            live window has no mass where the reference did, which is a claim
            about missing data rather than about a measured shift.
        epsilon: The floor actually applied.
        per_bin: Each bin's contribution, so a large PSI can be attributed to a
            region rather than accepted as a scalar.
    """

    feature: str
    psi: float
    n_live: int
    n_reference: int
    bins_smoothed: int
    epsilon: float
    per_bin: tuple[float, ...]

    #: Share of the index's magnitude contributed by bins that were floored at
    #: ``epsilon``. Computed where the mask is known rather than reconstructed
    #: from ``per_bin``, which cannot say which bins were smoothed.
    smoothed_share: float = 0.0

    @property
    def driven_by_smoothing(self) -> bool:
        """True when floored bins contribute most of the index.

        Half is the threshold: if the bins with no live data account for more
        of the index than the bins with data, the number is mostly an artefact
        of the epsilon and belongs in a report as *absence of data in a region*
        rather than as a measured shift of that size.
        """
        return self.bins_smoothed > 0 and self.smoothed_share > 0.5

    def to_payload(self) -> dict:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "n_live": self.n_live,
            "n_reference": self.n_reference,
            "bins_smoothed": self.bins_smoothed,
            "epsilon": self.epsilon,
            "smoothed_share": self.smoothed_share,
            "driven_by_smoothing": self.driven_by_smoothing,
            "per_bin": list(self.per_bin),
        }


def population_stability_index(
    feature: EnvelopeFeature,
    live_values: Sequence[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> PsiResult:
    """PSI of a live sample against a reference feature's stored bins.

    The reference bins come from the warrant, not from the live data. Rebinning
    on the live sample would compare each window against itself and report
    stability forever, which is the failure mode that makes a drift monitor
    worse than none.

    Args:
        feature: Reference bins and probabilities, from the warrant's envelope.
        live_values: The live window's values for this feature.
        epsilon: Floor applied to a zero proportion before the logarithm.

    Returns:
        A :class:`PsiResult`.

    Raises:
        ValueError: If the live window is empty. An empty window has no
            distribution, and returning 0.0 would read as "no drift".
    """
    live = np.asarray(list(live_values), dtype=float)
    if live.size == 0:
        raise ValueError(
            f"{feature.name}: empty live window. There is no distribution to "
            "compare, and a PSI of 0.0 would read as stability."
        )

    edges = np.asarray(feature.bin_edges, dtype=float)
    expected = np.asarray(feature.bin_probabilities, dtype=float)

    # Values outside the reference range land in the first or last bin rather
    # than being dropped. Discarding them would hide the clearest possible
    # drift signal: traffic that has moved off the reference support entirely.
    indices = np.clip(np.digitize(live, edges[1:-1], right=False), 0, len(expected) - 1)
    counts = np.bincount(indices, minlength=len(expected)).astype(float)
    actual = counts / counts.sum()

    smoothed_mask = (actual <= 0.0) & (expected > 0.0)
    actual_safe = np.where(actual <= 0.0, epsilon, actual)
    expected_safe = np.where(expected <= 0.0, epsilon, expected)

    per_bin = (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)
    psi = float(per_bin.sum())

    magnitude = float(np.abs(per_bin).sum())
    smoothed_share = (
        float(np.abs(per_bin[smoothed_mask]).sum() / magnitude) if magnitude > 0 else 0.0
    )

    return PsiResult(
        feature=feature.name,
        psi=psi,
        n_live=int(live.size),
        n_reference=0,
        bins_smoothed=int(smoothed_mask.sum()),
        epsilon=epsilon,
        per_bin=tuple(float(x) for x in per_bin),
        smoothed_share=smoothed_share,
    )


def state_for_psi(psi: float, *, stable: float, significant: float) -> EnvelopeState:
    """Map a PSI onto a rung of the revocation ladder.

    Bands from ``config.drift``. Boundaries are inclusive at the lower edge so a
    PSI exactly at ``psi_significant`` revokes rather than sitting one rung
    below it: at the boundary the conservative reading is the correct one.
    """
    if not math.isfinite(psi):
        return EnvelopeState.SIGNIFICANT_SHIFT
    if psi < stable:
        return EnvelopeState.INSIDE
    if psi < significant:
        return EnvelopeState.MODERATE_SHIFT
    return EnvelopeState.SIGNIFICANT_SHIFT
