"""The (detector × envelope) warrant matrix. ``SPEC.md`` §3.

A list of warrants answers *"is this detector any good?"*. A matrix answers the
question that actually arises in production: *"what do we know about this
detector **on the traffic we are seeing right now**?"* — and those have
different answers, because an envelope violation is a property of the input
distribution and invalidates every detector measured on that distribution at
once (``CLAUDE.md`` invariant 1).

**The three states, and why the third is an absence.** A cell holds `VALID`,
`STALE`, `REVOKED` or `REFUSED` when a warrant record occupies it. It reports
`UNVALIDATED` when **no record exists** — not by storing a placeholder, but by
there being nothing there (``DECISIONS.md`` 024). That is deliberate: an
absence cannot be dereferenced into bounds by accident, whereas a record with
empty metrics can and eventually is.

``UNVALIDATED`` is the **modal state in production** and the matrix is built
expecting most cells to be empty. Collapsing it into `REFUSED` makes the system
unusable on day one; collapsing it into `VALID` is the failure the whole product
argues against. :func:`WarrantMatrix.status` never returns one when it means the
other, and ``test_three_states`` asserts it.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from typing import Iterable, Optional, Sequence

from ..model import Warrant, WarrantKey, WarrantStatus, utc_now
from ..store import Ledger, RecordKind

__all__ = ["MatrixCell", "WarrantMatrix"]

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MatrixCell:
    """One (detector, operating point, envelope) position and what fills it.

    Args:
        key: The three-part warrant key addressing this cell.
        warrant: The most recent warrant filed here, or None for ``UNVALIDATED``.
        status: What the cell reports *now*, which is not always the warrant's
            stored status — an expired ``VALID`` warrant reports ``STALE``,
            because age is a reason to stop relying on a number independently of
            drift.
        reason: Why, in words, when the status is anything but ``VALID``.
    """

    key: WarrantKey
    warrant: Optional[Warrant]
    status: WarrantStatus
    reason: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        """Whether this cell has never been measured."""
        return self.warrant is None

    @property
    def detector_id(self) -> str:
        return self.key.detector_id

    @property
    def eval_set_id(self) -> str:
        return self.key.eval_set_id


def _calibration_note(warrant) -> str:
    """Annotate a cell whose operating point no longer spends what it declared.

    A warrant makes two separable claims -- how well the detector ranks, and
    what its threshold spends -- and can hold one while losing the other.
    ``T1-last_token`` transferred to long context with AUROC 0.826 -> 0.813 and
    a flag rate of 0.042 -> 0.065 against a declared 0.05 target: sound about
    ranking, questionable about cost.

    Shown only when it is the exception, so the annotation carries information
    rather than decorating every cell. ``CAL:n/a`` marks the case that would
    otherwise be read as reassurance: drift not shown, at an ``n`` too small to
    have shown it.
    """
    claim = getattr(warrant, "calibration", None)
    status = getattr(claim, "status", None)
    if status is None:
        return ""
    if status.value == "DRIFTED":
        return " · **CAL:DRIFTED**"
    if getattr(claim, "underpowered", False):
        return " · CAL:n/a"
    return ""


class WarrantMatrix:
    """Warrants indexed by detector and envelope, with the empty cells visible.

    Built from the ledger rather than held in memory, so the matrix and the
    audit log cannot disagree. Reading it twice at different times can give
    different answers — a warrant expires — which is correct and is why
    :meth:`status` takes a clock.

    Args:
        cells: Every populated cell.
        detectors: Every detector id the matrix knows about, including ones with
            no warrant anywhere.
        envelopes: Every envelope id, likewise. Supplying these explicitly is
            what makes the empty cells *visible*: a matrix built only from the
            warrants that exist cannot show you the ones that do not.
        now: Clock, injectable so expiry is testable.
    """

    def __init__(
        self,
        cells: Sequence[MatrixCell],
        *,
        detectors: Sequence[str],
        envelopes: Sequence[str],
        now: Optional[datetime] = None,
    ) -> None:
        self._by_key = {cell.key.as_string(): cell for cell in cells}
        self.detectors = tuple(dict.fromkeys(detectors))
        self.envelopes = tuple(dict.fromkeys(envelopes))
        self.now = now or utc_now()

    # -- construction --------------------------------------------------------- #

    @classmethod
    def from_ledger(
        cls,
        ledger: Ledger,
        *,
        detectors: Optional[Sequence[str]] = None,
        envelopes: Optional[Sequence[str]] = None,
        now: Optional[datetime] = None,
    ) -> "WarrantMatrix":
        """Build the matrix from every warrant in the ledger.

        The latest warrant per key wins, because revalidation appends rather than
        replaces and the history stays readable underneath.

        Args:
            ledger: The audit store.
            detectors: Detector ids to include. Defaults to those seen in the
                ledger; pass more to make their empty cells visible.
            envelopes: Envelope ids, likewise. **Pass the declared eval sets
                here**, or a set that has never been validated against will be
                missing from the matrix entirely rather than showing as
                ``UNVALIDATED`` — which is the difference between "we have not
                measured this" and "we have not thought about this".
            now: Clock.

        Returns:
            A :class:`WarrantMatrix`.
        """
        clock = now or utc_now()
        latest: dict[str, Warrant] = {}
        for record in ledger.query(kind=RecordKind.WARRANT):
            warrant = ledger.get_warrant(record.record_id)
            latest[warrant.key.as_string()] = warrant

        cells = [cls._cell_for(warrant, clock) for warrant in latest.values()]
        seen_detectors = sorted({w.detector_id for w in latest.values()})
        seen_envelopes = sorted({w.eval_set_id for w in latest.values()})
        return cls(
            cells,
            detectors=list(detectors) if detectors is not None else seen_detectors,
            envelopes=list(envelopes) if envelopes is not None else seen_envelopes,
            now=clock,
        )

    @staticmethod
    def _cell_for(warrant: Warrant, now: datetime) -> MatrixCell:
        """Decide what a populated cell reports, given the clock.

        A ``VALID`` warrant past its expiry reports ``STALE`` rather than
        ``VALID``. Age and drift are separate reasons to stop relying on a
        number, and a warrant can sit inside its envelope and still be too old
        to quote.
        """
        if warrant.status is WarrantStatus.VALID and warrant.is_expired(now):
            age_hours = warrant.age(now).total_seconds() / 3600.0
            return MatrixCell(
                key=warrant.key,
                warrant=warrant,
                status=WarrantStatus.STALE,
                reason=(
                    f"warrant expired: issued {age_hours:.1f}h ago, expiry was "
                    f"{warrant.expires_at.isoformat()}"
                ),
            )
        return MatrixCell(
            key=warrant.key,
            warrant=warrant,
            status=warrant.status,
            reason=warrant.status_reason,
        )

    # -- lookup ---------------------------------------------------------------- #

    def cell(self, key: WarrantKey) -> MatrixCell:
        """The cell at a key, empty if nothing has been filed there.

        Never raises for a missing cell. "We have not measured this" is an
        answer, and the modal one.
        """
        found = self._by_key.get(key.as_string())
        if found is not None:
            return found
        return MatrixCell(key=key, warrant=None, status=WarrantStatus.UNVALIDATED)

    def status(self, key: WarrantKey) -> WarrantStatus:
        """What is known at this cell right now."""
        return self.cell(key).status

    def warrant(self, key: WarrantKey) -> Optional[Warrant]:
        """The warrant at this cell, or None if the cell is empty."""
        return self.cell(key).warrant

    def cells_for_envelope(self, envelope_id: str) -> tuple[MatrixCell, ...]:
        """Every populated cell on one envelope, in detector order."""
        return tuple(
            sorted(
                (c for c in self._by_key.values() if c.eval_set_id == envelope_id),
                key=lambda c: c.detector_id,
            )
        )

    def valid_warrants(self, envelope_id: str) -> tuple[Warrant, ...]:
        """Warrants that can be relied upon on this envelope, right now.

        ``VALID`` and unexpired, and nothing else. Deliberately not
        "not REFUSED": ``STALE`` and ``REVOKED`` both mean the bounds are
        unknown or known-wrong, and a caller wanting to proceed anyway has to
        say which case it is handling.
        """
        return tuple(
            cell.warrant
            for cell in self.cells_for_envelope(envelope_id)
            if cell.warrant is not None and cell.status.can_be_relied_upon
            and not cell.warrant.is_expired(self.now)
        )

    def unvalidated_cells(self, envelope_id: str) -> tuple[WarrantKey, ...]:
        """Cells on this envelope that have never been measured.

        What :func:`~src.matrix.routing.route` enqueues, and how the matrix
        self-populates from live traffic. Requires ``detectors`` to have been
        supplied at construction — a matrix that only knows the detectors it has
        warrants for cannot tell you which ones it has not tried.
        """
        populated = {c.detector_id for c in self.cells_for_envelope(envelope_id)}
        return tuple(
            WarrantKey(detector, self._operating_point_for(detector), envelope_id)
            for detector in self.detectors
            if detector not in populated
        )

    def _operating_point_for(self, detector_id: str) -> str:
        """The operating point this detector is filed under, where one is known.

        A detector validated somewhere carries its operating point with it. One
        that has never been validated anywhere has no operating point yet, and
        the placeholder makes that visible rather than inventing a threshold.
        """
        for cell in self._by_key.values():
            if cell.detector_id == detector_id and cell.warrant is not None:
                return cell.warrant.operating_point.operating_point_id
        return "P-unassigned"

    # -- reporting ------------------------------------------------------------- #

    def summary(self) -> dict[str, int]:
        """How many cells sit in each state, empty ones included."""
        counts: dict[str, int] = {state.value: 0 for state in WarrantStatus}
        for detector in self.detectors:
            for envelope in self.envelopes:
                key = WarrantKey(detector, self._operating_point_for(detector), envelope)
                counts[self.status(key).value] += 1
        return counts

    def to_payload(self) -> dict:
        """JSON-serialisable form for ``results/``."""
        rows = []
        for detector in self.detectors:
            operating_point = self._operating_point_for(detector)
            row: dict[str, object] = {"detector_id": detector, "cells": {}}
            for envelope in self.envelopes:
                key = WarrantKey(detector, operating_point, envelope)
                cell = self.cell(key)
                entry: dict[str, object] = {"status": cell.status.value}
                if cell.reason:
                    entry["reason"] = cell.reason
                if cell.warrant is not None:
                    warrant = cell.warrant
                    entry["warrant_id"] = warrant.warrant_id
                    entry["n_test"] = warrant.n_test
                    entry["base_rate"] = warrant.base_rate
                    entry["controls_run"] = len(warrant.controls) - len(
                        warrant.inapplicable_controls()
                    )
                    entry["metrics"] = {
                        m.name: {
                            "value": m.value,
                            "ci_low": m.ci_low,
                            "ci_high": m.ci_high,
                            "kind": m.kind.value,
                            "n": m.n,
                        }
                        for m in warrant.metrics.all_metrics()
                    }
                row["cells"][envelope] = entry
            rows.append(row)
        return {
            "detectors": list(self.detectors),
            "envelopes": list(self.envelopes),
            "summary": self.summary(),
            "rows": rows,
            "rendered_at": self.now.isoformat(),
        }

    def render(self, *, mask_synthetic: bool = False) -> str:
        """A markdown table with every cell populated, empty ones included.

        Empty cells render as ``UNVALIDATED`` rather than as a blank, because a
        blank reads as "nothing to report" and the whole point is that "we have
        not measured this here" *is* the report.

        Args:
            mask_synthetic: Replace the numbers on synthetic envelopes with a
                marker. Set by the ``RESULTS.md`` renderer, because that is the
                document a judge reads and a fixture number on it would be a
                claim about a language model that no language model produced.
                The standalone ``warrant_matrix.md`` leaves them visible: it is
                an internal diagnostic, and the status still says which cells
                are which.
        """
        header = "| detector | " + " | ".join(self.envelopes) + " |"
        divider = "|---|" + "---|" * len(self.envelopes)
        lines = [header, divider]
        for detector in self.detectors:
            operating_point = self._operating_point_for(detector)
            cells = []
            for envelope in self.envelopes:
                cell = self.cell(WarrantKey(detector, operating_point, envelope))
                label = cell.status.value
                # Fail-closed, exactly as the RESULTS renderer does: an
                # envelope that does not declare a data_source cannot be shown
                # to be measured, and defaulting to "measured" would let a new
                # code path that forgets the field print fixture numbers.
                unverified = (
                    cell.warrant is not None
                    and getattr(cell.warrant.envelope, "data_source", None)
                    != "measured"
                )
                if mask_synthetic and unverified:
                    cells.append(f"{label} · FIXTURE — NOT MEASURED")
                    continue
                if cell.status is WarrantStatus.VALID and cell.warrant is not None:
                    recall = cell.warrant.metrics.recall
                    if recall is not None:
                        # Recall never travels alone (invariant 5). It moves with
                        # the budget spent, so "R=0.08 -> R=0.13" across two
                        # envelopes reads as a detector improving when the flag
                        # rate went 0.042 -> 0.065 at a frozen threshold and the
                        # lift is flat. The budget belongs in the cell that shows
                        # the recall, not in a footnote (DECISIONS 067).
                        flag = cell.warrant.metrics.flag_rate
                        label += (
                            f" R={recall.value:.2f} "
                            f"[{recall.ci_low:.2f}, {recall.ci_high:.2f}]"
                            f" @f={flag.value:.3f}"
                        )
                        # Calibration shows only when it is the exception. A
                        # VALID cell whose operating point no longer delivers
                        # its declared budget is sound about ranking and wrong
                        # about cost; unannotated it reads as unqualified.
                        label += _calibration_note(cell.warrant)
                    else:
                        fpr = cell.warrant.metrics.fpr_hard_negatives
                        if fpr is not None:
                            label += f" FPR={fpr.value:.3f} [{fpr.ci_low:.3f}, {fpr.ci_high:.3f}]"
                elif cell.status is WarrantStatus.REFUSED:
                    label = f"**{label}**"
                cells.append(label)
            lines.append(f"| {detector} | " + " | ".join(cells) + " |")
        counts = self.summary()
        lines.append("")
        lines.append(
            "Cell states: "
            + ", ".join(f"{name} {count}" for name, count in counts.items() if count)
            + f". {counts['UNVALIDATED']} of "
            f"{len(self.detectors) * len(self.envelopes)} cells have never been "
            "measured, which is the expected shape: UNVALIDATED is the modal "
            "state in production."
        )
        return "\n".join(lines)
