"""Rendering ``results/RESULTS.md``, with a hard refusal to print fixture numbers.

**The guard this module exists for.** Every activation-tier number in this repo
is currently produced by a synthetic generator we wrote. It is internally valid
and it is not evidence about a language model. Until the real extraction lands,
the only thing standing between a fixture number and a slide is somebody
remembering — and "somebody remembering" is exactly the control this project
argues is insufficient.

So the renderer **refuses**. A cell whose envelope carries
``data_source="synthetic"`` renders as ``FIXTURE — NOT MEASURED`` and its
numbers are not printed at all. Not greyed, not footnoted: absent. A number that
is not on the page cannot be read off it.

The refusal is fail-closed in both directions. An envelope that does not declare
its ``data_source`` at all is treated as unverified and refused too, because the
failure mode is a new code path that forgets to set it.

``DECISIONS.md`` 046.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..matrix import WarrantMatrix
from ..model import Metric, MetricKind, Warrant, WarrantStatus

__all__ = [
    "FIXTURE_MARKER",
    "RequiredExtraction",
    "ResultsRefusal",
    "render_metric",
    "render_results",
]

_LOG = logging.getLogger(__name__)

#: What a fixture cell renders as instead of a number.
FIXTURE_MARKER = "FIXTURE — NOT MEASURED"


class ResultsRefusal(RuntimeError):
    """Raised when a caller asks the renderer to print an unmeasured number."""


@dataclasses.dataclass(frozen=True)
class RequiredExtraction:
    """An eval set the submission needs measured, and what blocks without it.

    Listed in ``RESULTS.md`` so the outstanding GPU work is visible in the
    artifact a reader opens first, rather than living in a plan. A dependency
    that is only tracked in someone's head is a dependency that slips.
    """

    eval_set_id: str
    needed_for: str
    blocks: str


#: The extractions the submission cannot ship without. Beat 4 and the tier
#: ladder both need real activations, and there is no fallback for either.
REQUIRED_EXTRACTIONS: tuple[RequiredExtraction, ...] = (
    RequiredExtraction(
        eval_set_id="triviaqa-600",
        needed_for="the tier ladder — what T1 access actually buys over T2 and T3",
        blocks="Phase 10 (demo shows real numbers), Phase 12 (README numbers must "
        "trace to results/)",
    ),
    RequiredExtraction(
        eval_set_id="triviaqa-longctx-600",
        needed_for="Beat 4 — the envelope shift that drives revocation and routing",
        blocks="Phase 10; the drift demo has no measured basis without it",
    ),
)


def render_metric(metric: Optional[Metric], *, digits: int = 4) -> str:
    """Render one metric, or say plainly that it does not exist.

    A missing metric renders as ``n/a`` with no number anywhere near it. The
    alternative — a dash, a blank, a zero — is read as a value by at least one
    reader.
    """
    if metric is None:
        return "n/a"
    return metric.render(digits)


def _envelope_is_measured(warrant: Warrant) -> tuple[bool, str]:
    """Whether a warrant's numbers may be printed as results.

    Returns ``(allowed, reason)``. Fail-closed: anything other than an envelope
    explicitly marked ``measured`` is refused, because the failure mode is a new
    code path that forgets to set the field rather than one that sets it wrong.
    """
    envelope = warrant.envelope
    source = getattr(envelope, "data_source", None)
    if source == "measured":
        return True, ""
    if source == "synthetic":
        return False, (
            f"{envelope.eval_set_id} is a synthetic fixture; its numbers describe "
            "a generator we wrote, not a language model"
        )
    return False, (
        f"{envelope.eval_set_id} does not declare a data_source, so it cannot be "
        "shown to be measured. Refusing rather than assuming."
    )


def render_results(
    matrix: WarrantMatrix,
    *,
    provenance: dict[str, Any],
    extra_sections: Sequence[tuple[str, str]] = (),
) -> str:
    """Render ``RESULTS.md`` from the matrix, refusing every fixture number.

    Args:
        matrix: The populated warrant matrix.
        provenance: The provenance block from any artifact in this run.
        extra_sections: ``(heading, body)`` pairs appended after the matrix.

    Returns:
        Markdown. Every number in it comes from an envelope marked ``measured``;
        every other cell reads :data:`FIXTURE_MARKER`.
    """
    lines: list[str] = ["# RESULTS", ""]

    measured: list[Warrant] = []
    fixtures: list[tuple[Warrant, str]] = []
    for envelope_id in matrix.envelopes:
        for cell in matrix.cells_for_envelope(envelope_id):
            if cell.warrant is None:
                continue
            allowed, reason = _envelope_is_measured(cell.warrant)
            (measured if allowed else fixtures).append(
                cell.warrant if allowed else (cell.warrant, reason)
            )

    # -- the refusal banner, first, so it cannot be scrolled past ------------- #
    if fixtures:
        lines += [
            "> [!WARNING]",
            f"> **{len(fixtures)} of {len(fixtures) + len(measured)} populated "
            "cells are synthetic fixtures and their numbers are not printed "
            "below.**",
            ">",
            "> A fixture number is internally valid and is not evidence about a "
            "language model. The renderer refuses to print them rather than "
            "relying on a reader noticing a footnote. Cells affected:",
            ">",
        ]
        for warrant, reason in fixtures:
            lines.append(f"> - `{warrant.detector_id}` on `{warrant.eval_set_id}` — {reason}")
        lines.append("")

    # -- outstanding measurement --------------------------------------------- #
    outstanding = [
        required
        for required in REQUIRED_EXTRACTIONS
        if not any(w.eval_set_id == required.eval_set_id for w in measured)
    ]
    if outstanding:
        lines += [
            "## Outstanding measurement",
            "",
            "These extractions are hard dependencies of the submission. Until "
            "they land, the sections below are structurally complete and "
            "empirically unbacked.",
            "",
            "| eval set | needed for | blocks |",
            "|---|---|---|",
        ]
        for required in outstanding:
            lines.append(
                f"| `{required.eval_set_id}` | {required.needed_for} | {required.blocks} |"
            )
        lines.append("")

    # -- the matrix ----------------------------------------------------------- #
    lines += [
        "## Warrant matrix",
        "",
        "Cells on synthetic envelopes show their status but not their numbers.",
        "",
        matrix.render(mask_synthetic=True),
        "",
    ]

    # -- measured results ------------------------------------------------------ #
    lines += ["## Measured results", ""]
    if not measured:
        lines += [
            "**No measured results yet.** Every populated cell in the matrix is a "
            "synthetic fixture. This section stays empty until a real extraction "
            "lands, and that is the honest state rather than an omission.",
            "",
        ]
    else:
        lines += [
            "| detector | envelope | status | AUROC | recall | precision | "
            "flag rate | base rate | lift | n |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for warrant in measured:
            metrics = warrant.metrics
            lift = "n/a"
            if metrics.recall is not None and metrics.flag_rate.value > 0:
                lift_metric = metrics.lift
                ceiling = metrics.lift_ceiling
                lift = lift_metric.render(3)
                if ceiling is not None:
                    lift += (
                        f" — {metrics.lift_fraction_of_ceiling:.0%} of the "
                        f"{ceiling:.2f} ceiling at base rate {metrics.base_rate:.3f}"
                    )
            # Flag rate and base rate travel with recall, always. Recall moves
            # with the budget spent, so a row showing recall rising without the
            # flag rate beside it reads as a detector improving when the
            # detector may not have changed at all: last_token's recall goes
            # 0.079 -> 0.126 across the envelope shift purely because the frozen
            # threshold flags 25 items on one and 39 on the other, and its lift
            # -- the budget-normalised quantity -- is flat (DECISIONS 067).
            lines.append(
                f"| `{warrant.detector_id}` | `{warrant.eval_set_id}` | "
                f"{warrant.status.value} | {render_metric(metrics.auroc, digits=3)} | "
                f"{render_metric(metrics.recall, digits=3)} | "
                f"{render_metric(metrics.precision, digits=3)} | "
                f"{render_metric(metrics.flag_rate, digits=4)} | "
                f"{metrics.base_rate:.4f} | {lift} | "
                f"{warrant.n_test} |"
            )
        lines.append("")

    lines += _detection_sensitivity_section(measured)

    for heading, body in extra_sections:
        lines += [f"## {heading}", "", body, ""]

    lines += [
        "## Provenance",
        "",
        f"- config hash `{provenance.get('config_hash')}`",
        f"- git commit `{provenance.get('git_commit')}`",
        f"- dirty tree: `{provenance.get('dirty')}`",
        f"- seed `{provenance.get('seed')}`",
        f"- generated `{provenance.get('timestamp_utc')}`",
        "",
    ]
    return "\n".join(lines)


def _detection_sensitivity_section(measured) -> list:
    """What the calibration claims can and cannot detect at the n actually run.

    A warrant asserts a budget as well as a ranking, and the budget half is
    only as good as the sample behind it. Stating the boundary turns "how
    sensitive is your drift detection?" from a shrug into a number and a limit
    -- the same discipline as the price list: what can be proved, and what it
    would cost to prove more.
    """
    claims = [
        (w, w.calibration) for w in measured if getattr(w, "calibration", None)
    ]
    if not claims:
        return []

    rows, tolerances, shortfalls = [], set(), []
    for warrant, claim in claims:
        realised = claim.realised
        n = realised.n if realised is not None else None
        tolerances.add(claim.tolerance)
        if claim.n_to_detect and n and claim.n_to_detect > n:
            shortfalls.append((n, claim.n_to_detect))
        rows.append(
            "| `%s` | `%s` | %s | %s | %s |"
            % (
                warrant.detector_id,
                warrant.eval_set_id,
                claim.status.value,
                "yes" if claim.underpowered else "no",
                "n=%s, needs n>=%s" % (n, claim.n_to_detect)
                if claim.n_to_detect
                else "n/a",
            )
        )

    tolerance = sorted(tolerances)[0] if len(tolerances) == 1 else None
    lines = [
        "## Detection sensitivity — a declared limitation",
        "",
    ]
    if tolerance is not None and shortfalls:
        worst_n = min(n for n, _ in shortfalls)
        needed = max(need for _, need in shortfalls)
        lines += [
            "**Calibration drift is detectable at a %.0f%% deviation from the "
            "declared flag-rate budget, and is not detectable at %.0f%%, on the "
            "evidence here.**"
            % (tolerance * 100, tolerance * 100 / 2.5),
            "",
            "Separating a %.0f%% deviation from the budget needs **n >= %d**. "
            "These envelopes were measured at **n = %d**. So every budget claim "
            "below is either refused outright or unresolved -- the interval "
            "extends past the band, and this sample could not have narrowed it."
            % (tolerance * 100, needed, worst_n),
            "",
            "This is a limitation of sample size, not of the method: the "
            "ranking claims on the same warrants are supported, because AUROC "
            "intervals at this n are tight enough to separate the detectors "
            "from each other. Closing it needs more test items, not more code.",
            "",
        ]
    lines += [
        "| detector | envelope | calibration | unresolved | power |",
        "|---|---|---|---|---|",
    ] + rows + [""]
    return lines
