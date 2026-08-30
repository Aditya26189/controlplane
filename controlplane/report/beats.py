"""Assemble the demo's beats from committed artifacts. Reads; decides nothing.

``CLAUDE.md``: no logic in the demo runner. Every number a beat carries is
read out of ``results/`` and cited with the file and field it came from, so a
viewer can open the artifact and find it. A beat whose artifact is missing
reports itself absent rather than rendering a blank -- a demo that silently
skips a beat is a demo that shows only what happened to work.

The beats answer the problem statement directly:

===============================  =============================================
problem statement                beat
===============================  =============================================
different use cases, different   2 -- three profiles, three actions, one input
risk tolerance and latency
consume a model via API,         3 -- the tier ladder: activations, logprobs,
limited internal access               text-only, each separately warranted
hallucination and privacy        4 -- the dual-labelled pilot, and PII measured
overlap                               out of sample
no reliable real-time ground     5 -- envelope drift and revocation, which
truth                                 needs no ground truth to fire
over- and under-flagging         1 -- the refusal, and what it costs to fix
audit trail behind every         6 -- hash-chained certificates
decision
===============================  =============================================
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

__all__ = ["Beat", "assemble_beats"]


@dataclasses.dataclass(frozen=True)
class Beat:
    """One demo beat, with the artifact every number in it came from.

    Args:
        number: Running order.
        title: What the beat shows.
        answers: The problem-statement complexity it addresses.
        rows: ``(label, value)`` pairs, already formatted.
        artifacts: Files a viewer can open to check the rows.
        note: The caveat that must be said aloud with it, if any.
        missing: Why the beat could not be assembled, if it could not.
    """

    number: int
    title: str
    answers: str
    rows: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[str, ...] = ()
    note: Optional[str] = None
    missing: Optional[str] = None

    def to_payload(self) -> dict:
        return dataclasses.asdict(self)


def _load(root: Path, name: str) -> Optional[dict]:
    path = root / "results" / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fmt_interval(metric: Optional[dict]) -> str:
    """A rate with its interval and n, or an honest absence."""
    if not metric:
        return "absent"
    low, high, n = metric.get("ci_low"), metric.get("ci_high"), metric.get("n")
    value = metric.get("value")
    if low is None or high is None:
        return f"{value:.4f} (n={n})"
    return f"{value:.4f} [{low:.4f}, {high:.4f}] n={n}"


def _beat_refusal(root: Path) -> Beat:
    """What the system refuses to certify, and the price of fixing it."""
    from ..validation.warrant_stats import min_n_for

    rows = [
        ("customer-support FPR budget", "0.015"),
        ("clean negatives for ONE profile", f"{min_n_for(0.015, 0.05)}"),
        ("across THREE profiles (Bonferroni)", f"{min_n_for(0.015, 0.05 / 3)}"),
        ("convention", "one-sided 95%, per DECISIONS 110"),
    ]
    return Beat(
        number=1,
        title="The refusal, and what it costs to lift",
        answers="over-flagging vs under-flagging is tuned, not solved; "
        "reporting to a skeptical stakeholder",
        rows=tuple(rows),
        artifacts=("controlplane/validation/warrant_stats.py", "DECISIONS.md 110"),
        note=(
            "These are cluster-uncorrected. Inflating them needs a MEASURED ICC "
            "on the target set, which does not exist -- so the number is not "
            "inflated by an assumed factor (DECISIONS 113)."
        ),
    )


def _beat_profiles(root: Path) -> Beat:
    """Three profiles, three actions, one input."""
    payload = _load(root, "policy-triviaqa-2400-t960.json")
    if not payload or "comparison" not in payload:
        return Beat(
            2, "Three profiles, three actions, one input",
            "different use cases carry different risk tolerance and latency",
            missing="results/policy-triviaqa-2400-t960.json absent; run scripts/07_policy.py",
        )
    rows = []
    for row in payload["comparison"]["rows"]:
        rows.append((
            f"{row['profile']} -> {row['action']}",
            f"recall {row['recall']:.4f} "
            f"[{row['recall_ci'][0]:.4f}, {row['recall_ci'][1]:.4f}] "
            f"| flag rate {row['measured_flag_rate']:.4f} | {row['rule_id']}",
        ))
    score = payload["comparison"]["request"]["detector"]["score"]
    rows.insert(0, ("one input, one score", f"{score:.4f}"))
    return Beat(
        number=2,
        title="Three profiles, three actions, one input",
        answers="different use cases carry different risk tolerance and latency; "
        "tiered responses; a configurable policy layer",
        rows=tuple(rows),
        artifacts=("results/policy-triviaqa-2400-t960.json", "policies/"),
        note=(
            "The score is identical in all three rows. The ACTION differs "
            "because the policy differs -- not because the detector changed."
        ),
    )


def _beat_tiers(root: Path) -> Beat:
    """The access-tier ladder: what survives when you cannot see inside."""
    payload = _load(root, "tier_ladder.json")
    if not payload:
        return Beat(
            3, "The tier ladder", "enterprises consume a model via API",
            missing="results/tier_ladder.json absent",
        )
    rows = []
    for name, rung in payload["rungs"].items():
        auroc = rung.get("metrics", {}).get("auroc")
        rows.append((
            f"{rung.get('access_tier', '?')} / {name}",
            f"AUROC {_fmt_interval(auroc)} -- warrant {rung.get('warrant_status')}",
        ))
    return Beat(
        number=3,
        title="The tier ladder -- what survives without model internals",
        answers="enterprises consume a foundation model via API, limiting how "
        "deeply a checker can inspect internals",
        rows=tuple(rows),
        artifacts=("results/tier_ladder.json", "results/tier_ladder.png"),
        note=(
            "Each tier is warranted SEPARATELY. A tier that cannot clear the "
            "issuance bar is refused rather than reported with a caveat."
        ),
    )


def _beat_overlap(root: Path) -> Beat:
    """Hallucination and privacy in the same interaction, and PII out of sample."""
    rows: list[tuple[str, str]] = []
    pilot = _load(root, "pilot_run.json")
    if pilot:
        gen = pilot.get("generation", {})
        rows.append((
            "banking pilot, dual-labelled",
            f"{gen.get('n_items')} items over {gen.get('n_questions')} questions; "
            f"correctness MEASURED, identifier presence AUTHORED",
        ))
        band = pilot.get("acceptance_band", {})
        rows.append((
            "acceptance band (pre-registered)",
            f"{band.get('observed')} of 12 wrong, band "
            f"[{band.get('low')}, {band.get('high')}] -- "
            f"{'IN BAND' if band.get('in_band') else 'OUT OF BAND'}",
        ))
    detectors = _load(root, "detectors.json")
    if detectors:
        for run in detectors.get("runs", []):
            recall = (run.get("metrics") or {}).get("recall")
            status = (run.get("warrant") or {}).get("status")
            det = run.get("detector_id", "")
            if "presidio" in det or "pii-reference" in det:
                rows.append((
                    f"{det} on {run.get('eval_set_id')}",
                    f"recall {_fmt_interval(recall)} -- {status}",
                ))
    if not rows:
        return Beat(
            4, "Hallucination and privacy overlap", "the two risks co-occur",
            missing="neither results/pilot_run.json nor results/detectors.json present",
        )
    return Beat(
        number=4,
        title="Hallucination and privacy in the same interaction",
        answers="a fabricated detail about a person is simultaneously a "
        "hallucination and a privacy concern",
        rows=tuple(rows[:8]),
        artifacts=("results/pilot_run.json", "results/detectors.json"),
        note=(
            "The pilot's privacy axis is identifier-presence-in-PROMPT, which "
            "is not the output-side quantity a production path measures "
            "(DECISIONS 104). Declared, not glossed."
        ),
    )


def _beat_revocation(root: Path) -> Beat:
    """Drift and revocation -- the part that needs no ground truth."""
    matrix = _load(root, "warrant_matrix.json")
    rows: list[tuple[str, str]] = []
    if matrix:
        summary = matrix.get("matrix", {}).get("summary", {})
        total = sum(summary.values()) if summary else 0
        # UNVALIDATED first and by name, because it is the modal state and the
        # one a demo is tempted to hide. CLAUDE.md invariant 2: it must never
        # collapse into VALID or REFUSED.
        for state in ("VALID", "REFUSED", "UNVALIDATED", "REVOKED", "STALE"):
            if state in summary:
                rows.append((f"cells {state}", f"{summary[state]} of {total}"))
        rows.append((
            "why UNVALIDATED dominates",
            "a warrant is keyed by (detector, operating point, eval set). Most "
            "pairs were never tested here, and saying so is the point.",
        ))
    rows.append((
        "revocation trigger",
        "anytime-valid; P(false revoke) <= alpha over the whole deployment",
    ))
    rows.append((
        "estimand",
        "the profile's declared max_fpr_hard_negatives (DECISIONS 112)",
    ))
    return Beat(
        number=5,
        title="Drift, revocation, and why it needs no ground truth",
        answers="there is often no reliable real-time ground truth; regulatory "
        "expectations evolve, so hard-coded rules age",
        rows=tuple(rows),
        artifacts=(
            "results/warrant_matrix.json",
            "controlplane/drift/",
            "controlplane/validation/warrant_stats.py",
        ),
        note=(
            "Revocation fires on the INPUT DISTRIBUTION moving, which is "
            "observable without labels. The false-revocation bound was "
            "validated on i.i.d. negatives and is NOT validated under session "
            "correlation -- see LIMITATIONS."
        ),
    )


def _beat_audit(root: Path) -> Beat:
    """The audit trail, and the thing that makes it more than a log."""
    return Beat(
        number=6,
        title="A hash-chained decision record",
        answers="a clear audit trail behind every decision",
        rows=(
            ("per decision", "a certificate naming detector, operating point, "
             "envelope, warrant status and expiry"),
            ("chain", "each record hashes its predecessor; a rewritten history "
             "fails verification"),
            ("refusal", "has no override -- no flag, no env var, no admin bypass"),
        ),
        artifacts=("controlplane/store/ledger.py", "controlplane/model/certificate.py"),
        note=None,
    )


def assemble_beats(root: Path) -> tuple[Beat, ...]:
    """Every beat, in running order, from whatever artifacts exist.

    Args:
        root: The repository root.

    Returns:
        Six :class:`Beat` records. A beat whose artifact is missing carries
        ``missing`` rather than empty rows, so the renderer can say so.
    """
    return (
        _beat_refusal(root),
        _beat_profiles(root),
        _beat_tiers(root),
        _beat_overlap(root),
        _beat_revocation(root),
        _beat_audit(root),
    )
