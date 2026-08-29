"""Frozen per-item scores — the evidence a clean clone can actually check.

Block E, E.8 follow-up. ``make verify``'s second check re-derives the committed
metrics rather than trusting them, and until now it could only do that from the
extraction caches. Those are ~100 MB, gitignored, and therefore absent from
exactly the clone a judge has, so the substantive half of verification reported
SKIPPED for the one person it was written for.

The fix is not to ship activations. **Every number in the claim table is a
function of four arrays** — labels, scores, question ids and a threshold —
because every path in this codebase funnels through one
:func:`~controlplane.validation.metrics_builder.build_warrant_metrics` call
(``DECISIONS.md`` 048 made sure of that). Those arrays are kilobytes.

So they are frozen here, committed, and the verifier recomputes the metric
blocks from them with the same estimator, the same bootstrap count and the same
seed.

**What this tier proves and what it does not.** It proves the committed metrics
are what the metric builder produces from the recorded scores: every interval,
every rate, every refusal threshold. It does **not** prove the scores came from
the model and probe the artifact names — for that you need the activations, and
that remains the deeper tier, run when the cache is present. The two are
reported separately and neither is described as the other.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..config import Config
from ..model import WarrantMetrics, to_jsonable
from ..model.serde import content_hash
from .metrics_builder import build_warrant_metrics

__all__ = [
    "Scoring",
    "ScoreSet",
    "ScoreTarget",
    "load_score_set",
    "metrics_for_target",
    "write_score_set",
]


@dataclasses.dataclass(frozen=True)
class Scoring:
    """The arrays one validation scored, carried in memory off a run.

    Deliberately **not** written into a ``results/validation-*.json`` payload:
    those artifacts are the warrant record and adding a 600-element array to
    each would bury the thing a reader opens them for. Frozen separately, under
    ``results/scores/``, by ``scripts/10_freeze_scores.py``.
    """

    labels: np.ndarray
    scores: np.ndarray
    question_ids: np.ndarray
    threshold: float
    is_hard_negative_set: bool = False


@dataclasses.dataclass(frozen=True)
class ScoreTarget:
    """One metrics block this score set reproduces.

    A single score vector can back several blocks: the three policy profiles
    are three thresholds on one scoring, which is the whole point of calling
    them three points on one curve. Each target names its own threshold and the
    artifact field it must reproduce.
    """

    artifact: str
    metrics_path: str
    threshold: float
    operating_point_id: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ScoreSet:
    """Frozen scores for one (detector, envelope) scoring, plus what it backs."""

    score_set_id: str
    detector_id: str
    variant: str
    eval_set_id: str
    envelope_id: str
    labels: tuple[int, ...]
    scores: tuple[float, ...]
    question_ids: tuple[str, ...]
    targets: tuple[ScoreTarget, ...]
    is_hard_negative_set: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        n = len(self.labels)
        if not (len(self.scores) == len(self.question_ids) == n):
            raise ValueError(
                f"{self.score_set_id}: labels, scores and question_ids must be "
                f"the same length; got {n}, {len(self.scores)}, "
                f"{len(self.question_ids)}"
            )
        if n == 0:
            raise ValueError(f"{self.score_set_id}: an empty score set proves nothing")
        if not self.targets:
            raise ValueError(
                f"{self.score_set_id}: a score set with no target reproduces "
                "nothing. Name the artifact and metrics block it backs."
            )

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def base_rate(self) -> float:
        return float(np.mean(self.labels))

    @property
    def content_hash(self) -> str:
        """Identity of the arrays, so an edited score file is detectable.

        Covers the scores, labels and groups and nothing else -- not the
        targets, which are bookkeeping about what the arrays are compared
        against and may legitimately grow.
        """
        return content_hash(
            {
                "score_set_id": self.score_set_id,
                "detector_id": self.detector_id,
                "variant": self.variant,
                "eval_set_id": self.eval_set_id,
                "envelope_id": self.envelope_id,
                "labels": list(self.labels),
                # repr-stable rounding: JSON floats round-trip exactly at 17
                # significant digits, and the hash must survive a save/load.
                "scores": [float(s) for s in self.scores],
                "question_ids": list(self.question_ids),
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "score_set_id": self.score_set_id,
            "detector_id": self.detector_id,
            "variant": self.variant,
            "eval_set_id": self.eval_set_id,
            "envelope_id": self.envelope_id,
            "is_hard_negative_set": self.is_hard_negative_set,
            "n": self.n,
            "base_rate": self.base_rate,
            "content_hash": self.content_hash,
            "note": self.note,
            "targets": [dataclasses.asdict(t) for t in self.targets],
            "question_ids": list(self.question_ids),
            "labels": list(self.labels),
            "scores": list(self.scores),
        }


def from_scoring(
    score_set_id: str,
    scoring: Scoring,
    *,
    detector_id: str,
    variant: str,
    eval_set_id: str,
    envelope_id: str,
    targets: Sequence[ScoreTarget],
    note: str = "",
) -> ScoreSet:
    """Freeze a :class:`Scoring` captured off a validation run."""
    return ScoreSet(
        score_set_id=score_set_id,
        detector_id=detector_id,
        variant=variant,
        eval_set_id=eval_set_id,
        envelope_id=envelope_id,
        labels=tuple(int(x) for x in np.asarray(scoring.labels).tolist()),
        scores=tuple(float(x) for x in np.asarray(scoring.scores).tolist()),
        question_ids=tuple(str(x) for x in np.asarray(scoring.question_ids).tolist()),
        targets=tuple(targets),
        is_hard_negative_set=scoring.is_hard_negative_set,
        note=note,
    )


def write_score_set(score_set: ScoreSet, path: Path, provenance: dict[str, Any]) -> None:
    """Write a score set to ``results/scores/`` with the usual provenance block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance, **score_set.to_payload()}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_score_set(path: Path) -> ScoreSet:
    """Read a score set, refusing one whose arrays no longer hash as recorded."""
    data = json.loads(path.read_text(encoding="utf-8"))
    score_set = ScoreSet(
        score_set_id=data["score_set_id"],
        detector_id=data["detector_id"],
        variant=data["variant"],
        eval_set_id=data["eval_set_id"],
        envelope_id=data["envelope_id"],
        labels=tuple(data["labels"]),
        scores=tuple(data["scores"]),
        question_ids=tuple(data["question_ids"]),
        targets=tuple(ScoreTarget(**t) for t in data["targets"]),
        is_hard_negative_set=data.get("is_hard_negative_set", False),
        note=data.get("note", ""),
    )
    recorded = data.get("content_hash")
    if recorded and recorded != score_set.content_hash:
        raise ValueError(
            f"{path.name} records content hash {recorded[:16]} but its arrays "
            f"hash to {score_set.content_hash[:16]}. The file was edited after "
            "it was frozen; re-freeze it rather than trusting it."
        )
    return score_set


def metrics_for_target(
    config: Config, score_set: ScoreSet, target: ScoreTarget
) -> WarrantMetrics:
    """Recompute one metrics block from frozen scores.

    Calls the same builder every runner calls, with the config's bootstrap
    count, coverage level and seed -- so a difference here is a difference in
    the numbers, never a difference in how they were computed.
    """
    return build_warrant_metrics(
        config,
        np.asarray(score_set.labels),
        np.asarray(score_set.scores, dtype=float),
        target.threshold,
        groups=np.asarray(score_set.question_ids),
        is_hard_negative_set=score_set.is_hard_negative_set,
    )


def metrics_as_payload(metrics: WarrantMetrics) -> dict[str, Any]:
    """The metrics block in the shape a ``results/`` artifact stores it."""
    return to_jsonable(metrics)
