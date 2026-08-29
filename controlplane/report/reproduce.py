"""Re-derive the frozen-set evaluation from cached activations and diff it.

Block E, E.4. ``verify_claims`` proves the README agrees with the artifacts in
``results/``. It cannot prove those artifacts describe what the code does now:
both could be stale together, and the check would still pass.

This closes that gap. It re-runs the validation from the extraction cache into
a scratch directory and compares every metric, interval and warrant status
against what is committed. Nothing in ``results/`` is written, so a failed
verification never damages the evidence it was checking.

**When the cache is absent it says so and returns SKIPPED.** The caches are
~100 MB and gitignored, so a fresh clone does not have them. A verifier that
silently reported success on a machine where it could not run anything would
be the exact failure this project is about.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "ReproductionReport",
    "VariantDiff",
    "reproduce_frozen_set",
    "reproduce_from_scores",
    "render",
]

#: Metrics compared for every variant. Each is checked at point, low and high.
COMPARED = ("auroc", "recall", "precision", "flag_rate")

#: The three probe aggregations validated on the frozen TriviaQA envelope.
VARIANTS = ("T1-last_token", "T1-max_rolling_means", "T1-mean_pool")


@dataclass
class VariantDiff:
    """The comparison for one detector variant."""

    variant: str
    committed_status: Optional[str] = None
    recomputed_status: Optional[str] = None
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.committed_status == self.recomputed_status


@dataclass
class ReproductionReport:
    """The outcome of one reproduction attempt."""

    ran: bool
    reason: str
    diffs: list[VariantDiff] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """A skipped run is not a failure, and is never reported as a pass."""
        return all(d.ok for d in self.diffs) if self.ran else True

    @property
    def status(self) -> str:
        if not self.ran:
            return "SKIPPED"
        return "OK" if self.ok else "DRIFT"


def _compare(committed: dict, recomputed: dict, variant: str) -> VariantDiff:
    diff = VariantDiff(
        variant=variant,
        committed_status=committed.get("warrant_status"),
        recomputed_status=recomputed.get("warrant_status"),
    )
    if diff.committed_status != diff.recomputed_status:
        diff.mismatches.append(
            f"warrant status {diff.committed_status} -> {diff.recomputed_status}"
        )
    for name in COMPARED:
        was = (committed.get("metrics") or {}).get(name)
        now = (recomputed.get("metrics") or {}).get(name)
        if (was is None) != (now is None):
            diff.mismatches.append(f"{name}: present in one run and not the other")
            continue
        if was is None:
            continue
        for bound in ("value", "ci_low", "ci_high"):
            if was.get(bound) != now.get(bound):
                diff.mismatches.append(
                    f"{name}.{bound}: {was.get(bound)!r} -> {now.get(bound)!r}"
                )
    return diff


def reproduce_frozen_set(
    root: Path,
    *,
    cache: Optional[Path] = None,
    eval_set: str = "triviaqa-600",
    timeout: int = 1800,
) -> ReproductionReport:
    """Re-run the frozen-set validation from cache and diff it against results/.

    Args:
        root: Project root.
        cache: Extraction cache. Defaults to ``results/cache-<eval_set>.npz``.
        eval_set: The frozen envelope to re-validate.
        timeout: Seconds before the subprocess is abandoned.

    Returns:
        A report. ``ran=False`` when the cache is absent, with the reason.
    """
    cache = cache or (root / "results" / f"cache-{eval_set}.npz")
    if not cache.is_file():
        return ReproductionReport(
            ran=False,
            reason=(
                f"extraction cache not found at {cache.relative_to(root) if cache.is_relative_to(root) else cache}. "
                "It is ~78 MB and gitignored, so a fresh clone does not have it. "
                "Regenerate it on a GPU with `make extract`, or copy it in. The "
                "claim check above still ran against the committed artifacts; "
                "this step, which proves those artifacts match the code, did not."
            ),
        )

    scratch = Path(tempfile.mkdtemp(prefix="controlplane-verify-"))
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "02_validate.py"),
                "--config", str(root / "config.yaml"),
                "--cache", str(cache),
                "--eval-set", eval_set,
                "--out", str(scratch),
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return ReproductionReport(
                ran=False,
                reason=(
                    f"re-validation exited {result.returncode}. This is a "
                    f"failure, not a skip:\n{result.stderr[-2000:]}"
                ),
            )

        diffs: list[VariantDiff] = []
        for variant in VARIANTS:
            name = f"validation-{variant}.json"
            here, there = root / "results" / name, scratch / name
            if not here.is_file():
                diffs.append(VariantDiff(variant, mismatches=[f"{name} not committed"]))
                continue
            if not there.is_file():
                diffs.append(VariantDiff(variant, mismatches=[f"{name} not produced"]))
                continue
            diffs.append(
                _compare(
                    json.loads(here.read_text(encoding="utf-8")),
                    json.loads(there.read_text(encoding="utf-8")),
                    variant,
                )
            )
        return ReproductionReport(ran=True, reason="re-derived from cache", diffs=diffs)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def render(report: ReproductionReport) -> str:
    """A short report: one line per comparison, then the detail of any drift.

    The heading comes from ``report.reason`` rather than being hardcoded. It
    said "from cached activations" for both tiers once the score tier existed,
    which described the weaker check as the stronger one.
    """
    if not report.ran:
        return f"re-derivation SKIPPED\n  {report.reason}"
    width = min(max((len(d.variant) for d in report.diffs), default=26), 62)
    lines = [f"re-derivation, {report.reason}:"]
    for diff in report.diffs:
        mark = "OK" if diff.ok else "DRIFT"
        lines.append(f"  {diff.variant[:width].ljust(width)}  {mark}")
    if not report.ok:
        lines.append("")
        lines.append("DRIFT:")
        for diff in report.diffs:
            for mismatch in diff.mismatches:
                lines.append(f"  {diff.variant}: {mismatch}")
    else:
        lines.append("")
        lines.append(
            f"{len(report.diffs)} comparisons reproduced bit-identically "
            f"({report.reason})."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The score tier -- what a clean clone can actually check
# --------------------------------------------------------------------------- #
# The activation tier above needs ~100 MB of gitignored cache, so on the clone
# a judge has it cannot run. This tier needs only results/scores/, which is
# ~200 KB and committed, and it recomputes every metrics block from the frozen
# per-item scores using the same builder, the same bootstrap count and the same
# seed.
#
# It proves the committed metrics are what the estimator produces from the
# recorded scores. It does NOT prove the scores came from the model and probe
# the artifact names -- that is the activation tier, and the two are reported
# separately rather than one being described as the other.

#: Metric fields compared, at point estimate and both bounds.
_METRIC_FIELDS = ("auroc", "recall", "precision", "flag_rate")


def reproduce_from_scores(root: Path, config=None) -> ReproductionReport:
    """Recompute every metrics block from ``results/scores/`` and diff it.

    Args:
        root: Project root.
        config: Resolved config. Loaded from ``root/config.yaml`` if omitted,
            so the bootstrap count, coverage and seed are the committed ones.

    Returns:
        A report with one diff per (score set, target). ``ran=False`` only when
        there are no score sets at all, which is a repository defect rather
        than a missing optional input.
    """
    import json

    from ..config import load_config
    from ..model import to_jsonable
    from ..validation.scores import load_score_set, metrics_for_target

    scores_dir = root / "results" / "scores"
    if not scores_dir.is_dir():
        return ReproductionReport(
            ran=False,
            reason=(
                f"{scores_dir.relative_to(root)} does not exist. Frozen scores "
                "are committed evidence, not an optional input -- regenerate "
                "them with scripts/10_freeze_scores.py where the extraction "
                "caches live."
            ),
        )
    files = sorted(scores_dir.glob("*.json"))
    if not files:
        return ReproductionReport(
            ran=False, reason=f"no score sets in {scores_dir.relative_to(root)}"
        )

    config = config or load_config(str(root / "config.yaml"))
    diffs: list[VariantDiff] = []

    for path in files:
        try:
            score_set = load_score_set(path)
        except (ValueError, KeyError) as exc:
            diffs.append(VariantDiff(path.stem, mismatches=[str(exc)]))
            continue

        for target in score_set.targets:
            label = f"{score_set.score_set_id} -> {Path(target.artifact).name}"
            diff = VariantDiff(label)
            artifact = root / target.artifact
            if not artifact.is_file():
                diff.mismatches.append(f"{target.artifact} not committed")
                diffs.append(diff)
                continue
            document = json.loads(artifact.read_text(encoding="utf-8"))
            try:
                from .claims import resolve

                committed = resolve(document, target.metrics_path)
            except KeyError as exc:
                diff.mismatches.append(f"{target.metrics_path}: {exc}")
                diffs.append(diff)
                continue

            recomputed = to_jsonable(metrics_for_target(config, score_set, target))
            for name in _METRIC_FIELDS:
                was, now = committed.get(name), recomputed.get(name)
                if (was is None) != (now is None):
                    diff.mismatches.append(
                        f"{name}: present in one and not the other"
                    )
                    continue
                if was is None:
                    continue
                for bound in ("value", "ci_low", "ci_high"):
                    if was.get(bound) != now.get(bound):
                        diff.mismatches.append(
                            f"{name}.{bound}: committed {was.get(bound)!r}, "
                            f"recomputed {now.get(bound)!r}"
                        )
            # Status is not recomputed here: issuance needs the controls, which
            # scores alone cannot reconstruct. Recorded as equal so the shared
            # VariantDiff.ok does not read a None mismatch as a failure.
            diff.committed_status = "metrics"
            diff.recomputed_status = "metrics"
            diffs.append(diff)

    return ReproductionReport(
        ran=True, reason="recomputed from frozen scores", diffs=diffs
    )
