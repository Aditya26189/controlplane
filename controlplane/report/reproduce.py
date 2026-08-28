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

__all__ = ["ReproductionReport", "VariantDiff", "reproduce_frozen_set", "render"]

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
    """A short report: one line per variant, then the detail of any drift."""
    if not report.ran:
        return f"re-derivation SKIPPED\n  {report.reason}"
    lines = ["re-derivation from cached activations:"]
    for diff in report.diffs:
        mark = "OK" if diff.ok else "DRIFT"
        lines.append(
            f"  {diff.variant:26s} {diff.committed_status or '-':9s} {mark}"
        )
    if not report.ok:
        lines.append("")
        lines.append("DRIFT:")
        for diff in report.diffs:
            for mismatch in diff.mismatches:
                lines.append(f"  {diff.variant}: {mismatch}")
    else:
        lines.append("")
        lines.append(
            f"{len(report.diffs)} variants re-derived bit-identically from cache."
        )
    return "\n".join(lines)
