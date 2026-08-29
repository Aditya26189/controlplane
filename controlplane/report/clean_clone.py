"""Clone this repository into a temporary directory and run the gates in it.

Block E, E.8. The Phase 10 gate, moved earlier because it is the check most
likely to fail in a way that takes hours to fix.

**What it actually catches.** Everything on a developer's machine works partly
by accident: an untracked file that was never committed, a cache the pipeline
quietly needs, a path that resolves only because of where the interpreter was
started. A fresh clone has *only what is tracked*, and running the gates inside
one is the only way to find out whether that is enough.

Lives under ``report`` rather than in the script because it writes an artifact
and because ``CLAUDE.md`` rules logic out of scripts. It is repo-level QA
rather than a measurement, and the artifact says so.

**On the verify tiers.** ``verify`` runs three checks and they do not stand or
fall together, so each gets its own row: a green exit code says *nothing
drifted*, not *everything was checked*. On a fresh clone the first two run --
the claim table against the artifacts, and every metrics block recomputed from
the frozen scores in ``results/scores/`` -- and the third skips, because the
extraction caches are ~100 MB and gitignored.

A skipped tier is recorded as skipped, never as a pass. So is a tier whose
outcome line is missing from the captured output: unknown is not success. A
gate reporting a check it could not run as green would be this project's own
failure mode, one level up.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["GateResult", "StepResult", "run_clean_clone_gate", "render"]


@dataclass
class StepResult:
    """One command run inside the clone."""

    name: str
    command: list[str]
    returncode: Optional[int]
    duration_seconds: float
    ok: bool
    skipped: bool = False
    detail: str = ""
    tail: str = ""


@dataclass
class GateResult:
    """The whole gate."""

    ok: bool
    commit: str
    clone_path: str
    tracked_files: int
    steps: list[StepResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "commit": self.commit,
            "clone_path": self.clone_path,
            "tracked_files": self.tracked_files,
            "steps": [asdict(s) for s in self.steps],
            "notes": self.notes,
        }


def _run(
    name: str,
    command: list[str],
    cwd: Path,
    timeout: int,
    env: Optional[dict] = None,
) -> StepResult:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or {**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return StepResult(
            name=name,
            command=command,
            returncode=None,
            duration_seconds=time.time() - started,
            ok=False,
            detail=f"timed out after {timeout}s",
        )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return StepResult(
        name=name,
        command=command,
        returncode=proc.returncode,
        duration_seconds=round(time.time() - started, 2),
        ok=proc.returncode == 0,
        detail="" if proc.returncode == 0 else f"exited {proc.returncode}",
        tail="\n".join(combined.strip().splitlines()[-25:]),
    )


#: How each verify tier announces its outcome. Matched against the captured
#: output so the gate reports what actually happened rather than inferring it
#: from an exit code, which only says "nothing drifted".
_TIERS = (
    ("claim table vs artifacts", "claims reproduce"),
    ("metrics recomputed from frozen scores", "recomputed from frozen scores"),
    ("scores re-derived from activations", "re-derived from cache"),
)


def _verify_tiers(tail: str) -> list[StepResult]:
    """One row per verify tier, saying whether it ran or was skipped.

    A tier is counted as having RUN only when its own success line is present.
    Absence is reported as skipped rather than assumed to have passed -- the
    captured tail is truncated to its last lines, so a tier whose line fell off
    the end is unknown, and unknown is not a pass.
    """
    rows: list[StepResult] = []
    for label, marker in _TIERS:
        ran = marker in tail
        rows.append(
            StepResult(
                name=f"verify: {label}",
                command=[],
                returncode=None,
                duration_seconds=0.0,
                ok=True,
                skipped=not ran,
                detail=(
                    ""
                    if ran
                    else (
                        "not reported in the captured output. On a fresh clone "
                        "the activation tier legitimately skips -- the caches "
                        "are ~100 MB and gitignored. Any other tier missing "
                        "here means the output was truncated, which is not a "
                        "pass either."
                    )
                ),
            )
        )
    return rows


def _remove_clone(path: Path, attempts: int = 3) -> str:
    """Delete the clone, and say so when it cannot rather than swallowing it.

    ``shutil.rmtree(ignore_errors=True)`` silently leaves the tree behind when
    Windows still holds a handle on a file -- git's pack files, most often,
    just after a subprocess exits. Two earlier gate runs each left 26 MB in the
    temp directory and reported nothing.

    Retries briefly, then returns what went wrong so the caller can put it in
    the report. A cleanup that reports success without completing is a small
    version of exactly what this gate exists to catch.

    Returns:
        An empty string on success, or a short description of the failure.
    """
    last = ""
    for attempt in range(attempts):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return ""
        last = f"still present after attempt {attempt + 1}"
        time.sleep(0.5)
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return "" if not path.exists() else last


def run_clean_clone_gate(
    root: Path,
    *,
    keep: bool = False,
    run_tests: bool = True,
    python: Optional[str] = None,
) -> GateResult:
    """Clone ``root``, then run smoke, test and verify inside the clone.

    Args:
        root: The repository to clone. Cloned from the local path, so the clone
            contains exactly what a ``git clone`` of the remote would once this
            branch is pushed: tracked files only, no gitignored caches.
        keep: Leave the clone on disk for inspection.
        run_tests: Run the full suite. Off makes the gate a two-minute check.
        python: Interpreter to use inside the clone. Defaults to this one, so
            the gate tests the *tree* rather than the environment. Point it at
            a venv built from requirements.lock.txt to test both.

    Returns:
        A result whose ``ok`` is False if any step failed. A step that could
        not run is recorded as skipped and does not count as a pass.
    """
    python = python or sys.executable
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root),
        capture_output=True, text=True,
    ).stdout.strip()

    clone = Path(tempfile.mkdtemp(prefix="controlplane-clean-clone-"))
    target = clone / "controlplane"
    result = GateResult(ok=True, commit=commit, clone_path=str(target), tracked_files=0)

    try:
        cloned = _run(
            "git clone",
            ["git", "clone", "--quiet", str(root), str(target)],
            cwd=clone,
            timeout=600,
        )
        result.steps.append(cloned)
        if not cloned.ok:
            result.ok = False
            return result

        listing = subprocess.run(
            ["git", "ls-files"], cwd=str(target), capture_output=True, text=True
        )
        result.tracked_files = len([x for x in listing.stdout.splitlines() if x])
        result.notes.append(
            f"{result.tracked_files} tracked files in the clone; no gitignored "
            "caches, databases or long-context eval sets came with it."
        )

        # The notes ref carries the correction to the withdrawn claim at
        # 67167ed. A judge is told to fetch it; the gate checks it is fetchable.
        notes = _run(
            "fetch notes",
            ["git", "fetch", "origin", "refs/notes/*:refs/notes/*"],
            cwd=target,
            timeout=300,
        )
        result.steps.append(notes)

        steps: list[tuple[str, list[str], int]] = [
            ("make smoke", [python, "scripts/smoke.py"], 600),
        ]
        if run_tests:
            steps.append(("make test", [python, "-m", "pytest", "tests/", "-q"], 3600))
        steps.append(("make verify", [python, "scripts/verify.py"], 1800))

        for name, command, timeout in steps:
            step = _run(name, command, cwd=target, timeout=timeout)
            result.steps.append(step)

        # verify runs three tiers and they do not stand or fall together. A
        # green exit code says "nothing drifted", not "everything was checked",
        # so each tier is recorded on its own row.
        #
        # The first version of this matched the literal string "SKIPPED"
        # anywhere in verify's output and filed ONE skip row saying the claim
        # table was all that ran. Once the score tier existed that was simply
        # false -- tier 2 reproduced 24 comparisons in the clone and the gate
        # reported it as not having run. Understating what was verified is the
        # safe direction to be wrong in, and it is still wrong, in a gate whose
        # whole job is saying accurately what happened.
        verify = next((s for s in result.steps if s.name == "make verify"), None)
        if verify is not None:
            result.steps.extend(_verify_tiers(verify.tail))
            ran = [s.name for s in result.steps if s.name.startswith("verify:") and not s.skipped]
            skipped = [s.name for s in result.steps if s.name.startswith("verify:") and s.skipped]
            result.notes.append(
                "verify tiers that ran in the clone: "
                + (", ".join(n.split(": ", 1)[1] for n in ran) or "none")
                + ". Skipped: "
                + (", ".join(n.split(": ", 1)[1] for n in skipped) or "none")
                + "."
            )

        result.ok = all(s.ok for s in result.steps)
        return result
    finally:
        if keep:
            result.notes.append(f"clone kept at {target}")
        else:
            leftover = _remove_clone(clone)
            if leftover:
                result.notes.append(
                    f"could not fully remove the clone at {clone}: {leftover}. "
                    "Not a gate failure -- the checks above all ran -- but it "
                    "leaves ~26 MB behind, and earlier runs did so silently."
                )


def render(result: GateResult) -> str:
    lines = [
        "clean-clone gate",
        f"  commit        {result.commit[:12]}",
        f"  tracked files {result.tracked_files}",
        "",
    ]
    for step in result.steps:
        mark = "SKIP" if step.skipped else ("OK" if step.ok else "FAIL")
        lines.append(f"  {mark:5s} {step.name:34s} {step.duration_seconds:7.1f}s")
        if step.detail:
            lines.append(f"        {step.detail}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    lines.append("")
    lines.append("CLEAN CLONE GATE: " + ("PASSED" if result.ok else "FAILED"))
    if not result.ok:
        for step in result.steps:
            if not step.ok:
                lines.append("")
                lines.append(f"--- {step.name} ---")
                lines.append(step.tail)
    return "\n".join(lines)


def write_artifact(result: GateResult, path: Path, provenance: dict) -> None:
    """Write the gate's outcome to ``results/`` with the usual provenance block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance, **result.to_dict()}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
