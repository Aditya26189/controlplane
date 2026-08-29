"""Frozen scores, and the tier of verification they make possible.

Block E, E.8 follow-up. The committed score sets are what lets a clean clone
run the substantive half of ``make verify`` instead of skipping it, so they
have to be checkable themselves: an edited score file must be caught, a target
naming a block that is not there must be caught, and a score set that no longer
reproduces its artifact must fail rather than being quietly re-derived.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlplane.config import Config
from controlplane.report.reproduce import reproduce_from_scores
from controlplane.validation.scores import (
    ScoreSet,
    ScoreTarget,
    load_score_set,
    metrics_for_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES = PROJECT_ROOT / "results" / "scores"


def _any_score_file() -> Path:
    files = sorted(SCORES.glob("*.json"))
    assert files, "no frozen score sets committed"
    return files[0]


# --------------------------------------------------------------------------- #
# The committed sets
# --------------------------------------------------------------------------- #


def test_score_sets_are_committed() -> None:
    """Without these a fresh clone can only check the README against itself."""
    files = sorted(SCORES.glob("*.json"))
    assert len(files) >= 12, (
        f"only {len(files)} score sets in results/scores/. They are the evidence "
        "a clean clone verifies against; regenerate with "
        "scripts/10_freeze_scores.py where the extraction caches live."
    )


def test_the_score_sets_stay_small_enough_to_commit() -> None:
    """The whole argument for shipping scores instead of activations.

    If this ever fails, something is being frozen that should not be -- most
    likely a feature matrix rather than a score vector.
    """
    total = sum(p.stat().st_size for p in SCORES.glob("*.json"))
    assert total < 2_000_000, (
        f"results/scores/ is {total / 1e6:.1f} MB. Scores are meant to be "
        "kilobytes; anything this large is not a score vector."
    )


def test_every_score_set_names_at_least_one_target() -> None:
    for path in sorted(SCORES.glob("*.json")):
        score_set = load_score_set(path)
        assert score_set.targets, f"{path.name} reproduces nothing"
        for target in score_set.targets:
            assert (PROJECT_ROOT / target.artifact).is_file(), (
                f"{path.name} targets {target.artifact}, which does not exist"
            )


def test_every_committed_metrics_block_recomputes(config: Config) -> None:
    """The headline: the committed numbers follow from the committed scores.

    This is what ``make verify`` check 2 runs, and what a judge without the
    extraction caches actually gets to check.
    """
    report = reproduce_from_scores(PROJECT_ROOT, config)
    assert report.ran, report.reason
    failed = [d for d in report.diffs if not d.ok]
    assert not failed, "\n".join(
        f"{d.variant}: {'; '.join(d.mismatches)}" for d in failed
    )


# --------------------------------------------------------------------------- #
# Tampering
# --------------------------------------------------------------------------- #


def test_an_edited_score_file_is_refused_on_load(tmp_path: Path) -> None:
    """The arrays are content-hashed, so an edit cannot pass as evidence."""
    data = json.loads(_any_score_file().read_text(encoding="utf-8"))
    data["scores"][0] = data["scores"][0] + 0.5
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_score_set(path)


def test_a_tampered_score_set_fails_verification(tmp_path: Path, config: Config) -> None:
    """End to end: an edited score set must fail check 2, not be re-derived.

    Built as a whole fake results tree so the real one is untouched.
    """
    source = SCORES / "text-pii-reference-hinglish-pii-200.json"
    if not source.is_file():
        pytest.skip("the reference score set is not committed")
    data = json.loads(source.read_text(encoding="utf-8"))

    root = tmp_path / "repo"
    (root / "results" / "scores").mkdir(parents=True)
    # Flip a label, then re-hash so it loads: the point is that the *metrics*
    # no longer follow, not that the file is malformed.
    data["labels"][0] = 1 - data["labels"][0]
    data.pop("content_hash", None)
    (root / "results" / "scores" / source.name).write_text(
        json.dumps(data), encoding="utf-8"
    )
    for target in data["targets"]:
        dest = root / target["artifact"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            (PROJECT_ROOT / target["artifact"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    report = reproduce_from_scores(root, config)
    assert report.ran
    assert not report.ok, (
        "a flipped label did not change any metric, which means the check is "
        "not comparing what it claims to compare"
    )


def test_an_absent_scores_directory_is_a_defect_not_a_skip(tmp_path: Path) -> None:
    """Unlike the activation cache, missing scores are a repository defect.

    The caches are legitimately absent on a fresh clone. The scores are
    committed evidence, so their absence must not read as "nothing to check".
    """
    report = reproduce_from_scores(tmp_path)
    assert report.ran is False
    assert "not an optional input" in report.reason or "does not exist" in report.reason


# --------------------------------------------------------------------------- #
# The dataclass
# --------------------------------------------------------------------------- #


def test_a_score_set_with_mismatched_array_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        ScoreSet(
            score_set_id="x", detector_id="d", variant="text",
            eval_set_id="e", envelope_id="sha256:0",
            labels=(0, 1), scores=(0.1,), question_ids=("q1", "q2"),
            targets=(ScoreTarget("results/x.json", "metrics", 0.5),),
        )


def test_a_score_set_with_no_target_is_refused() -> None:
    """A score set that reproduces nothing is decoration."""
    with pytest.raises(ValueError, match="reproduces nothing|no target"):
        ScoreSet(
            score_set_id="x", detector_id="d", variant="text",
            eval_set_id="e", envelope_id="sha256:0",
            labels=(0, 1), scores=(0.1, 0.9), question_ids=("q1", "q2"),
            targets=(),
        )


def test_recomputation_uses_the_configured_estimator(config: Config) -> None:
    """The interval comes from config, not from a constant in this module."""
    score_set = load_score_set(_any_score_file())
    metrics = metrics_for_target(config, score_set, score_set.targets[0])
    for metric in metrics.all_metrics():
        if metric.ci_level is not None:
            assert metric.ci_level == config.validation.ci
