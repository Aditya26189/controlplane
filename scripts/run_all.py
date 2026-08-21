"""Run the whole pipeline: 01 extract -> 02 probe -> 03 economics -> 04 latency -> 05 report.

Parses arguments, shells out to the stage scripts, reports timings. No logic
lives here.

Each stage is a separate process reading the previous stage's output from disk,
so any stage can be re-run alone without repeating the expensive extraction::

    python scripts/run_all.py --config config.yaml            # full run
    python scripts/run_all.py --config config.yaml --smoke    # n=100, for CI
    python scripts/run_all.py --from 02                       # reuse activations
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, setup_logging  # noqa: E402

LOGGER = logging.getLogger("run_all")

SMOKE_N_EXAMPLES = 100
SMOKE_BOOTSTRAP_SAMPLES = 200
SMOKE_SUBDIR = "smoke"

STAGES = [
    ("01", "01_extract.py", "extract activations and labels"),
    ("02", "02_train_probe.py", "sweep, select, score test once"),
    ("03", "03_economics.py", "three-policy comparison and lift"),
    ("04", "04_latency.py", "probe cost vs generation cost"),
    ("05", "05_report.py", "render RESULTS.md, README.md, plots"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"fast end-to-end check at n_examples={SMOKE_N_EXAMPLES} and "
            f"{SMOKE_BOOTSTRAP_SAMPLES} bootstrap resamples"
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        default="01",
        choices=[stage for stage, _, _ in STAGES],
        help="start from this stage, reusing earlier stages' artifacts",
    )
    parser.add_argument(
        "--no-readme", action="store_true", help="do not rewrite README.md"
    )
    return parser.parse_args()


def build_smoke_config(config_path: Path, destination: Path) -> Path:
    """Write a smoke variant of the config, writing into ``results/smoke/``.

    A derived file rather than a second committed config, so the two cannot
    drift apart: everything except the sample size, the bootstrap count and the
    output paths is copied from the real config verbatim. The smoke config has a
    different config hash, which is correct -- it is a different run.

    **Output paths are redirected to a subdirectory.** A smoke run that wrote to
    ``results/`` would overwrite the artifacts of a real run, and the real run
    costs a GPU hour. The same reasoning applies to README.md, which the caller
    redirects rather than letting stage 05 rewrite the published one with n=100
    numbers.

    The base-rate sanity band is deliberately *not* relaxed: at n=100 with the
    real model it is still a meaningful check, and a smoke mode that disables
    its own safety checks tests nothing worth testing.
    """
    import yaml

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["data"]["n_examples"] = SMOKE_N_EXAMPLES
    raw["evaluation"]["bootstrap_samples"] = SMOKE_BOOTSTRAP_SAMPLES

    smoke_dir = Path(raw["paths"]["results_dir"]) / SMOKE_SUBDIR
    raw["paths"] = {
        "results_dir": smoke_dir.as_posix(),
        "activations": (smoke_dir / "activations.npz").as_posix(),
        "labels": (smoke_dir / "labels.parquet").as_posix(),
        "splits": (smoke_dir / "splits.parquet").as_posix(),
    }

    # A clean checkout has no results/ directory: git does not track empty
    # directories, so this is the first thing that would fail on the clone-and-
    # run path the Stage 7 gate exercises.
    destination.parent.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return destination


def run_stage(script: str, config_path: Path, extra: list[str]) -> float:
    """Run one stage script as a subprocess, returning its wall-clock seconds."""
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / script),
        "--config",
        str(config_path),
        *extra,
    ]
    LOGGER.info("$ %s", " ".join(command))
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(
            f"stage {script} failed with exit code {result.returncode} after "
            f"{elapsed:.1f}s; the pipeline stops here rather than reporting "
            "numbers from a partial run"
        )
    return elapsed


def main() -> int:
    args = parse_args()
    setup_logging()

    config_path = Path(args.config)
    smoke_readme: Path | None = None
    if args.smoke:
        config_path = build_smoke_config(
            config_path, REPO_ROOT / "results" / SMOKE_SUBDIR / "config.smoke.yaml"
        )
        smoke_readme = REPO_ROOT / "results" / SMOKE_SUBDIR / "README.md"
        LOGGER.info(
            "smoke mode: config %s, artifacts under results/%s/ so a real run's "
            "results and README are left untouched",
            config_path,
            SMOKE_SUBDIR,
        )

    config = load_config(config_path)
    LOGGER.info(
        "pipeline start | config hash %s | seed %d | n_examples %d | model %s",
        config.config_hash,
        config.seed,
        config.data.n_examples,
        config.model.name,
    )

    start_index = [stage for stage, _, _ in STAGES].index(args.from_stage)
    timings: list[tuple[str, float]] = []
    total_started = time.perf_counter()

    for stage, script, description in STAGES[start_index:]:
        LOGGER.info("=== stage %s: %s ===", stage, description)
        extra: list[str] = []
        if stage == "05":
            if args.no_readme:
                extra = ["--no-readme"]
            elif smoke_readme is not None:
                # Exercise the README renderer without publishing n=100 numbers.
                extra = ["--readme", str(smoke_readme)]
        timings.append((stage, run_stage(script, config_path, extra)))

    total = time.perf_counter() - total_started
    LOGGER.info("=== pipeline complete in %.1f s ===", total)
    for stage, seconds in timings:
        LOGGER.info("  stage %s: %6.1f s (%4.1f%%)", stage, seconds, 100 * seconds / total)
    LOGGER.info("results are in %s/", config.paths.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
