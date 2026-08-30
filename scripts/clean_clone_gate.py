"""Clone the repository into a temporary directory and run the gates in it.

Block E, E.8. Thin wrapper: parses arguments, calls ``controlplane/``, writes
files. No logic (``CLAUDE.md``).

Run this before submission. It catches the class of failure that only exists on
the machine the work was done on: an untracked file the pipeline needs, a cache
nothing declares, a path that resolves only because of where the interpreter
started.

Usage:
    python scripts/clean_clone_gate.py                  # full, ~10 min
    python scripts/clean_clone_gate.py --no-tests       # ~2 min
    python scripts/clean_clone_gate.py --keep           # leave the clone behind
    python scripts/clean_clone_gate.py --python .venv/bin/python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import load_config, provenance, setup_logging  # noqa: E402
from controlplane.report.clean_clone import (  # noqa: E402
    render,
    run_clean_clone_gate,
    write_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results" / "clean_clone.json"),
        help="where the gate's outcome is recorded",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="skip the full suite inside the clone; smoke and verify still run",
    )
    parser.add_argument(
        "--keep", action="store_true", help="leave the clone on disk for inspection"
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "interpreter to use inside the clone. Defaults to this one, which "
            "tests the tree rather than the environment. Point it at a venv "
            "built from requirements.lock.txt to test both."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    result = run_clean_clone_gate(
        PROJECT_ROOT,
        keep=args.keep,
        run_tests=not args.no_tests,
        python=args.python,
    )
    print(render(result))

    write_artifact(result, Path(args.out), provenance(config))
    print(f"\nwrote {args.out}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
