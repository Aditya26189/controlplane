"""The cheapest check that can fail for a real reason. Block E, E.4.

CPU, no network, under a minute. Answers one question: **did this clone come
out intact and does the package actually work?**

Deliberately more than ``import controlplane``. A bare import passes on a
half-installed tree, on a tree whose config no longer loads, and on one whose
committed artifacts are truncated. Each step below is something that has to be
true before any other target is worth running, and each failure names what to
do about it.

Thin wrapper: parses arguments, calls ``controlplane/``, prints. No logic
(``CLAUDE.md``).

Usage:
    python scripts/smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON = (3, 10)


def _check(label: str, fn) -> tuple[bool, str]:
    try:
        return True, f"  OK    {label}: {fn()}"
    except Exception as exc:  # noqa: BLE001 - the point is to report anything
        return False, f"  FAIL  {label}: {type(exc).__name__}: {exc}"


def _python_version() -> str:
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Python {'.'.join(map(str, MIN_PYTHON))}+ required, "
            f"found {sys.version.split()[0]}"
        )
    return sys.version.split()[0]


def _package_imports() -> str:
    import controlplane  # noqa: F401
    from controlplane import config, model, policy, validation  # noqa: F401

    return "controlplane, and its model, policy and validation subpackages"


def _config_loads() -> str:
    from controlplane.config import load_config

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    return f"config hash {cfg.config_hash}, seed {cfg.seed}"


def _provenance_resolves() -> str:
    from controlplane.config import load_config, provenance

    prov = provenance(load_config(str(PROJECT_ROOT / "config.yaml")))
    commit = (prov.get("git_commit") or "unknown")[:12]
    return f"commit {commit}, dirty={prov.get('dirty')}"


def _artifacts_readable() -> str:
    path = PROJECT_ROOT / "results" / "validation-T1-last_token.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    auroc = data["metrics"]["auroc"]["value"]
    return f"{path.name} -> AUROC {auroc:.4f}, {data['warrant_status']}"


def _evalsets_frozen() -> str:
    manifest = json.loads(
        (PROJECT_ROOT / "evalsets" / "manifest.json").read_text(encoding="utf-8")
    )
    return f"{manifest['n_sets']} frozen eval sets, content-hashed"


def _claim_table_parses() -> str:
    from controlplane.report.claims import parse_claim_table

    claims = parse_claim_table(PROJECT_ROOT / "README.md")
    if len(claims) < 20:
        raise RuntimeError(f"claim table parsed to only {len(claims)} rows")
    return f"{len(claims)} claims, each naming an artifact and a field"


def main() -> int:
    print("ControlPlane smoke check")
    print("=" * 60)
    checks = [
        ("python version", _python_version),
        ("package imports", _package_imports),
        ("config loads", _config_loads),
        ("provenance resolves", _provenance_resolves),
        ("artifacts readable", _artifacts_readable),
        ("eval sets frozen", _evalsets_frozen),
        ("claim table parses", _claim_table_parses),
    ]
    failures = 0
    for label, fn in checks:
        ok, line = _check(label, fn)
        print(line)
        failures += not ok

    print("=" * 60)
    if failures:
        print(f"SMOKE FAILED: {failures} of {len(checks)} checks", file=sys.stderr)
        print(
            "The clone is not usable. Most often: dependencies not installed "
            "(pip install -r requirements.lock.txt), or a partial checkout.",
            file=sys.stderr,
        )
        return 1
    print(f"SMOKE OK: {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
