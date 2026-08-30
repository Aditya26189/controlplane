"""Shared pytest fixtures.

Puts the project root on ``sys.path`` so ``import controlplane...`` resolves regardless
of where pytest was invoked from, and provides the fixtures more than one test
file needs.

Everything here is built offline. The suite must run on a clean checkout with
no network and no model cache, because a test suite that needs a download is a
test suite that stops being run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import Config, load_config  # noqa: E402  (path set up above)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the Round 2 project root (the directory holding config.yaml)."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config() -> Config:
    """The committed ``config.yaml``, loaded and validated.

    Session-scoped because the config is frozen: no test can mutate it for
    another, so there is nothing to isolate.
    """
    return load_config(PROJECT_ROOT / "config.yaml")
