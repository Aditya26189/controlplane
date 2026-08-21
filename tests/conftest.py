"""Shared pytest fixtures.

Puts the repository root on ``sys.path`` so ``import src...`` works regardless
of where pytest is invoked from, and provides the fixtures that more than one
test file needs.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config  # noqa: E402  (path set up above)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def config() -> Config:
    """The committed ``config.yaml``, loaded and validated.

    Session-scoped: the config is frozen, so no test can mutate it for another.
    """
    return load_config(REPO_ROOT / "config.yaml")
