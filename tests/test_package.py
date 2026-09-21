"""Smoke tests: the package is installed, importable, and running on the pinned Python."""

import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import nhlsim

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_installed_metadata() -> None:
    assert nhlsim.__version__ == version("nhlsim")


def test_installed_version_matches_pyproject() -> None:
    # Fails if pyproject.toml's version was changed without re-running `uv sync`.
    with PYPROJECT.open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert nhlsim.__version__ == declared


def test_running_on_pinned_python() -> None:
    # pyproject.toml pins requires-python to 3.12; CI must run the same version.
    assert sys.version_info[:2] == (3, 12)
