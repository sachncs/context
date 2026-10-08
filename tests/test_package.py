import importlib.metadata
from pathlib import Path

import foveate

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject_and_metadata():
    declared = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    assert foveate.__version__ == declared
    assert importlib.metadata.version("foveate") == declared


def test_public_api_is_importable_and_documented():
    for name in foveate.__all__:
        assert hasattr(foveate, name)
    assert foveate.Context.__doc__ and foveate.Runtime.__doc__


def test_every_package_declares_all():
    for init in (ROOT / "foveate").rglob("__init__.py"):
        if init.parent.name == "internals" or init.parent.name == "foveate":
            continue
        assert "__all__" in init.read_text(encoding="utf-8"), init


def test_package_data_declared_for_resources():
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    data = config["tool"]["setuptools"]["package-data"]["foveate"]
    assert "bench/fixtures/*.jsonl" in data and "bench/seeds/*.md" in data
