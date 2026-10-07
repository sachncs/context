import importlib.metadata
from pathlib import Path

import ceng

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject_and_metadata():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    assert ceng.__version__ == declared
    assert importlib.metadata.version("ceng-context") == declared


def test_public_api_is_importable_and_documented():
    for name in ceng.__all__:
        assert hasattr(ceng, name)
    assert ceng.Context.__doc__ and ceng.Runtime.__doc__


def test_every_package_declares_all():
    for init in (ROOT / "src" / "ceng").rglob("__init__.py"):
        if init.parent.name == "internals" or init.parent.name == "ceng":
            continue
        assert "__all__" in init.read_text(), init


def test_package_data_declared_for_resources():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    data = config["tool"]["setuptools"]["package-data"]["ceng"]
    assert "bench/fixtures/*.jsonl" in data and "bench/seeds/*.md" in data
