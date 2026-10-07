import pathlib
import runpy

import pytest

EXAMPLES = sorted(
    (pathlib.Path(__file__).resolve().parents[1] / "examples").glob("*.py")
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_offline(path, capsys):
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_examples_exist():
    assert len(EXAMPLES) >= 4
