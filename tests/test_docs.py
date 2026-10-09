"""Documentation stays true: runnable examples run and relative links resolve."""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(
    p
    for p in [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")]
    if p.name != "plan.md"
)
RUN = re.compile(r"<!-- run -->\s*```python\n(.*?)```", re.DOTALL)
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path):
    broken = []
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("mailto:"):
            continue
        if not (path.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{path.name}: {broken}"


def test_runnable_examples_in_the_docs_execute():
    blocks = [
        (path, code)
        for path in FILES
        for code in RUN.findall(path.read_text(encoding="utf-8"))
    ]
    assert blocks, "no runnable doc examples found"
    for path, code in blocks:
        exec(compile(code, str(path), "exec"), {"__name__": "__doc__"})  # noqa: S102


def test_no_stale_names_in_the_docs():
    stale = ("ceng", "CENG_", "src/foveate", "v2.0", "NVIDIA_API_KEY secret")
    hits = []
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        hits += [f"{path.name}: {s}" for s in stale if s in text]
    assert not hits


def test_roadmap_items_are_complete_and_listed_in_the_body():
    text = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    items = json.loads(front)["items"]
    assert items
    for item in items:
        assert item["status"] in {
            "shipping",
            "building",
            "planned",
            "exploring",
        }
        assert item["summary"] and item["why"]
        assert item["title"] in body, item["title"]
