"""Enforces the project's no-underscore naming rule over src/."""

import ast
import pathlib

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "src" / "ceng"


def is_semi_private(name: str) -> bool:
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


def violations(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.append(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx, ast.Store
        ):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.alias):
            names.append(node.asname or node.name.split(".")[-1])
        for name in names:
            if is_semi_private(name):
                line = getattr(node, "lineno", 0)
                found.append(f"{path.relative_to(SOURCE)}:{line} {name}")
    return found


def test_no_semi_private_names() -> None:
    found = [v for p in sorted(SOURCE.rglob("*.py")) for v in violations(p)]
    assert not found, "underscore-prefixed names:\n" + "\n".join(found)
