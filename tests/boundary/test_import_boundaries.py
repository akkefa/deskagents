"""Import rules for src/deskagents.

1. Product code (PepGPT and friends) and libraries we decided against never appear.
2. Every other third-party import is a declared core dependency, or belongs to an
   extra and appears only in the modules registered for that extra below.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "deskagents"

FORBIDDEN_PREFIXES = ("app", "m365", "onereach_profile", "pepgpt", "langchain", "sqlalchemy")

# Distribution name -> import name, for dependencies whose names differ.
IMPORT_NAMES = {"deepseek-cordis": "deepseek_cordis", "pyyaml": "yaml"}

# Extra name -> (import names it provides, modules allowed to import them).
# Each phase that adds an extra registers it here, e.g.
#   "openai": ({"httpx"}, {"deskagents.llm.openai"}),
EXTRA_IMPORTS: dict[str, tuple[set[str], set[str]]] = {
    "deepseek": ({"httpx"}, {"deskagents.llm.deepseek"}),
    "openai": ({"httpx"}, {"deskagents.llm.openai"}),
}


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def absolute_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def requirement_name(requirement: str) -> str:
    for separator in "<>=!~;[ ":
        requirement = requirement.split(separator)[0]
    return requirement.lower().replace("_", "-")


def core_imports() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    names = (requirement_name(r) for r in project["dependencies"])
    return {IMPORT_NAMES.get(name, name.replace("-", "_")) for name in names}


def boundary_problems() -> list[str]:
    allowed = set(sys.stdlib_module_names) | {"deskagents"} | core_imports()
    problems: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        module = module_name(path)
        for name in sorted(absolute_imports(path)):
            if name.startswith(FORBIDDEN_PREFIXES):
                problems.append(f"{module} imports forbidden {name}")
            elif name in allowed:
                continue
            elif any(
                name in imports and module in modules for imports, modules in EXTRA_IMPORTS.values()
            ):
                continue
            else:
                problems.append(f"{module} imports undeclared {name}")
    return problems


def test_source_respects_import_boundaries() -> None:
    assert boundary_problems() == []


def test_extras_are_registered_for_every_declared_extra() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    declared = set(project.get("optional-dependencies", {})) - {"all"}
    assert declared == set(EXTRA_IMPORTS)


def test_core_dependency_import_names() -> None:
    assert core_imports() == {
        "deepseek_cordis",
        "pydantic",
        "typing_extensions",
        "yaml",
        "zstandard",
    }
