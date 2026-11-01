"""Run with a clean wheel environment: /path/to/env/bin/python -I scripts/check_installation.py.

The environment must contain the built wheel and its core dependencies only.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import re
import sys
import tomllib
from pathlib import Path

import deskagents


def normalize_requirement(requirement: str) -> tuple[str, frozenset[str]]:
    """Name and specifier set; build backends may reorder specifiers."""
    match = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*", requirement)
    assert match is not None, requirement
    name = re.sub(r"[-_.]+", "-", match[1]).lower()
    specifiers = frozenset(s.replace(" ", "") for s in match[2].split(",") if s.strip())
    return name, specifiers


assert deskagents.__file__ is not None
installed = Path(deskagents.__file__).resolve()
assert installed.is_relative_to(Path(sys.prefix).resolve()), "imported from the checkout"
with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as stream:
    project = tomllib.load(stream)["project"]
expected_name = str(project["name"])
expected_version = str(project["version"])
assert importlib.metadata.version(expected_name) == expected_version
requires = importlib.metadata.requires(expected_name) or []
core = {normalize_requirement(r) for r in requires if "extra ==" not in r}
declared = {normalize_requirement(r) for r in project["dependencies"]}
assert core == declared, "installed requirements differ from pyproject"
assert importlib.resources.files("deskagents").joinpath("py.typed").is_file()
assert all(hasattr(deskagents, name) for name in deskagents.__all__)
direct_url_text = importlib.metadata.distribution(expected_name).read_text("direct_url.json")
assert direct_url_text is not None
direct_url = json.loads(direct_url_text)
assert direct_url["url"].endswith(".whl") and "dir_info" not in direct_url
print(
    json.dumps(
        {
            "python": sys.version.split()[0],
            "name": expected_name,
            "version": expected_version,
            "installed": str(installed),
            "runtime_dependencies": sorted(name for name, _ in core),
        }
    )
)
