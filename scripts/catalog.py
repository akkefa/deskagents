"""Keep docs/compatibility-cases.json in step with @pytest.mark.upstream tests.

Usage:
    python scripts/catalog.py check
    python scripts/catalog.py sync [--upstream PATH]

``check`` prints every inconsistency (the test suite runs the same check).
``sync`` adds a case for each marked test that has none, records each referenced
spec file with its SHA-256 at the pinned upstream commit, and keeps existing
classifications and notes. It never deletes a case.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "compatibility-cases.json"
UPSTREAM_MAP = ROOT / "docs" / "upstream.json"
# "contract": a case from an upstream shared contract module (e.g. contract.ts), run through
# deskagents.testing by a parametrized test; its id is the upstream title, so it carries no mark.
CLASSIFICATIONS = {"ported", "adapted", "not-applicable", "deferred", "contract"}
SOURCE = re.compile(r"(?P<path>packages/[a-z0-9-]+/[a-z0-9-]+/\S+\.ts)::(?P<title>.+)")


def _string(node: ast.expr, constants: dict[str, str]) -> str | None:
    """Evaluate a string literal, f-string or ``+`` over module-level string constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string(node.left, constants), _string(node.right, constants)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if value.conversion != -1 or value.format_spec is not None:
                    return None
                part = _string(value.value, constants)
            else:
                part = _string(value, constants)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _is_upstream_mark(node: ast.expr, constants: dict[str, str]) -> str | None:
    if not (isinstance(node, ast.Call) and node.args):
        return None
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "upstream"):
        return None
    mark = func.value
    if not (isinstance(mark, ast.Attribute) and mark.attr == "mark"):
        return None
    if not (isinstance(mark.value, ast.Name) and mark.value.id == "pytest"):
        return None
    source = _string(node.args[0], constants)
    return "" if source is None else source


def _module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _string(node.value, constants)
                if value is not None:
                    constants[target.id] = value
    return constants


def upstream_marks(tests_root: Path) -> dict[str, list[str]]:
    """Map each marked test node id to the upstream sources its marks name, in order."""
    marks: dict[str, list[str]] = {}
    for path in sorted(tests_root.rglob("test_*.py")):
        relative = path.relative_to(tests_root.parent).as_posix()
        tree = ast.parse(path.read_text())
        constants = _module_constants(tree)
        for node in tree.body:
            members: list[tuple[str, ast.AST]] = (
                [(node.name, node)]
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                else []
            )
            if isinstance(node, ast.ClassDef):
                members = [
                    (f"{node.name}::{child.name}", child)
                    for child in node.body
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                ]
            for name, member in members:
                assert isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
                sources: list[str] = []
                for decorator in member.decorator_list:
                    source = _is_upstream_mark(decorator, constants)
                    if source is not None and source not in sources:
                        sources.append(source)
                if sources:
                    marks[f"{relative}::{name}"] = sources
    return marks


def catalog_problems(
    catalog: dict[str, Any], marks: dict[str, list[str]], mapped: set[str]
) -> list[str]:
    problems: list[str] = []
    cases: list[dict[str, Any]] = catalog["cases"]
    files = {entry["path"]: entry for entry in catalog["source_files"]}
    for case_id, count in Counter(case["id"] for case in cases).items():
        if count > 1:
            problems.append(f"duplicate case id {case_id}")
    if len(files) != len(catalog["source_files"]):
        problems.append("duplicate source file")
    for entry in catalog["source_files"]:
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            problems.append(f"bad sha256 for {entry['path']}")
    used_files: set[str] = set()
    claimed: set[tuple[str, str]] = set()
    for case in cases:
        match = SOURCE.fullmatch(case["source"])
        if match is None:
            problems.append(f"{case['id']}: source must be 'packages/.../x.spec.ts::title'")
            continue
        spec = match["path"]
        used_files.add(spec)
        if not any(spec.startswith(package + "/") for package in mapped):
            problems.append(f"{case['id']}: {spec} is outside every mapped package")
        if spec not in files:
            problems.append(f"{case['id']}: {spec} missing from source_files")
        if case["classification"] not in CLASSIFICATIONS:
            problems.append(f"{case['id']}: unknown classification {case['classification']}")
        if (
            case["classification"] not in ("not-applicable", "deferred", "contract")
            and not case["python_tests"]
        ):
            problems.append(f"{case['id']}: no Python tests")
        for test in case["python_tests"]:
            claimed.add((test, case["source"]))
            if case["source"] not in marks.get(test, ()):
                problems.append(f"{case['id']}: {test} lacks the matching upstream mark")
    for test, sources in marks.items():
        for source in sources:
            if (test, source) not in claimed:
                problems.append(f"{test} is marked upstream({source!r}) but not catalogued")
    for path in files.keys() - used_files:
        problems.append(f"source file {path} has no cases")
    return problems


def _case_id(source: str) -> str:
    match = SOURCE.fullmatch(source)
    assert match is not None, source
    package = match["path"].split("/tests/")[0].removeprefix("packages/")
    slug = re.sub(r"[^a-z0-9]+", "-", match["title"].lower()).strip("-")
    return f"{package}::{slug}"


def sync(upstream: Path) -> dict[str, Any]:
    catalog: dict[str, Any] = json.loads(CATALOG.read_text())
    pin = json.loads(UPSTREAM_MAP.read_text())["upstream"]["commit"]
    catalog["upstream_commit"] = pin
    marks = upstream_marks(ROOT / "tests")
    by_source: dict[str, dict[str, Any]] = {case["source"]: case for case in catalog["cases"]}
    for test, source in sorted(
        (test, source) for test, sources in marks.items() for source in sources
    ):
        case = by_source.get(source)
        if case is None:
            case = {
                "id": _case_id(source),
                "source": source,
                "classification": "ported",
                "python_tests": [],
                "notes": "",
            }
            catalog["cases"].append(case)
            by_source[source] = case
        if test not in case["python_tests"]:
            case["python_tests"].append(test)
    specs = sorted({SOURCE.fullmatch(case["source"])["path"] for case in catalog["cases"]})  # type: ignore[index]
    files = []
    missing = []
    for spec in specs:
        blob = subprocess.run(
            ["git", "-C", str(upstream), "show", f"{pin}:{spec}"], check=True, capture_output=True
        ).stdout
        files.append({"path": spec, "sha256": hashlib.sha256(blob).hexdigest()})
        text = blob.decode()
        for case in catalog["cases"]:
            path, title = case["source"].split("::", 1)
            # Every describe/it segment must be a literal title in the spec file.
            if path == spec and not all(
                f"'{part}'" in text or f'"{part}"' in text or f"`{part}`" in text
                for part in title.split(" > ")
            ):
                missing.append(case["source"])
    if missing:
        raise SystemExit("titles not found in their upstream spec:\n  " + "\n  ".join(missing))
    catalog["source_files"] = files
    catalog["cases"].sort(key=lambda case: (case["source"].split("::")[0], case["id"]))
    CATALOG.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    return catalog


def current_problems() -> list[str]:
    catalog = json.loads(CATALOG.read_text())
    upstream_map = json.loads(UPSTREAM_MAP.read_text())
    problems = []
    if catalog["upstream_commit"] != upstream_map["upstream"]["commit"]:
        problems.append("catalog upstream_commit differs from docs/upstream.json")
    mapped = {package["upstream"] for package in upstream_map["packages"]}
    return problems + catalog_problems(catalog, upstream_marks(ROOT / "tests"), mapped)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from upstream_changes import default_upstream

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check", "sync"])
    parser.add_argument("--upstream", type=Path, default=default_upstream())
    args = parser.parse_args()
    if args.command == "sync":
        catalog = sync(args.upstream)
        print(f"{len(catalog['cases'])} cases from {len(catalog['source_files'])} spec files")
    problems = current_problems()
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
