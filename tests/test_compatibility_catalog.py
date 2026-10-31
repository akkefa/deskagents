"""Keep docs/compatibility-cases.json and @pytest.mark.upstream tests in step.

Every Python test that claims an upstream origin must be listed in the catalog under
that origin, and every catalog case must point at tests that exist and carry the mark.
The logic lives in scripts/catalog.py, which also syncs the catalog.
"""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
_catalog = runpy.run_path(str(ROOT / "scripts" / "catalog.py"))
upstream_marks = cast(Callable[[Path], dict[str, list[str]]], _catalog["upstream_marks"])
catalog_problems = cast(
    Callable[[dict[str, Any], dict[str, list[str]], set[str]], list[str]],
    _catalog["catalog_problems"],
)
current_problems = cast(Callable[[], list[str]], _catalog["current_problems"])


def test_catalog_matches_marked_tests() -> None:
    assert current_problems() == []


SPEC = "packages/core/session/tests/surface.spec.ts"
GOOD: dict[str, Any] = {
    "source_files": [{"path": SPEC, "sha256": "0" * 64}],
    "cases": [
        {
            "id": "session-surface-1",
            "source": f"{SPEC}::replaces a range",
            "classification": "ported",
            "python_tests": ["tests/test_x.py::test_replace"],
        }
    ],
}
MARKS = {"tests/test_x.py::test_replace": [f"{SPEC}::replaces a range"]}
MAPPED = {"packages/core/session"}


def test_checker_accepts_a_consistent_catalog() -> None:
    assert catalog_problems(GOOD, MARKS, MAPPED) == []


def test_checker_rejects_an_uncatalogued_mark() -> None:
    extra = {**MARKS, "tests/test_x.py::test_other": [f"{SPEC}::other"]}
    assert any("not catalogued" in p for p in catalog_problems(GOOD, extra, MAPPED))


def test_checker_rejects_an_uncatalogued_mark_stacked_on_a_catalogued_test() -> None:
    stacked = {"tests/test_x.py::test_replace": [f"{SPEC}::replaces a range", f"{SPEC}::other"]}
    assert catalog_problems(GOOD, stacked, MAPPED) == [
        f"tests/test_x.py::test_replace is marked upstream('{SPEC}::other') but not catalogued"
    ]


def test_checker_rejects_a_case_without_its_mark() -> None:
    assert any("lacks the matching" in p for p in catalog_problems(GOOD, {}, MAPPED))


def test_checker_rejects_an_unmapped_source() -> None:
    assert any("outside every mapped" in p for p in catalog_problems(GOOD, MARKS, set()))


def test_marks_are_found_in_functions_and_classes(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "import pytest\n\n"
        f"@pytest.mark.upstream('{SPEC}::a')\n"
        f"@pytest.mark.upstream('{SPEC}::c')\n"
        "def test_a() -> None: ...\n\n"
        "@pytest.mark.asyncio\n"
        "def test_unmarked() -> None: ...\n\n"
        "class TestGroup:\n"
        f"    @pytest.mark.upstream('{SPEC}::b')\n"
        "    async def test_b(self) -> None: ...\n"
    )
    assert upstream_marks(tests) == {
        "tests/test_sample.py::test_a": [f"{SPEC}::a", f"{SPEC}::c"],
        "tests/test_sample.py::TestGroup::test_b": [f"{SPEC}::b"],
    }
