"""scripts/upstream_changes.py detects upstream drift against a scratch git history."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
_script = runpy.run_path(str(ROOT / "scripts" / "upstream_changes.py"))
check = cast(Callable[[Path, dict[str, Any], dict[str, Any]], list[str]], _script["check"])
changes = cast(
    Callable[[Path, dict[str, Any], dict[str, Any], str], dict[str, Any]], _script["changes"]
)
has_changes = cast(Callable[[dict[str, Any]], bool], _script["has_changes"])


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), *identity, "commit", "-q", "-m", message], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def scratch_upstream(tmp_path: Path) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "--template=", str(repo)], check=True)
    asset = '{"formatVersion": 4}\n'
    pin = commit(
        repo,
        {
            "packages/core/tools/package.json": "{}\n",
            "packages/util/timeout/package.json": "{}\n",
            "packages/core/session/schema.json": asset,
            "vendor/cordis/package.json": json.dumps({"version": "4.0.6-alpha.1"}),
        },
        "initial",
    )
    data = {
        "upstream": {"commit": pin, "vendor": {"vendor/cordis": "4.0.6-alpha.1"}},
        "packages": [{"upstream": "packages/core/tools", "module": "deskagents.core.tools"}],
        "watched": [{"upstream": "packages/util/timeout", "used_by": ["packages/core/tools"]}],
    }
    assets = {
        "commit": pin,
        "files": [
            {
                "upstream": "packages/core/session/schema.json",
                "sha256": hashlib.sha256(asset.encode()).hexdigest(),
            }
        ],
    }
    return repo, pin, data, assets


def test_a_matching_pin_checks_clean_and_reports_nothing(tmp_path: Path) -> None:
    repo, pin, data, assets = scratch_upstream(tmp_path)
    assert check(repo, data, assets) == []
    assert not has_changes(changes(repo, data, assets, pin))


def test_check_rejects_a_missing_watched_package(tmp_path: Path) -> None:
    repo, pin, data, assets = scratch_upstream(tmp_path)
    data["watched"].append({"upstream": "packages/util/gone", "used_by": ["packages/core/tools"]})
    assert check(repo, data, assets) == [
        f"watched packages/util/gone is not a tracked package at {pin[:7]}"
    ]


def test_check_rejects_a_misrecorded_vendored_version(tmp_path: Path) -> None:
    repo, pin, data, assets = scratch_upstream(tmp_path)
    data["upstream"]["vendor"]["vendor/cordis"] = "4.0.0-rc.11"
    assert check(repo, data, assets) == [
        f"vendor/cordis is 4.0.6-alpha.1 at {pin[:7]}, map records 4.0.0-rc.11"
    ]


def test_report_lists_package_asset_vendor_and_new_package_drift(tmp_path: Path) -> None:
    repo, _, data, assets = scratch_upstream(tmp_path)
    head = commit(
        repo,
        {
            "packages/core/tools/src.ts": "export {}\n",
            "packages/util/timeout/src.ts": "export {}\n",
            "packages/core/session/schema.json": '{"formatVersion": 5}\n',
            "packages/core/jobs/package.json": "{}\n",
            "vendor/cordis/package.json": json.dumps({"version": "4.0.0-rc.11"}),
        },
        "upstream moves",
    )
    report = changes(repo, data, assets, head)
    assert has_changes(report)
    assert [entry["upstream"] for entry in report["packages"]] == ["packages/core/tools"]
    assert report["new_packages_in_mapped_groups"] == ["packages/core/jobs"]
    assert report["changed_assets"] == ["packages/core/session/schema.json"]
    [watched] = report["watched"]
    assert (watched["upstream"], watched["module"]) == (
        "packages/util/timeout",
        "packages/core/tools",
    )
    [vendor] = report["vendor"]
    assert (vendor["path"], vendor["pinned"], vendor["current"]) == (
        "vendor/cordis",
        "4.0.6-alpha.1",
        "4.0.0-rc.11",
    )
    assert len(vendor["commits"]) == 1
