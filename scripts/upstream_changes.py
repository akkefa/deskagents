"""List upstream DeepSeek Harness changes since the pin in docs/upstream.json.

Usage:
    python scripts/upstream_changes.py [--upstream PATH] [--to REF] [--check] [--json]
                                       [--exit-code]

The upstream checkout defaults to $HARNESS_UPSTREAM, then ../deepseek-harness next to
this repository. It is read with git only; nothing in it is executed.

Without --check, prints every upstream commit since the pin that touches a mapped
package, a watched dependency of the core packages, or a pinned vendored Cordis
package, plus packages that appeared in a mapped group without being mapped. With
--exit-code, exits 3 when the report is not empty. With --check, verifies that every
mapped and watched package, pinned asset and vendored version matches the pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "docs" / "upstream.json"
ASSETS_PATH = ROOT / "tests" / "upstream_assets" / "MANIFEST.json"
CHANGED = 3


@dataclass
class PackageChanges:
    upstream: str
    module: str
    commits: list[str] = field(default_factory=list)


@dataclass
class VendorChanges:
    path: str
    pinned: str
    current: str | None
    commits: list[str] = field(default_factory=list)


def git(upstream: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream), *args], check=True, capture_output=True, text=True
    )
    return result.stdout


def load_map() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(MAP_PATH.read_text())
    return data


def default_upstream() -> Path:
    configured = os.environ.get("HARNESS_UPSTREAM")
    return Path(configured) if configured else ROOT.parent / "deepseek-harness"


def vendored_version(upstream: Path, ref: str, path: str) -> str | None:
    shown = subprocess.run(
        ["git", "-C", str(upstream), "show", f"{ref}:{path}/package.json"],
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        return None
    version = json.loads(shown.stdout).get("version")
    return version if isinstance(version, str) else None


def tracked_manifests(upstream: Path, ref: str) -> set[str]:
    listing = git(upstream, "ls-tree", "-r", "--name-only", ref, "--", "packages")
    return {
        line.removesuffix("/package.json")
        for line in listing.splitlines()
        if line.endswith("/package.json") and line.count("/") == 3
    }


def load_assets() -> dict[str, Any]:
    assets: dict[str, Any] = json.loads(ASSETS_PATH.read_text())
    return assets


def check(upstream: Path, data: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    pin = data["upstream"]["commit"]
    problems: list[str] = []
    present = tracked_manifests(upstream, pin)
    for package in data["packages"]:
        if package["upstream"] not in present:
            problems.append(f"{package['upstream']} is not a tracked package at {pin[:7]}")
    if assets["commit"] != pin:
        problems.append(f"assets pinned at {assets['commit'][:7]}, map pinned at {pin[:7]}")
    for asset in assets["files"]:
        blob = subprocess.run(
            ["git", "-C", str(upstream), "show", f"{assets['commit']}:{asset['upstream']}"],
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != asset["sha256"]:
            problems.append(f"{asset['upstream']} differs from its pinned copy")
    for path, pinned in data["upstream"].get("vendor", {}).items():
        actual = vendored_version(upstream, pin, path)
        if actual != pinned:
            problems.append(f"{path} is {actual} at {pin[:7]}, map records {pinned}")
    for watched in data.get("watched", ()):
        if watched["upstream"] not in present:
            problems.append(f"watched {watched['upstream']} is not a tracked package at {pin[:7]}")
    return problems


def changes(
    upstream: Path, data: dict[str, Any], assets: dict[str, Any], to: str
) -> dict[str, Any]:
    pin = data["upstream"]["commit"]
    report: list[PackageChanges] = []
    for package in data["packages"]:
        log = git(upstream, "log", "--format=%h %cs %s", f"{pin}..{to}", "--", package["upstream"])
        entry = PackageChanges(package["upstream"], package["module"], log.splitlines())
        if entry.commits:
            report.append(entry)
    mapped = {package["upstream"] for package in data["packages"]}
    groups = {path.rsplit("/", 1)[0] for path in mapped}
    unmapped = sorted(
        path
        for path in tracked_manifests(upstream, to) - tracked_manifests(upstream, pin)
        if path.rsplit("/", 1)[0] in groups
    )
    changed_assets = [
        asset["upstream"]
        for asset in assets["files"]
        if git(upstream, "log", "--format=%h", f"{assets['commit']}..{to}", "--", asset["upstream"])
    ]
    watched: list[PackageChanges] = []
    for item in data.get("watched", ()):
        log = git(upstream, "log", "--format=%h %cs %s", f"{pin}..{to}", "--", item["upstream"])
        if log:
            used_by = ", ".join(item["used_by"])
            watched.append(PackageChanges(item["upstream"], used_by, log.splitlines()))
    vendor: list[VendorChanges] = []
    for path, pinned in data["upstream"].get("vendor", {}).items():
        log = git(upstream, "log", "--format=%h %cs %s", f"{pin}..{to}", "--", path)
        if log:
            current = vendored_version(upstream, to, path)
            vendor.append(VendorChanges(path, pinned, current, log.splitlines()))
    return {
        "pin": pin,
        "to": git(upstream, "rev-parse", to).strip(),
        "packages": [asdict(entry) for entry in report],
        "new_packages_in_mapped_groups": unmapped,
        "changed_assets": changed_assets,
        "watched": [asdict(entry) for entry in watched],
        "vendor": [asdict(entry) for entry in vendor],
    }


def has_changes(report: dict[str, Any]) -> bool:
    keys = ("packages", "new_packages_in_mapped_groups", "changed_assets", "watched", "vendor")
    return any(report[key] for key in keys)


def print_report(report: dict[str, Any]) -> None:
    print(f"Upstream changes {report['pin'][:7]}..{report['to'][:7]}")
    if not report["packages"]:
        print("No mapped package changed.")
    for entry in report["packages"]:
        print(f"\n{entry['upstream']} -> {entry['module']} ({len(entry['commits'])} commits)")
        for line in entry["commits"]:
            print(f"  {line}")
    if report["new_packages_in_mapped_groups"]:
        print("\nNew upstream packages in mapped groups (not mapped yet):")
        for path in report["new_packages_in_mapped_groups"]:
            print(f"  {path}")
    if report["changed_assets"]:
        print("\nPinned assets changed upstream:")
        for path in report["changed_assets"]:
            print(f"  {path}")
    for entry in report["watched"]:
        count = len(entry["commits"])
        print(f"\n{entry['upstream']} (watched, used by {entry['module']}; {count} commits)")
        for line in entry["commits"]:
            print(f"  {line}")
    for entry in report["vendor"]:
        version = f"{entry['pinned']} -> {entry['current']}"
        print(f"\n{entry['path']} (deepseek-cordis runtime) {version}")
        for line in entry["commits"]:
            print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", type=Path, default=default_upstream())
    parser.add_argument("--to", default="HEAD", help="upstream ref to compare with the pin")
    parser.add_argument("--check", action="store_true", help="verify the map and assets")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--exit-code", action="store_true", help=f"exit {CHANGED} when anything changed"
    )
    args = parser.parse_args()
    data, assets = load_map(), load_assets()
    if args.check:
        problems = check(args.upstream, data, assets)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            pin = data["upstream"]["commit"][:7]
            vendor = len(data["upstream"].get("vendor", {}))
            print(
                f"{len(data['packages'])} packages, pinned assets and {vendor} vendored "
                f"versions verified at {pin}"
            )
        return 1 if problems else 0
    report = changes(args.upstream, data, assets, args.to)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return CHANGED if args.exit_code and has_changes(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
