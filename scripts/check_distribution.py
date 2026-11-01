"""Audit the exact local wheel/sdist pair; uses only the Python standard library."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import tarfile
import tomllib
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

IMPORT_NAME = "deskagents"
SDIST_FOLDERS = ["src", "tests", "docs", "scripts", "examples"]
SDIST_FILES = [
    ".gitignore",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
]


def normalize_requirement(requirement: str) -> tuple[str, frozenset[str]]:
    """Name and specifier set; build backends may reorder specifiers."""
    match = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*", requirement)
    assert match is not None, requirement
    name = re.sub(r"[-_.]+", "-", match[1]).lower()
    specifiers = frozenset(s.replace(" ", "") for s in match[2].split(",") if s.strip())
    return name, specifiers


def check_metadata(data: bytes, project: dict[str, Any], readme: str) -> None:
    metadata = BytesParser(policy=policy.default).parsebytes(data)
    assert str(metadata["Metadata-Version"]) == "2.4"
    for field, key in [
        ("Name", "name"),
        ("Version", "version"),
        ("Summary", "description"),
        ("Requires-Python", "requires-python"),
        ("License-Expression", "license"),
    ]:
        assert str(metadata[field]) == project[key], f"incorrect {field}"
    assert str(metadata["Author"]) == project["authors"][0]["name"]
    assert str(metadata["Description-Content-Type"]) == "text/markdown"
    payload = metadata.get_payload(decode=True)
    assert isinstance(payload, bytes) and payload.decode().strip() == readme.strip(), (
        "long description differs"
    )
    assert set(metadata.get_all("License-File", [])) == set(project["license-files"])
    assert set(metadata.get_all("Classifier", [])) == set(project["classifiers"])
    extras = project.get("optional-dependencies", {})
    requires = {str(value) for value in metadata.get_all("Requires-Dist", [])}
    core = {normalize_requirement(value) for value in requires if "extra ==" not in value}
    declared = {normalize_requirement(value) for value in project["dependencies"]}
    assert core == declared, "runtime dependencies differ from pyproject"
    assert set(metadata.get_all("Provides-Extra", [])) == set(extras), "extras differ"
    urls = dict(str(value).split(", ", 1) for value in metadata.get_all("Project-URL", []))
    assert urls == project["urls"], "project URLs differ"
    for url in urls.values():
        parsed = urlsplit(url)
        assert parsed.scheme == "https" and parsed.hostname
        assert parsed.username is None and parsed.password is None, "credentials in project URL"


def check_distributions(wheel: Path, sdist: Path, root: Path) -> dict[str, object]:
    with (root / "pyproject.toml").open("rb") as config_file:
        project = tomllib.load(config_file)["project"]
    name, version = project["name"], project["version"]
    archive_name = re.sub(r"[-_.]+", "_", name).lower()
    dist_info = f"{archive_name}-{version}.dist-info"
    readme = (root / project["readme"]).read_text()
    sources = {
        path.relative_to(root / "src").as_posix(): path.read_bytes()
        for path in (root / "src" / IMPORT_NAME).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (path.suffix in {".py", ".yml"} or path.name == "py.typed")
    }
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "duplicate wheel member"
        allowed = {
            *sources,
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/RECORD",
            *(f"{dist_info}/licenses/{file}" for file in project["license-files"]),
        }
        assert set(names) == allowed, "wheel has missing or unexpected contents"
        for path, data in sources.items():
            assert archive.read(path) == data, f"wheel source mismatch: {path}"
        assert f"{IMPORT_NAME}/py.typed" in names
        for file in project["license-files"]:
            assert archive.read(f"{dist_info}/licenses/{file}") == (root / file).read_bytes()
        check_metadata(archive.read(f"{dist_info}/METADATA"), project, readme)
        wheel_meta = BytesParser(policy=policy.default).parsebytes(
            archive.read(f"{dist_info}/WHEEL")
        )
        assert str(wheel_meta["Root-Is-Purelib"]) == "true"
        assert wheel_meta.get_all("Tag") == ["py3-none-any"]
        rows = list(csv.reader(io.StringIO(archive.read(f"{dist_info}/RECORD").decode())))
        assert len(rows) == len(names) and {row[0] for row in rows} == set(names)
        for path, digest, size in rows:
            if path.endswith("/RECORD"):
                assert digest == size == ""
                continue
            data = archive.read(path)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            assert digest == f"sha256={expected}" and int(size) == len(data), f"bad RECORD: {path}"
    prefix = f"{archive_name}-{version}"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names)), "duplicate sdist member"
        for member in members:
            member_path = PurePosixPath(member.name)
            assert not member_path.is_absolute() and ".." not in member_path.parts
            assert member_path.parts[0] == prefix and (member.isfile() or member.isdir())
            assert not set(member_path.parts) & {".git", ".venv", "__pycache__", "dist", ".env"}
        expected_files = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for folder in SDIST_FOLDERS
            if (root / folder).is_dir()
            for path in (root / folder).rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root).parts
            )
            and path.suffix != ".pyc"
        }
        for file in SDIST_FILES:
            expected_files[file] = (root / file).read_bytes()
        actual_files = {
            member.name.removeprefix(f"{prefix}/") for member in members if member.isfile()
        }
        assert actual_files == {*expected_files, "PKG-INFO"}, "source archive contents differ"
        for path, data in expected_files.items():
            stream = archive.extractfile(f"{prefix}/{path}")
            assert stream is not None and stream.read() == data, f"source archive mismatch: {path}"
        stream = archive.extractfile(f"{prefix}/PKG-INFO")
        assert stream is not None
        check_metadata(stream.read(), project, readme)
    return {
        "name": name,
        "version": version,
        "wheel_files": len(allowed),
        "sdist_files": len(actual_files),
        "runtime_dependencies": project["dependencies"],
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [wheel, sdist]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(check_distributions(args.wheel, args.sdist, root), indent=2))


if __name__ == "__main__":
    main()
