#!/usr/bin/env python3
"""Validate cross-service release identity, licensing and source indirection."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from lara_ltx.errors import LaraError

RELEASE_SCHEMA_VERSION = 1
MODEL_MANIFEST_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pyproject", required=True, type=Path)
    parser.add_argument("--readme", required=True, type=Path)
    parser.add_argument("--hf-readme", required=True, type=Path)
    parser.add_argument("--hf-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise LaraError("LARA-RELEASE-001", details={"reason": f"unreadable_toml:{path.name}"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-RELEASE-001", details={"reason": f"invalid_toml:{path.name}"})
    return value


def _section(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise LaraError("LARA-RELEASE-001", details={"reason": f"missing_section:{key}"})
    return result


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise LaraError("LARA-RELEASE-001", details={"reason": f"unreadable_file:{path.name}"}) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_release_bundle(
    *,
    project_root: Path,
    config_path: Path,
    pyproject_path: Path,
    readme_path: Path,
    hf_readme_path: Path,
    hf_manifest_path: Path,
) -> dict[str, Any]:
    config = _read_toml(config_path)
    release = _section(config, "release")
    github = _section(release, "github")
    hugging_face = _section(release, "hugging_face")
    upstream = _section(release, "upstream")
    license_config = _section(release, "license")
    pyproject = _read_toml(pyproject_path)
    project = _section(pyproject, "project")
    manifest = _read_toml(hf_manifest_path)
    manifest_header = _section(manifest, "manifest")
    manifest_release = _section(manifest, "release")
    manifest_source = _section(manifest, "source")

    version = str(release.get("version", "")).strip()
    github_id = f"{github.get('owner', '')}/{github.get('repository', '')}"
    hugging_face_id = f"{hugging_face.get('owner', '')}/{hugging_face.get('repository', '')}"
    upstream_id = str(upstream.get("repository", "")).strip()
    revision = str(upstream.get("revision", "")).strip()
    checks = {
        "release_schema": release.get("schema_version") == RELEASE_SCHEMA_VERSION,
        "package_version": bool(version) and version == str(project.get("version", "")).strip(),
        "github_identity": github_id.count("/") == 1 and not github_id.startswith("/") and not github_id.endswith("/"),
        "hugging_face_identity": (
            hugging_face_id.count("/") == 1
            and not hugging_face_id.startswith("/")
            and not hugging_face_id.endswith("/")
        ),
        "manifest_schema": manifest_header.get("schema_version") == MODEL_MANIFEST_SCHEMA_VERSION,
        "manifest_version": manifest_release.get("version") == version,
        "manifest_source_repository": manifest_release.get("source_repository") == github_id,
        "manifest_source_tag": manifest_release.get("source_tag") == f"v{version}",
        "upstream_identity": manifest_source.get("repository_id") == upstream_id,
        "upstream_revision": bool(revision) and manifest_source.get("revision") == revision,
    }

    root_readme = _text(readme_path)
    hf_readme = _text(hf_readme_path)
    checks.update(
        {
            "root_github_link": f"https://github.com/{github_id}" in root_readme,
            "root_hugging_face_id": hugging_face_id in root_readme,
            "hf_github_link": f"https://github.com/{github_id}" in hf_readme,
            "hf_upstream_id": upstream_id in hf_readme,
        }
    )
    license_path = project_root / str(license_config.get("file", ""))
    notice_path = project_root / str(license_config.get("notice", ""))
    checks["license_file"] = license_path.is_file()
    checks["notice_file"] = notice_path.is_file()
    if checks["license_file"]:
        checks["license_sha256"] = _sha256(license_path) == str(license_config.get("sha256", ""))
    if not all(checks.values()):
        failed = next(name for name, passed in checks.items() if not passed)
        raise LaraError("LARA-RELEASE-001", details={"reason": f"failed_check:{failed}"})

    files = (config_path, pyproject_path, readme_path, hf_readme_path, hf_manifest_path, license_path, notice_path)
    return {
        "schema_version": 1,
        "component": "release_bundle_validation",
        "version": version,
        "github_repository": github_id,
        "hugging_face_repository": hugging_face_id,
        "upstream": {"repository": upstream_id, "revision": revision},
        "checks": checks,
        "files": {path.name: _sha256(path) for path in files},
        "passed": True,
    }


def main() -> int:
    arguments = parse_arguments()
    report = validate_release_bundle(
        project_root=arguments.project_root.resolve(),
        config_path=arguments.config.resolve(),
        pyproject_path=arguments.pyproject.resolve(),
        readme_path=arguments.readme.resolve(),
        hf_readme_path=arguments.hf_readme.resolve(),
        hf_manifest_path=arguments.hf_manifest.resolve(),
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
