from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import re
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import jsonschema  # type: ignore[import-untyped]

from nvidia_converge.files import (
    BoundedFileError,
    fsync_directory,
    read_bounded_utf8,
)
from nvidia_converge.schemas import load_schema, strict_format_checker
from scripts.check_release_evidence import (
    EXPECTED_REPOSITORY,
    MAX_ARTIFACT_ARCHIVE_BYTES,
    MAX_EVIDENCE_BYTES,
    _artifact_evidence_members,
    _object_without_duplicates,
    check_commit_provenance,
    check_evidence,
    check_qualification_wheel_binding,
    verify_github_evidence,
)

MAX_BUNDLE_ARTIFACTS = 64
MAX_BUNDLE_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_BUNDLE_OUTPUT_BYTES = (
    MAX_BUNDLE_ARCHIVE_BYTES + 2 * MAX_EVIDENCE_BYTES + 1024 * 1024
)
BUNDLE_SCHEMA_VERSION = "1.1"
BUNDLE_KIND = "nvidia-converge-release-evidence-bundle"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ReleaseEvidenceBundleError(RuntimeError):
    pass


def _write_release_evidence_bundle(
    evidence: dict[str, Any],
    output: Path,
    *,
    verified_artifacts: dict[int, bytes],
) -> None:
    """Canonicalize already-verified artifacts into one durable release ZIP."""

    if output.exists() or output.is_symlink():
        raise ReleaseEvidenceBundleError(
            f"release evidence bundle target already exists: {output}"
        )
    expected_artifact_ids = _expected_artifact_ids(evidence)
    if set(verified_artifacts) != expected_artifact_ids:
        raise ReleaseEvidenceBundleError(
            "verified artifact set does not exactly match the release evidence"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    archive_count = 0
    total_archive_bytes = 0
    checksums: list[tuple[str, str]] = []
    manifest = (
        json.dumps(evidence, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    checksums.append((hashlib.sha256(manifest).hexdigest(), "integration-results.json"))
    retained_artifacts: list[dict[str, Any]] = []
    published = False
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w+b") as destination:
            descriptor = -1
            with zipfile.ZipFile(
                destination,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as bundle:
                bundle.writestr(_zip_info("integration-results.json"), manifest)
                seen_paths: set[str] = set()
                for run in evidence.get("runs", []):
                    if not isinstance(run, dict):
                        raise ReleaseEvidenceBundleError(
                            "evidence run must be an object"
                        )
                    matrix_id = run.get("matrix_id")
                    if not isinstance(
                        matrix_id, str
                    ) or not _SAFE_COMPONENT.fullmatch(matrix_id):
                        raise ReleaseEvidenceBundleError(
                            f"unsafe matrix identifier in release evidence: {matrix_id!r}"
                        )
                    artifacts = run.get("artifacts")
                    if not isinstance(artifacts, list):
                        raise ReleaseEvidenceBundleError(
                            f"run {matrix_id!r} has invalid artifact metadata"
                        )
                    for artifact in artifacts:
                        archive_count += 1
                        if archive_count > MAX_BUNDLE_ARTIFACTS:
                            raise ReleaseEvidenceBundleError(
                                "release evidence contains too many artifacts"
                            )
                        artifact_id, artifact_name, source_digest = (
                            _artifact_identity(matrix_id, artifact)
                        )
                        member = (
                            f"artifacts/{matrix_id}/{artifact_id}-{artifact_name}.zip"
                        )
                        if member in seen_paths:
                            raise ReleaseEvidenceBundleError(
                                f"duplicate release evidence member: {member}"
                            )
                        seen_paths.add(member)
                        source = verified_artifacts[artifact_id]
                        if not source or len(source) > MAX_ARTIFACT_ARCHIVE_BYTES:
                            raise ReleaseEvidenceBundleError(
                                f"artifact {artifact_name!r} exceeds the safety limit"
                            )
                        actual_source_digest = hashlib.sha256(source).hexdigest()
                        if not hmac.compare_digest(
                            actual_source_digest, source_digest
                        ):
                            raise ReleaseEvidenceBundleError(
                                f"artifact {artifact_name!r} source digest does not match verified evidence"
                            )
                        canonical = _canonical_artifact_archive(source)
                        if len(canonical) > MAX_ARTIFACT_ARCHIVE_BYTES:
                            raise ReleaseEvidenceBundleError(
                                f"canonical artifact {artifact_name!r} exceeds the safety limit"
                            )
                        retained_digest = hashlib.sha256(canonical).hexdigest()
                        total_archive_bytes += len(canonical)
                        if total_archive_bytes > MAX_BUNDLE_ARCHIVE_BYTES:
                            raise ReleaseEvidenceBundleError(
                                "release evidence bundle exceeds the safety limit"
                            )
                        bundle.writestr(_zip_info(member), canonical)
                        checksums.append((retained_digest, member))
                        retained_artifacts.append(
                            {
                                "matrix_id": matrix_id,
                                "artifact_id": artifact_id,
                                "artifact_name": artifact_name,
                                "source_sha256": source_digest,
                                "retained_path": member,
                                "retained_sha256": retained_digest,
                            }
                        )
                if archive_count == 0:
                    raise ReleaseEvidenceBundleError(
                        "release evidence contains no retained artifacts"
                    )
                bundle_manifest = _json_bytes(
                    {
                        "schema_version": BUNDLE_SCHEMA_VERSION,
                        "kind": BUNDLE_KIND,
                        "repository": evidence.get("repository"),
                        "release": evidence.get("release"),
                        "commit": evidence.get("commit"),
                        "qualification_wheel": evidence.get(
                            "qualification_wheel"
                        ),
                        "artifacts": retained_artifacts,
                    }
                )
                bundle.writestr(
                    _zip_info("bundle-manifest.json"), bundle_manifest
                )
                checksums.append(
                    (
                        hashlib.sha256(bundle_manifest).hexdigest(),
                        "bundle-manifest.json",
                    )
                )
                checksum_payload = "".join(
                    f"{digest}  {member}\n" for digest, member in checksums
                ).encode("ascii")
                bundle.writestr(_zip_info("SHA256SUMS"), checksum_payload)
            destination.flush()
            os.fsync(destination.fileno())
            size = os.fstat(destination.fileno()).st_size
            if size > MAX_BUNDLE_OUTPUT_BYTES:
                raise ReleaseEvidenceBundleError(
                    "release evidence bundle exceeds the safety limit"
                )
            source_stat = os.fstat(destination.fileno())
            path_stat = os.stat(temporary, follow_symlinks=False)
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or (path_stat.st_dev, path_stat.st_ino)
                != (source_stat.st_dev, source_stat.st_ino)
            ):
                raise ReleaseEvidenceBundleError(
                    "temporary release evidence bundle changed during creation"
                )
            try:
                os.link(temporary, output, follow_symlinks=False)
            except FileExistsError as exc:
                raise ReleaseEvidenceBundleError(
                    f"release evidence bundle target already exists: {output}"
                ) from exc
            published_stat = os.stat(output, follow_symlinks=False)
            if (
                not stat.S_ISREG(published_stat.st_mode)
                or (published_stat.st_dev, published_stat.st_ino)
                != (source_stat.st_dev, source_stat.st_ino)
            ):
                output.unlink(missing_ok=True)
                raise ReleaseEvidenceBundleError(
                    "release evidence bundle publication was replaced"
                )
            published = True
            temporary.unlink()
            fsync_directory(output.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if published:
            output.unlink(missing_ok=True)
        raise


def _expected_artifact_ids(evidence: dict[str, Any]) -> set[int]:
    artifact_ids: list[int] = []
    for run in evidence.get("runs", []):
        if not isinstance(run, dict) or not isinstance(run.get("artifacts"), list):
            raise ReleaseEvidenceBundleError("evidence has invalid artifact metadata")
        for artifact in run["artifacts"]:
            if not isinstance(artifact, dict):
                raise ReleaseEvidenceBundleError("evidence has a malformed artifact")
            artifact_id = artifact.get("id")
            if (
                not isinstance(artifact_id, int)
                or isinstance(artifact_id, bool)
                or artifact_id <= 0
                or artifact_id in artifact_ids
            ):
                raise ReleaseEvidenceBundleError(
                    "evidence has an invalid or duplicate artifact ID"
                )
            artifact_ids.append(artifact_id)
    return set(artifact_ids)


def _artifact_identity(
    matrix_id: str, artifact: Any
) -> tuple[int, str, str]:
    if not isinstance(artifact, dict):
        raise ReleaseEvidenceBundleError(
            f"run {matrix_id!r} has a malformed artifact"
        )
    artifact_id = artifact.get("id")
    artifact_name = artifact.get("name")
    digest = artifact.get("sha256")
    if (
        not isinstance(artifact_id, int)
        or isinstance(artifact_id, bool)
        or artifact_id <= 0
        or not isinstance(artifact_name, str)
        or not _SAFE_COMPONENT.fullmatch(artifact_name)
        or not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest) is None
    ):
        raise ReleaseEvidenceBundleError(
            f"run {matrix_id!r} has invalid artifact identity"
        )
    return artifact_id, artifact_name, digest


def _canonical_artifact_archive(source: bytes) -> bytes:
    try:
        members = _artifact_evidence_members(source)
    except ValueError as exc:
        raise ReleaseEvidenceBundleError(
            f"verified artifact cannot be canonicalized: {exc}"
        ) from exc
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name])
    return destination.getvalue()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bundle validated sanitized GPU evidence for durable release retention."
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", EXPECTED_REPOSITORY),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--qualification-wheel",
        type=Path,
        required=True,
        help="release wheel rebuilt from the tested commit and bound to GPU qualification",
    )
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--github-api-url",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    args = parser.parse_args(argv)

    evidence_path = Path(args.evidence)
    try:
        evidence = json.loads(
            read_bounded_utf8(evidence_path, max_bytes=MAX_EVIDENCE_BYTES),
            object_pairs_hook=_object_without_duplicates,
        )
        jsonschema.validate(
            evidence,
            load_schema("integration-results"),
            format_checker=strict_format_checker(),
        )
    except (
        OSError,
        BoundedFileError,
        json.JSONDecodeError,
        jsonschema.ValidationError,
        ValueError,
        RecursionError,
    ) as exc:
        print(f"release evidence bundle failed: {exc}", file=sys.stderr)
        return 2
    if not isinstance(evidence, dict):
        print("release evidence bundle failed: evidence must be an object", file=sys.stderr)
        return 2

    errors = check_evidence(
        evidence,
        release=args.release,
        expected_repository=args.repository,
    )
    tested_commit = evidence.get("commit")
    if isinstance(tested_commit, str) and re.fullmatch(
        r"[0-9a-f]{40}", tested_commit
    ):
        errors.extend(
            check_commit_provenance(tested_commit, args.commit, evidence_path)
        )
    if errors:
        for error in errors:
            print(f"release evidence bundle failed: {error}", file=sys.stderr)
        return 2
    errors.extend(
        check_qualification_wheel_binding(evidence, args.qualification_wheel)
    )
    if errors:
        for error in errors:
            print(f"release evidence bundle failed: {error}", file=sys.stderr)
        return 2
    verified_artifacts: dict[int, bytes] = {}
    errors.extend(
        verify_github_evidence(
            evidence,
            token=args.github_token,
            api_url=args.github_api_url,
            verified_artifacts=verified_artifacts,
        )
    )
    if errors:
        for error in errors:
            print(f"release evidence bundle failed: {error}", file=sys.stderr)
        return 2
    try:
        _write_release_evidence_bundle(
            evidence,
            Path(args.output),
            verified_artifacts=verified_artifacts,
        )
    except (OSError, ReleaseEvidenceBundleError, zipfile.BadZipFile) as exc:
        print(f"release evidence bundle failed: {exc}", file=sys.stderr)
        return 2
    print(f"release evidence bundle written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
