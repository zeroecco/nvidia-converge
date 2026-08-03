from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import jsonschema  # type: ignore[import-untyped]

from nvidia_converge.schemas import load_schema, strict_format_checker
from scripts.bundle_release_evidence import (
    BUNDLE_KIND,
    BUNDLE_SCHEMA_VERSION,
    MAX_BUNDLE_ARTIFACTS,
    MAX_BUNDLE_OUTPUT_BYTES,
)
from scripts.check_release_evidence import (
    EXPECTED_REPOSITORY,
    MAX_ARTIFACT_ARCHIVE_BYTES,
    MAX_ARTIFACT_ENTRIES,
    MAX_ARTIFACT_MEMBER_BYTES,
    MAX_ARTIFACT_MEMBER_NAME_BYTES,
    MAX_ARTIFACT_TOTAL_UNCOMPRESSED_BYTES,
    MAX_EVIDENCE_BYTES,
    ArtifactEvidenceError,
    _artifact_evidence_members,
    _object_without_duplicates,
    _verify_run_artifact_contents,
    check_evidence,
    check_qualification_wheel_binding,
)

MAX_BUNDLE_MANIFEST_BYTES = 256 * 1024
MAX_CHECKSUM_BYTES = 64 * 1024
MAX_OUTER_MEMBER_NAME_BYTES = 512
MAX_CANONICAL_ARTIFACT_BYTES = MAX_ARTIFACT_ARCHIVE_BYTES
_CANONICAL_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_CANONICAL_MODE = 0o100644 << 16
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ARTIFACT_MEMBER = re.compile(
    r"^artifacts/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/"
    r"([1-9][0-9]*)-([A-Za-z0-9][A-Za-z0-9._-]{0,127})\.zip$"
)
_BUNDLE_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "repository",
    "release",
    "commit",
    "qualification_wheel",
    "artifacts",
}
_BUNDLE_ARTIFACT_KEYS = {
    "matrix_id",
    "artifact_id",
    "artifact_name",
    "source_sha256",
    "retained_path",
    "retained_sha256",
}


class ReleaseEvidenceBundleVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactBinding:
    run: dict[str, Any]
    matrix_id: str
    artifact_id: int
    artifact_name: str
    source_sha256: str
    retained_path: str
    retained_sha256: str


def verify_release_evidence_bundle(
    bundle_path: Path,
    *,
    release: str,
    repository: str,
    qualification_wheel_path: Path | None = None,
) -> list[str]:
    """Verify a retained release evidence bundle without network access."""

    try:
        with _open_regular_bundle(bundle_path) as source:
            return _verify_open_bundle(
                source,
                release=release,
                repository=repository,
                qualification_wheel_path=qualification_wheel_path,
            )
    except (
        OSError,
        ReleaseEvidenceBundleVerificationError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        RuntimeError,
    ) as exc:
        return [str(exc)]


def _verify_open_bundle(
    source: BinaryIO,
    *,
    release: str,
    repository: str,
    qualification_wheel_path: Path | None,
) -> list[str]:
    archive_size = os.fstat(source.fileno()).st_size
    try:
        with zipfile.ZipFile(source, "r", allowZip64=True) as bundle:
            entries = bundle.infolist()
            _validate_outer_entries(source, entries, archive_size)
            indexed = {entry.filename: entry for entry in entries}

            evidence_payload = _read_member(
                bundle,
                indexed["integration-results.json"],
                MAX_EVIDENCE_BYTES,
            )
            evidence = _strict_json_object(
                evidence_payload,
                "integration-results.json",
            )
            _require_canonical_json(
                evidence_payload,
                evidence,
                "integration-results.json",
            )
            try:
                jsonschema.validate(
                    evidence,
                    load_schema("integration-results"),
                    format_checker=strict_format_checker(),
                )
            except jsonschema.ValidationError as exc:
                raise ReleaseEvidenceBundleVerificationError(
                    f"integration-results.json does not match the current schema: {exc.message}"
                ) from exc

            expected = _expected_artifacts(evidence)
            expected_paths = [binding.retained_path for binding in expected]
            expected_order = [
                "integration-results.json",
                *expected_paths,
                "bundle-manifest.json",
                "SHA256SUMS",
            ]
            actual_order = [entry.filename for entry in entries]
            if actual_order != expected_order:
                raise ReleaseEvidenceBundleVerificationError(
                    "bundle member inventory or order does not exactly match integration-results.json"
                )

            manifest_payload = _read_member(
                bundle,
                indexed["bundle-manifest.json"],
                MAX_BUNDLE_MANIFEST_BYTES,
            )
            manifest = _strict_json_object(
                manifest_payload,
                "bundle-manifest.json",
            )
            _require_canonical_json(
                manifest_payload,
                manifest,
                "bundle-manifest.json",
            )
            bound_artifacts = _bind_bundle_manifest(
                manifest,
                evidence,
                expected,
                release=release,
                repository=repository,
            )

            checksum_payload = _read_member(
                bundle,
                indexed["SHA256SUMS"],
                MAX_CHECKSUM_BYTES,
            )
            checksums = _parse_sha256sums(checksum_payload)
            checksum_order = [
                "integration-results.json",
                *expected_paths,
                "bundle-manifest.json",
            ]
            declared_digests = {
                "integration-results.json": hashlib.sha256(
                    evidence_payload
                ).hexdigest(),
                **{
                    binding.retained_path: binding.retained_sha256
                    for binding in bound_artifacts
                },
                "bundle-manifest.json": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
            }
            if [path for path, _digest in checksums] != checksum_order:
                raise ReleaseEvidenceBundleVerificationError(
                    "SHA256SUMS inventory or order does not exactly cover the retained members"
                )
            for path, digest in checksums:
                expected_digest = declared_digests.get(path)
                if expected_digest is None or not hmac.compare_digest(
                    digest, expected_digest
                ):
                    raise ReleaseEvidenceBundleVerificationError(
                        f"SHA256SUMS digest does not match the bundle manifest for {path!r}"
                    )
            canonical_checksums = "".join(
                f"{declared_digests[path]}  {path}\n" for path in checksum_order
            ).encode("ascii")
            if checksum_payload != canonical_checksums:
                raise ReleaseEvidenceBundleVerificationError(
                    "SHA256SUMS is not in the canonical format"
                )

            errors = check_evidence(
                evidence,
                release=release,
                expected_repository=repository,
            )
            if qualification_wheel_path is not None:
                errors.extend(
                    check_qualification_wheel_binding(
                        evidence, qualification_wheel_path
                    )
                )
            seen_operation_ids: dict[str, str] = {}
            bindings_by_run: dict[str, list[ArtifactBinding]] = {}
            for binding in bound_artifacts:
                run_id = str(binding.run.get("id"))
                bindings_by_run.setdefault(run_id, []).append(binding)
            for run in evidence.get("runs", []):
                if not isinstance(run, dict):
                    continue
                retained: list[tuple[str, bytes]] = []
                for binding in bindings_by_run.get(str(run.get("id")), []):
                    payload = _read_member(
                        bundle,
                        indexed[binding.retained_path],
                        MAX_CANONICAL_ARTIFACT_BYTES,
                    )
                    actual_digest = hashlib.sha256(payload).hexdigest()
                    if not hmac.compare_digest(
                        actual_digest, binding.retained_sha256
                    ):
                        errors.append(
                            f"retained artifact {binding.artifact_name!r} digest does not match its bundle manifest"
                        )
                        continue
                    try:
                        _validate_canonical_inner_archive(payload)
                    except ReleaseEvidenceBundleVerificationError as exc:
                        errors.append(
                            f"retained artifact {binding.artifact_name!r} is not canonical: {exc}"
                        )
                        continue
                    retained.append((binding.artifact_name, payload))
                errors.extend(
                    _verify_run_artifact_contents(
                        run,
                        retained,
                        seen_operation_ids=seen_operation_ids,
                    )
                )
            return errors
    except KeyError as exc:
        raise ReleaseEvidenceBundleVerificationError(
            f"bundle is missing required member {exc.args[0]!r}"
        ) from exc
    except (NotImplementedError, ArtifactEvidenceError) as exc:
        raise ReleaseEvidenceBundleVerificationError(str(exc)) from exc


def _open_regular_bundle(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseEvidenceBundleVerificationError(
                "bundle path is not a regular file"
            )
        if metadata.st_size <= 0 or metadata.st_size > MAX_BUNDLE_OUTPUT_BYTES:
            raise ReleaseEvidenceBundleVerificationError(
                "bundle archive size is outside the safety limit"
            )
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _validate_outer_entries(
    source: BinaryIO,
    entries: list[zipfile.ZipInfo],
    archive_size: int,
) -> None:
    if not entries or len(entries) > MAX_BUNDLE_ARTIFACTS + 3:
        raise ReleaseEvidenceBundleVerificationError(
            "bundle entry count is outside the safety limit"
        )
    names: set[str] = set()
    total_size = 0
    for entry in entries:
        _validate_safe_path(entry.filename, MAX_OUTER_MEMBER_NAME_BYTES)
        if entry.filename in names:
            raise ReleaseEvidenceBundleVerificationError(
                f"bundle contains duplicate path {entry.filename!r}"
            )
        names.add(entry.filename)
        if not (
            entry.filename
            in {
                "integration-results.json",
                "bundle-manifest.json",
                "SHA256SUMS",
            }
            or _ARTIFACT_MEMBER.fullmatch(entry.filename)
        ):
            raise ReleaseEvidenceBundleVerificationError(
                f"bundle contains an unexpected member name {entry.filename!r}"
            )
        member_limit = _outer_member_limit(entry.filename)
        if entry.file_size > member_limit:
            raise ReleaseEvidenceBundleVerificationError(
                f"bundle member {entry.filename!r} exceeds the safety limit"
            )
        total_size += entry.file_size
        if total_size > MAX_BUNDLE_OUTPUT_BYTES:
            raise ReleaseEvidenceBundleVerificationError(
                "bundle member sizes exceed the safety limit"
            )
    _validate_canonical_zip_layout(source, entries, archive_size)


def _outer_member_limit(name: str) -> int:
    if name == "integration-results.json":
        return MAX_EVIDENCE_BYTES
    if name == "bundle-manifest.json":
        return MAX_BUNDLE_MANIFEST_BYTES
    if name == "SHA256SUMS":
        return MAX_CHECKSUM_BYTES
    return MAX_CANONICAL_ARTIFACT_BYTES


def _validate_canonical_inner_archive(payload: bytes) -> None:
    if not payload or len(payload) > MAX_CANONICAL_ARTIFACT_BYTES:
        raise ReleaseEvidenceBundleVerificationError(
            "archive size is outside the safety limit"
        )
    import io

    source = io.BytesIO(payload)
    try:
        with zipfile.ZipFile(source, "r", allowZip64=True) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_ARTIFACT_ENTRIES:
                raise ReleaseEvidenceBundleVerificationError(
                    "archive entry count is outside the safety limit"
                )
            names = [entry.filename for entry in entries]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ReleaseEvidenceBundleVerificationError(
                    "archive members are not uniquely sorted"
                )
            total_size = 0
            for entry in entries:
                _validate_safe_path(
                    entry.filename,
                    MAX_ARTIFACT_MEMBER_NAME_BYTES,
                )
                if entry.file_size > MAX_ARTIFACT_MEMBER_BYTES:
                    raise ReleaseEvidenceBundleVerificationError(
                        f"archive member {entry.filename!r} exceeds the safety limit"
                    )
                total_size += entry.file_size
                if total_size > MAX_ARTIFACT_TOTAL_UNCOMPRESSED_BYTES:
                    raise ReleaseEvidenceBundleVerificationError(
                        "archive member sizes exceed the safety limit"
                    )
            _validate_canonical_zip_layout(source, entries, len(payload))
        _artifact_evidence_members(payload)
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        NotImplementedError,
        RuntimeError,
        ArtifactEvidenceError,
    ) as exc:
        raise ReleaseEvidenceBundleVerificationError(str(exc)) from exc


def _validate_canonical_zip_layout(
    source: BinaryIO,
    entries: list[zipfile.ZipInfo],
    archive_size: int,
) -> None:
    local_offset = 0
    encoded_names: list[bytes] = []
    for entry in entries:
        name = _canonical_entry_name(entry)
        encoded_names.append(name)
        if entry.header_offset != local_offset:
            raise ReleaseEvidenceBundleVerificationError(
                f"ZIP member {entry.filename!r} has a noncanonical offset"
            )
        expected_header = struct.pack(
            "<4s5H3L2H",
            b"PK\x03\x04",
            20,
            0,
            zipfile.ZIP_STORED,
            0,
            33,
            entry.CRC,
            entry.file_size,
            entry.file_size,
            len(name),
            0,
        ) + name
        if _read_exact_at(source, local_offset, len(expected_header)) != expected_header:
            raise ReleaseEvidenceBundleVerificationError(
                f"ZIP member {entry.filename!r} has noncanonical local metadata"
            )
        local_offset += len(expected_header) + entry.file_size

    central_offset = local_offset
    central_size = 0
    for entry, name in zip(entries, encoded_names):
        expected_header = struct.pack(
            "<4s6H3L5H2L",
            b"PK\x01\x02",
            (3 << 8) | 20,
            20,
            0,
            zipfile.ZIP_STORED,
            0,
            33,
            entry.CRC,
            entry.file_size,
            entry.file_size,
            len(name),
            0,
            0,
            0,
            0,
            _CANONICAL_MODE,
            entry.header_offset,
        ) + name
        offset = central_offset + central_size
        if _read_exact_at(source, offset, len(expected_header)) != expected_header:
            raise ReleaseEvidenceBundleVerificationError(
                f"ZIP member {entry.filename!r} has noncanonical central metadata"
            )
        central_size += len(expected_header)

    expected_eocd = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        len(entries),
        len(entries),
        central_size,
        central_offset,
        0,
    )
    eocd_offset = central_offset + central_size
    if _read_exact_at(source, eocd_offset, len(expected_eocd)) != expected_eocd:
        raise ReleaseEvidenceBundleVerificationError(
            "ZIP end record is not canonical"
        )
    if eocd_offset + len(expected_eocd) != archive_size:
        raise ReleaseEvidenceBundleVerificationError(
            "ZIP contains a preamble, gap, comment, or trailing bytes"
        )


def _canonical_entry_name(entry: zipfile.ZipInfo) -> bytes:
    if entry.orig_filename != entry.filename:
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} has an ambiguous name"
        )
    try:
        name = entry.filename.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} does not have a canonical ASCII name"
        ) from exc
    if entry.flag_bits & 0x1:
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} is encrypted"
        )
    if entry.compress_type != zipfile.ZIP_STORED:
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} uses unsupported compression"
        )
    file_type = stat.S_IFMT(entry.external_attr >> 16)
    if file_type == stat.S_IFLNK:
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} is a symbolic link"
        )
    if file_type != stat.S_IFREG or entry.is_dir():
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} is not a regular file"
        )
    if (
        entry.date_time != _CANONICAL_TIMESTAMP
        or entry.comment
        or entry.extra
        or entry.create_system != 3
        or entry.create_version != 20
        or entry.extract_version != 20
        or entry.reserved != 0
        or entry.flag_bits != 0
        or entry.volume != 0
        or entry.internal_attr != 0
        or entry.external_attr != _CANONICAL_MODE
        or entry.compress_size != entry.file_size
    ):
        raise ReleaseEvidenceBundleVerificationError(
            f"ZIP member {entry.filename!r} is not a canonical STORED regular file"
        )
    return name


def _read_exact_at(source: BinaryIO, offset: int, size: int) -> bytes:
    source.seek(offset)
    payload = source.read(size)
    if len(payload) != size:
        raise ReleaseEvidenceBundleVerificationError(
            "ZIP metadata is truncated"
        )
    return payload


def _read_member(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    limit: int,
) -> bytes:
    if entry.file_size < 0 or entry.file_size > limit:
        raise ReleaseEvidenceBundleVerificationError(
            f"bundle member {entry.filename!r} exceeds the safety limit"
        )
    try:
        with archive.open(entry, "r") as source:
            payload = source.read(limit + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ReleaseEvidenceBundleVerificationError(
            f"cannot read bundle member {entry.filename!r}: {exc}"
        ) from exc
    if len(payload) > limit or len(payload) != entry.file_size:
        raise ReleaseEvidenceBundleVerificationError(
            f"bundle member {entry.filename!r} has an invalid expanded size"
        )
    return payload


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceBundleVerificationError(
            f"{label} is not valid UTF-8"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceBundleVerificationError(
            f"{label} is invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise ReleaseEvidenceBundleVerificationError(
            f"{label} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseEvidenceBundleVerificationError(
            f"{label} must contain a JSON object"
        )
    return value


def _require_canonical_json(
    payload: bytes,
    value: dict[str, Any],
    label: str,
) -> None:
    canonical = (
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise ReleaseEvidenceBundleVerificationError(
            f"{label} is not in the canonical JSON format"
        )


def _expected_artifacts(evidence: dict[str, Any]) -> list[ArtifactBinding]:
    result: list[ArtifactBinding] = []
    paths: set[str] = set()
    for run in evidence.get("runs", []):
        if not isinstance(run, dict):
            raise ReleaseEvidenceBundleVerificationError(
                "integration evidence contains a malformed run"
            )
        matrix_id = run.get("matrix_id")
        artifacts = run.get("artifacts")
        if (
            not isinstance(matrix_id, str)
            or not _SAFE_COMPONENT.fullmatch(matrix_id)
            or not isinstance(artifacts, list)
        ):
            raise ReleaseEvidenceBundleVerificationError(
                "integration evidence contains invalid artifact metadata"
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ReleaseEvidenceBundleVerificationError(
                    "integration evidence contains a malformed artifact"
                )
            artifact_id = artifact.get("id")
            artifact_name = artifact.get("name")
            source_sha256 = artifact.get("sha256")
            if (
                not isinstance(artifact_id, int)
                or isinstance(artifact_id, bool)
                or artifact_id <= 0
                or not isinstance(artifact_name, str)
                or not _SAFE_COMPONENT.fullmatch(artifact_name)
                or not isinstance(source_sha256, str)
                or not _SHA256.fullmatch(source_sha256)
            ):
                raise ReleaseEvidenceBundleVerificationError(
                    "integration evidence contains an invalid artifact identity"
                )
            path = f"artifacts/{matrix_id}/{artifact_id}-{artifact_name}.zip"
            if path in paths:
                raise ReleaseEvidenceBundleVerificationError(
                    f"integration evidence derives duplicate artifact path {path!r}"
                )
            paths.add(path)
            result.append(
                ArtifactBinding(
                    run=run,
                    matrix_id=matrix_id,
                    artifact_id=artifact_id,
                    artifact_name=artifact_name,
                    source_sha256=source_sha256,
                    retained_path=path,
                    retained_sha256="",
                )
            )
    if not result or len(result) > MAX_BUNDLE_ARTIFACTS:
        raise ReleaseEvidenceBundleVerificationError(
            "integration evidence artifact count is outside the safety limit"
        )
    return result


def _bind_bundle_manifest(
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    expected: list[ArtifactBinding],
    *,
    release: str,
    repository: str,
) -> list[ArtifactBinding]:
    if set(manifest) != _BUNDLE_MANIFEST_KEYS:
        raise ReleaseEvidenceBundleVerificationError(
            "bundle-manifest.json has unexpected or missing fields"
        )
    expected_header = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "repository": repository,
        "release": release,
        "commit": evidence.get("commit"),
        "qualification_wheel": evidence.get("qualification_wheel"),
    }
    mismatches = sorted(
        key for key, value in expected_header.items() if manifest.get(key) != value
    )
    if evidence.get("repository") != repository:
        mismatches.append("integration repository")
    if evidence.get("release") != release:
        mismatches.append("integration release")
    if mismatches:
        raise ReleaseEvidenceBundleVerificationError(
            "bundle manifest context mismatch: " + ", ".join(mismatches)
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(expected):
        raise ReleaseEvidenceBundleVerificationError(
            "bundle manifest artifact inventory does not match integration evidence"
        )
    result: list[ArtifactBinding] = []
    for index, (item, source) in enumerate(zip(artifacts, expected)):
        if not isinstance(item, dict) or set(item) != _BUNDLE_ARTIFACT_KEYS:
            raise ReleaseEvidenceBundleVerificationError(
                f"bundle manifest artifact entry {index} has unexpected or missing fields"
            )
        exact_values = {
            "matrix_id": source.matrix_id,
            "artifact_id": source.artifact_id,
            "artifact_name": source.artifact_name,
            "source_sha256": source.source_sha256,
            "retained_path": source.retained_path,
        }
        mismatched = sorted(
            key for key, value in exact_values.items() if item.get(key) != value
        )
        retained_sha256 = item.get("retained_sha256")
        if not isinstance(retained_sha256, str) or not _SHA256.fullmatch(
            retained_sha256
        ):
            mismatched.append("retained_sha256")
        if mismatched:
            raise ReleaseEvidenceBundleVerificationError(
                f"bundle manifest artifact entry {index} mismatch: "
                + ", ".join(mismatched)
            )
        assert isinstance(retained_sha256, str)
        result.append(
            ArtifactBinding(
                run=source.run,
                matrix_id=source.matrix_id,
                artifact_id=source.artifact_id,
                artifact_name=source.artifact_name,
                source_sha256=source.source_sha256,
                retained_path=source.retained_path,
                retained_sha256=retained_sha256,
            )
        )
    return result


def _parse_sha256sums(payload: bytes) -> list[tuple[str, str]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceBundleVerificationError(
            "SHA256SUMS is not ASCII"
        ) from exc
    if not text or not text.endswith("\n"):
        raise ReleaseEvidenceBundleVerificationError(
            "SHA256SUMS must be nonempty and newline-terminated"
        )
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None:
            raise ReleaseEvidenceBundleVerificationError(
                f"SHA256SUMS line {line_number} is malformed"
            )
        digest, path = match.groups()
        _validate_safe_path(path, MAX_OUTER_MEMBER_NAME_BYTES)
        if path == "SHA256SUMS":
            raise ReleaseEvidenceBundleVerificationError(
                "SHA256SUMS must not contain a self entry"
            )
        if path in seen:
            raise ReleaseEvidenceBundleVerificationError(
                f"SHA256SUMS contains duplicate path {path!r}"
            )
        seen.add(path)
        result.append((path, digest))
    return result


def _validate_safe_path(name: str, max_bytes: int) -> None:
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReleaseEvidenceBundleVerificationError(
            "ZIP path is not valid UTF-8"
        ) from exc
    parts = name.split("/")
    if (
        not name
        or len(encoded) > max_bytes
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or any(part in {"", ".", ".."} for part in parts)
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ReleaseEvidenceBundleVerificationError(
            f"unsafe ZIP path {name!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify release evidence without network access; does not query GitHub."
        ),
        epilog=(
            "This command validates retained checksums, canonical archives, GPU "
            "evidence semantics, and the downloaded release wheel's qualification "
            "binding. Workflow, job, and source-artifact API metadata are not "
            "independently re-proven."
        ),
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--qualification-wheel",
        type=Path,
        required=True,
        help="downloaded release wheel that must match the qualified GPU-tested bytes",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", EXPECTED_REPOSITORY),
    )
    args = parser.parse_args(argv)

    errors = verify_release_evidence_bundle(
        Path(args.bundle),
        release=args.release,
        repository=args.repository,
        qualification_wheel_path=args.qualification_wheel,
    )
    if errors:
        for error in errors:
            print(
                f"offline release evidence bundle is invalid: {error}",
                file=sys.stderr,
            )
        return 2
    print(
        f"release evidence bundle passed offline: {args.bundle}; "
        "qualification wheel bytes matched; GitHub API provenance was not re-queried"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
