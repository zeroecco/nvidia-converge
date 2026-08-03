from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from nvidia_converge.dnf_module_transaction import (
    DNF_MODULE_ENABLE_SCRIPT,
    dnf_module_enable_command,
    parse_dnf_module_enable_proof,
)
from nvidia_converge.models import CommandResult
from nvidia_converge.schemas import load_schema
from nvidia_converge.verify import _CUDA_DRIVER_PROBE_SUCCESS

ATTESTATION_SCHEMA_VERSION = "1.3"
ATTESTATION_KIND = "nvidia-converge-integration-attestation"
CUDA_DRIVER_API_ATTESTATION = "[verified:cuda-driver-api]"
DNF_MODULE_PROOF_ATTESTATION_PREFIX = "[verified:dnf-module-proof-v2:"
REDACTIONS = [
    "command-output",
    "finding-detail-and-evidence",
    "gpu-device-topology",
    "host-identity",
    "managed-file-content",
    "private-snapshot-path",
    "verification-detail",
]
_CANONICAL_REPORT_PATHS = {
    "reports/doctor.json",
    "reports/doctor-missing-headers.json",
    "reports/doctor-module-unloaded.json",
    "reports/doctor-driver-mismatch.json",
    "reports/doctor-runtime-missing.json",
    "reports/doctor-fabric-manager-inactive.json",
    "reports/plan.json",
    "reports/install.json",
    "reports/verify.json",
    "reports/lock.json",
    "reports/snapshot.json",
    "reports/rollback.json",
    "reports/policy-rollback.json",
}
_CANONICAL_JOURNAL_PATHS = {
    f"{path[:-5]}.journal.jsonl" for path in _CANONICAL_REPORT_PATHS
}
_CANONICAL_SOURCE_PATHS = {
    *_CANONICAL_REPORT_PATHS,
    *_CANONICAL_JOURNAL_PATHS,
    "restoration/doctor-missing-headers.json",
    "restoration/doctor-module-unloaded.json",
    "restoration/doctor-driver-mismatch.json",
    "restoration/doctor-runtime-missing.json",
    "restoration/doctor-fabric-manager-inactive.json",
    "restoration/final-cleanup.json",
    "pre/rollback-snapshot.json",
    "pre/policy-rollback-snapshot.json",
}
_QUALIFICATION_WHEEL_NAME = re.compile(
    r"^nvidia_converge-[0-9][0-9A-Za-z.]*-py3-none-any\.whl$"
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_OPERATION_ID = re.compile(r"^[a-f0-9]{32}$")
_ExportEntryIdentity = tuple[int, int, int, int, int, int, int, int]
MAX_EXPORT_ENTRIES = 512
MAX_EXPORT_MEMBER_BYTES = 32 * 1024 * 1024
MAX_EXPORT_TOTAL_BYTES = 256 * 1024 * 1024
_SNAPSHOT_CHECKSUM_SOURCES = {
    "pre/rollback-snapshot.sha256": "pre/rollback-snapshot.json",
    "pre/policy-rollback-snapshot.sha256": (
        "pre/policy-rollback-snapshot.json"
    ),
}
_CANONICAL_EXPORTED_PATHS = {
    *_CANONICAL_SOURCE_PATHS,
    *_SNAPSHOT_CHECKSUM_SOURCES,
}
_RFC3339_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)
_REPORT_SCHEMA = load_schema("report")
_ROLLBACK_SCHEMA = {"$ref": "#/$defs/rollback"}
_JOURNAL_COMMON_KEYS = {"event", "operation_id", "timestamp"}
_SOURCE_JOURNAL_EVENT_KEYS = {
    "operation-started": _JOURNAL_COMMON_KEYS,
    "rollback-snapshot-persisted": _JOURNAL_COMMON_KEYS
    | {
        "snapshot_path",
        "snapshot_integrity_sha256",
        "snapshot_operation_id",
        "snapshot_host_id",
    },
    "launcher-release-authorized": _JOURNAL_COMMON_KEYS
    | {
        "release_target",
        "snapshot_path",
        "snapshot_integrity_sha256",
        "snapshot_operation_id",
        "snapshot_host_id",
    },
    "command-started": _JOURNAL_COMMON_KEYS | {"command", "mutating"},
    "command-finished": _JOURNAL_COMMON_KEYS
    | {
        "command",
        "mutating",
        "returncode",
        "skipped",
        "reason",
    },
    "operation-completed": _JOURNAL_COMMON_KEYS
    | {"exit_code", "incomplete", "outcome"},
    "report-persistence-failed": _JOURNAL_COMMON_KEYS | {"error"},
    "operation-recovered": _JOURNAL_COMMON_KEYS
    | {
        "recovery_operation_id",
        "snapshot_path",
        "snapshot_integrity_sha256",
        "snapshot_operation_id",
        "snapshot_host_id",
    },
}


class AttestationExportError(ValueError):
    pass


def build_attestation(
    source_members: dict[str, bytes],
    *,
    matrix_id: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    qualification_wheel_name: str,
    qualification_wheel_sha256: str,
    pseudonym_key: bytes | None = None,
) -> dict[str, bytes]:
    _validate_context(
        matrix_id,
        workflow_run_id,
        workflow_run_attempt,
        qualification_wheel_name,
        qualification_wheel_sha256,
    )
    expected_source_paths = {
        path
        for path in source_members
        if _is_source_evidence_path(path)
    }
    if expected_source_paths != set(source_members):
        unexpected = sorted(set(source_members) - expected_source_paths)
        raise AttestationExportError(
            "source contains unsupported attestation path(s): " + ", ".join(unexpected)
        )
    if not source_members:
        raise AttestationExportError("source contains no integration evidence")

    if pseudonym_key is None:
        pseudonym_key = secrets.token_bytes(32)
    elif not isinstance(pseudonym_key, bytes) or len(pseudonym_key) != 32:
        raise AttestationExportError(
            "pseudonym key must be exactly 32 bytes"
        )
    output: dict[str, bytes] = {}
    sources: dict[str, str] = {}
    snapshot_bindings: dict[
        tuple[str, str, str, str],
        tuple[str, str, str, str],
    ] = {}
    payload_path_bindings: dict[str, str] = {}
    payload_root_bindings: dict[str, str] = {}

    # Sanitize structured documents first so journal snapshot authorities can
    # be rebound to the exact transformed snapshot digest, path, creator, and
    # host instead of retaining raw private values or losing those fields.
    for path in sorted(source_members):
        if path.endswith(".journal.jsonl"):
            continue
        payload = source_members[path]
        document = _strict_json(payload, path)
        if not isinstance(document, dict):
            raise AttestationExportError(f"{path} is not a JSON object")
        standalone_snapshot = path in {
            "pre/rollback-snapshot.json",
            "pre/policy-rollback-snapshot.json",
        }
        document_schema = _ROLLBACK_SCHEMA if standalone_snapshot else _REPORT_SCHEMA
        _validate_evidence_document(
            document,
            document_schema,
            path=path,
            stage="source",
        )
        if standalone_snapshot:
            sanitized = _sanitize_snapshot(
                document,
                pseudonym_key,
                payload_path_bindings,
                payload_root_bindings,
            )
            _register_snapshot_binding(
                document,
                sanitized,
                snapshot_bindings,
                path,
            )
        else:
            sanitized = _sanitize_report(
                document,
                pseudonym_key,
                payload_path_bindings,
                payload_root_bindings,
            )
            source_snapshot = document.get("rollback")
            sanitized_snapshot = sanitized.get("rollback")
            if isinstance(source_snapshot, dict) and isinstance(
                sanitized_snapshot, dict
            ):
                _register_snapshot_binding(
                    source_snapshot,
                    sanitized_snapshot,
                    snapshot_bindings,
                    path,
                )
        _validate_evidence_document(
            sanitized,
            document_schema,
            path=path,
            stage="sanitized",
        )
        output[path] = _json_bytes(sanitized)
        sources[path] = path

    for path in sorted(source_members):
        if not path.endswith(".journal.jsonl"):
            continue
        output[path] = _sanitize_journal(
            source_members[path],
            path,
            pseudonym_key,
            snapshot_bindings,
            payload_path_bindings,
            payload_root_bindings,
        )
        sources[path] = path

    for snapshot_path in (
        "pre/rollback-snapshot.json",
        "pre/policy-rollback-snapshot.json",
    ):
        if snapshot_path not in output:
            continue
        snapshot_payload = output[snapshot_path]
        snapshot_digest = hashlib.sha256(snapshot_payload).hexdigest()
        checksum_path = snapshot_path.removesuffix(".json") + ".sha256"
        output[checksum_path] = f"{snapshot_digest}  {snapshot_path}\n".encode()
        sources[checksum_path] = snapshot_path

    files = []
    for path in sorted(output):
        source_path = sources[path]
        files.append(
            {
                "path": path,
                "source_path": source_path,
                "sha256": hashlib.sha256(output[path]).hexdigest(),
            }
        )
    manifest = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "matrix_id": matrix_id,
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
        "qualification_wheel": {
            "name": qualification_wheel_name,
            "sha256": qualification_wheel_sha256,
        },
        "redactions": REDACTIONS,
        "files": files,
    }
    output["attestation.json"] = _json_bytes(manifest)
    return output


class _EvidenceSchemaError(ValueError):
    pass


def _validate_evidence_document(
    document: dict[str, Any],
    schema: dict[str, Any],
    *,
    path: str,
    stage: str,
) -> None:
    try:
        _validate_schema_value(document, schema, _REPORT_SCHEMA, "$")
    except _EvidenceSchemaError as exc:
        raise AttestationExportError(
            f"{stage} {path} does not match the closed report schema: {exc}"
        ) from exc


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    location: str,
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise _EvidenceSchemaError(f"{location}: schema has a non-string $ref")
        _validate_schema_value(
            value,
            _resolve_local_schema_reference(reference, root_schema),
            root_schema,
            location,
        )
        return

    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of:
            raise _EvidenceSchemaError(f"{location}: schema has an invalid anyOf")
        branch_errors: list[str] = []
        for branch in any_of:
            if not isinstance(branch, dict):
                raise _EvidenceSchemaError(
                    f"{location}: schema has a non-object anyOf branch"
                )
            try:
                _validate_schema_value(value, branch, root_schema, location)
            except _EvidenceSchemaError as exc:
                branch_errors.append(str(exc))
            else:
                break
        else:
            detail = branch_errors[-1] if branch_errors else "no branch matched"
            raise _EvidenceSchemaError(f"{location}: no anyOf branch matched ({detail})")

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_json_type(value, expected_type):
        raise _EvidenceSchemaError(
            f"{location}: expected {_describe_json_types(expected_type)}, "
            f"got {_json_type_name(value)}"
        )

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise _EvidenceSchemaError(f"{location}: value does not match const")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not any(
            _json_equal(value, choice) for choice in choices
        ):
            raise _EvidenceSchemaError(f"{location}: value is not in enum")

    if isinstance(value, dict):
        _validate_schema_object(value, schema, root_schema, location)
    elif isinstance(value, list):
        _validate_schema_array(value, schema, root_schema, location)
    elif isinstance(value, str):
        _validate_schema_string(value, schema, location)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_schema_number(value, schema, location)

    all_of = schema.get("allOf")
    if all_of is not None:
        if not isinstance(all_of, list):
            raise _EvidenceSchemaError(f"{location}: schema has an invalid allOf")
        for branch in all_of:
            if not isinstance(branch, dict):
                raise _EvidenceSchemaError(
                    f"{location}: schema has a non-object allOf branch"
                )
            _validate_schema_value(value, branch, root_schema, location)

    condition = schema.get("if")
    consequent = schema.get("then")
    if isinstance(condition, dict) and isinstance(consequent, dict):
        try:
            _validate_schema_value(value, condition, root_schema, location)
        except _EvidenceSchemaError:
            pass
        else:
            _validate_schema_value(value, consequent, root_schema, location)


def _validate_schema_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    location: str,
) -> None:
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(key, str) for key in required
    ):
        raise _EvidenceSchemaError(f"{location}: schema has invalid required fields")
    missing = sorted(set(required) - set(value))
    if missing:
        raise _EvidenceSchemaError(
            f"{location}: missing required field(s): {', '.join(missing)}"
        )

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise _EvidenceSchemaError(f"{location}: schema has invalid properties")
    additional = schema.get("additionalProperties", True)
    unexpected = sorted(set(value) - set(properties))
    if additional is False and unexpected:
        raise _EvidenceSchemaError(
            f"{location}: unexpected field(s): {', '.join(unexpected)}"
        )
    for key, item in value.items():
        child_schema = properties.get(key)
        if child_schema is None:
            if isinstance(additional, dict):
                child_schema = additional
            else:
                continue
        if not isinstance(child_schema, dict):
            raise _EvidenceSchemaError(
                f"{location}: schema for field {key!r} is invalid"
            )
        _validate_schema_value(
            item,
            child_schema,
            root_schema,
            f"{location}.{key}",
        )


def _validate_schema_array(
    value: list[Any],
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    location: str,
) -> None:
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if isinstance(minimum, int) and len(value) < minimum:
        raise _EvidenceSchemaError(f"{location}: array is shorter than minItems")
    if isinstance(maximum, int) and len(value) > maximum:
        raise _EvidenceSchemaError(f"{location}: array is longer than maxItems")
    if schema.get("uniqueItems") is True:
        fingerprints = [_json_fingerprint(item) for item in value]
        if len(fingerprints) != len(set(fingerprints)):
            raise _EvidenceSchemaError(f"{location}: array items are not unique")
    item_schema = schema.get("items")
    if item_schema is None:
        return
    if not isinstance(item_schema, dict):
        raise _EvidenceSchemaError(f"{location}: schema has invalid items")
    for index, item in enumerate(value):
        _validate_schema_value(
            item,
            item_schema,
            root_schema,
            f"{location}[{index}]",
        )


def _validate_schema_string(
    value: str, schema: dict[str, Any], location: str
) -> None:
    minimum = schema.get("minLength")
    maximum = schema.get("maxLength")
    if isinstance(minimum, int) and len(value) < minimum:
        raise _EvidenceSchemaError(f"{location}: string is shorter than minLength")
    if isinstance(maximum, int) and len(value) > maximum:
        raise _EvidenceSchemaError(f"{location}: string is longer than maxLength")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        raise _EvidenceSchemaError(f"{location}: string does not match pattern")
    if schema.get("format") == "date-time" and not _is_rfc3339_date_time(value):
        raise _EvidenceSchemaError(f"{location}: string is not an RFC 3339 date-time")


def _validate_schema_number(
    value: float, schema: dict[str, Any], location: str
) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _EvidenceSchemaError(f"{location}: number is not finite")
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        raise _EvidenceSchemaError(f"{location}: number is below minimum")
    if isinstance(maximum, (int, float)) and value > maximum:
        raise _EvidenceSchemaError(f"{location}: number is above maximum")


def _resolve_local_schema_reference(
    reference: str, root_schema: dict[str, Any]
) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise _EvidenceSchemaError(f"unsupported non-local schema reference: {reference}")
    current: Any = root_schema
    for encoded_part in reference[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise _EvidenceSchemaError(f"unresolvable schema reference: {reference}")
        current = current[part]
    if not isinstance(current, dict):
        raise _EvidenceSchemaError(f"schema reference is not an object: {reference}")
    return current


def _matches_json_type(value: Any, expected: Any) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    if not names or not all(isinstance(name, str) for name in names):
        return False
    return any(
        {
            "null": value is None,
            "boolean": isinstance(value, bool),
            "integer": (
                isinstance(value, int) and not isinstance(value, bool)
            ) or (isinstance(value, float) and value.is_integer()),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "string": isinstance(value, str),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }.get(name, False)
        for name in names
    )


def _describe_json_types(expected: Any) -> str:
    names = expected if isinstance(expected, list) else [expected]
    return " or ".join(str(name) for name in names)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _json_equal(left: Any, right: Any) -> bool:
    return _json_fingerprint(left) == _json_fingerprint(right)


def _json_fingerprint(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_rfc3339_date_time(value: str) -> bool:
    if _RFC3339_DATE_TIME.fullmatch(value) is None:
        return False
    normalized = value[:10] + "T" + value[11:]
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def export_attestation(
    source_root: Path,
    output_root: Path,
    *,
    matrix_id: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    qualification_wheel_name: str,
    qualification_wheel_sha256: str,
) -> None:
    source_members = _read_source_members(source_root)
    exported = build_attestation(
        source_members,
        matrix_id=matrix_id,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        qualification_wheel_name=qualification_wheel_name,
        qualification_wheel_sha256=qualification_wheel_sha256,
    )
    directories = sorted(
        {
            Path(path).parent
            for path in exported
            if Path(path).parent != Path(".")
        },
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    _validate_export_payload_limits(exported, directories)
    _validate_export_parent(output_root)
    try:
        output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise AttestationExportError(
            f"output path already exists: {output_root}"
        ) from exc
    output_root.chmod(0o700)
    for relative in directories:
        destination = output_root / relative
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        destination.chmod(0o700)
    for path, payload in sorted(exported.items()):
        _write_export_member(output_root / path, payload)
    verify_export_directory(
        output_root,
        matrix_id=matrix_id,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        qualification_wheel_name=qualification_wheel_name,
        qualification_wheel_sha256=qualification_wheel_sha256,
    )


def verify_export_directory(
    output_root: Path,
    *,
    matrix_id: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    qualification_wheel_name: str,
    qualification_wheel_sha256: str,
) -> None:
    """Verify the exact sanitized directory immediately before artifact upload."""

    _validate_context(
        matrix_id,
        workflow_run_id,
        workflow_run_attempt,
        qualification_wheel_name,
        qualification_wheel_sha256,
    )
    _validate_export_parent(output_root)
    try:
        parent_before = output_root.parent.lstat()
        root_before = output_root.lstat()
    except OSError as exc:
        raise AttestationExportError(
            f"export directory is unavailable: {output_root}"
        ) from exc
    _validate_export_directory_metadata(output_root, root_before)
    file_identities, directory_identities = _scan_export_directory(
        output_root
    )
    files = set(file_identities)
    directories = set(directory_identities)
    if "attestation.json" not in files:
        raise AttestationExportError("export directory has no attestation manifest")
    manifest_payload = _read_export_member(
        output_root / "attestation.json",
        "attestation.json",
    )
    manifest = _strict_json(manifest_payload, "attestation.json")
    bindings = _validate_export_manifest(
        manifest,
        matrix_id=matrix_id,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        qualification_wheel_name=qualification_wheel_name,
        qualification_wheel_sha256=qualification_wheel_sha256,
    )
    expected_files = {"attestation.json", *bindings}
    if files != expected_files:
        missing = sorted(expected_files - files)
        unexpected = sorted(files - expected_files)
        raise AttestationExportError(
            "export directory inventory does not match its manifest"
            f" (missing={missing!r}, unexpected={unexpected!r})"
        )
    expected_directories = {
        "/".join(path.split("/")[:-1])
        for path in expected_files
        if "/" in path
    }
    if directories != expected_directories:
        missing = sorted(expected_directories - directories)
        unexpected = sorted(directories - expected_directories)
        raise AttestationExportError(
            "export directory tree does not match its manifest"
            f" (missing={missing!r}, unexpected={unexpected!r})"
        )
    for path, expected_digest in bindings.items():
        payload = _read_export_member(output_root / path, path)
        actual_digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise AttestationExportError(
                f"export member digest does not match the manifest: {path}"
            )
    final_file_identities, final_directory_identities = (
        _scan_export_directory(output_root)
    )
    if (
        file_identities != final_file_identities
        or directory_identities != final_directory_identities
    ):
        raise AttestationExportError(
            "export directory entries changed during verification"
        )
    _validate_export_parent(output_root)
    try:
        parent_after = output_root.parent.lstat()
        root_after = output_root.lstat()
    except OSError as exc:
        raise AttestationExportError(
            "export directory or its parent changed during verification"
        ) from exc
    _validate_export_directory_metadata(output_root, root_after)
    if (
        _directory_identity(parent_before) != _directory_identity(parent_after)
        or _file_identity(root_before) != _file_identity(root_after)
    ):
        raise AttestationExportError(
            "export directory or its parent changed during verification"
        )


def _validate_export_parent(output_root: Path) -> None:
    if output_root.name in {"", ".", ".."}:
        raise AttestationExportError("output path has no safe directory name")
    parent = output_root.parent
    try:
        metadata = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise AttestationExportError(
            f"output parent is unavailable: {parent}"
        ) from exc
    canonical = Path(os.path.abspath(parent))
    if resolved != canonical:
        raise AttestationExportError(
            f"output parent has a symlinked or noncanonical path: {parent}"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or mode & 0o022
    ):
        raise AttestationExportError(
            "output parent must be a real current-user-owned directory that is "
            "not group/world writable"
        )


def _write_export_member(destination: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while creating export member")
            remaining = remaining[written:]
    finally:
        os.close(descriptor)


def _validate_export_payload_limits(
    exported: dict[str, bytes], directories: list[Path]
) -> None:
    if len(exported) + len(directories) > MAX_EXPORT_ENTRIES:
        raise AttestationExportError(
            "export directory entry count exceeds the safety limit"
        )
    total_size = 0
    for path, payload in exported.items():
        if not 0 < len(payload) <= MAX_EXPORT_MEMBER_BYTES:
            raise AttestationExportError(
                f"export member size is outside the safety limit: {path}"
            )
        total_size += len(payload)
    if total_size > MAX_EXPORT_TOTAL_BYTES:
        raise AttestationExportError(
            "export directory size exceeds the safety limit"
        )


def _scan_export_directory(
    output_root: Path,
) -> tuple[
    dict[str, _ExportEntryIdentity],
    dict[str, _ExportEntryIdentity],
]:
    try:
        root_metadata = output_root.lstat()
    except OSError as exc:
        raise AttestationExportError(
            f"export directory is unavailable: {output_root}"
        ) from exc
    _validate_export_directory_metadata(output_root, root_metadata)
    files: dict[str, _ExportEntryIdentity] = {}
    directories: dict[str, _ExportEntryIdentity] = {}
    total_size = 0
    entries = 0

    def raise_walk_error(error: OSError) -> None:
        raise AttestationExportError(
            f"cannot traverse export directory: {error.filename or output_root}"
        ) from error

    for current, directory_names, file_names in os.walk(
        output_root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            relative = path.relative_to(output_root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise AttestationExportError(
                    f"cannot inspect export directory {relative!r}"
                ) from exc
            _validate_export_directory_metadata(path, metadata)
            directories[relative] = _file_identity(metadata)
            entries += 1
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(output_root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise AttestationExportError(
                    f"cannot inspect export member {relative!r}"
                ) from exc
            _validate_export_file_metadata(path, metadata)
            if not 0 < metadata.st_size <= MAX_EXPORT_MEMBER_BYTES:
                raise AttestationExportError(
                    f"export member size is outside the safety limit: {relative}"
                )
            total_size += metadata.st_size
            files[relative] = _file_identity(metadata)
            entries += 1
        if entries > MAX_EXPORT_ENTRIES:
            raise AttestationExportError(
                "export directory entry count exceeds the safety limit"
            )
        if total_size > MAX_EXPORT_TOTAL_BYTES:
            raise AttestationExportError(
                "export directory size exceeds the safety limit"
            )
    return files, directories


def _validate_export_directory_metadata(
    path: Path, metadata: os.stat_result
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise AttestationExportError(
            f"export directory has unsafe type, owner, or mode: {path}"
        )


def _validate_export_file_metadata(path: Path, metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise AttestationExportError(
            f"export member has unsafe type, owner, mode, or link count: {path}"
        )


def _read_export_member(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        _validate_export_file_metadata(path, before)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AttestationExportError(
            f"cannot safely open export member {label!r}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise AttestationExportError(
                f"export member changed while being opened: {label}"
            )
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_EXPORT_MEMBER_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_EXPORT_MEMBER_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if total > MAX_EXPORT_MEMBER_BYTES:
        raise AttestationExportError(
            f"export member exceeds the safety limit: {label}"
        )
    if _file_identity(opened) != _file_identity(after) or total != after.st_size:
        raise AttestationExportError(
            f"export member changed while being read: {label}"
        )
    return b"".join(chunks)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
    )


def _file_identity(
    metadata: os.stat_result,
) -> _ExportEntryIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_export_manifest(
    value: Any,
    *,
    matrix_id: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    qualification_wheel_name: str,
    qualification_wheel_sha256: str,
) -> dict[str, str]:
    expected_keys = {
        "schema_version",
        "kind",
        "matrix_id",
        "workflow_run_id",
        "workflow_run_attempt",
        "qualification_wheel",
        "redactions",
        "files",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise AttestationExportError(
            "attestation manifest does not have the exact expected shape"
        )
    qualification_wheel = value.get("qualification_wheel")
    if (
        value.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or value.get("kind") != ATTESTATION_KIND
        or value.get("matrix_id") != matrix_id
        or not isinstance(value.get("workflow_run_id"), int)
        or isinstance(value.get("workflow_run_id"), bool)
        or value.get("workflow_run_id") != workflow_run_id
        or not isinstance(value.get("workflow_run_attempt"), int)
        or isinstance(value.get("workflow_run_attempt"), bool)
        or value.get("workflow_run_attempt") != workflow_run_attempt
        or qualification_wheel
        != {
            "name": qualification_wheel_name,
            "sha256": qualification_wheel_sha256,
        }
        or value.get("redactions") != REDACTIONS
    ):
        raise AttestationExportError(
            "attestation manifest does not match the expected upload context"
        )
    entries = value.get("files")
    if (
        not isinstance(entries, list)
        or not entries
        or len(entries) > MAX_EXPORT_ENTRIES
    ):
        raise AttestationExportError(
            "attestation manifest has an invalid file inventory"
        )
    bindings: dict[str, str] = {}
    observed_paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "source_path",
            "sha256",
        }:
            raise AttestationExportError(
                "attestation manifest has an invalid file entry"
            )
        path = entry.get("path")
        source_path = entry.get("source_path")
        digest = entry.get("sha256")
        expected_source = (
            _SNAPSHOT_CHECKSUM_SOURCES.get(path)
            if isinstance(path, str)
            else None
        )
        if expected_source is None:
            expected_source = path
        if (
            not isinstance(path, str)
            or path not in _CANONICAL_EXPORTED_PATHS
            or path in bindings
            or not isinstance(source_path, str)
            or source_path != expected_source
            or source_path not in _CANONICAL_SOURCE_PATHS
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise AttestationExportError(
                "attestation manifest has a noncanonical or duplicate file entry"
            )
        bindings[path] = digest
        observed_paths.append(path)
    if observed_paths != sorted(observed_paths):
        raise AttestationExportError(
            "attestation manifest file inventory is not canonical"
        )
    for checksum_path, source_path in _SNAPSHOT_CHECKSUM_SOURCES.items():
        if (checksum_path in bindings) != (source_path in bindings):
            raise AttestationExportError(
                "attestation manifest has an incomplete snapshot checksum pair"
            )
    return bindings


def _read_source_members(source_root: Path) -> dict[str, bytes]:
    candidates = [
        *sorted((source_root / "reports").glob("*.json")),
        *sorted((source_root / "reports").glob("*.journal.jsonl")),
        *sorted((source_root / "restoration").glob("*.json")),
        source_root / "pre" / "rollback-snapshot.json",
        source_root / "pre" / "policy-rollback-snapshot.json",
    ]
    members: dict[str, bytes] = {}
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(source_root.resolve(strict=True)).as_posix()
        except ValueError as exc:
            raise AttestationExportError(
                f"source evidence escapes its root: {candidate}"
            ) from exc
        if relative in members:
            raise AttestationExportError(f"duplicate source evidence path: {relative}")
        members[relative] = resolved.read_bytes()
    return members


def _sanitize_report(
    report: dict[str, Any],
    pseudonym_key: bytes,
    payload_path_bindings: dict[str, str],
    payload_root_bindings: dict[str, str],
) -> dict[str, Any]:
    source_rollback = report.get("rollback")
    if isinstance(source_rollback, dict):
        _register_payload_path_bindings(
            source_rollback,
            pseudonym_key,
            payload_path_bindings,
            payload_root_bindings,
        )
    sanitized = copy.deepcopy(report)
    sanitized["host_id"] = _pseudonym(
        sanitized.get("host_id"), pseudonym_key, "host-id"
    )
    audit = sanitized.get("audit")
    if isinstance(audit, dict):
        audit["gpu_uuids"] = _pseudonymize_gpu_uuids(
            audit.get("gpu_uuids"), pseudonym_key
        )
        audit["mig_device_uuids"] = _pseudonymize_optional_gpu_uuids(
            audit.get("mig_device_uuids"), pseudonym_key
        )
        _pseudonymize_mig_geometry(
            audit.get("mig_geometry"), pseudonym_key
        )
        module = audit.get("module")
        if isinstance(module, dict):
            module["devices"] = []
        for key in (
            "nvidia_smi",
            "nvml",
            "package_inventory_result",
            "fabric_manager_health_result",
        ):
            _sanitize_command_result(
                audit.get(key),
                pseudonym_key,
                payload_path_bindings,
                payload_root_bindings,
            )
        for result in audit.get("mig_geometry_results") or []:
            _sanitize_command_result(
                result,
                pseudonym_key,
                payload_path_bindings,
                payload_root_bindings,
            )
        package_policy = audit.get("package_policy")
        if isinstance(package_policy, dict):
            _sanitize_command_result(
                package_policy.get("observation"),
                pseudonym_key,
                payload_path_bindings,
                payload_root_bindings,
            )
        for compatibility in audit.get("cuda_compatibility") or []:
            if isinstance(compatibility, dict):
                _sanitize_command_result(
                    compatibility.get("library_probe"),
                    pseudonym_key,
                    payload_path_bindings,
                    payload_root_bindings,
                )
    for finding in sanitized.get("findings") or []:
        if isinstance(finding, dict):
            finding["detail"] = "[redacted]"
            finding["evidence"] = {}
            if isinstance(finding.get("remediation"), str):
                finding["remediation"] = "[redacted]"
    for result in sanitized.get("command_results") or []:
        _sanitize_command_result(
            result,
            pseudonym_key,
            payload_path_bindings,
            payload_root_bindings,
        )
    for action in sanitized.get("plan") or []:
        if not isinstance(action, dict):
            continue
        for command in action.get("commands") or []:
            _pseudonymize_command(
                command,
                pseudonym_key,
                payload_path_bindings,
                payload_root_bindings,
            )
    for verification in sanitized.get("verification") or []:
        if isinstance(verification, dict):
            verification["detail"] = None
            result = verification.get("command")
            if verification.get("name") == "container.gpu" and verification.get(
                "ok"
            ) is True:
                if (
                    not isinstance(result, dict)
                    or result.get("returncode") != 0
                    or result.get("skipped") is not False
                    or not isinstance(result.get("stdout"), str)
                    or not any(
                        _CUDA_DRIVER_PROBE_SUCCESS.fullmatch(line) is not None
                        for line in result["stdout"].splitlines()
                    )
                ):
                    raise AttestationExportError(
                        "successful container verification has no CUDA Driver API success marker"
                    )
                _sanitize_command_result(
                    result,
                    pseudonym_key,
                    payload_path_bindings,
                    payload_root_bindings,
                )
                result["stdout"] = CUDA_DRIVER_API_ATTESTATION
            else:
                _sanitize_command_result(
                    result,
                    pseudonym_key,
                    payload_path_bindings,
                    payload_root_bindings,
                )
    rollback = sanitized.get("rollback")
    if isinstance(rollback, dict):
        sanitized["rollback"] = _sanitize_snapshot(
            rollback,
            pseudonym_key,
            payload_path_bindings,
            payload_root_bindings,
        )
    return sanitized


def _sanitize_snapshot(
    snapshot: dict[str, Any],
    pseudonym_key: bytes,
    payload_path_bindings: dict[str, str],
    payload_root_bindings: dict[str, str],
) -> dict[str, Any]:
    _validate_snapshot_integrity(snapshot)
    _register_payload_path_bindings(
        snapshot,
        pseudonym_key,
        payload_path_bindings,
        payload_root_bindings,
    )
    sanitized = copy.deepcopy(snapshot)
    snapshot_path = sanitized.get("path")
    sanitized["path"] = (
        _pseudonym(snapshot_path, pseudonym_key, "snapshot-path")
        if snapshot_path is not None
        else None
    )
    sanitized["host_id"] = _pseudonym(
        sanitized.get("host_id"), pseudonym_key, "host-id"
    )
    sanitized["gpu_uuids"] = _pseudonymize_gpu_uuids(
        sanitized.get("gpu_uuids"), pseudonym_key
    )
    for command in sanitized.get("commands") or []:
        _pseudonymize_command(
            command,
            pseudonym_key,
            payload_path_bindings,
            payload_root_bindings,
        )
    _pseudonymize_mig_geometry(
        sanitized.get("mig_geometry"), pseudonym_key
    )
    for managed_file in sanitized.get("managed_files") or []:
        if isinstance(managed_file, dict):
            managed_file["content_base64"] = _managed_file_content_attestation(
                managed_file,
                pseudonym_key,
            )
    sanitized.pop("integrity_sha256", None)
    canonical = json.dumps(
        sanitized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    sanitized["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return sanitized


def _validate_snapshot_integrity(snapshot: dict[str, Any]) -> None:
    claimed = snapshot.get("integrity_sha256")
    if not isinstance(claimed, str) or re.fullmatch(r"[a-f0-9]{64}", claimed) is None:
        raise AttestationExportError("source rollback snapshot has no valid integrity digest")
    canonical_payload = copy.deepcopy(snapshot)
    canonical_payload.pop("integrity_sha256", None)
    canonical = json.dumps(
        canonical_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(actual, claimed):
        raise AttestationExportError("source rollback snapshot failed its integrity check")


def _register_payload_path_bindings(
    snapshot: dict[str, Any],
    pseudonym_key: bytes,
    payload_path_bindings: dict[str, str],
    payload_root_bindings: dict[str, str],
) -> None:
    snapshot_path = snapshot.get("path")
    bundle = snapshot.get("package_payloads")
    if snapshot_path is None and bundle is None:
        return
    if not isinstance(snapshot_path, str) or not isinstance(bundle, dict):
        raise AttestationExportError(
            "source rollback snapshot has inconsistent package payload authority"
        )
    source_snapshot = Path(snapshot_path)
    if (
        not source_snapshot.is_absolute()
        or str(source_snapshot) != snapshot_path
    ):
        raise AttestationExportError(
            "source rollback snapshot has a noncanonical package payload path"
        )
    directory = bundle.get("directory")
    payloads = bundle.get("packages")
    if (
        not isinstance(directory, str)
        or directory != f"{source_snapshot.name}.payloads"
        or not isinstance(payloads, list)
    ):
        raise AttestationExportError(
            "source rollback snapshot has an invalid package payload binding"
        )
    source_root = source_snapshot.parent / directory
    attested_snapshot_path = _pseudonym(
        snapshot_path,
        pseudonym_key,
        "snapshot-path",
    )
    attested_root = (
        Path("/attested")
        / attested_snapshot_path.removeprefix("attested:")
        / directory
    )
    previous_root = payload_root_bindings.setdefault(
        str(source_root),
        str(attested_root),
    )
    if previous_root != str(attested_root):
        raise AttestationExportError(
            "source evidence has conflicting package payload roots"
        )
    filenames: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            raise AttestationExportError(
                "source rollback snapshot has an invalid package payload manifest"
            )
        filename = payload.get("filename")
        digest = payload.get("sha256")
        payload_format = payload.get("format")
        if (
            not isinstance(filename, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or payload_format not in {"deb", "rpm"}
            or filename != f"{digest}.{payload_format}"
            or filename in filenames
        ):
            raise AttestationExportError(
                "source rollback snapshot has an invalid package payload manifest"
            )
        filenames.add(filename)
        source = str(source_root / filename)
        attested = str(attested_root / filename)
        previous = payload_path_bindings.setdefault(source, attested)
        if previous != attested:
            raise AttestationExportError(
                "source evidence has conflicting package payload paths"
            )


def _managed_file_content_attestation(
    managed_file: dict[str, Any], pseudonym_key: bytes
) -> str | None:
    path = managed_file.get("path")
    existed = managed_file.get("existed")
    content = managed_file.get("content_base64")
    mode = managed_file.get("mode")
    if not isinstance(path, str) or not path:
        raise AttestationExportError("source rollback snapshot has an invalid managed-file path")
    if existed is False and content is None and mode is None:
        return None
    if (
        existed is not True
        or not isinstance(content, str)
        or not isinstance(mode, int)
        or isinstance(mode, bool)
    ):
        raise AttestationExportError(
            "source rollback snapshot has inconsistent managed-file state"
        )
    return _keyed_attestation(
        pseudonym_key,
        "managed-file-content",
        path,
        content,
    )


def _sanitize_command_result(
    value: Any,
    pseudonym_key: bytes,
    payload_path_bindings: dict[str, str] | None = None,
    payload_root_bindings: dict[str, str] | None = None,
) -> None:
    if not isinstance(value, dict):
        return
    dnf_module_marker = _validated_dnf_module_marker(value)
    _pseudonymize_command(
        value.get("command"),
        pseudonym_key,
        payload_path_bindings,
        payload_root_bindings,
    )
    value["stdout"] = "[redacted]" if value.get("stdout") else ""
    value["stderr"] = "[redacted]" if value.get("stderr") else ""
    value["reason"] = None
    if dnf_module_marker is not None:
        value["stdout"] = dnf_module_marker


def _validated_dnf_module_marker(value: dict[str, Any]) -> str | None:
    command = value.get("command")
    if (
        not isinstance(command, list)
        or command[:4] != ["python3", "-I", "-c", DNF_MODULE_ENABLE_SCRIPT]
    ):
        return None
    if (
        value.get("returncode") != 0
        or value.get("skipped") is not False
        or value.get("reason") is not None
    ):
        return None
    if len(command) not in {7, 8} or not all(
        isinstance(part, str) for part in command
    ):
        raise AttestationExportError(
            "successful DNF module transaction has an invalid command shape"
        )
    applied = command[4] == "--apply"
    if command[4] not in {"--check", "--apply"}:
        raise AttestationExportError(
            "successful DNF module transaction has an invalid mode"
        )
    stream = command[6]
    binding = command[7] if applied and len(command) == 8 else None
    try:
        expected = dnf_module_enable_command(
            apply=applied,
            stream=stream,
            preflight_sha256=binding,
        )
    except ValueError as exc:
        raise AttestationExportError(
            "successful DNF module transaction has an invalid proof binding"
        ) from exc
    if command != expected or len(command) != (8 if applied else 7):
        raise AttestationExportError(
            "successful DNF module transaction is not the exact fixed command"
        )
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    proof = parse_dnf_module_enable_proof(
        CommandResult(
            command=command,
            returncode=0,
            stdout=stdout if isinstance(stdout, str) else "",
            stderr=stderr if isinstance(stderr, str) else "",
            skipped=False,
        ),
        applied=applied,
        stream=stream,
        preflight_sha256=binding,
    )
    if proof is None:
        raise AttestationExportError(
            "successful DNF module transaction lacks a valid schema-2 proof"
        )
    mode = "apply" if applied else "check"
    return (
        f"{DNF_MODULE_PROOF_ATTESTATION_PREFIX}{mode}:"
        f"{proof.preflight_sha256}]"
    )


def _sanitize_journal(
    payload: bytes,
    path: str,
    pseudonym_key: bytes,
    snapshot_bindings: dict[
        tuple[str, str, str, str],
        tuple[str, str, str, str],
    ],
    payload_path_bindings: dict[str, str],
    payload_root_bindings: dict[str, str],
) -> bytes:
    lines = payload.splitlines()
    if not lines:
        raise AttestationExportError(f"{path} is empty")
    exported: list[bytes] = []
    for index, line in enumerate(lines, start=1):
        entry = _strict_json(line, f"{path} line {index}")
        if not isinstance(entry, dict):
            raise AttestationExportError(f"{path} line {index} is not an object")
        _validate_source_journal_entry(entry, path, index)
        sanitized = copy.deepcopy(entry)
        if entry["event"] == "report-persistence-failed":
            sanitized.pop("error")
        if "reason" in sanitized:
            sanitized["reason"] = None
        if entry.get("event") in {
            "rollback-snapshot-persisted",
            "launcher-release-authorized",
            "operation-recovered",
        }:
            source_binding = _journal_snapshot_binding(entry, path, index)
            sanitized_binding = snapshot_bindings.get(source_binding)
            if sanitized_binding is None:
                raise AttestationExportError(
                    f"{path} line {index} references an unretained or mismatched "
                    "rollback snapshot authority"
                )
            (
                sanitized["snapshot_path"],
                sanitized["snapshot_integrity_sha256"],
                sanitized["snapshot_operation_id"],
                sanitized["snapshot_host_id"],
            ) = sanitized_binding
        _pseudonymize_command(
            sanitized.get("command"),
            pseudonym_key,
            payload_path_bindings,
            payload_root_bindings,
        )
        exported.append(
            (json.dumps(sanitized, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
    return b"".join(exported)


def _validate_source_journal_entry(
    entry: dict[str, Any], path: str, line: int
) -> None:
    event = entry.get("event")
    expected_keys = (
        _SOURCE_JOURNAL_EVENT_KEYS.get(event)
        if isinstance(event, str)
        else None
    )
    if expected_keys is None:
        raise AttestationExportError(
            f"{path} line {line} has an unknown journal event"
        )
    if set(entry) != expected_keys:
        raise AttestationExportError(
            f"{path} line {line} does not match the exact source journal event shape"
        )
    operation_id = entry["operation_id"]
    timestamp = entry["timestamp"]
    if (
        not isinstance(operation_id, str)
        or _OPERATION_ID.fullmatch(operation_id) is None
        or not isinstance(timestamp, str)
        or not timestamp
    ):
        raise AttestationExportError(
            f"{path} line {line} has invalid common journal fields"
        )
    if event in {"command-started", "command-finished"}:
        command = entry["command"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
            or not isinstance(entry["mutating"], bool)
        ):
            raise AttestationExportError(
                f"{path} line {line} has invalid command fields"
            )
    if event == "command-finished":
        returncode = entry["returncode"]
        reason = entry["reason"]
        if (
            not (
                returncode is None
                or (
                    isinstance(returncode, int)
                    and not isinstance(returncode, bool)
                )
            )
            or not isinstance(entry["skipped"], bool)
            or not (reason is None or isinstance(reason, str))
        ):
            raise AttestationExportError(
                f"{path} line {line} has invalid command completion fields"
            )
    if event in {
        "rollback-snapshot-persisted",
        "launcher-release-authorized",
        "operation-recovered",
    }:
        _journal_snapshot_binding(entry, path, line)
    if event == "launcher-release-authorized" and entry["release_target"] not in {
        "install-target",
        "operation-target",
        "rollback-baseline",
    }:
        raise AttestationExportError(
            f"{path} line {line} has an invalid launcher release target"
        )
    if event == "operation-completed":
        exit_code = entry["exit_code"]
        incomplete = entry["incomplete"]
        outcome = entry["outcome"]
        if (
            not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or not 0 <= exit_code <= 255
            or not isinstance(incomplete, bool)
            or outcome not in {"succeeded", "failed"}
            or (exit_code == 0) != (outcome == "succeeded")
            or (incomplete and outcome != "failed")
        ):
            raise AttestationExportError(
                f"{path} line {line} has invalid operation completion fields"
            )
    if event == "report-persistence-failed" and (
        not isinstance(entry["error"], str) or not entry["error"]
    ):
        raise AttestationExportError(
            f"{path} line {line} has an invalid persistence failure"
        )
    if event == "operation-recovered":
        recovery_id = entry["recovery_operation_id"]
        if (
            not isinstance(recovery_id, str)
            or _OPERATION_ID.fullmatch(recovery_id) is None
            or recovery_id == operation_id
        ):
            raise AttestationExportError(
                f"{path} line {line} has an invalid recovery operation ID"
            )


def _register_snapshot_binding(
    source: dict[str, Any],
    sanitized: dict[str, Any],
    bindings: dict[
        tuple[str, str, str, str],
        tuple[str, str, str, str],
    ],
    path: str,
) -> None:
    source_binding = _snapshot_binding(source, f"{path} rollback snapshot")
    sanitized_binding = _snapshot_binding(
        sanitized,
        f"sanitized {path} rollback snapshot",
        require_attested=True,
    )
    previous = bindings.setdefault(source_binding, sanitized_binding)
    if previous != sanitized_binding:
        raise AttestationExportError(
            f"{path} conflicts with another retained rollback snapshot binding"
        )


def _journal_snapshot_binding(
    entry: dict[str, Any], path: str, line: int
) -> tuple[str, str, str, str]:
    return _snapshot_binding(
        {
            "path": entry.get("snapshot_path"),
            "integrity_sha256": entry.get("snapshot_integrity_sha256"),
            "operation_id": entry.get("snapshot_operation_id"),
            "host_id": entry.get("snapshot_host_id"),
        },
        f"{path} line {line}",
    )


def _snapshot_binding(
    snapshot: dict[str, Any],
    description: str,
    *,
    require_attested: bool = False,
) -> tuple[str, str, str, str]:
    snapshot_path = snapshot.get("path")
    integrity = snapshot.get("integrity_sha256")
    operation_id = snapshot.get("operation_id")
    host_id = snapshot.get("host_id")
    path_ok = bool(
        isinstance(snapshot_path, str)
        and (
            re.fullmatch(r"attested:[a-f0-9]{64}", snapshot_path)
            if require_attested
            else Path(snapshot_path).is_absolute()
        )
    )
    host_ok = bool(
        isinstance(host_id, str)
        and (
            re.fullmatch(r"attested:[a-f0-9]{64}", host_id)
            if require_attested
            else re.fullmatch(
                r"(?:machine-id|hostname)-sha256:[a-f0-9]{64}",
                host_id,
            )
        )
    )
    if (
        not path_ok
        or not isinstance(integrity, str)
        or _SHA256.fullmatch(integrity) is None
        or not isinstance(operation_id, str)
        or re.fullmatch(r"[a-f0-9]{32}", operation_id) is None
        or not host_ok
    ):
        raise AttestationExportError(
            f"{description} has an invalid path/hash/creator/host binding"
        )
    assert isinstance(snapshot_path, str)
    assert isinstance(host_id, str)
    return snapshot_path, integrity, operation_id, host_id


def _pseudonymize_command(
    value: Any,
    pseudonym_key: bytes,
    payload_path_bindings: dict[str, str] | None = None,
    payload_root_bindings: dict[str, str] | None = None,
) -> None:
    if not isinstance(value, list):
        return
    payload_paths = payload_path_bindings or {}
    payload_roots = payload_root_bindings or {}
    for index, argument in enumerate(value):
        if not isinstance(argument, str):
            continue
        attested_payload_path = payload_paths.get(argument)
        if attested_payload_path is not None:
            value[index] = attested_payload_path
            continue
        argument_path = Path(argument)
        if argument_path.is_absolute():
            for source_root in payload_roots:
                try:
                    argument_path.relative_to(Path(source_root))
                except ValueError:
                    continue
                raise AttestationExportError(
                    "source command references an unbound package payload path"
                )
            if any(
                component.endswith(".payloads")
                for component in argument_path.parts
            ):
                raise AttestationExportError(
                    "source command leaks an unretained private package payload path"
                )
        if re.fullmatch(
            r"(?:GPU|MIG)-[A-Za-z0-9][A-Za-z0-9/.-]*", argument
        ):
            value[index] = _pseudonym(
                argument, pseudonym_key, "gpu-identity"
            )
            continue
        device_match = re.fullmatch(
            r"device=((?:GPU|MIG)-[A-Za-z0-9][A-Za-z0-9/.-]*)",
            argument,
        )
        if device_match:
            value[index] = (
                "device="
                + _pseudonym(
                    device_match.group(1),
                    pseudonym_key,
                    "gpu-identity",
                )
            )


def _keyed_attestation(
    pseudonym_key: bytes,
    purpose: str,
    *parts: str,
) -> str:
    mac = hmac.new(pseudonym_key, digestmod=hashlib.sha256)
    for part in (purpose, *parts):
        encoded = part.encode("utf-8")
        mac.update(len(encoded).to_bytes(8, "big"))
        mac.update(encoded)
    return f"attested:{mac.hexdigest()}"


def _pseudonym(
    value: Any,
    pseudonym_key: bytes,
    purpose: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise AttestationExportError("source evidence has no host identity")
    return _keyed_attestation(pseudonym_key, purpose, value)


def _pseudonymize_gpu_uuids(
    value: Any, pseudonym_key: bytes
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AttestationExportError("source evidence has invalid GPU UUID binding")
    return [
        _pseudonym(item, pseudonym_key, "gpu-identity")
        for item in value
    ]


def _pseudonymize_optional_gpu_uuids(
    value: Any, pseudonym_key: bytes
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AttestationExportError("source evidence has invalid MIG UUID binding")
    return [
        _pseudonym(item, pseudonym_key, "gpu-identity")
        for item in value
    ]


def _pseudonymize_mig_geometry(value: Any, pseudonym_key: bytes) -> None:
    if not isinstance(value, list):
        raise AttestationExportError("source evidence has invalid MIG geometry")
    for instance in value:
        if not isinstance(instance, dict):
            raise AttestationExportError("source evidence has invalid MIG geometry")
        instance["gpu_uuid"] = _pseudonym(
            instance.get("gpu_uuid"),
            pseudonym_key,
            "gpu-identity",
        )


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite number {value!r}")


def _strict_json(payload: bytes, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AttestationExportError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AttestationExportError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _is_source_evidence_path(path: str) -> bool:
    return path in _CANONICAL_SOURCE_PATHS


def _validate_context(
    matrix_id: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    qualification_wheel_name: str,
    qualification_wheel_sha256: str,
) -> None:
    if (
        not isinstance(matrix_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", matrix_id) is None
    ):
        raise AttestationExportError("invalid matrix identity")
    if (
        not isinstance(workflow_run_id, int)
        or isinstance(workflow_run_id, bool)
        or workflow_run_id <= 0
        or not isinstance(workflow_run_attempt, int)
        or isinstance(workflow_run_attempt, bool)
        or workflow_run_attempt <= 0
    ):
        raise AttestationExportError("invalid workflow run identity")
    if (
        not isinstance(qualification_wheel_name, str)
        or _QUALIFICATION_WHEEL_NAME.fullmatch(qualification_wheel_name) is None
    ):
        raise AttestationExportError("invalid qualification wheel name")
    if (
        not isinstance(qualification_wheel_sha256, str)
        or _SHA256.fullmatch(qualification_wheel_sha256) is None
        or set(qualification_wheel_sha256) == {"0"}
    ):
        raise AttestationExportError("invalid qualification wheel SHA256")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a privacy-minimized GPU integration attestation."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing exact export directory without writing it",
    )
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)
    parser.add_argument("--qualification-wheel-name", required=True)
    parser.add_argument("--qualification-wheel-sha256", required=True)
    args = parser.parse_args(argv)
    if args.verify_only and args.source is not None:
        parser.error("--source cannot be used with --verify-only")
    if not args.verify_only and args.source is None:
        parser.error("--source is required unless --verify-only is set")
    try:
        if args.verify_only:
            verify_export_directory(
                args.output,
                matrix_id=args.matrix_id,
                workflow_run_id=args.workflow_run_id,
                workflow_run_attempt=args.workflow_run_attempt,
                qualification_wheel_name=args.qualification_wheel_name,
                qualification_wheel_sha256=args.qualification_wheel_sha256,
            )
        else:
            assert args.source is not None
            export_attestation(
                args.source,
                args.output,
                matrix_id=args.matrix_id,
                workflow_run_id=args.workflow_run_id,
                workflow_run_attempt=args.workflow_run_attempt,
                qualification_wheel_name=args.qualification_wheel_name,
                qualification_wheel_sha256=args.qualification_wheel_sha256,
            )
    except (AttestationExportError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
