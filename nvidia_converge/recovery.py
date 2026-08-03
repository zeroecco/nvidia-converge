from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .files import open_private_directory
from .report import (
    ACTIVE_JOURNAL_DIRECTORY_NAME,
    ACTIVE_JOURNAL_INDEX_NAME,
    ACTIVE_JOURNAL_INDEX_SCHEMA,
    REPORT_DIR,
)

MAX_RECOVERY_JOURNALS = 1_024
MAX_JOURNAL_BYTES = 8 * 1024 * 1024
MAX_JOURNAL_EVENTS = 100_000
MAX_ACTIVE_MARKER_BYTES = 1_024
MAX_RECOVERY_REPORT_BYTES = 32 * 1024 * 1024
MAX_LEGACY_DIRECTORY_ENTRIES = 16_384
MAX_LEGACY_JOURNALS = 4_096
MAX_LEGACY_JOURNAL_TOTAL_BYTES = 64 * 1024 * 1024

_JOURNAL_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.json\.journal\.jsonl$"
)
_OPERATION_ID = re.compile(r"^[a-f0-9]{32}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_HOST_ID = re.compile(r"^(?:machine-id|hostname)-sha256:[a-f0-9]{64}$")
_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.json$")
_SAFE_PRIVATE_STATE_COMMANDS = frozenset(
    {
        ("persist-rollback-snapshot",),
        ("stage-package-payloads",),
    }
)
_RELEASE_TARGETS = frozenset(
    {"install-target", "operation-target", "rollback-baseline"}
)
_KNOWN_EVENTS = frozenset(
    {
        "operation-started",
        "rollback-snapshot-persisted",
        "launcher-release-authorized",
        "command-started",
        "command-finished",
        "operation-completed",
        "report-persistence-failed",
        "operation-recovered",
    }
)


class RecoveryStateError(ValueError):
    """Raised when durable operation state cannot be trusted."""


@dataclass(frozen=True)
class UnresolvedOperation:
    journal_path: Path
    operation_id: str
    snapshot_path: Path | None
    snapshot_integrity_sha256: str | None
    snapshot_operation_id: str | None
    snapshot_host_id: str | None
    launcher_release_target: str | None
    host_mutation_possible: bool


@dataclass(frozen=True)
class _PrivateJournal:
    payload: bytes
    trusted_length: int
    device: int
    inode: int


@dataclass(frozen=True)
class _ParsedJournal:
    operation_id: str
    operation: UnresolvedOperation | None
    completion: dict[str, Any] | None
    persistence_failed: bool
    recovery_operation_id: str | None
    private_snapshot_artifacts_possible: bool
    snapshot_binding_present: bool


@dataclass(frozen=True)
class _ActiveMarker:
    name: str
    operation_id: str
    device: int
    inode: int


def unresolved_operations(
    *,
    report_dir: Path = REPORT_DIR,
    expected_owner_uid: int | None = None,
) -> list[UnresolvedOperation]:
    """Return mutating operations that lack a trusted terminal recovery event.

    The caller must hold the process-wide operation lock while invoking this
    function. A journal that ended before any host mutation is ignored; a
    journal that can represent a partial host mutation remains authoritative
    until its operation is completed without incomplete state or explicitly
    recovered by a later rollback.
    """

    owner_uid = os.geteuid() if expected_owner_uid is None else expected_owner_uid
    directory_fd = _open_report_directory(report_dir, owner_uid)
    if directory_fd is None:
        return []
    active_path = report_dir / ACTIVE_JOURNAL_DIRECTORY_NAME
    active_directory_fd = _open_active_directory(active_path, owner_uid)
    try:
        _bootstrap_active_index(
            directory_fd,
            active_directory_fd,
            report_dir,
            owner_uid,
        )
        markers = _read_active_markers(active_directory_fd, owner_uid)
        indexed: list[tuple[_ActiveMarker, _ParsedJournal]] = []
        for marker in markers:
            journal = _read_private_journal(
                directory_fd,
                marker.name,
                owner_uid,
            )
            trusted_payload = journal.payload[: journal.trusted_length]
            parsed = _parse_journal(report_dir / marker.name, trusted_payload)
            if parsed.operation_id != marker.operation_id:
                raise RecoveryStateError(
                    "active journal marker operation ID does not match its journal: "
                    + marker.name
                )
            if journal.trusted_length != len(journal.payload):
                _truncate_torn_journal(
                    directory_fd,
                    marker.name,
                    owner_uid,
                    journal,
                )
            indexed.append((marker, parsed))

        completed = {
            parsed.operation_id
            for marker, parsed in indexed
            if _completion_has_final_report(
                directory_fd,
                report_dir,
                marker.name,
                parsed,
                owner_uid,
            )
        }
        unresolved: list[UnresolvedOperation] = []
        retired = False
        for marker, parsed in indexed:
            terminal = bool(
                parsed.operation is None
                or parsed.operation_id in completed
                or (
                    parsed.recovery_operation_id is not None
                    and parsed.recovery_operation_id in completed
                )
            )
            if terminal:
                if parsed.operation is None:
                    _cleanup_private_snapshot_artifacts(parsed, owner_uid)
                _retire_active_marker(
                    active_directory_fd,
                    marker,
                    owner_uid,
                )
                retired = True
            elif (
                parsed.operation is not None
                and not _journal_has_terminal_resolution(parsed)
            ):
                unresolved.append(parsed.operation)
                if len(unresolved) > MAX_RECOVERY_JOURNALS:
                    raise RecoveryStateError(
                        "unresolved applied operation count exceeds the "
                        "recovery scan limit"
                    )
        if retired:
            os.fsync(active_directory_fd)
        _verify_directory_binding(active_path, active_directory_fd, owner_uid)
        return unresolved
    finally:
        os.close(active_directory_fd)
        os.close(directory_fd)


def active_recovery_journal_names(
    *,
    report_dir: Path = REPORT_DIR,
    expected_owner_uid: int | None = None,
) -> list[str]:
    """Return the bounded, validated active journal inventory.

    The caller holds the operation lock.  This is the only inventory used by
    stale emergency-reserve cleanup, so completed report history never causes
    an unbounded top-level directory scan after the one-time bounded bootstrap.
    """

    owner_uid = os.geteuid() if expected_owner_uid is None else expected_owner_uid
    directory_fd = _open_report_directory(report_dir, owner_uid)
    if directory_fd is None:
        return []
    active_path = report_dir / ACTIVE_JOURNAL_DIRECTORY_NAME
    active_directory_fd = _open_active_directory(active_path, owner_uid)
    try:
        _bootstrap_active_index(
            directory_fd,
            active_directory_fd,
            report_dir,
            owner_uid,
        )
        markers = _read_active_markers(active_directory_fd, owner_uid)
        _verify_directory_binding(active_path, active_directory_fd, owner_uid)
        return [marker.name for marker in markers]
    finally:
        os.close(active_directory_fd)
        os.close(directory_fd)


def _open_active_directory(path: Path, owner_uid: int) -> int:
    try:
        return open_private_directory(
            path,
            required_owner_uid=owner_uid,
            create=True,
        )
    except (OSError, ValueError) as exc:
        raise RecoveryStateError(
            f"cannot open active recovery journal directory {path}: {exc}"
        ) from exc


def _bootstrap_active_index(
    directory_fd: int,
    active_directory_fd: int,
    report_dir: Path,
    owner_uid: int,
) -> None:
    if _active_index_present(active_directory_fd, owner_uid):
        return

    journal_names: list[str] = []
    reserve_names: list[str] = []
    total_entries = 0
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                total_entries += 1
                if total_entries > MAX_LEGACY_DIRECTORY_ENTRIES:
                    raise RecoveryStateError(
                        "legacy applied report directory exceeds the bounded index "
                        "bootstrap entry limit; archive preserved terminal report "
                        "history offline before retrying"
                    )
                name = entry.name
                if _legacy_reserve_report_name(name) is not None:
                    reserve_names.append(name)
                    continue
                if not name.endswith(".journal.jsonl"):
                    continue
                if _JOURNAL_NAME.fullmatch(name) is None:
                    raise RecoveryStateError(
                        "applied report directory contains an unsafe journal name: "
                        + name
                    )
                journal_names.append(name)
                if len(journal_names) > MAX_LEGACY_JOURNALS:
                    raise RecoveryStateError(
                        "legacy applied journal history exceeds the bounded index "
                        "bootstrap journal limit; archive preserved terminal journals "
                        "offline before retrying"
                    )
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot enumerate legacy applied report history: {exc}"
        ) from exc

    parsed_journals: list[tuple[str, _ParsedJournal]] = []
    total_bytes = 0
    for name in sorted(journal_names):
        journal = _read_private_journal(directory_fd, name, owner_uid)
        total_bytes += len(journal.payload)
        if total_bytes > MAX_LEGACY_JOURNAL_TOTAL_BYTES:
            raise RecoveryStateError(
                "legacy applied journal history exceeds the bounded index "
                "bootstrap byte limit; archive preserved terminal journals "
                "offline before retrying"
            )
        parsed = _parse_journal(
            report_dir / name,
            journal.payload[: journal.trusted_length],
        )
        if journal.trusted_length != len(journal.payload):
            _truncate_torn_journal(
                directory_fd,
                name,
                owner_uid,
                journal,
            )
        parsed_journals.append((name, parsed))

    completed = {
        parsed.operation_id
        for name, parsed in parsed_journals
        if _completion_has_final_report(
            directory_fd,
            report_dir,
            name,
            parsed,
            owner_uid,
        )
    }
    marker_count = 0
    for name, parsed in parsed_journals:
        terminal = bool(
            parsed.operation is None
            or parsed.operation_id in completed
            or (
                parsed.recovery_operation_id is not None
                and parsed.recovery_operation_id in completed
            )
        )
        if terminal:
            if parsed.operation is None:
                _cleanup_private_snapshot_artifacts(parsed, owner_uid)
            continue
        marker_count += 1
        if marker_count > MAX_RECOVERY_JOURNALS:
            raise RecoveryStateError(
                "active applied operation count exceeds the recovery index limit"
            )
        _ensure_active_marker(
            active_directory_fd,
            name,
            parsed.operation_id,
            owner_uid,
        )

    # The process-wide operation lock proves these allocations have no live
    # owner. They contain random capacity bytes, not audit evidence; journals
    # and reports remain untouched.
    released_reserve = False
    for name in sorted(reserve_names):
        released_reserve = (
            _unlink_private_legacy_reserve(directory_fd, name, owner_uid)
            or released_reserve
        )
    if released_reserve:
        os.fsync(directory_fd)

    # Marker files and their directory entries are durable before the sentinel
    # declares the one-time top-level scan complete. A crash before this point
    # simply repeats the idempotent bootstrap.
    os.fsync(active_directory_fd)
    _create_active_index(active_directory_fd, owner_uid)
    os.fsync(active_directory_fd)
    _verify_directory_binding(
        report_dir / ACTIVE_JOURNAL_DIRECTORY_NAME,
        active_directory_fd,
        owner_uid,
    )


def _bounded_directory_names(directory_fd: int) -> list[str]:
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > MAX_RECOVERY_JOURNALS + 1:
                    raise RecoveryStateError(
                        "active recovery journal directory exceeds its bounded entry limit"
                    )
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot enumerate active recovery journal index: {exc}"
        ) from exc
    return sorted(names)


def _read_active_markers(
    active_directory_fd: int,
    owner_uid: int,
) -> list[_ActiveMarker]:
    markers: list[_ActiveMarker] = []
    for name in _bounded_directory_names(active_directory_fd):
        if name == ACTIVE_JOURNAL_INDEX_NAME:
            continue
        if _JOURNAL_NAME.fullmatch(name) is None:
            raise RecoveryStateError(
                "active recovery journal directory contains an unsafe marker name: "
                + name
            )
        payload, metadata = _read_private_index_file(
            active_directory_fd,
            name,
            owner_uid,
            max_bytes=MAX_ACTIVE_MARKER_BYTES,
        )
        operation_id = _parse_active_marker(name, payload)
        markers.append(
            _ActiveMarker(
                name,
                operation_id,
                metadata.st_dev,
                metadata.st_ino,
            )
        )
    return markers


def _parse_active_marker(name: str, payload: bytes) -> str:
    try:
        marker = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecoveryStateError) as exc:
        raise RecoveryStateError(
            f"active recovery journal marker is invalid: {name}"
        ) from exc
    if not isinstance(marker, dict):
        raise RecoveryStateError(
            f"active recovery journal marker has invalid fields: {name}"
        )
    operation_id = marker.get("operation_id")
    if (
        not isinstance(operation_id, str)
        or _OPERATION_ID.fullmatch(operation_id) is None
        or marker
        != {
            "journal": name,
            "operation_id": operation_id,
            "schema_version": ACTIVE_JOURNAL_INDEX_SCHEMA,
        }
    ):
        raise RecoveryStateError(
            f"active recovery journal marker has invalid fields: {name}"
        )
    return operation_id


def _active_marker_payload(name: str, operation_id: str) -> bytes:
    return (
        json.dumps(
            {
                "journal": name,
                "operation_id": operation_id,
                "schema_version": ACTIVE_JOURNAL_INDEX_SCHEMA,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _ensure_active_marker(
    active_directory_fd: int,
    name: str,
    operation_id: str,
    owner_uid: int,
) -> None:
    payload = _active_marker_payload(name, operation_id)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=active_directory_fd)
    except FileExistsError:
        existing, _ = _read_private_index_file(
            active_directory_fd,
            name,
            owner_uid,
            max_bytes=MAX_ACTIVE_MARKER_BYTES,
        )
        if existing != payload:
            raise RecoveryStateError(
                f"active recovery journal marker conflicts with journal: {name}"
            )
        return
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot create active recovery journal marker {name}: {exc}"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _validate_private_index_file(metadata, name, owner_uid)
    except OSError as exc:
        try:
            os.unlink(name, dir_fd=active_directory_fd)
        except OSError:
            pass
        raise RecoveryStateError(
            f"cannot persist active recovery journal marker {name}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _active_index_present(active_directory_fd: int, owner_uid: int) -> bool:
    try:
        payload, _ = _read_private_index_file(
            active_directory_fd,
            ACTIVE_JOURNAL_INDEX_NAME,
            owner_uid,
            max_bytes=MAX_ACTIVE_MARKER_BYTES,
        )
    except FileNotFoundError:
        return False
    expected = (
        json.dumps(
            {"schema_version": ACTIVE_JOURNAL_INDEX_SCHEMA},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if payload != expected:
        raise RecoveryStateError("active recovery journal index has invalid contents")
    return True


def _create_active_index(active_directory_fd: int, owner_uid: int) -> None:
    if _active_index_present(active_directory_fd, owner_uid):
        return
    payload = (
        json.dumps(
            {"schema_version": ACTIVE_JOURNAL_INDEX_SCHEMA},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            ACTIVE_JOURNAL_INDEX_NAME,
            flags,
            0o600,
            dir_fd=active_directory_fd,
        )
    except FileExistsError:
        if not _active_index_present(active_directory_fd, owner_uid):
            raise RecoveryStateError("active recovery journal index is invalid")
        return
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot create active recovery journal index: {exc}"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _validate_private_index_file(
            os.fstat(descriptor),
            ACTIVE_JOURNAL_INDEX_NAME,
            owner_uid,
        )
    finally:
        os.close(descriptor)


def _read_private_index_file(
    directory_fd: int,
    name: str,
    owner_uid: int,
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot open private recovery index file {name}: {exc}"
        ) from exc
    try:
        try:
            before = os.fstat(descriptor)
            _validate_private_index_file(before, name, owner_uid)
            if before.st_size > max_bytes:
                raise RecoveryStateError(
                    f"private recovery index file exceeds {max_bytes} bytes: {name}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, max_bytes + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise RecoveryStateError(
                        "private recovery index file exceeds "
                        f"{max_bytes} bytes: {name}"
                    )
            after = os.fstat(descriptor)
            _validate_private_index_file(after, name, owner_uid)
            if _file_identity(before) != _file_identity(after):
                raise RecoveryStateError(
                    f"private recovery index file changed while reading: {name}"
                )
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            _validate_private_index_file(named, name, owner_uid)
            if (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
                raise RecoveryStateError(
                    f"private recovery index file changed while opening: {name}"
                )
            return b"".join(chunks), after
        except OSError as exc:
            raise RecoveryStateError(
                f"cannot read private recovery index file {name}: {exc}"
            ) from exc
    finally:
        os.close(descriptor)


def _validate_private_index_file(
    metadata: os.stat_result,
    name: str,
    owner_uid: int,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise RecoveryStateError(
            f"private recovery index entry is not trusted: {name}"
        )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while persisting recovery index")
        offset += written


def _verify_directory_binding(path: Path, descriptor: int, owner_uid: int) -> None:
    opened = os.fstat(descriptor)
    try:
        rebound_descriptor = open_private_directory(
            path,
            required_owner_uid=owner_uid,
        )
    except (OSError, ValueError) as exc:
        raise RecoveryStateError(
            f"cannot rebind private recovery directory {path}: {exc}"
        ) from exc
    try:
        rebound = os.fstat(rebound_descriptor)
        if (opened.st_dev, opened.st_ino) != (rebound.st_dev, rebound.st_ino):
            raise RecoveryStateError(
                f"private recovery directory changed while operating: {path}"
            )
    finally:
        os.close(rebound_descriptor)


def _retire_active_marker(
    active_directory_fd: int,
    marker: _ActiveMarker,
    owner_uid: int,
) -> None:
    payload, metadata = _read_private_index_file(
        active_directory_fd,
        marker.name,
        owner_uid,
        max_bytes=MAX_ACTIVE_MARKER_BYTES,
    )
    if (
        (metadata.st_dev, metadata.st_ino) != (marker.device, marker.inode)
        or _parse_active_marker(marker.name, payload) != marker.operation_id
    ):
        raise RecoveryStateError(
            f"active recovery journal marker changed before retirement: {marker.name}"
        )
    try:
        os.unlink(marker.name, dir_fd=active_directory_fd)
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot retire active recovery journal marker {marker.name}: {exc}"
        ) from exc


def _cleanup_private_snapshot_artifacts(
    parsed: _ParsedJournal,
    owner_uid: int,
) -> None:
    """Clean deterministic pre-mutation authority before retiring its marker."""

    if not parsed.private_snapshot_artifacts_possible:
        return
    # Import lazily so parsing recovery authority remains independent of the
    # rollback module's wider audit and execution dependencies.
    from .rollback import (
        RollbackSnapshotError,
        cleanup_staged_snapshot_artifacts,
    )

    try:
        cleanup_staged_snapshot_artifacts(
            parsed.operation_id,
            preserve_bound_authority=parsed.snapshot_binding_present,
            required_owner_uid=owner_uid,
        )
    except (OSError, RollbackSnapshotError) as exc:
        raise RecoveryStateError(
            "cannot clean safe pre-mutation rollback artifacts for operation "
            f"{parsed.operation_id}: {exc}"
        ) from exc


def _completion_has_final_report(
    directory_fd: int,
    report_dir: Path,
    journal_name: str,
    parsed: _ParsedJournal,
    owner_uid: int,
) -> bool:
    completion = parsed.completion
    if (
        completion is None
        or completion["incomplete"] is not False
        or parsed.persistence_failed
    ):
        return False
    suffix = ".journal.jsonl"
    if not journal_name.endswith(suffix):
        raise RecoveryStateError("cannot derive report name from active journal")
    report_name = journal_name[: -len(suffix)]
    if _REPORT_NAME.fullmatch(report_name) is None:
        raise RecoveryStateError(
            f"active journal derives an unsafe report name: {journal_name}"
        )
    try:
        payload, _ = _read_private_index_file(
            directory_fd,
            report_name,
            owner_uid,
            max_bytes=MAX_RECOVERY_REPORT_BYTES,
        )
    except FileNotFoundError:
        return False
    try:
        report = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecoveryStateError) as exc:
        raise RecoveryStateError(
            f"final applied report is invalid: {report_name}"
        ) from exc
    expected_path = str(report_dir / report_name)
    return bool(
        isinstance(report, dict)
        and report.get("operation_id") == parsed.operation_id
        and report.get("mode") == "apply"
        and report.get("report_path") == expected_path
        and report.get("incomplete") is False
        and type(report.get("exit_code")) is int
        and report.get("exit_code") == completion["exit_code"]
        and report.get("outcome") == completion["outcome"]
    )


def _journal_has_terminal_resolution(parsed: _ParsedJournal) -> bool:
    completion = parsed.completion
    return bool(
        parsed.recovery_operation_id is not None
        or (
            completion is not None
            and completion["incomplete"] is False
            and not parsed.persistence_failed
        )
    )


def _legacy_reserve_report_name(name: str) -> str | None:
    prefix = "."
    suffix = ".emergency-reserve"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    report_name = name[len(prefix) : -len(suffix)]
    return report_name if _REPORT_NAME.fullmatch(report_name) else None


def _unlink_private_legacy_reserve(
    directory_fd: int,
    name: str,
    owner_uid: int,
) -> bool:
    if _legacy_reserve_report_name(name) is None:
        raise RecoveryStateError("cannot remove an unsafe legacy reserve name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    try:
        metadata = os.fstat(descriptor)
        _validate_private_index_file(metadata, name, owner_uid)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _validate_private_index_file(named, name, owner_uid)
        if (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino):
            raise RecoveryStateError(
                f"legacy emergency reserve changed while opening: {name}"
            )
        os.unlink(name, dir_fd=directory_fd)
        return True
    finally:
        os.close(descriptor)


def recovery_snapshot_path(
    operations: list[UnresolvedOperation],
) -> Path | None:
    """Return the one exact snapshot that can recover all unresolved state."""

    if not operations:
        return None
    missing = [item.operation_id for item in operations if item.snapshot_path is None]
    if missing:
        raise RecoveryStateError(
            "interrupted host mutation has no durable rollback snapshot binding: "
            + ", ".join(missing)
        )
    bindings = {
        (
            item.snapshot_path,
            item.snapshot_integrity_sha256,
            item.snapshot_operation_id,
            item.snapshot_host_id,
        )
        for item in operations
    }
    if len(bindings) != 1:
        raise RecoveryStateError(
            "unresolved operations do not bind one unambiguous rollback snapshot"
        )
    snapshot_path, integrity, snapshot_operation_id, snapshot_host_id = next(
        iter(bindings)
    )
    if (
        snapshot_path is None
        or integrity is None
        or snapshot_operation_id is None
        or snapshot_host_id is None
    ):
        raise RecoveryStateError(
            "interrupted host mutation has an incomplete rollback snapshot binding"
        )
    return snapshot_path


def _open_report_directory(path: Path, owner_uid: int) -> int | None:
    try:
        return open_private_directory(
            path,
            required_owner_uid=owner_uid,
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RecoveryStateError(
            f"cannot open applied report directory {path}: {exc}"
        ) from exc


def _read_private_journal(
    directory_fd: int,
    name: str,
    owner_uid: int,
) -> _PrivateJournal:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RecoveryStateError(f"cannot open applied report journal {name}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != owner_uid
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RecoveryStateError(
                f"applied report journal is not a private singly linked file: {name}"
            )
        if metadata.st_size > MAX_JOURNAL_BYTES:
            raise RecoveryStateError(
                f"applied report journal exceeds {MAX_JOURNAL_BYTES} bytes: {name}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_JOURNAL_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_JOURNAL_BYTES:
                raise RecoveryStateError(
                    f"applied report journal exceeds {MAX_JOURNAL_BYTES} bytes: {name}"
                )
        payload = b"".join(chunks)
        trusted_length = len(payload)
        if payload and not payload.endswith(b"\n"):
            trusted_length = payload.rfind(b"\n") + 1
            if trusted_length == 0:
                raise RecoveryStateError(
                    "applied report journal has no complete durable record: "
                    + name
                )
        return _PrivateJournal(
            payload,
            trusted_length,
            metadata.st_dev,
            metadata.st_ino,
        )
    finally:
        os.close(descriptor)


def _truncate_torn_journal(
    directory_fd: int,
    name: str,
    owner_uid: int,
    journal: _PrivateJournal,
) -> None:
    """Durably discard only a parser-validated, unterminated final append."""

    flags = (
        os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot repair torn applied report journal {name}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != owner_uid
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_dev != journal.device
            or metadata.st_ino != journal.inode
            or metadata.st_size != len(journal.payload)
        ):
            raise RecoveryStateError(
                "applied report journal changed while repairing its torn "
                f"final record: {name}"
            )
        os.ftruncate(descriptor, journal.trusted_length)
        os.fsync(descriptor)
    except OSError as exc:
        raise RecoveryStateError(
            f"cannot durably repair torn applied report journal {name}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _parse_journal(path: Path, payload: bytes) -> _ParsedJournal:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryStateError(
            f"applied report journal is not valid UTF-8: {path.name}"
        ) from exc
    if not text or not text.endswith("\n"):
        raise RecoveryStateError(
            f"applied report journal is empty or unrepairable: {path.name}"
        )
    lines = text.splitlines()
    if len(lines) > MAX_JOURNAL_EVENTS:
        raise RecoveryStateError(
            f"applied report journal exceeds {MAX_JOURNAL_EVENTS} events: {path.name}"
        )
    events = [_load_event(line, path) for line in lines]
    operation_id = _validate_event_stream(events, path)

    recovery_events = [
        event for event in events if event["event"] == "operation-recovered"
    ]
    completions = [event for event in events if event["event"] == "operation-completed"]
    persistence_failed = any(
        event["event"] == "report-persistence-failed" for event in events
    )
    private_snapshot_artifacts_possible = any(
        event["event"] == "command-started"
        and event["mutating"] is True
        and tuple(event["command"]) in _SAFE_PRIVATE_STATE_COMMANDS
        for event in events
    )

    host_mutation_possible = any(
        event["event"] == "command-started"
        and event["mutating"] is True
        and tuple(event["command"]) not in _SAFE_PRIVATE_STATE_COMMANDS
        for event in events
    )
    if completions and completions[-1]["incomplete"] is True:
        host_mutation_possible = True

    snapshot_events = [
        event for event in events if event["event"] == "rollback-snapshot-persisted"
    ]
    snapshot_path: Path | None = None
    snapshot_integrity: str | None = None
    snapshot_operation_id: str | None = None
    snapshot_host_id: str | None = None
    if snapshot_events:
        bindings = {
            (
                event["snapshot_path"],
                event["snapshot_integrity_sha256"],
                event["snapshot_operation_id"],
                event["snapshot_host_id"],
            )
            for event in snapshot_events
        }
        if len(bindings) != 1:
            raise RecoveryStateError(
                f"journal contains conflicting rollback snapshot bindings: {path.name}"
            )
        (
            raw_path,
            snapshot_integrity,
            snapshot_operation_id,
            snapshot_host_id,
        ) = next(iter(bindings))
        snapshot_path = Path(raw_path)
    release_events = [
        event for event in events if event["event"] == "launcher-release-authorized"
    ]
    launcher_release_target = (
        release_events[-1]["release_target"] if release_events else None
    )
    operation = (
        UnresolvedOperation(
            path,
            operation_id,
            snapshot_path,
            snapshot_integrity,
            snapshot_operation_id,
            snapshot_host_id,
            launcher_release_target,
            host_mutation_possible,
        )
        if host_mutation_possible
        else None
    )
    return _ParsedJournal(
        operation_id=operation_id,
        operation=operation,
        completion=completions[-1] if completions else None,
        persistence_failed=persistence_failed,
        recovery_operation_id=(
            recovery_events[-1]["recovery_operation_id"]
            if recovery_events
            else None
        ),
        private_snapshot_artifacts_possible=private_snapshot_artifacts_possible,
        snapshot_binding_present=bool(snapshot_events),
    )


def _load_event(line: str, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(line, object_pairs_hook=_object_without_duplicates)
    except (json.JSONDecodeError, RecoveryStateError) as exc:
        raise RecoveryStateError(
            f"applied report journal contains invalid JSON: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise RecoveryStateError(
            f"applied report journal event is not an object: {path.name}"
        )
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RecoveryStateError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _validate_event_stream(events: list[dict[str, Any]], path: Path) -> str:
    if not events or events[0].get("event") != "operation-started":
        raise RecoveryStateError(
            f"applied report journal does not begin with operation-started: {path.name}"
        )
    journal_operation_id: str | None = None
    pending_command: tuple[tuple[str, ...], bool] | None = None
    snapshot_binding: tuple[str, str, str, str] | None = None
    release_targets: list[str] = []
    host_mutation_possible = False
    completion_seen = False
    persistence_failure_seen = False
    sealed_seen = False
    terminal_seen = False
    terminal_allows_persistence_failure = False
    for index, event in enumerate(events):
        event_name = event.get("event")
        operation_id = event.get("operation_id")
        timestamp = event.get("timestamp")
        if event_name not in _KNOWN_EVENTS:
            raise RecoveryStateError(
                f"applied report journal has an unknown event: {path.name}"
            )
        if not isinstance(operation_id, str) or not _OPERATION_ID.fullmatch(operation_id):
            raise RecoveryStateError(
                f"applied report journal has an invalid operation ID: {path.name}"
            )
        if not isinstance(timestamp, str) or not timestamp:
            raise RecoveryStateError(
                f"applied report journal has an invalid timestamp: {path.name}"
            )
        _validate_event_fields(event, event_name, path)
        if journal_operation_id is None:
            journal_operation_id = operation_id
        elif operation_id != journal_operation_id:
            raise RecoveryStateError(
                f"applied report journal mixes operation IDs: {path.name}"
            )
        if terminal_seen and not (
            terminal_allows_persistence_failure
            and event_name == "report-persistence-failed"
        ):
            raise RecoveryStateError(
                f"applied report journal has events after its terminal record: {path.name}"
            )
        if sealed_seen and event_name not in {
            "operation-recovered",
            "report-persistence-failed",
        }:
            raise RecoveryStateError(
                "applied report journal continues after sealed incomplete "
                f"state instead of recording recovery: {path.name}"
            )
        if event_name == "operation-started":
            if index != 0:
                raise RecoveryStateError(
                    f"applied report journal restarts its operation: {path.name}"
                )
        elif event_name == "command-started":
            _validate_command_event(event, path)
            if pending_command is not None:
                raise RecoveryStateError(
                    "applied report journal starts a command before the prior "
                    f"command completed: {path.name}"
                )
            command = tuple(event["command"])
            mutating = event["mutating"]
            if (
                mutating is True
                and command not in _SAFE_PRIVATE_STATE_COMMANDS
                and snapshot_binding is None
            ):
                raise RecoveryStateError(
                    "applied report journal records a possible host mutation "
                    f"before rollback snapshot authority: {path.name}"
                )
            if (
                mutating is True
                and command not in _SAFE_PRIVATE_STATE_COMMANDS
            ):
                host_mutation_possible = True
            pending_command = (command, mutating)
        elif event_name == "command-finished":
            _validate_command_event(event, path)
            finished_command = (
                tuple(event["command"]),
                event["mutating"],
            )
            if pending_command != finished_command:
                raise RecoveryStateError(
                    "applied report journal command completion does not match "
                    f"one pending command: {path.name}"
                )
            returncode = event["returncode"]
            reason = event["reason"]
            if (
                not isinstance(event["skipped"], bool)
                or not (
                    returncode is None
                    or (
                        isinstance(returncode, int)
                        and not isinstance(returncode, bool)
                    )
                )
                or not (reason is None or isinstance(reason, str))
            ):
                raise RecoveryStateError(
                    f"applied report journal has invalid command completion: {path.name}"
                )
            pending_command = None
        elif event_name == "rollback-snapshot-persisted":
            if pending_command is not None or snapshot_binding is not None:
                raise RecoveryStateError(
                    "applied report journal records a misplaced or duplicate "
                    f"rollback snapshot binding: {path.name}"
                )
            snapshot_binding = _snapshot_event_binding(event, path)
        elif event_name == "launcher-release-authorized":
            if pending_command is not None or snapshot_binding is None:
                raise RecoveryStateError(
                    "applied report journal authorizes launcher release outside "
                    f"a completed snapshot-bound phase: {path.name}"
                )
            if _snapshot_event_binding(event, path) != snapshot_binding:
                raise RecoveryStateError(
                    "applied report journal launcher release does not match its "
                    f"exact rollback snapshot binding: {path.name}"
                )
            release_target = event["release_target"]
            if (
                release_target not in _RELEASE_TARGETS
                or release_target in release_targets
                or (
                    release_target in {"install-target", "operation-target"}
                    and release_targets
                )
                or (
                    release_target == "rollback-baseline"
                    and release_targets
                    not in ([], ["install-target"], ["operation-target"])
                )
                or not host_mutation_possible
            ):
                raise RecoveryStateError(
                    f"applied report journal has invalid launcher release order: {path.name}"
                )
            release_targets.append(release_target)
        elif event_name == "operation-completed":
            if pending_command is not None or completion_seen:
                raise RecoveryStateError(
                    "applied report journal completes with pending or duplicate "
                    f"command state: {path.name}"
                )
            _validate_completion_event(event, path)
            completion_seen = True
            if event["incomplete"] is False:
                terminal_seen = True
                terminal_allows_persistence_failure = True
            else:
                if snapshot_binding is None:
                    raise RecoveryStateError(
                        "applied report journal seals incomplete host state "
                        f"without rollback snapshot authority: {path.name}"
                    )
                host_mutation_possible = True
                sealed_seen = True
        elif event_name == "report-persistence-failed":
            if pending_command is not None or persistence_failure_seen:
                raise RecoveryStateError(
                    "applied report journal records invalid report persistence "
                    f"failure state: {path.name}"
                )
            error = event["error"]
            if not isinstance(error, str) or not error:
                raise RecoveryStateError(
                    f"applied report journal has invalid persistence failure: {path.name}"
                )
            persistence_failure_seen = True
            terminal_seen = False
            terminal_allows_persistence_failure = False
            sealed_seen = True
        elif event_name == "operation-recovered":
            recovery_id = event["recovery_operation_id"]
            if (
                not isinstance(recovery_id, str)
                or not _OPERATION_ID.fullmatch(recovery_id)
                or recovery_id == journal_operation_id
                or snapshot_binding is None
                or not host_mutation_possible
                or _snapshot_event_binding(event, path) != snapshot_binding
            ):
                raise RecoveryStateError(
                    f"applied report journal has invalid recovery state: {path.name}"
                )
            terminal_seen = True
            terminal_allows_persistence_failure = False
    assert journal_operation_id is not None
    return journal_operation_id


def _validate_event_fields(
    event: dict[str, Any],
    event_name: str,
    path: Path,
) -> None:
    common = {"event", "operation_id", "timestamp"}
    details = {
        "operation-started": set(),
        "rollback-snapshot-persisted": {
            "snapshot_path",
            "snapshot_integrity_sha256",
            "snapshot_operation_id",
            "snapshot_host_id",
        },
        "launcher-release-authorized": {
            "release_target",
            "snapshot_path",
            "snapshot_integrity_sha256",
            "snapshot_operation_id",
            "snapshot_host_id",
        },
        "command-started": {"command", "mutating"},
        "command-finished": {
            "command",
            "mutating",
            "returncode",
            "skipped",
            "reason",
        },
        "operation-completed": {"exit_code", "incomplete", "outcome"},
        "report-persistence-failed": {"error"},
        "operation-recovered": {
            "recovery_operation_id",
            "snapshot_path",
            "snapshot_integrity_sha256",
            "snapshot_operation_id",
            "snapshot_host_id",
        },
    }[event_name]
    if set(event) != common | details:
        raise RecoveryStateError(
            f"applied report journal event has unexpected fields: {path.name}"
        )


def _validate_command_event(event: dict[str, Any], path: Path) -> None:
    command = event.get("command")
    if (
        not isinstance(event.get("mutating"), bool)
        or not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise RecoveryStateError(
            f"applied report journal has an invalid command event: {path.name}"
        )


def _snapshot_event_binding(
    event: dict[str, Any],
    path: Path,
) -> tuple[str, str, str, str]:
    snapshot_path = event.get("snapshot_path")
    integrity = event.get("snapshot_integrity_sha256")
    snapshot_operation_id = event.get("snapshot_operation_id")
    snapshot_host_id = event.get("snapshot_host_id")
    if (
        not isinstance(snapshot_path, str)
        or not Path(snapshot_path).is_absolute()
        or not isinstance(integrity, str)
        or not _SHA256.fullmatch(integrity)
        or not isinstance(snapshot_operation_id, str)
        or not _OPERATION_ID.fullmatch(snapshot_operation_id)
        or not isinstance(snapshot_host_id, str)
        or not _HOST_ID.fullmatch(snapshot_host_id)
    ):
        raise RecoveryStateError(
            f"applied report journal has an invalid rollback snapshot binding: {path.name}"
        )
    return snapshot_path, integrity, snapshot_operation_id, snapshot_host_id


def _validate_completion_event(event: dict[str, Any], path: Path) -> None:
    exit_code = event["exit_code"]
    incomplete = event["incomplete"]
    outcome = event["outcome"]
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not isinstance(incomplete, bool)
        or outcome not in {"succeeded", "failed"}
        or (exit_code == 0) != (outcome == "succeeded")
        or (incomplete and outcome != "failed")
    ):
        raise RecoveryStateError(
            f"applied report journal has invalid completion state: {path.name}"
        )
