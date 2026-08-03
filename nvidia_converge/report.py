from __future__ import annotations

import errno
import json
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .files import (
    atomic_write_text,
    atomic_write_text_trusted,
    ensure_private_directory,
    open_private_directory,
)
from .models import HostAudit, PackageInfo, Report, utc_now

REPORT_DIR = Path("/var/lib/nvidia-converge/reports")
MIN_APPLIED_STATE_FREE_BYTES = 64 * 1024 * 1024
MIN_APPLIED_STATE_FREE_INODES = 32
APPLIED_REPORT_EMERGENCY_RESERVE_BYTES = 4 * 1024 * 1024
_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.json$")
_EMERGENCY_RESERVE_SUFFIX = ".emergency-reserve"
ACTIVE_JOURNAL_DIRECTORY_NAME = ".active-journals"
ACTIVE_JOURNAL_INDEX_NAME = "index-v1.json"
ACTIVE_JOURNAL_INDEX_SCHEMA = "1.0"
MAX_ACTIVE_JOURNAL_MARKERS = 1_024
_ACTIVE_JOURNAL_MARKER_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.json\.journal\.jsonl$"
)
_CAPACITY_ERRNOS = frozenset(
    {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
)


class ReportWriteError(OSError):
    pass


class ReportJournalIntegrityError(ReportWriteError):
    """Raised when a failed journal append cannot be durably rolled back."""


def applied_report_path(
    command: str,
    requested_path: str | None,
    operation_id: str,
) -> Path:
    """Resolve an applied report to a private, root-controlled state directory."""
    _ensure_private_report_directory()
    if requested_path:
        path = Path(requested_path)
        if not path.is_absolute() or path.parent != REPORT_DIR:
            raise ReportWriteError(
                f"applied reports must be stored directly in {REPORT_DIR}"
            )
    else:
        path = REPORT_DIR / f"{command}-{operation_id}.json"
    if not _REPORT_NAME.fullmatch(path.name):
        raise ReportWriteError(
            "applied report filename must end in .json and contain only letters, "
            "digits, dots, underscores, or hyphens"
        )
    return path


def reserve_applied_report(
    report: Report,
    path: Path,
    *,
    capacity_paths: Iterable[Path] = (),
) -> None:
    """Reserve report, journal, and emergency recovery storage exclusively."""
    _validate_applied_report_path(path)
    report_text = report_json(report) + "\n"
    journal_path = report_journal_path(path)
    reserve_path = report_emergency_reserve_path(path)
    directory_fd = _open_report_directory(create=True)
    active_directory_fd = -1
    created: list[tuple[Path, os.stat_result]] = []
    active_marker: tuple[Path, os.stat_result] | None = None
    try:
        active_directory_fd = _open_active_journal_directory(create=True)
        created.append(
            (path, _create_private_file(directory_fd, path, report_text))
        )
        journal_metadata = _create_private_file(
            directory_fd,
            journal_path,
            json.dumps(
                {
                    "event": "operation-started",
                    "operation_id": report.operation_id,
                    "timestamp": report.started_at,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )
        created.append((journal_path, journal_metadata))
        # The journal directory entry must be durable before its active marker:
        # after reserve returns, recovery may rely exclusively on the bounded
        # marker index instead of rescanning all historical reports.
        os.fsync(directory_fd)
        _verify_report_directory_binding(directory_fd)
        _require_active_marker_capacity(active_directory_fd)
        marker_path = active_journal_marker_path(path)
        active_marker = (
            marker_path,
            _create_private_file(
                active_directory_fd,
                marker_path,
                _active_journal_marker_text(path, report.operation_id),
            ),
        )
        os.fsync(active_directory_fd)
        _verify_active_journal_directory_binding(active_directory_fd)
        reserve_metadata = _create_emergency_reserve(
            directory_fd,
            reserve_path,
        )
        created.append((reserve_path, reserve_metadata))
        os.fsync(directory_fd)
        _verify_report_directory_binding(directory_fd)
        require_applied_state_capacity(REPORT_DIR, *capacity_paths)
        _verify_report_directory_binding(directory_fd)
    except (OSError, ValueError) as exc:
        if active_marker is not None and active_directory_fd >= 0:
            try:
                _unlink_created_private_file(
                    active_directory_fd,
                    active_marker[0],
                    active_marker[1],
                )
                os.fsync(active_directory_fd)
            except OSError:
                pass
        for created_path, created_metadata in reversed(created):
            try:
                _unlink_created_private_file(
                    directory_fd,
                    created_path,
                    created_metadata,
                )
            except OSError:
                pass
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        raise ReportWriteError(
            f"cannot reserve applied report {str(path)!r}: {exc}"
        ) from exc
    finally:
        if active_directory_fd >= 0:
            os.close(active_directory_fd)
        os.close(directory_fd)


def append_report_journal(
    report_path: Path,
    operation_id: str,
    event: str,
    **details: Any,
) -> None:
    _validate_applied_report_path(report_path)
    payload = {
        "event": event,
        "operation_id": operation_id,
        "timestamp": utc_now(),
        **details,
    }
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    journal_path = report_journal_path(report_path)
    flags = (
        os.O_RDWR
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = _open_report_directory()
    try:
        fd = os.open(journal_path.name, flags, dir_fd=directory_fd)
        try:
            encoded_line = line.encode("utf-8")
            emergency_retry_available = True
            while True:
                metadata = os.fstat(fd)
                _validate_private_report_file(metadata, journal_path)
                _validate_open_private_file_binding(
                    directory_fd,
                    journal_path.name,
                    fd,
                    journal_path,
                )
                _validate_journal_append_boundary(fd, metadata, journal_path)
                try:
                    _write_all(fd, encoded_line)
                    os.fsync(fd)
                    _validate_open_private_file_binding(
                        directory_fd,
                        journal_path.name,
                        fd,
                        journal_path,
                    )
                    _verify_report_directory_binding(directory_fd)
                    break
                except OSError as append_error:
                    try:
                        _durably_rollback_journal_append(
                            fd,
                            directory_fd,
                            journal_path.name,
                            journal_path,
                            metadata,
                            len(encoded_line),
                        )
                    except OSError as rollback_error:
                        raise ReportJournalIntegrityError(
                            "cannot ensure applied report journal tail integrity after "
                            f"append failure for {str(journal_path)!r}: "
                            f"{append_error}; durable rollback failed: {rollback_error}"
                        ) from rollback_error
                    if (
                        emergency_retry_available
                        and _is_capacity_error(append_error)
                    ):
                        emergency_retry_available = False
                        if _unlink_private_reserve(
                            directory_fd,
                            report_emergency_reserve_path(report_path).name,
                        ):
                            os.fsync(directory_fd)
                            _verify_report_directory_binding(directory_fd)
                            continue
                    raise ReportWriteError(
                        f"cannot append applied report journal {str(journal_path)!r}: "
                        f"{append_error}; partial append was durably rolled back"
                    ) from append_error
        finally:
            os.close(fd)
    except ReportWriteError:
        raise
    except OSError as exc:
        raise ReportWriteError(
            f"cannot append applied report journal {str(journal_path)!r}: {exc}"
        ) from exc
    finally:
        os.close(directory_fd)


def report_journal_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.name}.journal.jsonl")


def report_emergency_reserve_path(report_path: Path) -> Path:
    return report_path.with_name(
        f".{report_path.name}{_EMERGENCY_RESERVE_SUFFIX}"
    )


def active_journal_marker_path(report_path: Path) -> Path:
    _validate_applied_report_path(report_path)
    return REPORT_DIR / ACTIVE_JOURNAL_DIRECTORY_NAME / report_journal_path(
        report_path
    ).name


def _active_journal_marker_text(report_path: Path, operation_id: str) -> str:
    return (
        json.dumps(
            {
                "journal": report_journal_path(report_path).name,
                "operation_id": operation_id,
                "schema_version": ACTIVE_JOURNAL_INDEX_SCHEMA,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def require_applied_state_capacity(*paths: Path) -> None:
    """Require the documented post-reservation budget on every filesystem.

    Missing state directories are created privately through descriptor-walked,
    no-follow ancestry. Filesystems are deduplicated by device so the report
    and snapshot budgets are not double-counted on one state partition.
    """

    checked_devices: set[int] = set()
    for path in paths:
        try:
            descriptor = open_private_directory(
                path,
                required_owner_uid=os.geteuid(),
                create=True,
            )
        except (OSError, ValueError) as exc:
            raise ReportWriteError(
                f"cannot inspect applied state storage capacity for {path}: {exc}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            capacity = os.fstatvfs(descriptor)
            if metadata.st_dev in checked_devices:
                continue
            checked_devices.add(metadata.st_dev)
            fragment_size = capacity.f_frsize or capacity.f_bsize
            if (
                fragment_size <= 0
                or capacity.f_bavail < 0
                or capacity.f_favail < 0
            ):
                raise ReportWriteError(
                    f"applied state filesystem returned invalid capacity for {path}"
                )
            free_bytes = capacity.f_bavail * fragment_size
            free_inodes = capacity.f_favail
            if free_bytes < MIN_APPLIED_STATE_FREE_BYTES:
                raise ReportWriteError(
                    "applied state filesystem has insufficient free space for host "
                    f"mutation: {free_bytes} bytes available at {path}; "
                    f"{MIN_APPLIED_STATE_FREE_BYTES} required after emergency reservation"
                )
            if free_inodes < MIN_APPLIED_STATE_FREE_INODES:
                raise ReportWriteError(
                    "applied state filesystem has insufficient free inodes for host "
                    f"mutation: {free_inodes} available at {path}; "
                    f"{MIN_APPLIED_STATE_FREE_INODES} required after emergency reservation"
                )
        finally:
            os.close(descriptor)


def release_applied_report_reserve(report_path: Path) -> bool:
    """Release one operation's emergency allocation after validating identity."""

    _validate_applied_report_path(report_path)
    directory_fd = _open_report_directory()
    try:
        released = _unlink_private_reserve(
            directory_fd,
            report_emergency_reserve_path(report_path).name,
        )
        if released:
            os.fsync(directory_fd)
        _verify_report_directory_binding(directory_fd)
        return released
    except ReportWriteError:
        raise
    except OSError as exc:
        raise ReportWriteError(
            f"cannot release applied report emergency reserve for {report_path}: {exc}"
        ) from exc
    finally:
        os.close(directory_fd)


def cleanup_stale_applied_report_reserves() -> None:
    """Reclaim reserves left by dead operations while holding the operation lock."""

    # Import lazily to avoid a module cycle: recovery imports the report path
    # constants, while applied startup intentionally bootstraps/validates the
    # bounded active index before reserve cleanup.
    from .recovery import active_recovery_journal_names

    journal_names = active_recovery_journal_names(
        report_dir=REPORT_DIR,
        expected_owner_uid=os.geteuid(),
    )
    directory_fd = _open_report_directory(create=True)
    released = False
    try:
        for journal_name in journal_names:
            suffix = ".journal.jsonl"
            if not journal_name.endswith(suffix):
                raise ReportWriteError(
                    "active recovery index returned an unsafe journal name"
                )
            report_name = journal_name[: -len(suffix)]
            reserve_name = f".{report_name}{_EMERGENCY_RESERVE_SUFFIX}"
            released = (
                _unlink_private_reserve(directory_fd, reserve_name) or released
            )
        if released:
            os.fsync(directory_fd)
        _verify_report_directory_binding(directory_fd)
    except ReportWriteError:
        raise
    except OSError as exc:
        raise ReportWriteError(
            f"cannot clean stale applied report emergency reserves: {exc}"
        ) from exc
    finally:
        os.close(directory_fd)


def _ensure_private_report_directory() -> None:
    try:
        ensure_private_directory(
            REPORT_DIR,
            required_owner_uid=os.geteuid(),
        )
    except (OSError, ValueError) as exc:
        raise ReportWriteError(
            f"cannot create applied report directory {str(REPORT_DIR)!r}: {exc}"
        ) from exc


def _validate_applied_report_path(path: Path) -> None:
    if not path.is_absolute() or path.parent != REPORT_DIR:
        raise ReportWriteError(
            f"applied reports must be stored directly in {REPORT_DIR}"
        )
    if not _REPORT_NAME.fullmatch(path.name):
        raise ReportWriteError("cannot derive an emergency reserve from an unsafe name")


def _open_report_directory(*, create: bool = False) -> int:
    try:
        return open_private_directory(
            REPORT_DIR,
            required_owner_uid=os.geteuid(),
            create=create,
        )
    except (OSError, ValueError) as exc:
        raise ReportWriteError(
            f"cannot open trusted applied report directory {REPORT_DIR}: {exc}"
        ) from exc


def _open_active_journal_directory(*, create: bool = False) -> int:
    path = REPORT_DIR / ACTIVE_JOURNAL_DIRECTORY_NAME
    try:
        return open_private_directory(
            path,
            required_owner_uid=os.geteuid(),
            create=create,
        )
    except (OSError, ValueError) as exc:
        raise ReportWriteError(
            f"cannot open trusted active journal directory {path}: {exc}"
        ) from exc


def _require_active_marker_capacity(directory_fd: int) -> None:
    marker_count = 0
    total_entries = 0
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                total_entries += 1
                if total_entries > MAX_ACTIVE_JOURNAL_MARKERS + 1:
                    raise ReportWriteError(
                        "active recovery journal index exceeds its bounded entry limit"
                    )
                if entry.name == ACTIVE_JOURNAL_INDEX_NAME:
                    continue
                if _ACTIVE_JOURNAL_MARKER_NAME.fullmatch(entry.name) is None:
                    raise ReportWriteError(
                        "active recovery journal index contains an unsafe entry: "
                        + entry.name
                    )
                marker_count += 1
    except OSError as exc:
        raise ReportWriteError(
            f"cannot inspect active recovery journal capacity: {exc}"
        ) from exc
    if marker_count >= MAX_ACTIVE_JOURNAL_MARKERS:
        raise ReportWriteError(
            "active recovery journal count exceeds the reservation limit"
        )


def _create_emergency_reserve(
    directory_fd: int,
    path: Path,
) -> os.stat_result:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    created: os.stat_result | None = None
    try:
        os.fchmod(fd, 0o600)
        created = os.fstat(fd)
        _validate_private_report_file(created, path)
        _allocate_emergency_bytes(fd, APPLIED_REPORT_EMERGENCY_RESERVE_BYTES)
        os.fsync(fd)
        _validate_open_private_file_binding(
            directory_fd,
            path.name,
            fd,
            path,
        )
        return os.fstat(fd)
    except BaseException:
        if created is not None:
            try:
                _unlink_created_private_file(directory_fd, path, created)
            except OSError:
                pass
        raise
    finally:
        os.close(fd)


def _allocate_emergency_bytes(fd: int, size: int) -> None:
    if size <= 0:
        raise OSError("emergency reserve size must be positive")
    allocator = getattr(os, "posix_fallocate", None)
    if allocator is not None:
        try:
            allocator(fd, 0, size)
            return
        except OSError as exc:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in unsupported:
                raise
            os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    remaining = size
    while remaining:
        chunk = os.urandom(min(64 * 1024, remaining))
        _write_all(fd, chunk)
        remaining -= len(chunk)


def _reserved_report_name(name: str) -> str | None:
    if not name.startswith(".") or not name.endswith(_EMERGENCY_RESERVE_SUFFIX):
        return None
    report_name = name[1 : -len(_EMERGENCY_RESERVE_SUFFIX)]
    return report_name if _REPORT_NAME.fullmatch(report_name) else None


def _unlink_private_reserve(directory_fd: int, name: str) -> bool:
    if _reserved_report_name(name) is None:
        raise ReportWriteError("cannot release an unsafe emergency reserve name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    try:
        metadata = os.fstat(fd)
        path = REPORT_DIR / name
        _validate_private_report_file(metadata, path)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _validate_private_report_file(named, path)
        if named.st_dev != metadata.st_dev or named.st_ino != metadata.st_ino:
            raise ReportWriteError(
                f"applied report emergency reserve changed while opening: {path}"
            )
        os.unlink(name, dir_fd=directory_fd)
        return True
    finally:
        os.close(fd)


def _is_capacity_error(error: OSError) -> bool:
    return error.errno in _CAPACITY_ERRNOS


def _create_private_file(
    directory_fd: int,
    path: Path,
    text: str,
) -> os.stat_result:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    created: os.stat_result | None = None
    try:
        os.fchmod(fd, 0o600)
        created = os.fstat(fd)
        _validate_private_report_file(created, path)
        _write_all(fd, text.encode("utf-8"))
        os.fsync(fd)
        _validate_open_private_file_binding(
            directory_fd,
            path.name,
            fd,
            path,
        )
        return os.fstat(fd)
    except BaseException:
        if created is not None:
            try:
                _unlink_created_private_file(directory_fd, path, created)
            except OSError:
                pass
        raise
    finally:
        os.close(fd)


def _unlink_created_private_file(
    directory_fd: int,
    path: Path,
    expected: os.stat_result,
) -> None:
    try:
        named = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if named.st_dev != expected.st_dev or named.st_ino != expected.st_ino:
        raise OSError(f"created applied report file changed before cleanup: {path}")
    os.unlink(path.name, dir_fd=directory_fd)


def _validate_open_private_file_binding(
    directory_fd: int,
    name: str,
    fd: int,
    path: Path,
) -> None:
    opened = os.fstat(fd)
    _validate_private_report_file(opened, path)
    named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    _validate_private_report_file(named, path)
    if named.st_dev != opened.st_dev or named.st_ino != opened.st_ino:
        raise OSError(f"applied report file changed while opening: {path}")


def _validate_private_report_file(metadata: os.stat_result, path: Path) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise OSError(
            f"applied report must be a private, singly linked regular file owned "
            f"by uid {os.geteuid()}: {path}"
        )


def _validate_journal_append_boundary(
    fd: int,
    metadata: os.stat_result,
    path: Path,
) -> None:
    """Refuse to append when the existing journal has no complete final record."""

    if metadata.st_size == 0:
        raise ReportJournalIntegrityError(
            f"cannot ensure applied report journal tail integrity: {path} is empty"
        )
    try:
        final_byte = os.pread(fd, 1, metadata.st_size - 1)
        current = os.fstat(fd)
    except OSError as exc:
        raise ReportJournalIntegrityError(
            f"cannot inspect applied report journal tail integrity for {path}: {exc}"
        ) from exc
    if (
        final_byte != b"\n"
        or current.st_dev != metadata.st_dev
        or current.st_ino != metadata.st_ino
        or current.st_size != metadata.st_size
    ):
        raise ReportJournalIntegrityError(
            f"cannot ensure applied report journal tail integrity before append: {path}"
        )


def _durably_rollback_journal_append(
    fd: int,
    directory_fd: int,
    name: str,
    path: Path,
    original: os.stat_result,
    append_length: int,
) -> None:
    """Restore a failed append on its original descriptor and make it durable."""

    current = os.fstat(fd)
    _validate_private_report_file(current, path)
    if (
        current.st_dev != original.st_dev
        or current.st_ino != original.st_ino
        or current.st_size < original.st_size
        or current.st_size > original.st_size + append_length
    ):
        raise OSError("journal changed outside the failed append")
    os.ftruncate(fd, original.st_size)
    os.fsync(fd)
    repaired = os.fstat(fd)
    _validate_private_report_file(repaired, path)
    if (
        repaired.st_dev != original.st_dev
        or repaired.st_ino != original.st_ino
        or repaired.st_size != original.st_size
    ):
        raise OSError("journal descriptor changed while rolling back failed append")
    named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    _validate_private_report_file(named, path)
    if (
        named.st_dev != original.st_dev
        or named.st_ino != original.st_ino
        or named.st_size != original.st_size
    ):
        raise OSError("journal path changed while rolling back failed append")


def _verify_report_directory_binding(directory_fd: int) -> None:
    opened = os.fstat(directory_fd)
    rebound_fd = _open_report_directory()
    try:
        rebound = os.fstat(rebound_fd)
        if rebound.st_dev != opened.st_dev or rebound.st_ino != opened.st_ino:
            raise ReportWriteError(
                f"applied report directory changed while operating: {REPORT_DIR}"
            )
    finally:
        os.close(rebound_fd)


def _verify_active_journal_directory_binding(directory_fd: int) -> None:
    opened = os.fstat(directory_fd)
    rebound_fd = _open_active_journal_directory()
    try:
        rebound = os.fstat(rebound_fd)
        if rebound.st_dev != opened.st_dev or rebound.st_ino != opened.st_ino:
            raise ReportWriteError(
                "active journal directory changed while operating: "
                f"{REPORT_DIR / ACTIVE_JOURNAL_DIRECTORY_NAME}"
            )
    finally:
        os.close(rebound_fd)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write while persisting applied report evidence")
        offset += written


def _fsync_report_directory() -> None:
    fd = _open_report_directory()
    try:
        os.fsync(fd)
        _verify_report_directory_binding(fd)
    finally:
        os.close(fd)


def write_report(report: Report, path: str | None) -> None:
    text = report_json(report)
    if path:
        try:
            output_path = Path(path)
            if report.mode == "apply":
                _validate_applied_report_path(output_path)
                _ensure_private_report_directory()
                atomic_write_text_trusted(
                    output_path,
                    text + "\n",
                    mode=0o600,
                    required_owner_uid=os.geteuid(),
                )
            else:
                atomic_write_text(output_path, text + "\n")
        except (OSError, ValueError) as exc:
            raise ReportWriteError(f"cannot write report {path!r}: {exc}") from exc
    else:
        print(text)


def report_json(report: Report) -> str:
    if (
        report.command in {"install", "verify", "lock", "snapshot"}
        and report.rollback is not None
        and report.rollback.operation_id != report.operation_id
    ):
        raise ReportWriteError(
            "in-operation rollback snapshot operation_id does not match its report"
        )
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def sbom_from_audit(audit: HostAudit) -> list[PackageInfo]:
    sbom = list(audit.packages)
    if audit.module.version:
        sbom.append(PackageInfo(name="nvidia-kernel-module", version=audit.module.version, manager="kernel", installed=audit.module.loaded))
    sbom.append(PackageInfo(name="linux-kernel", version=audit.kernel.running, manager="kernel", installed=True))
    return sbom
