import errno
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvidia_converge import cli
from nvidia_converge import report as report_module
from nvidia_converge.models import DesiredState, Report, RollbackSnapshot, utc_now
from nvidia_converge.recovery import unresolved_operations
from nvidia_converge.report import (
    APPLIED_REPORT_EMERGENCY_RESERVE_BYTES,
    MIN_APPLIED_STATE_FREE_BYTES,
    MIN_APPLIED_STATE_FREE_INODES,
    ReportJournalIntegrityError,
    ReportWriteError,
    append_report_journal,
    applied_report_path,
    cleanup_stale_applied_report_reserves,
    release_applied_report_reserve,
    report_emergency_reserve_path,
    report_journal_path,
    report_json,
    require_applied_state_capacity,
    reserve_applied_report,
)

SNAPSHOT_INTEGRITY = "c" * 64
SNAPSHOT_HOST_ID = "machine-id-sha256:" + "e" * 64


def test_applied_report_path_defaults_to_private_state_directory(
    monkeypatch, tmp_path
):
    report_dir = tmp_path / "reports"
    monkeypatch.setattr("nvidia_converge.report.REPORT_DIR", report_dir)

    path = applied_report_path("install", None, "a" * 32)

    assert path == report_dir / f"install-{'a' * 32}.json"
    assert stat.S_IMODE(report_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "requested",
    [
        "relative.json",
        "/tmp/report.json",
        "/var/lib/nvidia-converge/reports/../outside.json",
    ],
)
def test_applied_report_path_rejects_paths_outside_private_state_directory(
    monkeypatch, tmp_path, requested
):
    monkeypatch.setattr(
        "nvidia_converge.report.REPORT_DIR", tmp_path / "reports"
    )

    with pytest.raises(ReportWriteError, match="stored directly"):
        applied_report_path("install", requested, "a" * 32)


def test_applied_report_path_rejects_unsafe_filename(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    monkeypatch.setattr("nvidia_converge.report.REPORT_DIR", report_dir)

    with pytest.raises(ReportWriteError, match="filename"):
        applied_report_path(
            "install", str(report_dir / ".hidden.json"), "a" * 32
        )


def test_reserve_and_append_applied_report_evidence(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    monkeypatch.setattr("nvidia_converge.report.REPORT_DIR", report_dir)
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)

    reserve_applied_report(report, path)
    append_report_journal(
        path,
        report.operation_id,
        "command-finished",
        command=["apt-get", "--simulate", "install", "nvidia-open"],
        mutating=False,
        returncode=0,
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    journal = report_journal_path(path)
    reserve = report_emergency_reserve_path(path)
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600
    assert stat.S_IMODE(reserve.stat().st_mode) == 0o600
    assert reserve.stat().st_size == APPLIED_REPORT_EMERGENCY_RESERVE_BYTES
    assert json.loads(path.read_text(encoding="utf-8"))["incomplete"] is True
    events = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "operation-started",
        "command-finished",
    ]
    assert events[1]["mutating"] is False


@pytest.mark.parametrize(
    ("free_bytes", "free_inodes", "expected"),
    [
        (
            MIN_APPLIED_STATE_FREE_BYTES - 4096,
            MIN_APPLIED_STATE_FREE_INODES,
            "insufficient free space",
        ),
        (
            MIN_APPLIED_STATE_FREE_BYTES,
            MIN_APPLIED_STATE_FREE_INODES - 1,
            "insufficient free inodes",
        ),
    ],
)
def test_applied_state_capacity_budget_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    free_bytes: int,
    free_inodes: int,
    expected: str,
) -> None:
    monkeypatch.setattr(
        report_module.os,
        "fstatvfs",
        lambda descriptor: SimpleNamespace(
            f_frsize=4096,
            f_bsize=4096,
            f_bavail=free_bytes // 4096,
            f_favail=free_inodes,
        ),
    )

    with pytest.raises(ReportWriteError, match=expected):
        require_applied_state_capacity(tmp_path)


def test_applied_state_capacity_accepts_the_exact_post_reservation_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report_module.os,
        "fstatvfs",
        lambda descriptor: SimpleNamespace(
            f_frsize=4096,
            f_bsize=4096,
            f_bavail=MIN_APPLIED_STATE_FREE_BYTES // 4096,
            f_favail=MIN_APPLIED_STATE_FREE_INODES,
        ),
    )

    require_applied_state_capacity(tmp_path, tmp_path / "future-snapshots")


def test_reservation_capacity_failure_removes_every_new_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        report_module,
        "APPLIED_REPORT_EMERGENCY_RESERVE_BYTES",
        4096,
    )
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    monkeypatch.setattr(
        report_module.os,
        "fstatvfs",
        lambda descriptor: SimpleNamespace(
            f_frsize=4096,
            f_bsize=4096,
            f_bavail=(MIN_APPLIED_STATE_FREE_BYTES // 4096) - 1,
            f_favail=MIN_APPLIED_STATE_FREE_INODES,
        ),
    )

    with pytest.raises(ReportWriteError, match="cannot reserve"):
        reserve_applied_report(report, path)

    assert not path.exists()
    assert not report_journal_path(path).exists()
    assert not report_emergency_reserve_path(path).exists()


@pytest.mark.parametrize(
    "capacity_errno",
    [errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)],
)
def test_journal_capacity_failure_consumes_reserve_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capacity_errno: int,
) -> None:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        report_module,
        "APPLIED_REPORT_EMERGENCY_RESERVE_BYTES",
        4096,
    )
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    reserve_applied_report(report, path)
    real_write = os.write
    injected = False

    def fail_first_event_write(fd: int, payload: bytes) -> int:
        nonlocal injected
        if not injected and b'"event":"command-started"' in payload:
            injected = True
            raise OSError(capacity_errno, "injected capacity exhaustion")
        return real_write(fd, payload)

    monkeypatch.setattr(report_module.os, "write", fail_first_event_write)

    append_report_journal(
        path,
        report.operation_id,
        "command-started",
        command=["apt-get", "install", "driver"],
        mutating=True,
    )

    assert injected is True
    assert not report_emergency_reserve_path(path).exists()
    events = [
        json.loads(line)
        for line in report_journal_path(path).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "operation-started",
        "command-started",
    ]


def test_partial_enospc_append_is_rolled_back_before_emergency_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        report_module,
        "APPLIED_REPORT_EMERGENCY_RESERVE_BYTES",
        4096,
    )
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    reserve_applied_report(report, path)
    original_journal = report_journal_path(path).read_bytes()
    real_write = os.write
    stage = 0

    def partially_fill_then_fail(fd: int, payload: bytes) -> int:
        nonlocal stage
        if b'"event":"command-started"' in payload and stage == 0:
            stage = 1
            partial_length = max(1, len(payload) // 2)
            return real_write(fd, payload[:partial_length])
        if stage == 1:
            stage = 2
            raise OSError(errno.ENOSPC, "injected capacity exhaustion")
        return real_write(fd, payload)

    monkeypatch.setattr(report_module.os, "write", partially_fill_then_fail)

    append_report_journal(
        path,
        report.operation_id,
        "command-started",
        command=["apt-get", "install", "driver"],
        mutating=True,
    )

    payload = report_journal_path(path).read_bytes()
    assert stage == 2
    assert payload.startswith(original_journal)
    assert payload.count(b'"event":"command-started"') == 1
    assert payload.endswith(b"\n")


def test_stale_reserve_cleanup_preserves_reports_and_journals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        report_module,
        "APPLIED_REPORT_EMERGENCY_RESERVE_BYTES",
        4096,
    )
    paths = [
        applied_report_path("install", None, operation_id)
        for operation_id in ("a" * 32, "b" * 32)
    ]
    for path in paths:
        reserve_applied_report(_provisional_report(path), path)

    monkeypatch.setattr(
        report_module.os,
        "listdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unbounded listdir used")
        ),
    )

    cleanup_stale_applied_report_reserves()

    for path in paths:
        assert path.is_file()
        assert report_journal_path(path).is_file()
        assert not report_emergency_reserve_path(path).exists()


def test_reservation_refuses_to_exceed_bounded_active_marker_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(report_module, "MAX_ACTIVE_JOURNAL_MARKERS", 1)
    monkeypatch.setattr(
        report_module,
        "APPLIED_REPORT_EMERGENCY_RESERVE_BYTES",
        4096,
    )
    first = applied_report_path("install", None, "a" * 32)
    second = applied_report_path("install", None, "b" * 32)
    reserve_applied_report(_provisional_report(first), first)

    with pytest.raises(ReportWriteError, match="reservation limit"):
        reserve_applied_report(_provisional_report(second), second)

    assert first.is_file()
    assert report_journal_path(first).is_file()
    assert not second.exists()
    assert not report_journal_path(second).exists()
    active = report_dir / report_module.ACTIVE_JOURNAL_DIRECTORY_NAME
    assert [path.name for path in active.iterdir()] == [
        report_journal_path(first).name
    ]


def test_release_reserve_rejects_paths_outside_the_report_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(report_module, "REPORT_DIR", tmp_path / "reports")

    with pytest.raises(ReportWriteError, match="stored directly"):
        release_applied_report_reserve(tmp_path / "outside.json")


def test_release_reserve_does_not_recreate_a_missing_report_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "missing-reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)

    with pytest.raises(ReportWriteError, match="cannot open trusted"):
        release_applied_report_reserve(report_dir / "install-test.json")

    assert not report_dir.exists()


def test_partial_completion_append_is_rolled_back_before_failure_marker(
    monkeypatch, tmp_path
) -> None:
    report_dir, path, report, snapshot_path = _snapshot_bound_mutation(
        monkeypatch, tmp_path
    )
    real_write = os.write
    append_fault_stage = 0

    def fail_after_partial_completion_write(fd: int, payload: bytes) -> int:
        nonlocal append_fault_stage
        if append_fault_stage == 0 and b'"event":"operation-completed"' in payload:
            append_fault_stage = 1
            partial_length = max(1, len(payload) // 2)
            return real_write(fd, payload[:partial_length])
        if append_fault_stage == 1:
            append_fault_stage = 2
            raise OSError("injected journal append failure")
        return real_write(fd, payload)

    monkeypatch.setattr(report_module.os, "write", fail_after_partial_completion_write)

    with pytest.raises(ReportWriteError, match="durably rolled back"):
        cli.emit_report("install", report, str(path), False, True)

    assert append_fault_stage == 2
    events = [
        json.loads(line)
        for line in report_journal_path(path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events][-1] == "report-persistence-failed"
    assert all(event["event"] != "operation-completed" for event in events)

    operations = unresolved_operations(report_dir=report_dir)
    assert len(operations) == 1
    operation = operations[0]
    assert operation.operation_id == report.operation_id
    assert operation.snapshot_path == snapshot_path
    assert operation.snapshot_integrity_sha256 == SNAPSHOT_INTEGRITY
    assert operation.snapshot_operation_id == report.operation_id
    assert operation.snapshot_host_id == SNAPSHOT_HOST_ID


def test_failed_partial_append_rollback_preserves_fail_closed_torn_tail(
    monkeypatch, tmp_path
) -> None:
    report_dir, path, report, snapshot_path = _snapshot_bound_mutation(
        monkeypatch, tmp_path
    )
    real_write = os.write
    real_ftruncate = os.ftruncate
    append_fault_stage = 0

    def fail_after_partial_completion_write(fd: int, payload: bytes) -> int:
        nonlocal append_fault_stage
        if append_fault_stage == 0 and b'"event":"operation-completed"' in payload:
            append_fault_stage = 1
            partial_length = max(1, len(payload) // 2)
            return real_write(fd, payload[:partial_length])
        if append_fault_stage == 1:
            append_fault_stage = 2
            raise OSError("injected journal append failure")
        return real_write(fd, payload)

    def fail_truncate(fd: int, length: int) -> None:
        del fd, length
        raise OSError("injected durable rollback failure")

    monkeypatch.setattr(report_module.os, "write", fail_after_partial_completion_write)
    monkeypatch.setattr(report_module.os, "ftruncate", fail_truncate)

    with pytest.raises(
        ReportJournalIntegrityError,
        match="cannot ensure applied report journal tail integrity",
    ):
        cli.emit_report("install", report, str(path), False, True)

    journal = report_journal_path(path)
    torn_payload = journal.read_bytes()
    assert append_fault_stage == 2
    assert not torn_payload.endswith(b"\n")
    assert b'"event":"report-persistence-failed"' not in torn_payload

    # A later recovery scan can conservatively discard the sole unterminated
    # append once durable truncation is available again.
    monkeypatch.setattr(report_module.os, "ftruncate", real_ftruncate)
    operations = unresolved_operations(report_dir=report_dir)
    assert len(operations) == 1
    assert operations[0].snapshot_path == snapshot_path
    assert operations[0].snapshot_integrity_sha256 == SNAPSHOT_INTEGRITY
    assert journal.read_bytes().endswith(b"\n")


def test_reserve_refuses_to_overwrite_existing_report(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    monkeypatch.setattr("nvidia_converge.report.REPORT_DIR", report_dir)
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ReportWriteError, match="cannot reserve"):
        reserve_applied_report(report, path)

    assert path.read_text(encoding="utf-8") == "existing\n"


def test_reserve_refuses_to_reuse_an_existing_emergency_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    reserve = report_emergency_reserve_path(path)
    reserve.write_bytes(b"existing reserve")
    reserve.chmod(0o600)

    with pytest.raises(ReportWriteError, match="cannot reserve"):
        reserve_applied_report(report, path)

    assert reserve.read_bytes() == b"existing reserve"
    assert not path.exists()
    assert not report_journal_path(path).exists()


def test_applied_report_directory_must_be_private(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(mode=0o755)
    report_dir.chmod(0o755)
    monkeypatch.setattr("nvidia_converge.report.REPORT_DIR", report_dir)

    with pytest.raises(ReportWriteError, match="private"):
        applied_report_path("install", None, "a" * 32)


def test_applied_report_directory_rejects_symlinked_ancestry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actual_state = tmp_path / "actual-state"
    actual_state.mkdir(mode=0o700)
    alias = tmp_path / "state-alias"
    alias.symlink_to(actual_state, target_is_directory=True)
    monkeypatch.setattr(report_module, "REPORT_DIR", alias / "reports")

    with pytest.raises(ReportWriteError, match="report directory"):
        applied_report_path("install", None, "a" * 32)

    assert not (actual_state / "reports").exists()


def test_applied_report_directory_rejects_writable_ancestry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe_state = tmp_path / "unsafe-state"
    unsafe_state.mkdir(mode=0o700)
    unsafe_state.chmod(0o777)
    monkeypatch.setattr(report_module, "REPORT_DIR", unsafe_state / "reports")

    with pytest.raises(ReportWriteError, match="group/world-writable"):
        applied_report_path("install", None, "a" * 32)

    assert not (unsafe_state / "reports").exists()


def test_journal_append_detects_report_directory_swap_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    displaced = tmp_path / "reports-before-swap"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        report_module,
        "APPLIED_REPORT_EMERGENCY_RESERVE_BYTES",
        4096,
    )
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    reserve_applied_report(report, path)
    journal_name = report_journal_path(path).name
    original_journal = (report_dir / journal_name).read_bytes()
    real_open_directory = report_module._open_report_directory
    swapped = False

    def swap_after_open(*, create: bool = False) -> int:
        nonlocal swapped
        descriptor = real_open_directory(create=create)
        if not swapped:
            swapped = True
            report_dir.rename(displaced)
            report_dir.mkdir(mode=0o700)
        return descriptor

    monkeypatch.setattr(
        report_module,
        "_open_report_directory",
        swap_after_open,
    )

    with pytest.raises(ReportWriteError, match="durably rolled back"):
        append_report_journal(
            path,
            report.operation_id,
            "command-started",
            command=["apt-get", "install", "driver"],
            mutating=True,
        )

    assert swapped is True
    assert list(report_dir.iterdir()) == []
    assert (displaced / journal_name).read_bytes() == original_journal


@pytest.mark.parametrize("command", ["install", "verify", "lock", "snapshot"])
def test_snapshot_creating_report_requires_operation_binding(command):
    report = _provisional_report(Path("/var/lib/nvidia-converge/reports/a.json"))
    report.command = command
    report.rollback = RollbackSnapshot(
        None,
        [],
        "6.8.0-test",
        None,
        [],
        operation_id="b" * 32,
    )

    with pytest.raises(ReportWriteError, match="operation_id"):
        report_json(report)


def test_rollback_report_can_embed_prior_snapshot_operation():
    report = _provisional_report(Path("/var/lib/nvidia-converge/reports/a.json"))
    report.command = "rollback"
    report.rollback = RollbackSnapshot(
        None,
        [],
        "6.8.0-test",
        None,
        [],
        operation_id="b" * 32,
    )

    assert json.loads(report_json(report))["rollback"]["operation_id"] == "b" * 32


def _provisional_report(path: Path) -> Report:
    timestamp = utc_now()
    return Report(
        "1.2",
        timestamp,
        DesiredState(),
        command="install",
        mode="apply",
        tool_version="0.1.0",
        operation_id="a" * 32,
        started_at=timestamp,
        completed_at=timestamp,
        duration_seconds=0.0,
        outcome="failed",
        exit_code=255,
        incomplete=True,
        report_path=str(path),
    )


def _snapshot_bound_mutation(
    monkeypatch,
    tmp_path: Path,
) -> tuple[Path, Path, Report, Path]:
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(cli, "_host_identity", lambda: SNAPSHOT_HOST_ID)
    path = applied_report_path("install", None, "a" * 32)
    report = _provisional_report(path)
    snapshot_path = tmp_path / "snapshots" / "baseline.json"
    report.rollback = RollbackSnapshot(
        str(snapshot_path),
        [],
        "6.8.0-test",
        None,
        [],
        operation_id=report.operation_id,
        host_id=SNAPSHOT_HOST_ID,
        architecture="x86_64",
        integrity_sha256=SNAPSHOT_INTEGRITY,
    )
    reserve_applied_report(report, path)
    append_report_journal(
        path,
        report.operation_id,
        "rollback-snapshot-persisted",
        snapshot_path=str(snapshot_path),
        snapshot_integrity_sha256=SNAPSHOT_INTEGRITY,
        snapshot_operation_id=report.operation_id,
        snapshot_host_id=SNAPSHOT_HOST_ID,
    )
    command = ["apt-get", "install", "-y", "nvidia-open"]
    append_report_journal(
        path,
        report.operation_id,
        "command-started",
        command=command,
        mutating=True,
    )
    append_report_journal(
        path,
        report.operation_id,
        "command-finished",
        command=command,
        mutating=True,
        returncode=0,
        skipped=False,
        reason=None,
    )
    return report_dir, path, report, snapshot_path
