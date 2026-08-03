from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nvidia_converge import files as files_module
from nvidia_converge import recovery
from nvidia_converge.recovery import (
    RecoveryStateError,
    recovery_snapshot_path,
    unresolved_operations,
)

OPERATION_ID = "a" * 32
RECOVERY_ID = "b" * 32
SNAPSHOT_INTEGRITY = "c" * 64
SNAPSHOT_PATH = "/var/lib/nvidia-converge/snapshots/baseline.json"
SNAPSHOT_HOST_ID = "machine-id-sha256:" + "e" * 64


def _event(name: str, **values: object) -> dict[str, object]:
    return {
        "event": name,
        "operation_id": OPERATION_ID,
        "timestamp": "2026-08-02T00:00:00+00:00",
        **values,
    }


def _started() -> dict[str, object]:
    return _event("operation-started")


def _snapshot() -> dict[str, object]:
    return _event(
        "rollback-snapshot-persisted",
        snapshot_path=SNAPSHOT_PATH,
        snapshot_integrity_sha256=SNAPSHOT_INTEGRITY,
        snapshot_operation_id=OPERATION_ID,
        snapshot_host_id=SNAPSHOT_HOST_ID,
    )


def _command_started(
    command: list[str], *, mutating: bool = True
) -> dict[str, object]:
    return _event("command-started", command=command, mutating=mutating)


def _command_finished(
    command: list[str], *, mutating: bool = True
) -> dict[str, object]:
    return _event(
        "command-finished",
        command=command,
        mutating=mutating,
        returncode=0,
        skipped=False,
        reason=None,
    )


def _recovered(**values: object) -> dict[str, object]:
    event = _event(
        "operation-recovered",
        recovery_operation_id=RECOVERY_ID,
        snapshot_path=SNAPSHOT_PATH,
        snapshot_integrity_sha256=SNAPSHOT_INTEGRITY,
        snapshot_operation_id=OPERATION_ID,
        snapshot_host_id=SNAPSHOT_HOST_ID,
    )
    event.update(values)
    return event


def _release(target: str, **values: object) -> dict[str, object]:
    event = _event(
        "launcher-release-authorized",
        release_target=target,
        snapshot_path=SNAPSHOT_PATH,
        snapshot_integrity_sha256=SNAPSHOT_INTEGRITY,
        snapshot_operation_id=OPERATION_ID,
        snapshot_host_id=SNAPSHOT_HOST_ID,
    )
    event.update(values)
    return event


def _completed(*, incomplete: bool) -> dict[str, object]:
    return _event(
        "operation-completed",
        exit_code=2 if incomplete else 0,
        incomplete=incomplete,
        outcome="failed" if incomplete else "succeeded",
    )


def _journal_dir(tmp_path: Path) -> Path:
    path = tmp_path / "reports"
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _write_journal(
    directory: Path,
    events: list[dict[str, object]],
    *,
    name: str = "install-test.json.journal.jsonl",
) -> Path:
    path = directory / name
    payload = "".join(
        json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
        for event in events
    )
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return path


def _append_events(path: Path, events: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def _write_final_report(
    directory: Path,
    journal_name: str,
    operation_id: str,
    *,
    exit_code: int = 0,
    outcome: str = "succeeded",
) -> Path:
    suffix = ".journal.jsonl"
    assert journal_name.endswith(suffix)
    report = directory / journal_name[: -len(suffix)]
    report.write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "incomplete": False,
                "mode": "apply",
                "operation_id": operation_id,
                "outcome": outcome,
                "report_path": str(report),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report.chmod(0o600)
    return report


def _write_active_marker(
    directory: Path,
    journal_name: str,
    operation_id: str,
) -> Path:
    active = directory / recovery.ACTIVE_JOURNAL_DIRECTORY_NAME
    active.mkdir(mode=0o700, exist_ok=True)
    marker = active / journal_name
    marker.write_text(
        json.dumps(
            {
                "journal": journal_name,
                "operation_id": operation_id,
                "schema_version": recovery.ACTIVE_JOURNAL_INDEX_SCHEMA,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o600)
    return marker


def test_missing_report_directory_has_no_recovery_state(tmp_path: Path) -> None:
    assert unresolved_operations(report_dir=tmp_path / "missing") == []


def test_recovery_rejects_symlinked_report_directory_ancestry(
    tmp_path: Path,
) -> None:
    actual_state = tmp_path / "actual-state"
    directory = actual_state / "reports"
    directory.mkdir(mode=0o700, parents=True)
    alias = tmp_path / "state-alias"
    alias.symlink_to(actual_state, target_is_directory=True)

    with pytest.raises(RecoveryStateError, match="cannot open"):
        unresolved_operations(report_dir=alias / "reports")


def test_recovery_rejects_writable_report_directory_ancestry(
    tmp_path: Path,
) -> None:
    unsafe_state = tmp_path / "unsafe-state"
    directory = unsafe_state / "reports"
    directory.mkdir(mode=0o700, parents=True)
    unsafe_state.chmod(0o777)

    with pytest.raises(RecoveryStateError, match="group/world-writable"):
        unresolved_operations(report_dir=directory)


def test_recovery_detects_report_directory_swap_while_opening(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    displaced = tmp_path / "reports-before-swap"
    journal = _write_journal(directory, [_started()])
    real_open = files_module.os.open
    swapped = False

    def swapping_open(
        target: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(target, flags, mode, dir_fd=dir_fd)
        if target == directory.name and dir_fd is not None and not swapped:
            swapped = True
            directory.rename(displaced)
            directory.mkdir(mode=0o700)
        return descriptor

    monkeypatch.setattr(files_module.os, "open", swapping_open)

    with pytest.raises(RecoveryStateError, match="changed while opening"):
        unresolved_operations(report_dir=directory)

    assert swapped is True
    assert list(directory.iterdir()) == []
    assert (displaced / journal.name).is_file()


def test_crash_before_host_mutation_does_not_block_recovery(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started(), _snapshot()])

    assert unresolved_operations(report_dir=directory) == []


def test_private_snapshot_persistence_is_not_a_host_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import rollback as rollback_module

    directory = _journal_dir(tmp_path)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(mode=0o700)
    snapshot_dir.chmod(0o700)
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_dir)
    _write_journal(
        directory,
        [
            _started(),
            _command_started(["persist-rollback-snapshot"]),
        ],
    )

    assert unresolved_operations(report_dir=directory) == []


def test_completed_non_incomplete_operation_is_resolved(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["systemctl", "mask", "docker.socket"]),
            _command_finished(["systemctl", "mask", "docker.socket"]),
            _completed(incomplete=False),
        ],
    )

    assert unresolved_operations(report_dir=directory) == []


def test_mutating_crash_returns_exact_snapshot_binding(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    journal = _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["apt-get", "install", "-y", "package"]),
        ],
    )

    operations = unresolved_operations(report_dir=directory)

    assert len(operations) == 1
    assert operations[0].journal_path == journal
    assert operations[0].operation_id == OPERATION_ID
    assert operations[0].snapshot_path == Path(SNAPSHOT_PATH)
    assert operations[0].snapshot_integrity_sha256 == SNAPSHOT_INTEGRITY
    assert operations[0].snapshot_operation_id == OPERATION_ID
    assert operations[0].snapshot_host_id == SNAPSHOT_HOST_ID
    assert operations[0].launcher_release_target is None
    assert recovery_snapshot_path(operations) == Path(SNAPSHOT_PATH)


def test_incomplete_completion_remains_unresolved(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started(), _snapshot(), _completed(incomplete=True)])

    assert len(unresolved_operations(report_dir=directory)) == 1


def test_recovery_terminal_event_resolves_incomplete_operation(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _completed(incomplete=True),
            _recovered(),
        ],
    )

    assert unresolved_operations(report_dir=directory) == []


def test_recovery_terminal_event_cannot_be_laundered_by_persistence_failure(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _completed(incomplete=True),
            _recovered(),
            _event("report-persistence-failed", error="unexpected append"),
        ],
    )

    with pytest.raises(RecoveryStateError, match="after its terminal"):
        unresolved_operations(report_dir=directory)


def test_torn_final_append_is_trimmed_to_valid_durable_prefix(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    journal = _write_journal(directory, [_started(), _snapshot()])
    durable_prefix = journal.read_bytes()
    with journal.open("ab") as handle:
        handle.write(b'{"command":["apt-get"],"event":"command-started"')

    assert unresolved_operations(report_dir=directory) == []
    assert journal.read_bytes() == durable_prefix


def test_torn_command_finish_preserves_unresolved_mutation(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    journal = _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["apt-get", "install", "-y", "package"]),
        ],
    )
    durable_prefix = journal.read_bytes()
    with journal.open("ab") as handle:
        handle.write(b'{"event":"command-finished"')

    operations = unresolved_operations(report_dir=directory)

    assert len(operations) == 1
    assert operations[0].snapshot_path == Path(SNAPSHOT_PATH)
    assert journal.read_bytes() == durable_prefix


@pytest.mark.parametrize(
    "prefix_events, tail_event, expected_unresolved",
    [
        ([_started()], _snapshot(), 0),
        (
            [_started(), _snapshot()],
            _command_started(["apt-get", "install", "-y", "package"]),
            0,
        ),
        (
            [
                _started(),
                _snapshot(),
                _command_started(["apt-get", "install", "-y", "package"]),
            ],
            _command_finished(["apt-get", "install", "-y", "package"]),
            1,
        ),
        (
            [
                _started(),
                _snapshot(),
                _command_started(["apt-get", "install", "-y", "package"]),
                _command_finished(["apt-get", "install", "-y", "package"]),
            ],
            _completed(incomplete=True),
            1,
        ),
        (
            [
                _started(),
                _snapshot(),
                _command_started(["systemctl", "mask", "docker.socket"]),
                _command_finished(["systemctl", "mask", "docker.socket"]),
            ],
            _release("install-target"),
            1,
        ),
        (
            [_started(), _snapshot(), _completed(incomplete=True)],
            _recovered(),
            1,
        ),
    ],
)
def test_every_torn_final_event_prefix_has_conservative_recovery_state(
    tmp_path: Path,
    prefix_events: list[dict[str, object]],
    tail_event: dict[str, object],
    expected_unresolved: int,
) -> None:
    tail = (
        json.dumps(tail_event, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    for cut in range(1, len(tail)):
        case_root = tmp_path / str(cut)
        case_root.mkdir()
        directory = _journal_dir(case_root)
        journal = _write_journal(directory, prefix_events)
        durable_prefix = journal.read_bytes()
        with journal.open("ab") as handle:
            handle.write(tail[:cut])

        operations = unresolved_operations(report_dir=directory)

        assert len(operations) == expected_unresolved
        assert journal.read_bytes() == durable_prefix


def test_orphan_command_completion_fails_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_finished(["apt-get", "install", "-y", "package"]),
        ],
    )

    with pytest.raises(RecoveryStateError, match="does not match"):
        unresolved_operations(report_dir=directory)


def test_mismatched_command_completion_fails_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["systemctl", "mask", "docker.socket"]),
            _command_finished(["systemctl", "mask", "docker.service"]),
        ],
    )

    with pytest.raises(RecoveryStateError, match="does not match"):
        unresolved_operations(report_dir=directory)


def test_conflicting_snapshot_binding_fails_even_if_completed(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    conflicting = {
        **_snapshot(),
        "snapshot_path": "/var/lib/nvidia-converge/snapshots/other.json",
    }
    _write_journal(
        directory,
        [_started(), _snapshot(), conflicting, _completed(incomplete=False)],
    )

    with pytest.raises(RecoveryStateError, match="duplicate"):
        unresolved_operations(report_dir=directory)


def test_incomplete_completion_cannot_be_laundered_by_clean_completion(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _completed(incomplete=True),
            _completed(incomplete=False),
        ],
    )

    with pytest.raises(RecoveryStateError, match="sealed incomplete"):
        unresolved_operations(report_dir=directory)


@pytest.mark.parametrize(
    "field, value",
    [
        (
            "snapshot_path",
            "/var/lib/nvidia-converge/snapshots/other.json",
        ),
        ("snapshot_integrity_sha256", "f" * 64),
        ("snapshot_operation_id", "d" * 32),
        ("snapshot_host_id", "hostname-sha256:" + "f" * 64),
    ],
)
def test_recovery_must_match_exact_snapshot_binding(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["systemctl", "mask", "docker.socket"]),
            _recovered(**{field: value}),
        ],
    )

    with pytest.raises(RecoveryStateError, match="invalid recovery state"):
        unresolved_operations(report_dir=directory)


def test_snapshot_creator_operation_can_differ_from_recovery_consumer(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    creator_id = "d" * 32
    binding = {**_snapshot(), "snapshot_operation_id": creator_id}
    recovered = {**_recovered(), "snapshot_operation_id": creator_id}
    _write_journal(
        directory,
        [
            _started(),
            binding,
            _command_started(["systemctl", "mask", "docker.socket"]),
            recovered,
        ],
    )

    assert unresolved_operations(report_dir=directory) == []


def test_launcher_release_authorization_is_snapshot_bound(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    command = ["systemctl", "mask", "--now", "docker.socket"]
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(command),
            _command_finished(command),
            _release("install-target"),
        ],
    )

    operations = unresolved_operations(report_dir=directory)

    assert len(operations) == 1
    assert operations[0].launcher_release_target == "install-target"


def test_verify_or_lock_operation_target_can_authorize_launcher_release(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    command = ["systemctl", "mask", "--now", "docker.socket"]
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(command),
            _command_finished(command),
            _release("operation-target"),
        ],
    )

    operations = unresolved_operations(report_dir=directory)

    assert len(operations) == 1
    assert operations[0].launcher_release_target == "operation-target"


def test_install_release_can_be_followed_by_rollback_release(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    target_command = ["systemctl", "mask", "--now", "docker.socket"]
    rollback_command = ["apt-get", "install", "-y", "baseline"]
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(target_command),
            _command_finished(target_command),
            _release("install-target"),
            _command_started(rollback_command),
            _command_finished(rollback_command),
            _release("rollback-baseline"),
        ],
    )

    operations = unresolved_operations(report_dir=directory)

    assert len(operations) == 1
    assert operations[0].launcher_release_target == "rollback-baseline"


def test_launcher_release_authorization_rejects_mismatch_and_bad_order(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    command = ["systemctl", "mask", "--now", "docker.socket"]
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(command),
            _command_finished(command),
            _release("rollback-baseline"),
            _release("install-target"),
        ],
    )

    with pytest.raises(RecoveryStateError, match="release order"):
        unresolved_operations(report_dir=directory)


def test_persistence_failure_supersedes_uncertain_clean_completion(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    command = ["systemctl", "mask", "--now", "docker.socket"]
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(command),
            _command_finished(command),
            _completed(incomplete=False),
            _event("report-persistence-failed", error="fsync failed"),
        ],
    )

    operations = unresolved_operations(report_dir=directory)

    assert len(operations) == 1
    assert operations[0].snapshot_path == Path(SNAPSHOT_PATH)


def test_unresolved_mutation_without_snapshot_binding_fails_closed(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [_started(), _command_started(["systemctl", "mask", "docker.socket"])],
    )

    with pytest.raises(RecoveryStateError, match="before rollback snapshot"):
        unresolved_operations(report_dir=directory)


def test_multiple_unresolved_snapshots_are_ambiguous(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["systemctl", "mask", "docker.socket"]),
        ],
        name="install-one.json.journal.jsonl",
    )
    second_id = "d" * 32
    second_events = [
        {**_started(), "operation_id": second_id},
        {
            **_snapshot(),
            "operation_id": second_id,
            "snapshot_operation_id": second_id,
            "snapshot_path": "/var/lib/nvidia-converge/snapshots/other.json",
        },
        {
            **_command_started(["dnf", "install", "-y", "package"]),
            "operation_id": second_id,
        },
    ]
    _write_journal(
        directory,
        second_events,
        name="install-two.json.journal.jsonl",
    )

    operations = unresolved_operations(report_dir=directory)

    with pytest.raises(RecoveryStateError, match="one unambiguous"):
        recovery_snapshot_path(operations)


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"", "empty or unrepairable"),
        (b'{"event":"operation-started"}', "no complete durable record"),
        (
            b'{"event":"operation-started","event":"operation-started"}\n',
            "invalid JSON",
        ),
        (b"[]\n", "event is not an object"),
    ],
)
def test_malformed_journal_fails_closed(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    directory = _journal_dir(tmp_path)
    path = directory / "install-test.json.journal.jsonl"
    path.write_bytes(payload)
    path.chmod(0o600)

    with pytest.raises(RecoveryStateError, match=message):
        unresolved_operations(report_dir=directory)


def test_events_after_terminal_completion_fail_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(
        directory,
        [
            _started(),
            _completed(incomplete=False),
            _command_started(["systemctl", "start", "docker.service"]),
        ],
    )

    with pytest.raises(RecoveryStateError, match="after its terminal"):
        unresolved_operations(report_dir=directory)


def test_unknown_event_fails_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started(), _event("future-event")])

    with pytest.raises(RecoveryStateError, match="unknown event"):
        unresolved_operations(report_dir=directory)


def test_insecure_or_linked_journal_fails_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    journal = _write_journal(directory, [_started()])
    journal.chmod(0o644)

    with pytest.raises(RecoveryStateError, match="private singly linked"):
        unresolved_operations(report_dir=directory)

    journal.chmod(0o600)
    os.link(journal, directory / "journal-hardlink")
    with pytest.raises(RecoveryStateError, match="private singly linked"):
        unresolved_operations(report_dir=directory)


def test_symlink_journal_fails_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    source = tmp_path / "source"
    source.write_text("payload\n", encoding="utf-8")
    (directory / "install-test.json.journal.jsonl").symlink_to(source)

    with pytest.raises(RecoveryStateError, match="cannot open"):
        unresolved_operations(report_dir=directory)


def test_unsafe_journal_name_fails_closed(tmp_path: Path) -> None:
    directory = _journal_dir(tmp_path)
    path = directory / "-unsafe.json.journal.jsonl"
    path.write_text("payload\n", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(RecoveryStateError, match="unsafe journal name"):
        unresolved_operations(report_dir=directory)


def test_journal_count_bound_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    first = [
        _started(),
        _snapshot(),
        _command_started(["systemctl", "mask", "docker.socket"]),
    ]
    second_id = "d" * 32
    second = [
        {**event, "operation_id": second_id}
        for event in (
            _started(),
            _snapshot(),
            _command_started(["systemctl", "mask", "docker.service"]),
        )
    ]
    _write_journal(directory, first, name="one.json.journal.jsonl")
    _write_journal(directory, second, name="two.json.journal.jsonl")
    monkeypatch.setattr(recovery, "MAX_RECOVERY_JOURNALS", 1)

    with pytest.raises(RecoveryStateError, match="operation count exceeds"):
        unresolved_operations(report_dir=directory)


def test_completed_history_does_not_consume_unresolved_scan_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    first = [_started(), _completed(incomplete=False)]
    second_id = "d" * 32
    second = [
        {**event, "operation_id": second_id}
        for event in (_started(), _completed(incomplete=False))
    ]
    _write_journal(directory, first, name="one.json.journal.jsonl")
    _write_journal(directory, second, name="two.json.journal.jsonl")
    monkeypatch.setattr(recovery, "MAX_RECOVERY_JOURNALS", 1)

    assert unresolved_operations(report_dir=directory) == []


def test_active_index_bootstrap_is_idempotent_after_crash_before_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    journal = _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(["apt-get", "install", "-y", "package"]),
        ],
    )
    real_create_index = recovery._create_active_index

    def crash_before_sentinel(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated bootstrap crash")

    monkeypatch.setattr(recovery, "_create_active_index", crash_before_sentinel)
    with pytest.raises(RuntimeError, match="simulated bootstrap crash"):
        unresolved_operations(report_dir=directory)

    active = directory / recovery.ACTIVE_JOURNAL_DIRECTORY_NAME
    assert (active / journal.name).is_file()
    assert not (active / recovery.ACTIVE_JOURNAL_INDEX_NAME).exists()

    monkeypatch.setattr(recovery, "_create_active_index", real_create_index)
    operations = unresolved_operations(report_dir=directory)

    assert [operation.operation_id for operation in operations] == [OPERATION_ID]
    assert (active / recovery.ACTIVE_JOURNAL_INDEX_NAME).is_file()


def test_terminal_marker_waits_for_final_report_then_is_durably_retired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    command = ["apt-get", "install", "-y", "package"]
    journal = _write_journal(
        directory,
        [_started(), _snapshot(), _command_started(command)],
    )
    assert len(unresolved_operations(report_dir=directory)) == 1
    marker = (
        directory / recovery.ACTIVE_JOURNAL_DIRECTORY_NAME / journal.name
    )
    assert marker.is_file()

    _append_events(
        journal,
        [_command_finished(command), _completed(incomplete=False)],
    )
    # The terminal journal is sufficient not to block, but its active marker
    # remains until the matching final report is itself durable.
    assert unresolved_operations(report_dir=directory) == []
    assert marker.is_file()

    _write_final_report(directory, journal.name, OPERATION_ID)
    active_metadata = marker.parent.stat()
    real_fsync = os.fsync
    active_fsync_seen = False

    def recording_fsync(descriptor: int) -> None:
        nonlocal active_fsync_seen
        if os.path.samestat(os.fstat(descriptor), active_metadata):
            active_fsync_seen = True
        real_fsync(descriptor)

    monkeypatch.setattr(recovery.os, "fsync", recording_fsync)
    assert unresolved_operations(report_dir=directory) == []

    assert not marker.exists()
    assert active_fsync_seen is True


def test_recovery_marker_waits_for_completed_recovery_operation(
    tmp_path: Path,
) -> None:
    directory = _journal_dir(tmp_path)
    original_command = ["apt-get", "install", "-y", "package"]
    original = _write_journal(
        directory,
        [
            _started(),
            _snapshot(),
            _command_started(original_command),
            _command_finished(original_command),
            _completed(incomplete=True),
        ],
    )
    assert len(unresolved_operations(report_dir=directory)) == 1
    _append_events(original, [_recovered()])

    recovery_name = "rollback-recovery.json.journal.jsonl"
    recovery_command = ["systemctl", "mask", "docker.service"]
    recovery_events = [
        {**event, "operation_id": RECOVERY_ID}
        for event in (
            _started(),
            _snapshot(),
            _command_started(recovery_command),
        )
    ]
    recovery_journal = _write_journal(
        directory,
        recovery_events,
        name=recovery_name,
    )
    _write_active_marker(directory, recovery_name, RECOVERY_ID)

    operations = unresolved_operations(report_dir=directory)
    assert [operation.operation_id for operation in operations] == [RECOVERY_ID]
    assert (
        directory
        / recovery.ACTIVE_JOURNAL_DIRECTORY_NAME
        / original.name
    ).is_file()

    _append_events(
        recovery_journal,
        [
            {
                **event,
                "operation_id": RECOVERY_ID,
            }
            for event in (
                _command_finished(recovery_command),
                _completed(incomplete=False),
            )
        ],
    )
    _write_final_report(directory, recovery_name, RECOVERY_ID)

    assert unresolved_operations(report_dir=directory) == []
    active = directory / recovery.ACTIVE_JOURNAL_DIRECTORY_NAME
    assert sorted(path.name for path in active.iterdir()) == [
        recovery.ACTIVE_JOURNAL_INDEX_NAME
    ]


def test_safe_package_payload_staging_crash_does_not_require_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import rollback as rollback_module

    directory = _journal_dir(tmp_path)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(mode=0o700)
    snapshot_dir.chmod(0o700)
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_dir)
    _write_journal(
        directory,
        [_started(), _command_started(["stage-package-payloads"])],
    )

    assert unresolved_operations(report_dir=directory) == []


def test_pre_mutation_stage_crash_removes_only_exact_unbound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import rollback as rollback_module

    directory = _journal_dir(tmp_path)
    assert unresolved_operations(report_dir=directory) == []
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(mode=0o700)
    snapshot_dir.chmod(0o700)
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_dir)

    snapshot = snapshot_dir / f"snapshot-{OPERATION_ID}.json"
    snapshot_temp = snapshot_dir / f".{snapshot.name}.tmp"
    bundle = snapshot_dir / f"{snapshot.name}.payloads"
    bundle_temp = snapshot_dir / f".{bundle.name}.tmp"
    bundle_incoming = snapshot_dir / f".{bundle.name}.incoming"
    for path in (snapshot, snapshot_temp):
        path.write_text("interrupted\n", encoding="utf-8")
        path.chmod(0o600)
    for path in (bundle, bundle_temp, bundle_incoming):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        child = path / "payload"
        child.write_text("private\n", encoding="utf-8")
        child.chmod(0o600)
    unrelated = snapshot_dir / f"snapshot-{'d' * 32}.json"
    unrelated.write_text("unrelated\n", encoding="utf-8")
    unrelated.chmod(0o600)

    journal = _write_journal(
        directory,
        [_started(), _command_started(["stage-package-payloads"])],
    )
    marker = _write_active_marker(directory, journal.name, OPERATION_ID)

    assert unresolved_operations(report_dir=directory) == []
    assert not marker.exists()
    assert all(
        not path.exists()
        for path in (snapshot, snapshot_temp, bundle, bundle_temp, bundle_incoming)
    )
    assert unrelated.read_text(encoding="utf-8") == "unrelated\n"


def test_pre_mutation_snapshot_binding_preserves_authority_and_cleans_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import rollback as rollback_module

    directory = _journal_dir(tmp_path)
    assert unresolved_operations(report_dir=directory) == []
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(mode=0o700)
    snapshot_dir.chmod(0o700)
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_dir)

    snapshot = snapshot_dir / f"snapshot-{OPERATION_ID}.json"
    snapshot.write_text("bound snapshot\n", encoding="utf-8")
    snapshot.chmod(0o600)
    bundle = snapshot_dir / f"{snapshot.name}.payloads"
    bundle.mkdir(mode=0o700)
    bundle.chmod(0o700)
    payload = bundle / "payload"
    payload.write_text("bound payload\n", encoding="utf-8")
    payload.chmod(0o600)
    snapshot_temp = snapshot_dir / f".{snapshot.name}.tmp"
    snapshot_temp.write_text("temporary\n", encoding="utf-8")
    snapshot_temp.chmod(0o600)
    bundle_temp = snapshot_dir / f".{bundle.name}.tmp"
    bundle_incoming = snapshot_dir / f".{bundle.name}.incoming"
    for path in (bundle_temp, bundle_incoming):
        path.mkdir(mode=0o700)
        path.chmod(0o700)

    stage = ["stage-package-payloads"]
    persist = ["persist-rollback-snapshot"]
    journal = _write_journal(
        directory,
        [
            _started(),
            _command_started(stage),
            _command_finished(stage),
            _command_started(persist),
            _command_finished(persist),
            {**_snapshot(), "snapshot_path": str(snapshot)},
        ],
    )
    marker = _write_active_marker(directory, journal.name, OPERATION_ID)

    assert unresolved_operations(report_dir=directory) == []
    assert not marker.exists()
    assert snapshot.read_text(encoding="utf-8") == "bound snapshot\n"
    assert payload.read_text(encoding="utf-8") == "bound payload\n"
    assert not snapshot_temp.exists()
    assert not bundle_temp.exists()
    assert not bundle_incoming.exists()


def test_unsafe_pre_mutation_artifact_keeps_active_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import rollback as rollback_module

    directory = _journal_dir(tmp_path)
    assert unresolved_operations(report_dir=directory) == []
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(mode=0o700)
    snapshot_dir.chmod(0o700)
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_dir)
    snapshot = snapshot_dir / f"snapshot-{OPERATION_ID}.json"
    bundle_temp = snapshot_dir / f".{snapshot.name}.payloads.tmp"
    bundle_temp.mkdir(mode=0o700)
    bundle_temp.chmod(0o700)
    unsafe = bundle_temp / "link"
    unsafe.symlink_to(tmp_path / "outside")

    journal = _write_journal(
        directory,
        [_started(), _command_started(["stage-package-payloads"])],
    )
    marker = _write_active_marker(directory, journal.name, OPERATION_ID)

    with pytest.raises(
        RecoveryStateError,
        match="cannot clean safe pre-mutation rollback artifacts",
    ):
        unresolved_operations(report_dir=directory)
    assert marker.is_file()
    assert unsafe.is_symlink()


def test_bounded_pre_mutation_cleanup_failure_keeps_active_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import package_payloads
    from nvidia_converge import rollback as rollback_module

    directory = _journal_dir(tmp_path)
    assert unresolved_operations(report_dir=directory) == []
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(mode=0o700)
    snapshot_dir.chmod(0o700)
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(package_payloads, "MAX_PACKAGE_PAYLOAD_CLEANUP_ENTRIES", 1)
    snapshot = snapshot_dir / f"snapshot-{OPERATION_ID}.json"
    bundle_temp = snapshot_dir / f".{snapshot.name}.payloads.tmp"
    bundle_temp.mkdir(mode=0o700)
    bundle_temp.chmod(0o700)
    child = bundle_temp / "payload"
    child.write_text("private\n", encoding="utf-8")
    child.chmod(0o600)

    journal = _write_journal(
        directory,
        [_started(), _command_started(["stage-package-payloads"])],
    )
    marker = _write_active_marker(directory, journal.name, OPERATION_ID)

    with pytest.raises(RecoveryStateError, match="cleanup entry limit"):
        unresolved_operations(report_dir=directory)
    assert marker.is_file()
    assert child.is_file()


@pytest.mark.parametrize(
    "command",
    [
        ["stage-package-payloads", "--extra"],
        ["apt-get", "stage-package-payloads"],
        ["dnf", "download"],
    ],
)
def test_only_exact_package_payload_staging_is_safe_private_state(
    tmp_path: Path,
    command: list[str],
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started(), _command_started(command)])

    with pytest.raises(RecoveryStateError, match="before rollback snapshot"):
        unresolved_operations(report_dir=directory)


def test_pre_mutation_crash_markers_do_not_accumulate_at_the_active_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    # Establish the durable index before simulating later reservations.
    assert unresolved_operations(report_dir=directory) == []
    monkeypatch.setattr(recovery, "MAX_RECOVERY_JOURNALS", 1)

    for index, operation_id in enumerate(("a" * 32, "b" * 32, "c" * 32)):
        name = f"snapshot-{index}.json.journal.jsonl"
        _write_journal(
            directory,
            [{**_started(), "operation_id": operation_id}],
            name=name,
        )
        marker = _write_active_marker(directory, name, operation_id)

        assert unresolved_operations(report_dir=directory) == []
        assert not marker.exists()


def test_recovery_index_never_uses_unbounded_listdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started()])

    monkeypatch.setattr(
        recovery.os,
        "listdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unbounded listdir used")
        ),
    )

    assert unresolved_operations(report_dir=directory) == []


def test_legacy_bootstrap_entry_limit_is_enforced_before_sorting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started()])
    monkeypatch.setattr(recovery, "MAX_LEGACY_DIRECTORY_ENTRIES", 1)

    def unexpected_sort(*_args, **_kwargs):
        raise AssertionError("legacy names were sorted after exceeding the scan cap")

    monkeypatch.setattr(recovery, "sorted", unexpected_sort, raising=False)

    with pytest.raises(RecoveryStateError, match="bootstrap entry limit"):
        unresolved_operations(report_dir=directory)


def test_legacy_bootstrap_journal_count_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started()], name="one.json.journal.jsonl")
    _write_journal(directory, [_started()], name="two.json.journal.jsonl")
    monkeypatch.setattr(recovery, "MAX_LEGACY_JOURNALS", 1)

    with pytest.raises(RecoveryStateError, match="bootstrap journal limit"):
        unresolved_operations(report_dir=directory)


def test_legacy_bootstrap_total_journal_bytes_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _journal_dir(tmp_path)
    _write_journal(directory, [_started()])
    monkeypatch.setattr(recovery, "MAX_LEGACY_JOURNAL_TOTAL_BYTES", 1)

    with pytest.raises(RecoveryStateError, match="bootstrap byte limit"):
        unresolved_operations(report_dir=directory)
