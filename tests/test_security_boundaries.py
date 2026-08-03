from __future__ import annotations

import io
import json
import os
import shutil
import signal
import stat
import time
from pathlib import Path

import pytest

import nvidia_converge.audit as audit_module
import nvidia_converge.files as files_module
import nvidia_converge.rollback as rollback_module
import nvidia_converge.runner as runner_module
from nvidia_converge.audit import _secure_boot_state
from nvidia_converge.cli import main
from nvidia_converge.desired import (
    MAX_DESIRED_BYTES,
    DesiredConfigError,
    load_desired,
)
from nvidia_converge.models import (
    CommandResult,
    HostAudit,
    KernelInfo,
    ModuleInfo,
    RollbackSnapshot,
    RuntimeInfo,
)
from nvidia_converge.rollback import (
    MAX_SNAPSHOT_BYTES,
    RollbackSnapshotError,
    load_snapshot,
    validate_snapshot_for_apply,
)
from nvidia_converge.runner import TRUSTED_EXECUTABLE_PATH, CommandRunner


def test_desired_input_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "desired.yaml"
    path.write_bytes(b" " * (MAX_DESIRED_BYTES + 1))

    with pytest.raises(DesiredConfigError, match=rf"exceeds {MAX_DESIRED_BYTES} bytes"):
        load_desired(str(path))


def test_desired_input_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "desired.yaml"
    path.write_bytes(b"desired:\n  driver: \xff\n")

    with pytest.raises(DesiredConfigError, match="not valid UTF-8"):
        load_desired(str(path))


def test_desired_input_does_not_follow_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("desired: {}\n", encoding="utf-8")
    path = tmp_path / "desired.yaml"
    path.symlink_to(target)

    with pytest.raises(DesiredConfigError, match="cannot read desired-state file"):
        load_desired(str(path))


def test_desired_input_requires_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "desired.yaml"
    path.mkdir()

    with pytest.raises(DesiredConfigError, match="regular file"):
        load_desired(str(path))


def test_applied_desired_input_requires_root_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    _override_open_file_metadata(monkeypatch, uid=1000, mode=0o600)

    with pytest.raises(DesiredConfigError, match=r"owned by root.*observed uid 1000"):
        load_desired(str(path), require_root_controlled=True)


@pytest.mark.parametrize("mode", [0o620, 0o602, 0o622])
def test_applied_desired_input_rejects_group_or_world_write_bits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: int
) -> None:
    path = tmp_path / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    _override_open_file_metadata(monkeypatch, uid=0, mode=mode)

    with pytest.raises(
        DesiredConfigError,
        match=rf"must not be group/world-writable.*observed mode {mode:04o}",
    ):
        load_desired(str(path), require_root_controlled=True)


def test_applied_desired_input_accepts_root_owned_nonwritable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    _override_open_file_metadata(monkeypatch, uid=0, mode=0o644)

    desired = load_desired(str(path), require_root_controlled=True)

    assert desired.driver == "580-open"


def test_applied_desired_path_swap_fails_closed_when_open_file_is_unlinked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text(
        "desired:\n  driver: 595.71.05\n",
        encoding="utf-8",
    )
    _override_open_file_metadata(monkeypatch, uid=0, mode=0o644)
    real_open = files_module.os.open
    real_replace = files_module.os.replace
    swapped = False

    def swapping_open(
        opened_path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        fd = real_open(opened_path, flags, mode, dir_fd=dir_fd)
        if opened_path == path.name and dir_fd is not None and not swapped:
            real_replace(replacement, path)
            swapped = True
        return fd

    monkeypatch.setattr(files_module.os, "open", swapping_open)

    with pytest.raises(DesiredConfigError, match="input must be singly linked"):
        load_desired(str(path), require_root_controlled=True)

    assert swapped is True


def test_applied_desired_rejects_symlinked_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actual_parent = tmp_path.resolve() / "actual"
    actual_parent.mkdir()
    path = actual_parent / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    alias = tmp_path.resolve() / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)
    _override_open_file_metadata(monkeypatch, uid=0, mode=0o644)

    with pytest.raises(DesiredConfigError, match="cannot read desired-state file"):
        load_desired(
            str(alias / path.name),
            require_root_controlled=True,
        )


def test_applied_desired_rejects_writable_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe_parent = tmp_path.resolve() / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    path = unsafe_parent / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    _override_open_file_metadata(monkeypatch, uid=0, mode=0o644)

    with pytest.raises(DesiredConfigError, match="group/world-writable"):
        load_desired(str(path), require_root_controlled=True)


def test_applied_desired_rejects_non_root_ancestor_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path.resolve() / "untrusted-owner"
    parent.mkdir(mode=0o700)
    path = parent / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    parent_metadata = parent.stat()
    real_fstat = files_module.os.fstat

    def controlled_fstat(fd: int) -> os.stat_result:
        metadata = real_fstat(fd)
        fields = list(metadata)
        if stat.S_ISREG(metadata.st_mode):
            fields[stat.ST_UID] = 0
        elif os.path.samestat(metadata, parent_metadata):
            fields[stat.ST_UID] = 12345
        return os.stat_result(fields)

    monkeypatch.setattr(files_module.os, "fstat", controlled_fstat)

    with pytest.raises(DesiredConfigError, match="unsafe owner uid"):
        load_desired(str(path), require_root_controlled=True)


def test_applied_desired_detects_parent_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    parent = root / "managed"
    displaced = root / "managed-before-swap"
    parent.mkdir()
    path = parent / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    _override_open_file_metadata(monkeypatch, uid=0, mode=0o644)
    real_open = files_module.os.open
    swapped = False

    def swapping_open(
        opened_path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        fd = real_open(opened_path, flags, mode, dir_fd=dir_fd)
        if opened_path == path.name and dir_fd is not None and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            (parent / path.name).write_text(
                "desired:\n  driver: 595.71.05\n",
                encoding="utf-8",
            )
        return fd

    monkeypatch.setattr(files_module.os, "open", swapping_open)

    with pytest.raises(DesiredConfigError, match="parent changed"):
        load_desired(str(path), require_root_controlled=True)


def test_dry_run_desired_input_remains_user_writable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "desired.yaml"
    path.write_text("desired:\n  driver: 580-open\n", encoding="utf-8")
    _override_open_file_metadata(monkeypatch, uid=1000, mode=0o666)

    desired = load_desired(str(path))

    assert desired.driver == "580-open"


def test_applied_cli_requests_a_root_controlled_desired_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def reject_untrusted(
        path: str | None, *, require_root_controlled: bool = False
    ) -> None:
        observed.update(
            path=path,
            require_root_controlled=require_root_controlled,
        )
        raise DesiredConfigError("unsafe desired-state ownership")

    monkeypatch.setattr("nvidia_converge.cli.os.geteuid", lambda: 0)
    monkeypatch.setattr("nvidia_converge.cli.load_desired", reject_untrusted)

    rc = main(["lock", "--apply", "--desired", "/tmp/desired.yaml"])

    captured = capsys.readouterr()
    assert rc == 2
    assert observed == {
        "path": "/tmp/desired.yaml",
        "require_root_controlled": True,
    }
    assert captured.out == ""
    assert captured.err == "error: unsafe desired-state ownership\n"


def test_snapshot_input_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_bytes(b" " * (MAX_SNAPSHOT_BYTES + 1))

    with pytest.raises(
        RollbackSnapshotError,
        match=rf"exceeds {MAX_SNAPSHOT_BYTES} bytes",
    ):
        load_snapshot(str(path))


def test_snapshot_input_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_bytes(b'{"schema_version":"\xff"}')

    with pytest.raises(RollbackSnapshotError, match="not valid UTF-8"):
        load_snapshot(str(path))


def test_snapshot_input_does_not_follow_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    path = tmp_path / "snapshot.json"
    path.symlink_to(target)

    with pytest.raises(RollbackSnapshotError, match="cannot read rollback snapshot"):
        load_snapshot(str(path))


def test_snapshot_input_requires_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.mkdir()

    with pytest.raises(RollbackSnapshotError, match="regular file"):
        load_snapshot(str(path))


def test_snapshot_input_rejects_symlinked_parent(tmp_path: Path) -> None:
    actual_parent = tmp_path.resolve() / "actual"
    actual_parent.mkdir()
    (actual_parent / "snapshot.json").write_text("{}\n", encoding="utf-8")
    alias = tmp_path.resolve() / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(RollbackSnapshotError, match="cannot read rollback snapshot"):
        load_snapshot(str(alias / "snapshot.json"))


def test_snapshot_input_rejects_writable_ancestor(tmp_path: Path) -> None:
    unsafe_parent = tmp_path.resolve() / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    path = unsafe_parent / "snapshot.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RollbackSnapshotError, match="group/world-writable"):
        load_snapshot(str(path))


def test_snapshot_input_detects_parent_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    parent = root / "snapshots"
    displaced = root / "snapshots-before-swap"
    parent.mkdir()
    path = parent / "snapshot.json"
    path.write_text("{}\n", encoding="utf-8")
    real_open = files_module.os.open
    swapped = False

    def swapping_open(
        opened_path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        fd = real_open(opened_path, flags, mode, dir_fd=dir_fd)
        if opened_path == path.name and dir_fd is not None and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            (parent / path.name).write_text("{}\n", encoding="utf-8")
        return fd

    monkeypatch.setattr(files_module.os, "open", swapping_open)

    with pytest.raises(RollbackSnapshotError, match="parent changed"):
        load_snapshot(str(path))


def test_non_efi_boot_is_reported_as_secure_boot_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audit_module, "EFI_FIRMWARE_PATH", tmp_path / "missing-efi")

    assert _secure_boot_state(_UnexpectedRunner()) is False


def test_efi_boot_with_unreadable_state_remains_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    efi_path = tmp_path / "efi"
    variables_path = efi_path / "efivars"
    variables_path.mkdir(parents=True)
    monkeypatch.setattr(audit_module, "EFI_FIRMWARE_PATH", efi_path)
    monkeypatch.setattr(audit_module, "EFI_VARIABLE_PATH", variables_path)
    runner = _SecureBootRunner(CommandResult(["mokutil", "--sb-state"], 1))

    assert _secure_boot_state(runner) is None
    assert runner.commands == [["mokutil", "--sb-state"]]


@pytest.mark.parametrize(("value", "expected"), [(0, False), (1, True)])
def test_efi_secure_boot_variable_is_used_as_a_bounded_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: int,
    expected: bool,
) -> None:
    efi_path = tmp_path / "efi"
    variables_path = efi_path / "efivars"
    variables_path.mkdir(parents=True)
    (variables_path / "SecureBoot-test").write_bytes(b"\x07\x00\x00\x00" + bytes([value]))
    monkeypatch.setattr(audit_module, "EFI_FIRMWARE_PATH", efi_path)
    monkeypatch.setattr(audit_module, "EFI_VARIABLE_PATH", variables_path)

    assert _secure_boot_state(_SecureBootRunner(None, has_mokutil=False)) is expected


@pytest.fixture
def trusted_snapshot_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[RollbackSnapshot, Path, Path, HostAudit]:
    from test_rollback import _snapshot_document

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir(mode=0o700)
    snapshot_root.chmod(0o700)
    source = snapshot_root / "snapshot.json"
    host_id = "machine-id:" + "a" * 32
    audit = _trust_audit()
    monkeypatch.setattr(rollback_module, "SNAPSHOT_DIR", snapshot_root)
    monkeypatch.setattr(rollback_module, "_host_identity", lambda: host_id)
    source.write_text(
        json.dumps(_snapshot_document(source, packages=[])),
        encoding="utf-8",
    )
    source.chmod(0o600)
    snapshot = load_snapshot(str(source), require_private=True)
    return snapshot, source, snapshot_root, audit


def test_snapshot_apply_accepts_private_bound_file(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context

    validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_accepts_repeated_validation_of_loaded_private_file(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context

    validate_snapshot_for_apply(snapshot, str(source), audit)
    validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_file_outside_snapshot_root(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
    tmp_path: Path,
) -> None:
    snapshot, _, _, audit = trusted_snapshot_context
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    outside.chmod(0o600)
    snapshot.path = str(outside)

    with pytest.raises(RollbackSnapshotError, match="stored directly"):
        validate_snapshot_for_apply(snapshot, str(outside), audit)


def test_snapshot_apply_rejects_path_binding_mismatch(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
) -> None:
    snapshot, source, snapshot_root, audit = trusted_snapshot_context
    snapshot.path = str(snapshot_root / "different.json")

    with pytest.raises(RollbackSnapshotError, match="path binding"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_group_readable_file(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context
    source.chmod(0o640)

    with pytest.raises(RollbackSnapshotError, match="private"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_non_private_directory(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
) -> None:
    snapshot, source, snapshot_root, audit = trusted_snapshot_context
    snapshot_root.chmod(0o755)

    with pytest.raises(RollbackSnapshotError, match="directory must be private"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_hard_linked_file(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
    tmp_path: Path,
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context
    os.link(source, tmp_path / "snapshot-hard-link.json")
    assert source.stat().st_nlink == 2

    with pytest.raises(RollbackSnapshotError, match="singly linked"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_wrong_owner(
    monkeypatch: pytest.MonkeyPatch,
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context
    monkeypatch.setattr(rollback_module.os, "geteuid", lambda: source.stat().st_uid + 1)

    with pytest.raises(RollbackSnapshotError, match="owned by the effective uid"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_loaded_target_replacement(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
    tmp_path: Path,
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context
    payload = source.read_text(encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text(payload, encoding="utf-8")
    replacement.chmod(0o600)
    os.replace(replacement, source)

    with pytest.raises(RollbackSnapshotError, match="changed after loading"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_snapshot_apply_rejects_loaded_parent_replacement(
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
    tmp_path: Path,
) -> None:
    snapshot, source, snapshot_root, audit = trusted_snapshot_context
    payload = source.read_text(encoding="utf-8")
    displaced = tmp_path / "snapshots-before-swap"
    snapshot_root.rename(displaced)
    snapshot_root.mkdir(mode=0o700)
    snapshot_root.chmod(0o700)
    shutil.copytree(
        displaced / "snapshot.json.payloads",
        snapshot_root / "snapshot.json.payloads",
    )
    source.write_text(payload, encoding="utf-8")
    source.chmod(0o600)

    with pytest.raises(RollbackSnapshotError, match="changed after loading"):
        validate_snapshot_for_apply(snapshot, str(source), audit)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("host_id", "machine-id:" + "b" * 32),
        ("os_id", "rhel"),
        ("os_version", "9.6"),
        ("architecture", "unexpected-architecture"),
        ("package_manager", "dnf"),
    ],
)
def test_snapshot_apply_rejects_host_or_backend_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    trusted_snapshot_context: tuple[RollbackSnapshot, Path, Path, HostAudit],
    field: str,
    wrong_value: str,
) -> None:
    snapshot, source, _, audit = trusted_snapshot_context
    if field == "host_id":
        monkeypatch.setattr(rollback_module, "_host_identity", lambda: wrong_value)
    elif field == "architecture":
        monkeypatch.setattr(rollback_module.platform, "machine", lambda: wrong_value)
    else:
        setattr(audit, field, wrong_value)

    with pytest.raises(RollbackSnapshotError, match=field):
        validate_snapshot_for_apply(snapshot, str(source), audit)


def test_runner_terminates_process_group_when_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _InterruptingProcess()
    terminated: list[object] = []
    monkeypatch.setattr(
        runner_module.shutil,
        "which",
        lambda name, path: os.path.realpath("/usr/bin/true"),
    )
    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        runner_module,
        "_terminate_process_group",
        lambda target: terminated.append(target),
    )

    with pytest.raises(KeyboardInterrupt):
        CommandRunner().run(["nvidia-smi"])

    assert terminated == [process]


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("sh", path=TRUSTED_EXECUTABLE_PATH) is None,
    reason="requires POSIX process groups and a trusted-path shell",
)
def test_runner_kills_lingering_descendant_process_group() -> None:
    child_pid: int | None = None
    try:
        result = CommandRunner(timeout=5).run(["sh", "-c", "sleep 30 & echo $!"])
        child_pid = int(result.stdout)

        assert result.returncode == 125
        assert result.reason == "lingering-process-group-terminated"
        assert _wait_until_process_exits(child_pid)
    finally:
        if child_pid is not None and _process_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


class _UnexpectedRunner:
    def exists(self, name: str) -> bool:
        raise AssertionError(f"runner should not be consulted for non-EFI boot: {name}")


class _SecureBootRunner:
    def __init__(
        self,
        result: CommandResult | None,
        *,
        has_mokutil: bool = True,
    ) -> None:
        self.result = result
        self.has_mokutil = has_mokutil
        self.commands: list[list[str]] = []

    def exists(self, name: str) -> bool:
        return name == "mokutil" and self.has_mokutil

    def run(self, command: list[str], *, allow_fail: bool = True) -> CommandResult:
        del allow_fail
        self.commands.append(command)
        assert self.result is not None
        return self.result


class _InterruptingProcess:
    pid = 424242
    returncode: int | None = None

    def __init__(self) -> None:
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.stdin = None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        raise KeyboardInterrupt


def _override_open_file_metadata(
    monkeypatch: pytest.MonkeyPatch, *, uid: int, mode: int
) -> None:
    real_fstat = files_module.os.fstat

    def controlled_fstat(fd: int) -> os.stat_result:
        metadata = real_fstat(fd)
        if stat.S_ISDIR(metadata.st_mode):
            fields = list(metadata)
            fields[stat.ST_UID] = 0
            return os.stat_result(fields)
        if not stat.S_ISREG(metadata.st_mode):
            return metadata
        fields = list(metadata)
        fields[stat.ST_MODE] = stat.S_IFMT(metadata.st_mode) | mode
        fields[stat.ST_UID] = uid
        return os.stat_result(fields)

    monkeypatch.setattr(files_module.os, "fstat", controlled_fstat)


def _trust_audit() -> HostAudit:
    return HostAudit(
        timestamp="2026-08-02T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo(
            "6.8.0-test",
            headers_installed=True,
            compiler="/usr/bin/gcc",
            secure_boot_enabled=False,
        ),
        module=ModuleInfo(loaded=False),
        runtime=RuntimeInfo(
            docker_installed=False,
            nvidia_container_runtime_installed=False,
        ),
        packages=[],
        nvidia_smi=CommandResult(["nvidia-smi"], 127),
        nvml=CommandResult(["python3"], 1),
        fabric_manager_active=False,
        mig_mode=None,
    )


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until_process_exits(pid: int) -> bool:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.01)
    return not _process_exists(pid)
