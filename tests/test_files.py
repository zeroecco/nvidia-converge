import base64
import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import nvidia_converge.files as files_module
from nvidia_converge.files import (
    BoundedFileError,
    atomic_write_text,
    open_private_directory,
    read_bounded_utf8_with_metadata,
)
from nvidia_converge.models import FileSnapshot
from nvidia_converge.rollback import (
    RollbackSnapshotError,
    _capture_managed_files,
    _ensure_private_snapshot_directory,
    _restore_managed_files,
)


def test_atomic_write_fails_closed_when_parent_directory_open_fails(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "report.json"
    real_open = os.open

    def failing_open(target, flags, *args, **kwargs):
        if Path(target) == tmp_path and flags & getattr(os, "O_DIRECTORY", 0):
            raise OSError(errno.EIO, "directory open failed")
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr("nvidia_converge.files.os.open", failing_open)

    with pytest.raises(OSError, match="directory open failed"):
        atomic_write_text(path, "durable\n", mode=0o600)


def test_atomic_write_fails_closed_when_parent_directory_fsync_fails(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "report.json"
    real_fsync = os.fsync

    def failing_fsync(fd):
        if os.path.isdir(f"/dev/fd/{fd}"):
            raise OSError(errno.EIO, "directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr("nvidia_converge.files.os.fsync", failing_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        atomic_write_text(path, "durable\n", mode=0o600)


def test_absent_file_restore_fsyncs_parent_and_reports_failure(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "daemon.json"
    path.write_text("introduced\n", encoding="utf-8")
    parent_metadata = path.parent.stat()
    real_fsync = os.fsync

    def failing_directory_fsync(fd):
        if os.path.samestat(os.fstat(fd), parent_metadata):
            raise OSError(errno.EIO, "unlink metadata not durable")
        return real_fsync(fd)

    monkeypatch.setattr(
        "nvidia_converge.files.os.fsync",
        failing_directory_fsync,
    )
    runner = _AppliedExternalRunner()

    results = _restore_managed_files(
        [FileSnapshot(str(path), False, None, None)],
        runner,
    )

    assert results[0].returncode == 1
    assert "not durable" in results[0].stderr
    assert not path.exists()


def test_bounded_read_rejects_metadata_change_during_read(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}\n", encoding="utf-8")
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(fd):
        nonlocal calls
        metadata = real_fstat(fd)
        calls += 1
        if calls == 2:
            values = {
                name: getattr(metadata, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_uid",
                    "st_gid",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            values["st_size"] += 1
            return SimpleNamespace(**values)
        return metadata

    monkeypatch.setattr("nvidia_converge.files.os.fstat", changing_fstat)

    with pytest.raises(BoundedFileError, match="changed while"):
        read_bounded_utf8_with_metadata(path, max_bytes=1024)


def test_open_private_directory_creates_and_returns_bound_descriptor(tmp_path):
    directory = tmp_path.resolve() / "state" / "snapshots"

    fd = open_private_directory(
        directory,
        required_owner_uid=os.geteuid(),
        create=True,
    )
    try:
        assert os.path.samestat(os.fstat(fd), directory.stat())
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o700
    finally:
        os.close(fd)


def test_snapshot_directory_creation_rejects_symlinked_ancestor(tmp_path):
    actual = tmp_path.resolve() / "actual"
    actual.mkdir()
    alias = tmp_path.resolve() / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(RollbackSnapshotError, match="cannot create"):
        _ensure_private_snapshot_directory(alias / "snapshots")

    assert not (actual / "snapshots").exists()


def test_snapshot_directory_creation_rejects_writable_ancestor(tmp_path):
    unsafe = tmp_path.resolve() / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)

    with pytest.raises(RollbackSnapshotError, match="group/world-writable"):
        _ensure_private_snapshot_directory(unsafe / "snapshots")

    assert not (unsafe / "snapshots").exists()


def test_managed_file_capture_rejects_group_writable_privileged_input(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "daemon.json"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o660)
    monkeypatch.setattr("nvidia_converge.rollback._DOCKER_CONFIG_PATH", path)

    with pytest.raises(RollbackSnapshotError, match="group/world-writable"):
        _capture_managed_files(None)


def test_managed_file_capture_rejects_symlinked_parent(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    actual_parent = root / "actual"
    actual_parent.mkdir()
    (actual_parent / "daemon.json").write_text("{}\n", encoding="utf-8")
    alias = root / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)
    monkeypatch.setattr(
        "nvidia_converge.rollback._DOCKER_CONFIG_PATH",
        alias / "daemon.json",
    )

    with pytest.raises(RollbackSnapshotError, match="cannot capture managed state"):
        _capture_managed_files(None)


def test_managed_file_capture_rejects_writable_ancestor(monkeypatch, tmp_path):
    unsafe_parent = tmp_path.resolve() / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    path = unsafe_parent / "daemon.json"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr("nvidia_converge.rollback._DOCKER_CONFIG_PATH", path)

    with pytest.raises(RollbackSnapshotError, match="group/world-writable"):
        _capture_managed_files(None)


def test_absent_managed_file_capture_still_rejects_writable_ancestor(
    monkeypatch,
    tmp_path,
):
    unsafe_parent = tmp_path.resolve() / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    path = unsafe_parent / "missing" / "daemon.json"
    monkeypatch.setattr("nvidia_converge.rollback._DOCKER_CONFIG_PATH", path)

    with pytest.raises(RollbackSnapshotError, match="group/world-writable"):
        _capture_managed_files(None)


def test_managed_file_capture_detects_parent_swap(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    parent = root / "managed"
    displaced = root / "managed-before-swap"
    parent.mkdir()
    path = parent / "daemon.json"
    path.write_text("baseline\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr("nvidia_converge.rollback._DOCKER_CONFIG_PATH", path)
    real_open = files_module.os.open
    swapped = False

    def swapping_open(target, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        fd = real_open(target, flags, mode, dir_fd=dir_fd)
        if target == path.name and dir_fd is not None and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            replacement = parent / path.name
            replacement.write_text("substituted\n", encoding="utf-8")
            replacement.chmod(0o600)
        return fd

    monkeypatch.setattr(files_module.os, "open", swapping_open)

    with pytest.raises(RollbackSnapshotError, match="parent changed"):
        _capture_managed_files(None)

    assert (parent / path.name).read_text(encoding="utf-8") == "substituted\n"
    assert (displaced / path.name).read_text(encoding="utf-8") == "baseline\n"


def test_managed_file_restore_rejects_symlinked_parent(tmp_path):
    root = tmp_path.resolve()
    actual_parent = root / "actual"
    actual_parent.mkdir()
    target = actual_parent / "daemon.json"
    target.write_text("changed\n", encoding="utf-8")
    alias = root / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    results = _restore_managed_files(
        [
            FileSnapshot(
                str(alias / target.name),
                True,
                base64.b64encode(b"original\n").decode("ascii"),
                0o600,
            )
        ],
        _AppliedExternalRunner(),
    )

    assert results[0].returncode == 1
    assert target.read_text(encoding="utf-8") == "changed\n"


def test_managed_file_restore_detects_parent_swap_without_writing_substitute(
    monkeypatch,
    tmp_path,
):
    root = tmp_path.resolve()
    parent = root / "managed"
    displaced = root / "managed-before-swap"
    parent.mkdir()
    path = parent / "daemon.json"
    path.write_text("changed\n", encoding="utf-8")
    path.chmod(0o600)
    real_replace = files_module.os.replace
    swapped = False

    def swapping_replace(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        nonlocal swapped
        if not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            replacement = parent / path.name
            replacement.write_text("substituted\n", encoding="utf-8")
            replacement.chmod(0o600)
        return real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(files_module.os, "replace", swapping_replace)
    results = _restore_managed_files(
        [
            FileSnapshot(
                str(path),
                True,
                base64.b64encode(b"original\n").decode("ascii"),
                0o600,
            )
        ],
        _AppliedExternalRunner(),
    )

    assert results[0].returncode == 1
    assert "parent changed" in results[0].stderr
    assert (parent / path.name).read_text(encoding="utf-8") == "substituted\n"
    assert (displaced / path.name).read_text(encoding="utf-8") == "original\n"


def test_absent_managed_file_restore_detects_parent_swap_without_unlinking_substitute(
    monkeypatch,
    tmp_path,
):
    root = tmp_path.resolve()
    parent = root / "managed"
    displaced = root / "managed-before-swap"
    parent.mkdir()
    path = parent / "daemon.json"
    path.write_text("introduced\n", encoding="utf-8")
    path.chmod(0o600)
    real_unlink = files_module.os.unlink
    swapped = False

    def swapping_unlink(target, *, dir_fd=None):
        nonlocal swapped
        if target == path.name and dir_fd is not None and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            replacement = parent / path.name
            replacement.write_text("substituted\n", encoding="utf-8")
            replacement.chmod(0o600)
        return real_unlink(target, dir_fd=dir_fd)

    monkeypatch.setattr(files_module.os, "unlink", swapping_unlink)
    results = _restore_managed_files(
        [FileSnapshot(str(path), False, None, None)],
        _AppliedExternalRunner(),
    )

    assert results[0].returncode == 1
    assert "parent changed" in results[0].stderr
    assert (parent / path.name).read_text(encoding="utf-8") == "substituted\n"
    assert not (displaced / path.name).exists()


class _AppliedExternalRunner:
    apply = True

    def record_external_start(self, command, mutate):
        del command, mutate

    def record_external_result(self, result, mutate):
        del result, mutate
