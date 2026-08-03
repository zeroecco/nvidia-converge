import os
from pathlib import Path

import pytest

from nvidia_converge.locking import OperationLockError, operation_lock


def test_disabled_operation_lock_does_not_touch_files(monkeypatch, tmp_path):
    path = tmp_path / "lock"
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)
    with operation_lock(False):
        pass
    assert not path.exists()


def test_operation_lock_rejects_concurrent_holder(monkeypatch, tmp_path):
    path = tmp_path / "lock"
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)
    with operation_lock(True), pytest.raises(OperationLockError, match="another applied"), operation_lock(True):
        pass
    assert Path(path).stat().st_mode & 0o777 == 0o600


def test_operation_lock_creates_private_parent_directory(monkeypatch, tmp_path):
    path = tmp_path / "nvidia-converge" / "operation.lock"
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)

    with operation_lock(True):
        assert path.parent.stat().st_uid == os.geteuid()
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_operation_lock_rejects_symlink_without_changing_victim(monkeypatch, tmp_path):
    private_parent = tmp_path / "nvidia-converge"
    private_parent.mkdir(mode=0o700)
    victim = tmp_path / "victim"
    victim.write_text("do not change\n", encoding="utf-8")
    victim.chmod(0o640)
    original_mode = victim.stat().st_mode
    path = private_parent / "operation.lock"
    path.symlink_to(victim)
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)

    with pytest.raises(OperationLockError, match="cannot open operation lock"), operation_lock(True):
        pass

    assert victim.read_text(encoding="utf-8") == "do not change\n"
    assert victim.stat().st_mode == original_mode


def test_operation_lock_rejects_symlinked_parent_ancestor(monkeypatch, tmp_path):
    actual = tmp_path.resolve() / "actual"
    actual.mkdir(mode=0o700)
    alias = tmp_path.resolve() / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    path = alias / "nvidia-converge" / "operation.lock"
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)

    with pytest.raises(
        OperationLockError,
        match="cannot open operation lock",
    ), operation_lock(True):
        pass

    assert not (actual / "nvidia-converge").exists()


def test_operation_lock_rejects_writable_ancestor(monkeypatch, tmp_path):
    unsafe = tmp_path.resolve() / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)
    path = unsafe / "nvidia-converge" / "operation.lock"
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)

    with pytest.raises(
        OperationLockError,
        match="group/world-writable",
    ), operation_lock(True):
        pass

    assert not path.parent.exists()


def test_operation_lock_detects_parent_swap_without_locking_substitute(
    monkeypatch,
    tmp_path,
):
    parent = tmp_path.resolve() / "nvidia-converge"
    displaced = tmp_path.resolve() / "nvidia-converge-before-swap"
    parent.mkdir(mode=0o700)
    path = parent / "operation.lock"
    monkeypatch.setattr("nvidia_converge.locking.LOCK_PATH", path)
    real_open = os.open
    swapped = False

    def swapping_open(target, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        fd = real_open(target, flags, mode, dir_fd=dir_fd)
        if target == path.name and dir_fd is not None and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir(mode=0o700)
        return fd

    monkeypatch.setattr("nvidia_converge.locking.os.open", swapping_open)

    with pytest.raises(OperationLockError, match="directory changed"), operation_lock(
        True
    ):
        pass

    assert not path.exists()
    assert (displaced / path.name).exists()
