from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .files import BoundedFileError, open_private_directory

LOCK_PATH = Path("/run/nvidia-converge/operation.lock")


class OperationLockError(OSError):
    pass


def _open_operation_lock() -> tuple[int, int]:
    parent = Path(os.path.abspath(LOCK_PATH.parent))
    parent_fd = -1
    lock_fd = -1
    opened = False
    try:
        parent_fd = open_private_directory(
            parent,
            required_owner_uid=os.geteuid(),
            create=True,
        )

        lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        lock_fd = os.open(LOCK_PATH.name, lock_flags, 0o600, dir_fd=parent_fd)
        metadata = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise OperationLockError(
                f"operation lock must be a singly linked regular file owned by uid {os.geteuid()}: {LOCK_PATH}"
            )
        os.fchmod(lock_fd, 0o600)
        rebound_fd = open_private_directory(
            parent,
            required_owner_uid=os.geteuid(),
        )
        try:
            if not os.path.samestat(os.fstat(parent_fd), os.fstat(rebound_fd)):
                raise OperationLockError(
                    f"operation lock directory changed while opening: {parent}"
                )
        finally:
            os.close(rebound_fd)
        opened = True
        return parent_fd, lock_fd
    except OperationLockError:
        raise
    except (OSError, BoundedFileError, ValueError) as exc:
        raise OperationLockError(f"cannot open operation lock {str(LOCK_PATH)!r}: {exc}") from exc
    finally:
        if not opened:
            if lock_fd >= 0:
                os.close(lock_fd)
            if parent_fd >= 0:
                os.close(parent_fd)


@contextmanager
def operation_lock(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    parent_fd, lock_fd = _open_operation_lock()
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise OperationLockError(f"another applied nvidia-converge operation holds {LOCK_PATH}") from exc
            raise OperationLockError(f"cannot lock {LOCK_PATH}: {exc}") from exc
        yield
    finally:
        os.close(lock_fd)
        os.close(parent_fd)
