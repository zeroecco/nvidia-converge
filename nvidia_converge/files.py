from __future__ import annotations

import errno
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


class BoundedFileError(ValueError):
    pass


@dataclass(frozen=True)
class _BoundParent:
    path: Path
    name: str
    fd: int
    device: int
    inode: int
    required_owner_uid: int


@dataclass(frozen=True)
class TrustedFileRead:
    text: str
    file_metadata: os.stat_result
    parent_metadata: os.stat_result


_MAX_TRUSTED_SYMLINK_BYTES = 4 * 1024


def read_bounded_utf8(
    path: Path,
    *,
    max_bytes: int,
    require_root_controlled: bool = False,
    require_trusted_ancestors: bool = False,
) -> str:
    """Read a no-follow regular file with an explicit size and encoding bound."""
    text, _ = read_bounded_utf8_with_metadata(
        path,
        max_bytes=max_bytes,
        require_root_controlled=require_root_controlled,
        require_trusted_ancestors=require_trusted_ancestors,
    )
    return text


def read_bounded_utf8_with_metadata(
    path: Path,
    *,
    max_bytes: int,
    require_root_controlled: bool = False,
    required_owner_uid: int | None = None,
    require_trusted_ancestors: bool = False,
) -> tuple[str, os.stat_result]:
    """Read text and stable metadata from the same no-follow file descriptor."""

    text, metadata, _ = _read_bounded_utf8_with_metadata(
        path,
        max_bytes=max_bytes,
        require_root_controlled=require_root_controlled,
        required_owner_uid=required_owner_uid,
        require_trusted_ancestors=require_trusted_ancestors,
        require_private_parent=False,
    )
    return text, metadata


def read_trusted_utf8_with_metadata(
    path: Path,
    *,
    max_bytes: int,
    require_root_controlled: bool = False,
    required_owner_uid: int | None = None,
    require_private_parent: bool = False,
) -> TrustedFileRead:
    """Read through a bound parent and return file and parent metadata."""

    text, file_metadata, parent_metadata = _read_bounded_utf8_with_metadata(
        path,
        max_bytes=max_bytes,
        require_root_controlled=require_root_controlled,
        required_owner_uid=required_owner_uid,
        require_trusted_ancestors=True,
        require_private_parent=require_private_parent,
    )
    if parent_metadata is None:
        raise BoundedFileError("trusted read did not retain parent metadata")
    return TrustedFileRead(text, file_metadata, parent_metadata)


def read_root_controlled_utf8(
    path: Path,
    *,
    max_bytes: int,
    required_owner_uid: int = 0,
) -> str:
    """Read a trusted regular file or one root-controlled final symlink.

    ``/etc/os-release`` is normally a relative symlink to
    ``/usr/lib/os-release``.  This narrow helper permits that topology without
    allowing ordinary path traversal to follow symlinks: the lexical parent is
    opened component-by-component, the link identity and text are retained,
    and the normalized target is independently opened through trusted
    ancestry with a no-follow final descriptor.  The link must be owned by the
    required uid and singly linked; its permission bits are deliberately not
    consulted because POSIX symlink modes are not access-control metadata.
    The target must be a singly linked regular file owned by the required uid
    and not group/world-writable.
    """

    bound = _open_bound_parent(path, required_owner_uid=required_owner_uid)
    try:
        lexical_before = _stat_bound_target(bound)
        if lexical_before is None:
            _verify_bound_parent(bound)
            raise FileNotFoundError(errno.ENOENT, "input does not exist", str(path))
        if stat.S_ISREG(lexical_before.st_mode):
            # The ordinary trusted reader reopens and binds the same lexical
            # path. A replacement with a symlink is rejected by O_NOFOLLOW.
            return read_trusted_utf8_with_metadata(
                path,
                max_bytes=max_bytes,
                required_owner_uid=required_owner_uid,
            ).text
        if not stat.S_ISLNK(lexical_before.st_mode):
            raise OSError(
                errno.EINVAL,
                "input must be a regular file or trusted final symlink",
                str(path),
            )
        if lexical_before.st_uid != required_owner_uid or lexical_before.st_nlink != 1:
            raise BoundedFileError(
                "trusted symlink must be singly linked and owned by uid "
                f"{required_owner_uid}: {path}"
            )

        link_text = os.readlink(bound.name, dir_fd=bound.fd)
        try:
            link_size = len(link_text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise BoundedFileError("trusted symlink target is not valid UTF-8") from exc
        if not link_text or link_size > _MAX_TRUSTED_SYMLINK_BYTES:
            raise BoundedFileError("trusted symlink target is empty or too long")
        if link_text.startswith("//"):
            raise BoundedFileError(
                "trusted symlink target must not use an implementation-defined root"
            )
        target = Path(
            os.path.normpath(
                link_text if os.path.isabs(link_text) else str(path.parent / link_text)
            )
        )
        if not target.is_absolute() or target == path:
            raise BoundedFileError("trusted symlink target is not a distinct absolute path")

        trusted = read_trusted_utf8_with_metadata(
            target,
            max_bytes=max_bytes,
            required_owner_uid=required_owner_uid,
        )

        lexical_after = _stat_bound_target(bound)
        if lexical_after is None or _metadata_changed(
            lexical_before,
            lexical_after,
            fields=(
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            ),
        ):
            raise BoundedFileError(
                "trusted symlink changed while its target was being read"
            )
        if os.readlink(bound.name, dir_fd=bound.fd) != link_text:
            raise BoundedFileError(
                "trusted symlink target changed while it was being read"
            )
        _verify_bound_parent(bound)
        return trusted.text
    finally:
        os.close(bound.fd)


def _read_bounded_utf8_with_metadata(
    path: Path,
    *,
    max_bytes: int,
    require_root_controlled: bool,
    required_owner_uid: int | None,
    require_trusted_ancestors: bool,
    require_private_parent: bool,
) -> tuple[str, os.stat_result, os.stat_result | None]:
    if require_private_parent and not require_trusted_ancestors:
        raise ValueError("a private parent check requires trusted ancestry")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    bound: _BoundParent | None = None
    if require_trusted_ancestors:
        ancestor_owner = (
            required_owner_uid
            if required_owner_uid is not None
            else 0
            if require_root_controlled
            else os.geteuid()
        )
        bound = _open_bound_parent(path, required_owner_uid=ancestor_owner)
    parent_before: os.stat_result | None = None
    try:
        if bound is not None:
            parent_before = os.fstat(bound.fd)
            if require_private_parent:
                _validate_private_directory(
                    parent_before,
                    bound.path,
                    required_owner_uid=bound.required_owner_uid,
                )
        try:
            fd = (
                os.open(bound.name, flags, dir_fd=bound.fd)
                if bound is not None
                else os.open(path, flags)
            )
        except FileNotFoundError:
            if bound is not None:
                _verify_bound_parent(bound)
            raise
    except BaseException:
        if bound is not None:
            os.close(bound.fd)
        raise
    try:
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise OSError(errno.EINVAL, "input must be a regular file", str(path))
            if require_root_controlled:
                if before.st_uid != 0:
                    raise BoundedFileError(
                        "input must be owned by root (uid 0); "
                        f"observed uid {before.st_uid}"
                    )
                mode = stat.S_IMODE(before.st_mode)
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    raise BoundedFileError(
                        "input must not be group/world-writable; "
                        f"observed mode {mode:04o}"
                    )
            if required_owner_uid is not None and before.st_uid != required_owner_uid:
                raise BoundedFileError(
                    f"input must be owned by uid {required_owner_uid}; "
                    f"observed uid {before.st_uid}"
                )
            if required_owner_uid is not None:
                mode = stat.S_IMODE(before.st_mode)
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    raise BoundedFileError(
                        "input must not be group/world-writable; "
                        f"observed mode {mode:04o}"
                    )
            if before.st_nlink != 1:
                raise BoundedFileError("input must be singly linked")
            if before.st_size > max_bytes:
                raise OSError(
                    errno.EFBIG,
                    f"input exceeds {max_bytes} bytes",
                    str(path),
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise OSError(
                        errno.EFBIG,
                        f"input exceeds {max_bytes} bytes",
                        str(path),
                    )
            after = os.fstat(fd)
            stable_fields = (
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
            if any(
                getattr(before, name) != getattr(after, name)
                for name in stable_fields
            ):
                raise BoundedFileError("input changed while it was being read")
        finally:
            os.close(fd)
        parent_metadata: os.stat_result | None = None
        if bound is not None:
            parent_after = os.fstat(bound.fd)
            rebound_metadata = _verify_bound_parent(bound)
            if require_private_parent:
                _validate_private_directory(
                    parent_after,
                    bound.path,
                    required_owner_uid=bound.required_owner_uid,
                )
                _validate_private_directory(
                    rebound_metadata,
                    bound.path,
                    required_owner_uid=bound.required_owner_uid,
                )
                if parent_before is None:
                    raise BoundedFileError(
                        "trusted read lost its initial parent metadata"
                    )
                if _metadata_changed(
                    parent_before,
                    parent_after,
                    fields=(
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_nlink",
                        "st_uid",
                        "st_gid",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    ),
                ) or _metadata_changed(
                    parent_after,
                    rebound_metadata,
                    fields=(
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_nlink",
                        "st_uid",
                        "st_gid",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    ),
                ):
                    raise BoundedFileError(
                        "trusted path parent changed while the file was being read"
                    )
            parent_metadata = rebound_metadata
    finally:
        if bound is not None:
            os.close(bound.fd)
    try:
        return b"".join(chunks).decode("utf-8"), after, parent_metadata
    except UnicodeDecodeError as exc:
        raise BoundedFileError(f"input is not valid UTF-8 at byte {exc.start}") from exc


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    """Atomically and durably replace a text file before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    effective_mode = mode
    if effective_mode is None:
        try:
            effective_mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            effective_mode = 0o666 & ~_current_umask()

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, effective_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text_trusted(
    path: Path,
    text: str,
    *,
    mode: int,
    required_owner_uid: int,
    deterministic_temporary_name: str | None = None,
) -> None:
    """Durably replace a file through a no-follow, ancestry-bound parent FD."""

    if not 0 <= mode <= 0o777:
        raise ValueError("trusted file mode must be between 0000 and 0777")
    bound = _open_bound_parent(path, required_owner_uid=required_owner_uid)
    temporary_name: str | None = None
    fd = -1
    try:
        metadata = _stat_bound_target(bound)
        if metadata is not None and (
            not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        ):
            raise OSError(
                errno.EINVAL,
                "managed state target is not a singly linked regular file",
                str(path),
            )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if deterministic_temporary_name is not None and (
            deterministic_temporary_name in {"", ".", ".."}
            or os.sep in deterministic_temporary_name
            or (
                os.altsep is not None
                and os.altsep in deterministic_temporary_name
            )
            or "\x00" in deterministic_temporary_name
            or len(os.fsencode(deterministic_temporary_name)) > 255
        ):
            raise ValueError("trusted temporary name must be one safe path component")
        candidates = (
            [deterministic_temporary_name]
            if deterministic_temporary_name is not None
            else [f".{bound.name}.{secrets.token_hex(12)}" for _ in range(128)]
        )
        for candidate in candidates:
            assert candidate is not None
            try:
                fd = os.open(candidate, flags, 0o600, dir_fd=bound.fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_name is None or fd < 0:
            raise OSError(errno.EEXIST, "cannot allocate trusted temporary file")
        os.fchmod(fd, mode)
        payload = text.encode("utf-8")
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise OSError(errno.EIO, "short write to trusted temporary file")
            written += count
        os.fsync(fd)
        os.close(fd)
        fd = -1
        _verify_bound_parent(bound)
        os.replace(
            temporary_name,
            bound.name,
            src_dir_fd=bound.fd,
            dst_dir_fd=bound.fd,
        )
        temporary_name = None
        os.fsync(bound.fd)
        final_metadata = _stat_bound_target(bound)
        if final_metadata is None or (
            not stat.S_ISREG(final_metadata.st_mode)
            or final_metadata.st_nlink != 1
            or final_metadata.st_uid != required_owner_uid
            or stat.S_IMODE(final_metadata.st_mode) != mode
        ):
            raise BoundedFileError(
                "managed state target metadata changed during trusted replacement"
            )
        _verify_bound_parent(bound)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=bound.fd)
            except FileNotFoundError:
                pass
        os.close(bound.fd)


def trusted_path_metadata(
    path: Path,
    *,
    required_owner_uid: int,
) -> os.stat_result | None:
    """Inspect a final component without following it or any path ancestor."""

    bound = _open_bound_parent(path, required_owner_uid=required_owner_uid)
    try:
        metadata = _stat_bound_target(bound)
        _verify_bound_parent(bound)
        return metadata
    finally:
        os.close(bound.fd)


def unlink_trusted_path(
    path: Path,
    *,
    required_owner_uid: int,
) -> bool:
    """Durably unlink a safe final component through a bound parent FD."""

    bound = _open_bound_parent(path, required_owner_uid=required_owner_uid)
    try:
        metadata = _stat_bound_target(bound)
        if metadata is None:
            _verify_bound_parent(bound)
            return False
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError(
                errno.EINVAL,
                "managed state target is not a singly linked regular file",
                str(path),
            )
        _verify_bound_parent(bound)
        os.unlink(bound.name, dir_fd=bound.fd)
        os.fsync(bound.fd)
        _verify_bound_parent(bound)
        return True
    finally:
        os.close(bound.fd)


def open_private_directory(
    path: Path,
    *,
    required_owner_uid: int,
    create: bool = False,
    mode: int = 0o700,
) -> int:
    """Open a private directory through trusted ancestry; caller closes the FD."""

    if mode & 0o077 or mode & 0o700 != 0o700 or not 0 <= mode <= 0o777:
        raise ValueError("private directory mode must be owner-only and owner-accessible")
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise BoundedFileError("trusted directory path must be normalized and absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise BoundedFileError("platform lacks required no-follow directory support")
    fd = os.open(os.sep, flags)
    try:
        root_metadata = os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise
    opened: list[tuple[Path, os.stat_result]] = [(Path(os.sep), root_metadata)]
    current_path = Path(os.sep)
    try:
        for component in path.parts[1:]:
            _validate_trusted_directories(
                opened,
                required_owner_uid,
                allow_final_sticky=True,
            )
            created = False
            try:
                next_fd = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                if os.geteuid() != required_owner_uid:
                    raise BoundedFileError(
                        "cannot create a trusted directory for a different owner uid"
                    )
                try:
                    os.mkdir(component, mode, dir_fd=fd)
                    os.fsync(fd)
                    created = True
                except FileExistsError:
                    pass
                next_fd = os.open(component, flags, dir_fd=fd)
            previous_fd = fd
            fd = next_fd
            os.close(previous_fd)
            current_path /= component
            if created:
                os.fchmod(fd, mode)
                os.fsync(fd)
            opened.append((current_path, os.fstat(fd)))
        _validate_trusted_directories(opened, required_owner_uid)
        metadata = os.fstat(fd)
        _validate_private_directory(
            metadata,
            path,
            required_owner_uid=required_owner_uid,
        )
        rebound = _open_bound_parent(
            path / ".nvidia-converge-directory-binding",
            required_owner_uid=required_owner_uid,
        )
        try:
            rebound_metadata = os.fstat(rebound.fd)
            if _metadata_changed(
                metadata,
                rebound_metadata,
                fields=(
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_uid",
                    "st_gid",
                    "st_mtime_ns",
                    "st_ctime_ns",
                ),
            ):
                raise BoundedFileError(
                    f"trusted directory changed while opening: {path}"
                )
            _validate_private_directory(
                rebound_metadata,
                path,
                required_owner_uid=required_owner_uid,
            )
        finally:
            os.close(rebound.fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def ensure_private_directory(
    path: Path,
    *,
    required_owner_uid: int,
    mode: int = 0o700,
) -> None:
    """Create a private directory safely and durably when it is absent."""

    fd = open_private_directory(
        path,
        required_owner_uid=required_owner_uid,
        create=True,
        mode=mode,
    )
    os.close(fd)


def _open_bound_parent(path: Path, *, required_owner_uid: int) -> _BoundParent:
    if (
        not path.is_absolute()
        or not path.name
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise BoundedFileError("trusted file path must be normalized and absolute")
    directory = path.parent
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise BoundedFileError("platform lacks required no-follow directory support")
    fd = os.open(os.sep, flags)
    try:
        root_metadata = os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise
    opened: list[tuple[Path, os.stat_result]] = [(Path(os.sep), root_metadata)]
    current_path = Path(os.sep)
    try:
        try:
            for component in directory.parts[1:]:
                next_fd = os.open(component, flags, dir_fd=fd)
                previous_fd = fd
                fd = next_fd
                os.close(previous_fd)
                current_path /= component
                opened.append((current_path, os.fstat(fd)))
        except FileNotFoundError:
            _validate_trusted_directories(opened, required_owner_uid)
            raise
        _validate_trusted_directories(opened, required_owner_uid)
        metadata = os.fstat(fd)
        return _BoundParent(
            directory,
            path.name,
            fd,
            metadata.st_dev,
            metadata.st_ino,
            required_owner_uid,
        )
    except BaseException:
        os.close(fd)
        raise


def _validate_trusted_directories(
    opened: list[tuple[Path, os.stat_result]],
    required_owner_uid: int,
    *,
    allow_final_sticky: bool = False,
) -> None:
    allowed_owners = {0, required_owner_uid}
    for index, (path, metadata) in enumerate(opened):
        if not stat.S_ISDIR(metadata.st_mode):
            raise BoundedFileError(f"trusted path ancestor is not a directory: {path}")
        if metadata.st_uid not in allowed_owners:
            raise BoundedFileError(
                f"trusted path ancestor has unsafe owner uid {metadata.st_uid}: {path}"
            )
        mode = stat.S_IMODE(metadata.st_mode)
        if not mode & (stat.S_IWGRP | stat.S_IWOTH):
            continue
        sticky_boundary = (
            metadata.st_uid == 0
            and bool(mode & stat.S_ISVTX)
            and index + 1 < len(opened)
        )
        if (
            allow_final_sticky
            and metadata.st_uid == 0
            and bool(mode & stat.S_ISVTX)
            and index + 1 == len(opened)
        ):
            continue
        if sticky_boundary:
            _, child = opened[index + 1]
            child_mode = stat.S_IMODE(child.st_mode)
            if (
                child.st_uid in allowed_owners
                and not child_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                continue
        raise BoundedFileError(
            f"trusted path ancestor is group/world-writable: {path}"
        )


def _stat_bound_target(bound: _BoundParent) -> os.stat_result | None:
    try:
        return os.stat(bound.name, dir_fd=bound.fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _verify_bound_parent(bound: _BoundParent) -> os.stat_result:
    rebound = _open_bound_parent(
        bound.path / bound.name,
        required_owner_uid=bound.required_owner_uid,
    )
    try:
        if (rebound.device, rebound.inode) != (bound.device, bound.inode):
            raise BoundedFileError(
                f"trusted path parent changed while operating on {bound.path / bound.name}"
            )
        return os.fstat(rebound.fd)
    finally:
        os.close(rebound.fd)


def _validate_private_directory(
    metadata: os.stat_result,
    path: Path,
    *,
    required_owner_uid: int,
) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != required_owner_uid
        or mode & 0o077
        or mode & 0o700 != 0o700
    ):
        raise BoundedFileError(
            "trusted directory must be private, owner-accessible, "
            f"owned by uid {required_owner_uid}: {path}"
        )


def _metadata_changed(
    before: os.stat_result,
    after: os.stat_result,
    *,
    fields: tuple[str, ...],
) -> bool:
    return any(getattr(before, name) != getattr(after, name) for name in fields)


def _current_umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def fsync_directory(path: Path) -> None:
    """Durably commit directory metadata without following a symlink."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError(errno.ENOTDIR, "fsync target is not a directory", str(path))
        os.fsync(fd)
    finally:
        os.close(fd)
