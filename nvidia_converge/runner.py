from __future__ import annotations

import os
import pwd
import shutil
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from .models import CommandResult

TRUSTED_EXECUTABLE_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
TRUSTED_WORKING_DIRECTORY = "/"
MAX_CAPTURE_BYTES = 1024 * 1024
TERMINATION_GRACE_SECONDS = 2.0
_PASSTHROUGH_ENVIRONMENT = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_TRUSTED_EXECUTABLE_OWNER_UID = 0
_TRUSTED_EXECUTABLE_ANCESTOR_UIDS = frozenset({_TRUSTED_EXECUTABLE_OWNER_UID})


class _UntrustedExecutableError(OSError):
    pass


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )


def _file_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _normalized_absolute_path(path: Path) -> Path:
    if not path.is_absolute() or path.anchor != os.sep:
        raise _UntrustedExecutableError(f"trusted path is not absolute: {path}")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if normalized != path:
        raise _UntrustedExecutableError(f"trusted path is not normalized: {path}")
    return normalized


def _validate_trusted_directory_metadata(
    metadata: os.stat_result,
    path: Path,
) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in _TRUSTED_EXECUTABLE_ANCESTOR_UIDS
        or mode & 0o022
    ):
        raise _UntrustedExecutableError(
            "trusted executable ancestor must be root-owned and "
            f"non-group/world-writable: {path}"
        )


def _validate_trusted_symlink_metadata(
    metadata: os.stat_result,
    path: Path,
) -> None:
    if (
        not stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid not in _TRUSTED_EXECUTABLE_ANCESTOR_UIDS
    ):
        raise _UntrustedExecutableError(
            f"trusted executable symlink has an untrusted owner: {path}"
        )


def _validate_lexical_path(path: Path, *, directory: bool) -> None:
    """Authenticate every lexical component up to a root-controlled symlink.

    The canonical target is validated separately. Stopping at the first
    symlink preserves standard root-owned system aliases while ensuring the
    alias itself cannot be replaced through an untrusted parent directory.
    """

    path = _normalized_absolute_path(path)
    parts = path.parts
    parent_descriptor = os.open(path.anchor, _directory_open_flags())
    try:
        _validate_trusted_directory_metadata(
            os.fstat(parent_descriptor), Path(path.anchor)
        )
        current = Path(path.anchor)
        for index, component in enumerate(parts[1:]):
            current /= component
            metadata = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode):
                _validate_trusted_symlink_metadata(metadata, current)
                return
            is_leaf = index == len(parts) - 2
            if is_leaf and not directory:
                return
            descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
            try:
                opened_metadata = os.fstat(descriptor)
                if (
                    opened_metadata.st_dev != metadata.st_dev
                    or opened_metadata.st_ino != metadata.st_ino
                ):
                    raise _UntrustedExecutableError(
                        f"trusted executable ancestor changed while opening: {current}"
                    )
                _validate_trusted_directory_metadata(opened_metadata, current)
            except BaseException:
                os.close(descriptor)
                raise
            os.close(parent_descriptor)
            parent_descriptor = descriptor
    finally:
        os.close(parent_descriptor)


def _open_canonical_path(
    path: Path,
    *,
    directory: bool,
) -> os.stat_result:
    path = _normalized_absolute_path(path)
    parts = path.parts
    parent_descriptor = os.open(path.anchor, _directory_open_flags())
    try:
        _validate_trusted_directory_metadata(
            os.fstat(parent_descriptor), Path(path.anchor)
        )
        current = Path(path.anchor)
        for component in parts[1:-1]:
            descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
            try:
                current /= component
                _validate_trusted_directory_metadata(os.fstat(descriptor), current)
            except BaseException:
                os.close(descriptor)
                raise
            os.close(parent_descriptor)
            parent_descriptor = descriptor
        if len(parts) == 1:
            return os.fstat(parent_descriptor)
        flags = _directory_open_flags() if directory else _file_open_flags()
        descriptor = os.open(parts[-1], flags, dir_fd=parent_descriptor)
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _trusted_directory(path: Path) -> Path:
    path = _normalized_absolute_path(path)
    _validate_lexical_path(path, directory=True)
    try:
        resolved = path.resolve(strict=True)
    except RuntimeError as exc:
        raise _UntrustedExecutableError(
            f"trusted executable directory cannot be resolved safely: {path}"
        ) from exc
    metadata = _open_canonical_path(resolved, directory=True)
    _validate_trusted_directory_metadata(metadata, resolved)
    return resolved


def _trusted_executable(path: Path) -> tuple[Path, os.stat_result]:
    path = _normalized_absolute_path(path)
    _validate_lexical_path(path, directory=False)
    try:
        resolved = path.resolve(strict=True)
    except RuntimeError as exc:
        raise _UntrustedExecutableError(
            f"trusted executable cannot be resolved safely: {path}"
        ) from exc
    metadata = _open_canonical_path(resolved, directory=False)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != _TRUSTED_EXECUTABLE_OWNER_UID
        or mode & 0o022
        or not mode & stat.S_IXUSR
    ):
        raise _UntrustedExecutableError(
            "trusted executable must be a root-owned, non-group/world-writable "
            f"regular file with owner-execute permission: {path}"
        )
    if os.access in os.supports_effective_ids:
        can_execute = os.access(resolved, os.X_OK, effective_ids=True)
    else:
        can_execute = os.access(resolved, os.X_OK)
    if not can_execute:
        raise _UntrustedExecutableError(
            f"trusted executable is not executable by this process: {path}"
        )
    return resolved, metadata


def _trusted_executable_directories(path_value: str) -> frozenset[Path]:
    components = path_value.split(os.pathsep)
    if not components or any(not component for component in components):
        raise _UntrustedExecutableError(
            "trusted executable search path contains an empty component"
        )
    return frozenset(_trusted_directory(Path(component)) for component in components)


class CommandRunner:
    def __init__(
        self,
        apply: bool = False,
        timeout: int = 120,
        mutation_timeout: int = 1800,
        executable_path: str = TRUSTED_EXECUTABLE_PATH,
        start_callback: Callable[[list[str], bool], None] | None = None,
        result_callback: Callable[[CommandResult, bool], None] | None = None,
    ):
        self.apply = apply
        self.timeout = timeout
        self.mutation_timeout = mutation_timeout
        self.executable_path = executable_path
        self.start_callback = start_callback
        self.result_callback = result_callback
        self.results: list[CommandResult] = []
        self._private_state_scope: tuple[str, ...] | None = None

    def exists(self, name: str) -> bool:
        return self.resolve_executable(name) is not None

    def resolve_executable(self, name: str) -> str | None:
        try:
            trusted_directories = _trusted_executable_directories(
                self.executable_path
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if not name or "\x00" in name:
            return None
        if os.path.isabs(name):
            candidate = Path(name)
        else:
            if os.sep in name or (os.altsep and os.altsep in name):
                return None
            found = shutil.which(name, path=self.executable_path)
            if found is None:
                return None
            candidate = Path(found)
        try:
            parent = _trusted_directory(candidate.parent)
            if parent not in trusted_directories:
                return None
            executable, _ = _trusted_executable(candidate)
        except (OSError, RuntimeError, ValueError):
            return None
        if executable.parent not in trusted_directories:
            return None
        return str(executable)

    def run(
        self,
        command: list[str],
        *,
        mutate: bool = False,
        allow_fail: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        self.record_external_start(command, mutate)
        if mutate and not self.apply:
            result = CommandResult(command=command, returncode=None, skipped=True, reason="dry-run")
            self._record_result(result, mutate)
            return result
        executable = self.resolve_executable(command[0])
        if executable is None:
            result = CommandResult(
                command=command,
                returncode=127,
                stderr=f"trusted executable not found: {command[0]}",
            )
            self._record_result(result, mutate)
            if not allow_fail:
                raise RuntimeError(f"command failed: {' '.join(command)}: {result.stderr}")
            return result

        timeout = self.mutation_timeout if mutate else self.timeout
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in _PASSTHROUGH_ENVIRONMENT
        }
        try:
            home = pwd.getpwuid(os.geteuid()).pw_dir
        except KeyError:
            home = TRUSTED_WORKING_DIRECTORY
        environment.update(
            {"HOME": home, "LANG": "C", "LC_ALL": "C", "PATH": self.executable_path}
        )
        if mutate:
            environment["DEBIAN_FRONTEND"] = "noninteractive"
            # Report obsolete-library users without restarting services that
            # are outside this tool's explicitly modeled service boundary.
            environment["NEEDRESTART_MODE"] = "l"

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                executable=executable,
                cwd=TRUSTED_WORKING_DIRECTORY,
                start_new_session=True,
            )
        except OSError as exc:
            result = CommandResult(command=command, returncode=127, stderr=str(exc))
            self._record_result(result, mutate)
            if not allow_fail:
                raise RuntimeError(f"command failed: {' '.join(command)}: {result.stderr}") from exc
            return result

        stdout_capture = _BoundedCapture()
        stderr_capture = _BoundedCapture()
        assert proc.stdout is not None
        assert proc.stderr is not None
        readers = [
            threading.Thread(
                target=_drain_stream,
                args=(proc.stdout, stdout_capture),
                name="nvidia-converge-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=_drain_stream,
                args=(proc.stderr, stderr_capture),
                name="nvidia-converge-stderr",
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        lingering_descendants = False
        try:
            if input_text is not None:
                assert proc.stdin is not None
                try:
                    proc.stdin.write(input_text.encode("utf-8"))
                    proc.stdin.flush()
                except BrokenPipeError:
                    pass
                finally:
                    proc.stdin.close()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(proc)
            else:
                lingering_descendants = _process_group_exists(proc.pid)
                if lingering_descendants:
                    _terminate_process_group(proc)
        except BaseException:
            _terminate_process_group(proc)
            _join_readers(readers)
            raise
        _join_readers(readers)

        stdout = stdout_capture.text()
        stderr = stderr_capture.text()
        if timed_out:
            detail = (
                f"command process group timed out after {timeout} seconds and was terminated"
            )
            if command[0] in {"apt-get", "dpkg", "dnf", "yum", "rpm", "zypper"}:
                detail += "; package database recovery may be required"
            result = CommandResult(
                command=command,
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\n{detail}".strip(),
                reason="timeout-process-group-terminated",
            )
        elif lingering_descendants:
            result = CommandResult(
                command=command,
                returncode=125,
                stdout=stdout,
                stderr=f"{stderr}\ncommand left descendant processes; process group was terminated".strip(),
                reason="lingering-process-group-terminated",
            )
        else:
            result = CommandResult(
                command=command,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        self._record_result(result, mutate)
        if not allow_fail and result.returncode not in (0, None):
            raise RuntimeError(f"command failed: {' '.join(command)}: {result.stderr}")
        return result

    def run_private_state(
        self,
        command: list[str],
        *,
        allow_fail: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        """Run a long private-state subcommand inside an outer journaled phase.

        The caller must journal one narrowly recognized private-state phase.
        Subcommands use the package mutation timeout but cannot emit nested
        command events or acquire interrupted-host-mutation authority.
        """

        if self._private_state_scope != ("stage-package-payloads",):
            raise RuntimeError(
                "private-state subcommands require the exact package staging scope"
            )
        original_timeout = self.timeout
        original_start_callback = self.start_callback
        original_result_callback = self.result_callback
        self.timeout = self.mutation_timeout
        self.start_callback = None
        self.result_callback = None
        try:
            return self.run(
                command,
                mutate=False,
                allow_fail=allow_fail,
                input_text=input_text,
            )
        finally:
            self.timeout = original_timeout
            self.start_callback = original_start_callback
            self.result_callback = original_result_callback

    @contextmanager
    def private_state_scope(self, command: list[str]) -> Iterator[None]:
        """Journal one exact safe-private phase without nested command events."""

        identity = tuple(command)
        if identity != ("stage-package-payloads",):
            raise ValueError("unsupported private-state command scope")
        if self._private_state_scope is not None:
            raise RuntimeError("a private-state command scope is already active")
        self.record_external_start(command, True)
        self._private_state_scope = identity
        try:
            yield
        except BaseException as exc:
            self.record_external_result(
                CommandResult(command, 1, stderr=str(exc)),
                True,
            )
            raise
        else:
            self.record_external_result(CommandResult(command, 0), True)
        finally:
            self._private_state_scope = None

    def _record_result(self, result: CommandResult, mutate: bool) -> None:
        self.results.append(result)
        if self.result_callback is not None:
            self.result_callback(result, mutate)

    def record_external_start(self, command: list[str], mutate: bool) -> None:
        if self.start_callback is not None:
            self.start_callback(list(command), mutate)

    def record_external_result(
        self, result: CommandResult, mutate: bool
    ) -> None:
        self._record_result(result, mutate)


@dataclass
class _BoundedCapture:
    limit: int = MAX_CAPTURE_BYTES
    data: bytearray = field(default_factory=bytearray)
    total: int = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        remaining = self.limit - len(self.data)
        if remaining > 0:
            self.data.extend(chunk[:remaining])

    def text(self) -> str:
        text = bytes(self.data).decode("utf-8", errors="replace").strip()
        if self.total > self.limit:
            marker = (
                f"[output truncated: retained first {self.limit} of {self.total} bytes]"
            )
            text = f"{text}\n{marker}".strip()
        return text


def _drain_stream(stream: BinaryIO, capture: _BoundedCapture) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            capture.add(chunk)
    finally:
        stream.close()


def _join_readers(readers: list[threading.Thread]) -> None:
    for reader in readers:
        reader.join(timeout=TERMINATION_GRACE_SECONDS)
    if any(reader.is_alive() for reader in readers):
        raise RuntimeError("command output readers did not terminate after process-group cleanup")


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while _process_group_exists(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
