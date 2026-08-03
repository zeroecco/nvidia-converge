from __future__ import annotations

import errno
import json
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .files import BoundedFileError, read_bounded_utf8
from .models import CommandResult, HostAudit
from .runner import CommandRunner

_WORKLOAD_PROBE_COMMAND = ["probe-active-gpu-workloads"]
_COMPUTE_QUERY = [
    "nvidia-smi",
    "--query-compute-apps=pid",
    "--format=csv,noheader,nounits",
]
_PROC_ROOT = Path("/proc")
_MAX_PROCESSES = 131_072
_MAX_FDS_PER_PROCESS = 65_536
_MAX_TOTAL_FDS = 1_048_576
_MAX_MAP_BYTES_PER_PROCESS = 8 * 1024 * 1024
_MAX_TOTAL_MAP_BYTES = 128 * 1024 * 1024
_MAX_MAP_FILE_STATS_PER_PROCESS = 65_536
_MAX_TOTAL_MAP_FILE_STATS = 262_144
_MAX_PROC_SCAN_SECONDS = 10.0
_MAX_DEVICE_ROOT_ENTRIES = 65_536
_MAX_NVIDIA_DEVICE_NODES = 4_096
_MAX_PROC_MAP_LINE_BYTES = 64 * 1024
_MAX_RUNNING_CONTAINERS = 256
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_OUTPUT_TRUNCATED = "[output truncated:"
_MAX_CGROUP_BYTES = 64 * 1024
_MAX_PROC_STAT_BYTES = 4096
_NVIDIA_DEVICE_ROOT = Path("/dev")
_SERVICE_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
    "ExecCondition",
    "ExecStartPre",
    "ExecStart",
    "ExecStartPost",
    "ExecStop",
    "ExecStopPost",
    "Environment",
    "EnvironmentFiles",
    "RootDirectory",
    "RootImage",
    "BindPaths",
    "BindReadOnlyPaths",
    "TemporaryFileSystem",
    "MountImages",
    "ExtensionImages",
)
_DOCKER_SOCKET_UNIT = "docker.socket"
_DOCKER_SOCKET_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "FragmentPath",
    "DropInPaths",
    "Triggers",
    "ExecStartPre",
    "ExecStartPost",
    "ExecStopPre",
    "ExecStopPost",
    "Environment",
    "EnvironmentFiles",
    "RootDirectory",
    "RootImage",
    "BindPaths",
    "BindReadOnlyPaths",
    "TemporaryFileSystem",
    "MountImages",
    "ExtensionImages",
)
_TRUSTED_OWNER_UID = 0
_TRUSTED_ANCESTOR_UIDS = frozenset({_TRUSTED_OWNER_UID})
_TRUSTED_SYSTEMD_MASK_ROOTS = (
    Path("/etc/systemd/system"),
    Path("/run/systemd/system"),
)


@dataclass(frozen=True)
class TrustedGpuServiceSpec:
    unit: str
    executable: Path


@dataclass(frozen=True)
class TrustedGpuServiceIdentity:
    """Immutable identity of one validated active trusted service process."""

    unit: str
    main_pid: int
    process_start_time_ticks: int
    executable_device: int
    executable_inode: int
    cgroup_path: str


@dataclass(frozen=True)
class TrustedUnitFileIdentity:
    """Immutable metadata for one authenticated effective unit input."""

    path: str
    executable: bool
    device: int
    inode: int
    mode: int
    owner_uid: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class TrustedDockerSocketIdentity:
    """Binding for the loaded socket definition across a safe mask operation."""

    active: bool
    effective_properties: tuple[tuple[str, str], ...]
    files: tuple[TrustedUnitFileIdentity, ...]


@dataclass(frozen=True)
class _NvidiaDeviceIdentity:
    """Kernel identities for one NVIDIA character-device node."""

    rdev: int
    backing_device: int
    inode: int


TRUSTED_GPU_SERVICES = (
    TrustedGpuServiceSpec(
        "nvidia-fabricmanager.service",
        Path("/usr/bin/nv-fabricmanager"),
    ),
    TrustedGpuServiceSpec(
        "nvidia-persistenced.service",
        Path("/usr/bin/nvidia-persistenced"),
    ),
)
_DOCKER_SERVICE_SPEC = TrustedGpuServiceSpec(
    "docker.service",
    Path("/usr/bin/dockerd"),
)
_TRUSTED_GPU_SERVICE_BY_UNIT = {
    spec.unit: spec for spec in (*TRUSTED_GPU_SERVICES, _DOCKER_SERVICE_SPEC)
}


@dataclass(frozen=True)
class _TrustedServiceObservation:
    spec: TrustedGpuServiceSpec
    active: bool
    main_pid: int
    fragment_path: Path | None
    load_state: str
    active_state: str
    sub_state: str
    properties: dict[str, str]
    identity: TrustedGpuServiceIdentity | None = None


@dataclass
class TrustedGpuServiceGuard:
    """Track exact trusted service state while disruptive work is in progress."""

    runner: CommandRunner
    proc_root: Path
    specs: tuple[TrustedGpuServiceSpec, ...]
    results: list[CommandResult] = field(default_factory=list)
    originally_active: dict[str, TrustedGpuServiceSpec] = field(default_factory=dict)
    pending_restore: dict[str, TrustedGpuServiceSpec] = field(default_factory=dict)
    error: str | None = None
    restore_errors: list[str] = field(default_factory=list)
    requiesce_errors: list[str] = field(default_factory=list)
    mutation_started: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.error is None and not self.restore_errors and not self.requiesce_errors
        )

    @property
    def quiesced_service_names(self) -> list[str]:
        return sorted(self.pending_restore)

    def mark_mutation_started(self) -> None:
        self.mutation_started = True

    def relinquish(self, units: set[str]) -> None:
        """Let another transaction own the final state of selected services."""

        for unit in units:
            self.pending_restore.pop(unit, None)
            self.originally_active.pop(unit, None)

    def restore(self, *, units: set[str] | None = None) -> bool:
        """Restore selected originally-active services, once, with revalidation."""

        targets = [
            spec
            for unit, spec in reversed(list(self.pending_restore.items()))
            if units is None or unit in units
        ]
        for spec in targets:
            observation, error = _observe_trusted_service(
                self.runner,
                spec,
                self.results,
                proc_root=self.proc_root,
            )
            if error is not None or observation is None:
                self.restore_errors.append(
                    error or f"could not observe {spec.unit} while restoring it"
                )
                continue
            if not observation.active:
                validation_results, error = _validate_trusted_gpu_service_spec_start(
                    self.runner,
                    spec,
                    proc_root=self.proc_root,
                )
                self.results.extend(validation_results)
                if error is not None:
                    self.restore_errors.append(error)
                    continue
                result = self.runner.run(
                    ["systemctl", "start", spec.unit],
                    mutate=True,
                    allow_fail=True,
                )
                self.results.append(result)
                if result.returncode not in (0, None):
                    self.restore_errors.append(
                        f"could not restart {spec.unit}: {_command_error('systemctl start failed', result)}"
                    )
                    continue
                observation, error = _observe_trusted_service(
                    self.runner,
                    spec,
                    self.results,
                    proc_root=self.proc_root,
                )
                if error is not None or observation is None:
                    self.restore_errors.append(
                        error or f"could not verify restarted {spec.unit}"
                    )
                    continue
            if not observation.active:
                self.restore_errors.append(
                    f"{spec.unit} did not return to its original active state"
                )
                continue
            self.pending_restore.pop(spec.unit, None)
        return not self.restore_errors and not (
            set(self.pending_restore)
            if units is None
            else set(self.pending_restore) & units
        )

    def requiesce(self) -> bool:
        """Stop every trusted GPU service before the next mutation phase.

        Package maintainer scripts can start a service that was inactive at the
        initial gate. Such a service is stopped but deliberately not adopted
        into ``pending_restore``: only the exact originally-active set may be
        restored by this guard.
        """

        self.requiesce_errors.clear()
        for spec in self.specs:
            observation, error = _observe_trusted_service(
                self.runner,
                spec,
                self.results,
                proc_root=self.proc_root,
            )
            if error is not None or observation is None:
                self.requiesce_errors.append(
                    error or f"could not observe {spec.unit} while re-quiescing it"
                )
                continue
            if observation.active:
                result = self.runner.run(
                    ["systemctl", "stop", spec.unit],
                    mutate=True,
                    allow_fail=True,
                )
                self.results.append(result)
                if result.returncode not in (0, None):
                    self.requiesce_errors.append(
                        f"could not re-quiesce {spec.unit}: "
                        + _command_error("systemctl stop failed", result)
                    )
                    continue
                observation, error = _observe_trusted_service(
                    self.runner,
                    spec,
                    self.results,
                    proc_root=self.proc_root,
                )
                if error is not None or observation is None:
                    self.requiesce_errors.append(
                        error or f"could not verify re-quiesced {spec.unit}"
                    )
                    continue
            if observation.active:
                self.requiesce_errors.append(
                    f"{spec.unit} remained active after re-quiesce"
                )
                continue
            if spec.unit in self.originally_active:
                self.pending_restore[spec.unit] = spec
        return not self.requiesce_errors


def quiesce_trusted_gpu_services(
    runner: CommandRunner,
    *,
    proc_root: Path = _PROC_ROOT,
    specs: tuple[TrustedGpuServiceSpec, ...] = TRUSTED_GPU_SERVICES,
    restore_on_failure: bool = True,
) -> TrustedGpuServiceGuard:
    """Stop only strongly bound NVIDIA service processes before a workload scan.

    ``restore_on_failure`` preserves the standalone API's historical cleanup
    behavior. Transaction owners that can persistently quarantine every
    launcher disable it so an unverified path can never restart a service.
    """

    guard = TrustedGpuServiceGuard(runner, proc_root, specs)
    if not runner.exists("systemctl"):
        guard.error = "trusted systemctl executable is required to qualify GPU services"
        return guard
    try:
        for spec in specs:
            observation, error = _observe_trusted_service(
                runner,
                spec,
                guard.results,
                proc_root=proc_root,
            )
            if error is not None or observation is None:
                guard.error = error or f"could not observe {spec.unit}"
                if restore_on_failure:
                    guard.restore()
                return guard
            if not observation.active:
                continue
            guard.originally_active[spec.unit] = spec
            guard.pending_restore[spec.unit] = spec
            result = runner.run(
                ["systemctl", "stop", spec.unit],
                mutate=True,
                allow_fail=True,
            )
            guard.results.append(result)
            if result.returncode not in (0, None):
                guard.error = f"could not quiesce {spec.unit}: " + _command_error(
                    "systemctl stop failed", result
                )
                if restore_on_failure:
                    guard.restore()
                return guard
            stopped, error = _observe_trusted_service(
                runner,
                spec,
                guard.results,
                proc_root=proc_root,
            )
            if error is not None or stopped is None or stopped.active:
                guard.error = (
                    error or f"{spec.unit} remained active after systemctl stop"
                )
                if restore_on_failure:
                    guard.restore()
                return guard
        return guard
    except BaseException:
        if restore_on_failure:
            guard.restore()
        raise


def validate_trusted_docker_socket_unit(
    runner: CommandRunner,
    *,
    allow_masked: bool = False,
) -> tuple[list[CommandResult], str | None]:
    results, _, error = validate_trusted_docker_socket_unit_identity(
        runner,
        allow_masked=allow_masked,
    )
    return results, error


def validate_trusted_docker_socket_unit_identity(
    runner: CommandRunner,
    *,
    allow_masked: bool = False,
) -> tuple[
    list[CommandResult],
    TrustedDockerSocketIdentity | None,
    str | None,
]:
    """Authenticate the exact effective Docker socket before mutation.

    A masked or absent unit is accepted only when ``allow_masked`` is true.
    Those states are inert and are useful as the precondition for an unmask or
    an idempotent quarantine mask.  Once unmasked, callers must invoke this
    function again without that allowance before starting, stopping, or
    changing enablement.
    """

    results: list[CommandResult] = []
    identity: TrustedDockerSocketIdentity | None = None
    error: str | None = None
    if not runner.exists("systemctl"):
        error = "trusted systemctl executable is required to validate docker.socket"
    else:
        command = [
            "systemctl",
            "show",
            "--all",
            "--full",
            "--no-pager",
            "--property=" + ",".join(_DOCKER_SOCKET_PROPERTIES),
            _DOCKER_SOCKET_UNIT,
        ]
        result = runner.run(command, mutate=False, allow_fail=True)
        results.append(result)
        properties = _parse_docker_socket_properties(result.stdout)
        if properties is None:
            error = "docker.socket returned malformed or incomplete unit state"
        elif properties["Id"] != _DOCKER_SOCKET_UNIT:
            error = (
                "docker.socket resolved to unexpected unit "
                f"{properties['Id']!r}"
            )
        else:
            load_state = properties["LoadState"]
            active_state = properties["ActiveState"]
            sub_state = properties["SubState"]
            if load_state in {"masked", "not-found"}:
                error = _validate_inert_docker_socket_state(
                    result,
                    properties,
                    allow_masked=allow_masked,
                )
            elif result.returncode != 0:
                error = _command_error("could not observe docker.socket", result)
            elif load_state != "loaded":
                error = f"docker.socket has unsupported load state {load_state!r}"
            elif (active_state, sub_state) not in {
                ("active", "listening"),
                ("inactive", "dead"),
                ("failed", "failed"),
            }:
                if active_state not in {"active", "inactive", "failed"}:
                    error = (
                        "docker.socket is in transitional state "
                        f"{active_state!r}"
                    )
                else:
                    error = "docker.socket has an inconsistent stable unit state"
            else:
                identity, error = _validate_trusted_docker_socket_binding(properties)

    validation_command = ["validate-trusted-docker-socket-unit", _DOCKER_SOCKET_UNIT]
    _record_external_start(runner, validation_command, mutate=False)
    validation_result = CommandResult(
        validation_command,
        0 if error is None else 1,
        stderr=error or "",
    )
    _record_external_result(runner, validation_result, mutate=False)
    results.append(validation_result)
    return results, identity if error is None else None, error


def _validate_inert_docker_socket_state(
    result: CommandResult,
    properties: dict[str, str],
    *,
    allow_masked: bool,
) -> str | None:
    load_state = properties["LoadState"]
    if not allow_masked:
        return (
            "cannot trust docker.socket for this mutation: expected an exact "
            "loaded unit"
        )
    if result.returncode not in {0, 1, 4}:
        return _command_error("could not observe docker.socket", result)
    if (
        properties["ActiveState"] != "inactive"
        or properties["SubState"] != "dead"
    ):
        return f"docker.socket has inconsistent {load_state} unit state"
    fragment_path = properties["FragmentPath"]
    if load_state == "not-found" and fragment_path:
        return "docker.socket has inconsistent not-found fragment state"
    if load_state == "masked":
        if not fragment_path:
            return "docker.socket has inconsistent masked fragment state"
        try:
            _trusted_unit_mask_path(Path(fragment_path), _DOCKER_SOCKET_UNIT)
        except OSError as exc:
            return f"docker.socket has untrusted masked fragment state: {exc}"
    for property_name in _DOCKER_SOCKET_PROPERTIES:
        if property_name in {
            "Id",
            "LoadState",
            "ActiveState",
            "SubState",
            "FragmentPath",
        }:
            continue
        if properties[property_name]:
            return (
                f"docker.socket has unexpected effective {property_name} "
                f"in {load_state} state"
            )
    return None


def _validate_trusted_docker_socket_binding(
    properties: dict[str, str],
) -> tuple[TrustedDockerSocketIdentity | None, str | None]:
    files: dict[tuple[Path, bool], TrustedUnitFileIdentity] = {}

    def bind_file(path: Path, *, executable: bool) -> None:
        metadata = _trusted_regular_file(path, executable=executable)
        files[(path, executable)] = _unit_file_identity(
            path,
            executable=executable,
            metadata=metadata,
        )

    try:
        fragment_text = properties["FragmentPath"]
        if not fragment_text:
            raise OSError("effective FragmentPath is missing")
        bind_file(Path(fragment_text), executable=False)

        drop_in_paths = _parse_systemd_path_list(properties["DropInPaths"])
        if drop_in_paths is None:
            raise OSError("systemd returned malformed DropInPaths")
        for drop_in_path in drop_in_paths:
            bind_file(drop_in_path, executable=False)

        if properties["Triggers"] != "docker.service":
            raise OSError(
                "effective Triggers does not resolve exactly to docker.service"
            )
        if properties["Environment"]:
            raise OSError("effective inline Environment entries are unsupported")
        environment_files = _parse_systemd_environment_files(
            properties["EnvironmentFiles"]
        )
        if environment_files is None:
            raise OSError("systemd returned malformed EnvironmentFiles")
        for environment_file in environment_files:
            bind_file(environment_file, executable=False)

        for property_name in (
            "RootDirectory",
            "RootImage",
            "BindPaths",
            "BindReadOnlyPaths",
            "TemporaryFileSystem",
            "MountImages",
            "ExtensionImages",
        ):
            if properties[property_name]:
                raise OSError(
                    f"effective {property_name} executable-root or mount "
                    "remapping is unsupported"
                )

        for property_name in (
            "ExecStartPre",
            "ExecStartPost",
            "ExecStopPre",
            "ExecStopPost",
        ):
            hook_paths = _parse_systemd_exec_paths(properties[property_name])
            if hook_paths is None:
                raise OSError(f"systemd returned malformed {property_name}")
            for hook_path in hook_paths:
                bind_file(hook_path, executable=True)
    except OSError as exc:
        return None, f"cannot trust docker.socket: {exc}"
    return (
        TrustedDockerSocketIdentity(
            active=properties["ActiveState"] == "active",
            effective_properties=tuple(sorted(properties.items())),
            files=tuple(files.values()),
        ),
        None,
    )


def _unit_file_identity(
    path: Path,
    *,
    executable: bool,
    metadata: os.stat_result,
) -> TrustedUnitFileIdentity:
    return TrustedUnitFileIdentity(
        path=str(path),
        executable=executable,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        owner_uid=metadata.st_uid,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def revalidate_trusted_docker_socket_identity(
    runner: CommandRunner,
    identity: TrustedDockerSocketIdentity,
) -> tuple[list[CommandResult], str | None]:
    """Rebind pre-mask unit inputs without asking systemd to reload the unit."""

    error: str | None = None
    try:
        for expected in identity.files:
            path = Path(expected.path)
            metadata = _trusted_regular_file(
                path,
                executable=expected.executable,
            )
            if _unit_file_identity(
                path,
                executable=expected.executable,
                metadata=metadata,
            ) != expected:
                raise OSError(
                    f"trusted effective unit input changed across mask: {path}"
                )
    except OSError as exc:
        error = f"cannot trust docker.socket: {exc}"
    command = ["revalidate-trusted-docker-socket-identity", _DOCKER_SOCKET_UNIT]
    _record_external_start(runner, command, mutate=False)
    result = CommandResult(command, 0 if error is None else 1, stderr=error or "")
    _record_external_result(runner, result, mutate=False)
    return [result], error


def _parse_docker_socket_properties(text: str) -> dict[str, str] | None:
    properties: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if (
            not separator
            or key not in _DOCKER_SOCKET_PROPERTIES
            or key in properties
            or "\x00" in value
        ):
            return None
        properties[key] = value
    if set(properties) != set(_DOCKER_SOCKET_PROPERTIES):
        return None
    return properties


def validate_trusted_gpu_service_unit(
    runner: CommandRunner,
    unit: str,
    *,
    allow_masked: bool = False,
    proc_root: Path = _PROC_ROOT,
) -> tuple[list[CommandResult], str | None]:
    """Authenticate one exact stable trusted service before any mutation."""

    results: list[CommandResult] = []
    spec = _TRUSTED_GPU_SERVICE_BY_UNIT.get(unit)
    error: str | None = None
    if spec is None:
        error = (
            "unsupported trusted service; expected an exact canonical "
            "nvidia-fabricmanager.service, nvidia-persistenced.service, or "
            "docker.service unit"
        )
    elif not runner.exists("systemctl"):
        error = "trusted systemctl executable is required to validate service unit"
    else:
        observation, error = _observe_trusted_service(
            runner,
            spec,
            results,
            proc_root=proc_root,
        )
        if error is None and observation is not None:
            if observation.load_state == "loaded":
                if observation.fragment_path is None:
                    error = f"cannot trust {unit}: effective FragmentPath is missing"
                elif not observation.active:
                    _, error = _validate_trusted_service_binding(
                        runner,
                        spec,
                        observation.fragment_path,
                        None,
                        results,
                        properties=observation.properties,
                        proc_root=proc_root,
                    )
            elif observation.load_state in {"masked", "not-found"}:
                error = _validate_inert_trusted_service_state(
                    observation,
                    allow_masked=allow_masked,
                )
            else:
                error = (
                    f"cannot trust {unit}: unsupported load state "
                    f"{observation.load_state!r}"
                )
        elif error is None:
            error = f"cannot trust {unit}: service state is unavailable"

    command = ["validate-trusted-gpu-service-unit", unit]
    _record_external_start(runner, command, mutate=False)
    result = CommandResult(command, 0 if error is None else 1, stderr=error or "")
    _record_external_result(runner, result, mutate=False)
    results.append(result)
    return results, error


def _validate_inert_trusted_service_state(
    observation: _TrustedServiceObservation,
    *,
    allow_masked: bool,
) -> str | None:
    unit = observation.spec.unit
    load_state = observation.load_state
    if not allow_masked:
        return f"cannot trust {unit} for this mutation: expected an exact loaded unit"
    if observation.active or observation.main_pid != 0:
        return f"cannot trust {unit}: inconsistent {load_state} process state"
    fragment_text = observation.properties["FragmentPath"]
    if load_state == "not-found" and fragment_text:
        return f"cannot trust {unit}: inconsistent not-found fragment state"
    if load_state == "masked":
        if not fragment_text:
            return f"cannot trust {unit}: inconsistent masked fragment state"
        try:
            _trusted_unit_mask_path(Path(fragment_text), unit)
        except OSError as exc:
            return f"cannot trust {unit}: untrusted masked fragment state: {exc}"
    for property_name in _SERVICE_PROPERTIES:
        if property_name in {
            "Id",
            "LoadState",
            "ActiveState",
            "SubState",
            "MainPID",
            "FragmentPath",
        }:
            continue
        if observation.properties[property_name]:
            return (
                f"cannot trust {unit}: unexpected effective {property_name} "
                f"in {load_state} state"
            )
    return None


def validate_trusted_gpu_service_start(
    runner: CommandRunner,
    unit: str,
    *,
    proc_root: Path = _PROC_ROOT,
) -> tuple[list[CommandResult], str | None]:
    """Validate an exact trusted inactive service immediately before start.

    This function never starts the service.  Callers must treat any returned
    error as a hard stop and retain ownership of the quiesced service state.
    """

    spec = _TRUSTED_GPU_SERVICE_BY_UNIT.get(unit)
    if spec is None:
        error = (
            "unsupported trusted service; expected an exact canonical "
            "nvidia-fabricmanager.service, nvidia-persistenced.service, or "
            "docker.service unit"
        )
        return _trusted_start_validation_result(runner, unit, [], error)
    return _validate_trusted_gpu_service_spec_start(
        runner,
        spec,
        proc_root=proc_root,
    )


def validate_active_trusted_gpu_service(
    runner: CommandRunner,
    unit: str,
    *,
    proc_root: Path = _PROC_ROOT,
) -> tuple[list[CommandResult], str | None]:
    """Validate the exact active process binding after a trusted service start."""

    results, _, error = validate_active_trusted_gpu_service_identity(
        runner,
        unit,
        proc_root=proc_root,
    )
    return results, error


def validate_active_trusted_gpu_service_identity(
    runner: CommandRunner,
    unit: str,
    *,
    expected_identity: TrustedGpuServiceIdentity | None = None,
    proc_root: Path = _PROC_ROOT,
) -> tuple[
    list[CommandResult],
    TrustedGpuServiceIdentity | None,
    str | None,
]:
    """Validate and bind one active trusted service process identity.

    Pass a previously returned identity after an intervening health probe to
    prove that the same process remained bound throughout that probe.
    """

    results: list[CommandResult] = []
    spec = _TRUSTED_GPU_SERVICE_BY_UNIT.get(unit)
    identity: TrustedGpuServiceIdentity | None = None
    error: str | None = None
    if spec is None:
        error = (
            "unsupported trusted service; expected an exact canonical "
            "nvidia-fabricmanager.service, nvidia-persistenced.service, or "
            "docker.service unit"
        )
    elif expected_identity is not None and expected_identity.unit != unit:
        error = (
            "expected trusted GPU service identity belongs to unexpected unit "
            f"{expected_identity.unit!r}"
        )
    elif not runner.exists("systemctl"):
        error = "trusted systemctl executable is required to validate active service"
    else:
        observation, error = _observe_trusted_service(
            runner,
            spec,
            results,
            proc_root=proc_root,
        )
        if error is None and observation is not None:
            if (
                observation.load_state != "loaded"
                or observation.active_state != "active"
                or observation.sub_state != "running"
                or not observation.active
                or observation.main_pid <= 0
                or observation.fragment_path is None
                or observation.identity is None
            ):
                error = (
                    f"cannot trust active {unit}: expected an exact loaded, "
                    "active, running unit with MainPID greater than 0"
                )
            else:
                identity = observation.identity
                if expected_identity is not None and identity != expected_identity:
                    error = (
                        f"cannot trust active {unit}: service process identity "
                        "changed across the validation boundary"
                    )
        elif error is None:
            error = f"cannot trust active {unit}: service state is unavailable"
    results, error = _trusted_active_validation_result(
        runner,
        unit,
        results,
        error,
        identity=identity,
    )
    return results, identity if error is None else None, error


def revalidate_trusted_gpu_service_process_identity(
    runner: CommandRunner,
    expected_identity: TrustedGpuServiceIdentity,
    *,
    proc_root: Path = _PROC_ROOT,
) -> tuple[list[CommandResult], str | None]:
    """Rebind an active process after masking hides its loaded unit fragment."""

    error: str | None = None
    spec = _TRUSTED_GPU_SERVICE_BY_UNIT.get(expected_identity.unit)
    try:
        if spec is None:
            raise OSError("trusted service identity names an unsupported unit")
        executable_metadata = _trusted_regular_file(
            spec.executable,
            executable=True,
        )
        if (
            executable_metadata.st_dev != expected_identity.executable_device
            or executable_metadata.st_ino != expected_identity.executable_inode
        ):
            raise OSError("trusted executable identity changed across mask")
        main_pid = expected_identity.main_pid
        start_time_before = _process_start_time_ticks(
            proc_root / str(main_pid) / "stat",
            main_pid,
        )
        descriptor = os.open(
            proc_root / str(main_pid) / "exe",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            process_metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            process_metadata.st_dev != expected_identity.executable_device
            or process_metadata.st_ino != expected_identity.executable_inode
        ):
            raise OSError("trusted service process executable changed across mask")
        cgroups = read_bounded_utf8(
            proc_root / str(main_pid) / "cgroup",
            max_bytes=_MAX_CGROUP_BYTES,
        )
        cgroup_path = _service_cgroup_path(cgroups, expected_identity.unit)
        if cgroup_path != expected_identity.cgroup_path:
            raise OSError("trusted service process cgroup changed across mask")
        start_time_after = _process_start_time_ticks(
            proc_root / str(main_pid) / "stat",
            main_pid,
        )
        if (
            start_time_before != expected_identity.process_start_time_ticks
            or start_time_after != expected_identity.process_start_time_ticks
        ):
            raise OSError("trusted service process identity changed across mask")
    except (OSError, BoundedFileError) as exc:
        error = f"cannot trust active {expected_identity.unit}: {exc}"
    command = [
        "revalidate-trusted-gpu-service-process-identity",
        expected_identity.unit,
    ]
    _record_external_start(runner, command, mutate=False)
    result = CommandResult(command, 0 if error is None else 1, stderr=error or "")
    _record_external_result(runner, result, mutate=False)
    return [result], error


def _validate_trusted_gpu_service_spec_start(
    runner: CommandRunner,
    spec: TrustedGpuServiceSpec,
    *,
    proc_root: Path,
) -> tuple[list[CommandResult], str | None]:
    results: list[CommandResult] = []
    error: str | None = None
    if not runner.exists("systemctl"):
        error = "trusted systemctl executable is required to validate service start"
    else:
        observation, error = _observe_trusted_service(
            runner,
            spec,
            results,
            proc_root=proc_root,
        )
        if error is None and observation is not None:
            if (
                observation.load_state != "loaded"
                or observation.active_state != "inactive"
                or observation.sub_state != "dead"
                or observation.active
                or observation.main_pid != 0
                or observation.fragment_path is None
            ):
                error = (
                    f"cannot start {spec.unit}: expected an exact loaded, "
                    "inactive, dead unit with MainPID 0"
                )
            else:
                _, error = _validate_trusted_service_binding(
                    runner,
                    spec,
                    observation.fragment_path,
                    None,
                    results,
                    properties=observation.properties,
                    proc_root=proc_root,
                )
        elif error is None:
            error = f"cannot start {spec.unit}: service state is unavailable"
    return _trusted_start_validation_result(
        runner,
        spec.unit,
        results,
        error,
    )


def _trusted_start_validation_result(
    runner: CommandRunner,
    unit: str,
    results: list[CommandResult],
    error: str | None,
) -> tuple[list[CommandResult], str | None]:
    command = ["validate-trusted-gpu-service-start", unit]
    _record_external_start(runner, command, mutate=False)
    result = CommandResult(
        command,
        0 if error is None else 1,
        stderr=error or "",
    )
    _record_external_result(runner, result, mutate=False)
    results.append(result)
    return results, error


def _trusted_active_validation_result(
    runner: CommandRunner,
    unit: str,
    results: list[CommandResult],
    error: str | None,
    *,
    identity: TrustedGpuServiceIdentity | None,
) -> tuple[list[CommandResult], str | None]:
    command = ["validate-active-trusted-gpu-service", unit]
    _record_external_start(runner, command, mutate=False)
    result = CommandResult(
        command,
        0 if error is None else 1,
        stdout=(
            json.dumps(
                {
                    "cgroup_path": identity.cgroup_path,
                    "executable_device": identity.executable_device,
                    "executable_inode": identity.executable_inode,
                    "main_pid": identity.main_pid,
                    "process_start_time_ticks": (identity.process_start_time_ticks),
                    "unit": identity.unit,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            if identity is not None
            else ""
        ),
        stderr=error or "",
    )
    _record_external_result(runner, result, mutate=False)
    results.append(result)
    return results, error


def _observe_trusted_service(
    runner: CommandRunner,
    spec: TrustedGpuServiceSpec,
    results: list[CommandResult],
    *,
    proc_root: Path,
) -> tuple[_TrustedServiceObservation | None, str | None]:
    command = [
        "systemctl",
        "show",
        "--all",
        "--full",
        "--no-pager",
        "--property=" + ",".join(_SERVICE_PROPERTIES),
        spec.unit,
    ]
    result = runner.run(command, mutate=False, allow_fail=True)
    results.append(result)
    properties = _parse_service_properties(result.stdout)
    if properties is None:
        return None, f"{spec.unit} returned malformed or incomplete unit state"
    if properties["Id"] != spec.unit:
        return None, f"{spec.unit} resolved to unexpected unit {properties['Id']!r}"
    main_pid_text = properties["MainPID"]
    if not main_pid_text.isdigit():
        return None, f"{spec.unit} returned an invalid MainPID"
    main_pid = int(main_pid_text)
    active_state = properties["ActiveState"]
    sub_state = properties["SubState"]
    load_state = properties["LoadState"]
    fragment_text = properties["FragmentPath"]
    fragment_path = Path(fragment_text) if fragment_text else None

    if load_state == "not-found":
        if (
            result.returncode not in {0, 1, 4}
            or active_state != "inactive"
            or sub_state != "dead"
            or main_pid != 0
            or fragment_path is not None
        ):
            return None, f"{spec.unit} has inconsistent not-found unit state"
        return (
            _TrustedServiceObservation(
                spec,
                False,
                0,
                None,
                load_state,
                active_state,
                sub_state,
                properties,
            ),
            None,
        )
    if result.returncode != 0:
        return None, _command_error(f"could not observe {spec.unit}", result)
    if active_state == "active":
        if (
            load_state != "loaded"
            or sub_state != "running"
            or main_pid <= 0
            or fragment_path is None
        ):
            return None, f"{spec.unit} has an unsupported active unit state"
        identity, error = _validate_trusted_service_binding(
            runner,
            spec,
            fragment_path,
            main_pid,
            results,
            properties=properties,
            proc_root=proc_root,
        )
        if error is not None:
            return None, error
        if identity is None:
            return None, f"cannot trust {spec.unit}: process identity is unavailable"
        return (
            _TrustedServiceObservation(
                spec,
                True,
                main_pid,
                fragment_path,
                load_state,
                active_state,
                sub_state,
                properties,
                identity,
            ),
            None,
        )
    if active_state in {"inactive", "failed"}:
        if main_pid != 0 or sub_state not in {"dead", "failed"}:
            return None, f"{spec.unit} has an inconsistent inactive unit state"
        return (
            _TrustedServiceObservation(
                spec,
                False,
                0,
                fragment_path,
                load_state,
                active_state,
                sub_state,
                properties,
            ),
            None,
        )
    return None, f"{spec.unit} is in transitional state {active_state!r}"


def _parse_service_properties(text: str) -> dict[str, str] | None:
    properties: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if (
            not separator
            or key not in _SERVICE_PROPERTIES
            or key in properties
            or "\x00" in value
        ):
            return None
        properties[key] = value
    if set(properties) != set(_SERVICE_PROPERTIES):
        return None
    return properties


def _validate_trusted_service_binding(
    runner: CommandRunner,
    spec: TrustedGpuServiceSpec,
    fragment_path: Path,
    main_pid: int | None,
    results: list[CommandResult],
    *,
    properties: dict[str, str],
    proc_root: Path,
) -> tuple[TrustedGpuServiceIdentity | None, str | None]:
    command = [
        "validate-trusted-gpu-service",
        spec.unit,
        str(main_pid or 0),
    ]
    _record_external_start(runner, command, mutate=False)
    identity: TrustedGpuServiceIdentity | None = None
    error: str | None = None
    try:
        _trusted_regular_file(fragment_path, executable=False)
        drop_in_paths = _parse_systemd_path_list(properties["DropInPaths"])
        if drop_in_paths is None:
            raise OSError("systemd returned malformed DropInPaths")
        for drop_in_path in drop_in_paths:
            _trusted_regular_file(drop_in_path, executable=False)

        if properties["Environment"]:
            raise OSError(
                "effective inline Environment entries are unsupported"
            )
        environment_files = _parse_systemd_environment_files(
            properties["EnvironmentFiles"]
        )
        if environment_files is None:
            raise OSError("systemd returned malformed EnvironmentFiles")
        for environment_file in environment_files:
            _trusted_regular_file(environment_file, executable=False)

        for property_name in (
            "RootDirectory",
            "RootImage",
            "BindPaths",
            "BindReadOnlyPaths",
            "TemporaryFileSystem",
            "MountImages",
            "ExtensionImages",
        ):
            if properties[property_name]:
                raise OSError(
                    f"effective {property_name} executable-root or mount "
                    "remapping is unsupported"
                )

        exec_start_paths = _parse_systemd_exec_paths(properties["ExecStart"])
        if exec_start_paths != (spec.executable,):
            raise OSError(
                "effective ExecStart does not resolve to the exact expected "
                f"executable {spec.executable}"
            )
        executable_metadata = _trusted_regular_file(
            spec.executable,
            executable=True,
        )
        for property_name in (
            "ExecCondition",
            "ExecStartPre",
            "ExecStartPost",
            "ExecStop",
            "ExecStopPost",
        ):
            hook_paths = _parse_systemd_exec_paths(properties[property_name])
            if hook_paths is None:
                raise OSError(f"systemd returned malformed {property_name}")
            for hook_path in hook_paths:
                _trusted_regular_file(hook_path, executable=True)
        if main_pid is not None:
            start_time_before = _process_start_time_ticks(
                proc_root / str(main_pid) / "stat",
                main_pid,
            )
            process_executable = proc_root / str(main_pid) / "exe"
            descriptor = os.open(
                process_executable,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                process_metadata = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (
                process_metadata.st_dev != executable_metadata.st_dev
                or process_metadata.st_ino != executable_metadata.st_ino
            ):
                raise OSError(
                    f"MainPID {main_pid} executable does not match {spec.executable}"
                )
            cgroup_path = proc_root / str(main_pid) / "cgroup"
            cgroups = read_bounded_utf8(
                cgroup_path,
                max_bytes=_MAX_CGROUP_BYTES,
            )
            service_cgroup = _service_cgroup_path(cgroups, spec.unit)
            if service_cgroup is None:
                raise OSError(
                    f"MainPID {main_pid} is not bound to {spec.unit}'s system.slice cgroup"
                )
            start_time_after = _process_start_time_ticks(
                proc_root / str(main_pid) / "stat",
                main_pid,
            )
            if start_time_after != start_time_before:
                raise OSError(f"MainPID {main_pid} changed identity during validation")
            identity = TrustedGpuServiceIdentity(
                unit=spec.unit,
                main_pid=main_pid,
                process_start_time_ticks=start_time_after,
                executable_device=process_metadata.st_dev,
                executable_inode=process_metadata.st_ino,
                cgroup_path=service_cgroup,
            )
    except (OSError, BoundedFileError) as exc:
        error = f"cannot trust {spec.unit}: {exc}"
    result = CommandResult(command, 0 if error is None else 1, stderr=error or "")
    _record_external_result(runner, result, mutate=False)
    results.append(result)
    return identity if error is None else None, error


def _parse_systemd_path_list(text: str) -> tuple[Path, ...] | None:
    if not text:
        return ()
    if any(character.isspace() and character != " " for character in text):
        return None
    tokens = text.split(" ")
    if not tokens or any(not token or "\\" in token for token in tokens):
        return None
    paths = tuple(Path(token) for token in tokens)
    if any(not path.is_absolute() for path in paths) or len(paths) != len(set(paths)):
        return None
    return paths


def _parse_systemd_exec_paths(text: str) -> tuple[Path, ...] | None:
    if not text:
        return ()
    paths: list[Path] = []
    cursor = 0
    for match in re.finditer(r"\{([^{}]*)\}", text):
        if text[cursor : match.start()].strip():
            return None
        fields = match.group(1)
        path_matches = re.findall(r"(?:^|;)\s*path=([^;\s]+)\s*(?=;|$)", fields)
        if len(path_matches) != 1:
            return None
        path_text = path_matches[0]
        if "\\" in path_text or "\x00" in path_text:
            return None
        path = Path(path_text)
        if not path.is_absolute():
            return None
        paths.append(path)
        cursor = match.end()
    if text[cursor:].strip() or not paths:
        return None
    return tuple(paths)


def _parse_systemd_environment_files(text: str) -> tuple[Path, ...] | None:
    if not text:
        return ()
    if any(character.isspace() and character != " " for character in text):
        return None
    paths: list[Path] = []
    cursor = 0
    pattern = re.compile(
        r"(?P<path>/[^\\\s()\x00]+) "
        r"\(ignore_errors=(?:yes|no)\)"
    )
    for match in pattern.finditer(text):
        separator = text[cursor : match.start()]
        if separator not in {"", " "}:
            return None
        path = Path(match.group("path"))
        if not path.is_absolute() or path in paths:
            return None
        paths.append(path)
        cursor = match.end()
    if cursor != len(text) or not paths:
        return None
    return tuple(paths)


def _trusted_unit_mask_path(path: Path, unit: str) -> None:
    """Authenticate an exact systemd mask without following its symlink."""

    if Path(unit).name != unit or unit in {"", ".", ".."}:
        raise OSError(f"invalid unit name for trusted mask: {unit!r}")
    if path == Path("/dev/null"):
        metadata = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISCHR(metadata.st_mode)
            or metadata.st_uid not in _TRUSTED_ANCESTOR_UIDS
        ):
            raise OSError("canonical /dev/null is not a trusted character device")
        return

    allowed_paths = tuple(root / unit for root in _TRUSTED_SYSTEMD_MASK_ROOTS)
    if path not in allowed_paths:
        raise OSError(f"masked fragment is outside trusted systemd roots: {path}")

    parts = path.parts
    if not path.is_absolute() or not parts or len(parts) < 2:
        raise OSError(f"trusted mask path is not canonical: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    parent_descriptor = os.open(path.anchor, directory_flags)
    try:
        _validate_trusted_directory_metadata(
            os.fstat(parent_descriptor),
            Path(path.anchor),
        )
        current = Path(path.anchor)
        for component in parts[1:-1]:
            descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            try:
                current /= component
                _validate_trusted_directory_metadata(
                    os.fstat(descriptor),
                    current,
                )
            except BaseException:
                os.close(descriptor)
                raise
            os.close(parent_descriptor)
            parent_descriptor = descriptor

        before = os.stat(
            parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISLNK(before.st_mode) or before.st_uid != _TRUSTED_OWNER_UID:
            raise OSError(
                "trusted systemd mask must be an owner-controlled symbolic link"
            )
        target = os.readlink(parts[-1], dir_fd=parent_descriptor)
        after = os.stat(
            parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    finally:
        os.close(parent_descriptor)

    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise OSError(f"trusted systemd mask changed while inspecting it: {path}")
    if target != "/dev/null":
        raise OSError(f"trusted systemd mask has unexpected target {target!r}")


def _trusted_regular_file(path: Path, *, executable: bool) -> os.stat_result:
    if not path.is_absolute():
        raise OSError(f"trusted path is not absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except RuntimeError as exc:
        raise OSError(f"trusted path cannot be resolved safely: {path}") from exc
    if resolved != path:
        raise OSError(f"trusted path is not canonical: {path}")

    parts = resolved.parts
    if not parts or parts[0] != resolved.anchor or len(parts) < 2:
        raise OSError(f"trusted path has no canonical leaf: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(resolved.anchor, directory_flags)
    try:
        _validate_trusted_directory_metadata(
            os.fstat(parent_descriptor),
            Path(resolved.anchor),
        )
        current = Path(resolved.anchor)
        for component in parts[1:-1]:
            descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            try:
                current /= component
                _validate_trusted_directory_metadata(
                    os.fstat(descriptor),
                    current,
                )
            except BaseException:
                os.close(descriptor)
                raise
            os.close(parent_descriptor)
            parent_descriptor = descriptor
        descriptor = os.open(
            parts[-1],
            file_flags,
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != _TRUSTED_OWNER_UID
        or mode & 0o022
        or (executable and not mode & stat.S_IXUSR)
    ):
        kind = "executable" if executable else "unit file"
        raise OSError(
            f"trusted {kind} must be a root-owned, non-group/world-writable "
            f"regular file{' with owner-execute permission' if executable else ''}: "
            f"{path}"
        )
    return metadata


def _validate_trusted_directory_metadata(
    metadata: os.stat_result,
    path: Path,
) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in _TRUSTED_ANCESTOR_UIDS
        or mode & 0o022
    ):
        raise OSError(
            "trusted path ancestor must be a root-owned, "
            f"non-group/world-writable directory: {path}"
        )


def _process_start_time_ticks(path: Path, main_pid: int) -> int:
    text = read_bounded_utf8(path, max_bytes=_MAX_PROC_STAT_BYTES)
    line = text.rstrip("\n")
    if not line or "\n" in line or "\r" in line or "\x00" in line:
        raise OSError(f"MainPID {main_pid} returned malformed process stat data")
    open_paren = line.find("(")
    close_paren = line.rfind(")")
    if (
        open_paren <= 0
        or close_paren <= open_paren
        or line[:open_paren].strip() != str(main_pid)
    ):
        raise OSError(f"MainPID {main_pid} returned malformed process stat data")
    fields = line[close_paren + 1 :].split()
    if (
        len(fields) < 20
        or len(fields[0]) != 1
        or not fields[19].isdigit()
        or int(fields[19]) <= 0
    ):
        raise OSError(f"MainPID {main_pid} returned malformed process stat data")
    return int(fields[19])


def _service_cgroup_path(text: str, unit: str) -> str | None:
    matched_path: str | None = None
    for line in text.splitlines():
        hierarchy, separator, remainder = line.partition(":")
        controllers, separator_two, cgroup_path = remainder.partition(":")
        if (
            not separator
            or not separator_two
            or (hierarchy and not hierarchy.isdigit())
            or "\x00" in controllers
        ):
            return None
        parts = PurePosixPath(cgroup_path).parts
        if parts == ("/", "system.slice", unit):
            matched_path = f"/system.slice/{unit}"
    return matched_path


def _process_is_in_service_cgroup(text: str, unit: str) -> bool:
    return _service_cgroup_path(text, unit) is not None


def _record_external_start(
    runner: CommandRunner,
    command: list[str],
    *,
    mutate: bool,
) -> None:
    callback = getattr(runner, "record_external_start", None)
    if callable(callback):
        callback(command, mutate)


def _record_external_result(
    runner: CommandRunner,
    result: CommandResult,
    *,
    mutate: bool,
) -> None:
    callback = getattr(runner, "record_external_result", None)
    if callable(callback):
        callback(result, mutate)


def probe_active_gpu_workloads(
    runner: CommandRunner,
    audit: HostAudit | None = None,
    *,
    proc_root: Path | None = None,
    device_root: Path | None = None,
) -> tuple[CommandResult, list[str] | None]:
    """Observe GPU users from independent sources and fail closed on uncertainty."""

    _record_start(runner)
    evidence: dict[str, Any] = {}

    compute_pids, compute_error = _probe_compute_pids(runner, audit)
    evidence["compute_pids"] = compute_pids
    if compute_error is not None:
        # A complete host /proc descriptor scan below observes compute and graphics
        # clients independently of NVML. Keep the failed query as evidence without
        # blocking repair of a broken driver/userspace pair when /proc remains
        # authoritative.
        evidence["compute_query_error"] = compute_error

    device_pids, device_identities, error = _probe_device_user_pids(
        audit,
        proc_root=proc_root or _PROC_ROOT,
        device_root=device_root or _NVIDIA_DEVICE_ROOT,
    )
    evidence["nvidia_device_identities"] = device_identities
    if error is not None:
        return _finish(runner, evidence, error)
    evidence["device_user_pids"] = device_pids

    container_ids, error = _probe_docker_gpu_allocations(runner, audit)
    if error is not None:
        return _finish(runner, evidence, error)
    evidence["docker_gpu_containers"] = container_ids

    process_pids = sorted(set(compute_pids or []) | set(device_pids), key=int)
    workloads = [*(f"pid:{pid}" for pid in process_pids)]
    workloads.extend(f"docker:{container_id[:12]}" for container_id in container_ids)
    result = CommandResult(
        _WORKLOAD_PROBE_COMMAND,
        0,
        stdout=json.dumps(evidence, sort_keys=True, separators=(",", ":")),
    )
    _record_result(runner, result)
    return result, workloads


def is_workload_probe(command: list[str]) -> bool:
    return command == _WORKLOAD_PROBE_COMMAND or command[:2] == [
        "nvidia-smi",
        "--query-compute-apps=pid",
    ]


def _probe_compute_pids(
    runner: CommandRunner, audit: HostAudit | None
) -> tuple[list[str] | None, str | None]:
    driver_absent = bool(
        audit is not None and not audit.module.loaded and not audit.module.devices
    )
    if not runner.exists("nvidia-smi"):
        return (
            ([], None)
            if driver_absent
            else (None, "trusted nvidia-smi executable not found")
        )

    result = runner.run(_COMPUTE_QUERY, allow_fail=True)
    if result.returncode != 0:
        if driver_absent:
            return [], None
        return None, _command_error("nvidia-smi compute-process query failed", result)
    if _truncated(result):
        return None, "nvidia-smi compute-process output was truncated"

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or all("no running processes" in line.lower() for line in lines):
        return [], None
    if not all(line.isdigit() and int(line) > 0 for line in lines):
        return None, "nvidia-smi returned an invalid compute-process inventory"
    return sorted(set(lines), key=int), None


def _probe_device_user_pids(
    audit: HostAudit | None,
    *,
    proc_root: Path,
    device_root: Path,
) -> tuple[list[str], list[str], str | None]:
    if audit is not None and not audit.module.loaded and not audit.module.devices:
        return [], [], None

    device_identities, inventory_error = _nvidia_character_device_identities(
        device_root
    )
    device_numbers = frozenset(identity.rdev for identity in device_identities)
    identity_evidence = [
        f"{os.major(device_number)}:{os.minor(device_number)}"
        for device_number in sorted(device_numbers)
    ]
    if inventory_error is not None:
        return [], identity_evidence, inventory_error

    started = time.monotonic()
    process_count = 0
    total_fds = 0
    total_map_bytes = 0
    total_map_file_stats = 0
    users: set[str] = set()
    try:
        with os.scandir(proc_root) as process_entries:
            for process_entry in process_entries:
                if not process_entry.name.isdigit():
                    continue
                process_count += 1
                if process_count > _MAX_PROCESSES:
                    return (
                        [],
                        identity_evidence,
                        "process inventory exceeded the bounded scan limit",
                    )
                if time.monotonic() - started > _MAX_PROC_SCAN_SECONDS:
                    return (
                        [],
                        identity_evidence,
                        "GPU device-user inspection exceeded its time limit",
                    )

                fd_path = proc_root / process_entry.name / "fd"
                uses_device = False
                try:
                    with os.scandir(fd_path) as fd_entries:
                        process_fds = 0
                        for fd_entry in fd_entries:
                            process_fds += 1
                            total_fds += 1
                            if process_fds > _MAX_FDS_PER_PROCESS:
                                return (
                                    [],
                                    identity_evidence,
                                    f"PID {process_entry.name} exceeded the bounded descriptor scan limit",
                                )
                            if total_fds > _MAX_TOTAL_FDS:
                                return (
                                    [],
                                    identity_evidence,
                                    "file-descriptor inventory exceeded the bounded scan limit",
                                )
                            if time.monotonic() - started > _MAX_PROC_SCAN_SECONDS:
                                return (
                                    [],
                                    identity_evidence,
                                    "GPU device-user inspection exceeded its time limit",
                                )
                            try:
                                descriptor_metadata = _stat_open_descriptor(
                                    Path(fd_entry.path)
                                )
                            except FileNotFoundError:
                                continue
                            except ProcessLookupError:
                                continue
                            except OSError as exc:
                                if exc.errno in {errno.ENOENT, errno.ESRCH}:
                                    continue
                                return (
                                    [],
                                    identity_evidence,
                                    f"cannot inspect PID {process_entry.name} descriptor: {exc}",
                                )
                            if (
                                stat.S_ISCHR(descriptor_metadata.st_mode)
                                and descriptor_metadata.st_rdev in device_numbers
                            ):
                                users.add(process_entry.name)
                                uses_device = True
                                break
                except FileNotFoundError:
                    continue
                except ProcessLookupError:
                    continue
                except OSError as exc:
                    if exc.errno in {errno.ENOENT, errno.ESRCH}:
                        continue
                    return (
                        [],
                        identity_evidence,
                        f"cannot inspect file descriptors for PID {process_entry.name}: {exc}",
                    )
                if uses_device:
                    continue
                mapped, bytes_read, map_file_stats, map_error = (
                    _process_maps_use_nvidia(
                        proc_root / process_entry.name / "maps",
                        proc_root / process_entry.name / "map_files",
                        device_identities,
                    )
                )
                total_map_bytes += bytes_read
                total_map_file_stats += map_file_stats
                if total_map_bytes > _MAX_TOTAL_MAP_BYTES:
                    return (
                        [],
                        identity_evidence,
                        "process memory-map inventory exceeded the bounded scan limit",
                    )
                if map_error is not None:
                    return (
                        [],
                        identity_evidence,
                        f"cannot inspect memory maps for PID {process_entry.name}: {map_error}",
                    )
                if total_map_file_stats > _MAX_TOTAL_MAP_FILE_STATS:
                    return (
                        [],
                        identity_evidence,
                        "mapped-file identity inspection exceeded the bounded scan limit",
                    )
                if mapped:
                    users.add(process_entry.name)
    except OSError as exc:
        return [], identity_evidence, f"cannot enumerate {proc_root}: {exc}"

    return sorted(users, key=int), identity_evidence, None


def _nvidia_character_device_identities(
    device_root: Path,
) -> tuple[frozenset[_NvidiaDeviceIdentity], str | None]:
    """Inventory current NVIDIA character devices by kernel device identity."""

    device_identities: set[_NvidiaDeviceIdentity] = set()
    root_entries = 0
    device_nodes = 0

    def add_device(path: Path) -> str | None:
        nonlocal device_nodes
        device_nodes += 1
        if device_nodes > _MAX_NVIDIA_DEVICE_NODES:
            return "NVIDIA device inventory exceeded the bounded node limit"
        try:
            metadata = os.stat(path, follow_symlinks=True)
        except OSError as exc:
            return f"cannot inspect NVIDIA device node {path}: {exc}"
        if not stat.S_ISCHR(metadata.st_mode):
            return f"NVIDIA device path is not a character device: {path}"
        device_identities.add(
            _NvidiaDeviceIdentity(
                rdev=metadata.st_rdev,
                backing_device=metadata.st_dev,
                inode=metadata.st_ino,
            )
        )
        return None

    try:
        with os.scandir(device_root) as entries:
            for entry in entries:
                root_entries += 1
                if root_entries > _MAX_DEVICE_ROOT_ENTRIES:
                    return (
                        frozenset(),
                        "device-root inventory exceeded the bounded scan limit",
                    )
                if not entry.name.startswith("nvidia"):
                    continue
                path = Path(entry.path)
                try:
                    metadata = os.stat(path, follow_symlinks=True)
                except OSError as exc:
                    return (
                        frozenset(),
                        f"cannot inspect NVIDIA device path {path}: {exc}",
                    )
                if entry.name == "nvidia-caps" and stat.S_ISDIR(metadata.st_mode):
                    try:
                        with os.scandir(path) as cap_entries:
                            for cap_entry in cap_entries:
                                error = add_device(Path(cap_entry.path))
                                if error is not None:
                                    return frozenset(), error
                    except OSError as exc:
                        return (
                            frozenset(),
                            f"cannot enumerate NVIDIA capability devices {path}: {exc}",
                        )
                    continue
                error = add_device(path)
                if error is not None:
                    return frozenset(), error
    except OSError as exc:
        return frozenset(), f"cannot enumerate NVIDIA devices in {device_root}: {exc}"

    if not device_identities:
        return (
            frozenset(),
            (
                "NVIDIA stack is loaded or device-present, but no NVIDIA "
                "character-device identities could be established"
            ),
        )
    return frozenset(device_identities), None


def _stat_open_descriptor(path: Path) -> os.stat_result:
    """Stat the object held by a procfs fd link, independent of its link text."""

    return os.stat(path, follow_symlinks=True)


def _stat_mapped_file(path: Path) -> os.stat_result:
    """Stat a procfs map_files link to recover its held object identity."""

    return os.stat(path, follow_symlinks=True)


def _process_maps_use_nvidia(
    path: Path,
    map_files_root: Path,
    device_identities: frozenset[_NvidiaDeviceIdentity],
) -> tuple[bool, int, int, str | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return False, 0, 0, None
        return False, 0, 0, str(exc)
    total = 0
    map_file_stats = 0
    buffered = b""
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return False, 0, 0, "maps is not a regular procfs file"
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                if buffered:
                    mapped, inspected, error = _maps_line_uses_nvidia(
                        buffered,
                        map_files_root,
                        device_identities,
                    )
                    map_file_stats += inspected
                    if map_file_stats > _MAX_MAP_FILE_STATS_PER_PROCESS:
                        return (
                            False,
                            total,
                            map_file_stats,
                            "mapped-file identity inspection exceeded the per-process limit",
                        )
                    if error is not None or mapped:
                        return mapped, total, map_file_stats, error
                return False, total, map_file_stats, None
            total += len(chunk)
            if total > _MAX_MAP_BYTES_PER_PROCESS:
                return (
                    False,
                    total,
                    map_file_stats,
                    "memory map exceeded the per-process scan limit",
                )
            buffered += chunk
            lines = buffered.split(b"\n")
            buffered = lines.pop()
            if len(buffered) > _MAX_PROC_MAP_LINE_BYTES:
                return (
                    False,
                    total,
                    map_file_stats,
                    "memory map contains an overlong record",
                )
            for line in lines:
                mapped, inspected, error = _maps_line_uses_nvidia(
                    line,
                    map_files_root,
                    device_identities,
                )
                map_file_stats += inspected
                if map_file_stats > _MAX_MAP_FILE_STATS_PER_PROCESS:
                    return (
                        False,
                        total,
                        map_file_stats,
                        "mapped-file identity inspection exceeded the per-process limit",
                    )
                if error is not None or mapped:
                    return mapped, total, map_file_stats, error
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return False, total, map_file_stats, None
        return False, total, map_file_stats, str(exc)
    finally:
        os.close(descriptor)


def _maps_line_uses_nvidia(
    line: bytes,
    map_files_root: Path,
    device_identities: frozenset[_NvidiaDeviceIdentity],
) -> tuple[bool, int, str | None]:
    if not line:
        return False, 0, "memory map contains an empty record"
    if len(line) > _MAX_PROC_MAP_LINE_BYTES:
        return False, 0, "memory map contains an overlong record"
    fields = line.split(None, 5)
    if len(fields) < 5:
        return False, 0, "memory map contains a malformed record"
    address_field = fields[0]
    if re.fullmatch(rb"[0-9A-Fa-f]+-[0-9A-Fa-f]+", address_field) is None:
        return False, 0, "memory map contains an invalid address range"
    device_field = fields[3]
    parts = device_field.split(b":")
    if (
        len(parts) != 2
        or not parts[0]
        or not parts[1]
        or re.fullmatch(rb"[0-9A-Fa-f]+", parts[0]) is None
        or re.fullmatch(rb"[0-9A-Fa-f]+", parts[1]) is None
        or len(parts[0]) > 8
        or len(parts[1]) > 8
    ):
        return False, 0, "memory map contains an invalid device identity"
    inode_field = fields[4]
    if not inode_field.isdigit() or len(inode_field) > 20:
        return False, 0, "memory map contains an invalid inode identity"
    backing_identity = (
        int(parts[0], 16),
        int(parts[1], 16),
        int(inode_field),
    )
    known_backing_identities = {
        (
            os.major(identity.backing_device),
            os.minor(identity.backing_device),
            identity.inode,
        )
        for identity in device_identities
    }
    if backing_identity in known_backing_identities:
        return True, 0, None

    known_backing_devices = {
        (major, minor) for major, minor, _ in known_backing_identities
    }
    pathname = fields[5] if len(fields) == 6 else b""
    potentially_relevant = bool(
        (
            int(inode_field) > 0
            and backing_identity[:2] in known_backing_devices
        )
        or b"nvidia" in pathname.lower()
    )
    if not potentially_relevant:
        return False, 0, None

    try:
        map_file_name = address_field.decode("ascii")
    except UnicodeDecodeError:
        return False, 0, "memory map address range is not ASCII"
    try:
        metadata = _stat_mapped_file(map_files_root / map_file_name)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            try:
                os.stat(map_files_root, follow_symlinks=True)
            except OSError as root_exc:
                if root_exc.errno in {errno.ENOENT, errno.ESRCH}:
                    try:
                        os.stat(map_files_root.parent, follow_symlinks=True)
                    except OSError as process_exc:
                        if process_exc.errno in {errno.ENOENT, errno.ESRCH}:
                            # The process disappeared after the maps snapshot.
                            return False, 1, None
                        return (
                            False,
                            1,
                            (
                                "cannot establish process state while classifying "
                                f"mapped file {map_file_name}: {process_exc}"
                            ),
                        )
                    return (
                        False,
                        1,
                        (
                            "procfs map_files inventory is unavailable for a "
                            f"potentially relevant mapping {map_file_name}"
                        ),
                    )
                return (
                    False,
                    1,
                    (
                        "cannot inspect procfs map_files while classifying "
                        f"mapping {map_file_name}: {root_exc}"
                    ),
                )
            # The map_files directory remains observable, so this exact VMA
            # disappeared after the maps snapshot and is no longer active.
            return False, 1, None
        return (
            False,
            1,
            (
                "cannot classify potentially relevant mapped file "
                f"{map_file_name}: {exc}"
            ),
        )
    known_rdevs = {identity.rdev for identity in device_identities}
    return (
        stat.S_ISCHR(metadata.st_mode) and metadata.st_rdev in known_rdevs,
        1,
        None,
    )


def _probe_docker_gpu_allocations(
    runner: CommandRunner, audit: HostAudit | None
) -> tuple[list[str], str | None]:
    if audit is not None and audit.docker_service_active is False:
        # Querying an inactive socket-activated daemon can itself start Docker.
        # The complete host /proc scan still detects processes currently using
        # NVIDIA devices, while the audited inactive daemon cannot own a
        # separately observable running-container inventory.
        return [], None
    if audit is not None and audit.docker_service_active is not True:
        # Never use a client request to discover whether an unproven daemon is
        # reachable: docker.socket may turn that observation into a mutation.
        return [], "Docker service activity is not proven; refusing a client probe"
    docker_exists = runner.exists("docker")
    if not docker_exists:
        if audit is not None and audit.runtime.docker_installed:
            return (
                [],
                "Docker is installed but its trusted client executable is unavailable",
            )
        return [], None

    listed = runner.run(["docker", "ps", "--quiet", "--no-trunc"], allow_fail=True)
    if listed.returncode != 0:
        return [], _command_error(
            "running Docker containers are not observable", listed
        )
    if _truncated(listed):
        return [], "Docker container inventory was truncated"
    container_ids = [
        line.strip() for line in listed.stdout.splitlines() if line.strip()
    ]
    if len(container_ids) > _MAX_RUNNING_CONTAINERS:
        return [], "running Docker container inventory exceeded the bounded scan limit"
    if len(container_ids) != len(set(container_ids)) or not all(
        _CONTAINER_ID.fullmatch(container_id) for container_id in container_ids
    ):
        return [], "Docker returned an invalid running-container inventory"
    if not container_ids:
        return [], None

    inspected = runner.run(
        ["docker", "inspect", "--type=container", *container_ids],
        allow_fail=True,
    )
    if inspected.returncode != 0:
        return [], _command_error(
            "running Docker containers could not be inspected", inspected
        )
    if _truncated(inspected):
        return [], "Docker container inspection output was truncated"
    try:
        containers = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        return [], "Docker returned invalid container inspection JSON"
    if not isinstance(containers, list) or len(containers) != len(container_ids):
        return [], "Docker returned an incomplete container inspection"

    expected = set(container_ids)
    observed: set[str] = set()
    gpu_containers: list[str] = []
    for container in containers:
        if not isinstance(container, dict):
            return [], "Docker returned a malformed container inspection"
        container_id = container.get("Id")
        if not isinstance(container_id, str) or container_id not in expected:
            return [], "Docker inspection did not match the requested containers"
        if container_id in observed:
            return [], "Docker returned duplicate container inspection state"
        observed.add(container_id)
        state = container.get("State")
        if not isinstance(state, dict) or state.get("Running") is not True:
            return [], "Docker container state changed during workload inspection"
        uses_gpu = _container_uses_gpu(container)
        if uses_gpu is None:
            return (
                [],
                f"Docker container {container_id[:12]} has malformed GPU allocation state",
            )
        if uses_gpu:
            gpu_containers.append(container_id)
    if observed != expected:
        return [], "Docker returned an incomplete container inspection"
    return sorted(gpu_containers), None


def _container_uses_gpu(container: dict[str, Any]) -> bool | None:
    host_config = container.get("HostConfig")
    config = container.get("Config")
    mounts = container.get("Mounts", [])
    if not isinstance(host_config, dict) or not isinstance(config, dict):
        return None
    if not isinstance(mounts, list):
        return None

    runtime = host_config.get("Runtime")
    if runtime is not None and not isinstance(runtime, str):
        return None
    if isinstance(runtime, str) and runtime.lower() == "nvidia":
        return True

    requests = host_config.get("DeviceRequests", [])
    if requests is None:
        requests = []
    if not isinstance(requests, list):
        return None
    for request in requests:
        if not isinstance(request, dict):
            return None
        driver = request.get("Driver", "")
        capabilities = request.get("Capabilities", [])
        if not isinstance(driver, str) or not isinstance(capabilities, list):
            return None
        if driver.lower() == "nvidia":
            return True
        for capability_set in capabilities:
            if not isinstance(capability_set, list) or not all(
                isinstance(capability, str) for capability in capability_set
            ):
                return None
            if any(capability.lower() == "gpu" for capability in capability_set):
                return True

    devices = host_config.get("Devices", [])
    if devices is None:
        devices = []
    if not isinstance(devices, list):
        return None
    for device in devices:
        if not isinstance(device, dict):
            return None
        paths = (device.get("PathOnHost"), device.get("PathInContainer"))
        if any(
            isinstance(path, str) and path.startswith("/dev/nvidia") for path in paths
        ):
            return True

    binds = host_config.get("Binds", [])
    if binds is None:
        binds = []
    if not isinstance(binds, list) or not all(isinstance(bind, str) for bind in binds):
        return None
    if any(bind.startswith("/dev/nvidia") for bind in binds):
        return True
    for mount in mounts:
        if not isinstance(mount, dict):
            return None
        paths = (mount.get("Source"), mount.get("Destination"))
        if any(
            isinstance(path, str) and path.startswith("/dev/nvidia") for path in paths
        ):
            return True

    environment = config.get("Env", [])
    if environment is None:
        environment = []
    if not isinstance(environment, list) or not all(
        isinstance(value, str) for value in environment
    ):
        return None
    for value in environment:
        name, separator, setting = value.partition("=")
        if (
            name == "NVIDIA_VISIBLE_DEVICES"
            and separator
            and setting.strip().lower() not in {"", "none", "void"}
        ):
            return True
    return False


def _finish(
    runner: CommandRunner,
    evidence: dict[str, Any],
    error: str,
) -> tuple[CommandResult, None]:
    result = CommandResult(
        _WORKLOAD_PROBE_COMMAND,
        1,
        stdout=json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        stderr=error,
    )
    _record_result(runner, result)
    return result, None


def _record_start(runner: CommandRunner) -> None:
    callback = getattr(runner, "record_external_start", None)
    if callable(callback):
        callback(_WORKLOAD_PROBE_COMMAND, False)


def _record_result(runner: CommandRunner, result: CommandResult) -> None:
    callback = getattr(runner, "record_external_result", None)
    if callable(callback):
        callback(result, False)


def _truncated(result: CommandResult) -> bool:
    return _OUTPUT_TRUNCATED in result.stdout or _OUTPUT_TRUNCATED in result.stderr


def _command_error(summary: str, result: CommandResult) -> str:
    detail = result.stderr.strip() or result.stdout.strip()
    if detail:
        return f"{summary} (exit {result.returncode}): {detail}"
    return f"{summary} (exit {result.returncode})"
