from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import re
import stat
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import (
    _aggregate_mig_modes,
    _audit_mig_geometry,
    _interesting_package,
    _parse_service_state,
)
from .dnf_transaction import (
    DNF_LOCAL_TRANSACTION_SCRIPT,
    dnf_local_transaction_command,
)
from .files import (
    BoundedFileError,
    atomic_write_text_trusted,
    ensure_private_directory,
    read_bounded_utf8,
    read_bounded_utf8_with_metadata,
    read_trusted_utf8_with_metadata,
    trusted_path_metadata,
    unlink_trusted_path,
)
from .gpu_safety import (
    TrustedDockerSocketIdentity,
    TrustedGpuServiceIdentity,
    revalidate_trusted_docker_socket_identity,
    revalidate_trusted_gpu_service_process_identity,
    validate_active_trusted_gpu_service_identity,
    validate_trusted_docker_socket_unit,
    validate_trusted_docker_socket_unit_identity,
    validate_trusted_gpu_service_unit,
)
from .mig import (
    full_mig_geometry_matches,
    mig_geometry_create_command,
    mig_geometry_destroy_commands,
    restorable_mig_geometry,
)
from .models import (
    CommandResult,
    DesiredState,
    FileSnapshot,
    HostAudit,
    MigComputeInstance,
    MigGpuInstance,
    PackageInfo,
    PackagePayload,
    PackagePayloadBundle,
    RollbackSnapshot,
    Verification,
    utc_now,
)
from .module_safety import ModuleDependencyError, nvidia_module_unload_order
from .package_payloads import (
    PackagePayloadError,
    cleanup_snapshot_payload_artifacts,
    payload_bundle_directory,
    validate_package_payloads,
)
from .planner import package_install_targets, package_policy_package_targets
from .runner import CommandRunner

SNAPSHOT_DIR = Path("/var/lib/nvidia-converge/snapshots")
SNAPSHOT_SCHEMA_VERSION = "2.6"
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_SNAPSHOT_PACKAGES = 4096
MAX_SNAPSHOT_COMMANDS = 256
MAX_COMMAND_PARTS = 4096
MAX_MANAGED_FILE_BYTES = 1024 * 1024
MAX_MIG_GPU_INSTANCES = 64
MAX_MIG_COMPUTE_INSTANCES = 64
_DOCKER_CONFIG_PATH = Path("/etc/docker/daemon.json")
_DNF_MODULE_PATH = Path("/etc/dnf/modules.d/nvidia-driver.module")
_DNF_MODULE_FAILSAFE_DIRECTORY = Path("/var/lib/dnf/modulefailsafe")
_ZYPPER_LOCK_PATH = Path("/etc/zypp/locks")
_ALLOWED_MANAGED_PATHS = {
    str(_DOCKER_CONFIG_PATH),
    str(_DNF_MODULE_PATH),
    str(_ZYPPER_LOCK_PATH),
}
_DNF_MODULE_FAILSAFE_PATH_PATTERN = re.compile(
    r"/var/lib/dnf/modulefailsafe/"
    r"nvidia-driver:[1-9]\d{2,3}-(?:open|dkms):"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}\.yaml"
)


def _is_managed_path(path: str) -> bool:
    return path in _ALLOWED_MANAGED_PATHS or (
        _DNF_MODULE_FAILSAFE_PATH_PATTERN.fullmatch(path) is not None
    )


def _dnf_module_failsafe_path_from_audit(audit: HostAudit) -> str | None:
    if audit.package_manager != "dnf":
        return None
    policy = audit.package_policy
    if not policy.observable or policy.backend != "dnf":
        return None
    selectors = policy.selectors
    if not selectors:
        return None
    if len(selectors) != 1:
        raise RollbackSnapshotError(
            "cannot derive one DNF module fail-safe target from package policy"
        )
    selector = selectors[0]
    stream = selector.version
    if (
        selector.identifier != "nvidia-driver"
        or selector.name != "nvidia-driver"
        or selector.kind != "module"
        or selector.relation != "stream"
        or selector.repositories
        or stream is None
        or re.fullmatch(r"[1-9]\d{2,3}-(?:open|dkms)", stream) is None
    ):
        raise RollbackSnapshotError(
            "cannot derive DNF fail-safe target from a noncanonical module selector"
        )

    directory = _DNF_MODULE_FAILSAFE_DIRECTORY
    try:
        before = trusted_path_metadata(
            directory,
            required_owner_uid=os.geteuid(),
        )
        if before is None or not stat.S_ISDIR(before.st_mode):
            raise RollbackSnapshotError(
                "DNF module fail-safe directory is not a trusted directory"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(directory, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_mode != before.st_mode
                or opened.st_uid != before.st_uid
                or stat.S_IMODE(opened.st_mode) & 0o022
            ):
                raise RollbackSnapshotError(
                    "DNF module fail-safe directory changed or is not trusted"
                )
            names = sorted(os.listdir(descriptor))
            target_pattern = re.compile(
                r"nvidia-driver:"
                + re.escape(stream)
                + r":[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}\.yaml"
            )
            targets = [name for name in names if target_pattern.fullmatch(name)]
            if len(targets) != 1 or sorted(os.listdir(descriptor)) != names:
                raise RollbackSnapshotError(
                    "cannot identify one stable DNF module fail-safe target"
                )
            final = os.fstat(descriptor)
            if (
                final.st_dev != opened.st_dev
                or final.st_ino != opened.st_ino
                or final.st_mode != opened.st_mode
                or final.st_uid != opened.st_uid
                or final.st_mtime_ns != opened.st_mtime_ns
                or final.st_ctime_ns != opened.st_ctime_ns
            ):
                raise RollbackSnapshotError(
                    "DNF module fail-safe directory changed during inventory"
                )
        finally:
            os.close(descriptor)
    except RollbackSnapshotError:
        raise
    except (OSError, BoundedFileError) as exc:
        raise RollbackSnapshotError(
            f"cannot inventory DNF module fail-safe target: {exc}"
        ) from exc
    path = str(directory / targets[0])
    if not _is_managed_path(path):
        raise RollbackSnapshotError("derived DNF module fail-safe target is unsafe")
    return path
_ALLOWED_NVIDIA_MODULES = {
    "nvidia",
    "nvidia_drm",
    "nvidia_fs",
    "nvidia_modeset",
    "nvidia_peermem",
    "nvidia_uvm",
}
_NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE = (
    "nvidia-container-toolkit",
    "nvidia-container-toolkit-base",
    "libnvidia-container-tools",
    "libnvidia-container1",
)
_SNAPSHOT_UNIT_FILE_STATES = frozenset(
    {"enabled", "disabled", "static", "masked", "not-found"}
)
_OUTPUT_TRUNCATED = "[output truncated:"
_DOCKER_SOCKET_UNIT = "docker.socket"
_DOCKER_SERVICE_UNIT = "docker.service"
_NVIDIA_PERSISTENCED_UNIT = "nvidia-persistenced.service"
_FABRIC_MANAGER_UNIT = "nvidia-fabricmanager.service"
_SERVICE_STOP_ORDER = (
    _DOCKER_SOCKET_UNIT,
    _DOCKER_SERVICE_UNIT,
    _NVIDIA_PERSISTENCED_UNIT,
    _FABRIC_MANAGER_UNIT,
)
_SERVICE_ACTIVITY_RESTORE_ORDER = (
    _FABRIC_MANAGER_UNIT,
    _NVIDIA_PERSISTENCED_UNIT,
    _DOCKER_SERVICE_UNIT,
    _DOCKER_SOCKET_UNIT,
)


def _validate_launcher_mutation(
    runner: CommandRunner,
    service: str,
    *,
    allow_masked: bool,
) -> tuple[list[CommandResult], bool]:
    """Return a fail-closed precondition for an applied socket mutation."""

    if not bool(getattr(runner, "apply", False)):
        return [], True
    if service == _DOCKER_SOCKET_UNIT:
        results, error = validate_trusted_docker_socket_unit(
            runner,
            allow_masked=allow_masked,
        )
    else:
        results, error = validate_trusted_gpu_service_unit(
            runner,
            service,
            allow_masked=allow_masked,
        )
    return results, error is None


class RollbackSnapshotError(ValueError):
    pass


def new_snapshot_path(
    path: str | None = None,
    *,
    operation_id: str | None = None,
) -> Path:
    if path is None and operation_id is not None:
        if re.fullmatch(r"[a-f0-9]{32}", operation_id) is None:
            raise RollbackSnapshotError(
                "snapshot path operation_id must be a 32-character lowercase hex identifier"
            )
        return SNAPSHOT_DIR / f"snapshot-{operation_id}.json"
    timestamp = utc_now().replace(":", "-").replace("+", "-")
    return Path(
        os.path.abspath(
            path
            if path
            else SNAPSHOT_DIR / f"{timestamp}-{uuid4().hex}.json"
        )
    )


def cleanup_staged_snapshot_artifacts(
    operation_id: str,
    preserve_bound_authority: bool,
    required_owner_uid: int,
) -> None:
    """Clean exact safe-private artifacts for one interrupted operation."""

    snapshot_path = new_snapshot_path(operation_id=operation_id)
    try:
        cleanup_snapshot_payload_artifacts(
            snapshot_path,
            preserve_bound_authority=preserve_bound_authority,
            required_owner_uid=required_owner_uid,
        )
    except PackagePayloadError as exc:
        raise RollbackSnapshotError(
            f"cannot clean interrupted rollback snapshot authority: {exc}"
        ) from exc


@dataclass(frozen=True)
class _SnapshotSourceBinding:
    path: Path
    content_sha256: str
    file_metadata: tuple[int, ...]
    parent_metadata: tuple[int, ...]


def _unit_state_consistency_error(
    unit: str,
    active: bool | None,
    enabled: bool | None,
    unit_file_state: str | None,
) -> str | None:
    if unit_file_state not in _SNAPSHOT_UNIT_FILE_STATES:
        return f"{unit} has unsupported unit-file state {unit_file_state!r}"
    if active is None or enabled is None:
        return f"{unit} has unknown active or enabled state"
    if enabled is not (unit_file_state == "enabled"):
        return f"{unit} enabled state contradicts {unit_file_state!r}"
    if unit_file_state in {"masked", "not-found"} and active:
        return f"{unit} cannot be active while {unit_file_state}"
    return None


def _service_state_consistency_error(audit: HostAudit) -> str | None:
    for unit, active, enabled, unit_file_state in (
        (
            _DOCKER_SOCKET_UNIT,
            audit.docker_socket_active,
            audit.docker_socket_enabled,
            audit.docker_socket_unit_file_state,
        ),
        (
            _DOCKER_SERVICE_UNIT,
            audit.docker_service_active,
            audit.docker_service_enabled,
            audit.docker_service_unit_file_state,
        ),
        (
            _NVIDIA_PERSISTENCED_UNIT,
            audit.nvidia_persistenced_active,
            audit.nvidia_persistenced_enabled,
            audit.nvidia_persistenced_unit_file_state,
        ),
        (
            _FABRIC_MANAGER_UNIT,
            audit.fabric_manager_active,
            audit.fabric_manager_enabled,
            audit.fabric_manager_unit_file_state,
        ),
    ):
        error = _unit_state_consistency_error(
            unit,
            active,
            enabled,
            unit_file_state,
        )
        if error is not None:
            return error
    return None


def _snapshot_service_state_consistency_error(
    snapshot: RollbackSnapshot,
) -> str | None:
    for unit, active, enabled, unit_file_state in (
        (
            _DOCKER_SOCKET_UNIT,
            snapshot.docker_socket_active,
            snapshot.docker_socket_enabled,
            snapshot.docker_socket_unit_file_state,
        ),
        (
            _DOCKER_SERVICE_UNIT,
            snapshot.docker_service_active,
            snapshot.docker_service_enabled,
            snapshot.docker_service_unit_file_state,
        ),
        (
            _NVIDIA_PERSISTENCED_UNIT,
            snapshot.nvidia_persistenced_active,
            snapshot.nvidia_persistenced_enabled,
            snapshot.nvidia_persistenced_unit_file_state,
        ),
        (
            _FABRIC_MANAGER_UNIT,
            snapshot.fabric_manager_active,
            snapshot.fabric_manager_enabled,
            snapshot.fabric_manager_unit_file_state,
        ),
    ):
        error = _unit_state_consistency_error(
            unit,
            active,
            enabled,
            unit_file_state,
        )
        if error is not None:
            return error
    return None


def create_snapshot(
    audit: HostAudit,
    path: str | None = None,
    *,
    desired: DesiredState | None = None,
    persist: bool = True,
    operation_id: str | None = None,
    package_payloads: PackagePayloadBundle | None = None,
    dnf_module_failsafe_path: str | None = None,
) -> RollbackSnapshot:
    if persist and dnf_module_failsafe_path is None:
        dnf_module_failsafe_path = _dnf_module_failsafe_path_from_audit(audit)
    if persist and audit.package_manager == "dnf" and dnf_module_failsafe_path is None:
        raise RollbackSnapshotError(
            "schema-v2.6 DNF snapshot requires one exact module fail-safe target"
        )
    if dnf_module_failsafe_path is not None and (
        audit.package_manager != "dnf"
        or not _is_managed_path(dnf_module_failsafe_path)
        or not dnf_module_failsafe_path.startswith(
            str(_DNF_MODULE_FAILSAFE_DIRECTORY) + "/"
        )
    ):
        raise RollbackSnapshotError(
            "DNF module fail-safe rollback target is not an exact managed path"
        )
    if audit.mig_mode == "mixed":
        raise RollbackSnapshotError(
            "cannot snapshot a mixed per-GPU MIG baseline; normalize MIG mode before convergence"
        )
    if persist and audit.mig_mode_pending is None:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without an observable pending MIG mode"
        )
    if persist and audit.mig_mode_pending != audit.mig_mode:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot while a MIG transition is pending"
        )
    if persist and not audit.mig_geometry_complete:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without a complete MIG GI/CI geometry observation"
        )
    if (
        persist
        and audit.mig_mode == "disabled"
        and (audit.mig_geometry or audit.mig_device_uuids)
    ):
        raise RollbackSnapshotError(
            "cannot snapshot MIG devices while current MIG mode is disabled"
        )
    mig_change_requested = bool(
        desired
        and (
            audit.mig_mode != desired.mig
            or (
                desired.mig == "enabled"
                and not full_mig_geometry_matches(
                    audit.mig_geometry,
                    audit.gpu_uuids,
                )
            )
        )
    )
    if (
        persist
        and mig_change_requested
        and not restorable_mig_geometry(audit.mig_geometry, audit.gpu_uuids)
    ):
        raise RollbackSnapshotError(
            "cannot mutate MIG from a baseline whose exact GI/CI geometry is not safely restorable"
        )
    if persist and not audit.gpu_uuids:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without an observable GPU UUID inventory"
        )
    if persist and audit.package_manager is None:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without a supported package manager"
        )
    if persist and not audit.os_id:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without an observable OS identity"
        )
    if persist and not audit.package_inventory_complete:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot from an incomplete package inventory"
        )
    exact_service_states: dict[str, bool | str | None] = {
        "docker.service active": audit.docker_service_active,
        "docker.service enabled": audit.docker_service_enabled,
        "docker.service unit-file": audit.docker_service_unit_file_state,
        "docker.socket active": audit.docker_socket_active,
        "docker.socket enabled": audit.docker_socket_enabled,
        "docker.socket unit-file": audit.docker_socket_unit_file_state,
        "nvidia-persistenced.service active": audit.nvidia_persistenced_active,
        "nvidia-persistenced.service enabled": audit.nvidia_persistenced_enabled,
        "nvidia-persistenced.service unit-file": (
            audit.nvidia_persistenced_unit_file_state
        ),
        "nvidia-fabricmanager.service active": audit.fabric_manager_active,
        "nvidia-fabricmanager.service enabled": audit.fabric_manager_enabled,
        "nvidia-fabricmanager.service unit-file": (
            audit.fabric_manager_unit_file_state
        ),
    }
    unknown_service_states = sorted(
        name for name, state in exact_service_states.items() if state is None
    )
    if persist and unknown_service_states:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without exact "
            "transactional service state: " + ", ".join(unknown_service_states)
        )
    service_state_error = _service_state_consistency_error(audit)
    if persist and service_state_error is not None:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot with inconsistent "
            f"transactional service state: {service_state_error}"
        )
    snapshot_operation_id = operation_id or uuid4().hex
    if re.fullmatch(r"[a-f0-9]{32}", snapshot_operation_id) is None:
        raise RollbackSnapshotError(
            "rollback snapshot operation_id must be a 32-character lowercase hex identifier"
        )
    incomplete_versions = [
        pkg.name for pkg in audit.packages if pkg.installed and not pkg.version
    ]
    if persist and incomplete_versions:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot with unknown package versions: "
            + ", ".join(sorted(incomplete_versions))
        )
    try:
        module_names = nvidia_module_unload_order()
    except ModuleDependencyError as exc:
        if persist:
            raise RollbackSnapshotError(
                f"cannot capture a restorable NVIDIA module graph: {exc}"
            ) from exc
        module_names = []
    if persist and audit.module.loaded != ("nvidia" in module_names):
        raise RollbackSnapshotError(
            "loaded NVIDIA module audit does not match the observable sysfs module graph"
        )
    snapshot_path = new_snapshot_path(path)
    if persist and package_payloads is None:
        raise RollbackSnapshotError(
            "cannot persist an applicable rollback snapshot without a complete package payload bundle"
    )
    if persist:
        assert audit.package_manager is not None
        assert package_payloads is not None
        try:
            validate_package_payloads(
                snapshot_path,
                package_payloads,
                [package for package in audit.packages if package.installed],
                audit.package_manager,
                required_owner_uid=os.geteuid(),
            )
        except PackagePayloadError as exc:
            raise RollbackSnapshotError(
                f"cannot bind rollback package payloads: {exc}"
            ) from exc
    remove_packages: list[str] = []
    if desired and audit.package_manager:
        installed_names = {pkg.name for pkg in audit.packages if pkg.installed}
        targets = [
            *package_install_targets(desired, audit),
            *package_policy_package_targets(desired, audit),
        ]
        tracking_names = _package_tracking_names(
            audit.package_manager,
            targets,
            audit.kernel.running,
        )
        if "nvidia-container-toolkit" in tracking_names:
            # NVIDIA publishes the toolkit as a small dependency closure on both
            # DEB and RPM platforms.  Record every package that the direct target
            # can introduce so rollback does not leave runtime binaries or
            # libraries behind.  The complete pre-audit inventory below prevents
            # removal of closure members that were already part of the baseline.
            tracking_names = list(
                dict.fromkeys(
                    [
                        *tracking_names,
                        *_NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE,
                    ]
                )
            )
        remove_packages = [
            name for name in tracking_names if name not in installed_names
        ]
    commands = _rollback_commands(
        audit.packages,
        audit.package_manager,
        remove_packages=remove_packages,
        snapshot_path=str(snapshot_path) if persist else None,
        package_payloads=package_payloads,
    )
    snapshot = RollbackSnapshot(
        path=str(snapshot_path) if persist else None,
        packages=[pkg for pkg in audit.packages if pkg.installed],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=commands,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        created_at=utc_now(),
        operation_id=snapshot_operation_id,
        host_id=_host_identity(),
        os_id=audit.os_id,
        os_version=audit.os_version,
        architecture=platform.machine(),
        package_manager=audit.package_manager,
        introduced_packages=remove_packages,
        module_loaded=audit.module.loaded,
        module_names=module_names,
        module_open_module=audit.module.open_module,
        module_signed=audit.module.signed,
        module_installed_version=audit.module.installed_version,
        module_installed_open_module=audit.module.installed_open_module,
        module_installed_signed=audit.module.installed_signed,
        mig_mode=audit.mig_mode,
        docker_service_active=audit.docker_service_active,
        docker_service_enabled=audit.docker_service_enabled,
        docker_service_unit_file_state=audit.docker_service_unit_file_state,
        docker_socket_active=audit.docker_socket_active,
        docker_socket_enabled=audit.docker_socket_enabled,
        docker_socket_unit_file_state=audit.docker_socket_unit_file_state,
        nvidia_persistenced_active=audit.nvidia_persistenced_active,
        nvidia_persistenced_enabled=audit.nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state=(audit.nvidia_persistenced_unit_file_state),
        fabric_manager_active=audit.fabric_manager_active,
        fabric_manager_enabled=audit.fabric_manager_enabled,
        fabric_manager_unit_file_state=audit.fabric_manager_unit_file_state,
        managed_files=(
            (
                _capture_managed_files(audit.package_manager)
                if dnf_module_failsafe_path is None
                else _capture_managed_files(
                    audit.package_manager,
                    dnf_module_failsafe_path=dnf_module_failsafe_path,
                )
            )
            if persist
            else []
        ),
        package_payloads=package_payloads,
        gpu_uuids=list(audit.gpu_uuids),
        mig_geometry=list(audit.mig_geometry),
    )
    snapshot.integrity_sha256 = _snapshot_integrity(snapshot)
    if not persist:
        return snapshot
    _ensure_private_snapshot_directory(snapshot_path.parent)
    text = json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n"
    if len(text.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        raise RollbackSnapshotError(
            "rollback snapshot exceeds the supported serialized size"
        )
    try:
        atomic_write_text_trusted(
            Path(os.path.abspath(snapshot_path)),
            text,
            mode=0o600,
            required_owner_uid=os.geteuid(),
            deterministic_temporary_name=f".{snapshot_path.name}.tmp",
        )
    except (OSError, BoundedFileError) as exc:
        raise RollbackSnapshotError(
            f"cannot persist rollback snapshot {str(snapshot_path)!r}: {exc}"
        ) from exc
    return snapshot


def _ensure_private_snapshot_directory(directory: Path) -> None:
    try:
        ensure_private_directory(
            Path(os.path.abspath(directory)),
            required_owner_uid=os.geteuid(),
        )
    except (OSError, BoundedFileError, ValueError) as exc:
        raise RollbackSnapshotError(
            f"cannot create rollback snapshot directory {str(directory)!r}: {exc}"
        ) from exc


def _capture_managed_files(
    package_manager: str | None,
    *,
    dnf_module_failsafe_path: str | None = None,
) -> list[FileSnapshot]:
    paths = [_DOCKER_CONFIG_PATH]
    if package_manager in {"dnf", "yum"}:
        paths.append(_DNF_MODULE_PATH)
        if dnf_module_failsafe_path is not None:
            if package_manager != "dnf" or not _is_managed_path(
                dnf_module_failsafe_path
            ):
                raise RollbackSnapshotError(
                    "DNF module fail-safe rollback target is not managed"
                )
            paths.append(Path(dnf_module_failsafe_path))
    elif package_manager == "zypper":
        paths.append(_ZYPPER_LOCK_PATH)
    captured: list[FileSnapshot] = []
    for path in paths:
        try:
            content, metadata = read_bounded_utf8_with_metadata(
                path,
                max_bytes=MAX_MANAGED_FILE_BYTES,
                required_owner_uid=os.geteuid(),
                require_trusted_ancestors=True,
            )
        except FileNotFoundError:
            captured.append(FileSnapshot(str(path), False, None, None))
            continue
        except (OSError, BoundedFileError) as exc:
            raise RollbackSnapshotError(
                f"cannot capture managed state file {str(path)!r}: {exc}"
            ) from exc
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        captured.append(
            FileSnapshot(
                str(path),
                True,
                encoded,
                stat.S_IMODE(metadata.st_mode),
            )
        )
    return captured


def load_snapshot(path: str, *, require_private: bool = False) -> RollbackSnapshot:
    source_path = Path(os.path.abspath(path))
    try:
        source = read_trusted_utf8_with_metadata(
            source_path,
            max_bytes=MAX_SNAPSHOT_BYTES,
            required_owner_uid=os.geteuid() if require_private else None,
            require_private_parent=require_private,
        )
        if require_private and stat.S_IMODE(source.file_metadata.st_mode) & 0o077:
            raise BoundedFileError(
                "rollback snapshot file must be private to the effective uid"
            )
        data = json.loads(
            source.text,
            object_pairs_hook=_object_without_duplicates,
        )
    except OSError as exc:
        raise RollbackSnapshotError(
            f"cannot read rollback snapshot {path!r}: {exc.strerror}"
        ) from exc
    except BoundedFileError as exc:
        raise RollbackSnapshotError(
            f"cannot read rollback snapshot {path!r}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RollbackSnapshotError(
            f"invalid rollback snapshot JSON {path!r}: line {exc.lineno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise RollbackSnapshotError(
            f"invalid rollback snapshot JSON {path!r}: {exc}"
        ) from exc
    except RecursionError as exc:
        raise RollbackSnapshotError(
            "rollback snapshot structure is too deeply nested"
        ) from exc
    if not isinstance(data, dict):
        raise RollbackSnapshotError("rollback snapshot must be a JSON object")
    required = {
        "architecture",
        "commands",
        "created_at",
        "operation_id",
        "host_id",
        "integrity_sha256",
        "introduced_packages",
        "kernel",
        "managed_files",
        "module_loaded",
        "module_names",
        "module_open_module",
        "module_signed",
        "module_installed_version",
        "module_installed_open_module",
        "module_installed_signed",
        "module_version",
        "mig_geometry",
        "mig_mode",
        "docker_service_active",
        "docker_service_enabled",
        "docker_service_unit_file_state",
        "docker_socket_active",
        "docker_socket_enabled",
        "docker_socket_unit_file_state",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
        "nvidia_persistenced_unit_file_state",
        "fabric_manager_active",
        "fabric_manager_enabled",
        "fabric_manager_unit_file_state",
        "gpu_uuids",
        "os_id",
        "os_version",
        "package_manager",
        "package_payloads",
        "packages",
        "path",
        "schema_version",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RollbackSnapshotError(
            f"rollback snapshot missing required field(s): {', '.join(missing)}"
        )
    unknown = sorted(set(data) - required)
    if unknown:
        raise RollbackSnapshotError(
            f"rollback snapshot contains unknown field(s): {', '.join(unknown)}"
        )
    schema_version = _required_string(data["schema_version"], "schema_version")
    if schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise RollbackSnapshotError(
            f"unsupported rollback snapshot schema_version {schema_version!r}; expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    snapshot_path = _required_string(data["path"], "path")
    if Path(snapshot_path) != source_path:
        raise RollbackSnapshotError(
            "rollback snapshot path binding does not match the loaded file"
        )
    created_at = _required_string(data["created_at"], "created_at")
    try:
        datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise RollbackSnapshotError(
            "rollback snapshot created_at must be an ISO-8601 timestamp"
        ) from exc
    operation_id = _required_string(data["operation_id"], "operation_id")
    if re.fullmatch(r"[a-f0-9]{32}", operation_id) is None:
        raise RollbackSnapshotError(
            "rollback snapshot operation_id must be a 32-character lowercase hex identifier"
        )
    host_id = _required_string(data["host_id"], "host_id")
    os_id = _optional_string(data["os_id"], "os_id")
    os_version = _optional_string(data["os_version"], "os_version")
    architecture = _required_string(data["architecture"], "architecture")
    package_manager = _required_string(data["package_manager"], "package_manager")
    if package_manager not in {"apt-get", "dnf", "yum", "zypper"}:
        raise RollbackSnapshotError(
            "rollback snapshot package_manager is not supported"
        )
    kernel = _required_string(data["kernel"], "kernel")
    module_version = _optional_string(data.get("module_version"), "module_version")
    module_loaded = _required_boolean(data["module_loaded"], "module_loaded")
    module_names = _load_module_names(data["module_names"], module_loaded)
    module_open_module = _optional_boolean(
        data["module_open_module"], "module_open_module"
    )
    module_signed = _optional_boolean(data["module_signed"], "module_signed")
    module_installed_version = _optional_string(
        data["module_installed_version"], "module_installed_version"
    )
    module_installed_open_module = _optional_boolean(
        data["module_installed_open_module"], "module_installed_open_module"
    )
    module_installed_signed = _optional_boolean(
        data["module_installed_signed"], "module_installed_signed"
    )
    mig_mode = _optional_string(data["mig_mode"], "mig_mode")
    if mig_mode not in {None, "enabled", "disabled"}:
        raise RollbackSnapshotError("rollback snapshot mig_mode is invalid")
    docker_service_active = _required_boolean(
        data["docker_service_active"], "docker_service_active"
    )
    docker_service_enabled = _required_boolean(
        data["docker_service_enabled"], "docker_service_enabled"
    )
    docker_service_unit_file_state = _required_unit_file_state(
        data["docker_service_unit_file_state"],
        "docker_service_unit_file_state",
    )
    docker_socket_active = _required_boolean(
        data["docker_socket_active"], "docker_socket_active"
    )
    docker_socket_enabled = _required_boolean(
        data["docker_socket_enabled"], "docker_socket_enabled"
    )
    docker_socket_unit_file_state = _required_unit_file_state(
        data["docker_socket_unit_file_state"],
        "docker_socket_unit_file_state",
    )
    nvidia_persistenced_active = _required_boolean(
        data["nvidia_persistenced_active"], "nvidia_persistenced_active"
    )
    nvidia_persistenced_enabled = _required_boolean(
        data["nvidia_persistenced_enabled"], "nvidia_persistenced_enabled"
    )
    nvidia_persistenced_unit_file_state = _required_unit_file_state(
        data["nvidia_persistenced_unit_file_state"],
        "nvidia_persistenced_unit_file_state",
    )
    fabric_manager_active = _required_boolean(
        data["fabric_manager_active"], "fabric_manager_active"
    )
    fabric_manager_enabled = _required_boolean(
        data["fabric_manager_enabled"], "fabric_manager_enabled"
    )
    fabric_manager_unit_file_state = _required_unit_file_state(
        data["fabric_manager_unit_file_state"],
        "fabric_manager_unit_file_state",
    )
    gpu_uuids = _load_gpu_uuids(data["gpu_uuids"])
    mig_geometry = _load_mig_geometry(data["mig_geometry"], gpu_uuids)
    if mig_mode == "disabled" and mig_geometry:
        raise RollbackSnapshotError(
            "rollback snapshot cannot contain MIG geometry while mig_mode is disabled"
        )
    packages = _load_packages(data["packages"])
    expected_package_manager = "apt" if package_manager == "apt-get" else "rpm"
    mismatched_managers = sorted(
        pkg.name for pkg in packages if pkg.manager != expected_package_manager
    )
    if mismatched_managers:
        raise RollbackSnapshotError(
            "rollback snapshot packages do not match package_manager: "
            + ", ".join(mismatched_managers)
        )
    introduced_packages = _load_introduced_packages(data["introduced_packages"])
    package_payloads = _load_package_payloads(
        data["package_payloads"],
        package_manager,
        packages,
        snapshot_path,
    )
    managed_files = _load_managed_files(data["managed_files"], package_manager)
    commands = _load_commands(data["commands"])
    integrity = _required_string(data["integrity_sha256"], "integrity_sha256")
    if re.fullmatch(r"[a-f0-9]{64}", integrity) is None:
        raise RollbackSnapshotError(
            "rollback snapshot integrity_sha256 must be a lowercase SHA-256 digest"
        )
    snapshot = RollbackSnapshot(
        path=snapshot_path,
        packages=packages,
        kernel=kernel,
        module_version=module_version,
        commands=commands,
        schema_version=schema_version,
        created_at=created_at,
        operation_id=operation_id,
        host_id=host_id,
        os_id=os_id,
        os_version=os_version,
        architecture=architecture,
        package_manager=package_manager,
        introduced_packages=introduced_packages,
        module_loaded=module_loaded,
        module_names=module_names,
        module_open_module=module_open_module,
        module_signed=module_signed,
        module_installed_version=module_installed_version,
        module_installed_open_module=module_installed_open_module,
        module_installed_signed=module_installed_signed,
        mig_mode=mig_mode,
        docker_service_active=docker_service_active,
        docker_service_enabled=docker_service_enabled,
        docker_service_unit_file_state=docker_service_unit_file_state,
        docker_socket_active=docker_socket_active,
        docker_socket_enabled=docker_socket_enabled,
        docker_socket_unit_file_state=docker_socket_unit_file_state,
        nvidia_persistenced_active=nvidia_persistenced_active,
        nvidia_persistenced_enabled=nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state=(nvidia_persistenced_unit_file_state),
        fabric_manager_active=fabric_manager_active,
        fabric_manager_enabled=fabric_manager_enabled,
        fabric_manager_unit_file_state=fabric_manager_unit_file_state,
        managed_files=managed_files,
        package_payloads=package_payloads,
        gpu_uuids=gpu_uuids,
        mig_geometry=mig_geometry,
        integrity_sha256=integrity,
    )
    service_state_error = _snapshot_service_state_consistency_error(snapshot)
    if service_state_error is not None:
        raise RollbackSnapshotError(
            "rollback snapshot has inconsistent transactional service state: "
            + service_state_error
        )
    expected_commands = _rollback_commands(
        snapshot.packages,
        snapshot.package_manager,
        remove_packages=snapshot.introduced_packages,
        snapshot_path=snapshot.path,
        package_payloads=snapshot.package_payloads,
    )
    if commands != expected_commands:
        raise RollbackSnapshotError(
            "rollback snapshot commands are inconsistent with its declarative package state"
        )
    expected_integrity = _snapshot_integrity(snapshot)
    if not hmac.compare_digest(integrity, expected_integrity):
        raise RollbackSnapshotError("rollback snapshot integrity check failed")
    try:
        validate_package_payloads(
            Path(snapshot.path or ""),
            package_payloads,
            packages,
            package_manager,
            required_owner_uid=os.geteuid(),
        )
    except PackagePayloadError as exc:
        raise RollbackSnapshotError(
            f"rollback snapshot package payloads are unusable: {exc}"
        ) from exc
    source_bound_snapshot: Any = snapshot
    source_bound_snapshot._source_binding = _SnapshotSourceBinding(
        source_path,
        hashlib.sha256(source.text.encode("utf-8")).hexdigest(),
        _metadata_fingerprint(source.file_metadata),
        _metadata_fingerprint(source.parent_metadata),
    )
    return snapshot


def _load_mig_geometry(
    value: Any,
    gpu_uuids: list[str],
) -> list[MigGpuInstance]:
    if not isinstance(value, list):
        raise RollbackSnapshotError("rollback snapshot mig_geometry must be an array")
    if len(value) > MAX_MIG_GPU_INSTANCES:
        raise RollbackSnapshotError(
            f"rollback snapshot mig_geometry exceeds {MAX_MIG_GPU_INSTANCES} entries"
        )
    geometry: list[MigGpuInstance] = []
    placements: set[tuple[str, int]] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise RollbackSnapshotError(
                f"rollback snapshot mig_geometry[{index}] must be an object"
            )
        expected_fields = {
            "compute_instances",
            "gpu_uuid",
            "placement_size",
            "placement_start",
            "profile",
            "profile_id",
        }
        if set(entry) != expected_fields:
            raise RollbackSnapshotError(
                f"rollback snapshot mig_geometry[{index}] has invalid fields"
            )
        gpu_uuid = _required_string(
            entry["gpu_uuid"], f"mig_geometry[{index}].gpu_uuid"
        )
        if gpu_uuid not in gpu_uuids:
            raise RollbackSnapshotError(
                f"rollback snapshot mig_geometry[{index}] references an unknown GPU UUID"
            )
        profile = _load_mig_profile(entry["profile"], f"mig_geometry[{index}].profile")
        profile_id = _load_nonnegative_integer(
            entry["profile_id"], f"mig_geometry[{index}].profile_id"
        )
        placement_start = _load_nonnegative_integer(
            entry["placement_start"],
            f"mig_geometry[{index}].placement_start",
        )
        placement_size = _load_nonnegative_integer(
            entry["placement_size"],
            f"mig_geometry[{index}].placement_size",
            minimum=1,
        )
        placement = (gpu_uuid, placement_start)
        if placement in placements:
            raise RollbackSnapshotError(
                "rollback snapshot mig_geometry contains duplicate GPU placements"
            )
        placements.add(placement)
        compute_value = entry["compute_instances"]
        if not isinstance(compute_value, list):
            raise RollbackSnapshotError(
                f"rollback snapshot mig_geometry[{index}].compute_instances must be an array"
            )
        if len(compute_value) > MAX_MIG_COMPUTE_INSTANCES:
            raise RollbackSnapshotError(
                "rollback snapshot MIG compute instance count exceeds the supported maximum"
            )
        compute_instances: list[MigComputeInstance] = []
        for compute_index, compute_entry in enumerate(compute_value):
            name = f"mig_geometry[{index}].compute_instances[{compute_index}]"
            if not isinstance(compute_entry, dict) or set(compute_entry) != {
                "profile",
                "profile_id",
            }:
                raise RollbackSnapshotError(
                    f"rollback snapshot {name} has invalid fields"
                )
            compute_instances.append(
                MigComputeInstance(
                    profile=_load_mig_profile(
                        compute_entry["profile"], f"{name}.profile"
                    ),
                    profile_id=_load_nonnegative_integer(
                        compute_entry["profile_id"], f"{name}.profile_id"
                    ),
                )
            )
        geometry.append(
            MigGpuInstance(
                gpu_uuid=gpu_uuid,
                profile=profile,
                profile_id=profile_id,
                placement_start=placement_start,
                placement_size=placement_size,
                compute_instances=compute_instances,
            )
        )
    return geometry


def _load_mig_profile(value: Any, name: str) -> str:
    profile = _required_string(value, name)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}", profile) is None:
        raise RollbackSnapshotError(f"rollback snapshot {name} is invalid")
    return profile


def _load_nonnegative_integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RollbackSnapshotError(
            f"rollback snapshot {name} must be an integer >= {minimum}"
        )
    return value


def _load_packages(value: Any) -> list[PackageInfo]:
    if not isinstance(value, list):
        raise RollbackSnapshotError("rollback snapshot packages must be an array")
    if len(value) > MAX_SNAPSHOT_PACKAGES:
        raise RollbackSnapshotError(
            f"rollback snapshot packages exceeds {MAX_SNAPSHOT_PACKAGES} entries"
        )
    packages: list[PackageInfo] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise RollbackSnapshotError(
                f"rollback snapshot packages[{index}] must be an object"
            )
        expected_fields = {
            "architecture",
            "epoch",
            "installed",
            "manager",
            "name",
            "version",
        }
        if set(entry) != expected_fields:
            raise RollbackSnapshotError(
                f"rollback snapshot packages[{index}] must contain exactly: "
                + ", ".join(sorted(expected_fields))
            )
        name = _required_string(entry.get("name"), f"packages[{index}].name")
        if not _interesting_package(name):
            raise RollbackSnapshotError(
                f"rollback snapshot packages[{index}].name is outside the NVIDIA stack scope"
            )
        version = _optional_package_string(
            entry.get("version"), f"packages[{index}].version"
        )
        manager = _optional_package_string(
            entry.get("manager"), f"packages[{index}].manager"
        )
        architecture = _optional_package_string(
            entry.get("architecture"), f"packages[{index}].architecture"
        )
        epoch = _optional_package_string(entry.get("epoch"), f"packages[{index}].epoch")
        installed = entry.get("installed")
        if installed is not True:
            raise RollbackSnapshotError(
                f"rollback snapshot packages[{index}].installed must be true"
            )
        if version is None:
            raise RollbackSnapshotError(
                f"rollback snapshot packages[{index}].version must be known"
            )
        if architecture is None:
            raise RollbackSnapshotError(
                f"rollback snapshot packages[{index}].architecture must be known"
            )
        packages.append(
            PackageInfo(
                name=name,
                version=version,
                manager=manager,
                installed=installed,
                architecture=architecture,
                epoch=epoch,
            )
        )
    identities = [_package_state_identity(package) for package in packages]
    if len(set(identities)) != len(identities):
        raise RollbackSnapshotError("rollback snapshot packages contains duplicates")
    return packages


def _load_package_payloads(
    value: Any,
    package_manager: str,
    baseline_packages: list[PackageInfo],
    snapshot_path: str,
) -> PackagePayloadBundle:
    if not isinstance(value, dict) or set(value) != {
        "directory",
        "packages",
        "total_size_bytes",
    }:
        raise RollbackSnapshotError(
            "rollback snapshot package_payloads must contain exactly: "
            "directory, packages, total_size_bytes"
        )
    directory = _required_string(
        value["directory"], "package_payloads.directory"
    )
    try:
        expected_directory = payload_bundle_directory(Path(snapshot_path))
    except PackagePayloadError as exc:
        raise RollbackSnapshotError(
            f"rollback snapshot package payload path is invalid: {exc}"
        ) from exc
    if directory != expected_directory:
        raise RollbackSnapshotError(
            "rollback snapshot package payload directory is not bound to its path"
        )
    raw_packages = value["packages"]
    if not isinstance(raw_packages, list):
        raise RollbackSnapshotError(
            "rollback snapshot package_payloads.packages must be an array"
        )
    if len(raw_packages) > MAX_SNAPSHOT_PACKAGES * 2:
        raise RollbackSnapshotError(
            "rollback snapshot package payload manifest exceeds the supported entry count"
        )
    expected_format = "deb" if package_manager == "apt-get" else "rpm"
    expected_verification = (
        "apt-repository" if package_manager == "apt-get" else "rpm-signature"
    )
    payloads: list[PackagePayload] = []
    for index, raw in enumerate(raw_packages):
        prefix = f"package_payloads.packages[{index}]"
        if not isinstance(raw, dict) or set(raw) != {
            "architecture",
            "epoch",
            "filename",
            "format",
            "name",
            "roles",
            "sha256",
            "signer_ids",
            "size_bytes",
            "verification",
            "version",
        }:
            raise RollbackSnapshotError(
                f"rollback snapshot {prefix} has invalid fields"
            )
        name = _required_string(raw["name"], f"{prefix}.name")
        if not _interesting_package(name):
            raise RollbackSnapshotError(
                f"rollback snapshot {prefix}.name is outside the package scope"
            )
        architecture = _required_string(
            raw["architecture"], f"{prefix}.architecture"
        )
        version = _required_string(raw["version"], f"{prefix}.version")
        epoch = _optional_package_string(raw["epoch"], f"{prefix}.epoch")
        payload_format = _required_string(raw["format"], f"{prefix}.format")
        verification = _required_string(
            raw["verification"], f"{prefix}.verification"
        )
        filename = _required_string(raw["filename"], f"{prefix}.filename")
        digest = _required_string(raw["sha256"], f"{prefix}.sha256")
        if (
            payload_format != expected_format
            or verification != expected_verification
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or filename != f"{digest}.{payload_format}"
        ):
            raise RollbackSnapshotError(
                f"rollback snapshot {prefix} has invalid payload provenance"
            )
        size_bytes = _load_nonnegative_integer(
            raw["size_bytes"], f"{prefix}.size_bytes", minimum=1
        )
        roles = _load_payload_string_set(
            raw["roles"],
            f"{prefix}.roles",
            allowed={"baseline", "forward"},
            require_nonempty=True,
        )
        signer_ids = _load_payload_string_set(
            raw["signer_ids"],
            f"{prefix}.signer_ids",
            pattern=r"[0-9a-f]{8,40}",
            require_nonempty=payload_format == "rpm",
        )
        if payload_format == "deb" and signer_ids:
            raise RollbackSnapshotError(
                f"rollback snapshot {prefix}.signer_ids must be empty for DEB payloads"
            )
        payloads.append(
            PackagePayload(
                name=name,
                architecture=architecture,
                epoch=epoch,
                version=version,
                format=payload_format,
                filename=filename,
                sha256=digest,
                size_bytes=size_bytes,
                verification=verification,
                roles=tuple(roles),
                signer_ids=tuple(signer_ids),
            )
        )
    identities = [
        (payload.name, payload.architecture, payload.epoch, payload.version)
        for payload in payloads
    ]
    if len(set(identities)) != len(identities):
        raise RollbackSnapshotError(
            "rollback snapshot package payload manifest contains duplicate identities"
        )
    filenames = [payload.filename for payload in payloads]
    if len(set(filenames)) != len(filenames):
        raise RollbackSnapshotError(
            "rollback snapshot package payload manifest contains duplicate files"
        )
    baseline_identities = {
        (package.name, package.architecture, package.epoch, package.version)
        for package in baseline_packages
    }
    manifest_baseline = {
        (payload.name, payload.architecture, payload.epoch, payload.version)
        for payload in payloads
        if "baseline" in payload.roles
    }
    if manifest_baseline != baseline_identities:
        raise RollbackSnapshotError(
            "rollback snapshot package payload manifest is not one-to-one with baseline packages"
        )
    total_size_bytes = _load_nonnegative_integer(
        value["total_size_bytes"],
        "package_payloads.total_size_bytes",
    )
    if total_size_bytes != sum(payload.size_bytes for payload in payloads):
        raise RollbackSnapshotError(
            "rollback snapshot package payload total size is inconsistent"
        )
    return PackagePayloadBundle(directory, tuple(payloads), total_size_bytes)


def _load_payload_string_set(
    value: Any,
    name: str,
    *,
    allowed: set[str] | None = None,
    pattern: str | None = None,
    require_nonempty: bool,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RollbackSnapshotError(f"rollback snapshot {name} must be a string array")
    items = list(value)
    if (require_nonempty and not items) or len(items) != len(set(items)):
        raise RollbackSnapshotError(
            f"rollback snapshot {name} must contain unique values"
        )
    if allowed is not None and set(items) - allowed:
        raise RollbackSnapshotError(
            f"rollback snapshot {name} contains unsupported values"
        )
    if pattern is not None and any(re.fullmatch(pattern, item) is None for item in items):
        raise RollbackSnapshotError(
            f"rollback snapshot {name} contains invalid values"
        )
    return items


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_introduced_packages(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise RollbackSnapshotError(
            "rollback snapshot introduced_packages must be an array"
        )
    if len(value) > MAX_SNAPSHOT_PACKAGES:
        raise RollbackSnapshotError(
            f"rollback snapshot introduced_packages exceeds {MAX_SNAPSHOT_PACKAGES} entries"
        )
    packages: list[str] = []
    for index, entry in enumerate(value):
        name = _required_string(entry, f"introduced_packages[{index}]")
        if not _interesting_package(name) or not _valid_package_spec(name):
            raise RollbackSnapshotError(
                f"rollback snapshot introduced_packages[{index}] is outside the NVIDIA stack scope"
            )
        packages.append(name)
    if len(set(packages)) != len(packages):
        raise RollbackSnapshotError(
            "rollback snapshot introduced_packages contains duplicates"
        )
    return packages


def _load_module_names(value: Any, module_loaded: bool) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(module, str) and module in _ALLOWED_NVIDIA_MODULES
        for module in value
    ):
        raise RollbackSnapshotError(
            "rollback snapshot module_names must contain only supported NVIDIA modules"
        )
    if len(value) != len(set(value)):
        raise RollbackSnapshotError(
            "rollback snapshot module_names contains duplicates"
        )
    if module_loaded != ("nvidia" in value):
        raise RollbackSnapshotError(
            "rollback snapshot module_names is inconsistent with module_loaded"
        )
    return list(value)


def _load_gpu_uuids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RollbackSnapshotError(
            "rollback snapshot gpu_uuids must be a non-empty array"
        )
    if len(value) > 64:
        raise RollbackSnapshotError("rollback snapshot gpu_uuids exceeds 64 entries")
    if not all(
        isinstance(gpu_uuid, str) and re.fullmatch(r"GPU-[A-Fa-f0-9-]{16,}", gpu_uuid)
        for gpu_uuid in value
    ):
        raise RollbackSnapshotError(
            "rollback snapshot gpu_uuids contains an invalid GPU UUID"
        )
    if len(value) != len(set(value)):
        raise RollbackSnapshotError("rollback snapshot gpu_uuids contains duplicates")
    return list(value)


def _load_managed_files(value: Any, package_manager: str) -> list[FileSnapshot]:
    if not isinstance(value, list):
        raise RollbackSnapshotError("rollback snapshot managed_files must be an array")
    files: list[FileSnapshot] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or set(entry) != {
            "content_base64",
            "existed",
            "mode",
            "path",
        }:
            raise RollbackSnapshotError(
                f"rollback snapshot managed_files[{index}] has invalid fields"
            )
        path = _required_string(entry["path"], f"managed_files[{index}].path")
        if not _is_managed_path(path):
            raise RollbackSnapshotError(
                f"rollback snapshot managed_files[{index}].path is not managed"
            )
        existed = _required_boolean(entry["existed"], f"managed_files[{index}].existed")
        content = _optional_string(
            entry["content_base64"], f"managed_files[{index}].content_base64"
        )
        mode = entry["mode"]
        if existed:
            if content is None or type(mode) is not int or not 0 <= mode <= 0o777:
                raise RollbackSnapshotError(
                    f"rollback snapshot managed_files[{index}] lacks restorable content/mode"
                )
            try:
                decoded = base64.b64decode(content, validate=True)
                decoded.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as exc:
                raise RollbackSnapshotError(
                    f"rollback snapshot managed_files[{index}].content_base64 is invalid"
                ) from exc
            if len(decoded) > MAX_MANAGED_FILE_BYTES:
                raise RollbackSnapshotError(
                    f"rollback snapshot managed_files[{index}] content is too large"
                )
        elif content is not None or mode is not None:
            raise RollbackSnapshotError(
                f"rollback snapshot managed_files[{index}] records content for an absent file"
            )
        files.append(FileSnapshot(path, existed, content, mode))
    expected_paths = {str(_DOCKER_CONFIG_PATH)}
    if package_manager in {"dnf", "yum"}:
        expected_paths.add(str(_DNF_MODULE_PATH))
    elif package_manager == "zypper":
        expected_paths.add(str(_ZYPPER_LOCK_PATH))
    actual_paths = {file.path for file in files}
    if package_manager == "dnf":
        dynamic_paths = {
            path
            for path in actual_paths
            if _DNF_MODULE_FAILSAFE_PATH_PATTERN.fullmatch(path) is not None
        }
        if len(dynamic_paths) != 1:
            raise RollbackSnapshotError(
                "rollback snapshot must have one DNF module fail-safe target"
            )
        expected_paths.update(dynamic_paths)
    if actual_paths != expected_paths or len(actual_paths) != len(files):
        raise RollbackSnapshotError(
            "rollback snapshot managed_files does not match its package manager"
        )
    canonical_paths = [str(_DOCKER_CONFIG_PATH)]
    if package_manager in {"dnf", "yum"}:
        canonical_paths.append(str(_DNF_MODULE_PATH))
        canonical_paths.extend(sorted(actual_paths - set(canonical_paths)))
    elif package_manager == "zypper":
        canonical_paths.append(str(_ZYPPER_LOCK_PATH))
    if [file.path for file in files] != canonical_paths:
        raise RollbackSnapshotError(
            "rollback snapshot managed_files is not in canonical restore order"
        )
    return files


def _load_commands(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        raise RollbackSnapshotError("rollback snapshot commands must be an array")
    if len(value) > MAX_SNAPSHOT_COMMANDS:
        raise RollbackSnapshotError(
            f"rollback snapshot commands exceeds {MAX_SNAPSHOT_COMMANDS} entries"
        )
    commands: list[list[str]] = []
    for index, command in enumerate(value):
        if not isinstance(command, list) or not command:
            raise RollbackSnapshotError(
                f"rollback snapshot commands[{index}] must be a non-empty array"
            )
        if len(command) > MAX_COMMAND_PARTS:
            raise RollbackSnapshotError(
                f"rollback snapshot commands[{index}] is too long"
            )
        if not all(isinstance(part, str) and part for part in command):
            raise RollbackSnapshotError(
                f"rollback snapshot commands[{index}] entries must be non-empty strings"
            )
        if not _allowed_rollback_command(command):
            raise RollbackSnapshotError(
                f"rollback snapshot commands[{index}] is not a supported rollback command"
            )
        commands.append(command)
    return commands


def _allowed_rollback_command(command: list[str]) -> bool:
    if command[:5] == [
        "python3",
        "-I",
        "-c",
        DNF_LOCAL_TRANSACTION_SCRIPT,
        "--apply",
    ]:
        try:
            remove_marker = command.index("--remove", 5)
            install_marker = command.index(
                "--expect-install",
                remove_marker + 1,
            )
            expected_remove_marker = command.index(
                "--expect-remove",
                install_marker + 1,
            )
        except ValueError:
            return False
        restore_paths = command[5:remove_marker]
        remove_specs = command[remove_marker + 1 : install_marker]
        expected_installs = command[
            install_marker + 1 : expected_remove_marker
        ]
        expected_removals = command[expected_remove_marker + 1 :]
        return bool(
            command.count("--remove") == 1
            and command.count("--expect-install") == 1
            and command.count("--expect-remove") == 1
            and (restore_paths or remove_specs)
            and len(restore_paths) == len(expected_installs)
            and len(set(restore_paths)) == len(restore_paths)
            and len(set(remove_specs)) == len(remove_specs)
            and len(set(expected_installs)) == len(expected_installs)
            and len(set(expected_removals)) == len(expected_removals)
            and all(
                _valid_local_payload_path(path, extension="rpm")
                for path in restore_paths
            )
            and (not remove_specs or _valid_package_specs(remove_specs))
            and (not expected_installs or _valid_package_specs(expected_installs))
            and (not expected_removals or _valid_package_specs(expected_removals))
        )
    if command[:8] == [
        "apt-get",
        "install",
        "-y",
        "--allow-change-held-packages",
        "--allow-downgrades",
        "--no-download",
        "--no-install-recommends",
        "--purge",
    ]:
        operands = command[8:]
        return bool(operands) and all(
            (
                _valid_package_spec(operand[:-1])
                if operand.endswith("-")
                else _valid_local_payload_path(operand, extension="deb")
            )
            for operand in operands
        )
    if (
        len(command) >= 8
        and command[0] in {"dnf", "yum"}
        and command[1:7]
        == [
            "--disablerepo=*",
            "--disableplugin=versionlock",
            "--noautoremove",
            "--setopt=localpkg_gpgcheck=1",
            "install",
            "-y",
        ]
    ):
        return all(
            _valid_local_payload_path(path, extension="rpm")
            for path in command[7:]
        )
    if (
        len(command) >= 7
        and command[0] in {"dnf", "yum"}
        and command[1:6]
        == [
            "--disablerepo=*",
            "--disableplugin=versionlock",
            "--noautoremove",
            "remove-nevra",
            "-y",
        ]
    ):
        return _valid_package_specs(command[6:])
    if (
        len(command) >= 7
        and command[0] in {"dnf", "yum"}
        and command[1:6]
        == [
            "--disablerepo=*",
            "--disableplugin=versionlock",
            "--noautoremove",
            "remove-n",
            "-y",
        ]
    ):
        return _valid_package_specs(command[6:])
    if command[:8] == [
        "zypper",
        "--non-interactive",
        "--disable-repositories",
        "--no-refresh",
        "install",
        "--oldpackage",
        "--no-recommends",
        "--no-force-resolution",
    ]:
        if len(command) < 10 or command[8] != "--":
            return False
        operands = command[9:]
        return bool(operands) and all(
            _valid_package_spec(operand[1:])
            if operand.startswith("-")
            else _valid_local_payload_path(operand, extension="rpm")
            for operand in operands
        )
    return False


def _valid_local_payload_path(value: str, *, extension: str) -> bool:
    path = Path(value)
    return bool(
        path.is_absolute()
        and path.anchor == os.sep
        and os.path.normpath(value) == value
        and all(part not in {"", ".", ".."} for part in path.parts[1:])
        and path.parent.name.endswith(".payloads")
        and re.fullmatch(rf"[a-f0-9]{{64}}\.{extension}", path.name)
    )


def _valid_package_specs(specs: list[str]) -> bool:
    if not specs:
        return False
    return all(_valid_package_spec(spec) for spec in specs)


def _valid_package_spec(spec: str) -> bool:
    if spec.startswith("-"):
        return False
    return (
        re.match(
            r"^[A-Za-z0-9][A-Za-z0-9.+_:-]*(?:=[A-Za-z0-9][A-Za-z0-9.+:~_-]*)?$", spec
        )
        is not None
    )


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RollbackSnapshotError(
            f"rollback snapshot {name} must be a non-empty string"
        )
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RollbackSnapshotError(
            f"rollback snapshot {name} must be null or a non-empty string"
        )
    return value


def _required_unit_file_state(value: Any, name: str) -> str:
    state = _required_string(value, name)
    if state not in _SNAPSHOT_UNIT_FILE_STATES:
        raise RollbackSnapshotError(
            f"rollback snapshot {name} is not a supported exact systemd state"
        )
    return state


def _optional_package_string(value: Any, name: str) -> str | None:
    if value == "":
        return None
    return _optional_string(value, name)


def _required_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RollbackSnapshotError(f"rollback snapshot {name} must be a boolean")
    return value


def _optional_boolean(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    return _required_boolean(value, name)


def apply_rollback(
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    *,
    current_audit: HostAudit | None = None,
    restore_service_enablement: bool = False,
    restore_service_activity: bool = False,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    applying = bool(getattr(runner, "apply", False))
    if applying and current_audit is None:
        return [
            CommandResult(
                ["rollback-precondition", "current-audit"],
                1,
                stderr="an applied rollback requires a current host audit",
            )
        ]
    if applying and current_audit is not None:
        if snapshot.path is None:
            return [
                CommandResult(
                    ["rollback-precondition", "snapshot-authority"],
                    1,
                    stderr=(
                        "an applied rollback requires a trusted snapshot loaded "
                        "from its exact private source path"
                    ),
                )
            ]
        try:
            validate_snapshot_for_apply(
                snapshot,
                snapshot.path,
                current_audit,
                runner=runner,
            )
        except RollbackSnapshotError as exc:
            return [
                CommandResult(
                    ["rollback-precondition", "snapshot-authority"],
                    1,
                    stderr=str(exc),
                )
            ]
        from .preflight import PackagePreflightError, preflight_package_rollback

        solver_command = ["rollback-precondition", "package-solver"]
        _record_external_start(runner, solver_command, mutate=False)
        try:
            solver_results = preflight_package_rollback(
                snapshot,
                current_audit,
                runner,
            )
        except PackagePreflightError as exc:
            solver_failure = CommandResult(
                solver_command,
                1,
                stderr=str(exc),
            )
            _record_external_result(runner, solver_failure, mutate=False)
            return [*exc.results, solver_failure]
        solver_success = CommandResult(solver_command, 0)
        _record_external_result(runner, solver_success, mutate=False)
        results.extend([*solver_results, solver_success])
    if applying and (restore_service_enablement or restore_service_activity):
        return [
            CommandResult(
                ["rollback-precondition", "staged-service-restore"],
                1,
                stderr=(
                    "applied rollback service state must be restored with the "
                    "single-unit prepare, activity, and final enablement APIs "
                    "so every stage can be re-audited"
                ),
            )
        ]
    if (
        applying
        and current_audit is not None
        and current_audit.gpu_uuids != snapshot.gpu_uuids
    ):
        return [
            CommandResult(
                ["rollback-precondition", "gpu-inventory"],
                1,
                stderr=(
                    "current GPU UUID inventory does not exactly match the "
                    "rollback snapshot"
                ),
            )
        ]
    commands = rollback_package_commands(snapshot, current_audit)
    files_changed = any(
        not _managed_file_matches(file) for file in snapshot.managed_files
    )
    if (
        applying
        and current_audit is not None
        and not current_audit.mig_geometry_complete
    ):
        return [
            CommandResult(
                ["rollback-precondition", "mig-geometry"],
                1,
                stderr=(
                    "an applied rollback requires a complete current MIG GI/CI "
                    "geometry observation"
                ),
            )
        ]
    if (
        applying
        and current_audit is not None
        and current_audit.mig_mode == "disabled"
        and (current_audit.mig_geometry or current_audit.mig_device_uuids)
    ):
        return [
            CommandResult(
                ["rollback-precondition", "mig-state-consistency"],
                1,
                stderr="current MIG mode is disabled but MIG instances remain visible",
            )
        ]
    mig_mode_changed = bool(
        snapshot.mig_mode in {"enabled", "disabled"}
        and (current_audit is None or current_audit.mig_mode != snapshot.mig_mode)
    )
    mig_geometry_changed = bool(
        snapshot.mig_mode in {"enabled", "disabled"}
        and (
            current_audit is None or current_audit.mig_geometry != snapshot.mig_geometry
        )
    )
    mig_changed = mig_mode_changed or mig_geometry_changed
    snapshot_module_names = list(snapshot.module_names)
    if snapshot.module_loaded and not snapshot_module_names:
        # Directly constructed snapshots from callers predating schema 2.1 are
        # treated as base-module-only; persisted snapshots must carry the field.
        snapshot_module_names = ["nvidia"]
    unload_order = [
        "nvidia_peermem",
        "nvidia_fs",
        "nvidia_uvm",
        "nvidia_drm",
        "nvidia_modeset",
        "nvidia",
    ]
    current_module_names = (
        ["nvidia"] if current_audit is not None and current_audit.module.loaded else []
    )
    if applying:
        inspect_command = ["inspect-module-dependencies"]
        _record_external_start(runner, inspect_command, mutate=False)
        try:
            unload_order = nvidia_module_unload_order()
            current_module_names = list(unload_order)
            if current_audit and current_audit.module.loaded != (
                "nvidia" in current_module_names
            ):
                raise ModuleDependencyError(
                    "loaded NVIDIA module audit does not match the current sysfs graph"
                )
        except ModuleDependencyError as exc:
            inspect_result = CommandResult(
                inspect_command,
                1,
                stderr=str(exc),
            )
            _record_external_result(runner, inspect_result, mutate=False)
            return [inspect_result]
        inspect_result = CommandResult(inspect_command, 0)
        _record_external_result(runner, inspect_result, mutate=False)
        file_precondition_command = ["rollback-precondition", "managed-files"]
        _record_external_start(runner, file_precondition_command, mutate=False)
        file_precondition = _managed_file_restore_precondition(snapshot.managed_files)
        if file_precondition is not None:
            _record_external_result(runner, file_precondition, mutate=False)
            return [file_precondition]
        _record_external_result(
            runner,
            CommandResult(file_precondition_command, 0),
            mutate=False,
        )
    module_matches = bool(
        current_audit is not None
        and current_audit.module.loaded == snapshot.module_loaded
        and current_audit.module.version == snapshot.module_version
        and current_audit.module.open_module == snapshot.module_open_module
        and current_audit.module.signed == snapshot.module_signed
        and current_audit.module.installed_version == snapshot.module_installed_version
        and current_audit.module.installed_open_module
        == snapshot.module_installed_open_module
        and current_audit.module.installed_signed == snapshot.module_installed_signed
        and current_module_names == snapshot_module_names
    )
    module_unload_needed = bool(current_module_names and not module_matches)
    module_change = not module_matches
    module_reset_expected = bool(
        module_unload_needed or (snapshot.module_loaded and not module_matches)
    )
    mig_restore_required = bool(
        mig_changed or (module_reset_expected and snapshot.mig_mode == "enabled")
    )
    if mig_restore_required and len(snapshot.gpu_uuids) != 1:
        return [
            CommandResult(
                ["rollback-precondition", "mig-transaction-scope"],
                1,
                stderr=(
                    "MIG rollback is qualified only for one UUID-bound GPU; "
                    "refusing a potentially partial multi-GPU transition"
                ),
            )
        ]
    if mig_restore_required and not restorable_mig_geometry(
        snapshot.mig_geometry,
        snapshot.gpu_uuids,
    ):
        return [
            CommandResult(
                ["rollback-precondition", "mig-rollback-geometry"],
                1,
                stderr=(
                    "the snapshot MIG layout cannot be recreated exactly; only "
                    "empty geometry or one GI containing one full CI is qualified"
                ),
            )
        ]
    current_service_active = {
        _DOCKER_SOCKET_UNIT: (
            current_audit.docker_socket_active if current_audit else None
        ),
        _DOCKER_SERVICE_UNIT: (
            current_audit.docker_service_active if current_audit else None
        ),
        _NVIDIA_PERSISTENCED_UNIT: (
            current_audit.nvidia_persistenced_active if current_audit else None
        ),
        _FABRIC_MANAGER_UNIT: (
            current_audit.fabric_manager_active if current_audit else None
        ),
    }
    initial_service_enabled = {
        _DOCKER_SOCKET_UNIT: (
            current_audit.docker_socket_enabled if current_audit else None
        ),
        _DOCKER_SERVICE_UNIT: (
            current_audit.docker_service_enabled if current_audit else None
        ),
        _NVIDIA_PERSISTENCED_UNIT: (
            current_audit.nvidia_persistenced_enabled if current_audit else None
        ),
        _FABRIC_MANAGER_UNIT: (
            current_audit.fabric_manager_enabled if current_audit else None
        ),
    }
    current_service_enabled = dict(initial_service_enabled)
    current_service_unit_file_state = {
        _DOCKER_SOCKET_UNIT: (
            current_audit.docker_socket_unit_file_state if current_audit else None
        ),
        _DOCKER_SERVICE_UNIT: (
            current_audit.docker_service_unit_file_state if current_audit else None
        ),
        _NVIDIA_PERSISTENCED_UNIT: (
            current_audit.nvidia_persistenced_unit_file_state if current_audit else None
        ),
        _FABRIC_MANAGER_UNIT: (
            current_audit.fabric_manager_unit_file_state if current_audit else None
        ),
    }
    target_service_active = {
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_active,
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_active,
        _NVIDIA_PERSISTENCED_UNIT: snapshot.nvidia_persistenced_active,
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_active,
    }
    target_service_enabled = {
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_enabled,
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_enabled,
        _NVIDIA_PERSISTENCED_UNIT: snapshot.nvidia_persistenced_enabled,
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_enabled,
    }
    target_service_unit_file_state = {
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_unit_file_state,
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (snapshot.nvidia_persistenced_unit_file_state),
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_unit_file_state,
    }
    service_state_changed = any(
        all(
            state is not None
            for state in (
                current_service_active[service],
                current_service_enabled[service],
                current_service_unit_file_state[service],
                target_service_active[service],
                target_service_enabled[service],
                target_service_unit_file_state[service],
            )
        )
        and (
            current_service_active[service],
            current_service_enabled[service],
            current_service_unit_file_state[service],
        )
        != (
            target_service_active[service],
            target_service_enabled[service],
            target_service_unit_file_state[service],
        )
        for service in _SERVICE_STOP_ORDER
    )
    transaction_mutates_host = bool(
        commands
        or module_change
        or mig_restore_required
        or files_changed
        or service_state_changed
    )
    # Any package/module/MIG/config mutation can indirectly activate a launcher
    # through package scripts or socket activation.  Persistently mask every
    # transactional unit before the first mutation; staged commit restores the
    # exact baseline only after core rollback verification.
    quiesce = {service: transaction_mutates_host for service in _SERVICE_STOP_ORDER}
    if applying and transaction_mutates_host and current_audit is not None:
        current_service_error = _service_state_consistency_error(current_audit)
        if current_service_error is not None:
            return [
                *results,
                CommandResult(
                    ["rollback-precondition", "current-service-state"],
                    1,
                    stderr=(
                        "current transactional service state is inconsistent: "
                        + current_service_error
                    ),
                ),
            ]
        snapshot_service_error = _snapshot_service_state_consistency_error(snapshot)
        if snapshot_service_error is not None:
            return [
                *results,
                CommandResult(
                    ["rollback-precondition", "snapshot-service-state"],
                    1,
                    stderr=(
                        "snapshot transactional service state is inconsistent: "
                        + snapshot_service_error
                    ),
                ),
            ]
    for service in _SERVICE_STOP_ORDER:
        unknown_states = [
            name
            for name, state in (
                ("current active", current_service_active[service]),
                ("current enabled", initial_service_enabled[service]),
                (
                    "current unit-file",
                    current_service_unit_file_state[service],
                ),
                ("snapshot active", target_service_active[service]),
                ("snapshot enabled", target_service_enabled[service]),
                (
                    "snapshot unit-file",
                    target_service_unit_file_state[service],
                ),
            )
            if state is None
        ]
        if quiesce[service] and applying and unknown_states:
            return [
                *results,
                CommandResult(
                    ["rollback-precondition", "service-state", service],
                    1,
                    stderr=(
                        f"cannot safely quiesce {service}: "
                        + ", ".join(unknown_states)
                        + " state is unknown"
                    ),
                ),
            ]
    for service in _SERVICE_STOP_ORDER:
        if not quiesce[service]:
            continue
        if not applying:
            if current_service_unit_file_state[service] == "enabled":
                results.append(
                    runner.run(
                        ["systemctl", "disable", "--now", service],
                        mutate=True,
                        allow_fail=True,
                    )
                )
            elif current_service_unit_file_state[service] in {
                "disabled",
                "static",
            }:
                results.append(
                    runner.run(
                        ["systemctl", "stop", service],
                        mutate=True,
                        allow_fail=True,
                    )
                )
            if results and results[-1].returncode not in (0, None):
                return results
            results.append(
                runner.run(
                    ["systemctl", "mask", "--now", service],
                    mutate=True,
                    allow_fail=True,
                )
            )
            if results[-1].returncode not in (0, None):
                return results
            current_service_active[service] = False
            current_service_enabled[service] = False
            current_service_unit_file_state[service] = "masked"
            continue
        quarantine_results = _quarantine_service_for_rollback(runner, service)
        results.extend(quarantine_results)
        if any(
            result.returncode not in (0, None)
            for result in quarantine_results
        ):
            return results
        current_service_active[service] = False
        current_service_enabled[service] = False
        current_service_unit_file_state[service] = "masked"

    if applying and commands:
        payload_validation_command = [
            "rollback-precondition",
            "package-payloads",
        ]
        _record_external_start(
            runner,
            payload_validation_command,
            mutate=False,
        )
        try:
            if snapshot.path is None or snapshot.package_payloads is None:
                raise PackagePayloadError(
                    "applicable rollback snapshot has no retained package payloads"
                )
            validate_package_payloads(
                Path(snapshot.path),
                snapshot.package_payloads,
                snapshot.packages,
                snapshot.package_manager or "",
                runner=runner,
                required_owner_uid=os.geteuid(),
            )
            payload_validation_result = CommandResult(
                payload_validation_command,
                0,
            )
        except PackagePayloadError as exc:
            payload_validation_result = CommandResult(
                payload_validation_command,
                1,
                stderr=f"retained rollback package payload validation failed: {exc}",
            )
        _record_external_result(
            runner,
            payload_validation_result,
            mutate=False,
        )
        results.append(payload_validation_result)
        if payload_validation_result.returncode != 0:
            return results

    for command in commands:
        result = runner.run(command, mutate=True, allow_fail=True)
        results.append(result)
        if result.returncode not in (0, None):
            return results

    # Package maintainer scripts can rewrite policy/configuration files. Restore
    # the captured bytes only after every package transaction has completed.
    file_results = _restore_managed_files(
        snapshot.managed_files,
        runner,
    )
    results.extend(file_results)
    if any(result.returncode not in (0, None) for result in file_results):
        return results

    if applying:
        # Package scripts may start a unit even when it was quiesced above. Use
        # one strict, machine-readable observation and stop it again before
        # touching the kernel-module graph or MIG state.
        for service in _SERVICE_STOP_ORDER:
            if not quiesce[service]:
                continue
            quarantine_results = _quarantine_service_for_rollback(
                runner,
                service,
            )
            results.extend(quarantine_results)
            if any(
                result.returncode not in (0, None)
                for result in quarantine_results
            ):
                return results
            current_service_active[service] = False
            current_service_enabled[service] = False
            current_service_unit_file_state[service] = "masked"

        inspect_command = ["inspect-module-dependencies"]
        _record_external_start(runner, inspect_command, mutate=False)
        try:
            current_module_names = nvidia_module_unload_order()
        except ModuleDependencyError as exc:
            inspect_result = CommandResult(inspect_command, 1, stderr=str(exc))
            _record_external_result(runner, inspect_result, mutate=False)
            results.append(inspect_result)
            return results
        inspect_result = CommandResult(inspect_command, 0)
        _record_external_result(runner, inspect_result, mutate=False)
        results.append(inspect_result)
        unload_order = list(current_module_names)
        module_unload_needed = bool(current_module_names and module_change)
        module_reset_expected = bool(
            module_unload_needed or (snapshot.module_loaded and not module_matches)
        )
        mig_restore_required = bool(
            mig_changed or (module_reset_expected and snapshot.mig_mode == "enabled")
        )

    if module_unload_needed:
        unload_result = runner.run(
            ["modprobe", "-r", *unload_order],
            mutate=True,
            allow_fail=True,
        )
        results.append(unload_result)
        if unload_result.returncode not in (0, None):
            return results
    if snapshot.module_loaded and not module_matches:
        for module in reversed(snapshot_module_names):
            load_result = runner.run(["modprobe", module], mutate=True, allow_fail=True)
            results.append(load_result)
            if load_result.returncode not in (0, None):
                return results

    current_mode = current_audit.mig_mode if current_audit else None
    current_geometry = current_audit.mig_geometry if current_audit else []
    mig_state_may_have_changed = bool(
        commands
        or module_unload_needed
        or (snapshot.module_loaded and not module_matches)
    )
    if applying and (mig_restore_required or mig_state_may_have_changed):
        status, current_mode, current_geometry = _refresh_rollback_mig_state(
            snapshot,
            runner,
            results,
            stage="post-package-module",
        )
        if status != "stable":
            return results
        mig_restore_required = bool(
            current_mode != snapshot.mig_mode
            or current_geometry != snapshot.mig_geometry
        )

    if mig_restore_required:
        assert snapshot.mig_mode is not None
        gpu_uuid = snapshot.gpu_uuids[0]

        # A successful module graph replacement destroys MIG devices.  Without
        # a module replacement, explicitly destroy the current children before
        # changing mode or recreating snapshot geometry.
        if (
            not module_reset_expected
            and current_mode == "enabled"
            and current_geometry
            and (
                snapshot.mig_mode == "disabled"
                or current_geometry != snapshot.mig_geometry
            )
        ):
            for command in mig_geometry_destroy_commands(gpu_uuid):
                result = runner.run(
                    command,
                    mutate=True,
                    allow_fail=True,
                )
                results.append(result)
                if result.returncode not in (0, None):
                    return results
                if applying:
                    status, current_mode, current_geometry = (
                        _refresh_rollback_mig_state(
                            snapshot,
                            runner,
                            results,
                            stage="post-geometry-destroy",
                        )
                    )
                    if status != "stable":
                        return results

        if module_reset_expected or current_mode != snapshot.mig_mode:
            mig_value = "1" if snapshot.mig_mode == "enabled" else "0"
            mig_result = runner.run(
                ["nvidia-smi", "-i", gpu_uuid, "-mig", mig_value],
                mutate=True,
                allow_fail=True,
            )
            results.append(mig_result)
            if mig_result.returncode not in (0, None):
                return results
            if applying:
                status, current_mode, current_geometry = _refresh_rollback_mig_state(
                    snapshot,
                    runner,
                    results,
                    stage="post-mode-change",
                )
                if status != "stable":
                    return results

        if snapshot.mig_mode == "enabled" and (
            current_geometry != snapshot.mig_geometry
        ):
            create_command = mig_geometry_create_command(
                gpu_uuid,
                snapshot.mig_geometry,
            )
            if create_command is not None:
                create_result = runner.run(
                    create_command,
                    mutate=True,
                    allow_fail=True,
                )
                results.append(create_result)
                if create_result.returncode not in (0, None):
                    return results
                if applying:
                    status, current_mode, current_geometry = (
                        _refresh_rollback_mig_state(
                            snapshot,
                            runner,
                            results,
                            stage="post-geometry-create",
                        )
                    )
                    if status != "stable":
                        return results
                    if current_geometry != snapshot.mig_geometry:
                        failure = CommandResult(
                            [
                                "rollback-precondition",
                                "mig-geometry-mismatch",
                            ],
                            1,
                            stderr=(
                                "fresh MIG geometry does not exactly match the "
                                "rollback snapshot after creation"
                            ),
                        )
                        _record_external_start(
                            runner,
                            failure.command,
                            mutate=False,
                        )
                        _record_external_result(
                            runner,
                            failure,
                            mutate=False,
                        )
                        results.append(failure)
                        return results

    if restore_service_enablement or restore_service_activity:
        for service in _SERVICE_ACTIVITY_RESTORE_ORDER:
            service_results = _prepare_service_for_activity(
                runner,
                service,
                target_state=target_service_unit_file_state[service],
                current_state=current_service_unit_file_state[service],
            )
            results.extend(service_results)
            if any(result.returncode not in (0, None) for result in service_results):
                return results
            if service_results:
                current_service_unit_file_state[service] = _prepared_unit_file_state(
                    target_service_unit_file_state[service]  # type: ignore[arg-type]
                )
                current_service_enabled[service] = False

    if restore_service_activity:
        for service in _SERVICE_ACTIVITY_RESTORE_ORDER:
            target_state = target_service_unit_file_state[service]
            prepared_state = (
                _prepared_unit_file_state(target_state)
                if target_state in _SNAPSHOT_UNIT_FILE_STATES
                else None
            )
            if current_service_unit_file_state[service] != prepared_state:
                results.append(
                    CommandResult(
                        [
                            "rollback-precondition",
                            "service-unit-file-state",
                            service,
                        ],
                        1,
                        stderr=(
                            f"cannot restore {service} activity until its "
                            "unit-file state has been safely prepared"
                        ),
                    )
                )
                return results
            target_active = target_service_active[service]
            service_results = _restore_service_activity(
                runner,
                service,
                active=target_active,
                current_active=current_service_active[service],
            )
            results.extend(service_results)
            if any(result.returncode not in (0, None) for result in service_results):
                return results
            if service_results:
                current_service_active[service] = target_active

    if restore_service_enablement:
        for service in _SERVICE_ACTIVITY_RESTORE_ORDER:
            service_results = _restore_service_unit_file_state(
                runner,
                service,
                target_state=target_service_unit_file_state[service],
                current_state=current_service_unit_file_state[service],
            )
            results.extend(service_results)
            if any(result.returncode not in (0, None) for result in service_results):
                return results
            if service_results:
                current_service_unit_file_state[service] = (
                    target_service_unit_file_state[service]
                )
                current_service_enabled[service] = target_service_enabled[service]

    if not (restore_service_enablement and restore_service_activity) and any(
        target_service_active[service] != current_service_active[service]
        or target_service_unit_file_state[service]
        != current_service_unit_file_state[service]
        for service in _SERVICE_ACTIVITY_RESTORE_ORDER
    ):
        results.append(
            CommandResult(
                ["rollback", "service-state"],
                0,
                skipped=True,
                reason="deferred",
            )
        )
    if not results:
        results.append(
            CommandResult(
                ["rollback"],
                0,
                skipped=True,
                reason="already-restored",
            )
        )
    return results


def rollback_package_commands(
    snapshot: RollbackSnapshot,
    current_audit: HostAudit | None,
) -> list[list[str]]:
    """Derive the exact package delta that rollback will apply."""
    restore_packages = snapshot.packages
    remove_packages = snapshot.introduced_packages
    expected_rpm_installs: list[str] | None = None
    expected_rpm_removals: list[str] | None = None
    if current_audit is not None:
        current_states = {
            _package_state_identity(package)
            for package in current_audit.packages
            if package.installed
        }
        restore_packages = [
            package
            for package in snapshot.packages
            if package.installed
            and package.version
            and _package_state_identity(package) not in current_states
        ]
        baseline_states = {
            _package_state_identity(package)
            for package in snapshot.packages
            if package.installed
        }
        baseline_slots = {
            (package.name, package.architecture)
            for package in snapshot.packages
            if package.installed
        }
        current_baseline_slots = {
            (package.name, package.architecture)
            for package in current_audit.packages
            if package.installed and _package_state_identity(package) in baseline_states
        }
        added_packages = [
            package
            for package in current_audit.packages
            if package.installed
            and _package_state_identity(package) not in baseline_states
            and (
                (package.name, package.architecture) not in baseline_slots
                or (package.name, package.architecture) in current_baseline_slots
            )
        ]
        remove_packages = _package_removal_specs(
            added_packages, snapshot.package_manager
        )
        if snapshot.package_manager in {"dnf", "yum"}:
            restored_rpm_packages = [
                package
                for package in restore_packages
                if package.installed
                and package.manager == "rpm"
                and package.version
                and _interesting_package(package.name)
            ]
            restore_slots = {
                (package.name, package.architecture)
                for package in restored_rpm_packages
            }
            expected_rpm_installs = sorted(
                _rpm_package_spec(package)
                for package in restored_rpm_packages
            )
            expected_rpm_removals = sorted(
                {
                    *remove_packages,
                    *(
                        _rpm_package_spec(package)
                        for package in current_audit.packages
                        if package.installed
                        and package.manager == "rpm"
                        and package.version
                        and (package.name, package.architecture) in restore_slots
                        and _package_state_identity(package) not in baseline_states
                    ),
                }
            )
    return _rollback_commands(
        restore_packages,
        snapshot.package_manager,
        remove_packages=remove_packages,
        exact_removals=current_audit is not None,
        snapshot_path=snapshot.path,
        package_payloads=snapshot.package_payloads,
        expected_rpm_installs=expected_rpm_installs,
        expected_rpm_removals=expected_rpm_removals,
    )


def _restore_managed_files(
    files: list[FileSnapshot], runner: CommandRunner
) -> list[CommandResult]:
    results: list[CommandResult] = []
    ordered_files = sorted(
        files,
        key=lambda item: (
            0
            if item.path == str(_DOCKER_CONFIG_PATH)
            else 1
            if item.path in {str(_DNF_MODULE_PATH), str(_ZYPPER_LOCK_PATH)}
            else 2
            if _DNF_MODULE_FAILSAFE_PATH_PATTERN.fullmatch(item.path) is not None
            else 3
        ),
    )
    for snapshot in ordered_files:
        command = ["restore-file", snapshot.path]
        _record_external_start(runner, command)
        if _managed_file_matches(snapshot):
            result = CommandResult(
                command,
                0,
                skipped=True,
                reason="already-restored",
            )
            _record_external_result(runner, result)
            results.append(result)
            continue
        if not getattr(runner, "apply", True):
            result = CommandResult(
                command,
                None,
                skipped=True,
                reason="dry-run",
            )
            _record_external_result(runner, result)
            results.append(result)
            continue
        path = Path(snapshot.path)
        try:
            if snapshot.existed:
                if snapshot.content_base64 is None or snapshot.mode is None:
                    raise ValueError(
                        f"rollback file baseline is incomplete: {snapshot.path}"
                    )
                content = base64.b64decode(
                    snapshot.content_base64,
                    validate=True,
                ).decode("utf-8")
                atomic_write_text_trusted(
                    path,
                    content,
                    mode=snapshot.mode,
                    required_owner_uid=os.geteuid(),
                )
            else:
                unlink_trusted_path(path, required_owner_uid=os.geteuid())
            result = CommandResult(command, 0)
        except (OSError, BoundedFileError, ValueError, UnicodeDecodeError) as exc:
            result = CommandResult(command, 1, stderr=str(exc))
        _record_external_result(runner, result)
        results.append(result)
    return results


def _managed_file_restore_precondition(
    files: list[FileSnapshot],
) -> CommandResult | None:
    """Validate every changed file target before rollback mutates host state."""
    command = ["rollback-precondition", "managed-files"]
    for snapshot in files:
        if not _is_managed_path(snapshot.path):
            return CommandResult(
                command,
                1,
                stderr=f"rollback file target is outside the managed path set: {snapshot.path}",
            )
        if snapshot.existed:
            if snapshot.content_base64 is None or snapshot.mode is None:
                return CommandResult(
                    command,
                    1,
                    stderr=f"rollback file baseline is incomplete: {snapshot.path}",
                )
            try:
                content = base64.b64decode(snapshot.content_base64, validate=True)
                content.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as exc:
                return CommandResult(
                    command,
                    1,
                    stderr=f"rollback file baseline is invalid for {snapshot.path}: {exc}",
                )
            if len(content) > MAX_MANAGED_FILE_BYTES or not 0 <= snapshot.mode <= 0o777:
                return CommandResult(
                    command,
                    1,
                    stderr=f"rollback file baseline exceeds safety bounds: {snapshot.path}",
                )
        elif snapshot.content_base64 is not None or snapshot.mode is not None:
            return CommandResult(
                command,
                1,
                stderr=f"absent rollback file baseline carries content or mode: {snapshot.path}",
            )
        if _managed_file_matches(snapshot):
            continue
        path = Path(snapshot.path)
        try:
            metadata = trusted_path_metadata(
                path,
                required_owner_uid=os.geteuid(),
            )
        except (OSError, BoundedFileError) as exc:
            return CommandResult(command, 1, stderr=f"cannot inspect {path}: {exc}")
        if metadata is not None and (
            not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        ):
            return CommandResult(
                command,
                1,
                stderr=f"rollback file target is not a singly linked regular file: {path}",
            )
    return None


def _record_external_start(
    runner: CommandRunner,
    command: list[str],
    *,
    mutate: bool = True,
) -> None:
    callback = getattr(runner, "record_external_start", None)
    if callable(callback):
        callback(command, mutate)


def _record_external_result(
    runner: CommandRunner,
    result: CommandResult,
    *,
    mutate: bool = True,
) -> None:
    callback = getattr(runner, "record_external_result", None)
    if callable(callback):
        callback(result, mutate)


def _observe_service_state(
    runner: CommandRunner,
    service: str,
) -> tuple[CommandResult, bool | None, bool | None, str | None]:
    result = runner.run(
        [
            "systemctl",
            "show",
            "--no-pager",
            "--property=Id",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=UnitFileState",
            service,
        ],
        mutate=False,
        allow_fail=True,
    )
    active, enabled, unit_file_state = _parse_service_state(result, service)
    if active is None:
        masked_state = _parse_masked_unit_file_state(result, service)
        if masked_state is not None:
            active, enabled, unit_file_state = masked_state
    return result, active, enabled, unit_file_state


def _parse_masked_unit_file_state(
    result: CommandResult,
    service: str,
) -> tuple[bool, bool, str] | None:
    """Parse the exact transient created by `systemctl mask` without --now."""

    if (
        result.returncode != 0
        or _OUTPUT_TRUNCATED in result.stdout
        or _OUTPUT_TRUNCATED in result.stderr
    ):
        return None
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in properties:
            return None
        properties[key] = value
    if set(properties) != {
        "Id",
        "LoadState",
        "ActiveState",
        "UnitFileState",
    }:
        return None
    if (
        properties["Id"] != service
        or properties["LoadState"] not in {"loaded", "masked"}
        or properties["UnitFileState"] != "masked"
    ):
        return None
    if properties["ActiveState"] == "active":
        return True, False, "masked"
    if properties["ActiveState"] in {"inactive", "failed"}:
        return False, False, "masked"
    return None


def _quarantine_service_for_rollback(
    runner: CommandRunner,
    service: str,
) -> list[CommandResult]:
    """Mask without hooks, then stop only an authenticated active unit."""

    results: list[CommandResult] = []
    before, active_before, _, state_before = _observe_service_state(
        runner,
        service,
    )
    results.append(before)
    if (
        before.returncode != 0
        or active_before is None
        or state_before is None
    ):
        failure = CommandResult(
            ["rollback-precondition", "service-quarantine", service],
            1,
            stderr="launcher state is incomplete before quarantine",
        )
        _record_external_start(runner, failure.command, mutate=False)
        _record_external_result(runner, failure, mutate=False)
        results.append(failure)
        return results

    service_identity: TrustedGpuServiceIdentity | None = None
    socket_identity: TrustedDockerSocketIdentity | None = None
    if service == _DOCKER_SOCKET_UNIT:
        trust_results, socket_identity, error = (
            validate_trusted_docker_socket_unit_identity(
                runner,
                allow_masked=True,
            )
        )
    elif active_before:
        trust_results, service_identity, error = (
            validate_active_trusted_gpu_service_identity(runner, service)
        )
    else:
        trust_results, error = validate_trusted_gpu_service_unit(
            runner,
            service,
            allow_masked=True,
        )
    results.extend(trust_results)
    if error is not None:
        return results

    mask_result = runner.run(
        ["systemctl", "mask", service],
        mutate=True,
        allow_fail=True,
    )
    results.append(mask_result)
    if mask_result.returncode not in (0, None):
        return results

    observation, active, _, unit_file_state = _observe_service_state(
        runner,
        service,
    )
    results.append(observation)
    if (
        observation.returncode != 0
        or active is None
        or unit_file_state is None
        or unit_file_state != "masked"
    ):
        failure = CommandResult(
            ["rollback-precondition", "service-quarantine", service],
            1,
            stderr=(
                "fresh systemd state does not prove the launcher was masked "
                "before its activity transition"
            ),
        )
        _record_external_start(runner, failure.command, mutate=False)
        _record_external_result(runner, failure, mutate=False)
        results.append(failure)
        return results

    if active:
        if service == _DOCKER_SOCKET_UNIT and socket_identity is not None:
            revalidation, error = revalidate_trusted_docker_socket_identity(
                runner,
                socket_identity,
            )
        elif service != _DOCKER_SOCKET_UNIT and service_identity is not None:
            revalidation, error = (
                revalidate_trusted_gpu_service_process_identity(
                    runner,
                    service_identity,
                )
            )
        else:
            revalidation = []
            error = (
                f"{service} became active without a pre-mask trusted identity"
            )
        results.extend(revalidation)
        if error is not None:
            if not revalidation:
                failure = CommandResult(
                    [
                        "rollback-precondition",
                        "service-process-identity",
                        service,
                    ],
                    1,
                    stderr=error,
                )
                _record_external_start(runner, failure.command, mutate=False)
                _record_external_result(runner, failure, mutate=False)
                results.append(failure)
            return results
        stop_result = runner.run(
            ["systemctl", "stop", service],
            mutate=True,
            allow_fail=True,
        )
        results.append(stop_result)
        if stop_result.returncode not in (0, None):
            return results
        observation, active, _, unit_file_state = _observe_service_state(
            runner,
            service,
        )
        results.append(observation)

    if (
        observation.returncode != 0
        or active is not False
        or unit_file_state != "masked"
    ):
        failure = CommandResult(
            ["rollback-precondition", "service-quarantine", service],
            1,
            stderr="fresh systemd state does not prove inactive/masked quarantine",
        )
        _record_external_start(runner, failure.command, mutate=False)
        _record_external_result(runner, failure, mutate=False)
        results.append(failure)
    return results


def _observe_rollback_mig_state(
    runner: CommandRunner,
    expected_gpu_uuids: list[str],
) -> tuple[CommandResult, str | None, str | None]:
    result = runner.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,mig.mode.current,mig.mode.pending",
            "--format=csv,noheader,nounits",
        ],
        mutate=False,
        allow_fail=True,
    )
    if (
        result.returncode != 0
        or _OUTPUT_TRUNCATED in result.stdout
        or _OUTPUT_TRUNCATED in result.stderr
    ):
        return result, None, None
    rows = [
        [part.strip() for part in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    valid_modes = {"enabled", "disabled", "n/a"}
    if (
        not rows
        or len(rows) != len(expected_gpu_uuids)
        or any(
            len(row) != 3
            or row[0] != expected_gpu_uuids[index]
            or row[1].lower() not in valid_modes
            or row[2].lower() not in valid_modes
            for index, row in enumerate(rows)
        )
    ):
        return result, None, None
    current = _aggregate_mig_modes([row[1].lower() for row in rows])
    pending = _aggregate_mig_modes([row[2].lower() for row in rows])
    return result, current, pending


def _fresh_rollback_mig_observation(
    runner: CommandRunner,
    gpu_uuids: list[str],
) -> tuple[
    list[CommandResult],
    str | None,
    str | None,
    list[MigGpuInstance],
    bool,
]:
    state_result, current, pending = _observe_rollback_mig_state(
        runner,
        gpu_uuids,
    )
    results = [state_result]
    if current is None or pending is None:
        return results, None, None, [], False
    geometry, _, complete, geometry_results = _audit_mig_geometry(
        runner,
        current,
        pending,
        gpu_uuids,
    )
    results.extend(geometry_results)
    return results, current, pending, geometry, complete


def _refresh_rollback_mig_state(
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    results: list[CommandResult],
    *,
    stage: str,
) -> tuple[str, str | None, list[MigGpuInstance]]:
    observation, current, pending, geometry, complete = _fresh_rollback_mig_observation(
        runner, snapshot.gpu_uuids
    )
    results.extend(observation)
    if current is None or pending is None:
        failure = CommandResult(
            ["rollback-precondition", "mig-observation", stage],
            1,
            stderr="fresh UUID-bound MIG mode observation is incomplete",
        )
        _record_external_start(runner, failure.command, mutate=False)
        _record_external_result(runner, failure, mutate=False)
        results.append(failure)
        return "failed", None, []
    if current != pending:
        if pending == snapshot.mig_mode:
            marker = CommandResult(
                ["rollback", "mig-reboot-pending"],
                0,
                skipped=True,
                reason="reboot-required",
                stderr=(
                    f"MIG target {pending} is pending while current mode is "
                    f"{current}; geometry restoration is deferred until reboot"
                ),
            )
            _record_external_start(runner, marker.command, mutate=False)
            _record_external_result(runner, marker, mutate=False)
            results.append(marker)
            return "pending-reboot", current, []
        failure = CommandResult(
            ["rollback-precondition", "mig-pending", stage],
            1,
            stderr=(
                f"fresh MIG state is current={current}, pending={pending}, "
                f"not target={snapshot.mig_mode}"
            ),
        )
        _record_external_start(runner, failure.command, mutate=False)
        _record_external_result(runner, failure, mutate=False)
        results.append(failure)
        return "failed", current, []
    if not complete:
        failure = CommandResult(
            ["rollback-precondition", "mig-geometry-observation", stage],
            1,
            stderr="fresh MIG geometry observation is incomplete",
        )
        _record_external_start(runner, failure.command, mutate=False)
        _record_external_result(runner, failure, mutate=False)
        results.append(failure)
        return "failed", current, []
    return "stable", current, geometry


def _restore_service_unit_file_state(
    runner: CommandRunner,
    service: str,
    *,
    target_state: str | None,
    current_state: str | None,
) -> list[CommandResult]:
    if target_state == current_state:
        return []
    if target_state not in _SNAPSHOT_UNIT_FILE_STATES:
        return [
            CommandResult(
                ["rollback-precondition", "service-unit-file-state", service],
                1,
                stderr=(
                    f"cannot restore {service}: snapshot unit-file state "
                    f"{target_state!r} is unknown or unsupported"
                ),
            )
        ]
    if current_state not in _SNAPSHOT_UNIT_FILE_STATES:
        return [
            CommandResult(
                ["rollback-precondition", "service-unit-file-state", service],
                1,
                stderr=(
                    f"cannot restore {service}: current unit-file state "
                    f"{current_state!r} is unknown or unsupported"
                ),
            )
        ]
    prepared_state = _prepared_unit_file_state(target_state)
    if current_state != prepared_state:
        return [
            CommandResult(
                ["rollback-precondition", "service-unit-file-state", service],
                1,
                stderr=(
                    f"cannot finalize {service} from {current_state!r}; "
                    f"expected prepared state {prepared_state!r}"
                ),
            )
        ]
    if target_state != "enabled":
        # disabled/static/masked/not-found are already exact after preparation.
        return []
    results, launcher_trusted = _validate_launcher_mutation(
        runner,
        service,
        allow_masked=False,
    )
    if not launcher_trusted:
        return results
    results.append(
        runner.run(
            ["systemctl", "enable", service],
            mutate=True,
            allow_fail=True,
        )
    )
    return results


def _prepared_unit_file_state(target_state: str) -> str:
    return "disabled" if target_state == "enabled" else target_state


def _prepare_service_for_activity(
    runner: CommandRunner,
    service: str,
    *,
    target_state: str | None,
    current_state: str | None,
) -> list[CommandResult]:
    if target_state not in _SNAPSHOT_UNIT_FILE_STATES:
        return [
            CommandResult(
                ["rollback-precondition", "service-unit-file-state", service],
                1,
                stderr=(
                    f"cannot prepare {service}: snapshot unit-file state "
                    f"{target_state!r} is unknown or unsupported"
                ),
            )
        ]
    if current_state not in _SNAPSHOT_UNIT_FILE_STATES:
        return [
            CommandResult(
                ["rollback-precondition", "service-unit-file-state", service],
                1,
                stderr=(
                    f"cannot prepare {service}: current unit-file state "
                    f"{current_state!r} is unknown or unsupported"
                ),
            )
        ]
    prepared_state = _prepared_unit_file_state(target_state)
    if current_state == prepared_state:
        return []
    if current_state != "masked":
        return [
            CommandResult(
                ["rollback-precondition", "service-unit-file-state", service],
                1,
                stderr=(
                    f"cannot prepare {service} from {current_state!r}; the "
                    "unit must remain transaction-masked"
                ),
            )
        ]
    if target_state == "masked":
        return []

    results, launcher_trusted = _validate_launcher_mutation(
        runner,
        service,
        allow_masked=True,
    )
    if not launcher_trusted:
        return results
    results.append(
        runner.run(
            ["systemctl", "unmask", service],
            mutate=True,
            allow_fail=True,
        )
    )
    if results[-1].returncode not in (0, None):
        return results
    if target_state in {"enabled", "disabled"}:
        # Keep the launcher boot-disabled until its activity is validated.  An
        # enabled baseline is restored only as the final commit step.
        trust_results, launcher_trusted = _validate_launcher_mutation(
            runner,
            service,
            allow_masked=False,
        )
        results.extend(trust_results)
        if not launcher_trusted:
            return results
        results.append(
            runner.run(
                ["systemctl", "disable", service],
                mutate=True,
                allow_fail=True,
            )
        )
    return results


def _restore_service_activity(
    runner: CommandRunner,
    service: str,
    *,
    active: bool | None,
    current_active: bool | None,
) -> list[CommandResult]:
    if active is not None and active != current_active:
        verb = "start" if active else "stop"
        results, launcher_trusted = _validate_launcher_mutation(
            runner,
            service,
            allow_masked=False,
        )
        if not launcher_trusted:
            return results
        results.append(
            runner.run(
                ["systemctl", verb, service],
                mutate=True,
                allow_fail=True,
            )
        )
        return results
    return []


def restore_rollback_service_activity(
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    current_audit: HostAudit,
    *,
    units: set[str],
) -> list[CommandResult]:
    """Restore one explicitly selected launcher after its prerequisite gate."""

    requested = set(units)
    if len(requested) != 1:
        return [
            CommandResult(
                ["rollback-precondition", "service-restore-scope"],
                1,
                stderr="rollback service activity must be committed one unit at a time",
            )
        ]
    unsupported = sorted(requested - set(_SERVICE_ACTIVITY_RESTORE_ORDER))
    if unsupported:
        return [
            CommandResult(
                ["rollback-precondition", "service-restore-scope"],
                1,
                stderr="unsupported rollback service unit(s): "
                + ", ".join(unsupported),
            )
        ]
    target_active = {
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_active,
        _NVIDIA_PERSISTENCED_UNIT: snapshot.nvidia_persistenced_active,
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_active,
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_active,
    }
    current_active = {
        _FABRIC_MANAGER_UNIT: current_audit.fabric_manager_active,
        _NVIDIA_PERSISTENCED_UNIT: current_audit.nvidia_persistenced_active,
        _DOCKER_SERVICE_UNIT: current_audit.docker_service_active,
        _DOCKER_SOCKET_UNIT: current_audit.docker_socket_active,
    }
    target_unit_file_state = {
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (snapshot.nvidia_persistenced_unit_file_state),
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_unit_file_state,
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_unit_file_state,
    }
    current_unit_file_state = {
        _FABRIC_MANAGER_UNIT: current_audit.fabric_manager_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (current_audit.nvidia_persistenced_unit_file_state),
        _DOCKER_SERVICE_UNIT: current_audit.docker_service_unit_file_state,
        _DOCKER_SOCKET_UNIT: current_audit.docker_socket_unit_file_state,
    }
    results: list[CommandResult] = []
    for service in _SERVICE_ACTIVITY_RESTORE_ORDER:
        if service not in requested:
            continue
        if target_active[service] is None or current_active[service] is None:
            results.append(
                CommandResult(
                    ["rollback-precondition", "service-state", service],
                    1,
                    stderr=(
                        f"cannot restore {service} activity exactly: current "
                        "or snapshot active state is unknown"
                    ),
                )
            )
            return results
        target_state = target_unit_file_state[service]
        prepared_state = (
            _prepared_unit_file_state(target_state)
            if target_state in _SNAPSHOT_UNIT_FILE_STATES
            else None
        )
        if (
            prepared_state is None
            or current_unit_file_state[service] is None
            or current_unit_file_state[service] != prepared_state
        ):
            results.append(
                CommandResult(
                    [
                        "rollback-precondition",
                        "service-unit-file-state",
                        service,
                    ],
                    1,
                    stderr=(
                        f"cannot restore {service} activity until its current "
                        "unit-file state is safely prepared for the rollback "
                        "activity commit"
                    ),
                )
            )
            return results
        service_results = _restore_service_activity(
            runner,
            service,
            active=target_active[service],
            current_active=current_active[service],
        )
        results.extend(service_results)
        if any(result.returncode not in (0, None) for result in service_results):
            return results
        if service_results:
            current_active[service] = target_active[service]
    return results


def prepare_rollback_service_activity(
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    current_audit: HostAudit,
    *,
    units: set[str],
) -> list[CommandResult]:
    """Unmask one launcher while keeping persistent activation disabled."""

    requested = set(units)
    if len(requested) != 1:
        return [
            CommandResult(
                ["rollback-precondition", "service-restore-scope"],
                1,
                stderr=(
                    "rollback service preparation must be committed one unit at a time"
                ),
            )
        ]
    unsupported = sorted(requested - set(_SERVICE_ACTIVITY_RESTORE_ORDER))
    if unsupported:
        return [
            CommandResult(
                ["rollback-precondition", "service-restore-scope"],
                1,
                stderr="unsupported rollback service unit(s): "
                + ", ".join(unsupported),
            )
        ]
    target_unit_file_state = {
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (snapshot.nvidia_persistenced_unit_file_state),
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_unit_file_state,
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_unit_file_state,
    }
    current_unit_file_state = {
        _FABRIC_MANAGER_UNIT: current_audit.fabric_manager_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (current_audit.nvidia_persistenced_unit_file_state),
        _DOCKER_SERVICE_UNIT: current_audit.docker_service_unit_file_state,
        _DOCKER_SOCKET_UNIT: current_audit.docker_socket_unit_file_state,
    }
    service = next(iter(requested))
    return _prepare_service_for_activity(
        runner,
        service,
        target_state=target_unit_file_state[service],
        current_state=current_unit_file_state[service],
    )


def restore_rollback_service_enablement(
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    current_audit: HostAudit,
    *,
    units: set[str],
) -> list[CommandResult]:
    """Restore one selected unit's exact persistent systemd state."""

    requested = set(units)
    if len(requested) != 1:
        return [
            CommandResult(
                ["rollback-precondition", "service-restore-scope"],
                1,
                stderr=(
                    "rollback service enablement must be committed one unit at a time"
                ),
            )
        ]
    unsupported = sorted(requested - set(_SERVICE_ACTIVITY_RESTORE_ORDER))
    if unsupported:
        return [
            CommandResult(
                ["rollback-precondition", "service-restore-scope"],
                1,
                stderr="unsupported rollback service unit(s): "
                + ", ".join(unsupported),
            )
        ]
    target_unit_file_state = {
        _FABRIC_MANAGER_UNIT: snapshot.fabric_manager_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (snapshot.nvidia_persistenced_unit_file_state),
        _DOCKER_SERVICE_UNIT: snapshot.docker_service_unit_file_state,
        _DOCKER_SOCKET_UNIT: snapshot.docker_socket_unit_file_state,
    }
    current_unit_file_state = {
        _FABRIC_MANAGER_UNIT: current_audit.fabric_manager_unit_file_state,
        _NVIDIA_PERSISTENCED_UNIT: (current_audit.nvidia_persistenced_unit_file_state),
        _DOCKER_SERVICE_UNIT: current_audit.docker_service_unit_file_state,
        _DOCKER_SOCKET_UNIT: current_audit.docker_socket_unit_file_state,
    }
    service = next(iter(requested))
    return _restore_service_unit_file_state(
        runner,
        service,
        target_state=target_unit_file_state[service],
        current_state=current_unit_file_state[service],
    )


def verify_rollback(
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    *,
    include_service_state: bool = True,
) -> list[Verification]:
    installed = {
        _package_state_identity(pkg) for pkg in audit.packages if pkg.installed
    }
    expected_packages = {
        _package_state_identity(pkg)
        for pkg in snapshot.packages
        if pkg.installed and pkg.version
    }
    mismatched = sorted(
        _format_package_identity(identity)
        for identity in expected_packages
        if identity not in installed
    )
    remaining_added = sorted(
        _format_package_identity(identity)
        for identity in installed
        if identity not in expected_packages
    )
    module_names_ok = True
    if snapshot.module_names:
        try:
            module_names_ok = nvidia_module_unload_order() == snapshot.module_names
        except ModuleDependencyError:
            module_names_ok = False
    module_ok = (
        audit.module.loaded == snapshot.module_loaded
        and audit.module.version == snapshot.module_version
        and audit.module.open_module == snapshot.module_open_module
        and audit.module.signed == snapshot.module_signed
        and audit.module.installed_version == snapshot.module_installed_version
        and audit.module.installed_open_module == snapshot.module_installed_open_module
        and audit.module.installed_signed == snapshot.module_installed_signed
        and module_names_ok
    )
    checks = [
        Verification(
            "rollback.gpu-inventory",
            audit.gpu_uuids == snapshot.gpu_uuids,
            detail="GPU UUID inventory must exactly match the rollback snapshot.",
        ),
        Verification(
            "rollback.kernel",
            audit.kernel.running == snapshot.kernel,
            detail=f"Running kernel must match snapshot kernel {snapshot.kernel}.",
        ),
        Verification(
            "rollback.module-version",
            module_ok,
            detail=(
                "Loaded and on-disk module version, flavor, signature, and dependent-module "
                f"set must match snapshot value {snapshot.module_version or 'not loaded'}."
            ),
        ),
        Verification(
            "rollback.packages-restored",
            audit.package_inventory_complete and not mismatched,
            detail="Snapshot package versions restored."
            if not mismatched
            else f"Package versions not restored: {', '.join(mismatched)}.",
        ),
        Verification(
            "rollback.added-packages-removed",
            not remaining_added,
            detail="Packages added by convergence removed."
            if not remaining_added
            else f"Added packages still installed: {', '.join(remaining_added)}.",
        ),
    ]
    if snapshot.mig_mode is not None:
        checks.append(
            Verification(
                "rollback.mig-mode",
                audit.mig_mode == snapshot.mig_mode,
                detail=f"MIG mode must match snapshot state {snapshot.mig_mode}.",
            )
        )
        checks.append(
            Verification(
                "rollback.mig-pending",
                audit.mig_mode_pending == snapshot.mig_mode,
                detail=(
                    "Pending MIG mode must match the snapshot baseline so reboot cannot reintroduce drift."
                ),
            )
        )
        checks.append(
            Verification(
                "rollback.mig-geometry",
                bool(
                    audit.mig_geometry_complete
                    and audit.mig_geometry == snapshot.mig_geometry
                    and (snapshot.mig_mode == "enabled" or not audit.mig_device_uuids)
                ),
                detail=(
                    "Stable GPU-instance and compute-instance geometry must "
                    "exactly match the rollback baseline."
                ),
            )
        )
    service_states = (
        (
            "rollback.docker-service-active",
            snapshot.docker_service_active,
            audit.docker_service_active,
        ),
        (
            "rollback.docker-service-enabled",
            snapshot.docker_service_enabled,
            audit.docker_service_enabled,
        ),
        (
            "rollback.docker-service-unit-file-state",
            snapshot.docker_service_unit_file_state,
            audit.docker_service_unit_file_state,
        ),
        (
            "rollback.docker-socket-active",
            snapshot.docker_socket_active,
            audit.docker_socket_active,
        ),
        (
            "rollback.docker-socket-enabled",
            snapshot.docker_socket_enabled,
            audit.docker_socket_enabled,
        ),
        (
            "rollback.docker-socket-unit-file-state",
            snapshot.docker_socket_unit_file_state,
            audit.docker_socket_unit_file_state,
        ),
        (
            "rollback.nvidia-persistenced-active",
            snapshot.nvidia_persistenced_active,
            audit.nvidia_persistenced_active,
        ),
        (
            "rollback.nvidia-persistenced-enabled",
            snapshot.nvidia_persistenced_enabled,
            audit.nvidia_persistenced_enabled,
        ),
        (
            "rollback.nvidia-persistenced-unit-file-state",
            snapshot.nvidia_persistenced_unit_file_state,
            audit.nvidia_persistenced_unit_file_state,
        ),
        (
            "rollback.fabric-manager-active",
            snapshot.fabric_manager_active,
            audit.fabric_manager_active,
        ),
        (
            "rollback.fabric-manager-enabled",
            snapshot.fabric_manager_enabled,
            audit.fabric_manager_enabled,
        ),
        (
            "rollback.fabric-manager-unit-file-state",
            snapshot.fabric_manager_unit_file_state,
            audit.fabric_manager_unit_file_state,
        ),
    )
    if include_service_state:
        for name, expected_state, actual in service_states:
            if expected_state is not None:
                checks.append(
                    Verification(
                        name,
                        actual == expected_state,
                        detail=f"Observed service state must match snapshot value {expected_state}.",
                    )
                )
    mismatched_files = [
        file.path for file in snapshot.managed_files if not _managed_file_matches(file)
    ]
    checks.append(
        Verification(
            "rollback.managed-files",
            not mismatched_files,
            detail=(
                "Managed configuration and package-policy files restored."
                if not mismatched_files
                else "Managed files not restored: " + ", ".join(mismatched_files)
            ),
        )
    )
    return checks


def _managed_file_matches(snapshot: FileSnapshot) -> bool:
    path = Path(snapshot.path)
    try:
        current, metadata = read_bounded_utf8_with_metadata(
            path,
            max_bytes=MAX_MANAGED_FILE_BYTES,
            required_owner_uid=os.geteuid(),
            require_trusted_ancestors=True,
        )
    except FileNotFoundError:
        return not snapshot.existed
    except (OSError, BoundedFileError):
        return False
    if not snapshot.existed:
        return False
    if snapshot.mode is None or stat.S_IMODE(metadata.st_mode) != snapshot.mode:
        return False
    try:
        expected = base64.b64decode(
            snapshot.content_base64 or "", validate=True
        ).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    return current == expected


def _rollback_commands(
    packages: list[PackageInfo],
    pm: str | None,
    *,
    remove_packages: list[str] | None = None,
    exact_removals: bool = False,
    snapshot_path: str | None = None,
    package_payloads: PackagePayloadBundle | None = None,
    expected_rpm_installs: list[str] | None = None,
    expected_rpm_removals: list[str] | None = None,
) -> list[list[str]]:
    installed = [
        pkg
        for pkg in packages
        if pkg.installed and pkg.version and _interesting_package(pkg.name)
    ]
    remove_specs = sorted(set(remove_packages or []))
    local_payloads: list[str] | None = None
    if package_payloads is not None:
        if snapshot_path is None:
            raise RollbackSnapshotError(
                "rollback package payloads require an exact snapshot path"
            )
        by_identity = {
            (payload.name, payload.architecture, payload.epoch, payload.version): payload
            for payload in package_payloads.packages
            if "baseline" in payload.roles
        }
        local_payloads = []
        for package in installed:
            identity = (
                package.name,
                package.architecture or "",
                package.epoch,
                package.version or "",
            )
            payload = by_identity.get(identity)
            if payload is None:
                raise RollbackSnapshotError(
                    "rollback package payload manifest omits baseline identity "
                    + _format_package_identity(identity)
                )
            local_payloads.append(
                str(
                    Path(snapshot_path).parent
                    / package_payloads.directory
                    / payload.filename
                )
            )
    elif snapshot_path is not None:
        raise RollbackSnapshotError(
            "applicable rollback snapshot has no retained package payloads"
        )
    if pm == "apt-get":
        specs = local_payloads if local_payloads is not None else [
            f"{pkg.name}{f':{pkg.architecture}' if pkg.architecture else ''}={pkg.version}"
            for pkg in installed
            if pkg.manager == "apt"
        ]
        operands = [*specs, *(f"{package}-" for package in remove_specs)]
        return (
            [
                [
                    "apt-get",
                    "install",
                    "-y",
                    "--allow-change-held-packages",
                    "--allow-downgrades",
                    "--no-download",
                    "--no-install-recommends",
                    "--purge",
                    *operands,
                ]
            ]
            if operands
            else []
        )
    if pm in {"dnf", "yum"}:
        specs = local_payloads if local_payloads is not None else [
            _rpm_package_spec(pkg) for pkg in installed if pkg.manager == "rpm"
        ]
        if local_payloads is not None:
            expected_installs = sorted(
                expected_rpm_installs
                if expected_rpm_installs is not None
                else [
                    _rpm_package_spec(package)
                    for package in installed
                    if package.manager == "rpm"
                ]
            )
            expected_removals = sorted(
                expected_rpm_removals
                if expected_rpm_removals is not None
                else remove_specs
            )
            return (
                [
                    dnf_local_transaction_command(
                        apply=True,
                        restore_paths=specs,
                        remove_specs=remove_specs,
                        expected_installs=expected_installs,
                        expected_removals=expected_removals,
                    )
                ]
                if specs or remove_specs
                else []
            )
        return [
            *(
                [
                    [
                        pm,
                        "--disablerepo=*",
                        "--disableplugin=versionlock",
                        "--noautoremove",
                        "--setopt=localpkg_gpgcheck=1",
                        "install" if local_payloads is not None else "install-nevra",
                        "-y",
                        *specs,
                    ]
                ]
                if specs
                else []
            ),
            *_remove_commands(
                pm,
                remove_specs,
                exact=exact_removals,
            ),
        ]
    if pm == "zypper":
        specs = local_payloads if local_payloads is not None else [
            _zypper_package_spec(pkg) for pkg in installed if pkg.manager == "rpm"
        ]
        operands = [*specs, *(f"-{package}" for package in remove_specs)]
        return (
            [
                [
                    "zypper",
                    "--non-interactive",
                    "--disable-repositories",
                    "--no-refresh",
                    "install",
                    "--oldpackage",
                    "--no-recommends",
                    "--no-force-resolution",
                    "--",
                    *operands,
                ]
            ]
            if operands
            else []
        )
    return []


def _rpm_package_spec(package: PackageInfo) -> str:
    epoch = f"{package.epoch}:" if package.epoch else ""
    architecture = f".{package.architecture}" if package.architecture else ""
    return f"{package.name}-{epoch}{package.version}{architecture}"


def _zypper_package_spec(package: PackageInfo) -> str:
    architecture = f".{package.architecture}" if package.architecture else ""
    epoch = f"{package.epoch}:" if package.epoch else ""
    return f"{package.name}{architecture}={epoch}{package.version}"


def _package_state_identity(
    package: PackageInfo,
) -> tuple[str, str | None, str | None, str | None]:
    return (
        package.name,
        package.architecture,
        package.epoch,
        package.version,
    )


def _format_package_identity(
    identity: tuple[str, str | None, str | None, str | None],
) -> str:
    name, architecture, epoch, version = identity
    architecture_suffix = f":{architecture}" if architecture else ""
    version_suffix = f"={f'{epoch}:' if epoch else ''}{version}" if version else ""
    return f"{name}{architecture_suffix}{version_suffix}"


def _package_removal_specs(
    packages: list[PackageInfo], package_manager: str | None
) -> list[str]:
    specs: list[str] = []
    for package in packages:
        if package_manager == "apt-get":
            specs.append(
                f"{package.name}{f':{package.architecture}' if package.architecture else ''}"
            )
        elif package_manager in {"dnf", "yum"}:
            specs.append(_rpm_package_spec(package))
        elif package_manager == "zypper":
            architecture = f".{package.architecture}" if package.architecture else ""
            epoch = f"{package.epoch}:" if package.epoch else ""
            version = f"={epoch}{package.version}" if package.version else ""
            specs.append(f"{package.name}{architecture}{version}")
    return sorted(set(specs))


def _package_tracking_names(
    package_manager: str,
    targets: list[str],
    kernel: str,
) -> list[str]:
    names: list[str] = []
    for target in targets:
        name = target.split("=", 1)[0]
        if package_manager in {"dnf", "yum"} and name == f"kernel-devel-{kernel}":
            name = "kernel-devel"
        if name not in names:
            names.append(name)
    return names


def validate_snapshot_for_apply(
    snapshot: RollbackSnapshot,
    source_path: str,
    audit: HostAudit,
    *,
    runner: CommandRunner | None = None,
) -> None:
    source = Path(os.path.abspath(source_path))
    snapshot_root = Path(os.path.abspath(SNAPSHOT_DIR))
    if source.parent != snapshot_root:
        raise RollbackSnapshotError(
            f"applied rollback snapshots must be stored directly in {SNAPSHOT_DIR}"
        )
    if (
        not snapshot.path
        or not Path(snapshot.path).is_absolute()
        or Path(os.path.abspath(snapshot.path)) != source
    ):
        raise RollbackSnapshotError(
            "rollback snapshot path binding does not match the loaded file"
        )
    try:
        observed_owner_uid = source.lstat().st_uid
    except OSError as exc:
        raise RollbackSnapshotError(
            f"cannot validate rollback snapshot trust: {exc}"
        ) from exc
    if observed_owner_uid != os.geteuid():
        raise RollbackSnapshotError(
            "applied rollback snapshot must be a private, singly linked regular "
            "file owned by the effective uid"
        )
    binding = getattr(snapshot, "_source_binding", None)
    try:
        if not isinstance(binding, _SnapshotSourceBinding):
            raise RollbackSnapshotError(
                "applied rollback requires the exact object returned by trusted snapshot loading"
            )
        trusted_snapshot = load_snapshot(str(source), require_private=True)
        trusted_binding = getattr(trusted_snapshot, "_source_binding", None)
        if trusted_binding != binding or trusted_snapshot != snapshot:
            raise RollbackSnapshotError(
                "rollback snapshot file or trusted ancestry changed after loading"
            )
    except (OSError, BoundedFileError) as exc:
        raise RollbackSnapshotError(
            f"cannot validate rollback snapshot trust: {exc}"
        ) from exc
    expected = {
        "host_id": _host_identity(),
        "os_id": audit.os_id,
        "os_version": audit.os_version,
        "architecture": platform.machine(),
        "package_manager": audit.package_manager,
    }
    mismatches = [
        field_name
        for field_name, current_value in expected.items()
        if getattr(snapshot, field_name) != current_value
    ]
    if mismatches:
        raise RollbackSnapshotError(
            "rollback snapshot does not belong to the current host/backend: "
            + ", ".join(mismatches)
        )
    if snapshot.package_payloads is None or snapshot.package_manager is None:
        raise RollbackSnapshotError(
            "applied rollback snapshot has no retained package payload authority"
        )
    try:
        validate_package_payloads(
            source,
            snapshot.package_payloads,
            snapshot.packages,
            snapshot.package_manager,
            runner=runner,
            required_owner_uid=os.geteuid(),
        )
    except PackagePayloadError as exc:
        raise RollbackSnapshotError(
            f"rollback package payload validation failed: {exc}"
        ) from exc


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _snapshot_integrity(snapshot: RollbackSnapshot) -> str:
    payload = asdict(snapshot)
    payload.pop("integrity_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _host_identity() -> str:
    machine_id_path = Path("/etc/machine-id")
    try:
        machine_id = read_bounded_utf8(machine_id_path, max_bytes=256).strip().lower()
    except (OSError, BoundedFileError):
        machine_id = ""
    if re.fullmatch(r"[a-f0-9]{32}", machine_id):
        digest = hashlib.sha256(machine_id.encode("ascii")).hexdigest()
        return f"machine-id-sha256:{digest}"
    hostname = platform.node().strip()
    if not hostname:
        raise RollbackSnapshotError(
            "cannot determine a stable host identity for rollback binding"
        )
    digest = hashlib.sha256(hostname.encode("utf-8")).hexdigest()
    return f"hostname-sha256:{digest}"


def _remove_commands(
    pm: str | None,
    packages: list[str],
    *,
    exact: bool = False,
) -> list[list[str]]:
    if not packages:
        return []
    if pm == "apt-get":
        raise AssertionError("APT removals must be part of one install transaction")
    if pm in {"dnf", "yum"}:
        operation = "remove-nevra" if exact else "remove-n"
        return [
            [
                pm,
                "--disablerepo=*",
                "--disableplugin=versionlock",
                "--noautoremove",
                operation,
                "-y",
                *packages,
            ]
        ]
    if pm == "zypper":
        raise AssertionError("Zypper removals must be part of one install transaction")
    return []
