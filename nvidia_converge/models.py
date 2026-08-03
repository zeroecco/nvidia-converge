from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DesiredState:
    role: str = "compute"
    driver: str = "580-open"
    cuda_compat: str = "none"
    secure_boot: str = "signed"
    container_runtime: str = "docker"
    container_test_image: str = (
        "nvidia/cuda:13.1.2-devel-ubuntu22.04"
        "@sha256:d8332c008e2ef270e82d286e5245e771f839645683075bba21ad9e4fa59dbcbb"
    )
    fabric_manager: bool = False
    mig: str = "disabled"
    mig_profile: str = "none"
    kernel_policy: str = "pin-compatible"

    @property
    def driver_major(self) -> str:
        match = re.match(r"^(\d+)", self.driver)
        return match.group(1) if match else self.driver.split("-", 1)[0]

    @property
    def open_kernel_module(self) -> bool:
        return self.driver.endswith("-open")

    @property
    def exact_driver_version(self) -> bool:
        return "." in self.driver

    @property
    def driver_match_label(self) -> str:
        return (
            f"version {self.driver}"
            if self.exact_driver_version
            else f"branch {self.driver_major}"
        )

    def matches_driver_version(self, version: str | None) -> bool:
        if not version:
            return False
        if self.exact_driver_version:
            return version == self.driver
        match = re.match(r"^(\d+)(?:\.|$)", version)
        return bool(match and match.group(1) == self.driver_major)


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False
    reason: str | None = None


@dataclass
class Finding:
    id: str
    severity: Severity
    summary: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None


@dataclass
class PackageInfo:
    name: str
    version: str | None = None
    manager: str | None = None
    installed: bool = False
    architecture: str | None = None
    epoch: str | None = None


@dataclass
class KernelInfo:
    running: str
    headers_installed: bool
    compiler: str | None = None
    secure_boot_enabled: bool | None = None


@dataclass
class ModuleInfo:
    loaded: bool
    version: str | None = None
    open_module: bool | None = None
    signed: bool | None = None
    devices: list[str] = field(default_factory=list)
    installed_version: str | None = None
    installed_open_module: bool | None = None
    installed_signed: bool | None = None


@dataclass
class RuntimeInfo:
    docker_installed: bool
    nvidia_container_runtime_installed: bool
    docker_gpus_usable: bool | None = None


@dataclass
class CudaCompatibilityInfo:
    version: str
    package_name: str
    package_version: str | None
    library_path: str
    library_present: bool
    library_probe: CommandResult


@dataclass(frozen=True)
class MigComputeInstance:
    profile: str
    profile_id: int


@dataclass(frozen=True)
class MigGpuInstance:
    gpu_uuid: str
    profile: str
    profile_id: int
    placement_start: int
    placement_size: int
    compute_instances: list[MigComputeInstance] = field(default_factory=list)


@dataclass
class PackagePolicySelector:
    identifier: str
    name: str
    kind: str
    relation: str | None = None
    version: str | None = None
    repositories: list[str] = field(default_factory=list)


@dataclass
class PackagePolicyInfo:
    backend: str | None
    observable: bool
    selectors: list[PackagePolicySelector] = field(default_factory=list)
    observation: CommandResult | None = None


@dataclass
class HostAudit:
    timestamp: str
    os_id: str | None
    os_version: str | None
    package_manager: str | None
    kernel: KernelInfo
    module: ModuleInfo
    runtime: RuntimeInfo
    packages: list[PackageInfo]
    nvidia_smi: CommandResult
    nvml: CommandResult
    fabric_manager_active: bool | None
    mig_mode: str | None
    cuda_compatibility: list[CudaCompatibilityInfo] = field(default_factory=list)
    package_inventory_complete: bool = True
    package_inventory_result: CommandResult | None = None
    docker_service_active: bool | None = None
    docker_service_enabled: bool | None = None
    docker_service_unit_file_state: str | None = None
    docker_socket_active: bool | None = None
    docker_socket_enabled: bool | None = None
    docker_socket_unit_file_state: str | None = None
    nvidia_persistenced_active: bool | None = None
    nvidia_persistenced_enabled: bool | None = None
    nvidia_persistenced_unit_file_state: str | None = None
    fabric_manager_enabled: bool | None = None
    fabric_manager_unit_file_state: str | None = None
    fabric_manager_version: str | None = None
    package_policy: PackagePolicyInfo = field(
        default_factory=lambda: PackagePolicyInfo(None, False)
    )
    open_kernel_module_supported: bool | None = None
    mig_capable: bool | None = None
    mig_mode_pending: str | None = None
    fabric_manager_applicable: bool | None = None
    fabric_manager_healthy: bool | None = None
    fabric_manager_health_result: CommandResult | None = None
    gpu_uuids: list[str] = field(default_factory=list)
    mig_geometry: list[MigGpuInstance] = field(default_factory=list)
    mig_device_uuids: list[str] = field(default_factory=list)
    mig_geometry_complete: bool = False
    mig_geometry_results: list[CommandResult] = field(default_factory=list)


@dataclass
class PlanAction:
    id: str
    description: str
    commands: list[list[str]]
    destructive: bool = False
    reason: str | None = None


@dataclass
class Verification:
    name: str
    ok: bool
    command: CommandResult | None = None
    detail: str | None = None


@dataclass
class FileSnapshot:
    path: str
    existed: bool
    content_base64: str | None
    mode: int | None


@dataclass(frozen=True)
class PackagePayload:
    name: str
    architecture: str
    epoch: str | None
    version: str
    format: str
    filename: str
    sha256: str
    size_bytes: int
    verification: str
    roles: tuple[str, ...] = ()
    signer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackagePayloadBundle:
    directory: str
    packages: tuple[PackagePayload, ...]
    total_size_bytes: int


@dataclass
class RollbackSnapshot:
    path: str | None
    packages: list[PackageInfo]
    kernel: str
    module_version: str | None
    commands: list[list[str]]
    schema_version: str = "2.6"
    created_at: str = ""
    operation_id: str = ""
    host_id: str = ""
    os_id: str | None = None
    os_version: str | None = None
    architecture: str = ""
    package_manager: str | None = None
    introduced_packages: list[str] = field(default_factory=list)
    module_loaded: bool = False
    module_names: list[str] = field(default_factory=list)
    module_open_module: bool | None = None
    module_signed: bool | None = None
    module_installed_version: str | None = None
    module_installed_open_module: bool | None = None
    module_installed_signed: bool | None = None
    mig_mode: str | None = None
    docker_service_active: bool | None = None
    docker_service_enabled: bool | None = None
    docker_service_unit_file_state: str | None = None
    docker_socket_active: bool | None = None
    docker_socket_enabled: bool | None = None
    docker_socket_unit_file_state: str | None = None
    nvidia_persistenced_active: bool | None = None
    nvidia_persistenced_enabled: bool | None = None
    nvidia_persistenced_unit_file_state: str | None = None
    fabric_manager_active: bool | None = None
    fabric_manager_enabled: bool | None = None
    fabric_manager_unit_file_state: str | None = None
    managed_files: list[FileSnapshot] = field(default_factory=list)
    package_payloads: PackagePayloadBundle | None = None
    gpu_uuids: list[str] = field(default_factory=list)
    mig_geometry: list[MigGpuInstance] = field(default_factory=list)
    integrity_sha256: str | None = None


@dataclass
class Report:
    schema_version: str
    generated_at: str
    desired: DesiredState
    audit: HostAudit | None = None
    findings: list[Finding] = field(default_factory=list)
    plan: list[PlanAction] = field(default_factory=list)
    command_results: list[CommandResult] = field(default_factory=list)
    verification: list[Verification] = field(default_factory=list)
    rollback: RollbackSnapshot | None = None
    sbom: list[PackageInfo] = field(default_factory=list)
    command: str | None = None
    mode: str = "dry-run"
    tool_version: str = ""
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    host_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    outcome: str = "running"
    exit_code: int | None = None
    reboot_required: bool | None = None
    incomplete: bool = False
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
