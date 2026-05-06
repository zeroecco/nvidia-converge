from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DesiredState:
    role: str = "compute"
    driver: str = "580-open"
    cuda_compat: str = "13.0"
    secure_boot: str = "signed"
    container_runtime: str = "docker"
    fabric_manager: bool = True
    mig: str = "disabled"
    kernel_policy: str = "pin-compatible"

    @property
    def driver_major(self) -> str:
        return self.driver.split("-", 1)[0]

    @property
    def open_kernel_module(self) -> bool:
        return self.driver.endswith("-open")


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


@dataclass
class RuntimeInfo:
    docker_installed: bool
    nvidia_container_runtime_installed: bool
    docker_gpus_usable: bool | None = None


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
class RollbackSnapshot:
    path: str | None
    packages: list[PackageInfo]
    kernel: str
    module_version: str | None
    commands: list[list[str]]


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
