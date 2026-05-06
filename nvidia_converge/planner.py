from __future__ import annotations

from .models import DesiredState, Finding, HostAudit, PlanAction, Severity


def build_plan(desired: DesiredState, audit: HostAudit, findings: list[Finding]) -> list[PlanAction]:
    pm = audit.package_manager
    actions: list[PlanAction] = []
    if pm is None:
        return [PlanAction("unsupported.package-manager", "Cannot converge without a supported package manager.", [], reason="No apt/dnf/yum/zypper found")]

    package_cmd = _package_install_command(pm, desired, audit.kernel.running)
    if any(f.severity == Severity.ERROR for f in findings) or _needs_driver_install(desired, audit):
        actions.append(PlanAction("snapshot.current-state", "Record installed NVIDIA packages and kernel/module state before changes.", [["nvidia-converge", "snapshot"]], reason="Required for rollback."))
        actions.append(PlanAction("install.packages", "Install desired NVIDIA driver, CUDA compatibility, Fabric Manager, container runtime, and kernel build dependencies.", package_cmd, reason=f"Converge to driver {desired.driver} and CUDA compat {desired.cuda_compat}."))

    if desired.container_runtime == "docker":
        actions.append(PlanAction("configure.docker-runtime", "Configure Docker to use the NVIDIA container runtime.", [["nvidia-ctk", "runtime", "configure", "--runtime=docker"], ["systemctl", "restart", "docker"]], reason="Required for container GPU tests."))

    if desired.fabric_manager:
        actions.append(PlanAction("enable.fabric-manager", "Enable and start NVIDIA Fabric Manager.", [["systemctl", "enable", "--now", "nvidia-fabricmanager"]], reason="Desired state requires fabric_manager: true."))

    if desired.mig == "disabled" and audit.mig_mode == "enabled":
        actions.append(PlanAction("disable.mig", "Disable MIG mode on all GPUs.", [["nvidia-smi", "-mig", "0"]], destructive=True, reason="May require GPU reset and workload drain."))
    if desired.mig == "enabled" and audit.mig_mode == "disabled":
        actions.append(PlanAction("enable.mig", "Enable MIG mode on all GPUs.", [["nvidia-smi", "-mig", "1"]], destructive=True, reason="May require GPU reset and workload drain."))

    if desired.kernel_policy == "pin-compatible":
        actions.extend(lock_actions(desired, audit))

    actions.append(PlanAction("verify.stack", "Validate module, nvidia-smi, NVML, and container GPU access.", [["nvidia-converge", "verify"]], reason="Post-convergence validation."))
    return actions


def lock_actions(desired: DesiredState, audit: HostAudit) -> list[PlanAction]:
    pm = audit.package_manager
    if pm == "apt-get":
        packages = _apt_package_names(desired)
        return [PlanAction("lock.apt", "Pin NVIDIA driver/toolkit packages and hold running kernel compatibility.", [["apt-mark", "hold", *packages]], reason="Prevent accidental driver/kernel/toolkit skew.")]
    if pm in {"dnf", "yum"}:
        packages = _rpm_package_names(desired)
        return [PlanAction("lock.rpm", "Version-lock NVIDIA driver/toolkit packages and running kernel compatibility.", [[pm, "versionlock", "add", *packages]], reason="Prevent accidental driver/kernel/toolkit skew.")]
    if pm == "zypper":
        packages = _rpm_package_names(desired)
        return [PlanAction("lock.zypper", "Lock NVIDIA driver/toolkit packages and running kernel compatibility.", [["zypper", "--non-interactive", "addlock", *packages]], reason="Prevent accidental driver/kernel/toolkit skew.")]
    return []


def _needs_driver_install(desired: DesiredState, audit: HostAudit) -> bool:
    return not audit.module.version or not audit.module.version.startswith(desired.driver_major)


def _package_install_command(pm: str, desired: DesiredState, kernel: str) -> list[list[str]]:
    if pm == "apt-get":
        return [["apt-get", "update"], ["apt-get", "install", "-y", *_apt_package_names(desired), f"linux-headers-{kernel}", "build-essential"]]
    if pm in {"dnf", "yum"}:
        return [[pm, "install", "-y", *_rpm_package_names(desired), f"kernel-devel-{kernel}", "gcc", "make"]]
    if pm == "zypper":
        return [["zypper", "--non-interactive", "install", *_rpm_package_names(desired), "gcc", "make"]]
    return []


def _apt_package_names(desired: DesiredState) -> list[str]:
    driver_pkg = f"nvidia-driver-{desired.driver_major}-open" if desired.open_kernel_module else f"nvidia-driver-{desired.driver_major}"
    packages = [driver_pkg, f"cuda-compat-{desired.cuda_compat.replace('.', '-')}"]
    if desired.fabric_manager:
        packages.append(f"nvidia-fabricmanager-{desired.driver_major}")
    return [*packages, "nvidia-container-toolkit", "docker-ce"]


def _rpm_package_names(desired: DesiredState) -> list[str]:
    module_pkg = f"nvidia-open-{desired.driver_major}" if desired.open_kernel_module else f"nvidia-driver-{desired.driver_major}"
    packages = [module_pkg, f"cuda-compat-{desired.cuda_compat.replace('.', '-')}"]
    if desired.fabric_manager:
        packages.append(f"nvidia-fabric-manager-{desired.driver_major}")
    return [*packages, "nvidia-container-toolkit", "docker-ce"]
