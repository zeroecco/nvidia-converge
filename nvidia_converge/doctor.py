from __future__ import annotations

from .models import DesiredState, Finding, HostAudit, Severity


def diagnose(desired: DesiredState, audit: HostAudit) -> list[Finding]:
    findings: list[Finding] = []
    if audit.package_manager is None:
        findings.append(Finding("package-manager.missing", Severity.ERROR, "No supported package manager found", "Expected apt, dnf, yum, or zypper to manage NVIDIA stack packages."))
    if not audit.kernel.headers_installed:
        findings.append(Finding("kernel.headers.missing", Severity.ERROR, "Kernel headers are missing", f"Headers for running kernel {audit.kernel.running} are required to compile or install the NVIDIA kernel module.", remediation="Install matching kernel headers or switch to a kernel with available headers."))
    if audit.kernel.compiler is None:
        findings.append(Finding("compiler.missing", Severity.ERROR, "No C compiler found", "The NVIDIA kernel module build needs gcc or cc.", remediation="Install build-essential or gcc/make for this distribution."))
    if desired.secure_boot == "disabled" and audit.kernel.secure_boot_enabled is True:
        findings.append(Finding("secure-boot.enabled", Severity.ERROR, "Secure Boot is enabled", "Desired state requires Secure Boot disabled.", remediation="Disable Secure Boot in firmware or change desired.secure_boot."))
    if desired.secure_boot == "signed" and audit.kernel.secure_boot_enabled and audit.module.signed is False:
        findings.append(Finding("secure-boot.unsigned-module", Severity.ERROR, "Secure Boot requires a signed NVIDIA module", "Secure Boot is enabled and modinfo did not show a module signer.", remediation="Install signed packages or enroll a MOK and sign the module."))
    if not audit.module.loaded:
        findings.append(Finding("module.not-loaded", Severity.ERROR, "NVIDIA kernel module is not loaded", "The host has no loaded nvidia kernel module, so GPUs will not be exposed to NVML or containers.", evidence={"devices": audit.module.devices}, remediation="Install the desired driver, rebuild initramfs if needed, and load nvidia."))
    if audit.module.version and not audit.module.version.startswith(desired.driver_major):
        findings.append(Finding("driver.version-mismatch", Severity.ERROR, "Loaded NVIDIA module does not match desired driver", f"Loaded module version {audit.module.version} does not match desired {desired.driver}.", remediation="Replace the installed driver with the desired branch."))
    if desired.open_kernel_module and audit.module.open_module is False:
        findings.append(Finding("driver.closed-module", Severity.ERROR, "Closed NVIDIA module detected", "Desired state requires the open kernel module variant.", remediation="Install the open module package for the desired driver branch."))
    if audit.nvidia_smi.returncode != 0:
        findings.append(Finding("nvidia-smi.failed", Severity.ERROR, "nvidia-smi failed", "Driver userspace cannot communicate with the NVIDIA stack.", evidence={"stdout": audit.nvidia_smi.stdout, "stderr": audit.nvidia_smi.stderr}, remediation="Repair driver/module/userspace version alignment."))
    if audit.nvml.returncode != 0:
        findings.append(Finding("nvml.failed", Severity.ERROR, "NVML library load failed", "libnvidia-ml.so.1 is unavailable or not loadable.", evidence={"stderr": audit.nvml.stderr}, remediation="Install the driver userspace libraries for the desired branch."))
    if desired.container_runtime == "docker":
        if not audit.runtime.docker_installed:
            findings.append(Finding("docker.missing", Severity.ERROR, "Docker is not installed", "Desired state requires Docker as the container runtime.", remediation="Install Docker and configure the NVIDIA container toolkit."))
        if not audit.runtime.nvidia_container_runtime_installed:
            findings.append(Finding("container-toolkit.missing", Severity.ERROR, "NVIDIA container toolkit is missing", "Docker cannot run GPU workloads without the NVIDIA container runtime/toolkit.", remediation="Install nvidia-container-toolkit and run nvidia-ctk runtime configure."))
        if audit.runtime.docker_gpus_usable is False:
            findings.append(Finding("docker.nvidia-runtime-missing", Severity.ERROR, "Docker is not configured with the NVIDIA runtime", "docker info does not list an nvidia runtime.", remediation="Run nvidia-ctk runtime configure --runtime=docker and restart Docker."))
    if desired.fabric_manager and audit.fabric_manager_active is False:
        findings.append(Finding("fabric-manager.inactive", Severity.WARNING, "NVIDIA Fabric Manager is inactive", "Desired state requires Fabric Manager to be installed and active.", remediation="Install and enable the matching nvidia-fabricmanager package."))
    if desired.mig == "disabled" and audit.mig_mode == "enabled":
        findings.append(Finding("mig.enabled", Severity.ERROR, "MIG mode is enabled", "Desired state requires MIG disabled.", remediation="Disable MIG with nvidia-smi -mig 0 and reset GPUs during a maintenance window."))
    if desired.mig == "enabled" and audit.mig_mode == "disabled":
        findings.append(Finding("mig.disabled", Severity.ERROR, "MIG mode is disabled", "Desired state requires MIG enabled.", remediation="Enable MIG with nvidia-smi -mig 1 and reset GPUs during a maintenance window."))
    if not findings:
        findings.append(Finding("stack.healthy", Severity.INFO, "NVIDIA stack matches observed desired-state checks", "No blocking audit issues were detected."))
    return findings
