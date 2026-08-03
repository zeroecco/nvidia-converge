from __future__ import annotations

from .mig import full_mig_geometry_matches
from .models import DesiredState, Finding, HostAudit, Severity


def diagnose(desired: DesiredState, audit: HostAudit) -> list[Finding]:
    findings: list[Finding] = []
    if audit.package_manager is None:
        findings.append(Finding("package-manager.missing", Severity.ERROR, "No supported package manager found", "Expected apt, dnf, yum, or zypper to manage NVIDIA stack packages."))
    if not audit.package_inventory_complete:
        evidence = {}
        if audit.package_inventory_result:
            evidence = {
                "returncode": audit.package_inventory_result.returncode,
                "stderr": audit.package_inventory_result.stderr,
            }
        findings.append(Finding("package-inventory.incomplete", Severity.ERROR, "Installed package inventory is incomplete", "A complete package inventory is required before convergence so rollback state can be trusted.", evidence=evidence, remediation="Repair the package database/query tooling and rerun doctor before applying changes."))
    if not audit.kernel.headers_installed:
        findings.append(
            Finding(
                "kernel.headers.missing",
                Severity.ERROR,
                "Running-kernel headers are not ready",
                f"A trusted, package-owned, prepared header tree for running kernel {audit.kernel.running} is required to compile or install the NVIDIA kernel module.",
                remediation="Install matching kernel headers, repair the running kernel's build link, or switch to a kernel with available headers.",
            )
        )
    if audit.kernel.compiler is None:
        findings.append(Finding("compiler.missing", Severity.ERROR, "No C compiler found", "The NVIDIA kernel module build needs gcc or cc.", remediation="Install build-essential or gcc/make for this distribution."))
    if desired.secure_boot == "disabled" and audit.kernel.secure_boot_enabled is True:
        findings.append(Finding("secure-boot.enabled", Severity.ERROR, "Secure Boot is enabled", "Desired state requires Secure Boot disabled.", remediation="Disable Secure Boot in firmware or change desired.secure_boot."))
    if desired.secure_boot == "signed" and audit.module.signed is not True:
        findings.append(Finding("secure-boot.unsigned-module", Severity.ERROR, "Desired policy requires a verified loaded NVIDIA module", "The loaded module's kernel taint state did not prove a verified signature.", remediation="Install signed packages, enroll a trusted key if needed, and reload the NVIDIA module."))
    if desired.secure_boot == "signed" and audit.module.installed_signed is not True:
        findings.append(Finding("secure-boot.installed-module-unsigned", Severity.ERROR, "Desired policy requires a signed on-disk NVIDIA module", "modinfo did not show a signer for the NVIDIA module selected for the running kernel.", remediation="Install a signed NVIDIA module package or sign it with an enrolled key before reloading."))
    if audit.kernel.secure_boot_enabled is None:
        findings.append(Finding("secure-boot.unknown", Severity.ERROR, "Secure Boot state could not be determined", "The desired Secure Boot policy cannot be proven without an observable firmware state.", remediation="Install mokutil and ensure the host exposes its UEFI Secure Boot state."))
    if desired.open_kernel_module:
        if audit.open_kernel_module_supported is False:
            findings.append(
                Finding(
                    "gpu.open-module-unsupported",
                    Severity.ERROR,
                    "GPU hardware does not support the open kernel module",
                    "At least one observed GPU has compute capability older than Turing (7.5).",
                    remediation="Use a closed-module desired driver for Maxwell, Pascal, or Volta GPUs.",
                )
            )
        elif audit.open_kernel_module_supported is None:
            findings.append(
                Finding(
                    "gpu.open-module-support-unknown",
                    Severity.ERROR,
                    "Open kernel-module hardware support is unknown",
                    "The compute capability of every GPU must be observed before selecting the open module flavor.",
                    remediation="Restore nvidia-smi GPU capability queries or select a closed-module desired driver.",
                )
            )
    if not audit.module.loaded:
        findings.append(Finding("module.not-loaded", Severity.ERROR, "NVIDIA kernel module is not loaded", "The host has no loaded nvidia kernel module, so GPUs will not be exposed to NVML or containers.", evidence={"devices": audit.module.devices}, remediation="Install the desired driver, rebuild initramfs if needed, and load nvidia."))
    else:
        if audit.module.version is None:
            findings.append(
                Finding(
                    "driver.loaded-version-unknown",
                    Severity.ERROR,
                    "Loaded NVIDIA module version is unknown",
                    "The loaded module is present, but /sys/module/nvidia/version did not provide its version.",
                    remediation="Restore sysfs observability and verify the loaded NVIDIA module before convergence.",
                )
            )
        if audit.module.installed_version is None:
            findings.append(
                Finding(
                    "driver.installed-version-unknown",
                    Severity.ERROR,
                    "On-disk NVIDIA module version is unknown",
                    "modinfo did not provide the version of the NVIDIA module selected for the running kernel.",
                    remediation="Install working kmod tooling and a complete NVIDIA kernel module package for the running kernel.",
                )
            )
        if (
            audit.module.version is not None
            and audit.module.installed_version is not None
            and audit.module.version != audit.module.installed_version
        ):
            findings.append(
                Finding(
                    "driver.module-version-mismatch",
                    Severity.ERROR,
                    "Loaded and on-disk NVIDIA modules differ",
                    f"Loaded module {audit.module.version} does not match modinfo-selected module {audit.module.installed_version}.",
                    remediation="Drain GPU users and reload the complete NVIDIA module stack, then rerun doctor.",
                )
            )
    if audit.module.version and not desired.matches_driver_version(audit.module.version):
        findings.append(Finding("driver.version-mismatch", Severity.ERROR, "Loaded NVIDIA module does not match desired driver", f"Loaded module version {audit.module.version} does not match desired {desired.driver_match_label}.", remediation="Replace the installed driver with the desired version or branch."))
    if desired.open_kernel_module and audit.module.open_module is False:
        findings.append(Finding("driver.closed-module", Severity.ERROR, "Closed NVIDIA module detected", "Desired state requires the open kernel module variant.", remediation="Install the open module package for the desired driver branch."))
    if not desired.open_kernel_module and audit.module.open_module is True:
        findings.append(Finding("driver.open-module", Severity.ERROR, "Open NVIDIA module detected", "Desired state requires the closed kernel module variant.", remediation="Install the closed module package for the desired driver branch."))
    if desired.open_kernel_module and audit.module.open_module is None:
        findings.append(Finding("driver.module-flavor-unknown", Severity.ERROR, "Loaded NVIDIA module flavor could not be determined", "/proc/driver/nvidia/version did not prove whether the loaded module is the open or closed variant.", remediation="Restore NVIDIA procfs observability and reload the desired module variant."))
    if not desired.open_kernel_module and audit.module.open_module is None:
        findings.append(Finding("driver.module-flavor-unknown", Severity.ERROR, "Loaded NVIDIA module flavor could not be determined", "/proc/driver/nvidia/version did not prove whether the loaded module is the open or closed variant.", remediation="Restore NVIDIA procfs observability and reload the desired module variant."))
    if desired.open_kernel_module and audit.module.installed_open_module is False:
        findings.append(Finding("driver.installed-closed-module", Severity.ERROR, "Closed on-disk NVIDIA module detected", "modinfo shows that the running kernel would load the closed variant, while desired state requires the open variant.", remediation="Install the open module package for the desired driver branch."))
    if not desired.open_kernel_module and audit.module.installed_open_module is True:
        findings.append(Finding("driver.installed-open-module", Severity.ERROR, "Open on-disk NVIDIA module detected", "modinfo shows that the running kernel would load the open variant, while desired state requires the closed variant.", remediation="Install the closed module package for the desired driver branch."))
    if desired.open_kernel_module and audit.module.installed_open_module is None:
        findings.append(Finding("driver.installed-module-flavor-unknown", Severity.ERROR, "On-disk NVIDIA module flavor could not be determined", "modinfo did not prove which NVIDIA module variant is selected for the running kernel.", remediation="Install working kmod tooling and verify the NVIDIA module license metadata."))
    if not desired.open_kernel_module and audit.module.installed_open_module is None:
        findings.append(Finding("driver.installed-module-flavor-unknown", Severity.ERROR, "On-disk NVIDIA module flavor could not be determined", "modinfo did not prove which NVIDIA module variant is selected for the running kernel.", remediation="Install working kmod tooling and verify the NVIDIA module license metadata."))
    if (
        audit.module.loaded
        and audit.module.open_module is not None
        and audit.module.installed_open_module is not None
        and audit.module.open_module != audit.module.installed_open_module
    ):
        findings.append(Finding("driver.module-flavor-mismatch", Severity.ERROR, "Loaded and on-disk NVIDIA module flavors differ", "The loaded NVIDIA module variant differs from the module selected by modinfo for the running kernel.", remediation="Drain GPU users and reload the complete NVIDIA module stack."))
    if audit.nvidia_smi.returncode != 0:
        findings.append(Finding("nvidia-smi.failed", Severity.ERROR, "nvidia-smi failed", "Driver userspace cannot communicate with the NVIDIA stack.", evidence={"stdout": audit.nvidia_smi.stdout, "stderr": audit.nvidia_smi.stderr}, remediation="Repair driver/module/userspace version alignment."))
    if audit.nvml.returncode != 0:
        findings.append(Finding("nvml.failed", Severity.ERROR, "NVML library load failed", "libnvidia-ml.so.1 is unavailable or not loadable.", evidence={"stderr": audit.nvml.stderr}, remediation="Install the driver userspace libraries for the desired branch."))
    if desired.cuda_compat != "none":
        findings.append(
            Finding(
                "cuda-compat.deployment-unmodeled",
                Severity.ERROR,
                "CUDA forward-compatibility deployment is not modeled",
                "Installing a cuda-compat package alone does not configure the "
                "application loader path, and this release cannot deploy or roll "
                "back that path safely.",
                remediation="Use cuda_compat: none in this release.",
            )
        )
    if desired.container_runtime == "docker":
        if not audit.runtime.docker_installed:
            findings.append(Finding("docker.missing", Severity.ERROR, "Docker is not installed", "Desired state requires Docker as the container runtime.", remediation="Install Docker and configure the NVIDIA container toolkit."))
        if audit.docker_service_active is False:
            findings.append(
                Finding(
                    "docker.service-inactive",
                    Severity.ERROR,
                    "Docker service is inactive",
                    "Desired state requires the Docker daemon to be active so GPU containers can run.",
                    remediation="Start the Docker service and rerun verification.",
                )
            )
        elif audit.docker_service_active is None:
            findings.append(
                Finding(
                    "docker.service-state-unknown",
                    Severity.ERROR,
                    "Docker service activity is unknown",
                    "The Docker unit's active state could not be observed, so convergence cannot safely decide whether to start it.",
                    remediation="Restore systemd service-state observability and rerun doctor.",
                )
            )
        if audit.docker_service_enabled is False:
            findings.append(
                Finding(
                    "docker.service-disabled",
                    Severity.ERROR,
                    "Docker service is not enabled at boot",
                    "A reboot could leave the otherwise healthy GPU container runtime unavailable.",
                    remediation="Enable the Docker service persistently and rerun verification.",
                )
            )
        elif audit.docker_service_enabled is None:
            findings.append(
                Finding(
                    "docker.service-enablement-unknown",
                    Severity.ERROR,
                    "Docker boot enablement is unknown",
                    "The Docker unit's boot-persistent enablement could not be observed.",
                    remediation="Restore systemd enablement observability and rerun doctor.",
                )
            )
        if not audit.runtime.nvidia_container_runtime_installed:
            findings.append(Finding("container-toolkit.missing", Severity.ERROR, "NVIDIA container toolkit is missing", "Docker cannot run GPU workloads without the NVIDIA container runtime/toolkit.", remediation="Install nvidia-container-toolkit and run nvidia-ctk runtime configure."))
        if audit.runtime.docker_gpus_usable is False:
            findings.append(Finding("docker.nvidia-runtime-missing", Severity.ERROR, "Docker is not configured with the NVIDIA runtime", "docker info does not list an nvidia runtime.", remediation="Run nvidia-ctk runtime configure --runtime=docker and restart Docker."))
        if audit.runtime.docker_installed and audit.runtime.docker_gpus_usable is None:
            findings.append(Finding("docker.runtime-unknown", Severity.ERROR, "Docker runtime state could not be determined", "docker info failed, so NVIDIA runtime configuration cannot be proven.", remediation="Ensure the Docker daemon is running and accessible, then rerun doctor."))
    if desired.fabric_manager:
        if audit.fabric_manager_applicable is False:
            findings.append(Finding("fabric-manager.not-applicable", Severity.ERROR, "Fabric Manager is not applicable to the observed topology", "Every GPU reported that the Fabric Manager handshake is unsupported.", remediation="Use fabric_manager: false or deploy this profile only to a qualified NVSwitch system."))
        elif audit.fabric_manager_applicable is None:
            findings.append(Finding("fabric-manager.applicability-unknown", Severity.ERROR, "Fabric Manager topology could not be qualified", "The GPU Fabric handshake state was unavailable or malformed.", remediation="Restore `nvidia-smi -q -d FABRIC` observability before using the Fabric Manager profile."))
        if not _fabric_manager_package_installed(audit):
            findings.append(Finding("fabric-manager.missing", Severity.ERROR, "NVIDIA Fabric Manager package is missing", "This release verifies pre-provisioned Fabric Manager but does not install it automatically.", remediation="Provision the exact driver-matched Fabric Manager package using the platform vendor procedure, then rerun doctor."))
        if audit.fabric_manager_active is False:
            findings.append(Finding("fabric-manager.inactive", Severity.ERROR, "NVIDIA Fabric Manager is inactive", "Desired state requires the pre-provisioned Fabric Manager service to be active.", remediation="Validate the NVSwitch topology and start Fabric Manager using the platform vendor procedure."))
        elif audit.fabric_manager_active is None:
            findings.append(Finding("fabric-manager.unknown", Severity.ERROR, "Fabric Manager state could not be determined", "Desired state requires an active Fabric Manager service, but service state is unavailable.", remediation="Run on a systemd host and verify the nvidia-fabricmanager unit."))
        if audit.fabric_manager_enabled is False:
            findings.append(Finding("fabric-manager.disabled", Severity.ERROR, "NVIDIA Fabric Manager is not enabled at boot", "A reboot would leave the required Fabric Manager service inactive.", remediation="Enable the qualified nvidia-fabricmanager service and rerun doctor."))
        elif audit.fabric_manager_enabled is None:
            findings.append(Finding("fabric-manager.enablement-unknown", Severity.ERROR, "Fabric Manager boot enablement is unknown", "The required service's boot-persistent state could not be observed.", remediation="Restore systemd enablement observability and rerun doctor."))
        if audit.fabric_manager_healthy is False:
            findings.append(Finding("fabric-manager.fabric-unhealthy", Severity.ERROR, "GPU Fabric registration is unhealthy", "At least one GPU did not complete a successful Fabric Manager handshake.", remediation="Repair the NVSwitch/Fabric Manager topology before convergence."))
        elif audit.fabric_manager_healthy is None:
            findings.append(Finding("fabric-manager.fabric-health-unknown", Severity.ERROR, "GPU Fabric registration health is unknown", "A successful Fabric Manager handshake could not be proven for every GPU.", remediation="Restore GPU Fabric health observability and rerun doctor."))
    if desired.fabric_manager and _fabric_manager_package_installed(audit):
        if audit.fabric_manager_version is None:
            findings.append(Finding("fabric-manager.version-unknown", Severity.ERROR, "Fabric Manager version could not be determined", "Driver/Fabric Manager compatibility cannot be proven without a version.", remediation="Install a matching versioned Fabric Manager package and ensure nv-fabricmanager -v works."))
        elif not desired.matches_driver_version(audit.fabric_manager_version):
            findings.append(Finding("fabric-manager.version-mismatch", Severity.ERROR, "Fabric Manager does not match the desired driver", f"Fabric Manager {audit.fabric_manager_version} does not match desired {desired.driver_match_label}.", remediation="Install the Fabric Manager build matching the NVIDIA driver."))
        elif (
            audit.module.version is not None
            and audit.fabric_manager_version != audit.module.version
        ):
            findings.append(
                Finding(
                    "fabric-manager.version-mismatch",
                    Severity.ERROR,
                    "Fabric Manager does not exactly match the loaded driver",
                    f"Loaded NVIDIA module {audit.module.version} and Fabric Manager {audit.fabric_manager_version} must use the same version.",
                    remediation="Install the Fabric Manager build that exactly matches the loaded NVIDIA driver.",
                )
            )
    if desired.mig == "enabled" and audit.mig_capable is False:
        findings.append(Finding("mig.unsupported", Severity.ERROR, "At least one GPU does not support MIG", "MIG cannot be enabled host-wide because the GPU inventory includes a device reporting MIG N/A.", remediation="Use desired MIG disabled or target only a homogeneous MIG-capable node."))
    elif desired.mig == "enabled" and audit.mig_capable is None:
        findings.append(Finding("mig.capability-unknown", Severity.ERROR, "MIG capability could not be determined", "Every GPU must report a current and pending MIG state before host-wide enablement.", remediation="Restore nvidia-smi MIG capability queries before convergence."))
    if audit.mig_mode_pending is None:
        findings.append(Finding("mig.pending-unknown", Severity.ERROR, "Pending MIG mode could not be determined", "A pending firmware/driver MIG transition could change state after reboot and must be observable.", remediation="Use a driver that exposes current and pending MIG mode for every GPU."))
    elif audit.mig_mode is not None and audit.mig_mode_pending != audit.mig_mode:
        findings.append(Finding("mig.pending-reboot", Severity.ERROR, "A MIG mode transition is pending", f"Current MIG mode is {audit.mig_mode}, while pending mode is {audit.mig_mode_pending}.", remediation="Reboot the node during the maintenance window, then rerun doctor and applied verification."))
    if audit.mig_mode is None:
        findings.append(Finding("mig.unknown", Severity.ERROR, "MIG mode could not be determined", "Desired MIG state cannot be proven from nvidia-smi output.", remediation="Ensure nvidia-smi can query every GPU's MIG mode."))
    elif audit.mig_mode != desired.mig:
        summary = "MIG mode is mixed across GPUs" if audit.mig_mode == "mixed" else f"MIG mode is {audit.mig_mode}"
        findings.append(Finding(f"mig.{audit.mig_mode}", Severity.ERROR, summary, f"Desired state requires MIG {desired.mig} on every GPU.", remediation=f"Set MIG {desired.mig} on every GPU during a maintenance window."))
    elif audit.mig_mode_pending == audit.mig_mode:
        if not audit.mig_geometry_complete:
            findings.append(
                Finding(
                    "mig.geometry-unobservable",
                    Severity.ERROR,
                    "MIG instance geometry could not be determined",
                    "Rollback and container binding require a complete GI/CI and MIG-device inventory.",
                    remediation=(
                        "Restore UUID-bound `nvidia-smi mig -lgi`, `-lci`, and "
                        "`nvidia-smi -L` observations before convergence."
                    ),
                )
            )
        elif desired.mig == "enabled" and not full_mig_geometry_matches(
            audit.mig_geometry,
            audit.gpu_uuids,
        ):
            finding_id = (
                "mig.geometry-missing"
                if not audit.mig_geometry
                else "mig.geometry-mismatch"
            )
            findings.append(
                Finding(
                    finding_id,
                    Severity.ERROR,
                    "MIG mode does not expose the desired usable instance",
                    "Desired MIG state requires exactly one full-profile GPU instance containing one full-size compute instance.",
                    remediation=(
                        "Recreate the UUID-bound MIG geometry with GPU instance "
                        "profile 0 at placement 0 and its default compute instance."
                    ),
                )
            )
        elif desired.mig == "enabled" and len(audit.mig_device_uuids) != 1:
            findings.append(
                Finding(
                    "mig.device-unobservable",
                    Severity.ERROR,
                    "The desired MIG compute device UUID is not unique",
                    "Container verification requires exactly one retained MIG device UUID.",
                    remediation="Repair MIG enumeration before running a workload.",
                )
            )
        elif desired.mig == "disabled" and (
            audit.mig_geometry or audit.mig_device_uuids
        ):
            findings.append(
                Finding(
                    "mig.disabled-with-instances",
                    Severity.ERROR,
                    "MIG devices remain visible while MIG mode is disabled",
                    "The mode and instance observations are internally inconsistent.",
                    remediation="Reset the GPU and rerun audit before convergence.",
                )
            )
    if not findings:
        findings.append(Finding("stack.healthy", Severity.INFO, "NVIDIA stack matches observed desired-state checks", "No blocking audit issues were detected."))
    return findings


def _fabric_manager_package_installed(audit: HostAudit) -> bool:
    names = ("nvidia-fabricmanager", "nvidia-fabric-manager")
    versioned_prefixes = ("nvidia-fabricmanager-", "nvidia-fabric-manager-")
    return any(pkg.installed and (pkg.name in names or pkg.name.startswith(versioned_prefixes)) for pkg in audit.packages)
