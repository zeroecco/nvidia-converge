from __future__ import annotations

from .dnf_module_transaction import dnf_module_enable_command
from .mig import (
    desired_full_mig_geometry_command,
    full_mig_geometry_matches,
    mig_geometry_destroy_commands,
    restorable_mig_geometry,
)
from .models import (
    DesiredState,
    Finding,
    HostAudit,
    PackagePolicySelector,
    PlanAction,
)

_APT_RELEASES = {
    "debian": {"12", "13"},
    "ubuntu": {"22", "24", "26"},
}
_RHEL_IDS = {"almalinux", "rhel", "rocky"}
_RHEL_RELEASES = {"8", "9"}
_SUSE_IDS = {"opensuse-leap", "sles"}
_SUSE_RELEASES = {"15", "16"}
_SUSE_KERNEL_VARIANTS = {"64k", "azure", "default"}

_PACKAGE_REMEDIABLE_FINDINGS = {
    "compiler.missing",
    "container-toolkit.missing",
    "cuda-compat.library-missing",
    "cuda-compat.library-unknown",
    "cuda-compat.library-unusable",
    "cuda-compat.package-missing",
    "docker.missing",
    "driver.closed-module",
    "driver.installed-version-unknown",
    "driver.installed-closed-module",
    "driver.installed-open-module",
    "driver.installed-module-flavor-unknown",
    "driver.loaded-version-unknown",
    "driver.module-flavor-unknown",
    "driver.module-flavor-mismatch",
    "driver.module-version-mismatch",
    "driver.open-module",
    "driver.version-mismatch",
    "kernel.headers.missing",
    "module.not-loaded",
    "nvidia-smi.failed",
    "nvml.failed",
    "secure-boot.unsigned-module",
    "secure-boot.installed-module-unsigned",
}

_AUTOMATICALLY_REMEDIABLE_FINDINGS = _PACKAGE_REMEDIABLE_FINDINGS | {
    "docker.nvidia-runtime-missing",
    "docker.service-disabled",
    "docker.service-inactive",
    "mig.disabled",
    "mig.enabled",
    "mig.geometry-mismatch",
    "mig.geometry-missing",
}


def _automatically_remediable(finding: Finding, audit: HostAudit) -> bool:
    if finding.id in _AUTOMATICALLY_REMEDIABLE_FINDINGS:
        return True
    return bool(
        finding.id == "docker.runtime-unknown"
        and audit.docker_service_active is False
    )


def build_plan(desired: DesiredState, audit: HostAudit, findings: list[Finding]) -> list[PlanAction]:
    pm = audit.package_manager
    actions: list[PlanAction] = []
    if pm is None:
        return [PlanAction("unsupported.package-manager", "Cannot converge without a supported package manager.", [], reason="No apt/dnf/yum/zypper found")]

    recipe_error = _package_recipe_error(desired, audit)
    if recipe_error:
        return [_unsupported_package_recipe(recipe_error)]
    if audit.mig_mode == "mixed":
        return [
            PlanAction(
                "unsupported.mig-rollback-baseline",
                "Cannot safely converge a mixed per-GPU MIG baseline.",
                [],
                reason=(
                    "Rollback schema v2.6 records exact UUID-bound MIG geometry, "
                    "but mixed per-GPU transitions are not transactionally qualified. "
                    "Normalize MIG mode manually before convergence."
                ),
            )
        ]
    mig_mode_change_required = bool(
        audit.mig_mode in {"enabled", "disabled"}
        and audit.mig_mode != desired.mig
    )
    mig_geometry_change_required = bool(
        desired.mig == "enabled"
        and audit.mig_mode == "enabled"
        and not full_mig_geometry_matches(
            audit.mig_geometry,
            audit.gpu_uuids,
        )
    )
    mig_change_required = mig_mode_change_required or mig_geometry_change_required
    if mig_change_required and len(audit.gpu_uuids) != 1:
        return [
            PlanAction(
                "unsupported.mig-transaction-scope",
                "Cannot safely mutate MIG mode for this GPU topology.",
                [],
                reason=(
                    "Applied MIG changes are qualified only for exactly one "
                    "observed stable GPU UUID; multi-GPU updates can partially "
                    "succeed and cannot yet be compensated transactionally."
                ),
            )
        ]
    if mig_change_required and not audit.mig_geometry_complete:
        return [
            PlanAction(
                "unsupported.mig-geometry-unobservable",
                "Cannot safely mutate MIG without a complete GI/CI baseline.",
                [],
                reason=(
                    "Applied MIG changes require complete UUID-bound GPU-instance, "
                    "compute-instance, and MIG-device observations so rollback cannot "
                    "silently lose a pre-existing partition."
                ),
            )
        ]
    if mig_change_required and not restorable_mig_geometry(
        audit.mig_geometry,
        audit.gpu_uuids,
    ):
        return [
            PlanAction(
                "unsupported.mig-rollback-geometry",
                "Cannot safely restore the observed MIG geometry.",
                [],
                reason=(
                    "Transactional MIG changes are qualified only from an empty "
                    "baseline or one GPU instance containing one full-size compute "
                    "instance. Complex GI/CI layouts must be managed with a dedicated "
                    "partition manager."
                ),
            )
        ]
    unresolved = [
        finding
        for finding in findings
        if finding.severity.value == "error"
        and not _automatically_remediable(finding, audit)
    ]
    if unresolved:
        return [
            PlanAction(
                "unsupported.precondition",
                "Cannot safely mutate the host while required preconditions are unresolved.",
                [],
                reason="; ".join(
                    f"{finding.id}: {finding.summary}" for finding in unresolved
                ),
            )
        ]

    package_targets = package_install_targets(desired, audit)
    needs_packages = bool(package_targets)
    driver_target = _driver_package_name(desired)
    module_reset_required = bool(
        audit.module.loaded
        and (
            driver_target in package_targets
            or module_reload_required(audit)
        )
    )
    if module_reset_required and audit.mig_mode == "enabled":
        if len(audit.gpu_uuids) != 1:
            return [
                PlanAction(
                    "unsupported.module-reset-mig-transaction-scope",
                    "Cannot safely reset the module under this MIG topology.",
                    [],
                    reason=(
                        "A module reset destroys MIG geometry and is qualified "
                        "only for exactly one stable GPU UUID."
                    ),
                )
            ]
        if not audit.mig_geometry_complete:
            return [
                PlanAction(
                    "unsupported.module-reset-mig-geometry-unobservable",
                    "Cannot safely reset the module without exact MIG geometry.",
                    [],
                    reason=(
                        "The current GI/CI baseline must be completely observable "
                        "before a module reset can destroy and later reconcile it."
                    ),
                )
            ]
        if not restorable_mig_geometry(audit.mig_geometry, audit.gpu_uuids):
            return [
                PlanAction(
                    "unsupported.module-reset-mig-rollback-geometry",
                    "Cannot safely restore MIG geometry after a module reset.",
                    [],
                    reason=(
                        "Only empty geometry or one full-profile GI containing "
                        "one full-size CI is transactionally qualified."
                    ),
                )
            ]
    policy_actions = (
        lock_actions(desired, audit)
        if desired.kernel_policy == "pin-compatible"
        else []
    )
    if any(action.id.startswith("unsupported.") for action in policy_actions):
        return policy_actions
    if (
        needs_packages
        and policy_actions
        and _package_policy_staging_required(package_targets)
    ):
        return [
            PlanAction(
                "unsupported.package-policy-staging",
                "Stage the desired package policy before resolving NVIDIA packages.",
                [],
                reason=(
                    "The active package policy differs from desired state, so the "
                    "planned NVIDIA transaction cannot yet be preflighted against "
                    "the policy it will use. Run `nvidia-converge lock --apply` in "
                    "the maintenance window, then re-audit and rerun install."
                ),
            )
        ]
    if needs_packages:
        actions.append(PlanAction("snapshot.current-state", "Record installed NVIDIA packages and kernel/module state before changes.", [["nvidia-converge", "snapshot", "--apply"]], reason="Required for rollback."))
        if desired.kernel_policy == "pin-compatible":
            # NVIDIA branch/version selection must be in effect before the package
            # manager resolves an unversioned nvidia-open/cuda-drivers meta-package.
            actions.extend(policy_actions)
        actions.append(
            PlanAction(
                "install.packages",
                "Install only package components that differ from observed desired state.",
                package_install_commands(desired, audit),
                destructive=True,
                reason="Drifted targets: " + ", ".join(package_targets),
            )
        )

    if desired.kernel_policy == "pin-compatible" and not needs_packages:
        actions.extend(policy_actions)

    if (
        module_reset_required
        and audit.mig_mode == "enabled"
        and audit.mig_geometry
    ):
        actions.append(
            PlanAction(
                "prepare.mig-geometry-teardown",
                "Destroy the exact UUID-bound MIG geometry before resetting the module.",
                mig_geometry_destroy_commands(audit.gpu_uuids[0]),
                destructive=True,
                reason=(
                    "A module reset destroys MIG devices; explicit teardown keeps "
                    "the transition observable and compensatable."
                ),
            )
        )

    if not audit.module.loaded or module_reset_required:
        actions.append(
            PlanAction(
                "prepare.module",
                "Load the selected on-disk NVIDIA module stack.",
                [],
                destructive=True,
                reason=(
                    "The NVIDIA module is not loaded."
                    if not audit.module.loaded
                    else (
                        "The driver package transaction or observed on-disk module "
                        "requires an exact module-stack reset."
                    )
                ),
            )
        )

    if (
        not module_reset_required
        and desired.mig == "disabled"
        and audit.mig_mode == "enabled"
    ):
        commands = [
            *(
                mig_geometry_destroy_commands(audit.gpu_uuids[0])
                if audit.mig_geometry
                else []
            ),
            ["nvidia-smi", "-i", audit.gpu_uuids[0], "-mig", "0"],
        ]
        actions.append(
            PlanAction(
                "disable.mig",
                "Destroy the UUID-bound GPU's MIG instances, then disable MIG mode.",
                commands,
                destructive=True,
                reason="Destroys the current GI/CI and may require a GPU reset.",
            )
        )
    if (
        not module_reset_required
        and desired.mig == "enabled"
        and audit.mig_mode == "disabled"
    ):
        actions.append(
            PlanAction(
                "enable.mig",
                "Enable MIG and create one full-profile GI with its default full CI.",
                [
                    ["nvidia-smi", "-i", audit.gpu_uuids[0], "-mig", "1"],
                    desired_full_mig_geometry_command(audit.gpu_uuids[0]),
                ],
                destructive=True,
                reason="May require a GPU reset and creates a new MIG device UUID.",
            )
        )
    if (
        desired.mig == "enabled"
        and not module_reset_required
        and audit.mig_mode == "enabled"
        and mig_geometry_change_required
    ):
        actions.append(
            PlanAction(
                "configure.mig-geometry",
                "Replace drifted MIG geometry with one full-profile GI and full CI.",
                [
                    *(
                        mig_geometry_destroy_commands(audit.gpu_uuids[0])
                        if audit.mig_geometry
                        else []
                    ),
                    desired_full_mig_geometry_command(audit.gpu_uuids[0]),
                ],
                destructive=True,
                reason="Destroys and recreates the UUID-bound GPU's MIG instances.",
            )
        )

    if module_reset_required:
        actions.append(
            PlanAction(
                "reconcile.mig-after-module",
                "Reobserve and reconcile MIG mode and UUID-bound geometry after module reset.",
                [],
                destructive=True,
                reason=(
                    "A module reset can destroy MIG devices, so applied execution "
                    "derives the exact MIG command only from a fresh post-reset audit."
                ),
            )
        )

    # Keep Docker stopped (when it began stopped) and avoid restarting it until
    # every package/module-adjacent and MIG mutation is complete. Starting or
    # restarting the daemon can launch queued restart-policy GPU containers;
    # the installer performs a second workload checkpoint immediately after
    # these actions and before module preparation/container verification.
    runtime_configuration_required = bool(
        desired.container_runtime == "docker"
        and audit.runtime.docker_gpus_usable is not True
    )
    if desired.container_runtime == "docker":
        service_action = _docker_service_action(
            audit,
            defer_start=runtime_configuration_required,
        )
        if service_action is not None:
            actions.append(service_action)

    if desired.container_runtime == "docker" and runtime_configuration_required:
        configure_commands = []
        if audit.docker_service_active is True:
            configure_commands.append(
                ["systemctl", "stop", "docker.service"]
            )
        configure_commands.append(
            ["nvidia-ctk", "runtime", "configure", "--runtime=docker"]
        )
        configure_commands.append(["systemctl", "start", "docker.service"])
        actions.append(
            PlanAction(
                "configure.docker-runtime",
                "Configure Docker to use the NVIDIA container runtime.",
                configure_commands,
                destructive=True,
                reason=(
                    "Configures Docker while stopped, then starts it exactly "
                    "once before an immediate GPU-workload checkpoint."
                ),
            )
        )

    actions.append(
        PlanAction(
            "verify.stack",
            "Validate module, nvidia-smi, NVML, and container GPU access.",
            [["nvidia-converge", "verify"]],
            destructive=True,
            reason=(
                "Post-convergence validation launches a temporary GPU-bound "
                "Docker container and therefore requires the same maintenance "
                "window and active-workload gate as host convergence."
            ),
        )
    )
    return actions


def _docker_service_action(
    audit: HostAudit,
    *,
    defer_start: bool = False,
) -> PlanAction | None:
    active = audit.docker_service_active
    enabled = audit.docker_service_enabled
    if defer_start:
        if enabled is False:
            return PlanAction(
                "service.docker",
                "Enable Docker persistently without starting it yet.",
                [["systemctl", "enable", "docker.service"]],
                reason=(
                    "Docker remains inactive until runtime configuration is complete."
                ),
            )
        return None
    if active is False and enabled is False:
        return PlanAction(
            "service.docker",
            "Enable Docker persistently and start it for the current boot.",
            [["systemctl", "enable", "--now", "docker.service"]],
            destructive=True,
            reason="Starting Docker can make queued containers runnable.",
        )
    if active is False:
        return PlanAction(
            "service.docker",
            "Start Docker for the current boot.",
            [["systemctl", "start", "docker.service"]],
            destructive=True,
            reason="Starting Docker can make queued containers runnable.",
        )
    if enabled is False:
        return PlanAction(
            "service.docker",
            "Enable the active Docker service persistently at boot.",
            [["systemctl", "enable", "docker.service"]],
            reason="Changes Docker's boot-persistent unit state.",
        )
    return None


def mig_reconciliation_actions(
    desired: DesiredState,
    audit: HostAudit,
) -> list[PlanAction]:
    """Plan MIG convergence from a fresh post-module-reset observation."""

    if audit.mig_mode not in {"enabled", "disabled"}:
        return [
            PlanAction(
                "unsupported.mig-state-unobservable",
                "Cannot reconcile MIG from an unknown current mode.",
                [],
                reason="Post-module-reset MIG mode was not exactly observable.",
            )
        ]
    if audit.mig_mode_pending not in {"enabled", "disabled"}:
        return [
            PlanAction(
                "unsupported.mig-pending-unobservable",
                "Cannot reconcile MIG with an unknown pending mode.",
                [],
                reason="Post-module-reset pending MIG mode was not observable.",
            )
        ]
    if audit.mig_mode != audit.mig_mode_pending:
        if audit.mig_mode_pending == desired.mig:
            # The driver has already staged the desired mode for reboot. Do not
            # mutate geometry against the still-current pre-reboot mode.
            return []
        return [
            PlanAction(
                "unsupported.mig-pending-conflict",
                "Cannot reconcile MIG across a conflicting pending transition.",
                [],
                reason=(
                    "Pending MIG mode differs from both current and desired state; "
                    "reboot or normalize it before another mutation."
                ),
            )
        ]
    mode_change = audit.mig_mode != desired.mig
    geometry_change = bool(
        desired.mig == "enabled"
        and audit.mig_mode == "enabled"
        and not full_mig_geometry_matches(
            audit.mig_geometry,
            audit.gpu_uuids,
        )
    )
    if not mode_change and not geometry_change:
        return []
    if len(audit.gpu_uuids) != 1:
        return [
            PlanAction(
                "unsupported.mig-transaction-scope",
                "Cannot safely mutate MIG mode for this GPU topology.",
                [],
                reason="MIG reconciliation is qualified for exactly one stable GPU UUID.",
            )
        ]
    if not audit.mig_geometry_complete:
        return [
            PlanAction(
                "unsupported.mig-geometry-unobservable",
                "Cannot safely reconcile MIG without complete GI/CI state.",
                [],
                reason="Post-module-reset MIG geometry is incomplete.",
            )
        ]
    if not restorable_mig_geometry(audit.mig_geometry, audit.gpu_uuids):
        return [
            PlanAction(
                "unsupported.mig-rollback-geometry",
                "Cannot safely reconcile the observed MIG geometry.",
                [],
                reason="Observed MIG geometry is outside the exact rollback model.",
            )
        ]
    gpu_uuid = audit.gpu_uuids[0]
    if desired.mig == "disabled":
        return [
            PlanAction(
                "disable.mig",
                "Destroy the UUID-bound GPU's MIG instances, then disable MIG mode.",
                [
                    *(
                        mig_geometry_destroy_commands(gpu_uuid)
                        if audit.mig_geometry
                        else []
                    ),
                    ["nvidia-smi", "-i", gpu_uuid, "-mig", "0"],
                ],
                destructive=True,
                reason="Destroys current GI/CI state and may require a GPU reset.",
            )
        ]
    if audit.mig_mode == "disabled":
        return [
            PlanAction(
                "enable.mig",
                "Enable MIG and create one full-profile GI with its default full CI.",
                [
                    ["nvidia-smi", "-i", gpu_uuid, "-mig", "1"],
                    desired_full_mig_geometry_command(gpu_uuid),
                ],
                destructive=True,
                reason="Enables MIG and creates the exact supported geometry.",
            )
        ]
    return [
        PlanAction(
            "configure.mig-geometry",
            "Replace drifted MIG geometry with one full-profile GI and full CI.",
            [
                *(
                    mig_geometry_destroy_commands(gpu_uuid)
                    if audit.mig_geometry
                    else []
                ),
                desired_full_mig_geometry_command(gpu_uuid),
            ],
            destructive=True,
            reason="Reconciles exact UUID-bound MIG geometry after module preparation.",
        )
    ]


def lock_actions(desired: DesiredState, audit: HostAudit) -> list[PlanAction]:
    pm = audit.package_manager
    recipe_error = _package_recipe_error(desired, audit)
    if recipe_error:
        return [_unsupported_package_recipe(recipe_error)]
    policy_error = _package_policy_error(audit)
    if policy_error:
        return [_unsupported_package_policy(policy_error)]
    if pm == "apt-get":
        selector_error = _apt_policy_selector_error(audit.package_policy.selectors)
        if selector_error:
            return [_unsupported_package_policy(selector_error)]
        pinning_package = _apt_pinning_package(desired)
        installed = {
            selector.name
            for selector in audit.package_policy.selectors
            if selector.name.startswith("nvidia-driver-pinning-")
        }
        conflicting = sorted(installed - {pinning_package})
        if not conflicting and pinning_package in installed:
            return []
        commands = [
            [
                "apt-get",
                "install",
                "-y",
                "--allow-downgrades",
                "--no-install-recommends",
                "--purge",
                pinning_package,
                *(f"{package}-" for package in conflicting),
            ]
        ]
        return [
            PlanAction(
                "lock.apt",
                "Select the NVIDIA driver branch or exact version with the vendor APT pinning package.",
                commands,
                destructive=True,
                reason="Apply NVIDIA's package-wide pin before resolving the driver meta-package.",
            )
        ]
    if pm == "dnf":
        stream = _dnf_module_stream(desired)
        selectors = audit.package_policy.selectors
        selector_error = _dnf_policy_selector_error(selectors)
        if selector_error:
            return [_unsupported_package_policy(selector_error)]
        current_stream = selectors[0].version if selectors else None
        if current_stream == stream:
            return []
        if current_stream is not None:
            return [
                _unsupported_package_policy(
                    "Changing an existing NVIDIA DNF module stream requires a "
                    "vendor switch-to transaction with explicitly bounded removals; "
                    "this release fails closed instead of resetting the live stream."
                )
            ]
        commands = [dnf_module_enable_command(apply=True, stream=stream)]
        return [
            PlanAction(
                "lock.rpm",
                "Select the NVIDIA DNF module stream for the desired driver branch and module flavor.",
                commands,
                destructive=True,
                reason="RHEL 8/9 use module streams to keep the driver on the selected branch.",
            )
        ]
    if pm == "zypper":
        upper_bound = _next_driver_branch(desired.driver_major)
        relevant = [
            selector
            for selector in audit.package_policy.selectors
            if "nvidia" in selector.name.lower()
        ]
        expected = PackagePolicySelector(
            identifier="",
            name="*nvidia*",
            kind="package",
            relation="ge",
            version=upper_bound,
        )
        if relevant:
            if len(relevant) == 1 and _same_zypper_policy(relevant[0], expected):
                return []
            return [
                _unsupported_package_policy(
                    "Observed NVIDIA Zypper locks conflict with the managed branch "
                    "constraint; refusing to remove or reinterpret administrator locks."
                )
            ]
        return [
            PlanAction(
                "lock.zypper",
                "Exclude NVIDIA packages newer than the desired driver branch before installation.",
                [["zypper", "--non-interactive", "addlock", f"*nvidia* >= {upper_bound}"]],
                destructive=True,
                reason="Zypper locks are exclusion rules; this keeps resolution within the desired branch.",
            )
        ]
    return []


def _needs_driver_install(desired: DesiredState, audit: HostAudit) -> bool:
    module = audit.module
    on_disk_ready = bool(
        desired.matches_driver_version(module.installed_version)
        and module.installed_open_module is desired.open_kernel_module
        and (
            desired.secure_boot != "signed"
            or module.installed_signed is True
        )
    )
    return not on_disk_ready


def module_reload_required(audit: HostAudit) -> bool:
    """Return whether a loaded module differs from observable on-disk metadata."""
    module = audit.module
    if not module.loaded:
        return False
    comparisons = (
        (module.version, module.installed_version),
        (module.open_module, module.installed_open_module),
        (module.signed, module.installed_signed),
    )
    return any(
        installed is not None and loaded != installed
        for loaded, installed in comparisons
    )


def package_install_targets(
    desired: DesiredState,
    audit: HostAudit,
) -> list[str]:
    """Return direct package names for only components observed to be drifted."""
    targets: list[str] = []
    if _needs_driver_install(desired, audit):
        targets.append(_driver_package_name(desired))
    if not audit.runtime.nvidia_container_runtime_installed:
        targets.append("nvidia-container-toolkit")
    if not audit.runtime.docker_installed:
        targets.append("docker-ce")
    if not audit.kernel.headers_installed:
        targets.extend(_kernel_build_targets(audit))
    if audit.kernel.compiler is None:
        targets.extend(
            ["build-essential"]
            if audit.package_manager == "apt-get"
            else ["gcc", "make"]
        )
    return _deduplicate(targets)


def package_policy_package_targets(
    desired: DesiredState,
    audit: HostAudit,
) -> list[str]:
    """Return policy packages that the observed lock plan can introduce."""
    if audit.package_manager != "apt-get":
        return []
    actions = lock_actions(desired, audit)
    if any(action.id.startswith("unsupported.") for action in actions):
        return []
    desired_pin = _apt_pinning_package(desired)
    installed = {selector.name for selector in audit.package_policy.selectors}
    return [desired_pin] if desired_pin not in installed and actions else []


def package_install_operands(
    desired: DesiredState,
    audit: HostAudit,
) -> list[str]:
    """Return the exact package-manager operands used by apply and preflight."""
    targets = package_install_targets(desired, audit)
    if audit.package_manager != "zypper":
        return targets
    lower_bound = desired.driver_major
    upper_bound = _next_driver_branch(lower_bound)
    operands: list[str] = []
    for target in targets:
        if target in {"cuda-drivers", "nvidia-fabricmanager", "nvidia-open"}:
            operands.extend(
                [f"{target}>={lower_bound}", f"{target}<{upper_bound}"]
            )
        else:
            operands.append(target)
    return operands


def package_install_commands(
    desired: DesiredState,
    audit: HostAudit,
) -> list[list[str]]:
    operands = package_install_operands(desired, audit)
    if not operands:
        return []
    pm = audit.package_manager
    if pm == "apt-get":
        return [
            [
                "apt-get",
                "install",
                "-y",
                "--allow-downgrades",
                "--no-install-recommends",
                *operands,
            ]
        ]
    if pm in {"dnf", "yum"}:
        return [
            [
                pm,
                "-C",
                "--setopt=install_weak_deps=False",
                "install",
                "-y",
                *operands,
            ]
        ]
    if pm == "zypper":
        return [
            [
                "zypper",
                "--non-interactive",
                "--no-refresh",
                "install",
                "--no-recommends",
                *operands,
            ]
        ]
    return []


def _driver_package_name(desired: DesiredState) -> str:
    return "nvidia-open" if desired.open_kernel_module else "cuda-drivers"


def _package_policy_staging_required(targets: list[str]) -> bool:
    return any(
        target in {"cuda-drivers", "nvidia-fabricmanager", "nvidia-open"}
        or target.startswith("cuda-compat-")
        for target in targets
    )


def _kernel_build_targets(audit: HostAudit) -> list[str]:
    pm = audit.package_manager
    if pm == "apt-get":
        return [f"linux-headers-{audit.kernel.running}"]
    if pm in {"dnf", "yum"}:
        dependencies = _rhel_build_dependencies(
            audit.kernel.running,
            audit.os_version,
        )
        return [name for name in dependencies if name not in {"gcc", "make"}]
    if pm == "zypper":
        kernel_devel = _suse_kernel_devel(audit.kernel.running)
        return [kernel_devel] if kernel_devel else []
    return []


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _package_policy_error(audit: HostAudit) -> str | None:
    policy = audit.package_policy
    if policy.backend != audit.package_manager:
        return (
            "Observed package-policy backend does not match the detected package "
            f"manager ({policy.backend!r} != {audit.package_manager!r})."
        )
    if not policy.observable:
        return (
            f"The {audit.package_manager} package-policy state could not be "
            "observed completely; refusing to mutate an unknown policy baseline."
        )
    return None


def _apt_policy_selector_error(
    selectors: list[PackagePolicySelector],
) -> str | None:
    for selector in selectors:
        if not selector.name.startswith("nvidia-driver-pinning-"):
            continue
        if (
            selector.identifier != selector.name
            or selector.kind != "package"
            or selector.relation is not None
            or selector.version is not None
            or selector.repositories
        ):
            return (
                "Observed an unsupported NVIDIA APT pin selector shape; refusing "
                "to infer package-policy intent."
            )
    return None


def _dnf_policy_selector_error(
    selectors: list[PackagePolicySelector],
) -> str | None:
    if len(selectors) > 1:
        return "Observed multiple NVIDIA DNF module selectors."
    if not selectors:
        return None
    selector = selectors[0]
    if (
        selector.identifier != "nvidia-driver"
        or selector.name != "nvidia-driver"
        or selector.kind != "module"
        or selector.relation != "stream"
        or selector.version is None
        or selector.repositories
    ):
        return (
            "Observed an unsupported NVIDIA DNF module selector shape; refusing "
            "to infer module-stream intent."
        )
    return None


def _same_zypper_policy(
    observed: PackagePolicySelector,
    expected: PackagePolicySelector,
) -> bool:
    return bool(
        observed.name == expected.name
        and observed.kind == expected.kind
        and observed.relation == expected.relation
        and observed.version == expected.version
        and observed.repositories == expected.repositories
    )


def _apt_package_names(desired: DesiredState) -> list[str]:
    driver_pkg = "nvidia-open" if desired.open_kernel_module else "cuda-drivers"
    return [driver_pkg, "nvidia-container-toolkit", "docker-ce"]


def _rpm_package_names(desired: DesiredState) -> list[str]:
    module_pkg = "nvidia-open" if desired.open_kernel_module else "cuda-drivers"
    return [module_pkg, "nvidia-container-toolkit", "docker-ce"]


def _apt_pinning_package(desired: DesiredState) -> str:
    selector = desired.driver if desired.exact_driver_version else desired.driver_major
    return f"nvidia-driver-pinning-{selector}"


def _dnf_module_stream(desired: DesiredState) -> str:
    suffix = "open" if desired.open_kernel_module else "dkms"
    return f"{desired.driver_major}-{suffix}"


def _next_driver_branch(driver_major: str) -> str:
    major = int(driver_major)
    return str((major // 10 + 1) * 10)


def _rhel_build_dependencies(kernel: str, os_version: str | None) -> list[str]:
    if _release_major(os_version) == "9":
        # RHEL 9's vendor-documented kernel-devel-matched transaction can select
        # a newer kernel. Missing running-kernel build inputs are therefore a
        # planner precondition (below), not an automatic package mutation.
        return ["gcc", "make"]
    return [f"kernel-devel-{kernel}", "kernel-headers", "gcc", "make"]


def _suse_kernel_devel(kernel: str) -> str | None:
    version, separator, variant = kernel.rpartition("-")
    if not separator or not version or variant not in _SUSE_KERNEL_VARIANTS:
        return None
    return f"kernel-{variant}-devel={version}"


def _package_recipe_error(desired: DesiredState, audit: HostAudit) -> str | None:
    pm = audit.package_manager
    os_id = (audit.os_id or "").lower()
    release = _release_major(audit.os_version)
    if desired.cuda_compat != "none":
        return (
            "CUDA forward-compatibility packages require an explicit loader "
            "deployment and workload contract that this release does not model; "
            "use cuda_compat: none."
        )
    installed_version = audit.module.installed_version
    installed_flavor = audit.module.installed_open_module
    on_disk_ready = bool(
        desired.matches_driver_version(installed_version)
        and installed_flavor is desired.open_kernel_module
        and (
            desired.secure_boot != "signed"
            or audit.module.installed_signed is True
        )
    )
    if on_disk_ready and (
        audit.nvml.returncode != 0
        or (
            audit.module.loaded
            and audit.nvidia_smi.returncode != 0
            and not module_reload_required(audit)
        )
    ):
        return (
            "The installed driver matches desired metadata but its userspace "
            "integrity checks fail. Ordinary package installation may be a no-op; "
            "candidate-bound reinstall/repair is not qualified in this release."
        )
    if installed_version is not None and not desired.matches_driver_version(
        installed_version
    ):
        return (
            "In-place NVIDIA driver branch or exact-version transitions are not "
            "qualified; use a pre-baked image or the vendor migration procedure."
        )
    if (
        installed_flavor is not None
        and installed_flavor is not desired.open_kernel_module
    ):
        return (
            "In-place NVIDIA open/closed kernel-module flavor transitions are not "
            "qualified; use a pre-baked image or the vendor migration procedure."
        )
    if pm == "apt-get":
        supported = _APT_RELEASES.get(os_id)
        if supported is None or release not in supported:
            return f"No NVIDIA APT recipe is defined for {audit.os_id or 'unknown'} {audit.os_version or 'unknown'}."
        return None
    if pm == "dnf":
        if os_id not in _RHEL_IDS or release not in _RHEL_RELEASES:
            return f"No NVIDIA DNF module-stream recipe is defined for {audit.os_id or 'unknown'} {audit.os_version or 'unknown'}."
        if not audit.kernel.running.endswith(".x86_64"):
            return (
                "NVIDIA DNF convergence is qualified only for x86_64 kernels; "
                "aarch64 and 64K-kernel dependency recipes are not yet supported."
            )
        if release == "9" and not audit.kernel.headers_installed:
            return (
                "RHEL 9 running-kernel development files must be installed before "
                "convergence; kernel-devel-matched can select a newer kernel, so "
                "this release refuses to create an implicit kernel transition."
            )
        if desired.exact_driver_version:
            return "Exact NVIDIA driver versions require candidate-aware DNF version locking, which this planner cannot safely express."
        return None
    if pm == "yum":
        return "Current RHEL 8/9 NVIDIA recipes require DNF module streams; yum-only hosts are not supported."
    if pm == "zypper":
        if os_id not in _SUSE_IDS or release not in _SUSE_RELEASES:
            return f"No NVIDIA Zypper recipe is defined for {audit.os_id or 'unknown'} {audit.os_version or 'unknown'}."
        if desired.exact_driver_version:
            return "Exact NVIDIA driver versions require candidate-aware Zypper constraints, which this planner cannot safely express."
        if _suse_kernel_devel(audit.kernel.running) is None:
            return f"Cannot derive a supported SUSE kernel variant and version from {audit.kernel.running!r}."
        return None
    return f"No NVIDIA package recipe is defined for package manager {pm or 'unknown'}."


def _release_major(version: str | None) -> str | None:
    if not version:
        return None
    return version.split(".", 1)[0]


def _unsupported_package_recipe(reason: str) -> PlanAction:
    return PlanAction(
        "unsupported.package-recipe",
        "Cannot safely converge with the detected distribution/package recipe.",
        [],
        reason=reason,
    )


def _unsupported_package_policy(reason: str) -> PlanAction:
    return PlanAction(
        "unsupported.package-policy",
        "Cannot safely converge the package-policy selector from the observed baseline.",
        [],
        reason=reason,
    )
