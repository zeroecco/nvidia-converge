import pytest

from nvidia_converge.dnf_module_transaction import dnf_module_enable_command
from nvidia_converge.doctor import diagnose
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    HostAudit,
    KernelInfo,
    MigComputeInstance,
    MigGpuInstance,
    ModuleInfo,
    PackageInfo,
    PackagePolicyInfo,
    PackagePolicySelector,
    RuntimeInfo,
)
from nvidia_converge.planner import (
    build_plan,
    lock_actions,
    package_install_operands,
    package_install_targets,
    package_policy_package_targets,
)


def test_plan_requires_policy_staging_before_previewable_install_actions():
    desired = DesiredState()
    audit = _audit()
    findings = diagnose(desired, audit)
    plan = build_plan(desired, audit, findings)
    assert [action.id for action in plan] == [
        "unsupported.package-policy-staging"
    ]
    assert "lock --apply" in (plan[0].reason or "")

    _stage_policy(audit, desired)
    plan = build_plan(desired, audit, findings)
    ids = [action.id for action in plan]
    assert "snapshot.current-state" in ids
    assert "install.packages" in ids
    assert "configure.docker-runtime" in ids
    assert "lock.apt" not in ids
    assert "verify.stack" in ids
    snapshot = next(action for action in plan if action.id == "snapshot.current-state")
    assert snapshot.commands == [["nvidia-converge", "snapshot", "--apply"]]
    install = next(action for action in plan if action.id == "install.packages")
    assert install.destructive is True
    flattened = [part for command in install.commands for part in command]
    assert "nvidia-open" in flattened
    assert not any(part.startswith("cuda-compat-") for part in flattened)


def test_lock_actions_pin_compatibility():
    locks = lock_actions(DesiredState(), _audit())
    assert locks[0].commands == [
        [
            "apt-get",
                "install",
                "-y",
                "--allow-downgrades",
                "--no-install-recommends",
                "--purge",
                "nvidia-driver-pinning-580",
        ],
    ]


def test_fabric_manager_false_omits_fabric_packages_and_service_action():
    desired = DesiredState(fabric_manager=False)
    audit = _audit()
    plan = build_plan(desired, audit, diagnose(desired, audit))
    ids = [action.id for action in plan]
    assert "enable.fabric-manager" not in ids
    flattened = [part for action in plan for command in action.commands for part in command]
    assert "nvidia-fabricmanager" not in flattened


def test_fabric_manager_false_omits_rpm_fabric_packages_from_locks():
    desired = DesiredState(fabric_manager=False)
    audit = _suse_audit()
    plan = build_plan(desired, audit, diagnose(desired, audit))
    flattened = [part for action in plan for command in action.commands for part in command]
    assert "nvidia-fabricmanager" not in flattened


def test_lock_actions_support_zypper():
    audit = _suse_audit()
    locks = lock_actions(DesiredState(), audit)
    assert locks[0].id == "lock.zypper"
    assert locks[0].commands[0][0:3] == ["zypper", "--non-interactive", "addlock"]
    assert locks[0].commands[0][-1] == "*nvidia* >= 590"


def test_zypper_install_includes_kernel_development_package():
    audit = _suse_audit()
    _stage_policy(audit, DesiredState())
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    install = next(action for action in plan if action.id == "install.packages")
    assert "nvidia-open>=580" in install.commands[0]
    assert "nvidia-open<590" in install.commands[0]
    assert "kernel-default-devel=6.4.0-150600.23.53" in install.commands[0]
    assert "--force-resolution" not in install.commands[0]


def test_zypper_uses_running_azure_kernel_variant_and_version():
    audit = _suse_audit()
    _stage_policy(audit, DesiredState())
    audit.kernel.running = "6.4.0-150600.23.53-azure"
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    install = next(action for action in plan if action.id == "install.packages")
    assert "kernel-azure-devel=6.4.0-150600.23.53" in install.commands[0]


def test_zypper_fails_closed_when_running_kernel_variant_is_unknown():
    audit = _suse_audit()
    audit.kernel.running = "6.4.0-custom"
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert plan[0].commands == []
    assert "kernel variant" in (plan[0].reason or "")


def test_rhel_8_requires_stream_staging_before_install():
    audit = _rhel_audit("8.10")
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    assert [action.id for action in plan] == [
        "unsupported.package-policy-staging"
    ]
    lock = lock_actions(DesiredState(), audit)[0]
    assert lock.commands == [
        dnf_module_enable_command(apply=True, stream="580-open"),
    ]
    _stage_policy(audit, DesiredState())
    staged_plan = build_plan(
        DesiredState(), audit, diagnose(DesiredState(), audit)
    )
    install = next(
        action for action in staged_plan if action.id == "install.packages"
    )
    assert "nvidia-open" in install.commands[0]
    assert "kernel-devel-5.14.0-427.el8.x86_64" in install.commands[0]
    assert "kernel-headers" in install.commands[0]
    assert "--allowerasing" not in install.commands[0]


def test_rhel_9_fails_closed_when_running_kernel_headers_are_missing():
    audit = _rhel_audit("9.6")
    desired = DesiredState(driver="580")
    plan = build_plan(desired, audit, diagnose(desired, audit))
    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "implicit kernel transition" in (plan[0].reason or "")


def test_rhel_9_never_plans_unversioned_kernel_development_packages():
    audit = _rhel_audit("9.6")
    audit.kernel.headers_installed = True
    desired = DesiredState(driver="580")
    lock = lock_actions(desired, audit)[0]
    assert lock.commands == [
        dnf_module_enable_command(apply=True, stream="580-dkms")
    ]
    _stage_policy(audit, desired)
    plan = build_plan(desired, audit, diagnose(desired, audit))
    install = next(action for action in plan if action.id == "install.packages")
    assert "cuda-drivers" in install.commands[0]
    assert "kernel-devel-matched" not in install.commands[0]


def test_rhel_aarch64_and_64k_kernel_recipes_fail_closed():
    audit = _rhel_audit("9.6")
    audit.kernel.running = "5.14.0-503.el9.aarch64+64k"

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "qualified only for x86_64" in (plan[0].reason or "")


def test_exact_driver_version_fails_closed_on_dnf_and_zypper():
    desired = DesiredState(driver="595.71.05")
    rhel = _rhel_audit("9.6")
    rhel.kernel.headers_installed = True
    for audit in (rhel, _suse_audit()):
        plan = build_plan(desired, audit, diagnose(desired, audit))
        assert [action.id for action in plan] == ["unsupported.package-recipe"]
        assert plan[0].commands == []
        assert "Exact NVIDIA driver versions" in (plan[0].reason or "")


def test_unsupported_distribution_release_fails_closed():
    audit = _audit()
    audit.os_version = "20.04"
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert plan[0].commands == []


def test_oracle_linux_fails_closed_until_uek_recipes_are_qualified():
    audit = _rhel_audit("9.6")
    audit.os_id = "ol"

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "No NVIDIA DNF module-stream recipe" in (plan[0].reason or "")


def test_yum_only_host_fails_closed_instead_of_using_obsolete_flat_recipe():
    audit = _rhel_audit("8.10")
    audit.package_manager = "yum"
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "require DNF module streams" in (plan[0].reason or "")


def test_debian_uses_vendor_meta_and_pinning_packages():
    audit = _audit()
    audit.os_id = "debian"
    audit.os_version = "13"
    lock = lock_actions(DesiredState(), audit)[0]
    assert lock.commands[-1][-1] == "nvidia-driver-pinning-580"
    _stage_policy(audit, DesiredState())
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    install = next(action for action in plan if action.id == "install.packages")
    assert "nvidia-open" in install.commands[-1]


def test_plan_enables_mig_when_desired():
    audit = _audit()
    audit.mig_mode = "disabled"
    desired = DesiredState(mig="enabled", mig_profile="full")
    _stage_policy(audit, desired)
    plan = build_plan(desired, audit, diagnose(desired, audit))
    action = next(action for action in plan if action.id == "enable.mig")
    assert action.destructive is True
    assert action.commands == [
        [
            "nvidia-smi",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-mig",
            "1",
        ],
        [
            "nvidia-smi",
            "mig",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-cgi",
            "0:0",
            "-C",
        ],
    ]


def test_plan_repairs_enabled_mig_without_a_compute_instance():
    audit = _healthy_audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "enabled"
    desired = DesiredState(mig="enabled", mig_profile="full")

    plan = build_plan(desired, audit, diagnose(desired, audit))

    action = next(
        action for action in plan if action.id == "configure.mig-geometry"
    )
    assert action.commands == [[
        "nvidia-smi",
        "mig",
        "-i",
        "GPU-aaaaaaaaaaaaaaaa",
        "-cgi",
        "0:0",
        "-C",
    ]]


def test_plan_destroys_compute_then_gpu_instance_before_disabling_mig():
    audit = _healthy_audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "enabled"
    audit.mig_geometry = [_full_mig_geometry()]
    audit.mig_device_uuids = ["MIG-bbbbbbbbbbbbbbbb"]

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    action = next(action for action in plan if action.id == "disable.mig")
    assert action.commands == [
        [
            "nvidia-smi",
            "mig",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-dci",
        ],
        [
            "nvidia-smi",
            "mig",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-dgi",
        ],
        [
            "nvidia-smi",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-mig",
            "0",
        ],
    ]


def test_plan_refuses_to_mutate_complex_unrestorable_mig_baseline():
    audit = _healthy_audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "enabled"
    first = _full_mig_geometry(profile="3g.40gb", profile_id=9)
    second = MigGpuInstance(
        gpu_uuid="GPU-aaaaaaaaaaaaaaaa",
        profile="3g.40gb",
        profile_id=9,
        placement_start=4,
        placement_size=4,
        compute_instances=[MigComputeInstance("3g.40gb", 2)],
    )
    audit.mig_geometry = [first, second]
    audit.mig_device_uuids = [
        "MIG-bbbbbbbbbbbbbbbb",
        "MIG-cccccccccccccccc",
    ]

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert [action.id for action in plan] == [
        "unsupported.mig-rollback-geometry"
    ]


def test_mig_mutation_fails_closed_for_multi_gpu_topology():
    audit = _healthy_audit()
    audit.gpu_uuids.append("GPU-bbbbbbbbbbbbbbbb")

    plan = build_plan(
        DesiredState(mig="enabled", mig_profile="full"),
        audit,
        diagnose(DesiredState(mig="enabled", mig_profile="full"), audit),
    )

    assert [action.id for action in plan] == [
        "unsupported.mig-transaction-scope"
    ]


def test_exact_driver_version_mismatch_fails_closed():
    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "595.60.01"
    audit.module.signed = True
    audit.module.installed_version = "595.60.01"
    audit.module.installed_open_module = False
    audit.module.installed_signed = True
    audit.kernel.headers_installed = True
    audit.kernel.compiler = "/usr/bin/gcc"
    audit.runtime.docker_installed = True
    audit.runtime.nvidia_container_runtime_installed = True
    audit.runtime.docker_gpus_usable = True
    audit.nvidia_smi = CommandResult(["nvidia-smi"], 0, stdout="Driver Version: 595.60.01")
    audit.nvml = CommandResult(["python3"], 0)
    audit.fabric_manager_active = True
    audit.fabric_manager_version = "580.126.16"
    audit.packages.append(PackageInfo("nvidia-fabricmanager-595", manager="apt", installed=True))
    desired = DesiredState(driver="595.71.05")
    plan = build_plan(desired, audit, diagnose(desired, audit))
    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "exact-version transitions" in (plan[0].reason or "")


def test_healthy_host_plan_avoids_runtime_and_service_restarts():
    audit = _healthy_audit()
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    ids = [action.id for action in plan]
    assert "install.packages" not in ids
    assert "configure.docker-runtime" not in ids
    assert "enable.fabric-manager" not in ids
    verification = next(action for action in plan if action.id == "verify.stack")
    assert verification.destructive is True
    assert "GPU-bound Docker container" in (verification.reason or "")


@pytest.mark.parametrize(
    ("active", "enabled", "command", "destructive"),
    [
        (
            False,
            False,
            ["systemctl", "enable", "docker.service"],
            False,
        ),
        (False, True, None, None),
        (
            True,
            False,
            ["systemctl", "enable", "docker.service"],
            False,
        ),
    ],
)
def test_docker_service_drift_has_minimal_remediation(
    active,
    enabled,
    command,
    destructive,
):
    audit = _healthy_audit()
    audit.docker_service_active = active
    audit.docker_service_enabled = enabled
    if not active:
        audit.runtime.docker_gpus_usable = None

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert not any(action.id.startswith("unsupported.") for action in plan)
    service = next(
        (action for action in plan if action.id == "service.docker"),
        None,
    )
    if command is None:
        assert service is None
    else:
        assert service is not None
        assert service.commands == [command]
        assert service.destructive is destructive
    if not active:
        ids = [action.id for action in plan]
        configure = next(
            action for action in plan if action.id == "configure.docker-runtime"
        )
        assert configure.commands[-1] == [
            "systemctl",
            "start",
            "docker.service",
        ]
        assert ["systemctl", "restart", "docker"] not in configure.commands
        if service is not None:
            assert ids.index("service.docker") < ids.index(
                "configure.docker-runtime"
            )


def test_docker_start_and_restart_are_ordered_after_mig_mutation():
    audit = _healthy_audit()
    audit.docker_service_active = False
    audit.runtime.docker_gpus_usable = None
    desired = DesiredState(mig="enabled", mig_profile="full")

    plan = build_plan(desired, audit, diagnose(desired, audit))

    ids = [action.id for action in plan]
    assert ids.index("enable.mig") < ids.index("configure.docker-runtime")


@pytest.mark.parametrize(
    ("attribute", "finding_id"),
    [
        ("docker_service_active", "docker.service-state-unknown"),
        ("docker_service_enabled", "docker.service-enablement-unknown"),
    ],
)
def test_docker_service_unknown_state_fails_closed(attribute, finding_id):
    audit = _healthy_audit()
    setattr(audit, attribute, None)

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert [action.id for action in plan] == ["unsupported.precondition"]
    assert finding_id in (plan[0].reason or "")


def test_non_none_cuda_compat_fails_closed_before_package_planning():
    audit = _healthy_audit()
    desired = DesiredState(cuda_compat="13.1")

    plan = build_plan(desired, audit, diagnose(desired, audit))

    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "loader deployment" in (plan[0].reason or "")


def test_drifted_component_targets_do_not_reinstall_healthy_stack():
    audit = _healthy_audit()

    audit.runtime.nvidia_container_runtime_installed = False
    assert package_install_targets(DesiredState(), audit) == [
        "nvidia-container-toolkit"
    ]

    audit.runtime.nvidia_container_runtime_installed = True
    audit.runtime.docker_installed = False
    assert package_install_targets(DesiredState(), audit) == ["docker-ce"]

    audit.runtime.docker_installed = True
    audit.kernel.headers_installed = False
    assert package_install_targets(DesiredState(), audit) == [
        "linux-headers-6.8.0-test"
    ]

    audit.kernel.headers_installed = True
    audit.kernel.compiler = None
    assert package_install_targets(DesiredState(), audit) == ["build-essential"]


def test_inactive_fabric_manager_fails_closed_without_service_mutation():
    audit = _healthy_audit()
    audit.fabric_manager_active = False
    desired = DesiredState(fabric_manager=True)

    plan = build_plan(desired, audit, diagnose(desired, audit))

    assert [action.id for action in plan] == ["unsupported.precondition"]
    assert "fabric-manager.inactive" in (plan[0].reason or "")
    assert "nvidia-fabricmanager" not in package_install_targets(
        desired, audit
    )


def test_stale_loaded_module_uses_reload_marker_without_driver_install():
    audit = _healthy_audit()
    audit.module.version = "570.1"
    audit.module.open_module = False
    audit.module.signed = False
    audit.nvidia_smi.returncode = 1

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert "nvidia-open" not in package_install_targets(DesiredState(), audit)
    marker = next(action for action in plan if action.id == "prepare.module")
    assert marker.destructive is True


def test_broken_nvml_with_exact_driver_fails_closed_instead_of_noop_install():
    audit = _healthy_audit()
    audit.nvml = CommandResult(["python3", "-I", "-c", "probe"], 1)

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert [action.id for action in plan] == ["unsupported.package-recipe"]
    assert "candidate-bound reinstall" in (plan[0].reason or "")
    assert "nvidia-open" not in package_install_targets(
        DesiredState(), audit
    )


def test_lock_actions_are_idempotent_for_observed_apt_pin():
    audit = _healthy_audit()
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "nvidia-driver-pinning-580",
            "nvidia-driver-pinning-580",
            "package",
        )
    ]

    assert lock_actions(DesiredState(), audit) == []
    assert package_policy_package_targets(DesiredState(), audit) == []


def test_apt_lock_replaces_only_observed_conflicting_pin():
    audit = _healthy_audit()
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "nvidia-driver-pinning-570",
            "nvidia-driver-pinning-570",
            "package",
        )
    ]

    actions = lock_actions(DesiredState(), audit)

    assert actions[0].commands == [
        [
            "apt-get",
            "install",
            "-y",
            "--allow-downgrades",
            "--no-install-recommends",
            "--purge",
            "nvidia-driver-pinning-580",
            "nvidia-driver-pinning-570-",
        ],
    ]
    assert package_policy_package_targets(DesiredState(), audit) == [
        "nvidia-driver-pinning-580"
    ]


def test_dnf_lock_fails_closed_on_conflicting_observed_stream():
    audit = _rhel_audit("9.6")
    audit.kernel.headers_installed = True
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "nvidia-driver",
            "nvidia-driver",
            "module",
            "stream",
            "570-open",
        )
    ]

    actions = lock_actions(DesiredState(), audit)

    assert [action.id for action in actions] == ["unsupported.package-policy"]
    assert actions[0].commands == []
    assert "fails closed" in (actions[0].reason or "")
    audit.package_policy.selectors[0].version = "580-open"
    assert lock_actions(DesiredState(), audit) == []


def test_driver_branch_and_flavor_transitions_fail_closed():
    branch_audit = _healthy_audit()
    branch_audit.module.installed_version = "570.172.08"
    branch_plan = build_plan(
        DesiredState(), branch_audit, diagnose(DesiredState(), branch_audit)
    )
    assert [action.id for action in branch_plan] == [
        "unsupported.package-recipe"
    ]
    assert "branch or exact-version transitions" in (
        branch_plan[0].reason or ""
    )

    flavor_audit = _healthy_audit()
    flavor_audit.module.installed_open_module = False
    flavor_plan = build_plan(
        DesiredState(), flavor_audit, diagnose(DesiredState(), flavor_audit)
    )
    assert [action.id for action in flavor_plan] == [
        "unsupported.package-recipe"
    ]
    assert "open/closed" in (flavor_plan[0].reason or "")


def test_zypper_lock_is_idempotent_and_conflicts_fail_closed():
    audit = _suse_audit()
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "1", "*nvidia*", "package", "ge", "590"
        )
    ]
    assert lock_actions(DesiredState(), audit) == []

    audit.package_policy.selectors[0].version = "600"
    actions = lock_actions(DesiredState(), audit)
    assert [action.id for action in actions] == ["unsupported.package-policy"]


def test_unobservable_policy_backend_fails_closed():
    audit = _healthy_audit()
    audit.package_policy.observable = False

    actions = lock_actions(DesiredState(), audit)

    assert [action.id for action in actions] == ["unsupported.package-policy"]


def test_zypper_actual_operands_are_branch_bounded():
    audit = _suse_audit()

    operands = package_install_operands(DesiredState(), audit)

    assert "nvidia-open>=580" in operands
    assert "nvidia-open<590" in operands


def test_arbitrary_cuda_compat_version_remains_fail_closed():
    audit = _healthy_audit()
    desired = DesiredState(cuda_compat="13.2")

    plan = build_plan(desired, audit, diagnose(desired, audit))

    assert [action.id for action in plan] == ["unsupported.package-recipe"]


def test_mig_only_drift_does_not_reinstall_packages():
    audit = _healthy_audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "enabled"
    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))
    ids = [action.id for action in plan]
    assert "disable.mig" in ids
    assert "install.packages" not in ids


def test_unobservable_mig_without_driver_repair_fails_closed():
    audit = _healthy_audit()
    audit.mig_mode = None

    plan = build_plan(DesiredState(), audit, diagnose(DesiredState(), audit))

    assert [action.id for action in plan] == ["unsupported.precondition"]
    assert "mig.unknown" in (plan[0].reason or "")


def test_mixed_mig_state_fails_closed_when_rollback_cannot_represent_it():
    audit = _healthy_audit()
    audit.mig_mode = "mixed"

    disabled_plan = build_plan(
        DesiredState(mig="disabled"),
        audit,
        diagnose(DesiredState(mig="disabled"), audit),
    )
    enabled_plan = build_plan(
        DesiredState(mig="enabled", mig_profile="full"),
        audit,
        diagnose(DesiredState(mig="enabled", mig_profile="full"), audit),
    )

    assert [action.id for action in disabled_plan] == [
        "unsupported.mig-rollback-baseline"
    ]
    assert [action.id for action in enabled_plan] == [
        "unsupported.mig-rollback-baseline"
    ]


def _audit() -> HostAudit:
    return HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", headers_installed=False, compiler=None, secure_boot_enabled=True),
        module=ModuleInfo(loaded=False, version=None, open_module=None, signed=False, devices=[]),
        runtime=RuntimeInfo(docker_installed=False, nvidia_container_runtime_installed=False, docker_gpus_usable=False),
        packages=[PackageInfo("nvidia-driver", manager="apt", installed=False)],
        nvidia_smi=CommandResult(["nvidia-smi"], 127, stderr="not found"),
        nvml=CommandResult(["python3"], 1, stderr="libnvidia-ml.so.1: cannot open shared object file"),
        fabric_manager_active=False,
        mig_mode="disabled",
        docker_service_active=False,
        docker_service_enabled=False,
        docker_service_unit_file_state="not-found",
        docker_socket_active=False,
        docker_socket_enabled=False,
        docker_socket_unit_file_state="not-found",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="not-found",
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="not-found",
        open_kernel_module_supported=True,
        mig_capable=True,
        mig_mode_pending="disabled",
        package_policy=PackagePolicyInfo("apt-get", True),
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
        mig_geometry_complete=True,
    )


def _full_mig_geometry(
    *,
    profile: str = "7g.80gb",
    profile_id: int = 0,
) -> MigGpuInstance:
    return MigGpuInstance(
        gpu_uuid="GPU-aaaaaaaaaaaaaaaa",
        profile=profile,
        profile_id=profile_id,
        placement_start=0,
        placement_size=8 if profile_id == 0 else 4,
        compute_instances=[
            MigComputeInstance(
                profile=profile,
                profile_id=4 if profile_id == 0 else 2,
            )
        ],
    )


def _healthy_audit() -> HostAudit:
    audit = _audit()
    audit.kernel.headers_installed = True
    audit.kernel.compiler = "/usr/bin/gcc"
    audit.kernel.secure_boot_enabled = False
    audit.module = ModuleInfo(loaded=True, version="580.126.16", open_module=True, signed=True, devices=["/dev/nvidia0"], installed_version="580.126.16", installed_open_module=True, installed_signed=True)
    audit.runtime = RuntimeInfo(docker_installed=True, nvidia_container_runtime_installed=True, docker_gpus_usable=True)
    audit.packages = [
        PackageInfo("cuda-compat-13-1", "590.1-1", "apt", True),
        PackageInfo("nvidia-fabricmanager-580", manager="apt", installed=True),
    ]
    audit.nvidia_smi = CommandResult(["nvidia-smi"], 0)
    audit.nvml = CommandResult(["python3"], 0)
    audit.fabric_manager_active = True
    audit.fabric_manager_enabled = True
    audit.fabric_manager_unit_file_state = "enabled"
    audit.docker_service_active = True
    audit.docker_service_enabled = True
    audit.docker_service_unit_file_state = "enabled"
    audit.docker_socket_active = True
    audit.docker_socket_enabled = True
    audit.docker_socket_unit_file_state = "enabled"
    audit.nvidia_persistenced_active = True
    audit.nvidia_persistenced_enabled = True
    audit.nvidia_persistenced_unit_file_state = "enabled"
    audit.fabric_manager_version = "580.126.16"
    audit.fabric_manager_applicable = True
    audit.fabric_manager_healthy = True
    audit.mig_mode = "disabled"
    return audit


def _rhel_audit(version: str) -> HostAudit:
    audit = _audit()
    audit.os_id = "rhel"
    audit.os_version = version
    audit.package_manager = "dnf"
    audit.package_policy = PackagePolicyInfo("dnf", True)
    audit.kernel.running = "5.14.0-427.el8.x86_64"
    return audit


def _suse_audit() -> HostAudit:
    audit = _audit()
    audit.os_id = "sles"
    audit.os_version = "15.6"
    audit.package_manager = "zypper"
    audit.package_policy = PackagePolicyInfo("zypper", True)
    audit.kernel.running = "6.4.0-150600.23.53-default"
    return audit


def _stage_policy(
    audit: HostAudit, desired: DesiredState
) -> None:
    if audit.package_manager == "apt-get":
        selector = (
            desired.driver
            if desired.exact_driver_version
            else desired.driver_major
        )
        name = f"nvidia-driver-pinning-{selector}"
        audit.package_policy.selectors = [
            PackagePolicySelector(name, name, "package")
        ]
    elif audit.package_manager == "dnf":
        suffix = "open" if desired.open_kernel_module else "dkms"
        audit.package_policy.selectors = [
            PackagePolicySelector(
                "nvidia-driver",
                "nvidia-driver",
                "module",
                "stream",
                f"{desired.driver_major}-{suffix}",
            )
        ]
    elif audit.package_manager == "zypper":
        upper = str((int(desired.driver_major) // 10 + 1) * 10)
        audit.package_policy.selectors = [
            PackagePolicySelector(
                "1", "*nvidia*", "package", "ge", upper
            )
        ]
