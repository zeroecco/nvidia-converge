import pytest

from nvidia_converge.doctor import diagnose
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    HostAudit,
    KernelInfo,
    ModuleInfo,
    PackageInfo,
    RuntimeInfo,
    Severity,
)


def test_secure_boot_disabled_desired_state_flags_enabled_host():
    audit = HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", headers_installed=True, compiler="/usr/bin/gcc", secure_boot_enabled=True),
        module=ModuleInfo(loaded=True, version="580.126.16", open_module=True, signed=True, devices=["/dev/nvidia0"], installed_version="580.126.16", installed_open_module=True, installed_signed=True),
        runtime=RuntimeInfo(docker_installed=True, nvidia_container_runtime_installed=True, docker_gpus_usable=True),
        packages=[
            PackageInfo("cuda-compat-13-1", "590.1-1", "apt", True),
            PackageInfo("nvidia-fabricmanager-580", manager="apt", installed=True),
        ],
        nvidia_smi=CommandResult(["nvidia-smi"], 0),
        nvml=CommandResult(["python3"], 0),
        fabric_manager_active=True,
        mig_mode="disabled",
        docker_service_active=True,
        docker_service_enabled=True,
        open_kernel_module_supported=True,
        mig_capable=True,
        mig_mode_pending="disabled",
        fabric_manager_version="580.126.16",
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
        mig_geometry_complete=True,
    )
    findings = diagnose(DesiredState(secure_boot="disabled"), audit)
    assert [finding.id for finding in findings] == ["secure-boot.enabled"]


def test_mig_enabled_desired_state_flags_disabled_host():
    audit = _healthy_audit()
    audit.mig_mode = "disabled"
    findings = diagnose(
        DesiredState(mig="enabled", mig_profile="full"), audit
    )
    assert [finding.id for finding in findings] == ["mig.disabled"]


def test_open_module_desired_rejects_pre_turing_gpu():
    audit = _healthy_audit()
    audit.open_kernel_module_supported = False

    findings = diagnose(DesiredState(), audit)

    assert [finding.id for finding in findings] == [
        "gpu.open-module-unsupported"
    ]


def test_mig_enable_rejects_any_unsupported_gpu():
    audit = _healthy_audit()
    audit.mig_capable = False

    findings = diagnose(
        DesiredState(mig="enabled", mig_profile="full"), audit
    )

    assert {finding.id for finding in findings} == {
        "mig.unsupported",
        "mig.disabled",
    }


def test_pending_mig_transition_requires_reboot():
    audit = _healthy_audit()
    audit.mig_mode_pending = "enabled"

    findings = diagnose(DesiredState(), audit)

    assert [finding.id for finding in findings] == ["mig.pending-reboot"]


def test_fabric_manager_missing_is_blocking_when_desired():
    audit = _healthy_audit()
    audit.packages = [
        package for package in audit.packages if package.name == "cuda-compat-13-1"
    ]
    findings = diagnose(DesiredState(fabric_manager=True), audit)
    assert [finding.id for finding in findings] == ["fabric-manager.missing"]


def test_non_none_cuda_compat_is_blocked_as_unmodeled_deployment():
    audit = _healthy_audit()

    findings = diagnose(DesiredState(cuda_compat="13.1"), audit)

    assert [finding.id for finding in findings] == [
        "cuda-compat.deployment-unmodeled"
    ]


def test_exact_driver_version_mismatch_is_blocking():
    audit = _healthy_audit()
    audit.module.version = "595.60.01"
    audit.module.installed_version = "595.60.01"
    audit.module.open_module = False
    audit.module.installed_open_module = False
    audit.fabric_manager_version = "595.71.05"
    findings = diagnose(DesiredState(driver="595.71.05"), audit)
    assert [finding.id for finding in findings] == ["driver.version-mismatch"]


def test_loaded_and_on_disk_module_version_mismatch_is_blocking():
    audit = _healthy_audit()
    audit.module.installed_version = "595.71.05"

    findings = {finding.id for finding in diagnose(DesiredState(), audit)}

    assert "driver.module-version-mismatch" in findings


def test_loaded_and_on_disk_module_flavor_mismatch_is_blocking():
    audit = _healthy_audit()
    audit.module.installed_open_module = False

    findings = {finding.id for finding in diagnose(DesiredState(), audit)}

    assert "driver.installed-closed-module" in findings
    assert "driver.module-flavor-mismatch" in findings


def test_unknown_on_disk_module_provenance_is_blocking():
    audit = _healthy_audit()
    audit.module.installed_version = None
    audit.module.installed_open_module = None
    audit.module.installed_signed = None

    findings = {finding.id for finding in diagnose(DesiredState(), audit)}

    assert {
        "driver.installed-version-unknown",
        "driver.installed-module-flavor-unknown",
        "secure-boot.installed-module-unsigned",
    }.issubset(findings)


@pytest.mark.parametrize(
    ("unknown_state", "expected_finding"),
    [
        ("secure-boot", "secure-boot.unknown"),
        ("module-flavor", "driver.module-flavor-unknown"),
        ("docker-runtime", "docker.runtime-unknown"),
        ("fabric-manager", "fabric-manager.unknown"),
        ("mig", "mig.unknown"),
    ],
)
def test_doctor_treats_unknown_observability_as_blocking(unknown_state, expected_finding):
    audit = _healthy_audit()
    if unknown_state == "secure-boot":
        audit.kernel.secure_boot_enabled = None
    elif unknown_state == "module-flavor":
        audit.module.open_module = None
    elif unknown_state == "docker-runtime":
        audit.runtime.docker_gpus_usable = None
    elif unknown_state == "fabric-manager":
        audit.fabric_manager_active = None
    else:
        audit.mig_mode = None

    desired = DesiredState(fabric_manager=unknown_state == "fabric-manager")
    findings = {finding.id: finding for finding in diagnose(desired, audit)}
    assert findings[expected_finding].severity is Severity.ERROR


@pytest.mark.parametrize(
    ("attribute", "value", "expected_finding"),
    [
        ("docker_service_active", False, "docker.service-inactive"),
        ("docker_service_active", None, "docker.service-state-unknown"),
        ("docker_service_enabled", False, "docker.service-disabled"),
        (
            "docker_service_enabled",
            None,
            "docker.service-enablement-unknown",
        ),
    ],
)
def test_doctor_requires_active_boot_persistent_docker_service(
    attribute,
    value,
    expected_finding,
):
    audit = _healthy_audit()
    setattr(audit, attribute, value)

    findings = {finding.id: finding for finding in diagnose(DesiredState(), audit)}

    assert findings[expected_finding].severity is Severity.ERROR


def _healthy_audit() -> HostAudit:
    return HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", headers_installed=True, compiler="/usr/bin/gcc", secure_boot_enabled=False),
        module=ModuleInfo(loaded=True, version="580.126.16", open_module=True, signed=True, devices=["/dev/nvidia0"], installed_version="580.126.16", installed_open_module=True, installed_signed=True),
        runtime=RuntimeInfo(docker_installed=True, nvidia_container_runtime_installed=True, docker_gpus_usable=True),
        packages=[
            PackageInfo("cuda-compat-13-1", "590.1-1", "apt", True),
            PackageInfo("nvidia-fabricmanager-580", manager="apt", installed=True),
        ],
        nvidia_smi=CommandResult(["nvidia-smi"], 0),
        nvml=CommandResult(["python3"], 0),
        fabric_manager_active=True,
        mig_mode="disabled",
        docker_service_active=True,
        docker_service_enabled=True,
        fabric_manager_version="580.126.16",
        fabric_manager_enabled=True,
        fabric_manager_applicable=True,
        fabric_manager_healthy=True,
        open_kernel_module_supported=True,
        mig_capable=True,
        mig_mode_pending="disabled",
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
        mig_geometry_complete=True,
    )
