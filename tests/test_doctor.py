from nvidia_converge.doctor import diagnose
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    HostAudit,
    KernelInfo,
    ModuleInfo,
    PackageInfo,
    RuntimeInfo,
)


def test_secure_boot_disabled_desired_state_flags_enabled_host():
    audit = HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", headers_installed=True, compiler="/usr/bin/gcc", secure_boot_enabled=True),
        module=ModuleInfo(loaded=True, version="580.126.16", open_module=True, signed=True, devices=["/dev/nvidia0"]),
        runtime=RuntimeInfo(docker_installed=True, nvidia_container_runtime_installed=True, docker_gpus_usable=True),
        packages=[PackageInfo("nvidia-fabricmanager-580", manager="apt", installed=True)],
        nvidia_smi=CommandResult(["nvidia-smi"], 0),
        nvml=CommandResult(["python3"], 0),
        fabric_manager_active=True,
        mig_mode="disabled",
    )
    findings = diagnose(DesiredState(secure_boot="disabled"), audit)
    assert [finding.id for finding in findings] == ["secure-boot.enabled"]


def test_mig_enabled_desired_state_flags_disabled_host():
    audit = _healthy_audit()
    audit.mig_mode = "disabled"
    findings = diagnose(DesiredState(mig="enabled"), audit)
    assert [finding.id for finding in findings] == ["mig.disabled"]


def test_fabric_manager_missing_is_blocking_when_desired():
    audit = _healthy_audit()
    audit.packages = []
    audit.fabric_manager_active = False
    findings = diagnose(DesiredState(), audit)
    assert [finding.id for finding in findings] == ["fabric-manager.missing"]


def test_exact_driver_version_mismatch_is_blocking():
    audit = _healthy_audit()
    audit.module.version = "595.60.01"
    findings = diagnose(DesiredState(driver="595.71.05"), audit)
    assert [finding.id for finding in findings] == ["driver.version-mismatch"]


def _healthy_audit() -> HostAudit:
    return HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", headers_installed=True, compiler="/usr/bin/gcc", secure_boot_enabled=False),
        module=ModuleInfo(loaded=True, version="580.126.16", open_module=True, signed=True, devices=["/dev/nvidia0"]),
        runtime=RuntimeInfo(docker_installed=True, nvidia_container_runtime_installed=True, docker_gpus_usable=True),
        packages=[PackageInfo("nvidia-fabricmanager-580", manager="apt", installed=True)],
        nvidia_smi=CommandResult(["nvidia-smi"], 0),
        nvml=CommandResult(["python3"], 0),
        fabric_manager_active=True,
        mig_mode="disabled",
    )
