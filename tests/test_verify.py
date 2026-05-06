from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    HostAudit,
    KernelInfo,
    ModuleInfo,
    RuntimeInfo,
)
from nvidia_converge.runner import CommandRunner
from nvidia_converge.verify import verify_stack


def test_verify_fails_when_secure_boot_should_be_disabled():
    checks = verify_stack(DesiredState(secure_boot="disabled"), CommandRunner(), _audit(secure_boot_enabled=True, module_signed=True))
    policy = next(check for check in checks if check.name == "secure-boot.policy")
    assert policy.ok is False


def test_verify_fails_when_secure_boot_requires_signed_module():
    checks = verify_stack(DesiredState(secure_boot="signed"), CommandRunner(), _audit(secure_boot_enabled=True, module_signed=False))
    signed = next(check for check in checks if check.name == "secure-boot.module-signed")
    assert signed.ok is False


def test_verify_passes_signed_module_policy_when_secure_boot_enabled():
    checks = verify_stack(DesiredState(secure_boot="signed"), CommandRunner(), _audit(secure_boot_enabled=True, module_signed=True))
    signed = next(check for check in checks if check.name == "secure-boot.module-signed")
    assert signed.ok is True


def test_verify_fails_when_mig_mode_does_not_match():
    checks = verify_stack(DesiredState(mig="enabled"), CommandRunner(), _audit(secure_boot_enabled=False, module_signed=True))
    mig = next(check for check in checks if check.name == "mig.mode")
    assert mig.ok is False


def _audit(*, secure_boot_enabled: bool, module_signed: bool) -> HostAudit:
    return HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", headers_installed=True, compiler="/usr/bin/gcc", secure_boot_enabled=secure_boot_enabled),
        module=ModuleInfo(loaded=True, version="580.126.16", open_module=True, signed=module_signed, devices=["/dev/nvidia0"]),
        runtime=RuntimeInfo(docker_installed=False, nvidia_container_runtime_installed=False, docker_gpus_usable=None),
        packages=[],
        nvidia_smi=CommandResult(["nvidia-smi"], 0),
        nvml=CommandResult(["python3"], 0),
        fabric_manager_active=None,
        mig_mode="disabled",
    )
