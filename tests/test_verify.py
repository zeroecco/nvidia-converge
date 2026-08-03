import pytest

from nvidia_converge.kernel_headers import KernelHeaderReadiness
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    HostAudit,
    KernelInfo,
    MigComputeInstance,
    MigGpuInstance,
    ModuleInfo,
    RuntimeInfo,
)
from nvidia_converge.runner import CommandRunner
from nvidia_converge.verify import (
    _CUDA_DRIVER_PROBE_SHA256,
    _nvidia_smi_matches_driver,
    verify_stack,
)

_LOCAL_IMAGE_ID = "sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def _trusted_service_validation(monkeypatch):
    def validate(_runner, unit, **_kwargs):
        result = CommandResult(["validate-active-trusted-service", unit], 0)
        return [result], object(), None

    monkeypatch.setattr(
        "nvidia_converge.verify.validate_active_trusted_gpu_service_identity",
        validate,
    )


def test_verify_fails_when_secure_boot_should_be_disabled():
    checks = verify_stack(
        DesiredState(secure_boot="disabled"),
        CommandRunner(),
        _audit(secure_boot_enabled=True, module_signed=True),
    )
    policy = next(check for check in checks if check.name == "secure-boot.policy")
    assert policy.ok is False


def test_verify_uses_hardened_kernel_header_readiness(monkeypatch):
    monkeypatch.setattr(
        "nvidia_converge.verify.assess_running_kernel_headers",
        lambda *args, **kwargs: KernelHeaderReadiness(False, "sentinel failure"),
    )

    checks = verify_stack(
        DesiredState(),
        CommandRunner(),
        _audit(secure_boot_enabled=False, module_signed=True),
    )

    headers = next(check for check in checks if check.name == "kernel.headers")
    assert headers.ok is False
    assert headers.detail == "sentinel failure"


def test_verify_fails_when_secure_boot_requires_signed_module():
    checks = verify_stack(
        DesiredState(secure_boot="signed"),
        CommandRunner(),
        _audit(secure_boot_enabled=True, module_signed=False),
    )
    signed = next(
        check for check in checks if check.name == "secure-boot.module-signed"
    )
    assert signed.ok is False


def test_verify_passes_signed_module_policy_when_secure_boot_enabled():
    checks = verify_stack(
        DesiredState(secure_boot="signed"),
        CommandRunner(),
        _audit(secure_boot_enabled=True, module_signed=True),
    )
    signed = next(
        check for check in checks if check.name == "secure-boot.module-signed"
    )
    assert signed.ok is True


def test_verify_fails_when_mig_mode_does_not_match():
    checks = verify_stack(
        DesiredState(mig="enabled", mig_profile="full"),
        CommandRunner(),
        _audit(secure_boot_enabled=False, module_signed=True),
    )
    mig = next(check for check in checks if check.name == "mig.mode")
    assert mig.ok is False


def test_nvidia_smi_match_respects_exact_driver_version():
    assert (
        _nvidia_smi_matches_driver(
            DesiredState(driver="595.71.05"), "Driver Version: 595.71.05"
        )
        is True
    )
    assert (
        _nvidia_smi_matches_driver(
            DesiredState(driver="595.71.05"), "Driver Version: 595.60.01"
        )
        is False
    )
    assert (
        _nvidia_smi_matches_driver(
            DesiredState(driver="580-open"), "Driver Version: 580.126.16"
        )
        is True
    )


def test_nvidia_smi_match_rejects_unrelated_driver_branch_substring():
    output = "NVIDIA RTX 580 | Driver Version: 570.42.01 | CUDA Version: 13.0"
    assert _nvidia_smi_matches_driver(DesiredState(driver="580-open"), output) is False


def test_verify_fails_when_mig_state_is_unknown():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.mig_mode = None
    checks = verify_stack(DesiredState(), CommandRunner(), audit)
    assert next(check for check in checks if check.name == "mig.mode").ok is False


def test_verify_requires_proof_of_open_module_variant():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.module.open_module = None
    checks = verify_stack(DesiredState(), CommandRunner(), audit)
    assert (
        next(check for check in checks if check.name == "module.open-variant").ok
        is False
    )


def test_verify_rejects_loaded_and_on_disk_version_mismatch():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.module.installed_version = "595.71.05"

    checks = verify_stack(DesiredState(), CommandRunner(), audit)

    assert (
        next(check for check in checks if check.name == "module.provenance").ok is False
    )


def test_verify_rejects_loaded_and_on_disk_flavor_mismatch():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.module.installed_open_module = False

    checks = verify_stack(DesiredState(), CommandRunner(), audit)

    assert (
        next(check for check in checks if check.name == "module.flavor-provenance").ok
        is False
    )
    assert (
        next(
            check for check in checks if check.name == "module.on-disk-open-variant"
        ).ok
        is False
    )


def test_verify_requires_loaded_and_on_disk_signature_proof():
    audit = _audit(secure_boot_enabled=True, module_signed=True)
    audit.module.installed_signed = False

    checks = verify_stack(DesiredState(), CommandRunner(), audit)

    assert (
        next(check for check in checks if check.name == "secure-boot.module-signed").ok
        is True
    )
    assert (
        next(
            check
            for check in checks
            if check.name == "secure-boot.on-disk-module-signed"
        ).ok
        is False
    )


def test_verify_fails_closed_when_stack_observability_is_unknown():
    audit = _audit(secure_boot_enabled=None, module_signed=True)
    audit.module.open_module = None
    audit.mig_mode = None
    checks = {
        check.name: check
        for check in verify_stack(DesiredState(), _UnavailableRunner(), audit)
    }

    for name in (
        "secure-boot.observable",
        "module.open-variant",
        "mig.mode",
        "container.gpu",
    ):
        assert checks[name].ok is False


def test_verify_requires_observable_active_and_enabled_docker_service():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.docker_service_active = None
    audit.docker_service_enabled = False
    runner = _DockerSuccessRunner()

    checks = {
        check.name: check
        for check in verify_stack(DesiredState(), runner, audit)
    }

    assert checks["docker.service-active"].ok is False
    assert checks["docker.service-enabled"].ok is False
    assert checks["container.gpu"].ok is False
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)


def test_verify_accepts_active_boot_persistent_docker_service():
    checks = {
        check.name: check
        for check in verify_stack(
            DesiredState(),
            _DockerSuccessRunner(),
            _audit(secure_boot_enabled=False, module_signed=True),
        )
    }

    assert checks["docker.service-active"].ok is True
    assert checks["docker.service-enabled"].ok is True
    assert checks["docker.service-trust"].ok is True


def test_untrusted_docker_service_blocks_container_mutation(monkeypatch):
    runner = _DockerSuccessRunner()

    def reject(_runner, unit, **_kwargs):
        result = CommandResult(
            ["validate-active-trusted-service", unit],
            1,
            stderr="untrusted EnvironmentFiles input",
        )
        return [result], None, result.stderr

    monkeypatch.setattr(
        "nvidia_converge.verify.validate_active_trusted_gpu_service_identity",
        reject,
    )

    checks = verify_stack(
        DesiredState(),
        runner,
        _audit(secure_boot_enabled=False, module_signed=True),
    )

    trust = next(check for check in checks if check.name == "docker.service-trust")
    assert trust.ok is False
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)


def test_verify_requires_qualified_healthy_fabric_manager():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.fabric_manager_active = True
    audit.fabric_manager_enabled = True
    audit.fabric_manager_version = "580.126.16"
    audit.fabric_manager_applicable = True
    audit.fabric_manager_healthy = False

    checks = {
        check.name: check
        for check in verify_stack(
            DesiredState(fabric_manager=True), _UnavailableRunner(), audit
        )
    }

    assert checks["fabric-manager.fabric-health"].ok is False
    assert checks["fabric-manager.applicable"].ok is True
    assert checks["fabric-manager.service-trust"].ok is True


def test_verify_fails_closed_for_unmodeled_cuda_compat_deployment():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    checks = verify_stack(DesiredState(cuda_compat="13.1"), CommandRunner(), audit)

    policy = next(
        check for check in checks if check.name == "cuda-compat.deployment-policy"
    )
    assert policy.ok is False


def test_failed_container_probe_forcibly_removes_named_container():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    runner = _DockerFailureRunner()

    checks = verify_stack(
        DesiredState(fabric_manager=False),
        runner,
        audit,
    )

    container = next(check for check in checks if check.name == "container.gpu")
    assert container.ok is False
    run_command = next(
        command for command in runner.calls if command[:2] == ["docker", "run"]
    )
    name = run_command[run_command.index("--name") + 1]
    assert ["docker", "rm", "--force", name] in runner.calls
    assert next(
        check for check in checks if check.name == "container.cleanup-command"
    ).ok is True
    assert next(
        check for check in checks if check.name == "container.probe-absent"
    ).ok is True


def test_failed_container_cleanup_is_explicitly_reported():
    runner = _DockerCleanupFailureRunner()

    checks = verify_stack(
        DesiredState(fabric_manager=False),
        runner,
        _audit(secure_boot_enabled=False, module_signed=True),
    )

    assert next(
        check for check in checks if check.name == "container.cleanup-command"
    ).ok is False
    assert next(
        check for check in checks if check.name == "container.probe-absent"
    ).ok is True


def test_lingering_named_probe_container_fails_absence_proof():
    runner = _DockerLingeringRunner()

    checks = verify_stack(
        DesiredState(fabric_manager=False),
        runner,
        _audit(secure_boot_enabled=False, module_signed=True),
    )

    absence = next(
        check for check in checks if check.name == "container.probe-absent"
    )
    assert absence.ok is False
    assert absence.command is not None
    assert absence.command.stdout == "f" * 64


def test_container_probe_executes_cuda_driver_api_in_hardened_pinned_image():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    runner = _DockerSuccessRunner()
    desired = DesiredState(fabric_manager=False)

    checks = verify_stack(desired, runner, audit)

    compatibility = next(
        check for check in checks if check.name == "container.cuda-driver-compatibility"
    )
    container = next(check for check in checks if check.name == "container.gpu")
    assert compatibility.ok is True
    assert container.ok is True
    run_command = next(
        command for command in runner.calls if command[:2] == ["docker", "run"]
    )
    inspect_command = next(
        command
        for command in runner.calls
        if command[:3] == ["docker", "image", "inspect"]
    )
    assert runner.calls.index(inspect_command) < runner.calls.index(run_command)
    assert inspect_command[-1] == desired.container_test_image
    assert run_command[run_command.index("--gpus") + 1] == (
        "device=GPU-aaaaaaaaaaaaaaaa"
    )
    assert "--pull=never" in run_command
    assert desired.container_test_image in run_command
    assert "nvidia-smi" not in run_command
    assert "--network=none" in run_command
    assert "--read-only" in run_command
    assert "--cap-drop=ALL" in run_command
    assert "--security-opt=no-new-privileges" in run_command
    assert "--user=65534:65534" in run_command
    assert "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=64m,mode=1777" in run_command
    assert "NVIDIA_DRIVER_CAPABILITIES=compute" in run_command
    assert (
        f"io.nvidia-converge.cuda-probe-sha256={_CUDA_DRIVER_PROBE_SHA256}"
        in run_command
    )
    assert f"NVIDIA_CONVERGE_PROBE_SHA256={_CUDA_DRIVER_PROBE_SHA256}" in run_command
    assert "NVIDIA_CONVERGE_EXPECTED_CUDA_VERSION=13.1.2" in run_command
    script = run_command[-1]
    assert "LD_LIBRARY_PATH" in script
    assert "--cudart=none" in script
    assert "/compat/" in script
    assert "forward-compatibility libcuda.so.1" in script
    assert len(runner.inputs) == 1
    assert "cuInit(0)" in runner.inputs[0]
    assert "cuDeviceGetCount(&device_count)" in runner.inputs[0]


def test_disabled_mig_probes_every_audited_physical_gpu_uuid():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.gpu_uuids = ["GPU-aaaaaaaaaaaaaaaa", "GPU-bbbbbbbbbbbbbbbb"]
    runner = _DockerSuccessRunner()

    checks = verify_stack(DesiredState(), runner, audit)

    run_commands = [
        command for command in runner.calls if command[:2] == ["docker", "run"]
    ]
    assert [
        command[command.index("--gpus") + 1] for command in run_commands
    ] == [
        "device=GPU-aaaaaaaaaaaaaaaa",
        "device=GPU-bbbbbbbbbbbbbbbb",
    ]
    assert all("--pull=never" in command for command in run_commands)
    assert len(runner.inputs) == 2
    assert next(check for check in checks if check.name == "container.gpu").ok is True
    assert next(
        check for check in checks if check.name == "container.probe-absent"
    ).ok is True


def test_disabled_mig_aggregates_one_failed_physical_gpu_probe():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.gpu_uuids = ["GPU-aaaaaaaaaaaaaaaa", "GPU-bbbbbbbbbbbbbbbb"]
    runner = _DockerOneDeviceFailureRunner("GPU-bbbbbbbbbbbbbbbb")

    checks = verify_stack(DesiredState(), runner, audit)

    run_commands = [
        command for command in runner.calls if command[:2] == ["docker", "run"]
    ]
    container = next(check for check in checks if check.name == "container.gpu")
    assert len(run_commands) == 2
    assert container.ok is False
    assert container.detail is not None
    assert "GPU-bbbbbbbbbbbbbbbb" in container.detail
    assert next(
        check for check in checks if check.name == "container.cleanup-command"
    ).ok is True
    assert next(
        check for check in checks if check.name == "container.probe-absent"
    ).ok is True


def test_missing_digest_bound_image_blocks_every_container_run():
    runner = _DockerMissingImageRunner()
    desired = DesiredState()

    checks = verify_stack(
        desired,
        runner,
        _audit(secure_boot_enabled=False, module_signed=True),
    )

    image = next(
        check for check in checks if check.name == "container.image-present"
    )
    container = next(check for check in checks if check.name == "container.gpu")
    assert image.ok is False
    assert image.command is not None
    assert image.command.command[-1] == desired.container_test_image
    assert container.ok is False
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)
    assert not any(call[:3] == ["systemctl", "stop", "docker.service"] for call in runner.calls)


def test_mig_container_probe_is_bound_to_the_single_audited_mig_uuid():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "enabled"
    audit.mig_geometry = [
        MigGpuInstance(
            gpu_uuid="GPU-aaaaaaaaaaaaaaaa",
            profile="7g.80gb",
            profile_id=0,
            placement_start=0,
            placement_size=8,
            compute_instances=[MigComputeInstance("7c.7g.80gb", 4)],
        )
    ]
    audit.mig_device_uuids = ["MIG-bbbbbbbbbbbbbbbb"]
    runner = _DockerSuccessRunner()

    checks = verify_stack(
        DesiredState(mig="enabled", mig_profile="full"),
        runner,
        audit,
    )

    assert next(
        check for check in checks if check.name == "mig.geometry"
    ).ok is True
    assert next(
        check for check in checks if check.name == "mig.device-uuid"
    ).ok is True
    assert next(
        check for check in checks if check.name == "container.device-binding"
    ).ok is True
    run_command = next(
        command for command in runner.calls if command[:2] == ["docker", "run"]
    )
    assert run_command[run_command.index("--gpus") + 1] == (
        "device=MIG-bbbbbbbbbbbbbbbb"
    )


def test_container_probe_requires_cuda_driver_api_success_marker():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    runner = _DockerMissingMarkerRunner()

    checks = verify_stack(DesiredState(), runner, audit)

    container = next(check for check in checks if check.name == "container.gpu")
    assert container.ok is False


def test_container_probe_refuses_observed_driver_below_cuda_native_floor():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.module.version = "535.183.01"
    audit.module.installed_version = "535.183.01"
    runner = _DockerSuccessRunner()
    image = "nvidia/cuda:13.3.0-devel-ubuntu22.04@sha256:" + "a" * 64

    checks = verify_stack(
        DesiredState(driver="535-open", container_test_image=image),
        runner,
        audit,
    )

    compatibility = next(
        check for check in checks if check.name == "container.cuda-driver-compatibility"
    )
    container = next(check for check in checks if check.name == "container.gpu")
    assert compatibility.ok is False
    assert container.ok is False
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)


def test_container_probe_requires_observed_driver_version():
    audit = _audit(secure_boot_enabled=False, module_signed=True)
    audit.module.version = None
    runner = _DockerSuccessRunner()

    checks = verify_stack(DesiredState(), runner, audit)

    compatibility = next(
        check for check in checks if check.name == "container.cuda-driver-compatibility"
    )
    container = next(check for check in checks if check.name == "container.gpu")
    assert compatibility.ok is False
    assert container.ok is False
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)


def test_container_probe_source_integrity_failure_blocks_docker(tmp_path, monkeypatch):
    source = tmp_path / "cuda_driver_probe.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    monkeypatch.setattr(
        "nvidia_converge.verify._CUDA_DRIVER_PROBE_PATH",
        source,
    )
    runner = _DockerSuccessRunner()

    checks = verify_stack(
        DesiredState(),
        runner,
        _audit(secure_boot_enabled=False, module_signed=True),
    )

    container = next(check for check in checks if check.name == "container.gpu")
    assert container.ok is False
    assert container.command is not None
    assert container.command.reason == "cuda-driver-probe-source-invalid"
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)


def _audit(*, secure_boot_enabled: bool | None, module_signed: bool) -> HostAudit:
    return HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo(
            "6.8.0-test",
            headers_installed=True,
            compiler="/usr/bin/gcc",
            secure_boot_enabled=secure_boot_enabled,
        ),
        module=ModuleInfo(
            loaded=True,
            version="580.126.16",
            open_module=True,
            signed=module_signed,
            devices=["/dev/nvidia0"],
            installed_version="580.126.16",
            installed_open_module=True,
            installed_signed=module_signed,
        ),
        runtime=RuntimeInfo(
            docker_installed=False,
            nvidia_container_runtime_installed=False,
            docker_gpus_usable=None,
        ),
        packages=[],
        nvidia_smi=CommandResult(["nvidia-smi"], 0),
        nvml=CommandResult(["python3"], 0),
        fabric_manager_active=None,
        mig_mode="disabled",
        docker_service_active=True,
        docker_service_enabled=True,
        open_kernel_module_supported=True,
        mig_capable=True,
        mig_mode_pending="disabled",
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
        mig_geometry_complete=True,
    )


class _UnavailableRunner:
    def exists(self, name):
        del name
        return False

    def run(self, command, *, allow_fail=True):
        del allow_fail
        return CommandResult(command, 1, stderr="unavailable")


class _DockerFailureRunner:
    def __init__(self):
        self.calls = []
        self.inputs = []

    def exists(self, name):
        return name == "docker"

    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        del mutate, allow_fail
        self.calls.append(command)
        if input_text is not None:
            self.inputs.append(input_text)
        if command[:3] == ["docker", "image", "inspect"]:
            return CommandResult(command, 0, stdout=_LOCAL_IMAGE_ID)
        if command[:2] == ["docker", "run"]:
            return CommandResult(
                command,
                125,
                reason="lingering-process-group-terminated",
            )
        return CommandResult(command, 0)


class _DockerSuccessRunner(_DockerFailureRunner):
    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        del mutate, allow_fail
        self.calls.append(command)
        if input_text is not None:
            self.inputs.append(input_text)
        if command[:3] == ["docker", "image", "inspect"]:
            return CommandResult(command, 0, stdout=_LOCAL_IMAGE_ID)
        if command[:2] == ["docker", "run"]:
            return CommandResult(
                command,
                0,
                stdout="CUDA_DRIVER_API_OK driver_version=13010 device_count=1",
            )
        return CommandResult(command, 0)


class _DockerCleanupFailureRunner(_DockerFailureRunner):
    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        result = super().run(
            command,
            mutate=mutate,
            allow_fail=allow_fail,
            input_text=input_text,
        )
        if command[:3] == ["docker", "rm", "--force"]:
            result.returncode = 1
            result.stderr = "remove failed"
        return result


class _DockerLingeringRunner(_DockerCleanupFailureRunner):
    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        result = super().run(
            command,
            mutate=mutate,
            allow_fail=allow_fail,
            input_text=input_text,
        )
        if command[:2] == ["docker", "ps"]:
            result.stdout = "f" * 64
        return result


class _DockerMissingMarkerRunner(_DockerSuccessRunner):
    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        result = super().run(
            command,
            mutate=mutate,
            allow_fail=allow_fail,
            input_text=input_text,
        )
        if command[:2] == ["docker", "run"]:
            result.stdout = "nvidia-smi succeeded but CUDA was not initialized"
        return result


class _DockerOneDeviceFailureRunner(_DockerSuccessRunner):
    def __init__(self, failed_device):
        super().__init__()
        self.failed_device = failed_device

    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        result = super().run(
            command,
            mutate=mutate,
            allow_fail=allow_fail,
            input_text=input_text,
        )
        if command[:2] == ["docker", "run"] and command[
            command.index("--gpus") + 1
        ] == f"device={self.failed_device}":
            result.returncode = 1
            result.stdout = ""
            result.stderr = "CUDA initialization failed"
        return result


class _DockerMissingImageRunner(_DockerSuccessRunner):
    def run(
        self,
        command,
        *,
        mutate=False,
        allow_fail=True,
        input_text=None,
    ):
        if command[:3] == ["docker", "image", "inspect"]:
            del mutate, allow_fail, input_text
            self.calls.append(command)
            return CommandResult(command, 1, stderr="No such image")
        return super().run(
            command,
            mutate=mutate,
            allow_fail=allow_fail,
            input_text=input_text,
        )
