from __future__ import annotations

from pathlib import Path

from .models import DesiredState, HostAudit, Verification
from .runner import CommandRunner


def verify_stack(desired: DesiredState, runner: CommandRunner, audit: HostAudit | None = None) -> list[Verification]:
    checks: list[Verification] = []
    if audit:
        checks.extend(_secure_boot_checks(desired, audit))
    checks.append(_check("kernel.headers", Path(f"/lib/modules/{_kernel_release()}/build").exists(), "Running kernel headers are present."))
    dkms = runner.run(["dkms", "status", "-m", "nvidia"], allow_fail=True) if runner.exists("dkms") else None
    if dkms:
        checks.append(Verification("module.dkms-build", dkms.returncode == 0 and "installed" in dkms.stdout.lower(), dkms, "DKMS reports an installed NVIDIA module build for the running kernel."))
    modprobe = runner.run(["modprobe", "-n", "-v", "nvidia"], allow_fail=True) if runner.exists("modprobe") else None
    checks.append(Verification("module.compile-or-loadable", bool(modprobe and modprobe.returncode == 0), modprobe, "modprobe dry-run can resolve nvidia module."))
    load = runner.run(["modprobe", "nvidia"], mutate=True, allow_fail=True) if runner.exists("modprobe") else None
    checks.append(Verification("module.load", bool(load and load.returncode == 0), load, "Loads nvidia module when --apply is used; dry-run records the command without marking it verified."))
    smi = runner.run(["nvidia-smi"], allow_fail=True) if runner.exists("nvidia-smi") else None
    checks.append(Verification("nvidia-smi", bool(smi and smi.returncode == 0 and desired.driver_major in smi.stdout), smi, f"nvidia-smi must run and show driver branch {desired.driver_major}."))
    nvml = runner.run(["python3", "-c", "import ctypes; ctypes.CDLL('libnvidia-ml.so.1'); print('NVML load ok')"], allow_fail=True)
    checks.append(Verification("nvml", nvml.returncode == 0, nvml, "NVML shared library loads."))
    if desired.container_runtime == "docker":
        image = f"nvidia/cuda:{desired.cuda_compat}.0-base-ubuntu22.04"
        docker = runner.run(["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi"], mutate=True, allow_fail=True) if runner.exists("docker") else None
        checks.append(Verification("container.gpu", bool(docker and docker.returncode == 0), docker, "Runs nvidia-smi inside a CUDA container when --apply is used; dry-run records the command without marking it verified."))
    if desired.fabric_manager:
        fm = runner.run(["systemctl", "is-active", "nvidia-fabricmanager"], allow_fail=True) if runner.exists("systemctl") else None
        checks.append(Verification("fabric-manager", bool(fm and fm.returncode == 0), fm, "Fabric Manager service is active."))
    return checks


def _secure_boot_checks(desired: DesiredState, audit: HostAudit) -> list[Verification]:
    checks: list[Verification] = []
    if desired.secure_boot == "disabled":
        checks.append(_check("secure-boot.policy", audit.kernel.secure_boot_enabled is not True, "Secure Boot must be disabled by desired policy."))
    if desired.secure_boot == "signed" and audit.kernel.secure_boot_enabled:
        checks.append(_check("secure-boot.module-signed", audit.module.signed is True, "NVIDIA module must be signed when Secure Boot is enabled."))
    return checks


def _kernel_release() -> str:
    import platform

    return platform.uname().release


def _check(name: str, ok: bool, detail: str) -> Verification:
    return Verification(name=name, ok=ok, detail=detail)
