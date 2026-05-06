from __future__ import annotations

import platform
import re
import shutil
from pathlib import Path

from .models import CommandResult, HostAudit, KernelInfo, ModuleInfo, PackageInfo, RuntimeInfo, utc_now
from .runner import CommandRunner


def audit_host(runner: CommandRunner) -> HostAudit:
    os_id, os_version = _read_os_release()
    package_manager = detect_package_manager()
    kernel = _audit_kernel(runner)
    module = _audit_module(runner)
    runtime = _audit_runtime(runner)
    packages = _audit_packages(package_manager, runner)
    nvidia_smi = runner.run(["nvidia-smi"], allow_fail=True) if runner.exists("nvidia-smi") else CommandResult(["nvidia-smi"], 127, stderr="not found")
    nvml = _audit_nvml(runner)
    fabric_manager_active = _service_active(runner, "nvidia-fabricmanager")
    mig_mode = _mig_mode(nvidia_smi.stdout)
    return HostAudit(
        timestamp=utc_now(),
        os_id=os_id,
        os_version=os_version,
        package_manager=package_manager,
        kernel=kernel,
        module=module,
        runtime=runtime,
        packages=packages,
        nvidia_smi=nvidia_smi,
        nvml=nvml,
        fabric_manager_active=fabric_manager_active,
        mig_mode=mig_mode,
    )


def detect_package_manager() -> str | None:
    for name in ("apt-get", "dnf", "yum", "zypper"):
        if shutil.which(name):
            return name
    return None


def _read_os_release() -> tuple[str | None, str | None]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    return values.get("ID"), values.get("VERSION_ID")


def _audit_kernel(runner: CommandRunner) -> KernelInfo:
    running = platform.uname().release
    headers_installed = Path(f"/lib/modules/{running}/build").exists()
    compiler = shutil.which("gcc") or shutil.which("cc")
    sbctl = runner.run(["mokutil", "--sb-state"], allow_fail=True) if runner.exists("mokutil") else None
    secure_boot_enabled = None
    if sbctl:
        output = f"{sbctl.stdout}\n{sbctl.stderr}".lower()
        if "enabled" in output:
            secure_boot_enabled = True
        elif "disabled" in output:
            secure_boot_enabled = False
    return KernelInfo(running=running, headers_installed=headers_installed, compiler=compiler, secure_boot_enabled=secure_boot_enabled)


def _audit_module(runner: CommandRunner) -> ModuleInfo:
    loaded = Path("/sys/module/nvidia").exists()
    version = Path("/sys/module/nvidia/version").read_text(encoding="utf-8", errors="ignore").strip() if Path("/sys/module/nvidia/version").exists() else None
    modinfo = runner.run(["modinfo", "nvidia"], allow_fail=True) if runner.exists("modinfo") else None
    open_module = None
    signed = None
    if modinfo and modinfo.returncode == 0:
        text = modinfo.stdout.lower()
        if "open kernel module" in text or "license:        dual mit/gpl" in text:
            open_module = True
        if "signer:" in text:
            signer = re.search(r"^signer:\s*(.*)$", modinfo.stdout, re.MULTILINE)
            signed = bool(signer and signer.group(1).strip())
    devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    return ModuleInfo(loaded=loaded, version=version, open_module=open_module, signed=signed, devices=devices)


def _audit_runtime(runner: CommandRunner) -> RuntimeInfo:
    docker_installed = runner.exists("docker")
    nvidia_ctk = runner.exists("nvidia-ctk") or runner.exists("nvidia-container-runtime")
    usable = None
    if docker_installed:
        result = runner.run(["docker", "info", "--format", "{{json .Runtimes}}"], allow_fail=True)
        if result.returncode == 0:
            usable = "nvidia" in result.stdout
    return RuntimeInfo(docker_installed=docker_installed, nvidia_container_runtime_installed=nvidia_ctk, docker_gpus_usable=usable)


def _audit_packages(package_manager: str | None, runner: CommandRunner) -> list[PackageInfo]:
    if package_manager == "apt-get":
        result = runner.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n", "nvidia-*", "libnvidia-*", "cuda-*", "docker-ce", "nvidia-container-toolkit"], allow_fail=True)
        return _parse_dpkg_packages(result.stdout)
    elif package_manager in {"dnf", "yum", "zypper"}:
        result = runner.run(["rpm", "-qa", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\n"], allow_fail=True)
        return _parse_rpm_packages(result.stdout)
    return []


def _parse_dpkg_packages(text: str) -> list[PackageInfo]:
    packages: dict[str, PackageInfo] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        name, version = line.split("\t", 1)
        name = name.strip()
        version = version.strip()
        if _interesting_package(name):
            packages[name] = PackageInfo(name=name, version=version, manager="apt", installed=True)
    return sorted(packages.values(), key=lambda pkg: pkg.name)


def _parse_rpm_packages(text: str) -> list[PackageInfo]:
    packages: dict[str, PackageInfo] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        name, version = line.split("\t", 1)
        name = name.strip()
        version = version.strip()
        if _interesting_package(name):
            packages[name] = PackageInfo(name=name, version=version, manager="rpm", installed=True)
    return sorted(packages.values(), key=lambda pkg: pkg.name)


def _interesting_package(name: str) -> bool:
    prefixes = ("nvidia", "libnvidia", "cuda", "docker-ce")
    return name.startswith(prefixes) or name == "nvidia-container-toolkit"


def _audit_nvml(runner: CommandRunner) -> CommandResult:
    code = "import ctypes; ctypes.CDLL('libnvidia-ml.so.1'); print('NVML load ok')"
    return runner.run(["python3", "-c", code], allow_fail=True)


def _service_active(runner: CommandRunner, service: str) -> bool | None:
    if not runner.exists("systemctl"):
        return None
    result = runner.run(["systemctl", "is-active", service], allow_fail=True)
    if result.returncode == 0:
        return True
    if result.stdout.strip() in {"inactive", "failed", "unknown"} or result.returncode:
        return False
    return None


def _mig_mode(nvidia_smi_output: str) -> str | None:
    lowered = nvidia_smi_output.lower()
    if "mig mode" not in lowered:
        return None
    if re.search(r"mig mode\s*:\s*enabled", lowered):
        return "enabled"
    if re.search(r"mig mode\s*:\s*disabled", lowered):
        return "disabled"
    return None
