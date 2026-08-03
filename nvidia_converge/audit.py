from __future__ import annotations

import json
import os
import platform
import re
import shlex
from configparser import ConfigParser
from configparser import Error as ConfigParserError
from pathlib import Path

from .cuda_compat import (
    cuda_compat_library_path,
    cuda_compat_version_from_package,
    probe_cuda_compat_library,
)
from .files import BoundedFileError, read_bounded_utf8, read_root_controlled_utf8
from .kernel_headers import assess_running_kernel_headers
from .models import (
    CommandResult,
    CudaCompatibilityInfo,
    HostAudit,
    KernelInfo,
    MigComputeInstance,
    MigGpuInstance,
    ModuleInfo,
    PackageInfo,
    PackagePolicyInfo,
    PackagePolicySelector,
    RuntimeInfo,
    utc_now,
)
from .runner import CommandRunner
from .xmlsafe import SafeXmlError, parse_bounded_xml

EFI_FIRMWARE_PATH = Path("/sys/firmware/efi")
EFI_VARIABLE_PATH = EFI_FIRMWARE_PATH / "efivars"
MAX_MODULE_METADATA_BYTES = 64 * 1024
MAX_PACKAGE_POLICY_BYTES = 64 * 1024
MAX_OS_RELEASE_BYTES = 64 * 1024
DNF_NVIDIA_MODULE_PATH = Path("/etc/dnf/modules.d/nvidia-driver.module")
_OUTPUT_TRUNCATED = "[output truncated:"
_NVIDIA_SMI_XML_DOCTYPE = re.compile(
    r'<!DOCTYPE\s+nvidia_smi_log\s+SYSTEM\s+"nvsmi_device_v\d+\.dtd"\s*>'
)
_RESTORABLE_UNIT_FILE_STATES = frozenset(
    {"enabled", "disabled", "static", "masked"}
)
_DPKG_INSTALLED_STATUS_ABBREVIATIONS = frozenset({"ii ", "hi "})
_DPKG_STABLE_STATUS_ABBREVIATIONS = _DPKG_INSTALLED_STATUS_ABBREVIATIONS
_ZYPPER_RELATION_NAMES = {
    "=": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}


def audit_host(runner: CommandRunner) -> HostAudit:
    os_id, os_version = _read_os_release()
    package_manager = detect_package_manager(runner)
    kernel = _audit_kernel(runner, package_manager)
    module = _audit_module(runner)
    # Observe systemd before touching the Docker client.  On distributions
    # where docker.socket is enabled, even a read-only `docker info` would
    # otherwise start the daemon (and any queued restart-policy containers).
    (
        docker_socket_active,
        docker_socket_enabled,
        docker_socket_unit_file_state,
    ) = _service_state(runner, "docker.socket")
    (
        docker_service_active,
        docker_service_enabled,
        docker_service_unit_file_state,
    ) = _service_state(runner, "docker.service")
    runtime = _audit_runtime(runner, docker_service_active)
    packages, inventory_complete, inventory_result = _audit_packages(
        package_manager, runner, kernel.running
    )
    package_policy = _audit_package_policy(
        package_manager,
        packages,
        inventory_complete,
        inventory_result,
        runner,
    )
    cuda_compatibility = _audit_cuda_compatibility(packages, runner)
    nvidia_smi = runner.run(["nvidia-smi"], allow_fail=True) if runner.exists("nvidia-smi") else CommandResult(["nvidia-smi"], 127, stderr="not found")
    nvml = _audit_nvml(runner)
    (
        nvidia_persistenced_active,
        nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state,
    ) = _service_state(runner, "nvidia-persistenced.service")
    (
        fabric_manager_active,
        fabric_manager_enabled,
        fabric_manager_unit_file_state,
    ) = _service_state(runner, "nvidia-fabricmanager.service")
    fabric_manager_version = _fabric_manager_version(runner, packages)
    mig_mode, mig_mode_pending, mig_capable, gpu_uuids = _audit_mig_state(
        runner,
        nvidia_smi.stdout,
    )
    (
        fabric_manager_applicable,
        fabric_manager_healthy,
        fabric_manager_health_result,
    ) = _audit_fabric_manager_health(runner, gpu_uuids)
    (
        mig_geometry,
        mig_device_uuids,
        mig_geometry_complete,
        mig_geometry_results,
    ) = _audit_mig_geometry(
        runner,
        mig_mode,
        mig_mode_pending,
        gpu_uuids,
    )
    open_kernel_module_supported = _audit_open_module_support(runner)
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
        package_inventory_complete=inventory_complete,
        package_inventory_result=inventory_result,
        docker_service_active=docker_service_active,
        docker_service_enabled=docker_service_enabled,
        docker_service_unit_file_state=docker_service_unit_file_state,
        docker_socket_active=docker_socket_active,
        docker_socket_enabled=docker_socket_enabled,
        docker_socket_unit_file_state=docker_socket_unit_file_state,
        nvidia_persistenced_active=nvidia_persistenced_active,
        nvidia_persistenced_enabled=nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state=(
            nvidia_persistenced_unit_file_state
        ),
        fabric_manager_enabled=fabric_manager_enabled,
        fabric_manager_unit_file_state=fabric_manager_unit_file_state,
        fabric_manager_version=fabric_manager_version,
        cuda_compatibility=cuda_compatibility,
        package_policy=package_policy,
        open_kernel_module_supported=open_kernel_module_supported,
        mig_capable=mig_capable,
        mig_mode_pending=mig_mode_pending,
        fabric_manager_applicable=fabric_manager_applicable,
        fabric_manager_healthy=fabric_manager_healthy,
        fabric_manager_health_result=fabric_manager_health_result,
        gpu_uuids=gpu_uuids,
        mig_geometry=mig_geometry,
        mig_device_uuids=mig_device_uuids,
        mig_geometry_complete=mig_geometry_complete,
        mig_geometry_results=mig_geometry_results,
    )


def detect_package_manager(runner: CommandRunner) -> str | None:
    for name in ("apt-get", "dnf", "yum", "zypper"):
        if runner.exists(name):
            return name
    return None


def _read_os_release(
    path: Path = Path("/etc/os-release"),
    *,
    required_owner_uid: int = 0,
) -> tuple[str | None, str | None]:
    try:
        text = read_root_controlled_utf8(
            path,
            max_bytes=MAX_OS_RELEASE_BYTES,
            required_owner_uid=required_owner_uid,
        )
    except (OSError, BoundedFileError):
        return None, None
    return _parse_os_release(text)


def _parse_os_release(text: str) -> tuple[str | None, str | None]:
    if "\x00" in text:
        return None, None
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, encoded_value = line.partition("=")
        if (
            not separator
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None
            or key in values
        ):
            return None, None
        lexer = shlex.shlex(encoded_value, posix=True)
        lexer.commenters = "#"
        lexer.whitespace_split = True
        try:
            decoded = list(lexer)
        except ValueError:
            return None, None
        if len(decoded) > 1:
            return None, None
        values[key] = decoded[0] if decoded else ""

    os_id = values.get("ID")
    os_version = values.get("VERSION_ID")
    if os_id is not None and re.fullmatch(r"[a-z0-9._-]+", os_id) is None:
        return None, None
    if os_version is not None and re.fullmatch(
        r"[A-Za-z0-9._+-]+", os_version
    ) is None:
        return None, None
    return os_id, os_version


def _audit_kernel(
    runner: CommandRunner, package_manager: str | None = None
) -> KernelInfo:
    running = platform.uname().release
    headers_installed = assess_running_kernel_headers(
        runner,
        release=running,
        package_manager=package_manager,
    ).ready
    compiler = runner.resolve_executable("gcc") or runner.resolve_executable("cc")
    secure_boot_enabled = _secure_boot_state(runner)
    return KernelInfo(running=running, headers_installed=headers_installed, compiler=compiler, secure_boot_enabled=secure_boot_enabled)


def _secure_boot_state(runner: CommandRunner) -> bool | None:
    if not EFI_FIRMWARE_PATH.exists():
        return False
    if runner.exists("mokutil"):
        result = runner.run(["mokutil", "--sb-state"], allow_fail=True)
        if result.returncode == 0:
            output = f"{result.stdout}\n{result.stderr}".lower()
            if re.search(r"\bsecureboot\s+enabled\b", output):
                return True
            if re.search(r"\bsecureboot\s+disabled\b", output):
                return False
    try:
        variables = list(EFI_VARIABLE_PATH.glob("SecureBoot-*"))
    except OSError:
        return None
    for variable in variables:
        try:
            data = variable.read_bytes()
        except OSError:
            continue
        if len(data) >= 5 and data[4] in {0, 1}:
            return data[4] == 1
    return None


def _audit_module(runner: CommandRunner) -> ModuleInfo:
    module_path = Path("/sys/module/nvidia")
    loaded = module_path.exists()
    version = _read_module_version(module_path / "version")
    modinfo = runner.run(["modinfo", "nvidia"], allow_fail=True) if runner.exists("modinfo") else None
    open_module = _loaded_module_flavor(Path("/proc/driver/nvidia/version")) if loaded else None
    signed = _loaded_module_signed(module_path / "taint") if loaded else None
    installed_version = None
    installed_open_module = None
    installed_signed = None
    if modinfo and modinfo.returncode == 0:
        text = modinfo.stdout.lower()
        if "open kernel module" in text or "license:        dual mit/gpl" in text:
            installed_open_module = True
        elif re.search(r"^license:\s*nvidia\s*$", modinfo.stdout, re.MULTILINE | re.IGNORECASE):
            installed_open_module = False
        if "signer:" in text:
            signer = re.search(r"^signer:\s*(.*)$", modinfo.stdout, re.MULTILINE)
            installed_signed = bool(signer and signer.group(1).strip())
        installed_version = _modinfo_version(modinfo.stdout)
    devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    return ModuleInfo(
        loaded=loaded,
        version=version,
        open_module=open_module,
        signed=signed,
        devices=devices,
        installed_version=installed_version,
        installed_open_module=installed_open_module,
        installed_signed=installed_signed,
    )


def _read_module_version(path: Path) -> str | None:
    try:
        value = _read_kernel_metadata(path).strip()
    except OSError:
        return None
    return value or None


def _modinfo_version(output: str) -> str | None:
    match = re.search(
        r"^version:\s*(\S+)\s*$", output, re.MULTILINE | re.IGNORECASE
    )
    return match.group(1) if match else None


def _loaded_module_flavor(path: Path) -> bool | None:
    try:
        output = _read_kernel_metadata(path)
    except OSError:
        return None
    lowered = output.lower()
    if "nvidia unix open kernel module" in lowered:
        return True
    if "nvidia unix" in lowered and "kernel module" in lowered:
        return False
    return None


def _loaded_module_signed(path: Path) -> bool | None:
    try:
        taints = _read_kernel_metadata(path).strip()
    except OSError:
        return None
    # Linux marks a module whose signature could not be verified with E
    # (TAINT_UNSIGNED_MODULE). Other taints, such as O/P, do not imply an
    # unverified signature.
    return "E" not in taints


def _read_kernel_metadata(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(
                descriptor,
                min(16 * 1024, MAX_MODULE_METADATA_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_MODULE_METADATA_BYTES:
                raise OSError(
                    f"module metadata exceeds {MAX_MODULE_METADATA_BYTES} bytes"
                )
    finally:
        os.close(descriptor)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _audit_runtime(
    runner: CommandRunner,
    docker_service_active: bool | None,
) -> RuntimeInfo:
    docker_installed = runner.exists("docker")
    nvidia_ctk = runner.exists("nvidia-ctk") or runner.exists("nvidia-container-runtime")
    usable = None
    if docker_installed and docker_service_active is True:
        result = runner.run(["docker", "info", "--format", "{{json .Runtimes}}"], allow_fail=True)
        if result.returncode == 0:
            try:
                runtimes = json.loads(result.stdout)
            except json.JSONDecodeError:
                usable = False
            else:
                usable = isinstance(runtimes, dict) and "nvidia" in runtimes
    return RuntimeInfo(docker_installed=docker_installed, nvidia_container_runtime_installed=nvidia_ctk, docker_gpus_usable=usable)


def _audit_packages(
    package_manager: str | None, runner: CommandRunner, kernel: str
) -> tuple[list[PackageInfo], bool, CommandResult | None]:
    if package_manager == "apt-get":
        result = runner.run(
            [
                "dpkg-query",
                "-W",
                "-f=${db:Status-Abbrev}\t${Package}\t${Version}\t${Architecture}\n",
                "nvidia-*",
                "libnvidia-*",
                "cuda-*",
                "docker-ce*",
                "containerd.io",
                "docker-buildx-plugin",
                "docker-compose-plugin",
                "nvidia-container-toolkit",
                f"linux-headers-{kernel}",
                "build-essential",
            ],
            allow_fail=True,
        )
        packages, rows_complete = _parse_dpkg_package_rows(result.stdout)
        complete = _dpkg_query_complete(result) and rows_complete
        return packages, complete, result
    elif package_manager in {"dnf", "yum", "zypper"}:
        result = runner.run(
            ["rpm", "-qa", "--qf", "%{NAME}\t%{EPOCHNUM}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n"],
            allow_fail=True,
        )
        packages, rows_complete = _parse_rpm_package_rows(result.stdout)
        complete = (
            result.returncode == 0
            and "[output truncated:" not in result.stdout
            and "[output truncated:" not in result.stderr
            and rows_complete
        )
        return packages, complete, result
    return [], False, None


def _dpkg_query_complete(result: CommandResult) -> bool:
    if "[output truncated:" in result.stdout or "[output truncated:" in result.stderr:
        return False
    if result.returncode == 0:
        return True
    if result.returncode != 1:
        return False
    errors = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return bool(errors) and all(
        line.startswith("dpkg-query: no packages found matching pattern ") for line in errors
    )


def _parse_dpkg_packages(text: str) -> list[PackageInfo]:
    return _parse_dpkg_package_rows(text)[0]


def _parse_dpkg_package_rows(text: str) -> tuple[list[PackageInfo], bool]:
    packages: dict[tuple[str, str], PackageInfo] = {}
    seen_slots: set[tuple[str, str]] = set()
    complete = True
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            complete = False
            continue
        status, name, version, architecture = parts
        if (
            re.fullmatch(r"[A-Za-z?][A-Za-z?][A-Za-z? ]", status) is None
            or not _valid_package_inventory_field(name)
            or not _valid_package_inventory_field(version)
            or not _valid_package_inventory_field(architecture)
        ):
            complete = False
            continue
        slot = (name, architecture)
        if slot in seen_slots:
            complete = False
            continue
        seen_slots.add(slot)
        if status not in _DPKG_STABLE_STATUS_ABBREVIATIONS:
            # `${db:Status-Abbrev}` encodes desired state, current state, and
            # an error flag. Rollback deltas are trustworthy only for the
            # stable installed shapes modeled here. Residual-config (`rc`),
            # unpacked, half-configured, half-installed, and trigger-pending
            # states (or any error flag) must make the whole inventory
            # incomplete: snapshots cannot yet restore those package-database
            # states exactly.
            complete = False
            continue
        if _interesting_package(name):
            packages[slot] = PackageInfo(
                name=name,
                version=version,
                manager="apt",
                installed=True,
                architecture=architecture,
            )
    sorted_packages = sorted(
        packages.values(),
        key=lambda pkg: (
            pkg.name,
            pkg.architecture or "",
            pkg.epoch or "",
            pkg.version or "",
        ),
    )
    return sorted_packages, complete


def _audit_package_policy(
    package_manager: str | None,
    packages: list[PackageInfo],
    inventory_complete: bool,
    inventory_result: CommandResult | None,
    runner: CommandRunner,
) -> PackagePolicyInfo:
    if package_manager == "apt-get":
        selectors = [
            PackagePolicySelector(
                identifier=package.name,
                name=package.name,
                kind="package",
            )
            for package in packages
            if package.installed
            and package.name.startswith("nvidia-driver-pinning-")
        ]
        return PackagePolicyInfo(
            backend=package_manager,
            observable=inventory_complete,
            selectors=selectors,
            observation=inventory_result,
        )
    if package_manager == "dnf":
        return _audit_dnf_module_policy()
    if package_manager == "zypper":
        command = ["zypper", "--xmlout", "--non-interactive", "locks"]
        if not runner.exists("zypper"):
            return PackagePolicyInfo(package_manager, False)
        result = runner.run(command, allow_fail=True)
        zypper_selectors = _parse_zypper_policy_selectors(result)
        return PackagePolicyInfo(
            backend=package_manager,
            observable=zypper_selectors is not None,
            selectors=zypper_selectors or [],
            observation=result,
        )
    return PackagePolicyInfo(package_manager, False)


def _audit_dnf_module_policy(
    path: Path = DNF_NVIDIA_MODULE_PATH,
) -> PackagePolicyInfo:
    try:
        text = read_bounded_utf8(path, max_bytes=MAX_PACKAGE_POLICY_BYTES)
    except FileNotFoundError:
        return PackagePolicyInfo("dnf", True)
    except (OSError, BoundedFileError):
        return PackagePolicyInfo("dnf", False)
    parser = ConfigParser(interpolation=None, strict=True)
    try:
        parser.read_string(text)
    except ConfigParserError:
        return PackagePolicyInfo("dnf", False)
    if parser.sections() != ["nvidia-driver"]:
        return PackagePolicyInfo("dnf", False)
    section = parser["nvidia-driver"]
    if section.get("name", "nvidia-driver") != "nvidia-driver":
        return PackagePolicyInfo("dnf", False)
    state = section.get("state", "").strip().lower()
    if state in {"disabled", "reset"}:
        return PackagePolicyInfo("dnf", True)
    stream = section.get("stream", "").strip()
    if state != "enabled" or not re.fullmatch(r"\d+-(?:open|dkms)", stream):
        return PackagePolicyInfo("dnf", False)
    return PackagePolicyInfo(
        "dnf",
        True,
        selectors=[
            PackagePolicySelector(
                identifier="nvidia-driver",
                name="nvidia-driver",
                kind="module",
                relation="stream",
                version=stream,
            )
        ],
    )


def _parse_zypper_policy_selectors(
    result: CommandResult,
) -> list[PackagePolicySelector] | None:
    if (
        result.returncode != 0
        or "[output truncated:" in result.stdout
        or "[output truncated:" in result.stderr
    ):
        return None
    try:
        root = parse_bounded_xml(result.stdout)
    except SafeXmlError:
        return None
    locks_nodes = list(root.iter("locks"))
    if len(locks_nodes) != 1:
        return None
    locks = locks_nodes[0]
    lock_nodes = list(locks.findall("lock"))
    try:
        declared_size = int(locks.attrib["size"])
    except (KeyError, ValueError):
        return None
    if declared_size != len(lock_nodes):
        return None

    selectors: list[PackagePolicySelector] = []
    seen_identifiers: set[str] = set()
    for lock in lock_nodes:
        identifier = lock.attrib.get("number", "")
        if not identifier.isdigit() or int(identifier) <= 0 or identifier in seen_identifiers:
            return None
        seen_identifiers.add(identifier)
        names = [node.text or "" for node in lock.findall("name")]
        kinds = [node.text or "" for node in lock.findall("type")]
        repositories = [node.text or "" for node in lock.findall("repo")]
        ranges = lock.findall("range")
        if not names or any(not name for name in names):
            return None
        if any("nvidia" in name.lower() for name in names) and (
            len(names) != 1 or len(kinds) != 1 or len(ranges) > 1
        ):
            return None
        relation = None
        version = None
        if ranges:
            relation = _ZYPPER_RELATION_NAMES.get(
                ranges[0].attrib.get("flag", "")
            )
            version = ranges[0].attrib.get("version")
            if (
                not relation
                or not version
                or ranges[0].attrib.get("epoch", "")
                or ranges[0].attrib.get("release", "")
            ):
                return None
        for name in names:
            selectors.append(
                PackagePolicySelector(
                    identifier=identifier,
                    name=name,
                    kind=kinds[0] if len(kinds) == 1 else "",
                    relation=relation,
                    version=version,
                    repositories=repositories,
                )
            )
    return selectors


def _parse_rpm_packages(text: str) -> list[PackageInfo]:
    return _parse_rpm_package_rows(text)[0]


def _parse_rpm_package_rows(text: str) -> tuple[list[PackageInfo], bool]:
    packages: dict[
        tuple[str, str | None, str | None, str | None], PackageInfo
    ] = {}
    seen_identities: set[tuple[str, str, str | None, str]] = set()
    complete = True
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            complete = False
            continue
        name, epoch_text, version, architecture = parts
        if (
            not _valid_package_inventory_field(name)
            or re.fullmatch(r"(?:[0-9]+|\(none\))", epoch_text) is None
            or not _valid_package_inventory_field(version)
            or not _valid_package_inventory_field(architecture)
        ):
            complete = False
            continue
        epoch = epoch_text if epoch_text not in {"0", "(none)"} else None
        identity = (name, architecture, epoch, version)
        if identity in seen_identities:
            complete = False
            continue
        seen_identities.add(identity)
        if _interesting_package(name):
            packages[(name, epoch, version, architecture)] = PackageInfo(
                name=name,
                version=version,
                manager="rpm",
                installed=True,
                architecture=architecture,
                epoch=epoch,
            )
    sorted_packages = sorted(
        packages.values(),
        key=lambda pkg: (
            pkg.name,
            pkg.architecture or "",
            pkg.epoch or "",
            pkg.version or "",
        ),
    )
    return sorted_packages, complete


def _valid_package_inventory_field(value: str) -> bool:
    return bool(value) and value == value.strip() and not any(
        character.isspace() for character in value
    )


def _interesting_package(name: str) -> bool:
    prefixes = (
        "nvidia",
        "libnvidia",
        "cuda",
        "docker-ce",
        "linux-headers-",
        "kernel-devel-",
    )
    exact = {
        "build-essential",
        "containerd.io",
        "docker-buildx-plugin",
        "docker-compose-plugin",
        "gcc",
        "kernel-devel",
        "kernel-headers",
        "make",
        "nvidia-container-toolkit",
    }
    return (
        name.startswith(prefixes)
        or name in exact
        or re.fullmatch(r"kernel-(?:default|azure|64k)-devel", name) is not None
    )


def _audit_nvml(runner: CommandRunner) -> CommandResult:
    code = "import ctypes; ctypes.CDLL('libnvidia-ml.so.1'); print('NVML load ok')"
    return runner.run(["python3", "-I", "-S", "-c", code], allow_fail=True)


def _audit_cuda_compatibility(
    packages: list[PackageInfo], runner: CommandRunner
) -> list[CudaCompatibilityInfo]:
    observations: list[CudaCompatibilityInfo] = []
    seen_versions: set[str] = set()
    for package in packages:
        if not package.installed:
            continue
        version = cuda_compat_version_from_package(package.name)
        if version is None or version in seen_versions:
            continue
        seen_versions.add(version)
        library_path = cuda_compat_library_path(version)
        observations.append(
            CudaCompatibilityInfo(
                version=version,
                package_name=package.name,
                package_version=package.version,
                library_path=str(library_path),
                library_present=library_path.is_file(),
                library_probe=probe_cuda_compat_library(version, runner),
            )
        )
    return observations


def _service_state(
    runner: CommandRunner,
    service: str,
) -> tuple[bool | None, bool | None, str | None]:
    if not runner.exists("systemctl"):
        return None, None, None
    result = runner.run(
        [
            "systemctl",
            "show",
            "--no-pager",
            "--property=Id",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=UnitFileState",
            service,
        ],
        allow_fail=True,
    )
    return _parse_service_state(result, service)


def _parse_service_state(
    result: CommandResult,
    expected_unit: str,
) -> tuple[bool | None, bool | None, str | None]:
    if (
        result.returncode != 0
        or _OUTPUT_TRUNCATED in result.stdout
        or _OUTPUT_TRUNCATED in result.stderr
    ):
        return None, None, None
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            return None, None, None
        name, value = line.split("=", 1)
        if name in properties:
            return None, None, None
        properties[name] = value
    if set(properties) != {
        "Id",
        "LoadState",
        "ActiveState",
        "UnitFileState",
    }:
        return None, None, None
    if properties["Id"] != expected_unit:
        return None, None, None
    load_state = properties["LoadState"]
    active_state = properties["ActiveState"]
    unit_file_state = properties["UnitFileState"]
    if load_state == "not-found":
        if active_state != "inactive" or unit_file_state:
            return None, None, None
        return False, False, "not-found"
    if load_state not in {"loaded", "masked"}:
        return None, None, None
    if unit_file_state not in _RESTORABLE_UNIT_FILE_STATES:
        return None, None, None
    if load_state == "masked" and unit_file_state != "masked":
        return None, None, None
    if load_state == "loaded" and unit_file_state == "masked":
        return None, None, None
    if active_state == "active":
        active: bool | None = True
    elif active_state in {"inactive", "failed"}:
        active = False
    else:
        active = None
    if unit_file_state == "masked" and active is not False:
        return None, None, None
    enabled = unit_file_state == "enabled"
    return active, enabled, unit_file_state


def _fabric_manager_version(
    runner: CommandRunner, packages: list[PackageInfo]
) -> str | None:
    if runner.exists("nv-fabricmanager"):
        result = runner.run(["nv-fabricmanager", "-v"], allow_fail=True)
        if result.returncode == 0:
            match = re.search(r"\b(\d+(?:\.\d+){1,3})\b", result.stdout)
            if match:
                return match.group(1)
    for package in packages:
        if package.installed and (
            package.name in {"nvidia-fabricmanager", "nvidia-fabric-manager"}
            or package.name.startswith("nvidia-fabricmanager-")
            or package.name.startswith("nvidia-fabric-manager-")
        ):
            match = re.search(r"\d+(?:\.\d+){1,3}", package.version or "")
            if match:
                return match.group(0)
    return None


def _audit_fabric_manager_health(
    runner: CommandRunner,
    gpu_uuids: list[str],
) -> tuple[bool | None, bool | None, CommandResult | None]:
    """Prove a healthy Fabric Manager handshake for the exact GPU inventory."""
    if not runner.exists("nvidia-smi"):
        return None, None, None
    if (
        not gpu_uuids
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(
            re.fullmatch(r"GPU-[A-Fa-f0-9-]{16,}", gpu_uuid) is None
            for gpu_uuid in gpu_uuids
        )
    ):
        return None, None, None
    states: list[str] = []
    statuses: list[str] = []
    result: CommandResult | None = None
    for gpu_uuid in gpu_uuids:
        result = runner.run(
            ["nvidia-smi", "-q", "-x", "-i", gpu_uuid],
            allow_fail=True,
        )
        if (
            result.returncode != 0
            or _OUTPUT_TRUNCATED in result.stdout
            or _OUTPUT_TRUNCATED in result.stderr
        ):
            return None, None, result
        observation = _parse_fabric_manager_health_xml(result.stdout)
        if observation is None:
            result.reason = "fabric-health-output-malformed"
            return None, None, result
        observed_uuid, state, status = observation
        if observed_uuid != gpu_uuid:
            result.reason = "fabric-health-gpu-coverage-incomplete"
            return None, None, result
        states.append(state)
        statuses.append(status)
    assert result is not None
    unsupported = {"not supported", "n/a"}
    if any(state in unsupported for state in states):
        return False, False, result
    applicable = all(state not in unsupported for state in states)
    healthy = bool(
        applicable
        and all(state == "completed" for state in states)
        and all(status in {"success", "nvml_success"} for status in statuses)
    )
    return applicable, healthy, result


def _parse_fabric_manager_health_xml(
    output: str,
) -> tuple[str, str, str] | None:
    sanitized, doctype_count = _NVIDIA_SMI_XML_DOCTYPE.subn("", output)
    if doctype_count > 1:
        return None
    try:
        root = parse_bounded_xml(sanitized)
    except SafeXmlError:
        return None
    if root.tag != "nvidia_smi_log":
        return None
    gpus = root.findall("./gpu")
    if len(gpus) != 1:
        return None
    gpu = gpus[0]
    uuid_elements = gpu.findall("./uuid")
    fabric_elements = gpu.findall("./fabric")
    if len(uuid_elements) != 1 or len(fabric_elements) != 1:
        return None
    state_elements = fabric_elements[0].findall("./state")
    status_elements = fabric_elements[0].findall("./status")
    if len(state_elements) != 1 or len(status_elements) != 1:
        return None
    values = [
        (element.text or "").strip()
        for element in (uuid_elements[0], state_elements[0], status_elements[0])
    ]
    if any(not value for value in values):
        return None
    return values[0], values[1].lower(), values[2].lower()


def _audit_mig_state(
    runner: CommandRunner,
    nvidia_smi_output: str,
) -> tuple[str | None, str | None, bool | None, list[str]]:
    if runner.exists("nvidia-smi"):
        result = runner.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,mig.mode.current,mig.mode.pending",
                "--format=csv,noheader,nounits",
            ],
            allow_fail=True,
        )
        if result.returncode == 0 and _OUTPUT_TRUNCATED not in result.stdout:
            rows = [
                [part.strip() for part in line.split(",")]
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            valid = {"enabled", "disabled", "n/a"}
            if rows and all(
                len(row) == 3
                and re.fullmatch(
                    r"GPU-[A-Fa-f0-9-]{16,}", row[0]
                ) is not None
                and all(value.lower() in valid for value in row[1:])
                for row in rows
            ) and len({row[0] for row in rows}) == len(rows):
                gpu_uuids = [row[0] for row in rows]
                current = [row[1].lower() for row in rows]
                pending = [row[2].lower() for row in rows]
                return (
                    _aggregate_mig_modes(current),
                    _aggregate_mig_modes(pending),
                    all(mode != "n/a" for mode in current),
                    gpu_uuids,
                )
    return _mig_mode(nvidia_smi_output), None, None, []


def _audit_mig_geometry(
    runner: CommandRunner,
    mig_mode: str | None,
    mig_mode_pending: str | None,
    gpu_uuids: list[str],
) -> tuple[list[MigGpuInstance], list[str], bool, list[CommandResult]]:
    """Capture stable GI/CI geometry and container-addressable MIG UUIDs.

    MIG devices disappear across reset/reboot, so mode alone is not a usable
    desired state.  Geometry is accepted only from complete, non-truncated
    UUID-bound observations while both current and pending mode are enabled.
    """
    if (
        mig_mode == "disabled"
        and mig_mode_pending == "disabled"
        and gpu_uuids
    ):
        return [], [], True, []
    if (
        mig_mode != "enabled"
        or mig_mode_pending != "enabled"
        or not gpu_uuids
        or not runner.exists("nvidia-smi")
    ):
        return [], [], False, []

    results: list[CommandResult] = []
    geometry: list[MigGpuInstance] = []
    for gpu_uuid in gpu_uuids:
        gpu_instances_result = runner.run(
            ["nvidia-smi", "mig", "-i", gpu_uuid, "-lgi"],
            allow_fail=True,
        )
        results.append(gpu_instances_result)
        if (
            gpu_instances_result.returncode != 0
            or _OUTPUT_TRUNCATED in gpu_instances_result.stdout
        ):
            return [], [], False, results
        parsed_gpu_instances = _parse_mig_gpu_instances(
            gpu_instances_result.stdout,
            gpu_uuid,
        )
        if parsed_gpu_instances is None:
            return [], [], False, results
        if not parsed_gpu_instances:
            continue

        compute_instances_result = runner.run(
            ["nvidia-smi", "mig", "-i", gpu_uuid, "-lci"],
            allow_fail=True,
        )
        results.append(compute_instances_result)
        if (
            compute_instances_result.returncode != 0
            or _OUTPUT_TRUNCATED in compute_instances_result.stdout
        ):
            return [], [], False, results
        parsed_compute_instances = _parse_mig_compute_instances(
            compute_instances_result.stdout
        )
        if parsed_compute_instances is None:
            return [], [], False, results
        known_gpu_instance_ids = {
            instance_id for instance_id, _ in parsed_gpu_instances
        }
        if any(
            gpu_instance_id not in known_gpu_instance_ids
            for gpu_instance_id, _, _ in parsed_compute_instances
        ):
            return [], [], False, results
        compute_by_gpu_instance: dict[int, list[MigComputeInstance]] = {
            instance_id: [] for instance_id in known_gpu_instance_ids
        }
        for gpu_instance_id, _, compute_instance in parsed_compute_instances:
            compute_by_gpu_instance[gpu_instance_id].append(compute_instance)
        geometry.extend(
            MigGpuInstance(
                gpu_uuid=instance.gpu_uuid,
                profile=instance.profile,
                profile_id=instance.profile_id,
                placement_start=instance.placement_start,
                placement_size=instance.placement_size,
                compute_instances=compute_by_gpu_instance[instance_id],
            )
            for instance_id, instance in parsed_gpu_instances
        )

    list_result = runner.run(["nvidia-smi", "-L"], allow_fail=True)
    results.append(list_result)
    if list_result.returncode != 0 or _OUTPUT_TRUNCATED in list_result.stdout:
        return [], [], False, results
    parsed_devices = _parse_mig_device_uuids(list_result.stdout, gpu_uuids)
    if parsed_devices is None:
        return [], [], False, results
    expected_device_count = sum(
        len(instance.compute_instances) for instance in geometry
    )
    mig_device_uuids = [
        device_uuid
        for gpu_uuid in gpu_uuids
        for device_uuid in parsed_devices[gpu_uuid]
    ]
    if len(mig_device_uuids) != expected_device_count:
        return [], [], False, results
    geometry.sort(
        key=lambda instance: (
            gpu_uuids.index(instance.gpu_uuid),
            instance.placement_start,
            instance.profile_id,
        )
    )
    return geometry, mig_device_uuids, True, results


_MIG_GPU_INSTANCE_ROW = re.compile(
    r"^\|\s*\d+\s+MIG\s+(?P<profile>[A-Za-z0-9.+_-]+)\s+"
    r"(?P<profile_id>\d+)\s+(?P<instance_id>\d+)\s+"
    r"(?P<placement_start>\d+):(?P<placement_size>\d+)\s*\|$"
)
_MIG_COMPUTE_INSTANCE_ROW = re.compile(
    r"^\|\s*\d+\s+(?P<gpu_instance_id>\d+)\s+MIG\s+"
    r"(?P<profile>[A-Za-z0-9.+_-]+)\s+(?P<profile_id>\d+)\s+"
    r"(?P<instance_id>\d+)\s*\|$"
)


def _parse_mig_gpu_instances(
    output: str,
    gpu_uuid: str,
) -> list[tuple[int, MigGpuInstance]] | None:
    if re.search(r"no\s+gpu\s+instances\s+found", output, re.IGNORECASE):
        return []
    candidates = [
        line.strip()
        for line in output.splitlines()
        if re.search(r"\|\s*\d+\s+MIG\s+", line)
    ]
    matches = [_MIG_GPU_INSTANCE_ROW.fullmatch(line) for line in candidates]
    if not candidates or any(match is None for match in matches):
        return None
    instances: list[tuple[int, MigGpuInstance]] = []
    ids: set[int] = set()
    for match in matches:
        assert match is not None
        instance_id = int(match.group("instance_id"))
        if instance_id in ids:
            return None
        ids.add(instance_id)
        instances.append(
            (
                instance_id,
                MigGpuInstance(
                    gpu_uuid=gpu_uuid,
                    profile=match.group("profile"),
                    profile_id=int(match.group("profile_id")),
                    placement_start=int(match.group("placement_start")),
                    placement_size=int(match.group("placement_size")),
                ),
            )
        )
    instances.sort(key=lambda item: (item[1].placement_start, item[0]))
    return instances


def _parse_mig_compute_instances(
    output: str,
) -> list[tuple[int, int, MigComputeInstance]] | None:
    if re.search(r"no\s+compute\s+instances\s+found", output, re.IGNORECASE):
        return []
    candidates = [
        line.strip()
        for line in output.splitlines()
        if re.search(r"\|\s*\d+\s+\d+\s+MIG\s+", line)
    ]
    matches = [_MIG_COMPUTE_INSTANCE_ROW.fullmatch(line) for line in candidates]
    if not candidates or any(match is None for match in matches):
        return None
    instances: list[tuple[int, int, MigComputeInstance]] = []
    ids: set[tuple[int, int]] = set()
    for match in matches:
        assert match is not None
        gpu_instance_id = int(match.group("gpu_instance_id"))
        instance_id = int(match.group("instance_id"))
        identity = (gpu_instance_id, instance_id)
        if identity in ids:
            return None
        ids.add(identity)
        instances.append(
            (
                gpu_instance_id,
                instance_id,
                MigComputeInstance(
                    profile=match.group("profile"),
                    profile_id=int(match.group("profile_id")),
                ),
            )
        )
    instances.sort(key=lambda item: (item[0], item[1]))
    return instances


def _parse_mig_device_uuids(
    output: str,
    gpu_uuids: list[str],
) -> dict[str, list[str]] | None:
    devices: dict[str, list[str]] = {gpu_uuid: [] for gpu_uuid in gpu_uuids}
    current_gpu_uuid: str | None = None
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        gpu_match = re.fullmatch(
            r"GPU\s+\d+:.*\(UUID:\s*(GPU-[A-Fa-f0-9-]{16,})\)",
            line,
        )
        if gpu_match:
            current_gpu_uuid = gpu_match.group(1)
            if current_gpu_uuid not in devices:
                return None
            continue
        mig_match = re.fullmatch(
            r"MIG\s+.+?\s+Device\s+\d+:\s*"
            r"\(UUID:\s*(MIG-[A-Fa-f0-9-]{16,})\)",
            line,
        )
        if mig_match:
            device_uuid = mig_match.group(1)
            if current_gpu_uuid is None or device_uuid in seen:
                return None
            seen.add(device_uuid)
            devices[current_gpu_uuid].append(device_uuid)
        elif "UUID:" in line and "MIG-" in line:
            return None
    return devices


def _audit_open_module_support(runner: CommandRunner) -> bool | None:
    if not runner.exists("nvidia-smi"):
        return None
    result = runner.run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ],
        allow_fail=True,
    )
    if result.returncode != 0 or _OUTPUT_TRUNCATED in result.stdout:
        return None
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values:
        return None
    capabilities: list[tuple[int, int]] = []
    for value in values:
        match = re.fullmatch(r"(\d+)\.(\d+)", value)
        if match is None:
            return None
        capabilities.append((int(match.group(1)), int(match.group(2))))
    # NVIDIA supports the open kernel-module flavor on Turing (7.5) and newer.
    return all(capability >= (7, 5) for capability in capabilities)


def _aggregate_mig_modes(modes: list[str]) -> str | None:
    normalized = {"disabled" if mode == "n/a" else mode for mode in modes}
    if normalized == {"enabled"}:
        return "enabled"
    if normalized == {"disabled"}:
        return "disabled"
    if normalized == {"enabled", "disabled"}:
        return "mixed"
    return None


def _mig_mode(nvidia_smi_output: str) -> str | None:
    lowered = nvidia_smi_output.lower()
    modes = [line.strip().lower() for line in nvidia_smi_output.splitlines()]
    recognized = {"disabled" if mode == "n/a" else mode for mode in modes if mode in {"enabled", "disabled", "n/a"}}
    if recognized == {"enabled"}:
        return "enabled"
    if recognized == {"disabled"}:
        return "disabled"
    if recognized == {"enabled", "disabled"}:
        return "mixed"
    if re.search(r"mig mode\s*:\s*enabled", lowered):
        return "enabled"
    if re.search(r"mig mode\s*:\s*disabled", lowered):
        return "disabled"
    return None
