from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4

from .desired import (
    container_cuda_full_version,
    container_cuda_minor_compatibility_status,
    container_cuda_version,
)
from .files import BoundedFileError, read_bounded_utf8
from .gpu_safety import validate_active_trusted_gpu_service_identity
from .kernel_headers import assess_running_kernel_headers
from .mig import full_mig_geometry_matches
from .models import CommandResult, DesiredState, HostAudit, Verification
from .module_safety import ModuleDependencyError, nvidia_module_unload_order
from .runner import CommandRunner

_CUDA_DRIVER_PROBE_PATH = Path(__file__).with_name("probes") / "cuda_driver_probe.c"
_CUDA_DRIVER_PROBE_MAX_BYTES = 16 * 1024
_CUDA_DRIVER_PROBE_SHA256 = (
    "4d01cf2d3618c0b3e2a737a0a2ff125febd5b74e944f309badef7ed8b21669fd"
)
_CUDA_DRIVER_PROBE_SUCCESS = re.compile(
    r"^CUDA_DRIVER_API_OK driver_version=[1-9]\d* device_count=1$"
)
_DIGEST_PINNED_CUDA_IMAGE = re.compile(
    r"^nvidia/cuda:[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"@sha256:[a-f0-9]{64}$"
)
_DOCKER_IMAGE_ID = re.compile(r"^sha256:[a-f0-9]{64}$")
_CUDA_DRIVER_PROBE_SCRIPT = r"""
set -o pipefail
umask 077
source_path=/tmp/nvidia-converge-cuda-driver-probe.c
binary_path=/tmp/nvidia-converge-cuda-driver-probe
cat > "${source_path}"
actual_sha256="$(sha256sum "${source_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${NVIDIA_CONVERGE_PROBE_SHA256}" ]]; then
    echo "CUDA Driver API probe source integrity check failed" >&2
    exit 70
fi
if [[ "${CUDA_VERSION:-}" != "${NVIDIA_CONVERGE_EXPECTED_CUDA_VERSION}" ]]; then
    echo "Pinned container CUDA_VERSION does not match its configured tag" >&2
    exit 74
fi
/usr/local/cuda/bin/nvcc \
    -x cu \
    -std=c++17 \
    --cudart=none \
    -O2 \
    -Xcompiler=-fPIE,-fstack-protector-strong,-D_FORTIFY_SOURCE=2,-Wall,-Wextra,-Werror \
    -Xlinker=-pie,-z,relro,-z,now \
    -L/usr/local/cuda/lib64/stubs \
    -o "${binary_path}" \
    "${source_path}" \
    -lcuda
probe_library_path=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
libcuda_path="$(
    LD_LIBRARY_PATH="${probe_library_path}" ldd "${binary_path}" |
        awk '$1 == "libcuda.so.1" && $2 == "=>" { print $3 }'
)"
if [[ "${libcuda_path}" != /* || "${libcuda_path}" == *$'\n'* ]]; then
    echo "CUDA Driver API probe could not resolve exactly one libcuda.so.1" >&2
    exit 71
fi
resolved_libcuda="$(readlink -f -- "${libcuda_path}")"
if [[ -z "${resolved_libcuda}" ]]; then
    echo "CUDA Driver API probe could not canonicalize libcuda.so.1" >&2
    exit 72
fi
case "${libcuda_path}:${resolved_libcuda}" in
    *'/compat/'*|*'/usr/local/cuda/'*|*'/usr/local/cuda-'*)
        echo "CUDA Driver API probe refuses a forward-compatibility libcuda.so.1" >&2
        exit 73
        ;;
esac
LD_LIBRARY_PATH="${probe_library_path}" "${binary_path}"
""".strip()


def prepare_stack(
    runner: CommandRunner,
    audit: HostAudit | None = None,
    *,
    force_reload: bool = False,
    sys_module_root: Path = Path("/sys/module"),
) -> Verification:
    if audit and audit.module.loaded and not force_reload:
        return Verification(
            "module.load", True, detail="NVIDIA module is already loaded."
        )
    if force_reload:
        try:
            unload_order = nvidia_module_unload_order(sys_module_root=sys_module_root)
        except ModuleDependencyError as exc:
            result = CommandResult(
                ["modprobe", "-r", "nvidia-stack"],
                1,
                stderr=str(exc),
            )
            return Verification(
                "module.reload",
                False,
                result,
                "Refuses to unload the NVIDIA stack unless every loaded module holder is recognized.",
            )
        if unload_order:
            unload = (
                runner.run(
                    ["modprobe", "-r", *unload_order],
                    mutate=True,
                    allow_fail=True,
                )
                if runner.exists("modprobe")
                else CommandResult(
                    ["modprobe", "-r", *unload_order],
                    127,
                    stderr="trusted modprobe executable not found",
                )
            )
            if unload.returncode not in (0, None):
                return Verification(
                    "module.reload",
                    False,
                    unload,
                    "Unload every recognized NVIDIA dependent before loading the selected on-disk module.",
                )
    modules_to_load = (
        list(reversed(unload_order)) if force_reload and unload_order else ["nvidia"]
    )
    load = None
    for module in modules_to_load:
        load = (
            runner.run(["modprobe", module], mutate=True, allow_fail=True)
            if runner.exists("modprobe")
            else CommandResult(
                ["modprobe", module],
                127,
                stderr="trusted modprobe executable not found",
            )
        )
        if load.returncode not in (0, None):
            break
    return Verification(
        "module.reload" if force_reload else "module.load",
        bool(load and load.returncode == 0),
        load,
        (
            "Unloads recognized NVIDIA dependents, then restores every previously loaded module in dependency order when --apply is used."
            if force_reload
            else "Loads nvidia module when --apply is used; dry-run records the command without marking it verified."
        ),
    )


def verify_stack(
    desired: DesiredState,
    runner: CommandRunner,
    audit: HostAudit | None = None,
    *,
    include_docker: bool = True,
    include_fabric_manager: bool = True,
) -> list[Verification]:
    checks: list[Verification] = []
    checks.extend(_module_version_checks(desired, audit))
    if audit:
        checks.extend(_secure_boot_checks(desired, audit))
        checks.append(
            _check(
                "mig.mode",
                audit.mig_mode == desired.mig,
                f"MIG mode must be observable and match desired state {desired.mig}.",
            )
        )
        checks.append(
            _check(
                "mig.pending-observable",
                audit.mig_mode_pending is not None,
                "Pending MIG mode must be observable for every GPU.",
            )
        )
        checks.append(
            _check(
                "mig.no-pending-transition",
                bool(
                    audit.mig_mode is not None
                    and audit.mig_mode_pending == audit.mig_mode
                ),
                "Current and pending MIG modes must match; otherwise a reboot boundary remains.",
            )
        )
        if desired.mig == "enabled":
            checks.append(
                _check(
                    "mig.capable",
                    audit.mig_capable is True,
                    "Every GPU must explicitly report MIG capability before host-wide enablement.",
                )
            )
            checks.append(
                _check(
                    "mig.geometry-observable",
                    audit.mig_geometry_complete,
                    "GPU-instance, compute-instance, and MIG-device geometry must be completely observable.",
                )
            )
            checks.append(
                _check(
                    "mig.geometry",
                    bool(
                        audit.mig_geometry_complete
                        and full_mig_geometry_matches(
                            audit.mig_geometry,
                            audit.gpu_uuids,
                        )
                    ),
                    "MIG must expose one full-profile GI containing one full-size CI.",
                )
            )
            checks.append(
                _check(
                    "mig.device-uuid",
                    bool(
                        audit.mig_geometry_complete
                        and len(audit.mig_device_uuids) == 1
                    ),
                    "Exactly one container-addressable MIG compute-device UUID must be observed.",
                )
            )
        else:
            checks.append(
                _check(
                    "mig.geometry-observable",
                    audit.mig_geometry_complete,
                    "Disabled MIG state must have a complete empty geometry observation.",
                )
            )
            checks.append(
                _check(
                    "mig.geometry",
                    bool(
                        audit.mig_geometry_complete
                        and not audit.mig_geometry
                        and not audit.mig_device_uuids
                    ),
                    "Disabled MIG mode must expose no GPU or compute instances.",
                )
            )
        if desired.open_kernel_module:
            checks.append(
                _check(
                    "gpu.open-module-supported",
                    audit.open_kernel_module_supported is True,
                    "Every GPU must be Turing (compute capability 7.5) or newer for the open kernel module.",
                )
            )
            checks.append(
                _check(
                    "module.open-variant",
                    audit.module.open_module is True,
                    "Loaded NVIDIA module must be the open kernel module variant.",
                )
            )
            checks.append(
                _check(
                    "module.on-disk-open-variant",
                    audit.module.installed_open_module is True,
                    "modinfo-selected NVIDIA module must be the open kernel module variant.",
                )
            )
        else:
            checks.append(
                _check(
                    "module.closed-variant",
                    audit.module.open_module is False,
                    "Loaded NVIDIA module must be the closed kernel module variant.",
                )
            )
            checks.append(
                _check(
                    "module.on-disk-closed-variant",
                    audit.module.installed_open_module is False,
                    "modinfo-selected NVIDIA module must be the closed kernel module variant.",
                )
            )
        checks.append(
            _check(
                "module.flavor-provenance",
                bool(
                    audit.module.open_module is not None
                    and audit.module.installed_open_module is not None
                    and audit.module.open_module == audit.module.installed_open_module
                ),
                "Loaded and modinfo-selected NVIDIA module flavors must match.",
            )
        )
    header_readiness = assess_running_kernel_headers(
        runner,
        package_manager=audit.package_manager if audit else None,
    )
    checks.append(
        _check(
            "kernel.headers",
            header_readiness.ready,
            header_readiness.detail,
        )
    )
    dkms = (
        runner.run(["dkms", "status", "-m", "nvidia"], allow_fail=True)
        if runner.exists("dkms")
        else None
    )
    if dkms:
        checks.append(
            Verification(
                "module.dkms-build",
                dkms.returncode == 0 and "installed" in dkms.stdout.lower(),
                dkms,
                "DKMS reports an installed NVIDIA module build for the running kernel.",
            )
        )
    modprobe = (
        runner.run(["modprobe", "-n", "-v", "nvidia"], allow_fail=True)
        if runner.exists("modprobe")
        else None
    )
    checks.append(
        Verification(
            "module.compile-or-loadable",
            bool(modprobe and modprobe.returncode == 0),
            modprobe,
            "modprobe dry-run can resolve nvidia module.",
        )
    )
    smi = (
        runner.run(["nvidia-smi"], allow_fail=True)
        if runner.exists("nvidia-smi")
        else None
    )
    checks.append(
        Verification(
            "nvidia-smi",
            bool(
                smi
                and smi.returncode == 0
                and _nvidia_smi_matches_driver(desired, smi.stdout)
            ),
            smi,
            f"nvidia-smi must run and show driver {desired.driver_match_label}.",
        )
    )
    nvml = runner.run(
        [
            "python3",
            "-I",
            "-S",
            "-c",
            "import ctypes; ctypes.CDLL('libnvidia-ml.so.1'); print('NVML load ok')",
        ],
        allow_fail=True,
    )
    checks.append(
        Verification("nvml", nvml.returncode == 0, nvml, "NVML shared library loads.")
    )
    if desired.cuda_compat == "none":
        checks.append(
            _check(
                "cuda-compat.policy",
                True,
                "No host CUDA forward-compatibility deployment is requested; "
                "the container probe rejects compatibility-library paths.",
            )
        )
    else:
        checks.append(
            _check(
                "cuda-compat.deployment-policy",
                False,
                "Non-none CUDA forward-compatibility is unsupported because a "
                "reversible loader-path deployment is not modeled.",
            )
        )
    if include_docker and desired.container_runtime == "docker":
        docker_service_active = bool(
            audit and audit.docker_service_active is True
        )
        checks.append(
            _check(
                "docker.service-active",
                docker_service_active,
                "Docker service activity must be observable and active.",
            )
        )
        checks.append(
            _check(
                "docker.service-enabled",
                bool(audit and audit.docker_service_enabled is True),
                "Docker must be enabled persistently so GPU containers remain available after reboot.",
            )
        )
        docker_service_trusted = _append_active_service_trust_check(
            checks,
            runner,
            unit="docker.service",
            name="docker.service-trust",
            observed_active=docker_service_active,
        )
        observed_driver_major = _observed_driver_major(audit)
        container_compatibility = container_cuda_minor_compatibility_status(
            desired,
            driver_major=observed_driver_major or "unknown",
        )
        container_cuda = container_cuda_version(desired.container_test_image)
        container_cuda_full = container_cuda_full_version(desired.container_test_image)
        container_compatibility_ok = bool(
            desired.cuda_compat == "none" and container_compatibility == "compatible"
        )
        checks.append(
            _check(
                "container.cuda-driver-compatibility",
                container_compatibility_ok,
                (
                    f"Observed driver branch {observed_driver_major or 'unknown'} "
                    f"must natively support CUDA {container_cuda or 'unknown'}; "
                    "forward-compatibility libraries are not permitted."
                ),
            )
        )
        container_devices = _container_device_selectors(desired, audit)
        checks.append(
            _check(
                "container.device-binding",
                container_devices is not None,
                (
                    "Container verification must bind the single retained MIG UUID."
                    if desired.mig == "enabled"
                    else "Container verification must bind every exact audited physical GPU UUID."
                ),
            )
        )
        docker_available = bool(
            runner.exists("docker")
            and docker_service_active
            and docker_service_trusted
        )
        image_inspection = None
        if (
            docker_available
            and container_compatibility_ok
            and container_devices is not None
        ):
            image_inspection = _inspect_local_digest_image(
                runner,
                desired.container_test_image,
            )
        image_present = bool(
            image_inspection
            and image_inspection.returncode == 0
            and _DOCKER_IMAGE_ID.fullmatch(image_inspection.stdout.strip())
        )
        checks.append(
            Verification(
                "container.image-present",
                image_present,
                image_inspection,
                "The exact digest-bound CUDA image must already be present locally; verification never pulls it.",
            )
        )

        probe_source, probe_source_error = _load_cuda_driver_probe_source()
        probe_results: list[tuple[str, CommandResult]] = []
        cleanup_results: list[tuple[str, CommandResult]] = []
        absence_results: list[tuple[str, CommandResult]] = []
        synthetic_probe_result = None
        if probe_source_error is not None:
            unavailable_device = (
                container_devices[0] if container_devices else "unavailable"
            )
            synthetic_probe_result = CommandResult(
                _container_probe_command(
                    name=f"nvidia-converge-verify-{uuid4().hex}",
                    device=unavailable_device,
                    image=desired.container_test_image,
                    expected_cuda=container_cuda_full or "unknown",
                ),
                126,
                stderr=probe_source_error,
                reason="cuda-driver-probe-source-invalid",
            )
        elif (
            docker_available
            and container_compatibility_ok
            and container_devices is not None
            and image_present
        ):
            for device in container_devices:
                name = f"nvidia-converge-verify-{uuid4().hex}"
                command = _container_probe_command(
                    name=name,
                    device=device,
                    image=desired.container_test_image,
                    expected_cuda=container_cuda_full or "unknown",
                )
                try:
                    docker = runner.run(
                        command,
                        mutate=True,
                        allow_fail=True,
                        input_text=probe_source,
                    )
                except BaseException:
                    runner.run(
                        ["docker", "rm", "--force", name],
                        mutate=True,
                        allow_fail=True,
                    )
                    _probe_container_absence(runner, name)
                    raise
                probe_results.append((device, docker))
                if docker.returncode not in (0, None) or docker.reason in {
                    "timeout-process-group-terminated",
                    "lingering-process-group-terminated",
                }:
                    cleanup_results.append(
                        (
                            device,
                            runner.run(
                                ["docker", "rm", "--force", name],
                                mutate=True,
                                allow_fail=True,
                            ),
                        )
                    )
                absence_results.append(
                    (device, _probe_container_absence(runner, name))
                )
        failed_probe_devices = [
            device
            for device, result in probe_results
            if result.returncode != 0
            or not _cuda_driver_probe_succeeded(result.stdout)
        ]
        aggregate_probe = synthetic_probe_result or _aggregate_command_result(
            probe_results,
            failed_devices=failed_probe_devices,
        )
        checks.append(
            Verification(
                "container.gpu",
                bool(
                    container_devices
                    and len(probe_results) == len(container_devices)
                    and not failed_probe_devices
                ),
                aggregate_probe,
                "Builds the audited probe in tmpfs and requires cuInit plus "
                "cuDeviceGetCount to expose exactly one GPU inside the "
                "locally present digest-pinned CUDA container once for every "
                "audited physical GPU UUID, or once for the retained MIG UUID; "
                "dry-run records but does not execute the probe."
                + (
                    " Failed UUIDs: " + ", ".join(failed_probe_devices) + "."
                    if failed_probe_devices
                    else ""
                ),
            )
        )
        if cleanup_results:
            failed_cleanup_devices = [
                device
                for device, result in cleanup_results
                if result.returncode != 0
            ]
            checks.append(
                Verification(
                    "container.cleanup-command",
                    not failed_cleanup_devices,
                    _aggregate_command_result(
                        cleanup_results,
                        failed_devices=failed_cleanup_devices,
                    ),
                    "Every failed or interrupted verification container must be forcibly removed."
                    + (
                        " Cleanup failed for UUIDs: "
                        + ", ".join(failed_cleanup_devices)
                        + "."
                        if failed_cleanup_devices
                        else ""
                    ),
                )
            )
        failed_absence_devices = [
            device
            for device, result in absence_results
            if result.returncode != 0 or bool(result.stdout.strip())
        ]
        checks.append(
            Verification(
                "container.probe-absent",
                bool(
                    not probe_results
                    or (
                        len(absence_results) == len(probe_results)
                        and not failed_absence_devices
                    )
                ),
                _aggregate_command_result(
                    absence_results,
                    failed_devices=failed_absence_devices,
                ),
                "Every uniquely named verification container must be proven absent after its probe."
                + (
                    " Absence proof failed for UUIDs: "
                    + ", ".join(failed_absence_devices)
                    + "."
                    if failed_absence_devices
                    else ""
                ),
            )
        )
    if include_fabric_manager and desired.fabric_manager:
        _append_active_service_trust_check(
            checks,
            runner,
            unit="nvidia-fabricmanager.service",
            name="fabric-manager.service-trust",
            observed_active=bool(audit and audit.fabric_manager_active is True),
        )
        fm = (
            runner.run(
                ["systemctl", "is-active", "nvidia-fabricmanager"], allow_fail=True
            )
            if runner.exists("systemctl")
            else None
        )
        checks.append(
            Verification(
                "fabric-manager",
                bool(fm and fm.returncode == 0),
                fm,
                "Fabric Manager service is active.",
            )
        )
        checks.append(
            _check(
                "fabric-manager.enabled",
                bool(audit and audit.fabric_manager_enabled is True),
                "Fabric Manager must be enabled persistently at boot.",
            )
        )
        checks.append(
            _check(
                "fabric-manager.applicable",
                bool(audit and audit.fabric_manager_applicable is True),
                "Every GPU must expose an applicable Fabric Manager handshake.",
            )
        )
        checks.append(
            _check(
                "fabric-manager.fabric-health",
                bool(audit and audit.fabric_manager_healthy is True),
                "Every GPU must complete a successful Fabric Manager handshake.",
            )
        )
        checks.append(
            _check(
                "fabric-manager.version",
                bool(
                    audit
                    and desired.matches_driver_version(audit.fabric_manager_version)
                    and audit.module.version
                    and audit.fabric_manager_version == audit.module.version
                ),
                "Fabric Manager must match both the desired selector and the loaded driver version exactly.",
            )
        )
    return checks


def _module_version_checks(
    desired: DesiredState, audit: HostAudit | None
) -> list[Verification]:
    loaded_version = audit.module.version if audit else None
    installed_version = audit.module.installed_version if audit else None
    return [
        _check(
            "module.loaded-version",
            bool(
                audit
                and audit.module.loaded
                and desired.matches_driver_version(loaded_version)
            ),
            f"Loaded NVIDIA module version must be observable and match desired {desired.driver_match_label}.",
        ),
        _check(
            "module.on-disk-version",
            bool(audit and desired.matches_driver_version(installed_version)),
            f"modinfo-selected NVIDIA module version must be observable and match desired {desired.driver_match_label}.",
        ),
        _check(
            "module.provenance",
            bool(
                loaded_version
                and installed_version
                and loaded_version == installed_version
            ),
            "Loaded /sys NVIDIA module and modinfo-selected on-disk module versions must match exactly.",
        ),
    ]


def _secure_boot_checks(desired: DesiredState, audit: HostAudit) -> list[Verification]:
    checks: list[Verification] = [
        _check(
            "secure-boot.observable",
            audit.kernel.secure_boot_enabled is not None,
            "Secure Boot firmware state must be observable.",
        )
    ]
    if desired.secure_boot == "disabled":
        checks.append(
            _check(
                "secure-boot.policy",
                audit.kernel.secure_boot_enabled is not True,
                "Secure Boot must be disabled by desired policy.",
            )
        )
    if desired.secure_boot == "signed":
        checks.append(
            _check(
                "secure-boot.module-signed",
                audit.module.signed is True,
                "Loaded NVIDIA module must have a kernel-verified signature by desired policy.",
            )
        )
        checks.append(
            _check(
                "secure-boot.on-disk-module-signed",
                audit.module.installed_signed is True,
                "modinfo-selected NVIDIA module must carry a signature by desired policy.",
            )
        )
    return checks


def _observed_driver_major(audit: HostAudit | None) -> str | None:
    version = audit.module.version if audit else None
    if not version:
        return None
    match = re.match(r"^(\d+)(?:\.|$)", version)
    return match.group(1) if match else None


def _append_active_service_trust_check(
    checks: list[Verification],
    runner: CommandRunner,
    *,
    unit: str,
    name: str,
    observed_active: bool,
) -> bool:
    results: list[CommandResult] = []
    identity = None
    error: str | None = (
        "service activity must be observed before trust validation"
    )
    if observed_active:
        results, identity, error = validate_active_trusted_gpu_service_identity(
            runner,
            unit,
        )
    ok = bool(error is None and identity is not None)
    checks.append(
        Verification(
            name,
            ok,
            results[-1] if results else None,
            "The active service must bind to trusted effective systemd unit inputs, its exact executable, and its system.slice process identity.",
        )
    )
    return ok


def _container_device_selectors(
    desired: DesiredState,
    audit: HostAudit | None,
) -> list[str] | None:
    if audit is None or not audit.mig_geometry_complete:
        return None
    if desired.mig == "enabled":
        if (
            full_mig_geometry_matches(audit.mig_geometry, audit.gpu_uuids)
            and len(audit.mig_device_uuids) == 1
            and re.fullmatch(
                r"MIG-[A-Fa-f0-9-]{16,}", audit.mig_device_uuids[0]
            )
            is not None
        ):
            return [audit.mig_device_uuids[0]]
        return None
    if (
        audit.mig_mode == "disabled"
        and audit.mig_mode_pending == "disabled"
        and not audit.mig_geometry
        and not audit.mig_device_uuids
        and bool(audit.gpu_uuids)
        and len(set(audit.gpu_uuids)) == len(audit.gpu_uuids)
        and all(
            re.fullmatch(r"GPU-[A-Fa-f0-9-]{16,}", gpu_uuid) is not None
            for gpu_uuid in audit.gpu_uuids
        )
    ):
        return list(audit.gpu_uuids)
    return None


def _inspect_local_digest_image(
    runner: CommandRunner,
    image: str,
) -> CommandResult:
    command = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    if _DIGEST_PINNED_CUDA_IMAGE.fullmatch(image) is None:
        return CommandResult(
            command,
            2,
            stderr="CUDA verification image is not bound to an exact sha256 digest",
            reason="container-image-not-digest-pinned",
        )
    result = runner.run(command, allow_fail=True)
    if result.returncode == 0 and _DOCKER_IMAGE_ID.fullmatch(
        result.stdout.strip()
    ) is None:
        result.reason = "container-image-inspection-malformed"
    return result


def _container_probe_command(
    *,
    name: str,
    device: str,
    image: str,
    expected_cuda: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--name",
        name,
        "--label",
        "io.nvidia-converge.verification=true",
        "--label",
        f"io.nvidia-converge.cuda-probe-sha256={_CUDA_DRIVER_PROBE_SHA256}",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=1g",
        "--cpus=1",
        "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=64m,mode=1777",
        "--user=65534:65534",
        "--gpus",
        f"device={device}",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=compute",
        "--env",
        f"NVIDIA_CONVERGE_PROBE_SHA256={_CUDA_DRIVER_PROBE_SHA256}",
        "--env",
        f"NVIDIA_CONVERGE_EXPECTED_CUDA_VERSION={expected_cuda}",
        "--interactive",
        "--entrypoint=/bin/bash",
        image,
        "-ceu",
        _CUDA_DRIVER_PROBE_SCRIPT,
    ]


def _aggregate_command_result(
    results: list[tuple[str, CommandResult]],
    *,
    failed_devices: list[str],
) -> CommandResult | None:
    if not results:
        return None
    if failed_devices:
        failed = set(failed_devices)
        return next(result for device, result in results if device in failed)
    return results[-1][1]


def _probe_container_absence(
    runner: CommandRunner,
    name: str,
) -> CommandResult:
    return runner.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"name=^/{name}$",
        ],
        allow_fail=True,
    )


def _load_cuda_driver_probe_source() -> tuple[str | None, str | None]:
    try:
        source = read_bounded_utf8(
            _CUDA_DRIVER_PROBE_PATH,
            max_bytes=_CUDA_DRIVER_PROBE_MAX_BYTES,
        )
    except (OSError, BoundedFileError) as exc:
        return None, f"cannot read packaged CUDA Driver API probe source: {exc}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != _CUDA_DRIVER_PROBE_SHA256:
        return (
            None,
            "packaged CUDA Driver API probe source failed its SHA256 integrity check",
        )
    return source, None


def _cuda_driver_probe_succeeded(output: str) -> bool:
    return any(
        _CUDA_DRIVER_PROBE_SUCCESS.fullmatch(line) is not None
        for line in output.splitlines()
    )


def _check(name: str, ok: bool, detail: str) -> Verification:
    return Verification(name=name, ok=ok, detail=detail)


def _nvidia_smi_matches_driver(desired: DesiredState, output: str) -> bool:
    match = re.search(r"Driver Version:\s*([0-9]+(?:\.[0-9]+)+)", output, re.IGNORECASE)
    return bool(match and desired.matches_driver_version(match.group(1)))
