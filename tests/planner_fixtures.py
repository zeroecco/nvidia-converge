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


def _audit() -> HostAudit:
    return HostAudit(
        timestamp="2026-05-06T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo(
            "6.8.0-test",
            headers_installed=False,
            compiler=None,
            secure_boot_enabled=True,
        ),
        module=ModuleInfo(
            loaded=False,
            version=None,
            open_module=None,
            signed=False,
            devices=[],
        ),
        runtime=RuntimeInfo(
            docker_installed=False,
            nvidia_container_runtime_installed=False,
            docker_gpus_usable=False,
        ),
        packages=[
            PackageInfo("nvidia-driver", manager="apt", installed=False)
        ],
        nvidia_smi=CommandResult(
            ["nvidia-smi"], 127, stderr="not found"
        ),
        nvml=CommandResult(
            ["python3"],
            1,
            stderr="libnvidia-ml.so.1: cannot open shared object file",
        ),
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
    audit.module = ModuleInfo(
        loaded=True,
        version="580.126.16",
        open_module=True,
        signed=True,
        devices=["/dev/nvidia0"],
        installed_version="580.126.16",
        installed_open_module=True,
        installed_signed=True,
    )
    audit.runtime = RuntimeInfo(
        docker_installed=True,
        nvidia_container_runtime_installed=True,
        docker_gpus_usable=True,
    )
    audit.packages = [
        PackageInfo("cuda-compat-13-1", "590.1-1", "apt", True),
        PackageInfo(
            "nvidia-fabricmanager-580", manager="apt", installed=True
        ),
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


def _stage_policy(audit: HostAudit, desired: DesiredState) -> None:
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
