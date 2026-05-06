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
from nvidia_converge.planner import build_plan, lock_actions


def test_plan_includes_previewable_install_lock_verify_actions():
    desired = DesiredState()
    audit = _audit()
    findings = diagnose(desired, audit)
    plan = build_plan(desired, audit, findings)
    ids = [action.id for action in plan]
    assert "snapshot.current-state" in ids
    assert "install.packages" in ids
    assert "configure.docker-runtime" in ids
    assert "enable.fabric-manager" in ids
    assert "lock.apt" in ids
    assert "verify.stack" in ids
    snapshot = next(action for action in plan if action.id == "snapshot.current-state")
    assert snapshot.commands == [["nvidia-converge", "snapshot", "--apply"]]
    install = next(action for action in plan if action.id == "install.packages")
    flattened = [part for command in install.commands for part in command]
    assert "nvidia-driver-580-open" in flattened
    assert "cuda-compat-13-0" in flattened


def test_lock_actions_pin_compatibility():
    locks = lock_actions(DesiredState(), _audit())
    assert locks[0].commands[0][0:2] == ["apt-mark", "hold"]
    assert "nvidia-driver-580-open" in locks[0].commands[0]


def test_fabric_manager_false_omits_fabric_packages_and_service_action():
    desired = DesiredState(fabric_manager=False)
    audit = _audit()
    plan = build_plan(desired, audit, diagnose(desired, audit))
    ids = [action.id for action in plan]
    assert "enable.fabric-manager" not in ids
    flattened = [part for action in plan for command in action.commands for part in command]
    assert "nvidia-fabricmanager-580" not in flattened


def test_fabric_manager_false_omits_rpm_fabric_packages_from_locks():
    desired = DesiredState(fabric_manager=False)
    audit = _audit()
    audit.package_manager = "zypper"
    locks = lock_actions(desired, audit)
    assert "nvidia-fabric-manager-580" not in locks[0].commands[0]


def test_lock_actions_support_zypper():
    audit = _audit()
    audit.package_manager = "zypper"
    locks = lock_actions(DesiredState(), audit)
    assert locks[0].id == "lock.zypper"
    assert locks[0].commands[0][0:3] == ["zypper", "--non-interactive", "addlock"]
    assert "nvidia-open-580" in locks[0].commands[0]


def test_plan_enables_mig_when_desired():
    audit = _audit()
    audit.mig_mode = "disabled"
    plan = build_plan(DesiredState(mig="enabled"), audit, diagnose(DesiredState(mig="enabled"), audit))
    action = next(action for action in plan if action.id == "enable.mig")
    assert action.destructive is True
    assert action.commands == [["nvidia-smi", "-mig", "1"]]


def test_exact_driver_version_mismatch_plans_install():
    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "595.60.01"
    audit.module.signed = True
    audit.kernel.headers_installed = True
    audit.kernel.compiler = "/usr/bin/gcc"
    audit.runtime.docker_installed = True
    audit.runtime.nvidia_container_runtime_installed = True
    audit.runtime.docker_gpus_usable = True
    audit.nvidia_smi = CommandResult(["nvidia-smi"], 0, stdout="Driver Version: 595.60.01")
    audit.nvml = CommandResult(["python3"], 0)
    audit.fabric_manager_active = True
    audit.packages.append(PackageInfo("nvidia-fabricmanager-595", manager="apt", installed=True))
    desired = DesiredState(driver="595.71.05")
    plan = build_plan(desired, audit, diagnose(desired, audit))
    assert "install.packages" in [action.id for action in plan]


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
        mig_mode=None,
    )
