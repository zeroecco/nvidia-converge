import base64
import hashlib
import json
import platform
import stat
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    FileSnapshot,
    MigComputeInstance,
    MigGpuInstance,
    PackageInfo,
    PackagePayload,
    PackagePayloadBundle,
    PackagePolicyInfo,
    PackagePolicySelector,
    RollbackSnapshot,
    RuntimeInfo,
)
from nvidia_converge.module_safety import ModuleDependencyError
from nvidia_converge.package_payloads import payload_bundle_directory
from nvidia_converge.rollback import (
    MAX_SNAPSHOT_BYTES,
    RollbackSnapshotError,
    _capture_managed_files,
    _host_identity,
    _load_managed_files,
    _managed_file_restore_precondition,
    _quarantine_service_for_rollback,
    _restore_managed_files,
    _rollback_commands,
    _snapshot_integrity,
    apply_rollback,
    create_snapshot,
    load_snapshot,
    new_snapshot_path,
    prepare_rollback_service_activity,
    restore_rollback_service_activity,
    restore_rollback_service_enablement,
    verify_rollback,
)


def test_apt_rollback_only_restores_relevant_packages():
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-driver-580-open", "580.126.16-1", "apt", True),
            PackageInfo("cuda-compat-13-0", "13.0.0-1", "apt", True),
            PackageInfo("bash", "5.2", "apt", True),
            PackageInfo("libnvidia-gl", None, "apt", True),
        ],
        "apt-get",
    )
    assert commands == [
        [
            "apt-get",
            "install",
            "-y",
            "--allow-change-held-packages",
            "--allow-downgrades",
            "--no-download",
            "--no-install-recommends",
            "--purge",
            "nvidia-driver-580-open=580.126.16-1",
            "cuda-compat-13-0=13.0.0-1",
        ]
    ]


def test_rpm_rollback_only_restores_relevant_packages():
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-open-595", "595.71.05-1", "rpm", True),
            PackageInfo("nvidia-container-toolkit", "1.19.0-1", "rpm", True),
            PackageInfo("bash", "5.2-1", "rpm", True),
        ],
        "dnf",
    )
    assert commands == [
        [
            "dnf",
            "--disablerepo=*",
            "--disableplugin=versionlock",
            "--noautoremove",
            "--setopt=localpkg_gpgcheck=1",
            "install-nevra",
            "-y",
            "nvidia-open-595-595.71.05-1",
            "nvidia-container-toolkit-1.19.0-1",
        ]
    ]


def test_zypper_rollback_restores_versioned_rpm_packages():
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-open-595", "595.71.05-1", "rpm", True),
            PackageInfo("nvidia-container-toolkit", "1.19.0-1", "rpm", True),
            PackageInfo("bash", "5.2-1", "rpm", True),
        ],
        "zypper",
    )
    assert commands == [
        [
            "zypper",
            "--non-interactive",
            "--disable-repositories",
            "--no-refresh",
            "install",
            "--oldpackage",
            "--no-recommends",
            "--no-force-resolution",
            "--",
            "nvidia-open-595=595.71.05-1",
            "nvidia-container-toolkit=1.19.0-1",
        ]
    ]


def test_apt_rollback_restores_and_removes_in_one_solver_transaction():
    commands = _rollback_commands(
        [PackageInfo("nvidia-driver-570", "570.1-1", "apt", True)],
        "apt-get",
        remove_packages=["nvidia-driver-580-open", "cuda-compat-13-0"],
    )
    assert commands == [
        [
            "apt-get",
            "install",
            "-y",
            "--allow-change-held-packages",
            "--allow-downgrades",
            "--no-download",
            "--no-install-recommends",
            "--purge",
            "nvidia-driver-570=570.1-1",
            "cuda-compat-13-0-",
            "nvidia-driver-580-open-",
        ]
    ]


def test_zypper_rollback_removes_packages_absent_from_snapshot():
    commands = _rollback_commands([], "zypper", remove_packages=["nvidia-open-580"])
    assert commands == [
        [
            "zypper",
            "--non-interactive",
            "--disable-repositories",
            "--no-refresh",
            "install",
            "--oldpackage",
            "--no-recommends",
            "--no-force-resolution",
            "--",
            "-nvidia-open-580",
        ]
    ]


def test_load_snapshot_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="missing required"):
        load_snapshot(str(path))


def test_load_snapshot_accepts_valid_snapshot(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(
        path,
        package_manager="zypper",
        packages=[
            PackageInfo(
                "nvidia-open",
                "595.71.05-1",
                "rpm",
                True,
                architecture="x86_64",
            )
        ],
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    snapshot = load_snapshot(str(path))
    assert snapshot.kernel == "6.8.0-111-generic"
    assert snapshot.packages[0].name == "nvidia-open"
    assert snapshot.commands[0][0] == "zypper"


def test_load_snapshot_rejects_legacy_snapshot_without_trust_metadata(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "path": "/var/lib/nvidia-converge/snapshots/example.json",
                "packages": [
                    {
                        "name": "cuda-cub",
                        "version": "",
                        "manager": "apt",
                        "installed": True,
                    },
                    {
                        "name": "libnvidia-ml.so.1",
                        "version": "",
                        "manager": "",
                        "installed": True,
                    },
                ],
                "kernel": "6.8.0-111-generic",
                "module_version": "595.71.05",
                "commands": [
                    ["apt-get", "install", "-y", "nvidia-driver-595=595.71.05-1"]
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RollbackSnapshotError, match="missing required field"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_invalid_package_entry(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["packages"][0]["installed"] = "yes"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match=r"packages\[0\].installed"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_package_without_exact_architecture(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["packages"][0]["architecture"] = None
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RollbackSnapshotError, match="architecture must be known"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_invalid_command_entry(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["commands"] = [["zypper", ""]]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match=r"commands\[0\] entries"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_unsupported_command(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["commands"] = [["sh", "-c", "id"]]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="not a supported rollback command"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_rollback_option_operand(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["commands"] = [
        ["apt-get", "install", "-y", "-o", "Dpkg::Options::=--force-confold"]
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="not a supported rollback command"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_empty_rollback_command_specs(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["commands"] = [["dnf", "downgrade", "-y"]]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="not a supported rollback command"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_unknown_or_inconsistent_module_graph(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["module_loaded"] = True
    document["module_names"] = ["nvidia", "nvidia_vgpu_vfio"]
    snapshot = RollbackSnapshot(
        **{key: value for key, value in document.items() if key != "integrity_sha256"}
    )
    document["integrity_sha256"] = _snapshot_integrity(snapshot)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RollbackSnapshotError, match="module_names"):
        load_snapshot(str(path))


def test_load_snapshot_accepts_supported_remove_command(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path, introduced_packages=["nvidia-open"])
    path.write_text(json.dumps(document), encoding="utf-8")
    snapshot = load_snapshot(str(path))
    assert snapshot.commands[0][-1] == "nvidia-open-"


def test_apply_rollback_stops_after_package_failure_without_module_or_services():
    snapshot = RollbackSnapshot(
        path=None,
        packages=[PackageInfo("nvidia-driver-570", "570.1", "apt", True)],
        kernel="6.8.0-test",
        module_version=None,
        commands=[],
        package_manager="apt-get",
        introduced_packages=["nvidia-open"],
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
    )
    runner = _FakeRunner([100])
    results = apply_rollback(snapshot, runner)
    commands = [result.command for result in results]
    assert commands[:4] == _quarantine_commands(include_disable=False)
    assert commands[4] == [
        "apt-get",
        "install",
        "-y",
        "--allow-change-held-packages",
        "--allow-downgrades",
        "--no-download",
        "--no-install-recommends",
        "--purge",
        "nvidia-driver-570=570.1",
        "nvidia-open-",
    ]
    assert len(results) == 5


def test_package_failure_prevents_uuid_bound_mig_mutation():
    from test_planner import _audit

    audit = _audit()
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "apt",
            True,
            architecture="amd64",
        )
    ]
    audit.docker_service_active = False
    audit.docker_service_enabled = False
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    audit.fabric_manager_enabled = False
    audit.fabric_manager_active = False
    audit.fabric_manager_enabled = False
    audit.mig_mode = "enabled"
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        package_manager="apt-get",
        mig_mode="disabled",
        docker_service_active=False,
        docker_service_enabled=False,
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([100])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert _non_quarantine_commands(results)[0][0:2] == [
        "apt-get",
        "install",
    ]
    assert not any(result.command[0] == "nvidia-smi" for result in results)


def test_rollback_recreates_exact_supported_mig_baseline_geometry():
    from test_planner import _audit

    audit = _audit()
    geometry = [_full_mig_geometry()]
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=audit.module.loaded,
        commands=[],
        package_manager=audit.package_manager,
        mig_mode="enabled",
        gpu_uuids=list(audit.gpu_uuids),
        mig_geometry=geometry,
    )
    runner = _FakeRunner([0, 0])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert _non_quarantine_commands(results) == [
        [
            "nvidia-smi",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-mig",
            "1",
        ],
        [
            "nvidia-smi",
            "mig",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-cgi",
            "0:0",
            "-C",
        ],
    ]


def test_rollback_destroys_current_instances_before_restoring_disabled_mode():
    from test_planner import _audit

    audit = _audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "enabled"
    audit.mig_geometry = [_full_mig_geometry()]
    audit.mig_device_uuids = ["MIG-bbbbbbbbbbbbbbbb"]
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=audit.module.loaded,
        commands=[],
        package_manager=audit.package_manager,
        mig_mode="disabled",
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([0, 0, 0])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert _non_quarantine_commands(results) == [
        [
            "nvidia-smi",
            "mig",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-dci",
        ],
        [
            "nvidia-smi",
            "mig",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-dgi",
        ],
        [
            "nvidia-smi",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-mig",
            "0",
        ],
    ]


def test_package_scripts_run_before_managed_file_restoration(tmp_path):
    from test_planner import _audit

    path = tmp_path / "daemon.json"
    path.write_text("changed\n", encoding="utf-8")
    audit = _audit()
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "apt",
            True,
            architecture="amd64",
        )
    ]
    audit.docker_service_active = False
    audit.docker_service_enabled = False
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    audit.fabric_manager_active = False
    audit.fabric_manager_enabled = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        package_manager="apt-get",
        managed_files=[
            FileSnapshot(
                str(path),
                True,
                base64.b64encode(b"baseline\n").decode("ascii"),
                0o600,
            )
        ],
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([0])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    commands = _non_quarantine_commands(results)
    assert commands[0][:2] == ["apt-get", "install"]
    assert commands[1] == ["restore-file", str(path)]
    assert path.read_text(encoding="utf-8") == "baseline\n"


@pytest.mark.parametrize("post_mask_load_state", ["loaded", "masked"])
def test_active_socket_is_masked_rebound_stopped_and_proven_inactive(
    monkeypatch,
    post_mask_load_state,
):
    events = []

    class ActiveMaskedSocketRunner:
        apply = True

        def __init__(self):
            self.active = True
            self.unit_file_state = "enabled"
            self.load_state = "loaded"
            self.mutations = []

        @staticmethod
        def exists(name):
            return name == "systemctl"

        def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
            del allow_fail, input_text
            if command[:2] == ["systemctl", "show"]:
                return CommandResult(
                    command,
                    0,
                    stdout=(
                        "Id=docker.socket\n"
                        f"LoadState={self.load_state}\n"
                        f"ActiveState={'active' if self.active else 'inactive'}\n"
                        f"UnitFileState={self.unit_file_state}\n"
                    ),
                )
            assert mutate is True
            self.mutations.append(command)
            if command == ["systemctl", "mask", "docker.socket"]:
                events.append("mask")
                self.unit_file_state = "masked"
                self.load_state = post_mask_load_state
            elif command == ["systemctl", "stop", "docker.socket"]:
                events.append("stop")
                self.active = False
            else:
                raise AssertionError(f"unexpected command: {command}")
            return CommandResult(command, 0)

    identity = object()

    def validate(*args, **kwargs):
        del args, kwargs
        events.append("validate")
        return [CommandResult(["validate", "docker.socket"], 0)], identity, None

    def revalidate(runner, observed_identity):
        assert runner.load_state == post_mask_load_state
        assert runner.active is True
        assert observed_identity is identity
        events.append("revalidate")
        return [CommandResult(["revalidate", "docker.socket"], 0)], None

    monkeypatch.setattr(
        "nvidia_converge.rollback.validate_trusted_docker_socket_unit_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback.revalidate_trusted_docker_socket_identity",
        revalidate,
    )
    runner = ActiveMaskedSocketRunner()

    results = _quarantine_service_for_rollback(runner, "docker.socket")

    assert all(result.returncode in (0, None) for result in results)
    assert runner.mutations == [
        ["systemctl", "mask", "docker.socket"],
        ["systemctl", "stop", "docker.socket"],
    ]
    assert events == ["validate", "mask", "revalidate", "stop"]
    assert runner.active is False
    assert runner.unit_file_state == "masked"


def test_unsafe_active_service_hook_is_never_stopped_or_masked(monkeypatch):
    class ActiveServiceRunner:
        apply = True

        def __init__(self):
            self.mutations = []

        def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
            del allow_fail, input_text
            if command[:2] == ["systemctl", "show"]:
                return CommandResult(
                    command,
                    0,
                    stdout=(
                        "Id=docker.service\n"
                        "LoadState=loaded\n"
                        "ActiveState=active\n"
                        "UnitFileState=enabled\n"
                    ),
                )
            if mutate:
                self.mutations.append(command)
            raise AssertionError("an unsafe service must not be mutated")

    validation = CommandResult(["validate", "docker.service"], 1)
    monkeypatch.setattr(
        "nvidia_converge.rollback.validate_active_trusted_gpu_service_identity",
        lambda *args, **kwargs: (
            [validation],
            None,
            "cannot trust docker.service: unsafe ExecStopPost hook",
        ),
    )
    runner = ActiveServiceRunner()

    results = _quarantine_service_for_rollback(runner, "docker.service")

    assert results[-1] is validation
    assert runner.mutations == []


def test_applied_rollback_requarantines_service_state_after_package_phase(
    monkeypatch,
):
    from test_planner import _audit

    audit = _audit()
    audit.docker_service_active = False
    audit.docker_service_enabled = False
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    audit.fabric_manager_active = False
    audit.fabric_manager_enabled = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=False,
        module_signed=True,
        commands=[],
        package_manager="apt-get",
        mig_mode=audit.mig_mode,
        docker_service_active=False,
        docker_service_enabled=True,
        docker_socket_active=False,
        docker_socket_enabled=False,
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        gpu_uuids=list(audit.gpu_uuids),
    )
    snapshot = _bind_snapshot_service_state(snapshot, audit)
    snapshot.docker_service_enabled = True
    snapshot.docker_service_unit_file_state = "enabled"
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    quarantined: list[str] = []

    def quarantine(_runner, service):
        quarantined.append(service)
        return [CommandResult(["quarantine-service", service], 0)]

    monkeypatch.setattr(
        "nvidia_converge.rollback._quarantine_service_for_rollback",
        quarantine,
    )
    runner = _FakeRunner([])
    runner.apply = True
    monkeypatch.setattr(
        "nvidia_converge.rollback.nvidia_module_unload_order",
        list,
    )

    results = apply_rollback(snapshot, runner, current_audit=audit)

    commands = [result.command for result in results]
    units = [
        "docker.socket",
        "docker.service",
        "nvidia-persistenced.service",
        "nvidia-fabricmanager.service",
    ]
    assert quarantined == [*units, *units]
    assert commands.count(["quarantine-service", "docker.socket"]) == 2
    assert commands[-1] == ["rollback", "service-state"]


def test_applied_rollback_fails_closed_on_ambiguous_fresh_service_state(
    monkeypatch,
):
    from test_planner import _audit

    audit = _audit()
    audit.docker_service_active = False
    audit.docker_service_enabled = False
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    audit.fabric_manager_active = False
    audit.fabric_manager_enabled = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=False,
        module_signed=True,
        commands=[],
        package_manager="apt-get",
        docker_service_active=False,
        docker_service_enabled=False,
        docker_socket_active=False,
        docker_socket_enabled=False,
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        gpu_uuids=list(audit.gpu_uuids),
    )
    snapshot = _bind_snapshot_service_state(snapshot, audit)
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    ambiguous = CommandResult(
        [],
        0,
        stdout=(
            "Id=docker.socket\n"
            "LoadState=loaded\n"
            "ActiveState=activating\n"
            "UnitFileState=disabled\n"
        ),
    )
    runner = _FakeRunner([ambiguous])
    runner.apply = True
    monkeypatch.setattr(
        "nvidia_converge.rollback.nvidia_module_unload_order",
        list,
    )

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert results[-1].command == [
        "rollback-precondition",
        "service-quarantine",
        "docker.socket",
    ]
    assert results[-1].returncode == 1
    assert not any(result.command[0] == "nvidia-smi" for result in results)


def test_applied_rollback_rejects_gpu_topology_change_before_mutation(monkeypatch):
    from test_planner import _audit

    audit = _audit()
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=None,
        commands=[],
        package_manager="apt-get",
        gpu_uuids=["GPU-bbbbbbbbbbbbbbbb"],
    )
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    runner = _FakeRunner([])
    runner.apply = True

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert results[-1].command == ["rollback-precondition", "gpu-inventory"]
    assert runner.calls == []


def test_rollback_refuses_multi_gpu_mig_transition():
    from test_planner import _audit

    audit = _audit()
    audit.gpu_uuids = [
        "GPU-aaaaaaaaaaaaaaaa",
        "GPU-bbbbbbbbbbbbbbbb",
    ]
    audit.mig_mode = "enabled"
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=None,
        commands=[],
        package_manager="apt-get",
        mig_mode="disabled",
        gpu_uuids=list(audit.gpu_uuids),
    )
    snapshot = _bind_snapshot_service_state(snapshot, audit)
    runner = _FakeRunner([])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert results[0].command == [
        "rollback-precondition",
        "mig-transaction-scope",
    ]
    assert runner.calls == []


def test_apply_rollback_skips_state_that_already_matches_current_audit():
    from test_planner import _audit

    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "580.1"
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=audit.module.loaded,
        module_open_module=audit.module.open_module,
        module_signed=audit.module.signed,
        module_installed_version=audit.module.installed_version,
        module_installed_open_module=audit.module.installed_open_module,
        module_installed_signed=audit.module.installed_signed,
        commands=[],
        package_manager=audit.package_manager,
        mig_mode=audit.mig_mode,
        gpu_uuids=list(audit.gpu_uuids),
    )
    snapshot = _bind_snapshot_service_state(snapshot, audit)
    runner = _FakeRunner([])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert len(results) == 1
    assert results[0].command == ["rollback"]
    assert results[0].reason == "already-restored"


def test_apply_rollback_reloads_module_when_version_changed():
    from test_planner import _audit

    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "580.1"
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version="570.1",
        module_loaded=True,
        commands=[],
        package_manager="apt-get",
        docker_service_active=False,
        docker_service_enabled=False,
        docker_socket_active=False,
        docker_socket_enabled=False,
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([0, 0])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    commands = _non_quarantine_commands(results)
    assert commands[0][:2] == ["modprobe", "-r"]
    assert commands[1] == ["modprobe", "nvidia"]


def test_apply_rollback_reloads_same_version_module_when_flavor_changed():
    from test_planner import _audit

    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "580.1"
    audit.module.open_module = False
    audit.module.signed = True
    audit.module.installed_version = "580.1"
    audit.module.installed_open_module = False
    audit.module.installed_signed = True
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version="580.1",
        module_loaded=True,
        module_open_module=True,
        module_signed=True,
        module_installed_version="580.1",
        module_installed_open_module=True,
        module_installed_signed=True,
        commands=[],
        package_manager="apt-get",
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([0, 0])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    commands = _non_quarantine_commands(results)
    assert commands[0][:2] == ["modprobe", "-r"]
    assert commands[1] == ["modprobe", "nvidia"]


def test_apply_rollback_restores_exact_dependent_module_set():
    from test_planner import _audit

    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "580.1"
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version="570.1",
        module_loaded=True,
        module_names=["nvidia_uvm", "nvidia"],
        commands=[],
        package_manager="apt-get",
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([0, 0, 0])

    results = apply_rollback(snapshot, runner, current_audit=audit)

    commands = _non_quarantine_commands(results)
    assert commands[-3][:2] == ["modprobe", "-r"]
    assert commands[-2] == ["modprobe", "nvidia"]
    assert commands[-1] == ["modprobe", "nvidia_uvm"]


def test_applied_rollback_fails_before_mutation_on_unknown_module_dependent(
    monkeypatch,
):
    from test_planner import _audit

    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "580.1"
    audit.docker_service_active = False
    audit.fabric_manager_active = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version="570.1",
        module_loaded=True,
        commands=[],
        package_manager="apt-get",
        gpu_uuids=list(audit.gpu_uuids),
    )
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    runner = _FakeRunner([])
    runner.apply = True
    monkeypatch.setattr(
        "nvidia_converge.rollback.nvidia_module_unload_order",
        lambda: (_ for _ in ()).throw(
            ModuleDependencyError("unsupported dependent: nvidia_vgpu_vfio")
        ),
    )

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert len(results) == 1
    assert results[0].command == ["inspect-module-dependencies"]
    assert results[0].returncode == 1
    assert runner.calls == []


def test_applied_rollback_checks_all_service_states_before_stopping_any(
    monkeypatch,
):
    from test_planner import _audit

    audit = _audit()
    audit.fabric_manager_active = True
    audit.docker_service_active = None
    audit.docker_service_enabled = False
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version="570.1",
        module_loaded=True,
        commands=[],
        package_manager="apt-get",
        docker_service_active=False,
        docker_service_enabled=False,
        docker_socket_active=False,
        docker_socket_enabled=False,
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        gpu_uuids=list(audit.gpu_uuids),
    )
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    runner = _FakeRunner([])
    runner.apply = True
    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert results[0].command == ["rollback-precondition", "package-solver"]
    assert results[1].command == [
        "rollback-precondition",
        "current-service-state",
    ]
    assert results[1].returncode == 1
    assert runner.calls == []


def test_applied_rollback_rejects_unmanaged_file_before_stopping_services(
    monkeypatch,
):
    from test_planner import _audit

    audit = _audit()
    audit.docker_service_active = True
    audit.fabric_manager_active = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version="570.1",
        module_loaded=True,
        commands=[],
        package_manager="apt-get",
        managed_files=[FileSnapshot("/tmp/not-managed", False, None, None)],
        gpu_uuids=list(audit.gpu_uuids),
    )
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    runner = _FakeRunner([])
    runner.apply = True
    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)

    results = apply_rollback(snapshot, runner, current_audit=audit)

    assert results[0].command == ["rollback-precondition", "managed-files"]
    assert results[0].returncode == 1
    assert runner.calls == []


def test_apply_rollback_restores_managed_file_content_and_mode(tmp_path):
    path = tmp_path / "daemon.json"
    path.write_text("changed\n", encoding="utf-8")
    path.chmod(0o644)
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel="6.8.0-test",
        module_version=None,
        commands=[],
        package_manager="apt-get",
        managed_files=[
            FileSnapshot(
                str(path),
                True,
                base64.b64encode(b"original\n").decode("ascii"),
                0o600,
            )
        ],
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
    )
    runner = _FakeRunner([0])

    results = apply_rollback(snapshot, runner)

    restore_result = next(
        result for result in results if result.command == ["restore-file", str(path)]
    )
    assert restore_result.returncode == 0
    assert path.read_text(encoding="utf-8") == "original\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("failsafe_exists", [False, True])
def test_capture_managed_files_includes_exact_dnf_failsafe_target(
    monkeypatch,
    failsafe_exists,
):
    target = (
        "/var/lib/dnf/modulefailsafe/"
        "nvidia-driver:580-open:x86_64.yaml"
    )
    calls: list[str] = []

    def fake_read(path, **_kwargs):
        path_text = str(path)
        calls.append(path_text)
        if path_text == target and not failsafe_exists:
            raise FileNotFoundError(path_text)
        return (
            f"baseline:{path_text}\n",
            SimpleNamespace(st_mode=stat.S_IFREG | 0o640),
        )

    monkeypatch.setattr(
        "nvidia_converge.rollback.read_bounded_utf8_with_metadata",
        fake_read,
    )

    files = _capture_managed_files(
        "dnf",
        dnf_module_failsafe_path=target,
    )

    assert calls == [
        "/etc/docker/daemon.json",
        "/etc/dnf/modules.d/nvidia-driver.module",
        target,
    ]
    failsafe = files[-1]
    assert failsafe.path == target
    assert failsafe.existed is failsafe_exists
    if failsafe_exists:
        assert failsafe.mode == 0o640
        assert base64.b64decode(failsafe.content_base64 or "").decode() == (
            f"baseline:{target}\n"
        )
    else:
        assert failsafe.content_base64 is None
        assert failsafe.mode is None


def test_load_managed_files_requires_one_canonical_ordered_dnf_failsafe():
    target = (
        "/var/lib/dnf/modulefailsafe/"
        "nvidia-driver:580-open:x86_64.yaml"
    )
    other_target = (
        "/var/lib/dnf/modulefailsafe/"
        "nvidia-driver:580-open:aarch64.yaml"
    )

    def absent(path):
        return {
            "path": path,
            "existed": False,
            "content_base64": None,
            "mode": None,
        }

    canonical = [
        absent("/etc/docker/daemon.json"),
        absent("/etc/dnf/modules.d/nvidia-driver.module"),
        absent(target),
    ]
    assert [item.path for item in _load_managed_files(canonical, "dnf")] == [
        item["path"] for item in canonical
    ]

    invalid_sets = (
        canonical[:-1],
        [*canonical, absent(other_target)],
        [canonical[0], canonical[2], canonical[1]],
    )
    for invalid in invalid_sets:
        with pytest.raises(RollbackSnapshotError):
            _load_managed_files(invalid, "dnf")


def test_restore_managed_files_orders_dnf_module_before_failsafe():
    target = (
        "/var/lib/dnf/modulefailsafe/"
        "nvidia-driver:580-open:x86_64.yaml"
    )
    files = [
        FileSnapshot(target, False, None, None),
        FileSnapshot(
            "/etc/dnf/modules.d/nvidia-driver.module",
            False,
            None,
            None,
        ),
        FileSnapshot("/etc/docker/daemon.json", False, None, None),
    ]
    runner = _FakeRunner([])
    runner.apply = False

    results = _restore_managed_files(files, runner)

    assert [result.command for result in results] == [
        ["restore-file", "/etc/docker/daemon.json"],
        ["restore-file", "/etc/dnf/modules.d/nvidia-driver.module"],
        ["restore-file", target],
    ]


def test_rollback_precondition_rejects_tampered_dnf_failsafe_target(
    monkeypatch,
):
    target = (
        "/var/lib/dnf/modulefailsafe/"
        "nvidia-driver:580-open:x86_64.yaml"
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback._managed_file_matches",
        lambda _snapshot: False,
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback.trusted_path_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(
            st_mode=stat.S_IFLNK | 0o777,
            st_nlink=1,
        ),
    )

    result = _managed_file_restore_precondition(
        [FileSnapshot(target, False, None, None)]
    )

    assert result is not None
    assert result.returncode == 1
    assert "not a singly linked regular file" in result.stderr


def test_apply_rollback_quarantines_launchers_before_mig_and_defers_restore():
    from test_planner import _audit

    audit = _audit()
    audit.mig_mode = "enabled"
    audit.docker_service_active = True
    audit.docker_service_enabled = True
    audit.docker_service_unit_file_state = "enabled"
    audit.docker_socket_active = True
    audit.docker_socket_enabled = True
    audit.docker_socket_unit_file_state = "enabled"
    audit.nvidia_persistenced_active = True
    audit.nvidia_persistenced_enabled = True
    audit.nvidia_persistenced_unit_file_state = "enabled"
    audit.fabric_manager_active = False
    audit.fabric_manager_enabled = False
    audit.fabric_manager_unit_file_state = "not-found"
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=audit.module.loaded,
        commands=[],
        package_manager=audit.package_manager,
        mig_mode="disabled",
        docker_service_active=False,
        docker_service_enabled=False,
        docker_service_unit_file_state="disabled",
        docker_socket_active=False,
        docker_socket_enabled=False,
        docker_socket_unit_file_state="disabled",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="disabled",
        fabric_manager_active=True,
        fabric_manager_enabled=True,
        fabric_manager_unit_file_state="enabled",
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([0])

    results = apply_rollback(
        snapshot,
        runner,
        current_audit=audit,
    )
    commands = [result.command for result in results]

    assert commands == [
        ["systemctl", "disable", "--now", "docker.socket"],
        ["systemctl", "mask", "--now", "docker.socket"],
        ["systemctl", "disable", "--now", "docker.service"],
        ["systemctl", "mask", "--now", "docker.service"],
        ["systemctl", "disable", "--now", "nvidia-persistenced.service"],
        ["systemctl", "mask", "--now", "nvidia-persistenced.service"],
        ["systemctl", "mask", "--now", "nvidia-fabricmanager.service"],
        [
            "nvidia-smi",
            "-i",
            "GPU-aaaaaaaaaaaaaaaa",
            "-mig",
            "0",
        ],
        ["rollback", "service-state"],
    ]


def test_final_service_enablement_stops_after_enable_failure():
    from test_planner import _audit

    audit = _audit()
    audit.docker_service_active = False
    audit.docker_service_enabled = False
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    audit.fabric_manager_active = False
    audit.fabric_manager_enabled = False
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(
            path=None,
            packages=list(audit.packages),
            kernel=audit.kernel.running,
            module_version=audit.module.version,
            module_loaded=audit.module.loaded,
            module_open_module=audit.module.open_module,
            module_signed=audit.module.signed,
            module_installed_version=audit.module.installed_version,
            module_installed_open_module=audit.module.installed_open_module,
            module_installed_signed=audit.module.installed_signed,
            commands=[],
            package_manager=audit.package_manager,
            mig_mode=audit.mig_mode,
            docker_service_active=True,
            docker_service_enabled=True,
            docker_socket_active=False,
            docker_socket_enabled=False,
            nvidia_persistenced_active=False,
            nvidia_persistenced_enabled=False,
            fabric_manager_active=False,
            fabric_manager_enabled=False,
            gpu_uuids=list(audit.gpu_uuids),
        ),
        audit,
    )
    snapshot.docker_service_active = True
    snapshot.docker_service_enabled = True
    snapshot.docker_service_unit_file_state = "enabled"
    audit.docker_service_unit_file_state = "disabled"
    runner = _FakeRunner([1])

    results = restore_rollback_service_enablement(
        snapshot,
        runner,
        audit,
        units={"docker.service"},
    )

    assert [result.command for result in results] == [
        ["systemctl", "enable", "docker.service"]
    ]


def test_rollback_activity_can_be_deferred_until_core_verification():
    from test_planner import _audit

    audit = _audit()
    audit.docker_socket_active = False
    audit.docker_socket_enabled = False
    audit.nvidia_persistenced_active = False
    audit.nvidia_persistenced_enabled = False
    audit.fabric_manager_enabled = False
    audit.fabric_manager_unit_file_state = "not-found"
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=audit.module.loaded,
        module_open_module=audit.module.open_module,
        module_signed=audit.module.signed,
        module_installed_version=audit.module.installed_version,
        module_installed_open_module=audit.module.installed_open_module,
        module_installed_signed=audit.module.installed_signed,
        commands=[],
        package_manager=audit.package_manager,
        mig_mode=audit.mig_mode,
        docker_service_active=True,
        docker_service_enabled=False,
        docker_service_unit_file_state="disabled",
        docker_socket_active=True,
        docker_socket_enabled=False,
        docker_socket_unit_file_state="disabled",
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="disabled",
        fabric_manager_active=True,
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="disabled",
        gpu_uuids=list(audit.gpu_uuids),
    )
    runner = _FakeRunner([])

    results = apply_rollback(
        snapshot,
        runner,
        current_audit=audit,
    )

    commands = [result.command for result in results]
    assert commands[-1] == ["rollback", "service-state"]
    assert results[-1].reason == "deferred"
    assert not any(command[:2] == ["systemctl", "start"] for command in commands)


def test_staged_rollback_activity_uses_safe_launcher_commit_order():
    from test_planner import _audit

    audit = _audit()
    audit.fabric_manager_active = False
    audit.fabric_manager_unit_file_state = "disabled"
    audit.nvidia_persistenced_unit_file_state = "disabled"
    audit.docker_service_unit_file_state = "disabled"
    audit.docker_socket_unit_file_state = "disabled"
    audit.docker_socket_active = False
    audit.nvidia_persistenced_active = False
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=True,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=True,
        nvidia_persistenced_unit_file_state="enabled",
        fabric_manager_active=True,
        fabric_manager_enabled=True,
        fabric_manager_unit_file_state="enabled",
    )
    runner = _FakeRunner([0, 0, 0, 0])

    results = []
    for unit in (
        "nvidia-fabricmanager.service",
        "nvidia-persistenced.service",
        "docker.service",
        "docker.socket",
    ):
        results.extend(
            restore_rollback_service_activity(
                snapshot,
                runner,
                audit,
                units={unit},
            )
        )

    assert [result.command for result in results] == [
        ["systemctl", "start", "nvidia-fabricmanager.service"],
        ["systemctl", "start", "nvidia-persistenced.service"],
        ["systemctl", "start", "docker.service"],
        ["systemctl", "start", "docker.socket"],
    ]


def test_docker_socket_can_be_prepared_before_dependent_service():
    from test_planner import _audit

    audit = _audit()
    for field in (
        "docker_socket_unit_file_state",
        "docker_service_unit_file_state",
    ):
        setattr(audit, field, "masked")
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(None, [], audit.kernel.running, None, []),
        audit,
    )
    snapshot.docker_socket_enabled = True
    snapshot.docker_socket_unit_file_state = "enabled"
    snapshot.docker_service_enabled = True
    snapshot.docker_service_unit_file_state = "enabled"
    runner = _FakeRunner([0, 0, 0, 0])

    socket_results = prepare_rollback_service_activity(
        snapshot,
        runner,
        audit,
        units={"docker.socket"},
    )
    audit.docker_socket_unit_file_state = "disabled"
    service_results = prepare_rollback_service_activity(
        snapshot,
        runner,
        audit,
        units={"docker.service"},
    )

    assert [result.command for result in [*socket_results, *service_results]] == [
        ["systemctl", "unmask", "docker.socket"],
        ["systemctl", "disable", "docker.socket"],
        ["systemctl", "unmask", "docker.service"],
        ["systemctl", "disable", "docker.service"],
    ]


@pytest.mark.parametrize(
    ("target_state", "prepared_state", "expected_commands"),
    [
        ("enabled", "disabled", [["systemctl", "enable", "docker.service"]]),
        ("disabled", "disabled", []),
        ("static", "static", []),
        ("masked", "masked", []),
        ("not-found", "not-found", []),
    ],
)
def test_final_service_enablement_restores_exact_unit_file_state(
    target_state,
    prepared_state,
    expected_commands,
):
    from test_planner import _audit

    audit = _audit()
    audit.docker_service_unit_file_state = prepared_state
    audit.docker_service_enabled = prepared_state == "enabled"
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(None, [], audit.kernel.running, None, []),
        audit,
    )
    snapshot.docker_service_unit_file_state = target_state
    snapshot.docker_service_enabled = target_state == "enabled"
    runner = _FakeRunner([0] if expected_commands else [])

    results = restore_rollback_service_enablement(
        snapshot,
        runner,
        audit,
        units={"docker.service"},
    )

    assert [result.command for result in results] == expected_commands


def test_service_activity_requires_prepared_not_final_enabled_state():
    from test_planner import _audit

    audit = _audit()
    audit.docker_service_unit_file_state = "enabled"
    audit.docker_service_enabled = True
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(None, [], audit.kernel.running, None, []),
        audit,
    )
    snapshot.docker_service_active = True
    runner = _FakeRunner([])

    results = restore_rollback_service_activity(
        snapshot,
        runner,
        audit,
        units={"docker.service"},
    )

    assert results[0].command == [
        "rollback-precondition",
        "service-unit-file-state",
        "docker.service",
    ]
    assert runner.calls == []


@pytest.mark.parametrize("active", [False, True])
def test_static_service_is_stopped_then_masked_without_disable(active):
    from test_planner import _audit

    audit = _audit()
    audit.nvidia_persistenced_active = active
    audit.nvidia_persistenced_enabled = False
    audit.nvidia_persistenced_unit_file_state = "static"
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(None, [], audit.kernel.running, None, []),
        audit,
    )
    snapshot.docker_service_unit_file_state = "disabled"
    runner = _FakeRunner([])

    results = apply_rollback(snapshot, runner, current_audit=audit)
    commands = [result.command for result in results]

    stop_index = commands.index(["systemctl", "stop", "nvidia-persistenced.service"])
    assert commands[stop_index + 1] == [
        "systemctl",
        "mask",
        "--now",
        "nvidia-persistenced.service",
    ]
    assert [
        "systemctl",
        "disable",
        "--now",
        "nvidia-persistenced.service",
    ] not in commands


def test_not_found_service_is_masked_without_stop_or_disable():
    from test_planner import _audit

    audit = _audit()
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(None, [], audit.kernel.running, None, []),
        audit,
    )
    snapshot.docker_service_unit_file_state = "disabled"
    runner = _FakeRunner([])

    results = apply_rollback(snapshot, runner, current_audit=audit)
    commands = [result.command for result in results]

    assert [
        "systemctl",
        "mask",
        "--now",
        "nvidia-fabricmanager.service",
    ] in commands
    assert ["systemctl", "stop", "nvidia-fabricmanager.service"] not in commands
    assert [
        "systemctl",
        "disable",
        "--now",
        "nvidia-fabricmanager.service",
    ] not in commands


def test_pending_only_mig_target_defers_geometry_until_reboot(monkeypatch):
    from test_planner import _audit

    audit = _audit()
    for active_field, enabled_field, state_field in (
        (
            "docker_service_active",
            "docker_service_enabled",
            "docker_service_unit_file_state",
        ),
        (
            "docker_socket_active",
            "docker_socket_enabled",
            "docker_socket_unit_file_state",
        ),
        (
            "nvidia_persistenced_active",
            "nvidia_persistenced_enabled",
            "nvidia_persistenced_unit_file_state",
        ),
        (
            "fabric_manager_active",
            "fabric_manager_enabled",
            "fabric_manager_unit_file_state",
        ),
    ):
        setattr(audit, active_field, False)
        setattr(audit, enabled_field, False)
        setattr(audit, state_field, "masked")
    snapshot = _bind_snapshot_service_state(
        RollbackSnapshot(
            None,
            list(audit.packages),
            audit.kernel.running,
            audit.module.version,
            [],
            module_loaded=audit.module.loaded,
            module_open_module=audit.module.open_module,
            module_signed=audit.module.signed,
            module_installed_version=audit.module.installed_version,
            module_installed_open_module=audit.module.installed_open_module,
            module_installed_signed=audit.module.installed_signed,
            package_manager=audit.package_manager,
            mig_mode="enabled",
            gpu_uuids=list(audit.gpu_uuids),
            mig_geometry=[_full_mig_geometry()],
        ),
        audit,
    )
    snapshot.mig_mode = "enabled"
    snapshot.mig_geometry = [_full_mig_geometry()]
    _allow_downstream_applied_rollback(monkeypatch, snapshot)
    monkeypatch.setattr(
        "nvidia_converge.rollback._quarantine_service_for_rollback",
        lambda _runner, service: [CommandResult(["quarantine-service", service], 0)],
    )
    runner = _MigPendingRunner(audit.gpu_uuids[0])
    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)

    results = apply_rollback(snapshot, runner, current_audit=audit)
    commands = [result.command for result in results]

    assert runner.mig_query_count == 2
    assert ["nvidia-smi", "-i", audit.gpu_uuids[0], "-mig", "1"] in commands
    assert not any("-cgi" in command for command in commands)
    assert commands[-1] == ["rollback", "mig-reboot-pending"]
    assert results[-1].reason == "reboot-required"


@pytest.mark.parametrize(
    "field",
    [
        "docker_socket_active",
        "docker_socket_enabled",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
    ],
)
def test_snapshot_fails_closed_on_unknown_transactional_service_state(
    tmp_path,
    monkeypatch,
    field,
):
    from test_planner import _audit

    audit = _audit()
    setattr(audit, field, None)
    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)

    with pytest.raises(
        RollbackSnapshotError,
        match="without exact transactional service state",
    ):
        create_snapshot(audit, str(tmp_path / "snapshot.json"))


def test_persisted_snapshot_is_private_and_loadable(tmp_path, monkeypatch):
    from test_planner import _audit

    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)
    monkeypatch.setattr(
        "nvidia_converge.rollback._capture_managed_files",
        lambda _package_manager: [
            FileSnapshot("/etc/docker/daemon.json", False, None, None)
        ],
    )
    path = tmp_path / "snapshot.json"
    audit = _audit()
    package_payloads = _write_test_payload_bundle(path, audit.packages)
    snapshot = create_snapshot(
        audit,
        str(path),
        package_payloads=package_payloads,
    )
    assert snapshot.path == str(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_snapshot(str(path)).kernel == snapshot.kernel


def test_snapshot_producer_rejects_oversized_serialized_document(
    tmp_path,
    monkeypatch,
):
    from test_planner import _audit

    assert MAX_SNAPSHOT_BYTES > 256
    monkeypatch.setattr("nvidia_converge.rollback.MAX_SNAPSHOT_BYTES", 256)
    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)
    monkeypatch.setattr(
        "nvidia_converge.rollback._capture_managed_files",
        lambda _package_manager, **_kwargs: [
            FileSnapshot("/etc/docker/daemon.json", False, None, None)
        ],
    )
    path = tmp_path / "snapshot.json"
    audit = _audit()
    package_payloads = _write_test_payload_bundle(path, audit.packages)

    with pytest.raises(
        RollbackSnapshotError,
        match="exceeds the supported serialized size",
    ):
        create_snapshot(
            audit,
            str(path),
            package_payloads=package_payloads,
        )

    assert not path.exists()


def test_schema_26_dnf_snapshot_round_trips_dynamic_failsafe_target(tmp_path):
    path = tmp_path / "dnf-snapshot.json"
    document = _snapshot_document(
        path,
        package_manager="dnf",
        packages=[
            PackageInfo(
                "nvidia-open-580",
                "580.126.16-1",
                "rpm",
                True,
                architecture="x86_64",
            )
        ],
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    snapshot = load_snapshot(str(path))

    assert [item.path for item in snapshot.managed_files] == [
        "/etc/docker/daemon.json",
        "/etc/dnf/modules.d/nvidia-driver.module",
        (
            "/var/lib/dnf/modulefailsafe/"
            "nvidia-driver:580-open:x86_64.yaml"
        ),
    ]


def test_snapshot_fails_closed_when_mig_pending_state_is_unobservable(tmp_path):
    from test_planner import _audit

    audit = _audit()
    audit.mig_mode_pending = None

    with pytest.raises(RollbackSnapshotError, match="pending MIG mode"):
        create_snapshot(audit, str(tmp_path / "snapshot.json"))


def test_snapshot_fails_closed_during_pending_mig_transition(tmp_path):
    from test_planner import _audit

    audit = _audit()
    audit.mig_mode_pending = "enabled"

    with pytest.raises(RollbackSnapshotError, match="transition is pending"):
        create_snapshot(audit, str(tmp_path / "snapshot.json"))


def test_snapshot_records_removal_of_new_target_packages(tmp_path, monkeypatch):
    from test_planner import _audit

    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)
    monkeypatch.setattr(
        "nvidia_converge.rollback._capture_managed_files",
        lambda _package_manager, **_kwargs: [],
    )
    path = tmp_path / "snapshot.json"
    audit = _audit()
    package_payloads = _write_test_payload_bundle(path, audit.packages)
    snapshot = create_snapshot(
        audit,
        str(path),
        desired=DesiredState(),
        package_payloads=package_payloads,
    )
    assert len(snapshot.commands) == 1
    transaction = snapshot.commands[0]
    assert transaction[:8] == [
        "apt-get",
        "install",
        "-y",
        "--allow-change-held-packages",
        "--allow-downgrades",
        "--no-download",
        "--no-install-recommends",
        "--purge",
    ]
    assert "nvidia-driver-pinning-580-" in transaction
    assert "nvidia-open-" in transaction
    assert f"linux-headers-{snapshot.kernel}-" in transaction


@pytest.mark.parametrize(
    (
        "package_manager",
        "os_id",
        "os_version",
        "kernel",
        "package_kind",
        "selector",
    ),
    [
        (
            "apt-get",
            "ubuntu",
            "24.04",
            "6.8.0-64-generic",
            "apt",
            PackagePolicySelector(
                "nvidia-driver-pinning-580",
                "nvidia-driver-pinning-580",
                "package",
            ),
        ),
        (
            "dnf",
            "rhel",
            "9.5",
            "5.14.0-503.el9_5.x86_64",
            "rpm",
            PackagePolicySelector(
                "nvidia-driver",
                "nvidia-driver",
                "module",
                "stream",
                "580-open",
            ),
        ),
        (
            "zypper",
            "sles",
            "15.6",
            "6.4.0-150600.23.53-default",
            "rpm",
            PackagePolicySelector(
                "1",
                "*nvidia*",
                "package",
                "ge",
                "590",
            ),
        ),
    ],
)
def test_snapshot_tracks_missing_toolkit_dependency_closure(
    tmp_path,
    monkeypatch,
    package_manager,
    os_id,
    os_version,
    kernel,
    package_kind,
    selector,
):
    from test_planner import _audit

    audit = _audit()
    audit.package_manager = package_manager
    audit.os_id = os_id
    audit.os_version = os_version
    audit.kernel.running = kernel
    audit.kernel.headers_installed = True
    audit.kernel.compiler = "gcc"
    audit.module.loaded = True
    audit.module.version = "580.126.16"
    audit.module.open_module = True
    audit.module.signed = True
    audit.module.installed_version = "580.126.16"
    audit.module.installed_open_module = True
    audit.module.installed_signed = True
    audit.runtime = RuntimeInfo(True, False, False)
    audit.nvidia_smi = CommandResult(["nvidia-smi"], 0)
    audit.nvml = CommandResult(["python3"], 0)
    audit.packages = [
        PackageInfo(
            "libnvidia-container1",
            "1.19.1-1",
            package_kind,
            True,
            "x86_64",
        )
    ]
    audit.package_policy = PackagePolicyInfo(
        package_manager,
        True,
        [selector],
    )

    monkeypatch.setattr(
        "nvidia_converge.rollback.nvidia_module_unload_order",
        lambda: ["nvidia"],
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback._capture_managed_files",
        lambda _package_manager, **_kwargs: [],
    )
    snapshot_path = tmp_path / f"{package_manager}.json"
    package_payloads = _write_test_payload_bundle(snapshot_path, audit.packages)
    snapshot = create_snapshot(
        audit,
        str(snapshot_path),
        desired=DesiredState(fabric_manager=False),
        package_payloads=package_payloads,
        dnf_module_failsafe_path=(
            "/var/lib/dnf/modulefailsafe/"
            "nvidia-driver:580-open:x86_64.yaml"
            if package_manager == "dnf"
            else None
        ),
    )

    assert set(snapshot.introduced_packages) == {
        "nvidia-container-toolkit",
        "nvidia-container-toolkit-base",
        "libnvidia-container-tools",
    }
    assert "libnvidia-container1" not in snapshot.introduced_packages


def test_automatic_snapshot_paths_are_unique(monkeypatch, tmp_path):
    monkeypatch.setattr("nvidia_converge.rollback.SNAPSHOT_DIR", tmp_path)
    first = new_snapshot_path()
    second = new_snapshot_path()
    assert first != second
    assert not first.exists()
    assert not second.exists()


def test_host_identity_hashes_machine_id_before_reporting(monkeypatch):
    machine_id = "0123456789abcdef0123456789abcdef"
    monkeypatch.setattr(
        "nvidia_converge.rollback.read_bounded_utf8",
        lambda path, max_bytes: machine_id,
    )

    identity = _host_identity()

    assert identity == (
        "machine-id-sha256:" + hashlib.sha256(machine_id.encode("ascii")).hexdigest()
    )
    assert machine_id not in identity


def test_verify_rollback_compares_snapshot_package_module_and_removals():
    from test_planner import _audit

    audit = _audit()
    audit.packages = [PackageInfo("nvidia-driver-570", "570.1-1", "apt", True)]
    audit.module.version = "570.1"
    snapshot = RollbackSnapshot(
        path=None,
        packages=[PackageInfo("nvidia-driver-570", "570.1-1", "apt", True)],
        kernel=audit.kernel.running,
        module_version="570.1",
        module_open_module=audit.module.open_module,
        module_signed=audit.module.signed,
        module_installed_version=audit.module.installed_version,
        module_installed_open_module=audit.module.installed_open_module,
        module_installed_signed=audit.module.installed_signed,
        commands=[
            [
                "apt-get",
                "remove",
                "-y",
                "--allow-change-held-packages",
                "nvidia-driver-580-open",
            ]
        ],
        gpu_uuids=list(audit.gpu_uuids),
    )
    assert all(check.ok for check in verify_rollback(snapshot, audit))


def test_verify_rollback_can_defer_exact_service_checks_until_commit():
    from test_planner import _audit

    audit = _audit()
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        module_loaded=audit.module.loaded,
        module_open_module=audit.module.open_module,
        module_signed=audit.module.signed,
        module_installed_version=audit.module.installed_version,
        module_installed_open_module=audit.module.installed_open_module,
        module_installed_signed=audit.module.installed_signed,
        commands=[],
        mig_mode=audit.mig_mode,
        docker_service_active=True,
        docker_socket_active=True,
        nvidia_persistenced_active=True,
        gpu_uuids=list(audit.gpu_uuids),
    )

    core_checks = verify_rollback(
        snapshot,
        audit,
        include_service_state=False,
    )
    full_checks = verify_rollback(snapshot, audit)

    assert not any(
        "service" in check.name or "persistenced" in check.name for check in core_checks
    )
    assert (
        next(
            check
            for check in full_checks
            if check.name == "rollback.docker-socket-active"
        ).ok
        is False
    )
    assert (
        next(
            check
            for check in full_checks
            if check.name == "rollback.nvidia-persistenced-active"
        ).ok
        is False
    )


def test_verify_rollback_fails_when_added_package_remains():
    from test_planner import _audit

    audit = _audit()
    audit.packages = [PackageInfo("nvidia-driver-580-open", "580.1-1", "apt", True)]
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        package_manager="apt-get",
        introduced_packages=["nvidia-driver-580-open"],
        gpu_uuids=list(audit.gpu_uuids),
    )
    checks = verify_rollback(snapshot, audit)
    assert (
        next(
            check for check in checks if check.name == "rollback.added-packages-removed"
        ).ok
        is False
    )


def test_verify_rollback_rejects_same_version_wrong_module_flavor():
    from test_planner import _audit

    audit = _audit()
    audit.module.loaded = True
    audit.module.version = "580.1"
    audit.module.open_module = False
    audit.module.signed = True
    audit.module.installed_version = "580.1"
    audit.module.installed_open_module = False
    audit.module.installed_signed = True
    snapshot = RollbackSnapshot(
        path=None,
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version="580.1",
        module_loaded=True,
        module_open_module=True,
        module_signed=True,
        module_installed_version="580.1",
        module_installed_open_module=True,
        module_installed_signed=True,
        commands=[],
        gpu_uuids=list(audit.gpu_uuids),
    )

    checks = verify_rollback(snapshot, audit)

    assert (
        next(check for check in checks if check.name == "rollback.module-version").ok
        is False
    )


def test_load_snapshot_rejects_tampered_integrity(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["kernel"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="integrity check failed"):
        load_snapshot(str(path))


@pytest.mark.parametrize(
    "field",
    [
        "docker_socket_active",
        "docker_socket_enabled",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
    ],
)
def test_load_snapshot_integrity_binds_transactional_service_state(
    tmp_path,
    field,
):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document[field] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        RollbackSnapshotError,
        match="integrity check failed|inconsistent transactional service state",
    ):
        load_snapshot(str(path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_id", "c" * 32),
        ("docker_service_unit_file_state", "static"),
        ("docker_socket_unit_file_state", "static"),
        ("nvidia_persistenced_unit_file_state", "disabled"),
        ("fabric_manager_unit_file_state", "disabled"),
    ],
)
def test_load_snapshot_integrity_binds_operation_and_exact_unit_state(
    tmp_path,
    field,
    value,
):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RollbackSnapshotError, match="integrity check failed"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_commands_inconsistent_with_state(tmp_path):
    path = tmp_path / "snapshot.json"
    document = _snapshot_document(path)
    document["commands"] = [
        ["apt-get", "remove", "-y", "--allow-change-held-packages", "sudo"]
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="not a supported|inconsistent"):
        load_snapshot(str(path))


def _write_test_payload_bundle(
    path: Path,
    packages: list[PackageInfo],
) -> PackagePayloadBundle:
    packages = [package for package in packages if package.installed]
    bundle_path = path.parent / payload_bundle_directory(path)
    bundle_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    bundle_path.chmod(0o700)
    payloads: list[PackagePayload] = []
    total_payload_bytes = 0
    for package in packages:
        content = (
            f"retained-test-package:{package.name}:{package.architecture}:"
            f"{package.epoch or ''}:{package.version}"
        ).encode()
        digest = hashlib.sha256(content).hexdigest()
        rpm = package.manager == "rpm"
        extension = "rpm" if rpm else "deb"
        payload_path = bundle_path / f"{digest}.{extension}"
        payload_path.write_bytes(content)
        payload_path.chmod(0o600)
        total_payload_bytes += len(content)
        payloads.append(
            PackagePayload(
                name=package.name,
                architecture=package.architecture or "",
                epoch=package.epoch,
                version=package.version or "",
                format=extension,
                filename=payload_path.name,
                sha256=digest,
                size_bytes=len(content),
                verification="rpm-signature" if rpm else "apt-repository",
                roles=("baseline",),
                signer_ids=("deadbeef",) if rpm else (),
            )
        )
    return PackagePayloadBundle(
        directory=bundle_path.name,
        packages=tuple(payloads),
        total_size_bytes=total_payload_bytes,
    )


def _snapshot_document(
    path: Path,
    *,
    package_manager: str = "apt-get",
    packages: list[PackageInfo] | None = None,
    introduced_packages: list[str] | None = None,
):
    if packages is None:
        packages = [
            PackageInfo(
                "nvidia-driver-570",
                "570.1-1",
                "apt",
                True,
                architecture="amd64",
            )
        ]
    introduced = introduced_packages or []
    package_payloads = _write_test_payload_bundle(path, packages)
    managed_files = [FileSnapshot("/etc/docker/daemon.json", False, None, None)]
    if package_manager in {"dnf", "yum"}:
        managed_files.append(
            FileSnapshot(
                "/etc/dnf/modules.d/nvidia-driver.module",
                False,
                None,
                None,
            )
        )
        if package_manager == "dnf":
            managed_files.append(
                FileSnapshot(
                    "/var/lib/dnf/modulefailsafe/"
                    "nvidia-driver:580-open:x86_64.yaml",
                    False,
                    None,
                    None,
                )
            )
    elif package_manager == "zypper":
        managed_files.append(FileSnapshot("/etc/zypp/locks", False, None, None))
    snapshot = RollbackSnapshot(
        path=str(path),
        packages=packages,
        kernel="6.8.0-111-generic",
        module_version="570.1",
        commands=_rollback_commands(
            packages,
            package_manager,
            remove_packages=introduced,
            snapshot_path=str(path),
            package_payloads=package_payloads,
        ),
        schema_version="2.6",
        created_at="2026-08-02T00:00:00+00:00",
        operation_id="b" * 32,
        host_id="machine-id:" + "a" * 32,
        os_id="ubuntu" if package_manager == "apt-get" else "sles",
        os_version="24.04" if package_manager == "apt-get" else "15.6",
        architecture=platform.machine() or "x86_64",
        package_manager=package_manager,
        introduced_packages=introduced,
        docker_service_active=False,
        docker_service_enabled=False,
        docker_service_unit_file_state="disabled",
        docker_socket_active=False,
        docker_socket_enabled=False,
        docker_socket_unit_file_state="disabled",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="static",
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="not-found",
        managed_files=managed_files,
        package_payloads=package_payloads,
        gpu_uuids=["GPU-aaaaaaaaaaaaaaaa"],
    )
    snapshot.integrity_sha256 = _snapshot_integrity(snapshot)
    return asdict(snapshot)


def _full_mig_geometry() -> MigGpuInstance:
    return MigGpuInstance(
        gpu_uuid="GPU-aaaaaaaaaaaaaaaa",
        profile="7g.80gb",
        profile_id=0,
        placement_start=0,
        placement_size=8,
        compute_instances=[MigComputeInstance("7c.7g.80gb", 4)],
    )


def _quarantine_commands(*, include_disable: bool) -> list[list[str]]:
    commands: list[list[str]] = []
    for unit in (
        "docker.socket",
        "docker.service",
        "nvidia-persistenced.service",
        "nvidia-fabricmanager.service",
    ):
        if include_disable:
            commands.append(["systemctl", "disable", "--now", unit])
        commands.append(["systemctl", "mask", "--now", unit])
    return commands


def _non_quarantine_commands(results) -> list[list[str]]:
    return [
        result.command
        for result in results
        if result.command[:3]
        not in (
            ["systemctl", "disable", "--now"],
            ["systemctl", "mask", "--now"],
        )
        and result.command[:2] != ["systemctl", "stop"]
        and result.command != ["rollback", "service-state"]
    ]


def _bind_snapshot_service_state(
    snapshot: RollbackSnapshot,
    audit,
) -> RollbackSnapshot:
    for field in (
        "docker_service_active",
        "docker_service_enabled",
        "docker_service_unit_file_state",
        "docker_socket_active",
        "docker_socket_enabled",
        "docker_socket_unit_file_state",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
        "nvidia_persistenced_unit_file_state",
        "fabric_manager_active",
        "fabric_manager_enabled",
        "fabric_manager_unit_file_state",
    ):
        setattr(snapshot, field, getattr(audit, field))
    return snapshot


def _allow_downstream_applied_rollback(monkeypatch, snapshot: RollbackSnapshot) -> None:
    """Bypass separately covered authority gates in downstream sequencing tests."""
    snapshot.path = "/var/lib/nvidia-converge/snapshots/downstream-test.json"
    snapshot.package_payloads = PackagePayloadBundle(
        directory="downstream-test.payloads",
        packages=(),
        total_size_bytes=0,
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback.validate_snapshot_for_apply",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "nvidia_converge.preflight.preflight_package_rollback",
        lambda *args, **kwargs: [],
    )


def _systemd_state_result(
    unit: str,
    *,
    active: str = "inactive",
    unit_file_state: str = "masked",
) -> CommandResult:
    load_state = "masked" if unit_file_state == "masked" else "loaded"
    if unit_file_state == "not-found":
        load_state = "not-found"
        unit_file_state = ""
    return CommandResult(
        [],
        0,
        stdout=(
            f"Id={unit}\n"
            f"LoadState={load_state}\n"
            f"ActiveState={active}\n"
            f"UnitFileState={unit_file_state}\n"
        ),
    )


class _MigPendingRunner:
    apply = True

    def __init__(self, gpu_uuid: str):
        self.gpu_uuid = gpu_uuid
        self.calls: list[list[str]] = []
        self.mig_query_count = 0

    def exists(self, name):
        return name in {"nvidia-smi", "systemctl"}

    def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
        del mutate, allow_fail, input_text
        self.calls.append(command)
        if command[:2] == ["systemctl", "show"]:
            return _systemd_state_result(command[-1])
        if command[:3] == ["systemctl", "mask", "--now"]:
            return CommandResult(command, 0)
        if command[0:2] == [
            "nvidia-smi",
            "--query-gpu=uuid,mig.mode.current,mig.mode.pending",
        ]:
            self.mig_query_count += 1
            modes = (
                ("disabled", "disabled")
                if self.mig_query_count == 1
                else ("disabled", "enabled")
            )
            return CommandResult(
                command,
                0,
                stdout=f"{self.gpu_uuid}, {modes[0]}, {modes[1]}\n",
            )
        if command == [
            "nvidia-smi",
            "-i",
            self.gpu_uuid,
            "-mig",
            "1",
        ]:
            return CommandResult(command, 0)
        raise AssertionError(f"unexpected command: {command}")


class _FakeRunner:
    def __init__(self, returncodes):
        self.returncodes = list(returncodes)
        self.calls = []

    def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
        del mutate, allow_fail, input_text
        self.calls.append(command)
        if command[:3] in (
            ["systemctl", "disable", "--now"],
            ["systemctl", "mask", "--now"],
        ) or command[:2] == ["systemctl", "stop"]:
            return CommandResult(command, 0)
        value = self.returncodes.pop(0)
        if isinstance(value, CommandResult):
            return CommandResult(
                command,
                value.returncode,
                stdout=value.stdout,
                stderr=value.stderr,
                skipped=value.skipped,
                reason=value.reason,
            )
        return CommandResult(command, value)
