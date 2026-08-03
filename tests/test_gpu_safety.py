import json
import os
import shutil
from pathlib import Path

import pytest

from nvidia_converge import gpu_safety
from nvidia_converge.gpu_safety import (
    TrustedGpuServiceIdentity,
    TrustedGpuServiceSpec,
    probe_active_gpu_workloads,
    quiesce_trusted_gpu_services,
    revalidate_trusted_docker_socket_identity,
    revalidate_trusted_gpu_service_process_identity,
    validate_active_trusted_gpu_service,
    validate_active_trusted_gpu_service_identity,
    validate_trusted_docker_socket_unit,
    validate_trusted_docker_socket_unit_identity,
    validate_trusted_gpu_service_start,
    validate_trusted_gpu_service_unit,
)
from nvidia_converge.models import (
    CommandResult,
    HostAudit,
    KernelInfo,
    ModuleInfo,
    RuntimeInfo,
)


@pytest.fixture(autouse=True)
def _trusted_test_path_ownership(monkeypatch):
    """Model root-owned fixtures without weakening production constants."""

    test_uid = os.geteuid()
    monkeypatch.setattr(gpu_safety, "_TRUSTED_OWNER_UID", test_uid)
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_ANCESTOR_UIDS",
        frozenset({0, test_uid}),
    )


@pytest.fixture
def nvidia_device_root(tmp_path):
    device_root = tmp_path / "dev"
    device_root.mkdir()
    (device_root / "nvidia0").symlink_to("/dev/null")
    return device_root


def test_device_user_blocks_false_safe_result(tmp_path, nvidia_device_root):
    proc_root = _proc_with_gpu_fd(tmp_path, "2718", nvidia_device_root)
    runner = _ProbeRunner(
        executables={"nvidia-smi"},
        compute=CommandResult(["nvidia-smi"], 0),
    )

    result, workloads = probe_active_gpu_workloads(
        runner,
        _audit(module_loaded=True),
        proc_root=proc_root,
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:2718"]
    assert json.loads(result.stdout)["device_user_pids"] == ["2718"]


def test_compute_and_device_observations_deduplicate_pid(
    tmp_path,
    nvidia_device_root,
):
    proc_root = _proc_with_gpu_fd(tmp_path, "42", nvidia_device_root)
    runner = _ProbeRunner(
        executables={"nvidia-smi"},
        compute=CommandResult(["nvidia-smi"], 0, stdout="42\n7\n"),
    )

    result, workloads = probe_active_gpu_workloads(
        runner,
        _audit(module_loaded=True),
        proc_root=proc_root,
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:7", "pid:42"]


def test_memory_mapped_device_user_blocks_false_safe_result(
    tmp_path,
    nvidia_device_root,
):
    proc_root = tmp_path / "proc"
    process_root = proc_root / "314"
    (process_root / "fd").mkdir(parents=True)
    device_metadata = (nvidia_device_root / "nvidia0").stat()
    assert (
        os.major(device_metadata.st_dev),
        os.minor(device_metadata.st_dev),
    ) != (
        os.major(device_metadata.st_rdev),
        os.minor(device_metadata.st_rdev),
    )
    device_field, device_inode = _mapped_file_identity(
        nvidia_device_root / "nvidia0"
    )
    (process_root / "maps").write_text(
        (
            "7f000000-7f001000 rw-s 00000000 "
            f"{device_field} {device_inode} /dev/nvidiactl\n"
        ),
        encoding="utf-8",
    )
    runner = _ProbeRunner(
        executables={"nvidia-smi"},
        compute=CommandResult(["nvidia-smi"], 0),
    )

    result, workloads = probe_active_gpu_workloads(
        runner,
        _audit(module_loaded=True),
        proc_root=proc_root,
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:314"]


def test_bind_mounted_fd_alias_is_detected_by_character_device_identity(
    tmp_path,
    nvidia_device_root,
):
    alias = tmp_path / "container-device-alias"
    alias.symlink_to("/dev/null")
    fd_root = tmp_path / "proc" / "315" / "fd"
    fd_root.mkdir(parents=True)
    descriptor = fd_root / "8"
    descriptor.symlink_to(alias)
    assert "nvidia" not in os.readlink(descriptor)

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=tmp_path / "proc",
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:315"]


def test_deleted_or_renamed_fd_is_detected_without_link_text(
    tmp_path,
    nvidia_device_root,
    monkeypatch,
):
    fd_root = tmp_path / "proc" / "316" / "fd"
    fd_root.mkdir(parents=True)
    descriptor = fd_root / "9"
    descriptor.symlink_to(tmp_path / "renamed-device (deleted)")
    device_metadata = (nvidia_device_root / "nvidia0").stat()
    real_stat = gpu_safety._stat_open_descriptor

    def stat_descriptor(path):
        if path == descriptor:
            return device_metadata
        return real_stat(path)

    monkeypatch.setattr(gpu_safety, "_stat_open_descriptor", stat_descriptor)

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=tmp_path / "proc",
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:316"]


def test_mmap_after_close_uses_map_files_rdev_not_deleted_path_text(
    tmp_path,
    nvidia_device_root,
):
    process_root = tmp_path / "proc" / "317"
    (process_root / "fd").mkdir(parents=True)
    device_field, device_inode = _mapped_file_identity(
        nvidia_device_root / "nvidia0"
    )
    map_files_root = process_root / "map_files"
    map_files_root.mkdir()
    (map_files_root / "7f000000-7f001000").symlink_to(
        nvidia_device_root / "nvidia0"
    )
    (process_root / "maps").write_text(
        (
            "7f000000-7f001000 rw-s 00000000 "
            f"{device_field} {device_inode + 1} "
            "/mnt/renamed/gpu-device (deleted)\n"
        ),
        encoding="utf-8",
    )

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=tmp_path / "proc",
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:317"]


def test_potential_device_map_without_map_files_fails_closed(
    tmp_path,
    nvidia_device_root,
):
    process_root = tmp_path / "proc" / "320"
    (process_root / "fd").mkdir(parents=True)
    device_field, device_inode = _mapped_file_identity(
        nvidia_device_root / "nvidia0"
    )
    (process_root / "maps").write_text(
        (
            "7f000000-7f001000 rw-s 00000000 "
            f"{device_field} {device_inode + 1} /mnt/opaque/device-alias\n"
        ),
        encoding="utf-8",
    )

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=tmp_path / "proc",
        device_root=nvidia_device_root,
    )

    assert result.returncode == 1
    assert workloads is None
    assert "map_files inventory is unavailable" in result.stderr


def test_nvidia_capability_device_identity_is_included(tmp_path):
    device_root = tmp_path / "dev"
    capability_root = device_root / "nvidia-caps"
    capability_root.mkdir(parents=True)
    capability = capability_root / "nvidia-cap1"
    capability.symlink_to("/dev/zero")
    process_root = tmp_path / "proc" / "318"
    (process_root / "fd").mkdir(parents=True)
    device_field, device_inode = _mapped_file_identity(capability)
    (process_root / "maps").write_text(
        (
            "7f000000-7f001000 rw-s 00000000 "
            f"{device_field} {device_inode} /unrelated/capability-alias\n"
        ),
        encoding="utf-8",
    )
    audit = _audit(module_loaded=True)
    audit.module.devices = ["/dev/nvidia-caps/nvidia-cap1"]

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        audit,
        proc_root=tmp_path / "proc",
        device_root=device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:318"]


def test_loaded_stack_without_character_device_identity_fails_closed(tmp_path):
    device_root = tmp_path / "dev"
    device_root.mkdir()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=proc_root,
        device_root=device_root,
    )

    assert result.returncode == 1
    assert workloads is None
    assert "no NVIDIA character-device identities" in result.stderr


def test_nvidia_looking_maps_path_with_other_device_identity_is_not_a_user(
    tmp_path,
    nvidia_device_root,
):
    process_root = tmp_path / "proc" / "319"
    (process_root / "fd").mkdir(parents=True)
    map_files_root = process_root / "map_files"
    map_files_root.mkdir()
    (map_files_root / "7f000000-7f001000").symlink_to("/dev/zero")
    other_device, other_inode = _mapped_file_identity(Path("/dev/zero"))
    (process_root / "maps").write_text(
        (
            "7f000000-7f001000 rw-s 00000000 "
            f"{other_device} {other_inode} /dev/nvidia0 (deleted)\n"
        ),
        encoding="utf-8",
    )

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=tmp_path / "proc",
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == []


def test_running_docker_gpu_request_is_active_workload(tmp_path):
    container_id = "a" * 64
    runner = _ProbeRunner(
        executables={"docker"},
        docker_ps=CommandResult(["docker", "ps"], 0, stdout=container_id),
        docker_inspect=CommandResult(
            ["docker", "inspect"],
            0,
            stdout=json.dumps(
                [
                    {
                        "Id": container_id,
                        "State": {"Running": True},
                        "HostConfig": {
                            "Runtime": "runc",
                            "DeviceRequests": [
                                {"Driver": "nvidia", "Capabilities": [["gpu"]]}
                            ],
                        },
                        "Config": {"Env": []},
                        "Mounts": [],
                    }
                ]
            ),
        ),
    )

    result, workloads = probe_active_gpu_workloads(
        runner, _audit(docker_installed=True), proc_root=tmp_path / "unused"
    )

    assert result.returncode == 0
    assert workloads == ["docker:aaaaaaaaaaaa"]


def test_docker_nvidia_device_bind_is_active_workload(tmp_path):
    container_id = "b" * 64
    runner = _ProbeRunner(
        executables={"docker"},
        docker_ps=CommandResult(["docker", "ps"], 0, stdout=container_id),
        docker_inspect=CommandResult(
            ["docker", "inspect"],
            0,
            stdout=json.dumps(
                [
                    {
                        "Id": container_id,
                        "State": {"Running": True},
                        "HostConfig": {
                            "Runtime": "runc",
                            "DeviceRequests": [],
                            "Devices": [
                                {
                                    "PathOnHost": "/dev/nvidia0",
                                    "PathInContainer": "/dev/nvidia0",
                                }
                            ],
                        },
                        "Config": {"Env": []},
                        "Mounts": [],
                    }
                ]
            ),
        ),
    )

    _, workloads = probe_active_gpu_workloads(
        runner, _audit(docker_installed=True), proc_root=tmp_path / "unused"
    )

    assert workloads == ["docker:bbbbbbbbbbbb"]


def test_docker_daemon_failure_is_unknown_not_safe(tmp_path):
    runner = _ProbeRunner(
        executables={"docker"},
        docker_ps=CommandResult(["docker", "ps"], 1, stderr="cannot connect to daemon"),
    )

    result, workloads = probe_active_gpu_workloads(
        runner, _audit(docker_installed=True), proc_root=tmp_path / "unused"
    )

    assert result.returncode == 1
    assert workloads is None
    assert "not observable" in result.stderr


def test_known_inactive_docker_service_is_not_socket_activated_by_safety_probe(
    tmp_path,
):
    audit = _audit(docker_installed=True)
    audit.docker_service_active = False
    audit.docker_service_enabled = True
    audit.runtime.docker_gpus_usable = None
    runner = _ProbeRunner(executables={"docker"})

    result, workloads = probe_active_gpu_workloads(
        runner,
        audit,
        proc_root=tmp_path / "unused",
    )

    assert result.returncode == 0
    assert workloads == []
    assert not any(entry.command[:1] == ["docker"] for entry in runner.results)


def test_unknown_docker_service_is_not_socket_activated_and_fails_closed(
    tmp_path,
):
    audit = _audit(docker_installed=True)
    audit.docker_service_active = None
    runner = _ProbeRunner(executables={"docker"})

    result, workloads = probe_active_gpu_workloads(
        runner,
        audit,
        proc_root=tmp_path / "unused",
    )

    assert result.returncode == 1
    assert workloads is None
    assert "not proven" in result.stderr
    assert not any(entry.command[:1] == ["docker"] for entry in runner.results)


def test_unreadable_process_inventory_is_unknown(tmp_path, nvidia_device_root):
    runner = _ProbeRunner(
        executables={"nvidia-smi"},
        compute=CommandResult(["nvidia-smi"], 0),
    )

    result, workloads = probe_active_gpu_workloads(
        runner,
        _audit(module_loaded=True),
        proc_root=tmp_path / "missing-proc",
        device_root=nvidia_device_root,
    )

    assert result.returncode == 1
    assert workloads is None
    assert "cannot enumerate" in result.stderr


def test_broken_nvidia_smi_uses_complete_device_scan_for_repairability(
    tmp_path,
    nvidia_device_root,
):
    proc_root = _proc_with_gpu_fd(tmp_path, "99", nvidia_device_root)
    runner = _ProbeRunner(
        executables={"nvidia-smi"},
        compute=CommandResult(
            ["nvidia-smi"], 9, stderr="driver/library version mismatch"
        ),
    )

    result, workloads = probe_active_gpu_workloads(
        runner,
        _audit(module_loaded=True),
        proc_root=proc_root,
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:99"]
    evidence = json.loads(result.stdout)
    assert evidence["compute_pids"] is None
    assert "driver/library version mismatch" in evidence["compute_query_error"]


def test_absent_driver_and_runtime_are_observably_idle(tmp_path):
    runner = _ProbeRunner(executables=set())

    result, workloads = probe_active_gpu_workloads(
        runner, _audit(), proc_root=tmp_path / "unused"
    )

    assert result.returncode == 0
    assert workloads == []


@pytest.mark.parametrize(
    ("active_state", "sub_state"),
    [
        ("active", "listening"),
        ("inactive", "dead"),
        ("failed", "failed"),
    ],
)
def test_trusted_docker_socket_accepts_exact_stable_effective_unit(
    tmp_path,
    active_state,
    sub_state,
):
    state = _trusted_socket(tmp_path)
    hook = tmp_path / "bin" / "socket-hook"
    hook.parent.mkdir()
    hook.write_text("trusted hook", encoding="utf-8")
    hook.chmod(0o755)
    drop_in = tmp_path / "units" / "docker.socket.d" / "10-vendor.conf"
    drop_in.parent.mkdir()
    drop_in.write_text("[Socket]\n", encoding="utf-8")
    environment_file = tmp_path / "socket.env"
    environment_file.write_text("TRUSTED=yes\n", encoding="utf-8")
    state.update(
        ActiveState=active_state,
        SubState=sub_state,
        DropInPaths=str(drop_in),
        ExecStartPre=f"{{ path={hook} ; argv[]={hook} }}",
        ExecStartPost=f"{{ path={hook} ; argv[]={hook} }}",
        ExecStopPre=f"{{ path={hook} ; argv[]={hook} }}",
        ExecStopPost=f"{{ path={hook} ; argv[]={hook} }}",
        EnvironmentFiles=f"{environment_file} (ignore_errors=no)",
    )
    runner = _SocketRunner(state)

    results, error = validate_trusted_docker_socket_unit(runner)

    assert error is None
    assert results[-1].command == [
        "validate-trusted-docker-socket-unit",
        "docker.socket",
    ]
    assert results[-1].returncode == 0
    assert runner.mutations == []


@pytest.mark.parametrize(
    ("unsafe", "expected"),
    [
        ("alias", "unexpected unit"),
        ("dropin", "unit file"),
        ("start-hook", "executable"),
        ("stop-hook", "executable"),
        ("environment-file", "unit file"),
        ("inline-environment", "inline Environment"),
        ("root-remap", "RootDirectory"),
        ("trigger", "Triggers"),
    ],
)
def test_trusted_docker_socket_rejects_unsafe_effective_inputs(
    tmp_path,
    unsafe,
    expected,
):
    state = _trusted_socket(tmp_path)
    untrusted_file = tmp_path / "unsafe"
    untrusted_file.write_text("unsafe", encoding="utf-8")
    untrusted_file.chmod(0o777)
    if unsafe == "alias":
        state["Id"] = "container-runtime.socket"
    elif unsafe == "dropin":
        state["DropInPaths"] = str(untrusted_file)
    elif unsafe == "start-hook":
        state["ExecStartPre"] = (
            f"{{ path={untrusted_file} ; argv[]={untrusted_file} }}"
        )
    elif unsafe == "stop-hook":
        state["ExecStopPost"] = (
            f"{{ path={untrusted_file} ; argv[]={untrusted_file} }}"
        )
    elif unsafe == "environment-file":
        state["EnvironmentFiles"] = f"{untrusted_file} (ignore_errors=no)"
    elif unsafe == "inline-environment":
        state["Environment"] = "LD_PRELOAD=/tmp/untrusted.so"
    elif unsafe == "root-remap":
        state["RootDirectory"] = "/srv/untrusted-root"
    else:
        state["Triggers"] = "attacker.service"
    runner = _SocketRunner(state)

    results, error = validate_trusted_docker_socket_unit(runner)

    assert expected in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_trusted_docker_socket_masked_state_requires_explicit_allowance(tmp_path):
    state = _trusted_socket(tmp_path)
    state.update(
        LoadState="masked",
        ActiveState="inactive",
        SubState="dead",
        FragmentPath="/dev/null",
        Triggers="",
    )
    runner = _SocketRunner(state)

    refused, error = validate_trusted_docker_socket_unit(runner)
    accepted, allowed_error = validate_trusted_docker_socket_unit(
        runner,
        allow_masked=True,
    )

    assert "expected an exact loaded unit" in (error or "")
    assert refused[-1].returncode == 1
    assert allowed_error is None
    assert accepted[-1].returncode == 0
    assert runner.mutations == []


def test_trusted_docker_socket_accepts_authenticated_systemd_mask(
    tmp_path,
    monkeypatch,
):
    mask_root = tmp_path / "systemd"
    mask_root.mkdir()
    mask_path = mask_root / "docker.socket"
    mask_path.symlink_to("/dev/null")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_SYSTEMD_MASK_ROOTS",
        (mask_root,),
    )
    state = _trusted_socket(tmp_path)
    state.update(
        LoadState="masked",
        ActiveState="inactive",
        SubState="dead",
        FragmentPath=str(mask_path),
        Triggers="",
    )
    runner = _SocketRunner(state)

    results, error = validate_trusted_docker_socket_unit(
        runner,
        allow_masked=True,
    )

    assert error is None
    assert results[-1].returncode == 0
    assert runner.mutations == []


@pytest.mark.parametrize("fragment_path", ["", "/tmp/docker.socket"])
def test_trusted_docker_socket_rejects_unauthenticated_mask(
    tmp_path,
    fragment_path,
):
    state = _trusted_socket(tmp_path)
    state.update(
        LoadState="masked",
        ActiveState="inactive",
        SubState="dead",
        FragmentPath=fragment_path,
        Triggers="",
    )
    runner = _SocketRunner(state)

    results, error = validate_trusted_docker_socket_unit(
        runner,
        allow_masked=True,
    )

    assert "masked fragment state" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_trusted_docker_socket_identity_detects_unit_change_across_mask(tmp_path):
    state = _trusted_socket(tmp_path)
    state.update(ActiveState="active", SubState="listening")
    runner = _SocketRunner(state)

    _, identity, error = validate_trusted_docker_socket_unit_identity(runner)
    assert error is None
    assert identity is not None
    Path(state["FragmentPath"]).chmod(0o666)

    results, error = revalidate_trusted_docker_socket_identity(runner, identity)

    assert "cannot trust docker.socket" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_trusted_service_unit_validator_rejects_inactive_unsafe_stop_hook(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "docker", "119")
    state.update(ActiveState="inactive", SubState="dead", MainPID="0")
    unsafe_hook = tmp_path / "unsafe-stop-hook"
    unsafe_hook.write_text("unsafe", encoding="utf-8")
    unsafe_hook.chmod(0o777)
    state["ExecStopPost"] = (
        f"{{ path={unsafe_hook} ; argv[]={unsafe_hook} }}"
    )
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_unit(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert "ExecStopPost" not in (error or "")
    assert "executable" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_active_service_process_identity_rebinds_without_systemd_show(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "docker", "118")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})
    _, identity, error = validate_active_trusted_gpu_service_identity(
        runner,
        spec.unit,
        proc_root=proc_root,
    )
    assert error is None
    assert identity is not None
    show_count = runner.show_count

    results, error = revalidate_trusted_gpu_service_process_identity(
        runner,
        identity,
        proc_root=proc_root,
    )

    assert error is None
    assert results[-1].returncode == 0
    assert runner.show_count == show_count
    assert runner.mutations == []


def test_trusted_active_service_is_stopped_and_exactly_restored(tmp_path):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "101")
    runner = _ServiceRunner({spec.unit: state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.ok is True
    assert guard.quiesced_service_names == [spec.unit]
    assert runner.mutations == [["systemctl", "stop", spec.unit]]
    assert guard.restore() is True
    assert guard.quiesced_service_names == []
    assert runner.mutations == [
        ["systemctl", "stop", spec.unit],
        ["systemctl", "start", spec.unit],
    ]
    assert (
        sum(
            result.command[0] == "validate-trusted-gpu-service"
            for result in guard.results
        )
        == 3
    )


def test_inactive_trusted_service_is_never_started(tmp_path):
    spec, proc_root, state = _trusted_service(tmp_path, "persistenced", "102")
    state.update(ActiveState="inactive", SubState="dead", MainPID="0")
    runner = _ServiceRunner({spec.unit: state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.ok is True
    assert guard.quiesced_service_names == []
    assert guard.restore() is True
    assert runner.mutations == []


@pytest.mark.parametrize("service", ["fabric", "persistenced", "docker"])
def test_trusted_start_validation_accepts_only_exact_inactive_loaded_unit(
    tmp_path,
    monkeypatch,
    service,
):
    spec, proc_root, state = _trusted_service(tmp_path, service, "120")
    state.update(ActiveState="inactive", SubState="dead", MainPID="0")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert error is None
    assert results[-1].command == [
        "validate-trusted-gpu-service-start",
        spec.unit,
    ]
    assert results[-1].returncode == 0
    assert runner.mutations == []
    assert any(
        result.command[0] == "validate-trusted-gpu-service" for result in results
    )


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("nvidia-fabricmanager", "exact canonical"),
        ("nvidia-persistenced", "exact canonical"),
        ("docker", "exact canonical"),
    ],
)
def test_trusted_start_validation_rejects_alias_or_unlisted_unit(
    unit,
    expected,
):
    runner = _ServiceRunner({})

    results, error = validate_trusted_gpu_service_start(runner, unit)

    assert expected in (error or "")
    assert results[-1].returncode == 1
    assert runner.results == [results[-1]]
    assert runner.mutations == []


@pytest.mark.parametrize(
    ("state_changes", "expected"),
    [
        (
            {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": "0",
                "FragmentPath": "",
            },
            "expected an exact loaded",
        ),
        (
            {
                "LoadState": "loaded",
                "ActiveState": "failed",
                "SubState": "failed",
                "MainPID": "0",
            },
            "expected an exact loaded",
        ),
        (
            {
                "LoadState": "loaded",
                "ActiveState": "activating",
                "SubState": "start",
                "MainPID": "0",
            },
            "transitional state",
        ),
    ],
)
def test_trusted_start_validation_rejects_non_startable_unit_states(
    tmp_path,
    monkeypatch,
    state_changes,
    expected,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "121")
    state.update(state_changes)
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert expected in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_trusted_start_validation_rejects_systemd_alias_resolution(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "122")
    state.update(
        Id="fabric-manager-alias.service",
        ActiveState="inactive",
        SubState="dead",
        MainPID="0",
    )
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert "resolved to unexpected unit" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


@pytest.mark.parametrize(
    ("unsafe", "expected"),
    [
        ("unit-mode", "unit file"),
        ("executable-mode", "executable"),
        ("unit-ancestor", "path ancestor"),
        ("executable-ancestor", "path ancestor"),
        ("unit-alias", "not canonical"),
        ("executable-alias", "not canonical"),
        ("owner", "root-owned"),
    ],
)
def test_trusted_start_validation_rejects_unsafe_paths(
    tmp_path,
    monkeypatch,
    unsafe,
    expected,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "123")
    state.update(ActiveState="inactive", SubState="dead", MainPID="0")
    fragment = Path(state["FragmentPath"])
    if unsafe == "unit-mode":
        fragment.chmod(0o664)
    elif unsafe == "executable-mode":
        spec.executable.chmod(0o644)
    elif unsafe == "unit-ancestor":
        fragment.parent.chmod(0o775)
    elif unsafe == "executable-ancestor":
        spec.executable.parent.chmod(0o775)
    elif unsafe == "unit-alias":
        alias = fragment.with_name("unit-alias.service")
        alias.symlink_to(fragment)
        state["FragmentPath"] = str(alias)
    elif unsafe == "executable-alias":
        alias = spec.executable.with_name("executable-alias")
        alias.symlink_to(spec.executable)
        spec = TrustedGpuServiceSpec(spec.unit, alias)
        state["ExecStart"] = f"{{ path={alias} ; argv[]={alias} }}"
    else:
        monkeypatch.setattr(
            gpu_safety,
            "_TRUSTED_OWNER_UID",
            os.geteuid() + 1,
        )
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert expected in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_trusted_start_validation_binds_effective_execstart_exactly(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "126")
    replacement = spec.executable.with_name("replacement")
    replacement.write_text("replacement", encoding="utf-8")
    replacement.chmod(0o755)
    state.update(
        ActiveState="inactive",
        SubState="dead",
        MainPID="0",
        ExecStart=f"{{ path={replacement} ; argv[]={replacement} }}",
    )
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert "effective ExecStart" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_trusted_start_validation_accepts_all_trusted_effective_unit_inputs(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "persistenced", "127")
    drop_in_root = tmp_path / "dropins"
    drop_in_root.mkdir()
    drop_in = drop_in_root / "10-vendor.conf"
    drop_in.write_text("[Service]\n", encoding="utf-8")
    drop_in.chmod(0o644)
    hook = spec.executable.with_name("trusted-hook")
    hook.write_text("trusted hook", encoding="utf-8")
    hook.chmod(0o755)
    hook_property = f"{{ path={hook} ; argv[]={hook} }}"
    environment_file = tmp_path / "trusted-service.env"
    environment_file.write_text("TRUSTED_VALUE=yes\n", encoding="utf-8")
    environment_file.chmod(0o640)
    state.update(
        ActiveState="inactive",
        SubState="dead",
        MainPID="0",
        DropInPaths=str(drop_in),
        ExecCondition=hook_property,
        ExecStartPre=hook_property,
        ExecStartPost=hook_property,
        ExecStop=hook_property,
        ExecStopPost=hook_property,
        EnvironmentFiles=f"{environment_file} (ignore_errors=no)",
    )
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert error is None
    assert results[-1].returncode == 0
    assert runner.mutations == []


@pytest.mark.parametrize(
    "unsafe",
    [
        "dropin",
        "dropin-ancestor",
        "dropin-symlink",
        "start-hook",
        "stop-hook",
        "stop-post-hook",
        "environment-file",
        "environment-file-missing-optional",
        "environment-file-symlink",
        "inline-environment",
    ],
)
def test_trusted_start_validation_rejects_unsafe_effective_overrides(
    tmp_path,
    monkeypatch,
    unsafe,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "128")
    override_root = tmp_path / "overrides"
    override_root.mkdir()
    override = override_root / "override.conf"
    override.write_text("[Service]\n", encoding="utf-8")
    override.chmod(0o644)
    hook = spec.executable.with_name("unsafe-hook")
    hook.write_text("unsafe hook", encoding="utf-8")
    hook.chmod(0o755)
    environment_file = tmp_path / "unsafe-service.env"
    environment_file.write_text("LD_PRELOAD=/tmp/untrusted.so\n", encoding="utf-8")
    environment_file.chmod(0o640)
    state.update(ActiveState="inactive", SubState="dead", MainPID="0")
    if unsafe == "dropin":
        override.chmod(0o666)
        state["DropInPaths"] = str(override)
    elif unsafe == "dropin-ancestor":
        override_root.chmod(0o775)
        state["DropInPaths"] = str(override)
    elif unsafe == "dropin-symlink":
        alias = override.with_name("override-alias.conf")
        alias.symlink_to(override)
        state["DropInPaths"] = str(alias)
    elif unsafe == "start-hook":
        hook.chmod(0o777)
        state["ExecStartPre"] = f"{{ path={hook} ; argv[]={hook} }}"
    elif unsafe == "stop-hook":
        hook.chmod(0o777)
        state["ExecStop"] = f"{{ path={hook} ; argv[]={hook} }}"
    elif unsafe == "stop-post-hook":
        hook.chmod(0o777)
        state["ExecStopPost"] = f"{{ path={hook} ; argv[]={hook} }}"
    elif unsafe == "environment-file":
        environment_file.chmod(0o666)
        state["EnvironmentFiles"] = (
            f"{environment_file} (ignore_errors=no)"
        )
    elif unsafe == "environment-file-symlink":
        alias = environment_file.with_name("environment-alias")
        alias.symlink_to(environment_file)
        state["EnvironmentFiles"] = f"{alias} (ignore_errors=yes)"
    elif unsafe == "environment-file-missing-optional":
        state["EnvironmentFiles"] = (
            f"{environment_file.with_name('missing.env')} (ignore_errors=yes)"
        )
    else:
        state["Environment"] = "LD_PRELOAD=/tmp/untrusted.so"
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert "cannot trust" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


@pytest.mark.parametrize(
    ("property_name", "value"),
    [
        ("RootDirectory", "/srv/untrusted-root"),
        ("RootImage", "/srv/untrusted-root.img"),
        (
            "BindPaths",
            "/srv/replacement-dockerd:/usr/bin/dockerd:rbind",
        ),
        (
            "BindReadOnlyPaths",
            "/srv/replacement-dockerd:/usr/bin/dockerd:rbind",
        ),
        ("TemporaryFileSystem", "/usr:ro"),
        ("MountImages", "/srv/usr.img:/usr"),
        ("ExtensionImages", "/srv/extension.raw"),
    ],
)
def test_docker_prestart_rejects_executable_namespace_remapping(
    tmp_path,
    monkeypatch,
    property_name,
    value,
):
    spec, proc_root, state = _trusted_service(tmp_path, "docker", "129")
    state.update(ActiveState="inactive", SubState="dead", MainPID="0")
    state[property_name] = value
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_trusted_gpu_service_start(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert property_name in (error or "")
    assert "remapping is unsupported" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


@pytest.mark.parametrize("service", ["fabric", "persistenced", "docker"])
def test_active_trusted_service_validation_binds_exact_running_process(
    tmp_path,
    monkeypatch,
    service,
):
    spec, proc_root, state = _trusted_service(tmp_path, service, "130")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_active_trusted_gpu_service(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert error is None
    assert results[-1].command == [
        "validate-active-trusted-gpu-service",
        spec.unit,
    ]
    assert results[-1].returncode == 0
    assert runner.mutations == []
    assert any(
        result.command[0] == "validate-trusted-gpu-service" for result in results
    )


def test_active_trusted_service_identity_can_be_revalidated_across_probe(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "134")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    first_results, identity, first_error = validate_active_trusted_gpu_service_identity(
        runner,
        spec.unit,
        proc_root=proc_root,
    )
    second_results, repeated_identity, second_error = (
        validate_active_trusted_gpu_service_identity(
            runner,
            spec.unit,
            expected_identity=identity,
            proc_root=proc_root,
        )
    )

    executable_metadata = spec.executable.stat()
    assert first_error is None
    assert identity == TrustedGpuServiceIdentity(
        unit=spec.unit,
        main_pid=134,
        process_start_time_ticks=13400,
        executable_device=executable_metadata.st_dev,
        executable_inode=executable_metadata.st_ino,
        cgroup_path=f"/system.slice/{spec.unit}",
    )
    assert second_error is None
    assert repeated_identity == identity
    assert first_results[-1].returncode == 0
    assert second_results[-1].returncode == 0
    assert json.loads(first_results[-1].stdout) == {
        "cgroup_path": f"/system.slice/{spec.unit}",
        "executable_device": executable_metadata.st_dev,
        "executable_inode": executable_metadata.st_ino,
        "main_pid": 134,
        "process_start_time_ticks": 13400,
        "unit": spec.unit,
    }
    assert second_results[-1].stdout == first_results[-1].stdout
    assert runner.mutations == []


def test_active_trusted_service_identity_rejects_restart_between_probes(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "135")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})
    _, identity, error = validate_active_trusted_gpu_service_identity(
        runner,
        spec.unit,
        proc_root=proc_root,
    )
    assert error is None
    assert identity is not None

    # Model a fast same-PID restart: executable inode and cgroup remain the
    # same, while Linux's process start-time identity changes.
    (proc_root / "135" / "stat").write_text(
        _proc_stat("135", start_time_ticks=99999),
        encoding="utf-8",
    )

    results, repeated_identity, error = validate_active_trusted_gpu_service_identity(
        runner,
        spec.unit,
        expected_identity=identity,
        proc_root=proc_root,
    )

    assert repeated_identity is None
    assert "identity changed across the validation boundary" in (error or "")
    assert results[-1].returncode == 1
    assert json.loads(results[-1].stdout)["process_start_time_ticks"] == 99999
    assert runner.mutations == []


def test_active_trusted_service_identity_rejects_change_during_observation(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "persistenced", "136")
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    start_times = iter((13600, 13601))
    monkeypatch.setattr(
        gpu_safety,
        "_process_start_time_ticks",
        lambda path, main_pid: next(start_times),
    )
    runner = _ServiceRunner({spec.unit: state})

    results, identity, error = validate_active_trusted_gpu_service_identity(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert identity is None
    assert "changed identity during validation" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


@pytest.mark.parametrize(
    "unit",
    [
        "nvidia-fabricmanager",
        "nvidia-persistenced",
        "docker",
    ],
)
def test_active_trusted_service_validation_rejects_alias_or_unlisted_unit(
    unit,
):
    runner = _ServiceRunner({})

    results, error = validate_active_trusted_gpu_service(runner, unit)

    assert "exact canonical" in (error or "")
    assert results[-1].returncode == 1
    assert runner.results == [results[-1]]
    assert runner.mutations == []


@pytest.mark.parametrize(
    ("state_changes", "expected"),
    [
        (
            {
                "LoadState": "loaded",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": "0",
            },
            "expected an exact loaded",
        ),
        (
            {
                "LoadState": "loaded",
                "ActiveState": "failed",
                "SubState": "failed",
                "MainPID": "0",
            },
            "expected an exact loaded",
        ),
        (
            {
                "LoadState": "loaded",
                "ActiveState": "activating",
                "SubState": "start",
                "MainPID": "0",
            },
            "transitional state",
        ),
        (
            {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": "0",
                "FragmentPath": "",
            },
            "expected an exact loaded",
        ),
    ],
)
def test_active_trusted_service_validation_rejects_non_running_states(
    tmp_path,
    monkeypatch,
    state_changes,
    expected,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "131")
    state.update(state_changes)
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_active_trusted_gpu_service(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert expected in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_active_trusted_service_validation_rejects_systemd_alias_resolution(
    tmp_path,
    monkeypatch,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "132")
    state["Id"] = "fabric-manager-alias.service"
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_active_trusted_gpu_service(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert "resolved to unexpected unit" in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


@pytest.mark.parametrize(
    ("spoof", "expected"),
    [
        ("unit", "unit file"),
        ("dropin", "unsafe-drop-in.conf"),
        ("hook", "executable"),
        ("execstart", "effective ExecStart"),
        ("process-executable", "MainPID 133 executable"),
        ("cgroup", "system.slice cgroup"),
    ],
)
def test_active_trusted_service_validation_rejects_spoofed_binding(
    tmp_path,
    monkeypatch,
    spoof,
    expected,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "133")
    if spoof == "unit":
        Path(state["FragmentPath"]).chmod(0o666)
    elif spoof == "dropin":
        drop_in = tmp_path / "unsafe-drop-in.conf"
        drop_in.write_text("[Service]\n", encoding="utf-8")
        drop_in.chmod(0o666)
        state["DropInPaths"] = str(drop_in)
    elif spoof == "hook":
        hook = tmp_path / "unsafe-hook"
        hook.write_text("unsafe hook", encoding="utf-8")
        hook.chmod(0o777)
        state["ExecStartPre"] = f"{{ path={hook} ; argv[]={hook} }}"
    elif spoof == "execstart":
        replacement = tmp_path / "replacement"
        replacement.write_text("replacement", encoding="utf-8")
        replacement.chmod(0o755)
        state["ExecStart"] = f"{{ path={replacement} ; argv[]={replacement} }}"
    elif spoof == "process-executable":
        replacement = tmp_path / "process-replacement"
        replacement.write_text("replacement", encoding="utf-8")
        replacement.chmod(0o755)
        (proc_root / "133" / "exe").unlink()
        (proc_root / "133" / "exe").symlink_to(replacement)
    else:
        (proc_root / "133" / "cgroup").write_text(
            "0::/system.slice/untrusted.service\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        gpu_safety,
        "_TRUSTED_GPU_SERVICE_BY_UNIT",
        {spec.unit: spec},
    )
    runner = _ServiceRunner({spec.unit: state})

    results, error = validate_active_trusted_gpu_service(
        runner,
        spec.unit,
        proc_root=proc_root,
    )

    assert "cannot trust" in (error or "")
    assert expected in (error or "")
    assert results[-1].returncode == 1
    assert runner.mutations == []


def test_active_service_rejects_effective_execstart_override_before_stop(
    tmp_path,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "129")
    state["ExecStart"] = "{ path=/usr/bin/unexpected ; argv[]=/usr/bin/unexpected }"
    runner = _ServiceRunner({spec.unit: state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.ok is False
    assert "effective ExecStart" in (guard.error or "")
    assert runner.mutations == []


def test_guard_restore_preserves_pending_unit_when_start_validation_fails(
    tmp_path,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "124")
    runner = _ServiceRunner({spec.unit: state})
    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )
    Path(state["FragmentPath"]).parent.chmod(0o775)

    assert guard.restore() is False
    assert guard.quiesced_service_names == [spec.unit]
    assert "path ancestor" in " ".join(guard.restore_errors)
    assert runner.mutations == [["systemctl", "stop", spec.unit]]


def test_active_service_validation_rejects_unsafe_executable_ancestor(
    tmp_path,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "125")
    spec.executable.parent.chmod(0o775)
    runner = _ServiceRunner({spec.unit: state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.ok is False
    assert "path ancestor" in (guard.error or "")
    assert runner.mutations == []


@pytest.mark.parametrize("spoof", ["executable", "cgroup", "unit-mode"])
def test_trusted_service_spoof_is_rejected_without_stop(tmp_path, spoof):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "103")
    if spoof == "executable":
        replacement = tmp_path / "bin" / "replacement"
        replacement.write_text("replacement", encoding="utf-8")
        replacement.chmod(0o755)
        (proc_root / "103" / "exe").unlink()
        (proc_root / "103" / "exe").symlink_to(replacement)
    elif spoof == "cgroup":
        (proc_root / "103" / "cgroup").write_text(
            "0::/system.slice/untrusted.service\n",
            encoding="utf-8",
        )
    else:
        Path(state["FragmentPath"]).chmod(0o666)
    runner = _ServiceRunner({spec.unit: state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.ok is False
    assert guard.error is not None
    assert "cannot trust" in guard.error
    assert runner.mutations == []


def test_duplicate_service_properties_fail_closed(tmp_path):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "104")
    runner = _ServiceRunner(
        {spec.unit: state},
        show_suffix="ActiveState=active\n",
    )

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.ok is False
    assert "malformed or incomplete" in (guard.error or "")
    assert runner.mutations == []


def test_partial_quiesce_failure_restores_already_stopped_service(tmp_path):
    first, proc_root, first_state = _trusted_service(tmp_path, "fabric", "105")
    second, _, second_state = _trusted_service(
        tmp_path,
        "persistenced",
        "106",
        proc_root=proc_root,
    )
    (proc_root / "106" / "cgroup").write_text(
        "0::/system.slice/untrusted.service\n",
        encoding="utf-8",
    )
    runner = _ServiceRunner({first.unit: first_state, second.unit: second_state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(first, second),
    )

    assert guard.ok is False
    assert guard.quiesced_service_names == []
    assert runner.mutations == [
        ["systemctl", "stop", first.unit],
        ["systemctl", "start", first.unit],
    ]


def test_transactional_partial_quiesce_failure_never_restarts_service(
    tmp_path,
):
    first, proc_root, first_state = _trusted_service(
        tmp_path,
        "fabric",
        "205",
    )
    second, _, second_state = _trusted_service(
        tmp_path,
        "persistenced",
        "206",
        proc_root=proc_root,
    )
    (proc_root / "206" / "cgroup").write_text(
        "0::/system.slice/untrusted.service\n",
        encoding="utf-8",
    )
    runner = _ServiceRunner({first.unit: first_state, second.unit: second_state})

    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(first, second),
        restore_on_failure=False,
    )

    assert guard.ok is False
    assert guard.quiesced_service_names == [first.unit]
    assert runner.mutations == [["systemctl", "stop", first.unit]]


def test_signal_during_pre_mutation_quiesce_restores_service(tmp_path):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "107")
    runner = _ServiceRunner(
        {spec.unit: state},
        raise_on_show=2,
    )

    with pytest.raises(KeyboardInterrupt):
        quiesce_trusted_gpu_services(
            runner,
            proc_root=proc_root,
            specs=(spec,),
        )

    assert runner.mutations == [
        ["systemctl", "stop", spec.unit],
        ["systemctl", "start", spec.unit],
    ]


def test_transactional_signal_during_quiesce_never_restarts_service(tmp_path):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "207")
    runner = _ServiceRunner(
        {spec.unit: state},
        raise_on_show=2,
    )

    with pytest.raises(KeyboardInterrupt):
        quiesce_trusted_gpu_services(
            runner,
            proc_root=proc_root,
            specs=(spec,),
            restore_on_failure=False,
        )

    assert runner.mutations == [["systemctl", "stop", spec.unit]]


def test_failed_restore_remains_explicitly_pending(tmp_path):
    spec, proc_root, state = _trusted_service(tmp_path, "persistenced", "108")
    runner = _ServiceRunner({spec.unit: state}, fail_start={spec.unit})
    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.restore() is False
    assert guard.quiesced_service_names == [spec.unit]
    assert any("could not restart" in error for error in guard.restore_errors)


def test_failed_mutation_requiesces_service_restarted_by_package_script(
    tmp_path,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "110")
    runner = _ServiceRunner({spec.unit: state})
    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )
    state.update(
        ActiveState="active",
        SubState="running",
        MainPID=state["_pid"],
    )

    assert guard.requiesce() is True
    assert guard.quiesced_service_names == [spec.unit]
    assert runner.mutations == [
        ["systemctl", "stop", spec.unit],
        ["systemctl", "stop", spec.unit],
    ]
    assert state["ActiveState"] == "inactive"


def test_post_verification_compensation_requiesces_original_service(
    tmp_path,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "111")
    runner = _ServiceRunner({spec.unit: state})
    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )

    assert guard.restore() is True
    assert guard.quiesced_service_names == []
    assert guard.requiesce() is True
    assert guard.quiesced_service_names == [spec.unit]
    assert runner.mutations == [
        ["systemctl", "stop", spec.unit],
        ["systemctl", "start", spec.unit],
        ["systemctl", "stop", spec.unit],
    ]
    assert state["ActiveState"] == "inactive"


def test_unknown_gpu_user_still_blocks_after_trusted_service_quiesce(
    tmp_path,
    nvidia_device_root,
):
    spec, proc_root, state = _trusted_service(tmp_path, "fabric", "109")
    runner = _ServiceRunner({spec.unit: state})
    guard = quiesce_trusted_gpu_services(
        runner,
        proc_root=proc_root,
        specs=(spec,),
    )
    shutil.rmtree(proc_root / "109")
    unknown_fd = proc_root / "999" / "fd"
    unknown_fd.mkdir(parents=True)
    (unknown_fd / "7").symlink_to(nvidia_device_root / "nvidia0")

    result, workloads = probe_active_gpu_workloads(
        _ProbeRunner(
            executables={"nvidia-smi"},
            compute=CommandResult(["nvidia-smi"], 0),
        ),
        _audit(module_loaded=True),
        proc_root=proc_root,
        device_root=nvidia_device_root,
    )

    assert result.returncode == 0
    assert workloads == ["pid:999"]
    assert guard.restore() is False


def _proc_with_gpu_fd(
    tmp_path: Path,
    pid: str,
    device_root: Path,
) -> Path:
    proc_root = tmp_path / "proc"
    fd_root = proc_root / pid / "fd"
    fd_root.mkdir(parents=True)
    (fd_root / "9").symlink_to(device_root / "nvidia0")
    return proc_root


def _mapped_file_identity(path: Path) -> tuple[str, int]:
    metadata = path.stat()
    return (
        f"{os.major(metadata.st_dev):x}:{os.minor(metadata.st_dev):x}",
        metadata.st_ino,
    )


def _trusted_service(
    tmp_path: Path,
    service: str,
    pid: str,
    *,
    proc_root: Path | None = None,
):
    names = {
        "fabric": ("nvidia-fabricmanager.service", "nv-fabricmanager"),
        "persistenced": (
            "nvidia-persistenced.service",
            "nvidia-persistenced",
        ),
        "docker": ("docker.service", "dockerd"),
    }
    unit, executable_name = names[service]
    executable_root = tmp_path / "bin"
    executable_root.mkdir(exist_ok=True)
    executable = executable_root / executable_name
    executable.write_text("trusted executable", encoding="utf-8")
    executable.chmod(0o755)
    unit_root = tmp_path / "units"
    unit_root.mkdir(exist_ok=True)
    fragment = unit_root / unit
    fragment.write_text("[Service]\n", encoding="utf-8")
    fragment.chmod(0o644)
    process_root = (proc_root or tmp_path / "proc") / pid
    process_root.mkdir(parents=True)
    (process_root / "exe").symlink_to(executable)
    (process_root / "cgroup").write_text(
        f"0::/system.slice/{unit}\n",
        encoding="utf-8",
    )
    (process_root / "stat").write_text(
        _proc_stat(pid, start_time_ticks=int(pid) * 100),
        encoding="utf-8",
    )
    (process_root / "fd").mkdir()
    state = {
        "Id": unit,
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": pid,
        "FragmentPath": str(fragment),
        "DropInPaths": "",
        "ExecCondition": "",
        "ExecStartPre": "",
        "ExecStart": f"{{ path={executable} ; argv[]={executable} }}",
        "ExecStartPost": "",
        "ExecStop": "",
        "ExecStopPost": "",
        "Environment": "",
        "EnvironmentFiles": "",
        "RootDirectory": "",
        "RootImage": "",
        "BindPaths": "",
        "BindReadOnlyPaths": "",
        "TemporaryFileSystem": "",
        "MountImages": "",
        "ExtensionImages": "",
        "_pid": pid,
    }
    return TrustedGpuServiceSpec(unit, executable), process_root.parent, state


def _trusted_socket(tmp_path: Path) -> dict[str, str]:
    unit_root = tmp_path / "units"
    unit_root.mkdir(exist_ok=True)
    fragment = unit_root / "docker.socket"
    fragment.write_text("[Socket]\nListenStream=/run/docker.sock\n", encoding="utf-8")
    fragment.chmod(0o644)
    return {
        "Id": "docker.socket",
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "FragmentPath": str(fragment),
        "DropInPaths": "",
        "Triggers": "docker.service",
        "ExecStartPre": "",
        "ExecStartPost": "",
        "ExecStopPre": "",
        "ExecStopPost": "",
        "Environment": "",
        "EnvironmentFiles": "",
        "RootDirectory": "",
        "RootImage": "",
        "BindPaths": "",
        "BindReadOnlyPaths": "",
        "TemporaryFileSystem": "",
        "MountImages": "",
        "ExtensionImages": "",
    }


def _proc_stat(pid: str, *, start_time_ticks: int) -> str:
    fields_before_start_time = ["0"] * 18
    return (
        f"{pid} (trusted service) S "
        + " ".join([*fields_before_start_time, str(start_time_ticks)])
        + "\n"
    )


class _ServiceRunner:
    def __init__(
        self,
        states,
        *,
        show_suffix="",
        raise_on_show=None,
        fail_start=None,
    ):
        self.states = states
        self.show_suffix = show_suffix
        self.raise_on_show = raise_on_show
        self.fail_start = fail_start or set()
        self.show_count = 0
        self.results = []
        self.mutations = []

    def exists(self, name):
        return name == "systemctl"

    def run(self, command, *, mutate=False, allow_fail=True):
        del allow_fail
        unit = command[-1]
        state = self.states[unit]
        if command[1] == "show":
            self.show_count += 1
            if self.raise_on_show == self.show_count:
                self.raise_on_show = None
                raise KeyboardInterrupt
            keys = gpu_safety._SERVICE_PROPERTIES
            output = "".join(f"{key}={state[key]}\n" for key in keys)
            result = CommandResult(command, 0, stdout=output + self.show_suffix)
        elif command[1] == "stop":
            assert mutate is True
            self.mutations.append(command)
            state.update(ActiveState="inactive", SubState="dead", MainPID="0")
            result = CommandResult(command, 0)
        elif command[1] == "start":
            assert mutate is True
            self.mutations.append(command)
            if unit in self.fail_start:
                result = CommandResult(command, 1, stderr="start failed")
            else:
                state.update(
                    ActiveState="active",
                    SubState="running",
                    MainPID=state["_pid"],
                )
                result = CommandResult(command, 0)
        else:
            raise AssertionError(f"unexpected command: {command}")
        self.results.append(result)
        return result

    def record_external_start(self, command, mutate):
        del command, mutate

    def record_external_result(self, result, mutate):
        del mutate
        self.results.append(result)


class _SocketRunner:
    def __init__(self, state):
        self.state = state
        self.results = []
        self.mutations = []

    @staticmethod
    def exists(name):
        return name == "systemctl"

    def run(self, command, *, mutate=False, allow_fail=True):
        del allow_fail
        if mutate:
            self.mutations.append(command)
            result = CommandResult(command, 0)
        elif command[:2] == ["systemctl", "show"]:
            output = "".join(
                f"{key}={self.state[key]}\n"
                for key in gpu_safety._DOCKER_SOCKET_PROPERTIES
            )
            result = CommandResult(command, 0, stdout=output)
        else:
            raise AssertionError(f"unexpected command: {command}")
        self.results.append(result)
        return result

    def record_external_start(self, command, mutate):
        del command, mutate

    def record_external_result(self, result, mutate):
        del mutate
        self.results.append(result)


def _audit(*, module_loaded: bool = False, docker_installed: bool = False) -> HostAudit:
    return HostAudit(
        timestamp="2026-08-02T00:00:00+00:00",
        os_id="ubuntu",
        os_version="24.04",
        package_manager="apt-get",
        kernel=KernelInfo("6.8.0-test", True),
        module=ModuleInfo(
            loaded=module_loaded,
            version="580.126.16" if module_loaded else None,
            devices=["/dev/nvidia0"] if module_loaded else [],
            installed_version="580.126.16" if module_loaded else None,
        ),
        runtime=RuntimeInfo(docker_installed, docker_installed, docker_installed),
        packages=[],
        nvidia_smi=CommandResult(["nvidia-smi"], 0 if module_loaded else 127),
        nvml=CommandResult(["python3"], 0 if module_loaded else 1),
        fabric_manager_active=False,
        mig_mode="disabled" if module_loaded else None,
        docker_service_active=docker_installed,
        docker_service_enabled=docker_installed,
    )


class _ProbeRunner:
    def __init__(
        self,
        *,
        executables: set[str],
        compute: CommandResult | None = None,
        docker_ps: CommandResult | None = None,
        docker_inspect: CommandResult | None = None,
    ):
        self.executables = executables
        self.compute = compute
        self.docker_ps = docker_ps
        self.docker_inspect = docker_inspect
        self.results: list[CommandResult] = []

    def exists(self, name: str) -> bool:
        return name in self.executables

    def run(self, command, *, allow_fail=True):
        del allow_fail
        if command[:2] == ["nvidia-smi", "--query-compute-apps=pid"]:
            template = self.compute
        elif command[:2] == ["docker", "ps"]:
            template = self.docker_ps
        elif command[:2] == ["docker", "inspect"]:
            template = self.docker_inspect
        else:
            raise AssertionError(f"unexpected command: {command}")
        assert template is not None
        result = CommandResult(
            command,
            template.returncode,
            stdout=template.stdout,
            stderr=template.stderr,
            reason=template.reason,
        )
        self.results.append(result)
        return result

    def record_external_start(self, command, mutate):
        del command, mutate

    def record_external_result(self, result, mutate):
        del mutate
        self.results.append(result)
