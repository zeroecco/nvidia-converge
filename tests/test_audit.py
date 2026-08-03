import os

import pytest

import nvidia_converge.files as files_module
from nvidia_converge.audit import (
    _audit_cuda_compatibility,
    _audit_dnf_module_policy,
    _audit_fabric_manager_health,
    _audit_mig_geometry,
    _audit_mig_state,
    _audit_open_module_support,
    _audit_package_policy,
    _audit_packages,
    _audit_runtime,
    _loaded_module_flavor,
    _loaded_module_signed,
    _mig_mode,
    _modinfo_version,
    _parse_dpkg_package_rows,
    _parse_dpkg_packages,
    _parse_rpm_packages,
    _parse_service_state,
    _parse_zypper_policy_selectors,
    _read_os_release,
    audit_host,
)


def _os_release_tree(tmp_path):
    root = tmp_path.resolve() / "host"
    etc = root / "etc"
    vendor = root / "usr" / "lib"
    etc.mkdir(parents=True)
    vendor.mkdir(parents=True)
    target = vendor / "os-release"
    target.write_text(
        'NAME="Test Linux"\nID=test-linux\nVERSION_ID="24.04"\n',
        encoding="utf-8",
    )
    link = etc / "os-release"
    link.symlink_to("../usr/lib/os-release")
    return root, link, target


def test_os_release_accepts_bound_root_controlled_relative_symlink(tmp_path):
    _, link, _ = _os_release_tree(tmp_path)

    assert _read_os_release(
        link,
        required_owner_uid=os.geteuid(),
    ) == ("test-linux", "24.04")


def test_os_release_rejects_symlink_swap_during_target_read(monkeypatch, tmp_path):
    root, link, _ = _os_release_tree(tmp_path)
    replacement = root / "usr" / "lib" / "replacement"
    replacement.write_text("ID=ubuntu\nVERSION_ID=24.04\n", encoding="utf-8")
    real_read = files_module.read_trusted_utf8_with_metadata
    swapped = False

    def swapping_read(*args, **kwargs):
        nonlocal swapped
        result = real_read(*args, **kwargs)
        if not swapped:
            swapped = True
            link.unlink()
            link.symlink_to("../usr/lib/replacement")
        return result

    monkeypatch.setattr(
        "nvidia_converge.files.read_trusted_utf8_with_metadata",
        swapping_read,
    )

    assert _read_os_release(
        link,
        required_owner_uid=os.geteuid(),
    ) == (None, None)


def test_os_release_rejects_writable_ancestor(tmp_path):
    root, link, _ = _os_release_tree(tmp_path)
    root.chmod(0o777)

    assert _read_os_release(
        link,
        required_owner_uid=os.geteuid(),
    ) == (None, None)


def test_os_release_rejects_writable_target(tmp_path):
    _, link, target = _os_release_tree(tmp_path)
    target.chmod(0o666)

    assert _read_os_release(
        link,
        required_owner_uid=os.geteuid(),
    ) == (None, None)


def test_os_release_rejects_oversize_target(tmp_path):
    _, link, target = _os_release_tree(tmp_path)
    target.write_text("X" * (64 * 1024 + 1), encoding="utf-8")

    assert _read_os_release(
        link,
        required_owner_uid=os.geteuid(),
    ) == (None, None)


@pytest.mark.parametrize(
    "content",
    [
        'ID="unterminated\nVERSION_ID=24.04\n',
        "ID=ubuntu\nID=debian\nVERSION_ID=24.04\n",
        "ID=Ubuntu Linux\nVERSION_ID=24.04\n",
        "ID=ubuntu\nVERSION_ID=24.04\x00ignored\n",
    ],
)
def test_os_release_rejects_malformed_content(tmp_path, content):
    _, link, target = _os_release_tree(tmp_path)
    target.write_text(content, encoding="utf-8")

    assert _read_os_release(
        link,
        required_owner_uid=os.geteuid(),
    ) == (None, None)
from nvidia_converge.mig import full_mig_geometry_matches
from nvidia_converge.models import (
    CommandResult,
    MigComputeInstance,
    MigGpuInstance,
    PackageInfo,
)


def test_parse_dpkg_packages_filters_sorts_and_deduplicates():
    packages = _parse_dpkg_packages(
        """
ii \tzlib1g\t1.3\tamd64
ii \tlibnvidia-gl\t595.71.05-1ubuntu1\tamd64
ii \tlibnvidia-gl\t595.71.05-1ubuntu1\tamd64
ii \tcuda-toolkit-13-1\t13.1.2-1\tamd64
ii \tdocker-ce\t5:29.4.2-2\tamd64
bad line without tab
"""
    )
    assert [pkg.name for pkg in packages] == [
        "cuda-toolkit-13-1",
        "docker-ce",
        "libnvidia-gl",
    ]
    assert [pkg.manager for pkg in packages] == ["apt", "apt", "apt"]
    assert all(pkg.installed for pkg in packages)


def test_parse_dpkg_packages_only_accepts_installed_status_records():
    packages = _parse_dpkg_packages(
        "ii \tnvidia-open\t580.126.16-1\tamd64\n"
        "rc \tnvidia-driver-570\t570.1-1\tamd64\n"
        "un \tcuda-compat-12-8\t\tamd64\n"
    )

    assert [(package.name, package.architecture) for package in packages] == [
        ("nvidia-open", "amd64")
    ]


def test_parse_dpkg_inventory_retains_stable_installed_and_held_rows():
    packages, complete = _parse_dpkg_package_rows(
        "ii \tnvidia-open\t580.126.16-1\tamd64\n"
        "hi \tnvidia-container-toolkit\t1.18.2-1\tamd64\n"
    )

    assert complete is True
    assert [(package.name, package.version) for package in packages] == [
        ("nvidia-container-toolkit", "1.18.2-1"),
        ("nvidia-open", "580.126.16-1"),
    ]


def test_dpkg_inventory_rejects_residual_configuration_state():
    packages, complete = _parse_dpkg_package_rows(
        "ii \tnvidia-open\t580.126.16-1\tamd64\n"
        "rc \tnvidia-driver-570\t570.1-1\tamd64\n"
    )

    assert complete is False
    assert [(package.name, package.version) for package in packages] == [
        ("nvidia-open", "580.126.16-1")
    ]


@pytest.mark.parametrize(
    "status",
    ["iU ", "iF ", "iH ", "it ", "iW ", "iiR", "hiR", "rcR"],
)
def test_dpkg_inventory_rejects_transitional_or_error_status(status):
    packages, complete, _ = _audit_packages(
        "apt-get",
        _InventoryRunner(
            f"{status}\tnvidia-open\t580.126.16-1\tamd64\n"
        ),
        "6.8.0-test",
    )

    assert complete is False
    assert packages == []


def test_dpkg_inventory_accepts_valid_multiarch_rows():
    packages, complete, _ = _audit_packages(
        "apt-get",
        _InventoryRunner(
            "ii \tlibnvidia-gl\t580.126.16-1\tamd64\n"
            "ii \tlibnvidia-gl\t580.126.16-1\tarm64\n"
        ),
        "6.8.0-test",
    )

    assert complete is True
    assert [(package.name, package.architecture) for package in packages] == [
        ("libnvidia-gl", "amd64"),
        ("libnvidia-gl", "arm64"),
    ]


@pytest.mark.parametrize(
    "output",
    [
        "ii \tnvidia-open\t580.126.16-1\tamd64\textra\n",
        "ii\tnvidia-open\t580.126.16-1\tamd64\n",
        "ii \tnvidia-open\t\tamd64\n",
        "ii \tnvidia-open\t580.126.16-1\t\n",
        "rc \tnvidia-open\t\tamd64\n",
        "ii \tnvidia-open\t580.126.16-1\tamd64\nmalformed\n",
        (
            "ii \tnvidia-open\t580.126.16-1\tamd64\n"
            "ii \tnvidia-open\t580.126.16-2\tamd64\n"
        ),
        (
            "rc \tnvidia-open\t570.1-1\tamd64\n"
            "ii \tnvidia-open\t580.126.16-1\tamd64\n"
        ),
    ],
)
def test_dpkg_inventory_rejects_malformed_or_duplicate_slots(output):
    _, complete, _ = _audit_packages(
        "apt-get",
        _InventoryRunner(output),
        "6.8.0-test",
    )

    assert complete is False


def test_apt_package_policy_uses_only_installed_pinning_packages():
    inventory = CommandResult(["dpkg-query"], 0)
    policy = _audit_package_policy(
        "apt-get",
        [
            PackageInfo(
                "nvidia-driver-pinning-580",
                "580.1",
                "apt",
                True,
            ),
            PackageInfo(
                "nvidia-driver-pinning-570",
                "570.1",
                "apt",
                False,
            ),
        ],
        True,
        inventory,
        object(),
    )

    assert policy.observable is True
    assert [selector.name for selector in policy.selectors] == [
        "nvidia-driver-pinning-580"
    ]
    assert policy.observation is inventory


def test_dnf_module_policy_observes_missing_and_enabled_state(tmp_path):
    path = tmp_path / "nvidia-driver.module"
    missing = _audit_dnf_module_policy(path)
    assert missing.observable is True
    assert missing.selectors == []

    path.write_text(
        "[nvidia-driver]\n"
        "name=nvidia-driver\n"
        "stream=580-open\n"
        "profiles=default\n"
        "state=enabled\n",
        encoding="utf-8",
    )
    enabled = _audit_dnf_module_policy(path)

    assert enabled.observable is True
    assert len(enabled.selectors) == 1
    assert enabled.selectors[0].version == "580-open"


def test_dnf_module_policy_rejects_malformed_or_unknown_stream(tmp_path):
    path = tmp_path / "nvidia-driver.module"
    path.write_text(
        "[nvidia-driver]\nstate=enabled\nstream=latest-open\n",
        encoding="utf-8",
    )

    assert _audit_dnf_module_policy(path).observable is False


def test_zypper_policy_parser_reads_typed_lock_and_rejects_partial_xml():
    result = CommandResult(
        ["zypper", "--xmlout", "--non-interactive", "locks"],
        0,
        stdout=(
            "<?xml version='1.0'?>"
            "<stream><locks size='1'><lock number='1'>"
            "<name>*nvidia*</name><type>package</type>"
            "<range flag='&gt;=' epoch='' version='590' release=''/>"
            "</lock></locks></stream>"
        ),
    )

    selectors = _parse_zypper_policy_selectors(result)

    assert selectors is not None
    assert [(selector.name, selector.relation, selector.version) for selector in selectors] == [
        ("*nvidia*", "ge", "590")
    ]
    result.stdout += "<trailing>"
    assert _parse_zypper_policy_selectors(result) is None


def test_parse_rpm_packages_filters_sorts_and_deduplicates():
    packages = _parse_rpm_packages(
        """
bash\t0\t5.2-1\tx86_64
nvidia-container-toolkit\t0\t1.19.0-1\tx86_64
nvidia-container-toolkit\t0\t1.19.0-1\tx86_64
cuda-compat-13-0\t0\t13.0.0-1\tx86_64
nvidia-open-595\t0\t595.71.05-1\tx86_64
"""
    )
    assert [pkg.name for pkg in packages] == [
        "cuda-compat-13-0",
        "nvidia-container-toolkit",
        "nvidia-open-595",
    ]
    assert [pkg.manager for pkg in packages] == ["rpm", "rpm", "rpm"]


def test_parse_rpm_packages_preserves_coinstalled_kernel_build_versions():
    packages = _parse_rpm_packages(
        "kernel-devel\t0\t5.14.0-427.el9\tx86_64\n"
        "kernel-devel\t0\t5.14.0-503.el9\tx86_64\n"
        "kernel-headers\t0\t5.14.0-503.el9\tx86_64\n"
        "containerd.io\t1\t1.7.27-1.el9\tx86_64"
    )

    assert [(package.name, package.version) for package in packages] == [
        ("containerd.io", "1.7.27-1.el9"),
        ("kernel-devel", "5.14.0-427.el9"),
        ("kernel-devel", "5.14.0-503.el9"),
        ("kernel-headers", "5.14.0-503.el9"),
    ]
    assert packages[0].epoch == "1"


def test_rpm_inventory_accepts_valid_multiversion_and_multiarch_rows():
    packages, complete, _ = _audit_packages(
        "dnf",
        _InventoryRunner(
            "kernel-devel\t0\t5.14.0-427.el9\tx86_64\n"
            "kernel-devel\t0\t5.14.0-503.el9\tx86_64\n"
            "libnvidia-ml\t0\t580.126.16-1\tx86_64\n"
            "libnvidia-ml\t0\t580.126.16-1\ti686\n"
        ),
        "5.14.0-503.el9.x86_64",
    )

    assert complete is True
    assert [
        (package.name, package.version, package.architecture)
        for package in packages
    ] == [
        ("kernel-devel", "5.14.0-427.el9", "x86_64"),
        ("kernel-devel", "5.14.0-503.el9", "x86_64"),
        ("libnvidia-ml", "580.126.16-1", "i686"),
        ("libnvidia-ml", "580.126.16-1", "x86_64"),
    ]


@pytest.mark.parametrize(
    "output",
    [
        "nvidia-open\t0\t580.126.16-1\tx86_64\textra\n",
        "nvidia-open\tnone\t580.126.16-1\tx86_64\n",
        "nvidia-open\t0\t\tx86_64\n",
        "nvidia-open\t0\t580.126.16-1\t\n",
        "nvidia-open\t0\t580.126.16-1\tx86_64\nmalformed\n",
        (
            "nvidia-open\t0\t580.126.16-1\tx86_64\n"
            "nvidia-open\t0\t580.126.16-1\tx86_64\n"
        ),
    ],
)
def test_rpm_inventory_rejects_malformed_or_duplicate_identities(output):
    _, complete, _ = _audit_packages(
        "dnf",
        _InventoryRunner(output),
        "5.14.0-503.el9.x86_64",
    )

    assert complete is False


def test_mig_mode_parses_query_output():
    assert _mig_mode("Disabled\nDisabled\n") == "disabled"
    assert _mig_mode("N/A\n") == "disabled"


def test_mig_mode_reports_mixed_across_gpus():
    assert _mig_mode("Enabled\nDisabled\n") == "mixed"


def test_mig_mode_parses_verbose_output():
    assert _mig_mode("GPU 00000000:01:00.0\n    MIG Mode                    : Enabled\n") == "enabled"
    assert _mig_mode("GPU 00000000:01:00.0\n    MIG Mode                    : Disabled\n") == "disabled"


def test_mig_state_records_capability_and_pending_reboot():
    runner = _QueryRunner(
        "GPU-aaaaaaaaaaaaaaaa, Disabled, Enabled\n"
        "GPU-bbbbbbbbbbbbbbbb, N/A, N/A\n"
    )

    current, pending, capable, gpu_uuids = _audit_mig_state(runner, "")

    assert current == "disabled"
    assert pending == "mixed"
    assert capable is False
    assert gpu_uuids == [
        "GPU-aaaaaaaaaaaaaaaa",
        "GPU-bbbbbbbbbbbbbbbb",
    ]


@pytest.mark.parametrize(
    ("gpu_profile", "placement_size", "compute_profile", "compute_profile_id"),
    [
        ("7g.40gb", 8, "7g.40gb", 4),
        ("7g.80gb", 8, "7c.7g.80gb", 4),
    ],
)
def test_mig_geometry_parses_full_a100_and_h100_style_tables(
    gpu_profile,
    placement_size,
    compute_profile,
    compute_profile_id,
):
    gpu_uuid = "GPU-aaaaaaaaaaaaaaaa"
    mig_uuid = "MIG-bbbbbbbbbbbbbbbb"
    runner = _CommandMapRunner(
        {
            (
                "nvidia-smi",
                "mig",
                "-i",
                gpu_uuid,
                "-lgi",
            ): f"""
+----------------------------------------------------+
| GPU instances:                                     |
| GPU   Name          Profile  Instance   Placement  |
|                       ID       ID       Start:Size |
|====================================================|
|   0  MIG {gpu_profile}       0        1          0:{placement_size}     |
+----------------------------------------------------+
""",
            (
                "nvidia-smi",
                "mig",
                "-i",
                gpu_uuid,
                "-lci",
            ): f"""
+-------------------------------------------------------+
| Compute instances:                                    |
| GPU     GPU       Name             Profile   Instance |
|       Instance                       ID        ID     |
|         ID                                            |
|=======================================================|
|   0      1       MIG {compute_profile}       {compute_profile_id}         0     |
+-------------------------------------------------------+
""",
            ("nvidia-smi", "-L"): (
                f"GPU 0: NVIDIA GPU (UUID: {gpu_uuid})\n"
                f"  MIG {gpu_profile} Device 0: (UUID: {mig_uuid})\n"
            ),
        }
    )

    geometry, device_uuids, complete, results = _audit_mig_geometry(
        runner,
        "enabled",
        "enabled",
        [gpu_uuid],
    )

    assert complete is True
    assert device_uuids == [mig_uuid]
    assert geometry == [
        MigGpuInstance(
            gpu_uuid=gpu_uuid,
            profile=gpu_profile,
            profile_id=0,
            placement_start=0,
            placement_size=placement_size,
            compute_instances=[
                MigComputeInstance(
                    profile=compute_profile,
                    profile_id=compute_profile_id,
                )
            ],
        )
    ]
    assert full_mig_geometry_matches(geometry, [gpu_uuid]) is True
    assert [result.command for result in results] == [
        ["nvidia-smi", "mig", "-i", gpu_uuid, "-lgi"],
        ["nvidia-smi", "mig", "-i", gpu_uuid, "-lci"],
        ["nvidia-smi", "-L"],
    ]


def test_mig_geometry_accepts_stable_disabled_mode_without_instance_queries():
    runner = _CommandMapRunner({})

    geometry, device_uuids, complete, results = _audit_mig_geometry(
        runner,
        "disabled",
        "disabled",
        ["GPU-aaaaaaaaaaaaaaaa"],
    )

    assert (geometry, device_uuids, complete, results) == ([], [], True, [])
    assert runner.calls == []


def test_mig_geometry_fails_closed_when_device_uuid_count_is_incomplete():
    gpu_uuid = "GPU-aaaaaaaaaaaaaaaa"
    runner = _CommandMapRunner(
        {
            ("nvidia-smi", "mig", "-i", gpu_uuid, "-lgi"): (
                "| 0 MIG 7g.80gb 0 1 0:8 |\n"
            ),
            ("nvidia-smi", "mig", "-i", gpu_uuid, "-lci"): (
                "| 0 1 MIG 7g.80gb 4 0 |\n"
            ),
            ("nvidia-smi", "-L"): (
                f"GPU 0: NVIDIA H100 (UUID: {gpu_uuid})\n"
            ),
        }
    )

    geometry, device_uuids, complete, _ = _audit_mig_geometry(
        runner,
        "enabled",
        "enabled",
        [gpu_uuid],
    )

    assert (geometry, device_uuids, complete) == ([], [], False)


@pytest.mark.parametrize(
    ("output", "supported"),
    [
        ("7.5\n8.0\n", True),
        ("7.0\n8.0\n", False),
        ("unknown\n", None),
    ],
)
def test_open_module_support_requires_turing_or_newer(output, supported):
    assert _audit_open_module_support(_QueryRunner(output)) is supported


def test_docker_runtime_audit_parses_runtime_names_as_json():
    runtime = _audit_runtime(
        _RuntimeRunner('{"runc":{"path":"nvidia-helper"}}'), True
    )
    assert runtime.docker_gpus_usable is False


def test_docker_runtime_audit_detects_nvidia_runtime_key():
    runtime = _audit_runtime(_RuntimeRunner('{"runc":{},"nvidia":{}}'), True)
    assert runtime.docker_gpus_usable is True


@pytest.mark.parametrize("active", [False, None])
def test_docker_runtime_audit_never_socket_activates_inactive_or_unknown_daemon(
    active,
):
    runner = _RuntimeRunner('{"runc":{},"nvidia":{}}')

    runtime = _audit_runtime(runner, active)

    assert runtime.docker_installed is True
    assert runtime.docker_gpus_usable is None
    assert runner.calls == []


def test_audit_preserves_positive_service_observations_despite_missing_inventory():
    audit = audit_host(_ContradictoryServiceRunner())

    assert audit.runtime.docker_installed is False
    assert audit.docker_service_active is True
    assert audit.docker_service_enabled is True
    assert audit.docker_service_unit_file_state == "enabled"
    assert audit.docker_socket_active is True
    assert audit.docker_socket_enabled is True
    assert audit.docker_socket_unit_file_state == "enabled"
    assert audit.nvidia_persistenced_active is True
    assert audit.nvidia_persistenced_enabled is True
    assert audit.nvidia_persistenced_unit_file_state == "enabled"
    assert audit.packages == []
    assert audit.fabric_manager_active is True
    assert audit.fabric_manager_enabled is True
    assert audit.fabric_manager_unit_file_state == "enabled"


@pytest.mark.parametrize(
    ("load_state", "active_state", "unit_file_state", "expected"),
    [
        ("loaded", "active", "enabled", (True, True, "enabled")),
        ("loaded", "inactive", "disabled", (False, False, "disabled")),
        ("loaded", "inactive", "static", (False, False, "static")),
        ("masked", "inactive", "masked", (False, False, "masked")),
        ("loaded", "inactive", "enabled-runtime", (None, None, None)),
        ("loaded", "inactive", "linked", (None, None, None)),
    ],
)
def test_service_state_requires_exact_restorable_systemd_state(
    load_state,
    active_state,
    unit_file_state,
    expected,
):
    result = CommandResult(
        ["systemctl", "show", "docker.service"],
        0,
        stdout=(
            "Id=docker.service\n"
            f"LoadState={load_state}\n"
            f"ActiveState={active_state}\n"
            f"UnitFileState={unit_file_state}\n"
        ),
    )

    assert _parse_service_state(result, "docker.service") == expected


def test_missing_service_is_observed_as_known_inactive_and_disabled():
    result = CommandResult(
        ["systemctl", "show", "missing.service"],
        0,
        stdout=(
            "Id=missing.service\n"
            "LoadState=not-found\n"
            "ActiveState=inactive\n"
            "UnitFileState=\n"
        ),
    )

    assert _parse_service_state(result, "missing.service") == (
        False,
        False,
        "not-found",
    )


def test_service_state_rejects_systemd_alias_resolution():
    result = CommandResult(
        ["systemctl", "show", "docker.service"],
        0,
        stdout=(
            "Id=podman.service\n"
            "LoadState=loaded\n"
            "ActiveState=inactive\n"
            "UnitFileState=disabled\n"
        ),
    )

    assert _parse_service_state(result, "docker.service") == (None, None, None)


def test_audit_observes_docker_socket_before_any_docker_client_call():
    runner = _AuditOrderRunner()

    audit = audit_host(runner)

    assert audit.docker_socket_active is False
    assert audit.docker_socket_enabled is True
    assert audit.docker_socket_unit_file_state == "enabled"
    assert audit.docker_service_active is True
    systemd_show = [
        "systemctl",
        "show",
        "--no-pager",
        "--property=Id",
        "--property=LoadState",
        "--property=ActiveState",
        "--property=UnitFileState",
    ]
    assert runner.calls[:3] == [
        [*systemd_show, "docker.socket"],
        [*systemd_show, "docker.service"],
        ["docker", "info", "--format", "{{json .Runtimes}}"],
    ]


def test_module_provenance_parses_independent_loaded_and_on_disk_sources(tmp_path):
    loaded_version = tmp_path / "proc-version"
    loaded_version.write_text(
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.126.16\n",
        encoding="utf-8",
    )
    taint = tmp_path / "taint"
    taint.write_text("O\n", encoding="utf-8")

    assert _loaded_module_flavor(loaded_version) is True
    assert _loaded_module_signed(taint) is True
    assert _modinfo_version("filename: x\nversion: 580.126.16\n") == "580.126.16"


def test_closed_or_unsigned_loaded_module_is_detected(tmp_path):
    loaded_version = tmp_path / "proc-version"
    loaded_version.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.126.16\n",
        encoding="utf-8",
    )
    taint = tmp_path / "taint"
    taint.write_text("OE\n", encoding="utf-8")

    assert _loaded_module_flavor(loaded_version) is False
    assert _loaded_module_signed(taint) is False


def test_unobservable_loaded_module_provenance_fails_closed(tmp_path):
    missing = tmp_path / "missing"

    assert _loaded_module_flavor(missing) is None
    assert _loaded_module_signed(missing) is None


def test_cuda_compat_audit_probes_installed_version_once(monkeypatch, tmp_path):
    library = tmp_path / "libcuda.so.1"
    library.write_bytes(b"compatibility library")
    probed = []
    monkeypatch.setattr(
        "nvidia_converge.audit.cuda_compat_library_path",
        lambda version: library,
    )

    def probe(version, runner):
        probed.append((version, runner))
        return CommandResult(["python3", str(library)], 0)

    monkeypatch.setattr("nvidia_converge.audit.probe_cuda_compat_library", probe)
    runner = object()
    observations = _audit_cuda_compatibility(
        [
            PackageInfo("cuda-compat-13-0", "580.1", "apt", True),
            PackageInfo("cuda-compat-13-0", "580.1", "apt", True, "arm64"),
            PackageInfo("cuda-toolkit-13-0", "13.0", "apt", True),
        ],
        runner,
    )

    assert len(observations) == 1
    assert observations[0].version == "13.0"
    assert observations[0].package_name == "cuda-compat-13-0"
    assert observations[0].library_present is True
    assert observations[0].library_probe.returncode == 0
    assert probed == [("13.0", runner)]


def test_cuda_compat_audit_does_not_probe_without_compat_package(monkeypatch):
    monkeypatch.setattr(
        "nvidia_converge.audit.probe_cuda_compat_library",
        lambda version, runner: pytest.fail(f"unexpected probe: {version}, {runner}"),
    )

    assert (
        _audit_cuda_compatibility(
            [PackageInfo("cuda-toolkit-13-0", "13.0", "apt", True)],
            object(),
        )
        == []
    )


class _InventoryRunner:
    def __init__(self, stdout, *, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr

    def run(self, command, *, allow_fail=True):
        del allow_fail
        return CommandResult(
            command,
            self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


class _RuntimeRunner:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def exists(self, name):
        return name == "docker"

    def run(self, command, *, allow_fail=True):
        del allow_fail
        self.calls.append(command)
        return CommandResult(command, 0, stdout=self.output)


class _ContradictoryServiceRunner:
    def exists(self, name):
        return name == "systemctl"

    def resolve_executable(self, name):
        del name

    def run(self, command, *, allow_fail=True):
        del allow_fail
        if command[:2] == ["systemctl", "show"]:
            service = command[-1]
            return CommandResult(
                command,
                0,
                stdout=(
                    f"Id={service}\n"
                    "LoadState=loaded\n"
                    "ActiveState=active\n"
                    "UnitFileState=enabled\n"
                ),
            )
        return CommandResult(command, 127, stderr="not found")


class _AuditOrderRunner:
    def __init__(self):
        self.calls = []

    def exists(self, name):
        return name in {"docker", "systemctl"}

    def resolve_executable(self, name):
        del name

    def run(self, command, *, allow_fail=True):
        del allow_fail
        self.calls.append(command)
        if command[:2] == ["systemctl", "show"]:
            service = command[-1]
            active = "inactive" if service == "docker.socket" else "active"
            return CommandResult(
                command,
                0,
                stdout=(
                    f"Id={service}\n"
                    "LoadState=loaded\n"
                    f"ActiveState={active}\n"
                    "UnitFileState=enabled\n"
                ),
            )
        if command[:2] == ["docker", "info"]:
            return CommandResult(command, 0, stdout='{"nvidia": {}}')
        return CommandResult(command, 127, stderr="not found")


class _QueryRunner:
    def __init__(self, output):
        self.output = output

    def exists(self, name):
        return name == "nvidia-smi"

    def run(self, command, *, allow_fail=True):
        del allow_fail
        return CommandResult(command, 0, stdout=self.output)


class _CommandMapRunner:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def exists(self, name):
        return name == "nvidia-smi"

    def run(self, command, *, allow_fail=True):
        del allow_fail
        self.calls.append(command)
        output = self.outputs.get(tuple(command))
        if output is None:
            return CommandResult(command, 1, stderr="unexpected command")
        return CommandResult(command, 0, stdout=output)


def _fabric_health_xml(
    gpu_uuid,
    state,
    status,
    *,
    duplicate_gpu=False,
    omit_status=False,
):
    status_xml = "" if omit_status else f"<status>{status}</status>"
    gpu = (
        f"<gpu><uuid>{gpu_uuid}</uuid><fabric>"
        f"<state>{state}</state>{status_xml}</fabric></gpu>"
    )
    duplicate = gpu if duplicate_gpu else ""
    return (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">\n'
        f"<nvidia_smi_log>{gpu}{duplicate}</nvidia_smi_log>"
    )


def test_fabric_manager_health_requires_every_gpu_handshake_to_succeed():
    gpu_uuids = ["GPU-aaaaaaaaaaaaaaaa", "GPU-bbbbbbbbbbbbbbbb"]
    commands = [
        ("nvidia-smi", "-q", "-x", "-i", gpu_uuid)
        for gpu_uuid in gpu_uuids
    ]
    runner = _CommandMapRunner(
        {
            commands[0]: _fabric_health_xml(gpu_uuids[0], "Completed", "Success"),
            commands[1]: _fabric_health_xml(
                gpu_uuids[1], "Completed", "NVML_SUCCESS"
            ),
        }
    )
    applicable, healthy, result = _audit_fabric_manager_health(
        runner,
        gpu_uuids,
    )

    assert applicable is True
    assert healthy is True
    assert result is not None
    assert result.command == list(commands[1])
    assert runner.calls == [list(command) for command in commands]


def test_fabric_manager_health_rejects_unsupported_or_incomplete_topology():
    assert _audit_fabric_manager_health(
        _QueryRunner(
            _fabric_health_xml(
                "GPU-aaaaaaaaaaaaaaaa", "Not Supported", "N/A"
            )
        ),
        ["GPU-aaaaaaaaaaaaaaaa"],
    )[:2] == (False, False)
    gpu_uuids = ["GPU-aaaaaaaaaaaaaaaa", "GPU-bbbbbbbbbbbbbbbb"]
    assert _audit_fabric_manager_health(
        _CommandMapRunner(
            {
                ("nvidia-smi", "-q", "-x", "-i", gpu_uuids[0]): (
                    _fabric_health_xml(gpu_uuids[0], "Completed", "Success")
                ),
                ("nvidia-smi", "-q", "-x", "-i", gpu_uuids[1]): (
                    _fabric_health_xml(gpu_uuids[1], "In Progress", "Success")
                ),
            }
        ),
        gpu_uuids,
    )[:2] == (True, False)


@pytest.mark.parametrize(
    "output",
    [
        "",
        _fabric_health_xml("GPU-cccccccccccccccc", "Completed", "Success"),
        _fabric_health_xml(
            "GPU-aaaaaaaaaaaaaaaa",
            "Completed",
            "Success",
            duplicate_gpu=True,
        ),
        _fabric_health_xml(
            "GPU-aaaaaaaaaaaaaaaa",
            "Completed",
            "Success",
            omit_status=True,
        ),
    ],
)
def test_fabric_manager_health_requires_exact_gpu_uuid_coverage(output):
    applicable, healthy, result = _audit_fabric_manager_health(
        _QueryRunner(output),
        ["GPU-aaaaaaaaaaaaaaaa"],
    )

    assert applicable is None
    assert healthy is None
    assert result is not None
    assert result.reason in {
        "fabric-health-output-malformed",
        "fabric-health-gpu-coverage-incomplete",
    }
