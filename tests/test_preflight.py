import argparse
import hashlib
import json
from pathlib import Path

import pytest

from nvidia_converge.cli import _execute_command
from nvidia_converge.dnf_module_transaction import (
    _combined_state_digest,
    _proof_preflight_sha256,
    dnf_module_enable_command,
)
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    HostAudit,
    KernelInfo,
    ModuleInfo,
    PackageInfo,
    PackagePayload,
    PackagePayloadBundle,
    PackagePolicyInfo,
    PackagePolicySelector,
    PlanAction,
    RollbackSnapshot,
    RuntimeInfo,
)
from nvidia_converge.package_payloads import payload_bundle_directory
from nvidia_converge.planner import lock_actions, package_install_commands
from nvidia_converge.preflight import (
    PackagePreflightError,
    preflight_package_install,
    preflight_package_lock,
    preflight_package_rollback,
    preflight_snapshot_restore_availability,
)


def test_apt_preflight_simulates_all_targets_including_exact_pin():
    desired = DesiredState(driver="595.71.05")
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(["nvidia-driver-pinning-595.71.05"]),
            _apt_architecture_result(),
            _apt_forward_result(
                [
                    "cuda-drivers",
                    "nvidia-container-toolkit",
                    "docker-ce",
                    "linux-headers-6.8.0-test",
                    "build-essential",
                ]
            ),
        ]
    )

    results = preflight_package_install(desired, _audit("apt-get"), runner)

    assert results[0].returncode == 0
    assert runner.calls[1][0][-1] == "nvidia-driver-pinning-595.71.05"
    command = runner.calls[3][0]
    assert command[:4] == [
        "apt-get",
        "--simulate",
        "install",
        "--allow-downgrades",
    ]
    assert "--no-install-recommends" in command
    assert "nvidia-driver-pinning-595.71.05" not in command
    assert "nvidia-open" not in command
    assert "cuda-drivers" in command
    assert runner.calls[0][1:] == (False, True)


def test_apt_preflight_fails_closed_when_solver_fails():
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(["nvidia-driver-pinning-580"]),
            _apt_architecture_result(),
            CommandResult(
                [],
                100,
                stderr="E: Unable to locate package nvidia-open",
            ),
        ]
    )

    with pytest.raises(PackagePreflightError, match="could not resolve") as raised:
        preflight_package_install(DesiredState(), _audit("apt-get"), runner)

    assert raised.value.results[-1].returncode == 100
    assert "Unable to locate package" in str(raised.value)


def test_dnf_preflight_checks_future_stream_and_each_repository_target():
    audit = _audit("dnf")
    audit.os_id = "rhel"
    audit.os_version = "9.6"
    audit.kernel.running = "5.14.0-503.el9.x86_64"
    audit.kernel.headers_installed = True
    targets = [
        "nvidia-open",
        "nvidia-container-toolkit",
        "docker-ce",
        "gcc",
        "make",
    ]
    runner = _FakeRunner(
        [
            _dnf_module_result(),
            *(CommandResult([], 0, stdout="repository match") for _ in targets),
            _dnf_forward_result(targets),
        ]
    )

    preflight_package_install(DesiredState(), audit, runner)

    assert runner.calls[0][0] == dnf_module_enable_command(
        apply=False,
        stream="580-open",
    )
    package_queries = [call[0] for call in runner.calls[1:-1]]
    assert package_queries
    assert all(command[:6] == [
        "dnf",
        "-C",
        "-q",
        "repoquery",
        "--available",
        "--disable-modular-filtering",
    ] for command in package_queries)
    assert runner.calls[-1][0][:3] == ["python3", "-I", "-c"]
    assert "install_weak_deps = False" in runner.calls[-1][0][3]
    assert all(call[1:] == (False, True) for call in runner.calls)


def test_dnf_preflight_rejects_empty_successful_repoquery_output():
    audit = _audit("dnf")
    audit.kernel.headers_installed = True
    runner = _FakeRunner(
        [
            _dnf_module_result(),
            CommandResult([], 0, stdout=""),
        ]
    )

    with pytest.raises(PackagePreflightError, match="nvidia-open"):
        preflight_package_install(DesiredState(), audit, runner)

    assert len(runner.calls) == 2


def test_dnf_preflight_rejects_untracked_dependency_expansion():
    audit = _prepared_audit("dnf")
    runner = _FakeRunner(
        [
            CommandResult([], 0, stdout="repository match"),
            _dnf_forward_result(["nvidia-open", "curl"]),
        ]
    )

    with pytest.raises(PackagePreflightError, match="rollback-tracked target closure"):
        preflight_package_install(DesiredState(), audit, runner)


def test_dnf_preflight_rejects_pure_managed_package_removal():
    audit = _prepared_audit("dnf")
    audit.packages = [
        PackageInfo(
            "libnvidia-gl",
            "580.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            CommandResult([], 0, stdout="repository match"),
            _dnf_forward_result(
                ["nvidia-open"],
                removals=[("libnvidia-gl", "580.1-1", "x86_64")],
            ),
        ]
    )

    with pytest.raises(PackagePreflightError, match="without a planned replacement"):
        preflight_package_install(DesiredState(), audit, runner)


def test_dnf_preflight_accepts_target_scoped_exact_replacement():
    audit = _prepared_audit("dnf")
    audit.packages = [
        PackageInfo(
            "libnvidia-gl",
            "580.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            CommandResult([], 0, stdout="repository match"),
            _dnf_forward_result(
                ["nvidia-open", ("libnvidia-gl", "580.2-1", "x86_64")],
                removals=[("libnvidia-gl", "580.1-1", "x86_64")],
            ),
        ]
    )

    preflight_package_install(DesiredState(), audit, runner)


@pytest.mark.parametrize(
    "stdout",
    [
        "not-json",
        '{"install":[],"install":[],"remove":[]}',
        '{"install":[],"remove":[]}\n[output truncated: retained first 1 of 2 bytes]',
        (
            '{"install":['
            '{"architecture":"x86_64","epoch":null,"name":"nvidia-open","version":"580.1-1"},'
            '{"architecture":"x86_64","epoch":null,"name":"nvidia-open","version":"580.2-1"}'
            '],"remove":[]}'
        ),
    ],
)
def test_dnf_forward_preflight_rejects_untrusted_solver_evidence(stdout):
    audit = _prepared_audit("dnf")
    runner = _FakeRunner(
        [
            CommandResult([], 0, stdout="repository match"),
            CommandResult([], 0, stdout=stdout),
        ]
    )

    with pytest.raises(PackagePreflightError, match="malformed or truncated"):
        preflight_package_install(DesiredState(), audit, runner)


def test_zypper_preflight_is_no_refresh_dry_run_with_branch_bounds():
    audit = _audit("zypper")
    audit.os_id = "sles"
    audit.os_version = "15.6"
    audit.kernel.running = "6.4.0-150600.23.53-default"
    runner = _FakeRunner(
        [
            _zypper_locks_result(),
            _zypper_forward_result(
                [
                    "nvidia-open",
                    "nvidia-container-toolkit",
                    "docker-ce",
                    "kernel-default-devel",
                    "gcc",
                    "make",
                ]
            ),
        ]
    )

    preflight_package_install(DesiredState(), audit, runner)

    command = runner.calls[1][0]
    assert command[:8] == [
        "zypper",
        "--xmlout",
        "--non-interactive",
        "--no-refresh",
        "install",
        "--dry-run",
        "--no-recommends",
        "nvidia-open>=580",
    ]
    assert "nvidia-open>=580" in command
    assert "nvidia-open<590" in command
    assert not any("fabricmanager" in operand for operand in command)
    assert "kernel-default-devel=6.4.0-150600.23.53" in command
    assert runner.calls[0][1:] == (False, True)


def test_apt_preflight_rejects_unmanaged_solver_removal():
    audit = _prepared_audit("apt-get")
    audit.packages = [
        PackageInfo(
            "libnvidia-gl-580",
            "580.126.16-1",
            "apt",
            True,
            architecture="amd64",
        )
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(
                ["nvidia-open"],
                removals=[("libnvidia-gl-580", "580.126.16-1", "amd64")],
            ),
        ]
    )

    with pytest.raises(PackagePreflightError, match="without a planned replacement"):
        preflight_package_install(DesiredState(), audit, runner)


def test_apt_preflight_rejects_untracked_dependency_expansion():
    audit = _prepared_audit("apt-get")
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(["nvidia-open", "curl"]),
        ]
    )

    with pytest.raises(PackagePreflightError, match="rollback-tracked target closure"):
        preflight_package_install(DesiredState(), audit, runner)


def test_zypper_preflight_fails_closed_when_dry_run_cannot_resolve():
    audit = _audit("zypper")
    audit.kernel.running = "6.4.0-150600.23.53-default"
    runner = _FakeRunner(
        [
            _zypper_locks_result(),
            CommandResult(
                [],
                104,
                stderr="Package 'nvidia-open' not found",
            ),
        ]
    )

    with pytest.raises(PackagePreflightError, match="could not resolve") as raised:
        preflight_package_install(DesiredState(), audit, runner)

    assert raised.value.results[0].returncode == 104
    assert runner.calls[0][1:] == (False, True)


def test_zypper_preflight_rejects_untracked_dependency_expansion():
    audit = _prepared_audit("zypper")
    runner = _FakeRunner([_zypper_forward_result(["nvidia-open", "curl"])])

    with pytest.raises(PackagePreflightError, match="rollback-tracked target closure"):
        preflight_package_install(DesiredState(), audit, runner)


def test_zypper_preflight_rejects_exact_unplanned_removal():
    audit = _prepared_audit("zypper")
    audit.packages = [
        PackageInfo(
            "libnvidia-gl",
            "580.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            _zypper_forward_result(
                ["nvidia-open"],
                removals=[("libnvidia-gl", "580.1-1", "x86_64")],
            )
        ]
    )

    with pytest.raises(PackagePreflightError, match="without a planned replacement"):
        preflight_package_install(DesiredState(), audit, runner)


def test_zypper_preflight_accepts_target_scoped_exact_upgrade():
    audit = _prepared_audit("zypper")
    audit.packages = [
        PackageInfo(
            "libnvidia-gl",
            "580.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            _zypper_forward_result(
                ["nvidia-open"],
                upgrades=[("libnvidia-gl", "580.2-1", "x86_64")],
            )
        ]
    )

    preflight_package_install(DesiredState(), audit, runner)


def test_apt_lock_preflight_checks_pinning_package_without_mutation():
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(["nvidia-driver-pinning-580"]),
        ]
    )

    preflight_package_lock(DesiredState(), _audit("apt-get"), runner)

    assert runner.calls == [
        (["dpkg", "--print-architecture"], False, True),
        (
            [
                "apt-get",
                "--simulate",
                "install",
                "-y",
                "--allow-downgrades",
                "--no-install-recommends",
                "--purge",
                "nvidia-driver-pinning-580",
            ],
            False,
            True,
        )
    ]


def test_dnf_lock_preflight_rebinds_fresh_plan_to_accepted_proof():
    desired = DesiredState()
    audit = _audit("dnf")
    audit.kernel.headers_installed = True
    actions = lock_actions(desired, audit)
    result = _dnf_module_result()
    token = json.loads(result.stdout)["preflight_sha256"]
    runner = _FakeRunner([result])

    preflight_package_lock(
        desired,
        audit,
        runner,
        actions=actions,
    )

    assert runner.calls == [
        (
            dnf_module_enable_command(apply=False, stream="580-open"),
            False,
            True,
        )
    ]
    assert actions[0].commands == [
        dnf_module_enable_command(
            apply=True,
            stream="580-open",
            preflight_sha256=token,
        )
    ]


def test_dnf_lock_preflight_rejects_a_prebound_or_stale_plan():
    desired = DesiredState()
    audit = _audit("dnf")
    audit.kernel.headers_installed = True
    actions = lock_actions(desired, audit)
    actions[0].commands = [
        dnf_module_enable_command(
            apply=True,
            stream="580-open",
            preflight_sha256="f" * 64,
        )
    ]
    runner = _FakeRunner([])

    with pytest.raises(PackagePreflightError, match="fresh plan"):
        preflight_package_lock(
            desired,
            audit,
            runner,
            actions=actions,
        )

    assert runner.calls == []


def test_apt_lock_switches_pins_in_one_exact_solver_transaction():
    audit = _audit("apt-get")
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "nvidia-driver-pinning-570",
            "nvidia-driver-pinning-570",
            "package",
        )
    ]
    audit.packages = [
        PackageInfo(
            "nvidia-driver-pinning-570",
            "570.1",
            "apt",
            True,
            architecture="all",
        )
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(
                ["nvidia-driver-pinning-580"],
                architecture="all",
                removals=[("nvidia-driver-pinning-570", "570.1", "all")],
            ),
        ]
    )

    preflight_package_lock(DesiredState(), audit, runner)

    command = runner.calls[1][0]
    assert command[:3] == ["apt-get", "--simulate", "install"]
    assert "nvidia-driver-pinning-580" in command
    assert "nvidia-driver-pinning-570-" in command
    assert "--purge" in command


def test_apt_lock_rejects_unexpected_atomic_solver_expansion():
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(
                ["nvidia-driver-pinning-580", "nvidia-container-toolkit"]
            ),
        ]
    )

    with pytest.raises(PackagePreflightError, match="outside the desired pin"):
        preflight_package_lock(DesiredState(), _audit("apt-get"), runner)


def test_preflight_uses_the_same_apt_install_targets_as_apply():
    audit = _audit("apt-get")
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "nvidia-driver-pinning-580",
            "nvidia-driver-pinning-580",
            "package",
        )
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            _apt_forward_result(
                [
                    "nvidia-open",
                    "nvidia-container-toolkit",
                    "docker-ce",
                    "linux-headers-6.8.0-test",
                    "build-essential",
                ]
            ),
        ]
    )

    preflight_package_install(DesiredState(), audit, runner)

    actual = package_install_commands(DesiredState(), audit)[0]
    assert _command_operands(actual) == _command_operands(runner.calls[1][0])


def test_preflight_uses_the_same_zypper_install_targets_as_apply():
    audit = _audit("zypper")
    audit.kernel.running = "6.4.0-150600.23.53-default"
    audit.package_policy.selectors = [
        PackagePolicySelector("1", "*nvidia*", "package", "ge", "590")
    ]
    runner = _FakeRunner(
        [
            _zypper_forward_result(
                [
                    "nvidia-open",
                    "nvidia-container-toolkit",
                    "docker-ce",
                    "kernel-default-devel",
                    "gcc",
                    "make",
                ]
            )
        ]
    )

    preflight_package_install(DesiredState(), audit, runner)

    actual = package_install_commands(DesiredState(), audit)[0]
    assert _command_operands(actual) == _command_operands(runner.calls[0][0])


def test_apt_rollback_preflight_resolves_restore_before_any_mutation(
    tmp_path: Path,
):
    audit = _audit("apt-get")
    audit.packages = [
        PackageInfo(
            "nvidia-driver-580-open",
            "580.1",
            "apt",
            True,
            architecture="amd64",
        )
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            CommandResult([], 100, stderr="Version '570.1' was not found"),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="apt-get",
        packages=[
            PackageInfo(
                "nvidia-driver-570",
                "570.1",
                "apt",
                True,
                architecture="amd64",
            )
        ],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="exact rollback"):
        preflight_package_rollback(snapshot, audit, runner)

    transaction_calls = _transaction_calls(runner)
    assert transaction_calls[1][0][:3] == ["apt-get", "--simulate", "install"]
    assert "--no-download" in transaction_calls[1][0]
    assert str(tmp_path / snapshot.package_payloads.directory) in " ".join(
        transaction_calls[1][0]
    )
    assert transaction_calls[1][1:] == (False, True)


def test_dnf_rollback_preflight_requires_each_exact_restore_nevra(
    tmp_path: Path,
):
    audit = _audit("dnf")
    audit.packages = [
        PackageInfo(
            "nvidia-open",
            "580.2-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            CommandResult(
                [],
                0,
                stdout=(
                    '{"install":["nvidia-open-580.1-1.x86_64"],'
                    '"remove":["nvidia-open-580.2-1.x86_64"]}'
                ),
            ),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="dnf",
        packages=[
            PackageInfo(
                "nvidia-open",
                "580.1-1",
                "rpm",
                True,
                architecture="x86_64",
            )
        ],
        kernel=audit.kernel.running,
    )

    preflight_package_rollback(snapshot, audit, runner)

    transaction_calls = _transaction_calls(runner)
    assert len(transaction_calls) == 1
    command = transaction_calls[0][0]
    assert command[:3] == ["python3", "-I", "-c"]
    assert command[4] == "--check"
    assert command[5].startswith(str(tmp_path / snapshot.package_payloads.directory))
    assert command[command.index("--expect-install") + 1] == (
        "nvidia-open-580.1-1.x86_64"
    )
    assert command[command.index("--expect-remove") + 1] == (
        "nvidia-open-580.2-1.x86_64"
    )
    assert transaction_calls[0][1:] == (False, True)


def test_apt_rollback_preflight_resolves_restore_and_removal_atomically(
    tmp_path: Path,
):
    audit = _audit("apt-get")
    audit.packages = [
        PackageInfo(
            "nvidia-driver-580-open",
            "580.1",
            "apt",
            True,
            architecture="amd64",
        ),
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "apt",
            True,
            architecture="amd64",
        ),
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            CommandResult(
                [],
                0,
                stdout=(
                    "Inst nvidia-driver-580-open [580.1] (580.0 repo [amd64])\n"
                    "Remv nvidia-container-toolkit [1.19.1-1]\n"
                ),
            ),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="apt-get",
        packages=[
            PackageInfo(
                "nvidia-driver-580-open",
                "580.0",
                "apt",
                True,
                architecture="amd64",
            )
        ],
        kernel=audit.kernel.running,
    )

    preflight_package_rollback(snapshot, audit, runner)

    transaction_calls = _transaction_calls(runner)
    assert len(transaction_calls) == 2
    command = transaction_calls[1][0]
    assert command[:3] == ["apt-get", "--simulate", "install"]
    assert "--no-download" in command
    assert any(operand.endswith(".deb") for operand in command)
    assert "nvidia-container-toolkit:amd64-" in command


def test_apt_rollback_preflight_rejects_dependency_outside_baseline(
    tmp_path: Path,
):
    audit = _audit("apt-get")
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "apt",
            True,
            architecture="amd64",
        )
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            CommandResult(
                [],
                0,
                stdout=(
                    "Remv nvidia-container-toolkit [1.19.1-1]\n"
                    "Inst curl (8.0 repo [amd64])\n"
                ),
            ),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="apt-get",
        packages=[],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="outside the exact baseline"):
        preflight_package_rollback(snapshot, audit, runner)


def test_apt_rollback_preflight_preserves_multiarch_removal_identity(
    tmp_path: Path,
):
    audit = _audit("apt-get")
    audit.packages = [
        PackageInfo("libnvidia-gl", "580.1", "apt", True, architecture="amd64"),
        PackageInfo("libnvidia-gl", "580.1", "apt", True, architecture="arm64"),
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            CommandResult(
                [],
                0,
                stdout="Remv libnvidia-gl:amd64 [580.1]",
            ),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="apt-get",
        packages=[
            PackageInfo(
                "libnvidia-gl",
                "580.1",
                "apt",
                True,
                architecture="amd64",
            )
        ],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="outside the exact rollback set"):
        preflight_package_rollback(snapshot, audit, runner)

    assert "libnvidia-gl:arm64-" in _transaction_calls(runner)[1][0]


def test_apt_rollback_preflight_preserves_exact_restore_version_identity(
    tmp_path: Path,
):
    audit = _audit("apt-get")
    audit.packages = [
        PackageInfo("nvidia-open", "580.2", "apt", True, architecture="amd64")
    ]
    runner = _FakeRunner(
        [
            _apt_architecture_result(),
            CommandResult(
                [],
                0,
                stdout="Inst nvidia-open [580.2] (580.0 repo [amd64])",
            ),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="apt-get",
        packages=[
            PackageInfo(
                "nvidia-open",
                "580.1",
                "apt",
                True,
                architecture="amd64",
            )
        ],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="outside the exact baseline"):
        preflight_package_rollback(snapshot, audit, runner)


def test_dnf_rollback_preflight_rejects_external_reverse_dependency(
    tmp_path: Path,
):
    audit = _audit("dnf")
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            CommandResult(
                [],
                0,
                stdout=(
                    '{"install":[],"remove":['
                    '"docker-ce-27.0-1.x86_64",'
                    '"nvidia-container-toolkit-1.19.1-1.x86_64"]}'
                ),
            )
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="dnf",
        packages=[],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="outside the exact rollback set"):
        preflight_package_rollback(snapshot, audit, runner)

    assert _transaction_calls(runner)[0][0][:3] == ["python3", "-I", "-c"]


def test_dnf_rollback_preflight_accepts_dependency_closed_exact_removal(
    tmp_path: Path,
):
    audit = _audit("dnf")
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "rpm",
            True,
            architecture="x86_64",
        ),
        PackageInfo(
            "libnvidia-container1",
            "1.19.1-1",
            "rpm",
            True,
            architecture="x86_64",
        ),
    ]
    runner = _FakeRunner(
        [
            CommandResult(
                [],
                0,
                stdout=(
                    '{"install":[],"remove":['
                    '"libnvidia-container1-1.19.1-1.x86_64",'
                    '"nvidia-container-toolkit-1.19.1-1.x86_64"]}'
                ),
            ),
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="dnf",
        packages=[],
        kernel=audit.kernel.running,
    )

    preflight_package_rollback(snapshot, audit, runner)

    commands = [call[0] for call in _transaction_calls(runner)]
    assert len(commands) == 1
    assert commands[0][:3] == ["python3", "-I", "-c"]
    remove_marker = commands[0].index("--remove")
    assert commands[0][remove_marker : remove_marker + 3] == [
        "--remove",
        "libnvidia-container1-1.19.1-1.x86_64",
        "nvidia-container-toolkit-1.19.1-1.x86_64",
    ]


@pytest.mark.parametrize(
    "stdout",
    [
        "not-json",
        '{"install":[],"remove":[],"remove":[]}',
        '{"install":[],"remove":[]}\n[output truncated: retained first 1 of 2 bytes]',
        '{"install":[],"remove":[1]}',
        (
            '{"install":[],"remove":['
            '"nvidia-container-toolkit-1.19.1-1.x86_64",'
            '"nvidia-container-toolkit-1.19.1-1.x86_64"]}'
        ),
    ],
)
def test_dnf_rollback_preflight_rejects_untrusted_solver_evidence(
    stdout,
    tmp_path: Path,
):
    audit = _audit("dnf")
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner([CommandResult([], 0, stdout=stdout)])
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="dnf",
        packages=[],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="malformed or truncated"):
        preflight_package_rollback(snapshot, audit, runner)


def test_zypper_rollback_preflight_matches_atomic_cached_transaction(
    tmp_path: Path,
):
    audit = _audit("zypper")
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            CommandResult(
                [],
                0,
                stdout=(
                    "<stream><install-summary packages-to-change='1'>"
                    "<to-remove><solvable status='installed' kind='package' "
                    "name='nvidia-container-toolkit' edition='1.19.1-1' "
                    "arch='x86_64'/></to-remove></install-summary></stream>"
                ),
            )
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="zypper",
        packages=[],
        kernel=audit.kernel.running,
    )

    preflight_package_rollback(snapshot, audit, runner)

    assert _transaction_calls(runner)[0][0] == [
        "zypper",
        "--xmlout",
        "--non-interactive",
        "--disable-repositories",
        "--no-refresh",
        "install",
        "--oldpackage",
        "--no-recommends",
        "--no-force-resolution",
        "--dry-run",
        "--",
        "-nvidia-container-toolkit.x86_64=1.19.1-1",
    ]


def test_zypper_rollback_preflight_rejects_solver_expansion(
    tmp_path: Path,
):
    audit = _audit("zypper")
    audit.packages = [
        PackageInfo(
            "nvidia-container-toolkit",
            "1.19.1-1",
            "rpm",
            True,
            architecture="x86_64",
        )
    ]
    runner = _FakeRunner(
        [
            CommandResult(
                [],
                0,
                stdout=(
                    "<stream><install-summary packages-to-change='2'>"
                    "<to-remove><solvable status='installed' kind='package' "
                    "name='nvidia-container-toolkit' edition='1.19.1-1' "
                    "arch='x86_64'/></to-remove>"
                    "<to-install><solvable status='not-installed' kind='package' "
                    "name='curl' edition='8.0-1' arch='x86_64'/>"
                    "</to-install></install-summary></stream>"
                ),
            )
        ]
    )
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="zypper",
        packages=[],
        kernel=audit.kernel.running,
    )

    with pytest.raises(PackagePreflightError, match="outside the exact baseline"):
        preflight_package_rollback(snapshot, audit, runner)


def test_apt_snapshot_preflight_validates_retained_exact_baseline_payload(
    tmp_path: Path,
):
    runner = _FakeRunner()
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="apt-get",
        packages=[
            PackageInfo(
                "nvidia-driver-570",
                "570.1",
                "apt",
                True,
                architecture="amd64",
            )
        ],
        kernel="6.8.0-test",
    )

    results = preflight_snapshot_restore_availability(snapshot, runner)

    assert results[0].command == ["validate-package-payloads"]
    assert runner.calls[0][0][:3] == ["dpkg-deb", "--show", "--showformat=${Package}\\t${Version}\\t${Architecture}\\n"]
    assert runner.calls[0][0][-1].startswith(
        str(tmp_path / snapshot.package_payloads.directory)
    )
    assert runner.calls[0][1:] == (False, True)


def test_zypper_snapshot_preflight_validates_signed_retained_exact_payload(
    tmp_path: Path,
):
    runner = _FakeRunner()
    snapshot = _snapshot_with_retained_payloads(
        tmp_path,
        runner,
        package_manager="zypper",
        packages=[
            PackageInfo(
                "nvidia-open",
                "570.1-1",
                "rpm",
                True,
                architecture="x86_64",
            )
        ],
        kernel="6.4.0-test-default",
    )

    results = preflight_snapshot_restore_availability(snapshot, runner)

    assert results[0].command == ["validate-package-payloads"]
    assert [call[0][0] for call in runner.calls] == ["rpm", "rpmkeys"]
    assert all(call[1:] == (False, True) for call in runner.calls)


def test_apply_install_preflight_failure_precedes_snapshot(monkeypatch):
    audit = _audit("apt-get")
    events: list[str] = []
    emitted = []

    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)
    monkeypatch.setattr("nvidia_converge.cli.diagnose", lambda desired, host: [])
    monkeypatch.setattr(
        "nvidia_converge.cli.build_plan",
        lambda desired, host, findings: [
            PlanAction("install.packages", "Install packages.", [["apt-get", "install"]])
        ],
    )

    def fail_preflight(desired, host, runner):
        events.append("preflight")
        raise PackagePreflightError(
            "repository metadata is stale",
            package_manager="apt-get",
            packages=["nvidia-open"],
            results=[CommandResult(["apt-get", "--simulate", "install"], 100)],
        )

    def unexpected_snapshot(*args, **kwargs):
        events.append("snapshot")
        raise AssertionError("snapshot must not be created before preflight succeeds")

    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_install", fail_preflight
    )
    monkeypatch.setattr("nvidia_converge.cli.create_snapshot", unexpected_snapshot)
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, out_path, json_stdout, apply: emitted.append(report),
    )

    args = argparse.Namespace(
        command="install",
        allow_disruption=True,
        allow_active_workloads=True,
    )
    rc = _execute_command(args, DesiredState(), None, False, True, None)

    assert rc == 2
    assert events == ["preflight"]
    assert emitted[0].rollback is None
    finding = next(
        finding
        for finding in emitted[0].findings
        if finding.id == "packages.preflight.failed"
    )
    assert finding.evidence["packages"] == ["nvidia-open"]


def test_apply_lock_policy_preflight_failure_precedes_snapshot(monkeypatch):
    audit = _audit("apt-get")
    events: list[str] = []
    emitted = []

    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)
    monkeypatch.setattr("nvidia_converge.cli.diagnose", lambda desired, host: [])
    monkeypatch.setattr(
        "nvidia_converge.cli.lock_actions",
        lambda desired, host: [
            PlanAction("lock.apt", "Lock packages.", [["apt-get", "install"]])
        ],
    )

    def fail_lock_preflight(desired, host, runner):
        events.append("preflight")
        raise PackagePreflightError(
            "pin package unavailable",
            package_manager="apt-get",
            packages=["nvidia-driver-pinning-580"],
            results=[CommandResult(["apt-get", "--simulate"], 100)],
        )

    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_lock", fail_lock_preflight
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.create_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot must follow policy preflight")
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, out_path, json_stdout, apply: emitted.append(report),
    )

    args = argparse.Namespace(
        command="lock",
        allow_disruption=True,
        allow_active_workloads=True,
    )
    rc = _execute_command(args, DesiredState(), None, False, True, None)

    assert rc == 2
    assert events == ["preflight"]
    assert emitted[0].rollback is None
    assert any(
        finding.id == "packages.policy-preflight.failed"
        for finding in emitted[0].findings
    )


def test_apply_lock_satisfied_policy_is_a_safe_noop(monkeypatch):
    audit = _audit("apt-get")
    audit.package_policy.selectors = [
        PackagePolicySelector(
            "nvidia-driver-pinning-580",
            "nvidia-driver-pinning-580",
            "package",
        )
    ]
    emitted = []
    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)
    monkeypatch.setattr("nvidia_converge.cli.diagnose", lambda desired, host: [])
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_lock",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("a satisfied policy must not be preflighted again")
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._create_snapshot_with_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a satisfied policy must not create a snapshot")
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._maintenance_gate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a satisfied policy must not enter a disruption gate")
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, out_path, json_stdout, apply: emitted.append(
            report
        ),
    )
    args = argparse.Namespace(
        command="lock",
        allow_disruption=False,
        allow_active_workloads=False,
    )

    rc = _execute_command(args, DesiredState(), None, False, True, None)

    assert rc == 0
    assert emitted[0].plan == []
    assert emitted[0].verification[-1].name == "package-policy.lock"
    assert emitted[0].verification[-1].ok is True


def test_apply_rollback_package_preflight_failure_precedes_workload_probe(
    monkeypatch,
):
    audit = _audit("apt-get")
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=None,
        commands=[],
        package_manager="apt-get",
    )
    events: list[str] = []
    emitted = []
    monkeypatch.setattr("nvidia_converge.cli.load_snapshot", lambda path: snapshot)
    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_snapshot_for_apply",
        lambda *args: None,
    )

    def fail_rollback_preflight(snapshot_arg, host, runner):
        events.append("preflight")
        raise PackagePreflightError(
            "baseline package unavailable",
            package_manager="apt-get",
            packages=["nvidia-driver-570=570.1"],
            results=[CommandResult(["apt-get", "--simulate"], 100)],
        )

    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_rollback",
        fail_rollback_preflight,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._probe_active_gpu_workloads",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("workload probe must follow package preflight")
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, out_path, json_stdout, apply: emitted.append(report),
    )
    args = argparse.Namespace(
        command="rollback",
        snapshot=snapshot.path,
        allow_disruption=True,
        allow_active_workloads=True,
    )

    rc = _execute_command(args, DesiredState(), None, False, True, None)

    assert rc == 2
    assert events == ["preflight"]
    assert any(
        finding.id == "rollback.packages-preflight-failed"
        for finding in emitted[0].findings
    )


def _audit(package_manager: str) -> HostAudit:
    os_id = "ubuntu"
    os_version = "24.04"
    kernel = "6.8.0-test"
    if package_manager == "dnf":
        os_id = "rhel"
        os_version = "9.6"
        kernel = "5.14.0-503.el9.x86_64"
    elif package_manager == "zypper":
        os_id = "sles"
        os_version = "15.6"
        kernel = "6.4.0-150600.23.53-default"
    return HostAudit(
        timestamp="2026-08-02T00:00:00+00:00",
        os_id=os_id,
        os_version=os_version,
        package_manager=package_manager,
        kernel=KernelInfo(kernel, False),
        module=ModuleInfo(False, None, None, None, []),
        runtime=RuntimeInfo(False, False, False),
        packages=[],
        nvidia_smi=CommandResult(["nvidia-smi"], 127),
        nvml=CommandResult(["python3"], 1),
        fabric_manager_active=False,
        mig_mode=None,
        docker_service_active=False,
        docker_service_enabled=False,
        package_policy=PackagePolicyInfo(package_manager, True),
    )


def _prepared_audit(package_manager: str) -> HostAudit:
    audit = _audit(package_manager)
    audit.kernel.headers_installed = True
    audit.kernel.compiler = "/usr/bin/gcc"
    audit.runtime = RuntimeInfo(True, True, False)
    if package_manager == "apt-get":
        audit.package_policy.selectors = [
            PackagePolicySelector(
                "nvidia-driver-pinning-580",
                "nvidia-driver-pinning-580",
                "package",
            )
        ]
    elif package_manager == "dnf":
        audit.package_policy.selectors = [
            PackagePolicySelector(
                "nvidia-driver",
                "nvidia-driver",
                "module",
                "stream",
                "580-open",
            )
        ]
    elif package_manager == "zypper":
        audit.package_policy.selectors = [
            PackagePolicySelector("1", "*nvidia*", "package", "ge", "590")
        ]
    return audit


def _apt_architecture_result(architecture: str = "amd64") -> CommandResult:
    return CommandResult([], 0, stdout=architecture)


def _apt_forward_result(
    installs: list[str],
    *,
    architecture: str = "amd64",
    removals: list[tuple[str, str, str]] | None = None,
) -> CommandResult:
    lines = [
        f"Inst {name} (1.0-1 repository [{architecture}])" for name in installs
    ]
    lines.extend(
        f"Remv {name}:{package_architecture} [{version}]"
        for name, version, package_architecture in removals or []
    )
    return CommandResult([], 0, stdout="\n".join(lines))


def _dnf_forward_result(
    installs: list[str | tuple[str, str, str]],
    *,
    removals: list[tuple[str, str, str]] | None = None,
) -> CommandResult:
    def record(item: str | tuple[str, str, str]) -> dict[str, str | None]:
        if isinstance(item, str):
            name, version, architecture = item, "1.0-1", "x86_64"
        else:
            name, version, architecture = item
        return {
            "architecture": architecture,
            "epoch": None,
            "name": name,
            "version": version,
        }

    payload = {
        "install": [record(item) for item in installs],
        "remove": [record(item) for item in removals or []],
    }
    return CommandResult([], 0, stdout=json.dumps(payload, separators=(",", ":")))


def _dnf_module_result() -> CommandResult:
    module_digest = "1" * 64
    failsafe_digest = "3" * 64
    state_digest = _combined_state_digest(module_digest, failsafe_digest)
    payload: dict[str, object] = {
        "active_modules_count": 1,
        "active_modules_sha256": "d" * 64,
        "active_target": [
            {
                "architecture": "x86_64",
                "context": "rhel9",
                "name": "nvidia-driver",
                "repository": "cuda-rhel9-x86_64",
                "stream": "580-open",
                "version": "202608020001",
                "yaml_sha256": "c" * 64,
            }
        ],
        "applied": False,
        "changes": {
            "disable": [],
            "enable": {"nvidia-driver": "580-open"},
            "install_profiles": {},
            "remove_profiles": {},
            "reset": [],
            "switch": {},
        },
        "failsafe_changed_files": [],
        "failsafe_target": {
            "filename": "nvidia-driver:580-open:x86_64.yaml",
            "yaml_sha256": "c" * 64,
        },
        "module_changed_files": [],
        "module_failsafe_after_sha256": failsafe_digest,
        "module_failsafe_before_sha256": failsafe_digest,
        "module_platform_id": "platform:el9",
        "module_state_after_sha256": module_digest,
        "module_state_before_sha256": module_digest,
        "repositories_count": 1,
        "repositories_sha256": "e" * 64,
        "requirements": [],
        "schema": 2,
        "state_after_sha256": state_digest,
        "state_before_sha256": state_digest,
        "target": {"name": "nvidia-driver", "stream": "580-open"},
    }
    payload["preflight_sha256"] = _proof_preflight_sha256(payload)
    return CommandResult([], 0, stdout=json.dumps(payload, separators=(",", ":")))


def _zypper_forward_result(
    installs: list[str],
    *,
    removals: list[tuple[str, str, str]] | None = None,
    upgrades: list[tuple[str, str, str]] | None = None,
) -> CommandResult:
    groups: list[str] = []
    if installs:
        groups.append(
            "<to-install>"
            + "".join(
                "<solvable status='not-installed' kind='package' "
                f"name='{name}' edition='1.0-1' arch='x86_64'/>"
                for name in installs
            )
            + "</to-install>"
        )
    if upgrades:
        groups.append(
            "<to-upgrade>"
            + "".join(
                "<solvable status='other-version' kind='package' "
                f"name='{name}' edition='{version}' arch='{architecture}'/>"
                for name, version, architecture in upgrades
            )
            + "</to-upgrade>"
        )
    if removals:
        groups.append(
            "<to-remove>"
            + "".join(
                "<solvable status='installed' kind='package' "
                f"name='{name}' edition='{version}' arch='{architecture}'/>"
                for name, version, architecture in removals
            )
            + "</to-remove>"
        )
    count = len(installs) + len(upgrades or []) + len(removals or [])
    return CommandResult(
        [],
        0,
        stdout=(
            f"<stream><install-summary packages-to-change='{count}'>"
            + "".join(groups)
            + "</install-summary></stream>"
        ),
    )


def _zypper_locks_result() -> CommandResult:
    return CommandResult(
        ["zypper", "--xmlout", "--non-interactive", "locks"],
        0,
        stdout="<stream><locks size='0'/></stream>",
    )


def _command_operands(command: list[str]) -> list[str]:
    operation = command.index("install")
    return [part for part in command[operation + 1 :] if not part.startswith("-")]


def _snapshot_with_retained_payloads(
    tmp_path: Path,
    runner: "_FakeRunner",
    *,
    package_manager: str,
    packages: list[PackageInfo],
    kernel: str,
) -> RollbackSnapshot:
    snapshot_path = tmp_path / "snapshot.json"
    bundle_path = snapshot_path.parent / payload_bundle_directory(snapshot_path)
    bundle_path.mkdir(mode=0o700)
    bundle_path.chmod(0o700)
    payloads: list[PackagePayload] = []
    total_size_bytes = 0
    for package in packages:
        content = (
            f"retained-preflight-package:{package.name}:{package.architecture}:"
            f"{package.epoch or ''}:{package.version}"
        ).encode()
        digest = hashlib.sha256(content).hexdigest()
        extension = "deb" if package_manager == "apt-get" else "rpm"
        payload_path = bundle_path / f"{digest}.{extension}"
        payload_path.write_bytes(content)
        payload_path.chmod(0o600)
        total_size_bytes += len(content)
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
                verification=(
                    "apt-repository"
                    if package_manager == "apt-get"
                    else "rpm-signature"
                ),
                roles=("baseline",),
                signer_ids=() if package_manager == "apt-get" else ("deadbeef",),
            )
        )
        runner.payloads[str(payload_path)] = package
    bundle = PackagePayloadBundle(
        directory=bundle_path.name,
        packages=tuple(payloads),
        total_size_bytes=total_size_bytes,
    )
    return RollbackSnapshot(
        path=str(snapshot_path),
        packages=packages,
        kernel=kernel,
        module_version=None,
        commands=[],
        package_manager=package_manager,
        package_payloads=bundle,
    )


def _transaction_calls(runner: "_FakeRunner"):
    return [
        call
        for call in runner.calls
        if call[0][0] not in {"dpkg-deb", "rpm", "rpmkeys"}
    ]


class _FakeRunner:
    def __init__(self, results=None, *, default=None):
        self.results = list(results or [])
        self.default = default
        self.calls = []
        self.payloads: dict[str, PackageInfo] = {}

    def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
        del input_text
        self.calls.append((command, mutate, allow_fail))
        if command[0] in {"dpkg-deb", "rpm", "rpmkeys"}:
            package = self.payloads.get(command[-1])
            if package is None:
                raise AssertionError(f"unexpected payload inspection: {command}")
            if command[0] == "dpkg-deb":
                result = CommandResult(
                    [],
                    0,
                    stdout=(
                        f"{package.name}\t{package.version}\t"
                        f"{package.architecture}\n"
                    ),
                )
            elif command[0] == "rpm":
                result = CommandResult(
                    [],
                    0,
                    stdout=(
                        f"{package.name}\t{package.epoch or '(none)'}\t"
                        f"{package.version}\t{package.architecture}\n"
                    ),
                )
            else:
                result = CommandResult(
                    [],
                    0,
                    stdout="Header V4 RSA/SHA256 Signature, key ID deadbeef: OK\n",
                )
            return CommandResult(
                command,
                result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if self.results:
            result = self.results.pop(0)
        elif self.default is not None:
            result = self.default
        else:
            raise AssertionError(f"unexpected command: {command}")
        return CommandResult(
            command,
            result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            skipped=result.skipped,
            reason=result.reason,
        )
