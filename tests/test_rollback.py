from nvidia_converge.models import CommandResult, PackageInfo, RollbackSnapshot
from nvidia_converge.rollback import RollbackSnapshotError, _rollback_commands, apply_rollback, load_snapshot

import json

import pytest


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
    assert commands == [["apt-get", "install", "-y", "nvidia-driver-580-open=580.126.16-1", "cuda-compat-13-0=13.0.0-1"]]


def test_rpm_rollback_only_restores_relevant_packages():
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-open-595", "595.71.05-1", "rpm", True),
            PackageInfo("nvidia-container-toolkit", "1.19.0-1", "rpm", True),
            PackageInfo("bash", "5.2-1", "rpm", True),
        ],
        "dnf",
    )
    assert commands == [["dnf", "downgrade", "-y", "nvidia-open-595-595.71.05-1", "nvidia-container-toolkit-1.19.0-1"]]


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
            "install",
            "--oldpackage",
            "nvidia-open-595=595.71.05-1",
            "nvidia-container-toolkit=1.19.0-1",
        ]
    ]


def test_load_snapshot_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="missing required"):
        load_snapshot(str(path))


def test_load_snapshot_accepts_valid_snapshot(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "path": "/var/lib/nvidia-converge/snapshots/example.json",
                "packages": [
                    {"name": "nvidia-open-595", "version": "595.71.05-1", "manager": "rpm", "installed": True}
                ],
                "kernel": "6.8.0-111-generic",
                "module_version": "595.71.05",
                "commands": [["zypper", "--non-interactive", "install", "--oldpackage", "nvidia-open-595=595.71.05-1"]],
            }
        ),
        encoding="utf-8",
    )
    snapshot = load_snapshot(str(path))
    assert snapshot.kernel == "6.8.0-111-generic"
    assert snapshot.packages[0].name == "nvidia-open-595"
    assert snapshot.commands[0][0] == "zypper"


def test_load_snapshot_tolerates_empty_package_version_from_older_snapshots(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "path": "/var/lib/nvidia-converge/snapshots/example.json",
                "packages": [
                    {"name": "cuda-cub", "version": "", "manager": "apt", "installed": True},
                    {"name": "libnvidia-ml.so.1", "version": "", "manager": "", "installed": True},
                ],
                "kernel": "6.8.0-111-generic",
                "module_version": "595.71.05",
                "commands": [["apt-get", "install", "-y", "nvidia-driver-595=595.71.05-1"]],
            }
        ),
        encoding="utf-8",
    )
    snapshot = load_snapshot(str(path))
    assert snapshot.packages[0].version is None
    assert snapshot.packages[1].manager is None


def test_load_snapshot_rejects_invalid_package_entry(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"packages": [{"name": "nvidia-open-595", "installed": "yes"}], "kernel": "6.8.0", "commands": [["true"]]}),
        encoding="utf-8",
    )
    with pytest.raises(RollbackSnapshotError, match=r"packages\[0\].installed"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_invalid_command_entry(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"packages": [], "kernel": "6.8.0", "commands": [["zypper", ""]]}),
        encoding="utf-8",
    )
    with pytest.raises(RollbackSnapshotError, match=r"commands\[0\] entries"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_unsupported_command(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"packages": [], "kernel": "6.8.0", "commands": [["sh", "-c", "id"]]}),
        encoding="utf-8",
    )
    with pytest.raises(RollbackSnapshotError, match="not a supported rollback command"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_rollback_option_operand(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"packages": [], "kernel": "6.8.0", "commands": [["apt-get", "install", "-y", "-o", "Dpkg::Options::=--force-confold"]]}),
        encoding="utf-8",
    )
    with pytest.raises(RollbackSnapshotError, match="not a supported rollback command"):
        load_snapshot(str(path))


def test_load_snapshot_rejects_empty_rollback_command_specs(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"packages": [], "kernel": "6.8.0", "commands": [["dnf", "downgrade", "-y"]]}),
        encoding="utf-8",
    )
    with pytest.raises(RollbackSnapshotError, match="not a supported rollback command"):
        load_snapshot(str(path))


def test_apply_rollback_stops_after_first_failed_command():
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel="6.8.0-test",
        module_version=None,
        commands=[
            ["apt-get", "install", "-y", "nvidia-driver-580=580.1"],
            ["apt-get", "install", "-y", "cuda-compat-13-0=13.0"],
        ],
    )
    runner = _FakeRunner([100, 0])
    results = apply_rollback(snapshot, runner)
    assert [result.command for result in results] == [["apt-get", "install", "-y", "nvidia-driver-580=580.1"]]


class _FakeRunner:
    def __init__(self, returncodes):
        self.returncodes = list(returncodes)

    def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
        del mutate, allow_fail, input_text
        return CommandResult(command, self.returncodes.pop(0))
