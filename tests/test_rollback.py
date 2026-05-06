from nvidia_converge.models import PackageInfo
from nvidia_converge.rollback import RollbackSnapshotError, _rollback_commands, load_snapshot

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


def test_load_snapshot_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RollbackSnapshotError, match="missing required"):
        load_snapshot(str(path))
