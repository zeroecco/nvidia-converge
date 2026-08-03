from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nvidia_converge import package_payloads
from nvidia_converge.models import CommandResult, PackageInfo


def _package(
    name: str,
    version: str,
    manager: str,
    architecture: str,
    *,
    epoch: str | None = None,
) -> PackageInfo:
    return PackageInfo(
        name=name,
        version=version,
        manager=manager,
        installed=True,
        architecture=architecture,
        epoch=epoch,
    )


def test_apt_size_probe_is_no_download_and_parses_exact_records() -> None:
    packages = [
        _package("nvidia-driver", "1:580.1-1", "apt", "amd64"),
        _package("libnvidia-compute", "580.1-1", "apt", "amd64"),
    ]
    command = package_payloads._download_size_command(
        "apt-get",
        packages,
        Path("/private/payloads"),
    )

    assert "--print-uris" in command
    assert "--download-only" in command
    assert "--reinstall" in command
    output = (
        "'https://repo.invalid/driver.deb' driver.deb 1024 "
        "MD5Sum:00000000000000000000000000000000\n"
        "'https://repo.invalid/compute.deb' compute.deb 2048 "
        "11111111111111111111111111111111"
    )
    result = CommandResult(command, 0, stdout=output)

    assert package_payloads._parse_download_size(
        "apt-get",
        result,
        packages,
        4096,
    ) == 3072


@pytest.mark.parametrize(
    "output",
    [
        (
            "'https://repo.invalid/only.deb' only.deb 1024 "
            "MD5Sum:00000000000000000000000000000000"
        ),
        (
            "'https://repo.invalid/a.deb' ../a.deb 1024 "
            "MD5Sum:00000000000000000000000000000000\n"
            "'https://repo.invalid/b.deb' b.deb 2048 "
            "MD5Sum:11111111111111111111111111111111"
        ),
        "'https://repo.invalid/a.deb' a.deb 1024 SHA256:"
        + "0" * 64
        + "\n'https://repo.invalid/b.deb' b.deb 2048 MD5Sum:"
        + "1" * 32,
    ],
)
def test_apt_size_probe_rejects_non_exact_or_malformed_records(output: str) -> None:
    packages = [
        _package("nvidia-driver", "580.1-1", "apt", "amd64"),
        _package("libnvidia-compute", "580.1-1", "apt", "amd64"),
    ]

    with pytest.raises(package_payloads.PackagePayloadError):
        package_payloads._parse_download_size(
            "apt-get",
            CommandResult(["apt-get"], 0, stdout=output),
            packages,
            4096,
        )


def test_zypper_size_probe_requires_the_exact_resolved_identity_set() -> None:
    packages = [
        _package(
            "nvidia-open",
            "580.1-1",
            "rpm",
            "x86_64",
            epoch="1",
        )
    ]
    command = package_payloads._download_size_command(
        "zypper",
        packages,
        Path("/private/payloads"),
    )
    assert "--dry-run" in command
    assert "--pkg-cache-dir" in command
    assert command[command.index("install") + 1 :].count("--force") == 1
    valid = (
        "<stream><install-summary download-size='4096' packages-to-change='1'>"
        "<to-reinstall><solvable kind='package' name='nvidia-open' "
        "edition='1:580.1-1' arch='x86_64' status='installed'/>"
        "</to-reinstall></install-summary></stream>"
    )

    assert package_payloads._parse_download_size(
        "zypper",
        CommandResult(command, 0, stdout=valid),
        packages,
        8192,
    ) == 4096

    extra = valid.replace(
        "</to-reinstall>",
        "<solvable kind='package' name='unrequested' edition='1-1' "
        "arch='x86_64' status='not-installed'/></to-reinstall>",
    )
    with pytest.raises(
        package_payloads.PackagePayloadError,
        match="unrequested unrequested-1-1.x86_64",
    ):
        package_payloads._parse_download_size(
            "zypper",
            CommandResult(command, 0, stdout=extra),
            packages,
            8192,
        )


@pytest.mark.parametrize(
    "output",
    [
        (
            "<stream><install-summary download-size='4096' "
            "packages-to-change='0'/></stream>"
        ),
        (
            "<stream><install-summary download-size='0' packages-to-change='1'>"
            "<to-install><solvable kind='package' name='nvidia-open' "
            "edition='580.1-1' arch='x86_64'/></to-install>"
            "</install-summary></stream>"
        ),
        "<!DOCTYPE stream><stream/>",
    ],
)
def test_zypper_size_probe_rejects_missing_zero_or_unsafe_xml(output: str) -> None:
    package = _package("nvidia-open", "580.1-1", "rpm", "x86_64")
    with pytest.raises(package_payloads.PackagePayloadError):
        package_payloads._parse_download_size(
            "zypper",
            CommandResult(["zypper"], 0, stdout=output),
            [package],
            8192,
        )


def test_capacity_gate_includes_allocation_overhead_and_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment_size = 4096
    planned_size = 1024
    package_count = 2
    required = (
        planned_size
        + package_count * (fragment_size - 1)
        + package_payloads.PACKAGE_DOWNLOAD_FIXED_OVERHEAD_BYTES
        + package_payloads.MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES
    )

    monkeypatch.setattr(
        package_payloads.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=(required - 1) // fragment_size,
            f_frsize=fragment_size,
        ),
    )
    with pytest.raises(package_payloads.PackagePayloadError, match="headroom"):
        package_payloads._require_download_capacity(
            Path("/private/payloads"),
            planned_size,
            package_count,
        )

    monkeypatch.setattr(
        package_payloads.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=(required + fragment_size - 1) // fragment_size,
            f_frsize=fragment_size,
        ),
    )
    package_payloads._require_download_capacity(
        Path("/private/payloads"),
        planned_size,
        package_count,
    )
