from __future__ import annotations

import os
from pathlib import Path

import pytest

from nvidia_converge.audit import _audit_kernel
from nvidia_converge.kernel_headers import (
    KernelHeaderReadiness,
    assess_running_kernel_headers,
)
from nvidia_converge.models import CommandResult


@pytest.mark.parametrize(
    ("release", "target_parts", "manager", "owner_row", "marker_kind"),
    [
        (
            "6.8.0-1018-azure",
            ("linux-headers-6.8.0-1018-azure",),
            "apt-get",
            "linux-headers-6.8.0-1018-azure",
            "plain",
        ),
        (
            "5.14.0-503.el9.x86_64",
            ("kernels", "5.14.0-503.el9.x86_64"),
            "dnf",
            "kernel-devel\t5.14.0-503.el9\tx86_64",
            "uts",
        ),
        (
            "6.4.0-150600.21-default",
            (
                "linux-6.4.0-150600.21-obj",
                "x86_64",
                "default",
            ),
            "zypper",
            "kernel-default-devel\t6.4.0-150600.21.3\tx86_64",
            "plain",
        ),
    ],
)
def test_accepts_release_bound_ubuntu_rhel_and_suse_header_trees(
    tmp_path: Path,
    release: str,
    target_parts: tuple[str, ...],
    manager: str,
    owner_row: str,
    marker_kind: str,
) -> None:
    modules_root, source_root, target = _header_tree(
        tmp_path,
        release,
        target_parts,
        marker_release=release,
        marker_kind=marker_kind,
    )
    runner = _OwnerRunner(manager, owner_row)

    readiness = assess_running_kernel_headers(
        runner,  # type: ignore[arg-type]
        release=release,
        package_manager=manager,
        modules_root=modules_root,
        source_root=source_root,
        trusted_uid=os.getuid(),
    )

    assert readiness.ready is True
    assert readiness.build_path == str(target.resolve())
    assert readiness.package_owner == owner_row.split("\t", 1)[0]
    assert len(runner.calls) == 1


def test_rejects_a_dangling_running_kernel_build_link(tmp_path: Path) -> None:
    release = "6.8.0-1018-azure"
    modules_root = tmp_path / "lib/modules"
    release_directory = modules_root / release
    release_directory.mkdir(parents=True)
    source_root = tmp_path / "usr/src"
    source_root.mkdir(parents=True)
    (release_directory / "build").symlink_to(source_root / "missing")

    readiness = _assess_without_package_owner(release, modules_root, source_root)

    assert readiness.ready is False
    assert "missing or dangling" in readiness.detail


def test_rejects_an_unrelated_tree_even_with_a_forged_release_marker(
    tmp_path: Path,
) -> None:
    release = "6.8.0-1018-azure"
    modules_root, source_root, _ = _header_tree(
        tmp_path,
        release,
        ("unrelated-kernel-source",),
        marker_release=release,
    )

    readiness = _assess_without_package_owner(release, modules_root, source_root)

    assert readiness.ready is False
    assert "unrelated" in readiness.detail


def test_rejects_a_header_marker_for_a_different_kernel(tmp_path: Path) -> None:
    release = "5.14.0-503.el9.x86_64"
    modules_root, source_root, _ = _header_tree(
        tmp_path,
        release,
        ("kernels", release),
        marker_release="5.14.0-427.el9.x86_64",
    )

    readiness = _assess_without_package_owner(release, modules_root, source_root)

    assert readiness.ready is False
    assert "marker for the running release" in readiness.detail


def test_rejects_a_group_writable_header_tree(tmp_path: Path) -> None:
    release = "6.8.0-1018-azure"
    modules_root, source_root, target = _header_tree(
        tmp_path,
        release,
        (f"linux-headers-{release}",),
        marker_release=release,
    )
    target.chmod(0o775)

    readiness = _assess_without_package_owner(release, modules_root, source_root)

    assert readiness.ready is False
    assert "unsafe ownership or mode" in readiness.detail


def test_rejects_a_build_tree_outside_trusted_kernel_roots(tmp_path: Path) -> None:
    release = "6.8.0-1018-azure"
    modules_root = tmp_path / "lib/modules"
    release_directory = modules_root / release
    release_directory.mkdir(parents=True)
    source_root = tmp_path / "usr/src"
    source_root.mkdir(parents=True)
    outside = tmp_path / f"opt/linux-headers-{release}"
    _populate_header_tree(outside, release, "plain")
    (release_directory / "build").symlink_to(outside)

    readiness = _assess_without_package_owner(release, modules_root, source_root)

    assert readiness.ready is False
    assert "outside trusted kernel source roots" in readiness.detail


def test_rejects_a_release_marker_that_escapes_the_trusted_tree(
    tmp_path: Path,
) -> None:
    release = "6.8.0-1018-azure"
    modules_root, source_root, target = _header_tree(
        tmp_path,
        release,
        (f"linux-headers-{release}",),
        marker_release=release,
    )
    marker = target / "include/config/kernel.release"
    marker.unlink()
    outside_marker = tmp_path / "untrusted-kernel.release"
    outside_marker.write_text(f"{release}\n", encoding="utf-8")
    marker.symlink_to(outside_marker)

    readiness = _assess_without_package_owner(release, modules_root, source_root)

    assert readiness.ready is False
    assert "marker for the running release" in readiness.detail


def test_known_package_backend_must_prove_header_ownership(tmp_path: Path) -> None:
    release = "6.8.0-1018-azure"
    modules_root, source_root, _ = _header_tree(
        tmp_path,
        release,
        (f"linux-headers-{release}",),
        marker_release=release,
    )

    readiness = assess_running_kernel_headers(
        _OwnerRunner(None, ""),  # type: ignore[arg-type]
        release=release,
        package_manager="apt-get",
        modules_root=modules_root,
        source_root=source_root,
        trusted_uid=os.getuid(),
    )

    assert readiness.ready is False
    assert "matching kernel header package" in readiness.detail


@pytest.mark.parametrize(
    ("manager", "owner_row"),
    [
        ("apt-get", "linux-headers-6.8.0-1017-azure"),
        ("dnf", "kernel-devel\t5.14.0-427.el9\tx86_64"),
    ],
)
def test_rejects_a_marker_owned_by_unmatched_header_package(
    tmp_path: Path,
    manager: str,
    owner_row: str,
) -> None:
    release = "6.8.0-1018-azure" if manager == "apt-get" else "5.14.0-503.el9.x86_64"
    modules_root, source_root, _ = _header_tree(
        tmp_path,
        release,
        (f"linux-headers-{release}",),
        marker_release=release,
    )

    readiness = assess_running_kernel_headers(
        _OwnerRunner(manager, owner_row),  # type: ignore[arg-type]
        release=release,
        package_manager=manager,
        modules_root=modules_root,
        source_root=source_root,
        trusted_uid=os.getuid(),
    )

    assert readiness.ready is False
    assert "matching kernel header package" in readiness.detail


def test_audit_uses_the_shared_readiness_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def assess(*args: object, **kwargs: object) -> KernelHeaderReadiness:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return KernelHeaderReadiness(False, "sentinel failure")

    monkeypatch.setattr(
        "nvidia_converge.audit.assess_running_kernel_headers",
        assess,
    )
    kernel = _audit_kernel(_KernelRunner(), "apt-get")  # type: ignore[arg-type]

    assert kernel.headers_installed is False
    assert observed["kwargs"] == {
        "release": kernel.running,
        "package_manager": "apt-get",
    }


def _header_tree(
    tmp_path: Path,
    release: str,
    target_parts: tuple[str, ...],
    *,
    marker_release: str,
    marker_kind: str = "plain",
) -> tuple[Path, Path, Path]:
    modules_root = tmp_path / "lib/modules"
    release_directory = modules_root / release
    release_directory.mkdir(parents=True)
    source_root = tmp_path / "usr/src"
    target = source_root.joinpath(*target_parts)
    _populate_header_tree(target, marker_release, marker_kind)
    (release_directory / "build").symlink_to(target)
    return modules_root, source_root, target


def _populate_header_tree(target: Path, release: str, marker_kind: str) -> None:
    (target / "include/config").mkdir(parents=True)
    (target / "include/generated").mkdir(parents=True)
    (target / "Makefile").write_text("obj-m += probe.o\n", encoding="utf-8")
    (target / "include/generated/autoconf.h").write_text(
        "#define CONFIG_MODULES 1\n",
        encoding="utf-8",
    )
    if marker_kind == "uts":
        (target / "include/generated/utsrelease.h").write_text(
            f'#define UTS_RELEASE "{release}"\n',
            encoding="utf-8",
        )
    else:
        (target / "include/config/kernel.release").write_text(
            f"{release}\n",
            encoding="utf-8",
        )


def _assess_without_package_owner(
    release: str,
    modules_root: Path,
    source_root: Path,
) -> KernelHeaderReadiness:
    return assess_running_kernel_headers(
        _OwnerRunner(None, ""),  # type: ignore[arg-type]
        release=release,
        package_manager=None,
        modules_root=modules_root,
        source_root=source_root,
        trusted_uid=os.getuid(),
    )


class _OwnerRunner:
    def __init__(self, manager: str | None, owner_row: str) -> None:
        self.manager = manager
        self.owner_row = owner_row
        self.calls: list[list[str]] = []

    def exists(self, name: str) -> bool:
        return (self.manager == "apt-get" and name == "dpkg-query") or (
            self.manager in {"dnf", "yum", "zypper"} and name == "rpm"
        )

    def run(self, command: list[str], *, allow_fail: bool = True) -> CommandResult:
        del allow_fail
        self.calls.append(command)
        if command[0] == "dpkg-query":
            return CommandResult(
                command,
                0,
                stdout=f"{self.owner_row}: {command[-1]}\n",
            )
        if command[0] == "rpm":
            return CommandResult(command, 0, stdout=f"{self.owner_row}\n")
        return CommandResult(command, 127, stderr="unexpected command")


class _KernelRunner:
    def resolve_executable(self, name: str) -> str | None:
        del name
        return None

    def exists(self, name: str) -> bool:
        del name
        return False
