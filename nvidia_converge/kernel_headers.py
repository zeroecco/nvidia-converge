from __future__ import annotations

import os
import platform
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from .runner import CommandRunner

_DEFAULT_MODULES_ROOT = Path("/lib/modules")
_DEFAULT_SOURCE_ROOT = Path("/usr/src")
_MAX_HEADER_MARKER_BYTES = 16 * 1024
_SAFE_KERNEL_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,254}$")
_UTS_RELEASE = re.compile(
    r'^\s*#\s*define\s+UTS_RELEASE\s+"([^"\\\r\n]+)"\s*$',
)
_RPM_DEVEL_PACKAGE = re.compile(r"^kernel(?:-[A-Za-z0-9_+.-]+)?-devel$")


@dataclass(frozen=True)
class KernelHeaderReadiness:
    ready: bool
    detail: str
    build_path: str | None = None
    binding_marker: str | None = None
    package_owner: str | None = None


def assess_running_kernel_headers(
    runner: CommandRunner,
    *,
    release: str | None = None,
    package_manager: str | None = None,
    modules_root: Path = _DEFAULT_MODULES_ROOT,
    source_root: Path = _DEFAULT_SOURCE_ROOT,
    trusted_uid: int = 0,
) -> KernelHeaderReadiness:
    """Prove that the running kernel has a trusted, prepared header tree."""

    running = release if release is not None else platform.uname().release
    if not _SAFE_KERNEL_RELEASE.fullmatch(running):
        return _not_ready("Running kernel release is not a safe path component.")

    try:
        canonical_modules = modules_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return _not_ready(
            "The kernel module root is missing or cannot be canonicalized."
        )
    if not _trusted_directory(canonical_modules, trusted_uid):
        return _not_ready("The kernel module root is not a trusted directory.")

    release_directory = modules_root / running
    try:
        release_metadata = release_directory.lstat()
        canonical_release_directory = release_directory.resolve(strict=True)
    except (OSError, RuntimeError):
        return _not_ready("The running kernel module directory is missing or unsafe.")
    if stat.S_ISLNK(release_metadata.st_mode):
        return _not_ready("The running kernel module directory must not be a symlink.")
    if canonical_release_directory.parent != canonical_modules:
        return _not_ready(
            "The running kernel module directory escaped its trusted root."
        )
    if not _trusted_directory(canonical_release_directory, trusted_uid):
        return _not_ready("The running kernel module directory is not trusted.")

    build_entry = release_directory / "build"
    try:
        build_entry_metadata = build_entry.lstat()
        canonical_build = build_entry.resolve(strict=True)
    except (OSError, RuntimeError):
        return _not_ready("The running kernel build entry is missing or dangling.")
    if not _trusted_link_or_directory(build_entry_metadata, trusted_uid):
        return _not_ready("The running kernel build entry is not trusted.")
    if not canonical_build.is_dir():
        return _not_ready(
            "The running kernel build entry does not resolve to a directory."
        )

    canonical_source = _canonical_directory_or_none(source_root)
    allowed_roots = [canonical_release_directory]
    if canonical_source is not None:
        allowed_roots.append(canonical_source)
    build_root = _containing_root(canonical_build, allowed_roots)
    if build_root is None:
        return _not_ready(
            "The running kernel build tree resolves outside trusted kernel source roots."
        )
    if build_root == canonical_source and not _path_mentions_release(
        canonical_build, running
    ):
        return _not_ready(
            "The running kernel build tree is unrelated to the running kernel release."
        )
    if not _trusted_directory_chain(build_root, canonical_build, trusted_uid):
        return _not_ready("The running kernel build tree has unsafe ownership or mode.")

    makefile = _trusted_regular_file(
        canonical_build / "Makefile",
        allowed_roots,
        trusted_uid,
    )
    if makefile is None:
        return _not_ready("The kernel header tree has no trusted top-level Makefile.")

    binding_marker = _kernel_release_marker(
        canonical_build,
        running,
        allowed_roots,
        trusted_uid,
    )
    if binding_marker is None:
        return _not_ready(
            "The kernel header tree does not contain a trusted marker for the running release."
        )

    prepared_markers = (
        canonical_build / "include/generated/autoconf.h",
        canonical_build / "include/config/auto.conf",
    )
    if not any(
        _trusted_regular_file(marker, allowed_roots, trusted_uid) is not None
        for marker in prepared_markers
    ):
        return _not_ready(
            "The kernel header tree is not prepared for an external module build."
        )

    owner = _header_package_owner(
        runner,
        package_manager,
        running,
        binding_marker,
    )
    if package_manager in {"apt-get", "dnf", "yum", "zypper"} and owner is None:
        return _not_ready(
            "The running-release header marker is not owned by the matching kernel header package."
        )

    return KernelHeaderReadiness(
        True,
        "Running kernel headers are trusted, release-matched, and prepared.",
        build_path=str(canonical_build),
        binding_marker=str(binding_marker),
        package_owner=owner,
    )


def _not_ready(detail: str) -> KernelHeaderReadiness:
    return KernelHeaderReadiness(False, detail)


def _canonical_directory_or_none(path: Path) -> Path | None:
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return canonical if canonical.is_dir() else None


def _trusted_link_or_directory(metadata: os.stat_result, trusted_uid: int) -> bool:
    if metadata.st_uid != trusted_uid:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    return stat.S_ISDIR(metadata.st_mode) and not _writable_by_group_or_other(
        metadata.st_mode
    )


def _trusted_directory(path: Path, trusted_uid: int) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == trusted_uid
        and not _writable_by_group_or_other(metadata.st_mode)
    )


def _trusted_directory_chain(root: Path, path: Path, trusted_uid: int) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    if not _trusted_directory(current, trusted_uid):
        return False
    for component in relative.parts:
        current /= component
        if not _trusted_directory(current, trusted_uid):
            return False
    return True


def _writable_by_group_or_other(mode: int) -> bool:
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH))


def _containing_root(path: Path, roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        candidates.append(root)
    return max(candidates, key=lambda candidate: len(candidate.parts), default=None)


def _path_mentions_release(path: Path, release: str) -> bool:
    if any(release in component for component in path.parts):
        return True
    base, separator, flavor = release.rpartition("-")
    return bool(
        separator
        and base
        and flavor
        and any(base in component for component in path.parts)
        and flavor in path.parts
    )


def _trusted_regular_file(
    path: Path,
    allowed_roots: list[Path],
    trusted_uid: int,
) -> Path | None:
    lexical_root = _containing_root(path, allowed_roots)
    if lexical_root is None or not _trusted_lexical_chain(
        lexical_root, path, trusted_uid
    ):
        return None
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    root = _containing_root(canonical, allowed_roots)
    if root is None or not _trusted_directory_chain(
        root, canonical.parent, trusted_uid
    ):
        return None
    try:
        metadata = canonical.lstat()
    except OSError:
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != trusted_uid
        or _writable_by_group_or_other(metadata.st_mode)
    ):
        return None
    return canonical


def _trusted_lexical_chain(root: Path, path: Path, trusted_uid: int) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            metadata = current.lstat()
        except OSError:
            return False
        final = index == len(relative.parts) - 1
        if stat.S_ISLNK(metadata.st_mode):
            if metadata.st_uid != trusted_uid:
                return False
        elif final:
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != trusted_uid
                or _writable_by_group_or_other(metadata.st_mode)
            ):
                return False
        elif (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != trusted_uid
            or _writable_by_group_or_other(metadata.st_mode)
        ):
            return False
    return True


def _kernel_release_marker(
    build: Path,
    release: str,
    allowed_roots: list[Path],
    trusted_uid: int,
) -> Path | None:
    candidates = (
        (build / "include/config/kernel.release", "plain"),
        (build / "include/generated/utsrelease.h", "uts"),
        (build / "include/linux/utsrelease.h", "uts"),
    )
    for candidate, kind in candidates:
        canonical = _trusted_regular_file(candidate, allowed_roots, trusted_uid)
        if canonical is None:
            continue
        content = _read_small_regular_file(canonical, trusted_uid)
        if content is None:
            continue
        if kind == "plain" and content.strip() == release:
            return canonical
        match = _UTS_RELEASE.fullmatch(content)
        if kind == "uts" and match is not None and match.group(1) == release:
            return canonical
    return None


def _read_small_regular_file(path: Path, trusted_uid: int) -> str | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    chunks: list[bytes] = []
    total = 0
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != trusted_uid
            or _writable_by_group_or_other(metadata.st_mode)
        ):
            return None
        while total <= _MAX_HEADER_MARKER_BYTES:
            chunk = os.read(
                descriptor,
                min(4096, _MAX_HEADER_MARKER_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_HEADER_MARKER_BYTES:
            return None
    finally:
        os.close(descriptor)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _header_package_owner(
    runner: CommandRunner,
    package_manager: str | None,
    release: str,
    marker: Path,
) -> str | None:
    if package_manager == "apt-get":
        return _dpkg_header_owner(runner, release, marker)
    if package_manager in {"dnf", "yum", "zypper"}:
        return _rpm_header_owner(runner, release, marker)
    return None


def _dpkg_header_owner(
    runner: CommandRunner,
    release: str,
    marker: Path,
) -> str | None:
    if not runner.exists("dpkg-query"):
        return None
    result = runner.run(["dpkg-query", "-S", str(marker)], allow_fail=True)
    if result.returncode != 0:
        return None
    owners: set[str] = set()
    for line in result.stdout.splitlines():
        package_field, separator, owned_path = line.partition(": ")
        if not separator or owned_path != str(marker):
            return None
        package = package_field.split(":", 1)[0]
        if package != f"linux-headers-{release}":
            return None
        owners.add(package)
    if len(owners) != 1:
        return None
    return owners.pop()


def _rpm_header_owner(
    runner: CommandRunner,
    release: str,
    marker: Path,
) -> str | None:
    if not runner.exists("rpm"):
        return None
    result = runner.run(
        [
            "rpm",
            "-qf",
            "--qf",
            "%{NAME}\\t%{VERSION}-%{RELEASE}\\t%{ARCH}\\n",
            str(marker),
        ],
        allow_fail=True,
    )
    if result.returncode != 0:
        return None
    rows = [line.split("\t") for line in result.stdout.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 3:
        return None
    name, version_release, architecture = rows[0]
    if not _RPM_DEVEL_PACKAGE.fullmatch(name):
        return None
    if release == f"{version_release}.{architecture}":
        return name
    match = re.fullmatch(r"kernel-([A-Za-z0-9_+.-]+)-devel", name)
    if match is not None:
        flavor = match.group(1)
        kernel_version, separator, running_flavor = release.rpartition("-")
        if (
            separator
            and running_flavor == flavor
            and (
                version_release == kernel_version
                or version_release.startswith(f"{kernel_version}.")
            )
        ):
            return name
    return None
