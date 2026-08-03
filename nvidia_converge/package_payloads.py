from __future__ import annotations

import errno
import hashlib
import os
import re
import shlex
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .files import BoundedFileError, ensure_private_directory, open_private_directory
from .models import CommandResult, PackageInfo, PackagePayload, PackagePayloadBundle
from .xmlsafe import SafeXmlError, parse_bounded_xml

MAX_PACKAGE_PAYLOAD_BYTES = 8 * 1024 * 1024 * 1024
MAX_PACKAGE_PAYLOAD_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_BASELINE_PACKAGE_PAYLOADS = 4096
MAX_PACKAGE_PAYLOAD_ENTRIES = 8192
MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES = 64 * 1024 * 1024
PACKAGE_DOWNLOAD_FIXED_OVERHEAD_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_PAYLOAD_CLEANUP_ENTRIES = MAX_PACKAGE_PAYLOAD_ENTRIES * 4
MAX_PACKAGE_PAYLOAD_CLEANUP_BYTES = (
    MAX_PACKAGE_PAYLOAD_TOTAL_BYTES + MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES + 8 * 1024 * 1024
)
MAX_PACKAGE_PAYLOAD_CLEANUP_DEPTH = 64
_READ_CHUNK_BYTES = 1024 * 1024
_SPACE_RESERVE_NAME = ".free-space-reserve"
_DIGEST = re.compile(r"[a-f0-9]{64}")
_SIGNER_ID = re.compile(r"key ID ([0-9A-Fa-f]{8,40})", re.IGNORECASE)
_DNF_DOWNLOAD_SCRIPT = r"""
import dnf
import os
import sys

destination, per_file_limit, total_limit, free_reserve, *fields = sys.argv[1:]
if len(fields) % 4:
    raise SystemExit("invalid exact RPM identity arguments")
base = dnf.Base()
try:
    base.read_all_repos()
    for repository in base.repos.iter_enabled():
        repository.pkgdir = destination
    base.fill_sack_from_repos_in_cache(load_system_repo=False)
    selected = []
    for index in range(0, len(fields), 4):
        name, architecture, epoch, version = fields[index:index + 4]
        matches = []
        for package in base.sack.query().available().filter(
            name=name, arch=architecture
        ):
            observed_epoch = str(package.epoch or "")
            if observed_epoch in {"0", "None"}:
                observed_epoch = ""
            observed_version = str(package.version)
            release = str(package.release or "")
            if release:
                observed_version += "-" + release
            if observed_epoch == epoch and observed_version == version:
                matches.append(package)
        if not matches:
            raise RuntimeError("exact RPM identity is absent from cached metadata")
        checksums = {package.chksum for package in matches}
        if None in checksums or len(checksums) != 1:
            raise RuntimeError("exact RPM identity has ambiguous repository payloads")
        selected.append(
            sorted(matches, key=lambda item: (item.reponame, item.location))[0]
        )
    sizes = [int(package.downloadsize) for package in selected]
    if any(size <= 0 or size > int(per_file_limit) for size in sizes):
        raise RuntimeError("exact RPM payload exceeds the per-file size limit")
    if sum(sizes) > int(total_limit):
        raise RuntimeError("exact RPM payload set exceeds the total size limit")
    filesystem = os.statvfs(destination)
    available = filesystem.f_bavail * filesystem.f_frsize
    if available - sum(sizes) < int(free_reserve):
        raise RuntimeError("insufficient free space for exact RPM payload set")
    base.download_packages(selected)
    for package in selected:
        print(package.nevra)
finally:
    base.close()
""".strip()


class PackagePayloadError(ValueError):
    pass


class PayloadRunner(Protocol):
    def run(
        self,
        command: list[str],
        *,
        mutate: bool = False,
        allow_fail: bool = True,
        input_text: str | None = None,
    ) -> CommandResult: ...


@dataclass(frozen=True)
class _Candidate:
    relative_parent: tuple[str, ...]
    name: str
    size_bytes: int


@dataclass
class _CleanupBudget:
    entries: int = 0
    total_bytes: int = 0

    def consume(self, metadata: os.stat_result) -> None:
        self.entries += 1
        if self.entries > MAX_PACKAGE_PAYLOAD_CLEANUP_ENTRIES:
            raise PackagePayloadError(
                "interrupted rollback artifact tree exceeds the cleanup entry limit"
            )
        if stat.S_ISREG(metadata.st_mode):
            self.total_bytes += metadata.st_size
            if self.total_bytes > MAX_PACKAGE_PAYLOAD_CLEANUP_BYTES:
                raise PackagePayloadError(
                    "interrupted rollback artifact tree exceeds the cleanup byte limit"
                )


PackageIdentity = tuple[str, str, str | None, str]


def package_identity(package: PackageInfo | PackagePayload) -> PackageIdentity:
    return (
        package.name,
        package.architecture or "",
        package.epoch,
        package.version or "",
    )


def payload_bundle_directory(snapshot_path: Path) -> str:
    normalized = _normalized_absolute(snapshot_path)
    name = f"{normalized.name}.payloads"
    if not _safe_component(name):
        raise PackagePayloadError("rollback snapshot name cannot bind a payload bundle")
    return name


def cleanup_snapshot_payload_artifacts(
    snapshot_path: Path,
    *,
    preserve_bound_authority: bool,
    required_owner_uid: int,
) -> None:
    """Remove only deterministic artifacts belonging to one interrupted snapshot."""

    snapshot_path = _normalized_absolute(snapshot_path)
    final_bundle = payload_bundle_directory(snapshot_path)
    bundle_temps = (f".{final_bundle}.tmp", f".{final_bundle}.incoming")
    snapshot_temp = f".{snapshot_path.name}.tmp"
    cleanup_budget = _CleanupBudget()
    try:
        parent_fd = open_private_directory(
            snapshot_path.parent,
            required_owner_uid=required_owner_uid,
        )
    except FileNotFoundError:
        if preserve_bound_authority:
            raise PackagePayloadError(
                "bound rollback snapshot directory disappeared before recovery"
            )
        return
    except (OSError, BoundedFileError) as exc:
        raise PackagePayloadError(
            f"cannot bind interrupted rollback snapshot directory: {exc}"
        ) from exc
    snapshot_fd = -1
    bundle_fd = -1
    try:
        if preserve_bound_authority:
            snapshot_fd = _open_exact_regular_at(
                parent_fd,
                snapshot_path.name,
                required_owner_uid,
            )
            bundle_fd = _open_exact_directory_at(
                parent_fd,
                final_bundle,
                required_owner_uid,
            )
        for name in bundle_temps:
            _remove_exact_directory_at(
                parent_fd,
                name,
                required_owner_uid,
                cleanup_budget,
            )
        _remove_exact_regular_at(
            parent_fd,
            snapshot_temp,
            required_owner_uid,
            cleanup_budget,
        )
        if not preserve_bound_authority:
            _remove_exact_directory_at(
                parent_fd,
                final_bundle,
                required_owner_uid,
                cleanup_budget,
            )
            _remove_exact_regular_at(
                parent_fd,
                snapshot_path.name,
                required_owner_uid,
                cleanup_budget,
            )
        else:
            _revalidate_exact_entry(
                parent_fd,
                snapshot_path.name,
                snapshot_fd,
                required_owner_uid,
                require_directory=False,
            )
            _revalidate_exact_entry(
                parent_fd,
                final_bundle,
                bundle_fd,
                required_owner_uid,
                require_directory=True,
            )
        os.fsync(parent_fd)
        _verify_parent_binding(snapshot_path.parent, parent_fd, required_owner_uid)
    except (OSError, BoundedFileError) as exc:
        raise PackagePayloadError(
            f"cannot clean interrupted rollback snapshot artifacts: {exc}"
        ) from exc
    finally:
        if snapshot_fd >= 0:
            os.close(snapshot_fd)
        if bundle_fd >= 0:
            os.close(bundle_fd)
        os.close(parent_fd)


def local_payload_paths(
    snapshot_path: Path,
    bundle: PackagePayloadBundle,
    *,
    role: str,
) -> list[str]:
    if role not in {"baseline", "forward"}:
        raise PackagePayloadError(f"unsupported package payload role {role!r}")
    root = _normalized_absolute(snapshot_path).parent / bundle.directory
    return [
        str(root / payload.filename)
        for payload in bundle.packages
        if role in payload.roles
    ]


def forward_package_command(
    snapshot_path: Path,
    bundle: PackagePayloadBundle,
    package_manager: str,
    *,
    remove_specs: list[str] | None = None,
) -> list[str]:
    payloads = local_payload_paths(snapshot_path, bundle, role="forward")
    removals = sorted(set(remove_specs or []))
    if not payloads:
        raise PackagePayloadError(
            "package payload bundle contains no resolved forward transaction"
        )
    if package_manager == "apt-get":
        return [
            "apt-get",
            "install",
            "-y",
            "--allow-change-held-packages",
            "--allow-downgrades",
            "--no-download",
            "--no-install-recommends",
            "--purge",
            *payloads,
            *(f"{package}-" for package in removals),
        ]
    if package_manager in {"dnf", "yum"}:
        return [
            package_manager,
            "--disablerepo=*",
            "--setopt=localpkg_gpgcheck=1",
            "--setopt=install_weak_deps=False",
            "install",
            "-y",
            *payloads,
        ]
    if package_manager == "zypper":
        return [
            "zypper",
            "--non-interactive",
            "--disable-repositories",
            "--no-refresh",
            "install",
            "--no-recommends",
            "--",
            *payloads,
            *(f"-{package}" for package in removals),
        ]
    raise PackagePayloadError(
        f"local forward package transaction is unsupported for {package_manager}"
    )


def stage_package_payloads(
    snapshot_path: Path,
    packages: list[PackageInfo],
    package_manager: str,
    runner: PayloadRunner,
    *,
    forward_packages: list[PackageInfo] | None = None,
    required_owner_uid: int | None = None,
) -> PackagePayloadBundle:
    """Download, authenticate, and durably publish one private snapshot bundle."""

    snapshot_path = _normalized_absolute(snapshot_path)
    owner_uid = os.geteuid() if required_owner_uid is None else required_owner_uid
    _validate_requested_packages(packages, package_manager)
    forward_packages = list(forward_packages or [])
    _validate_requested_packages(forward_packages, package_manager)
    if len(packages) > MAX_BASELINE_PACKAGE_PAYLOADS:
        raise PackagePayloadError("rollback baseline package count exceeds the limit")
    requested: dict[PackageIdentity, tuple[PackageInfo, set[str]]] = {}
    for role, role_packages in (
        ("baseline", packages),
        ("forward", forward_packages),
    ):
        for package in role_packages:
            identity = package_identity(package)
            if identity not in requested:
                requested[identity] = (package, set())
            requested[identity][1].add(role)
    if len(requested) > MAX_PACKAGE_PAYLOAD_ENTRIES:
        raise PackagePayloadError("package payload bundle entry count exceeds the limit")
    ensure_private_directory(snapshot_path.parent, required_owner_uid=owner_uid)
    _require_free_space(snapshot_path.parent)
    parent_fd = open_private_directory(
        snapshot_path.parent,
        required_owner_uid=owner_uid,
    )
    final_name = payload_bundle_directory(snapshot_path)
    bundle_temp = f".{final_name}.tmp"
    incoming_temp = f".{final_name}.incoming"
    bundle_fd = -1
    incoming_fd = -1
    reserve_fd = -1
    published = False
    completed = False
    try:
        if _stat_at(parent_fd, final_name) is not None:
            raise PackagePayloadError(
                f"rollback payload bundle already exists: {snapshot_path.parent / final_name}"
            )
        bundle_fd = _create_private_child_directory(parent_fd, bundle_temp, owner_uid)
        incoming_fd = _create_private_child_directory(
            parent_fd, incoming_temp, owner_uid
        )
        reserve_fd = _create_space_reserve(incoming_fd, owner_uid)
        entries: list[PackagePayload] = []
        total_size = 0
        staged_size = 0
        observed_candidates: dict[
            PackageIdentity, tuple[_Candidate, list[str]]
        ] = {}
        baseline_identities = {
            identity for identity, (_package, roles) in requested.items()
            if "baseline" in roles
        }
        batches = (
            (
                "baseline",
                [
                    package
                    for identity, (package, _roles) in requested.items()
                    if identity in baseline_identities
                ],
            ),
            (
                "forward",
                [
                    package
                    for identity, (package, roles) in requested.items()
                    if "forward" in roles and identity not in baseline_identities
                ],
            ),
        )
        for download_name, batch_packages in batches:
            if not batch_packages:
                continue
            download_fd = _create_private_child_directory(
                incoming_fd, download_name, owner_uid
            )
            os.close(download_fd)
            destination = snapshot_path.parent / incoming_temp / download_name
            if package_manager == "apt-get":
                download_fd = _open_relative_directory(
                    incoming_fd, (download_name,), owner_uid
                )
                partial_fd = _create_private_child_directory(
                    download_fd,
                    "partial",
                    owner_uid,
                )
                os.close(partial_fd)
                os.close(download_fd)
            remaining_budget = MAX_PACKAGE_PAYLOAD_TOTAL_BYTES - staged_size
            planned_size: int | None = None
            if package_manager in {"apt-get", "zypper"}:
                size_result = runner.run(
                    _download_size_command(
                        package_manager,
                        batch_packages,
                        destination,
                    ),
                    mutate=False,
                    allow_fail=True,
                )
                planned_size = _parse_download_size(
                    package_manager,
                    size_result,
                    batch_packages,
                    remaining_budget,
                )
                _require_download_capacity(
                    snapshot_path.parent,
                    planned_size,
                    len(batch_packages),
                )
            result = runner.run(
                _download_command(
                    package_manager,
                    batch_packages,
                    destination,
                    total_limit=remaining_budget,
                ),
                mutate=False,
                allow_fail=True,
            )
            if result.returncode != 0:
                raise PackagePayloadError(
                    "could not retain the exact package payload set: "
                    f"{_command_diagnostic(result)}"
                )
            download_fd = _open_relative_directory(
                incoming_fd, (download_name,), owner_uid
            )
            try:
                extension = ".deb" if package_manager == "apt-get" else ".rpm"
                candidates, batch_size = _find_candidates(
                    download_fd,
                    extension=extension,
                    required_owner_uid=owner_uid,
                )
                staged_size += batch_size
                if staged_size > MAX_PACKAGE_PAYLOAD_TOTAL_BYTES:
                    raise PackagePayloadError(
                        "rollback package payload staging exceeds the total size limit"
                    )
                candidate_size = sum(candidate.size_bytes for candidate in candidates)
                if planned_size is not None and candidate_size > planned_size:
                    raise PackagePayloadError(
                        "package manager downloaded more payload bytes than its "
                        "authenticated pre-download plan"
                    )
                batch_candidates: dict[
                    PackageIdentity, tuple[_Candidate, list[str]]
                ] = {}
                for candidate in candidates:
                    candidate_path = destination.joinpath(
                        *candidate.relative_parent, candidate.name
                    )
                    observed, signer_ids = _inspect_payload(
                        candidate_path,
                        package_manager,
                        runner,
                    )
                    if observed in batch_candidates or observed in observed_candidates:
                        raise PackagePayloadError(
                            "package manager produced duplicate payloads for "
                            f"{_format_identity(observed)}"
                        )
                    batch_candidates[observed] = (
                        _Candidate(
                            (download_name, *candidate.relative_parent),
                            candidate.name,
                            candidate.size_bytes,
                        ),
                        signer_ids,
                    )
                expected_batch = {
                    package_identity(package) for package in batch_packages
                }
                if set(batch_candidates) != expected_batch:
                    missing = sorted(expected_batch - set(batch_candidates))
                    extra = sorted(set(batch_candidates) - expected_batch)
                    detail = []
                    if missing:
                        detail.append(
                            "missing "
                            + ", ".join(
                                _format_identity(identity) for identity in missing
                            )
                        )
                    if extra:
                        detail.append(
                            "unrequested "
                            + ", ".join(
                                _format_identity(identity) for identity in extra
                            )
                        )
                    raise PackagePayloadError(
                        "package manager payload output is not one-to-one with the exact request: "
                        + "; ".join(detail)
                    )
                observed_candidates.update(batch_candidates)
            finally:
                os.close(download_fd)
        if set(observed_candidates) != set(requested):
            raise PackagePayloadError(
                "package payload batches do not cover the exact retained identity set"
            )
        try:
            extension = ".deb" if package_manager == "apt-get" else ".rpm"
            for identity, (package, roles) in requested.items():
                candidate, signer_ids = observed_candidates[identity]
                source_parent_fd = _open_relative_directory(
                    incoming_fd,
                    candidate.relative_parent,
                    owner_uid,
                )
                try:
                    digest, size = _hash_bound_file(
                        source_parent_fd,
                        candidate.name,
                        required_owner_uid=owner_uid,
                        require_private_mode=False,
                    )
                    if total_size + size > MAX_PACKAGE_PAYLOAD_TOTAL_BYTES:
                        raise PackagePayloadError(
                            "rollback package payload bundle exceeds the total size limit"
                        )
                    filename = f"{digest}{extension}"
                    if _stat_at(bundle_fd, filename) is not None:
                        raise PackagePayloadError(
                            "rollback package payloads contain duplicate content"
                        )
                    source_fd = os.open(
                        candidate.name,
                        _file_read_flags(),
                        dir_fd=source_parent_fd,
                    )
                    try:
                        os.fchmod(source_fd, 0o600)
                        os.fsync(source_fd)
                    finally:
                        os.close(source_fd)
                    os.rename(
                        candidate.name,
                        filename,
                        src_dir_fd=source_parent_fd,
                        dst_dir_fd=bundle_fd,
                    )
                    os.fsync(source_parent_fd)
                    os.fsync(bundle_fd)
                finally:
                    os.close(source_parent_fd)
                total_size += size
                entries.append(
                    PackagePayload(
                        name=package.name,
                        architecture=package.architecture or "",
                        epoch=package.epoch,
                        version=package.version or "",
                        format="deb" if package_manager == "apt-get" else "rpm",
                        filename=filename,
                        sha256=digest,
                        size_bytes=size,
                        verification=(
                            "apt-repository"
                            if package_manager == "apt-get"
                            else "rpm-signature"
                        ),
                        roles=tuple(sorted(roles)),
                        signer_ids=tuple(signer_ids),
                    )
                )
        finally:
            pass
        for download_name, _batch_packages in batches:
            _discard_staging_tree_at(incoming_fd, download_name, owner_uid)
        os.close(reserve_fd)
        reserve_fd = -1
        _discard_staging_tree_at(incoming_fd, _SPACE_RESERVE_NAME, owner_uid)
        _require_free_space(snapshot_path.parent)
        manifest = PackagePayloadBundle(
            directory=final_name,
            packages=tuple(entries),
            total_size_bytes=total_size,
        )
        os.fsync(bundle_fd)
        os.close(bundle_fd)
        bundle_fd = -1
        _discard_staging_tree_at(parent_fd, incoming_temp, owner_uid)
        os.close(incoming_fd)
        incoming_fd = -1
        os.rename(bundle_temp, final_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        published = True
        os.fsync(parent_fd)
        validate_package_payloads(
            snapshot_path,
            manifest,
            packages,
            package_manager,
            forward_packages=forward_packages,
            runner=runner,
            required_owner_uid=owner_uid,
        )
        _verify_parent_binding(snapshot_path.parent, parent_fd, owner_uid)
        completed = True
        return manifest
    except (OSError, BoundedFileError) as exc:
        raise PackagePayloadError(f"cannot create rollback package payload bundle: {exc}") from exc
    finally:
        if bundle_fd >= 0:
            os.close(bundle_fd)
        if incoming_fd >= 0:
            os.close(incoming_fd)
        if reserve_fd >= 0:
            os.close(reserve_fd)
        for name in (bundle_temp, incoming_temp):
            try:
                _discard_staging_tree_at(parent_fd, name, owner_uid)
            except (FileNotFoundError, OSError):
                pass
        if published and not completed:
            # A published bundle is removed only when validation failed before
            # any snapshot could bind it. Successful returns leave it durable.
            try:
                _discard_staging_tree_at(parent_fd, final_name, owner_uid)
            except (FileNotFoundError, OSError):
                pass
        os.close(parent_fd)


def validate_package_payloads(
    snapshot_path: Path,
    bundle: PackagePayloadBundle,
    packages: list[PackageInfo],
    package_manager: str,
    *,
    forward_packages: list[PackageInfo] | None = None,
    runner: PayloadRunner | None = None,
    required_owner_uid: int | None = None,
) -> dict[PackageIdentity, Path]:
    """Validate a complete bundle and return its exact local package paths."""

    snapshot_path = _normalized_absolute(snapshot_path)
    owner_uid = os.geteuid() if required_owner_uid is None else required_owner_uid
    _validate_requested_packages(packages, package_manager)
    if len(packages) > MAX_BASELINE_PACKAGE_PAYLOADS:
        raise PackagePayloadError("rollback baseline package count exceeds the limit")
    if len(bundle.packages) > MAX_PACKAGE_PAYLOAD_ENTRIES:
        raise PackagePayloadError("package payload bundle entry count exceeds the limit")
    if forward_packages is not None:
        _validate_requested_packages(forward_packages, package_manager)
    expected_directory = payload_bundle_directory(snapshot_path)
    if bundle.directory != expected_directory:
        raise PackagePayloadError(
            "rollback package payload directory is not bound to the snapshot path"
        )
    expected = {package_identity(package): package for package in packages}
    if len(expected) != len(packages):
        raise PackagePayloadError("rollback baseline package identities are not unique")
    entries = {package_identity(entry): entry for entry in bundle.packages}
    baseline_entries = {
        identity for identity, entry in entries.items() if "baseline" in entry.roles
    }
    if len(entries) != len(bundle.packages) or baseline_entries != set(expected):
        raise PackagePayloadError(
            "rollback package payload manifest is not one-to-one with baseline packages"
        )
    if forward_packages is not None:
        expected_forward = {
            package_identity(package) for package in forward_packages
        }
        observed_forward = {
            identity for identity, entry in entries.items() if "forward" in entry.roles
        }
        if observed_forward != expected_forward:
            raise PackagePayloadError(
                "rollback package payload manifest does not match the resolved forward transaction"
            )
    filenames = [entry.filename for entry in bundle.packages]
    if len(set(filenames)) != len(filenames):
        raise PackagePayloadError("rollback package payload filenames are not unique")
    bundle_path = snapshot_path.parent / bundle.directory
    try:
        bundle_fd = open_private_directory(
            bundle_path,
            required_owner_uid=owner_uid,
        )
    except (OSError, BoundedFileError) as exc:
        raise PackagePayloadError(
            f"cannot open private rollback package payload bundle: {exc}"
        ) from exc
    before = os.fstat(bundle_fd)
    paths: dict[PackageIdentity, Path] = {}
    actual_total = 0
    try:
        observed_names = set(os.listdir(bundle_fd))
        if observed_names != set(filenames):
            raise PackagePayloadError(
                "rollback package payload bundle contains missing or unmanifested files"
            )
        expected_format = "deb" if package_manager == "apt-get" else "rpm"
        expected_verification = (
            "apt-repository" if package_manager == "apt-get" else "rpm-signature"
        )
        for identity, entry in entries.items():
            extension = ".deb" if entry.format == "deb" else ".rpm"
            if (
                entry.format != expected_format
                or entry.verification != expected_verification
                or _DIGEST.fullmatch(entry.sha256) is None
                or entry.filename != f"{entry.sha256}{extension}"
                or entry.size_bytes <= 0
                or entry.size_bytes > MAX_PACKAGE_PAYLOAD_BYTES
                or not entry.roles
                or len(set(entry.roles)) != len(entry.roles)
                or set(entry.roles) - {"baseline", "forward"}
                or (entry.format == "deb" and entry.signer_ids)
                or (entry.format == "rpm" and not entry.signer_ids)
                or any(re.fullmatch(r"[0-9a-f]{8,40}", item) is None for item in entry.signer_ids)
            ):
                raise PackagePayloadError(
                    f"rollback package payload manifest is invalid for {_format_identity(identity)}"
                )
            digest, size = _hash_bound_file(
                bundle_fd,
                entry.filename,
                required_owner_uid=owner_uid,
                require_private_mode=True,
            )
            if digest != entry.sha256 or size != entry.size_bytes:
                raise PackagePayloadError(
                    f"rollback package payload bytes changed for {_format_identity(identity)}"
                )
            path = bundle_path / entry.filename
            if runner is not None:
                observed, signer_ids = _inspect_payload(path, package_manager, runner)
                if observed != identity or tuple(signer_ids) != entry.signer_ids:
                    raise PackagePayloadError(
                        "rollback package payload header or signature changed for "
                        f"{_format_identity(identity)}"
                    )
            actual_total += size
            if actual_total > MAX_PACKAGE_PAYLOAD_TOTAL_BYTES:
                raise PackagePayloadError(
                    "rollback package payload bundle exceeds the total size limit"
                )
            paths[identity] = path
        if actual_total != bundle.total_size_bytes:
            raise PackagePayloadError(
                "rollback package payload bundle total size is inconsistent"
            )
        after = os.fstat(bundle_fd)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise PackagePayloadError(
                "rollback package payload directory changed during validation"
            )
    except OSError as exc:
        raise PackagePayloadError(
            f"cannot validate rollback package payload bundle: {exc}"
        ) from exc
    finally:
        os.close(bundle_fd)
    rebound_fd = open_private_directory(bundle_path, required_owner_uid=owner_uid)
    try:
        if _metadata_fingerprint(before) != _metadata_fingerprint(os.fstat(rebound_fd)):
            raise PackagePayloadError(
                "rollback package payload directory binding changed during validation"
            )
    finally:
        os.close(rebound_fd)
    return paths


def _download_command(
    package_manager: str,
    packages: list[PackageInfo],
    destination: Path,
    *,
    total_limit: int = MAX_PACKAGE_PAYLOAD_TOTAL_BYTES,
) -> list[str]:
    if total_limit < 0 or total_limit > MAX_PACKAGE_PAYLOAD_TOTAL_BYTES:
        raise PackagePayloadError("invalid remaining package payload size budget")
    if package_manager == "apt-get":
        specs = [
            f"{package.name}:{package.architecture}={package.version}"
            for package in packages
        ]
        return [
            "apt-get",
            "-o",
            f"Dir::Cache::archives={destination}/",
            "-o",
            "APT::Sandbox::User=root",
            "-o",
            "Acquire::AllowInsecureRepositories=false",
            "-o",
            "Acquire::AllowDowngradeToInsecureRepositories=false",
            "-o",
            "Acquire::AllowWeakRepositories=false",
            "-o",
            "APT::Get::AllowUnauthenticated=false",
            "--download-only",
            "install",
            "-y",
            "--reinstall",
            "--allow-change-held-packages",
            "--allow-downgrades",
            "--no-install-recommends",
            *specs,
        ]
    if package_manager in {"dnf", "yum"}:
        identities = [package_identity(package) for package in packages]
        return [
            "python3",
            "-I",
            "-c",
            _DNF_DOWNLOAD_SCRIPT,
            str(destination),
            str(MAX_PACKAGE_PAYLOAD_BYTES),
            str(total_limit),
            str(MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES),
            *(
                field
                for identity in identities
                for field in (
                    identity[0],
                    identity[1],
                    identity[2] or "",
                    identity[3],
                )
            ),
        ]
    if package_manager == "zypper":
        specs = [
            f"{package.name}.{package.architecture}="
            f"{f'{package.epoch}:' if package.epoch else ''}{package.version}"
            for package in packages
        ]
        return [
            "zypper",
            "--xmlout",
            "--non-interactive",
            "--no-refresh",
            "--pkg-cache-dir",
            str(destination),
            "download",
            *specs,
        ]
    raise PackagePayloadError(
        f"rollback package payload staging is unsupported for {package_manager}"
    )


def _download_size_command(
    package_manager: str,
    packages: list[PackageInfo],
    destination: Path,
) -> list[str]:
    """Return a no-download command that reports an authenticated byte bound."""

    if package_manager == "apt-get":
        specs = [
            f"{package.name}:{package.architecture}={package.version}"
            for package in packages
        ]
        return [
            "apt-get",
            "-qq",
            "-o",
            f"Dir::Cache::archives={destination}/",
            "-o",
            "APT::Sandbox::User=root",
            "-o",
            "Acquire::AllowInsecureRepositories=false",
            "-o",
            "Acquire::AllowDowngradeToInsecureRepositories=false",
            "-o",
            "Acquire::AllowWeakRepositories=false",
            "-o",
            "APT::Get::AllowUnauthenticated=false",
            "--download-only",
            "--print-uris",
            "install",
            "-y",
            "--reinstall",
            "--allow-change-held-packages",
            "--allow-downgrades",
            "--no-install-recommends",
            *specs,
        ]
    if package_manager == "zypper":
        specs = [
            f"{package.name}.{package.architecture}="
            f"{f'{package.epoch}:' if package.epoch else ''}{package.version}"
            for package in packages
        ]
        return [
            "zypper",
            "--xmlout",
            "--non-interactive",
            "--no-refresh",
            "--pkg-cache-dir",
            str(destination),
            "install",
            "--dry-run",
            "--details",
            "--force",
            "--no-recommends",
            "--",
            *specs,
        ]
    raise PackagePayloadError(
        f"package download size probing is unsupported for {package_manager}"
    )


def _parse_download_size(
    package_manager: str,
    result: CommandResult,
    packages: list[PackageInfo],
    remaining_budget: int,
) -> int:
    if result.returncode != 0:
        raise PackagePayloadError(
            "could not establish the exact package download size before staging: "
            f"{_command_diagnostic(result)}"
        )
    if "[output truncated:" in result.stdout or "[output truncated:" in result.stderr:
        raise PackagePayloadError(
            "package download size evidence was truncated before staging"
        )
    if package_manager == "apt-get":
        planned_size = _parse_apt_download_size(result.stdout, len(packages))
    elif package_manager == "zypper":
        planned_size = _parse_zypper_download_size(result.stdout, packages)
    else:
        raise PackagePayloadError(
            f"package download size parsing is unsupported for {package_manager}"
        )
    if planned_size <= 0:
        raise PackagePayloadError("package download plan has no positive payload size")
    if planned_size > remaining_budget:
        raise PackagePayloadError(
            "package download plan exceeds the remaining payload size limit"
        )
    return planned_size


def _parse_apt_download_size(output: str, expected_count: int) -> int:
    rows: list[tuple[str, str, int, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        try:
            fields = shlex.split(line, posix=True)
        except ValueError as exc:
            raise PackagePayloadError(
                "APT emitted malformed package download size evidence"
            ) from exc
        if len(fields) != 4:
            raise PackagePayloadError(
                "APT package download size evidence is not a four-field record"
            )
        uri, filename, size_text, digest = fields
        parsed_uri = urlsplit(uri)
        if (
            not parsed_uri.scheme
            or parsed_uri.scheme.lower() not in {"copy", "file", "ftp", "http", "https"}
            or not filename
            or Path(filename).name != filename
            or not filename.endswith(".deb")
            or re.fullmatch(r"[1-9][0-9]*", size_text) is None
            or re.fullmatch(r"(?:MD5Sum:)?[0-9A-Fa-f]{32}", digest) is None
        ):
            raise PackagePayloadError(
                "APT emitted invalid package download size evidence"
            )
        size = int(size_text)
        if size > MAX_PACKAGE_PAYLOAD_BYTES:
            raise PackagePayloadError(
                "APT package download plan exceeds the per-file size limit"
            )
        rows.append((uri, filename, size, digest.lower()))
    if len(rows) != expected_count:
        raise PackagePayloadError(
            "APT package download plan is not one-to-one with the exact request"
        )
    filenames = [row[1] for row in rows]
    uris = [row[0] for row in rows]
    if len(set(filenames)) != len(filenames) or len(set(uris)) != len(uris):
        raise PackagePayloadError(
            "APT package download plan contains duplicate payload records"
        )
    return sum(row[2] for row in rows)


def _parse_zypper_download_size(
    output: str,
    packages: list[PackageInfo],
) -> int:
    try:
        root = parse_bounded_xml(output)
    except SafeXmlError as exc:
        raise PackagePayloadError(
            "Zypper emitted malformed package download size evidence"
        ) from exc
    if root.tag != "stream":
        raise PackagePayloadError(
            "Zypper package download size evidence has an invalid root"
        )
    summaries = root.findall("install-summary")
    if len(summaries) != 1:
        raise PackagePayloadError(
            "Zypper package download size evidence has no singular transaction summary"
        )
    summary = summaries[0]
    size_text = summary.attrib.get("download-size", "")
    if re.fullmatch(r"[1-9][0-9]*", size_text) is None:
        raise PackagePayloadError(
            "Zypper transaction summary has no positive byte download size"
        )
    expected = {
        (
            package.name,
            f"{f'{package.epoch}:' if package.epoch else ''}{package.version}",
            package.architecture or "",
        )
        for package in packages
    }
    observed: set[tuple[str, str, str]] = set()
    for group in summary:
        if group.tag not in {
            "to-install",
            "to-upgrade",
            "to-downgrade",
            "to-reinstall",
        }:
            continue
        for solvable in group:
            if solvable.tag != "solvable" or solvable.attrib.get("kind") != "package":
                raise PackagePayloadError(
                    "Zypper package download size evidence contains an invalid action"
                )
            identity = (
                solvable.attrib.get("name", ""),
                solvable.attrib.get("edition", ""),
                solvable.attrib.get("arch", ""),
            )
            if not all(identity) or identity in observed:
                raise PackagePayloadError(
                    "Zypper package download size evidence contains an ambiguous identity"
                )
            observed.add(identity)
    missing = expected - observed
    unrequested = observed - expected
    if missing or unrequested:
        detail: list[str] = []
        if missing:
            detail.append(
                "missing "
                + ", ".join(
                    f"{name}-{edition}.{architecture}"
                    for name, edition, architecture in sorted(missing)
                )
            )
        if unrequested:
            detail.append(
                "unrequested "
                + ", ".join(
                    f"{name}-{edition}.{architecture}"
                    for name, edition, architecture in sorted(unrequested)
                )
            )
        raise PackagePayloadError(
            "Zypper package download plan is not the exact requested identity set: "
            + "; ".join(detail)
        )
    return int(size_text)


def _inspect_payload(
    path: Path,
    package_manager: str,
    runner: PayloadRunner,
) -> tuple[PackageIdentity, list[str]]:
    if package_manager == "apt-get":
        result = runner.run(
            [
                "dpkg-deb",
                "--show",
                "--showformat=${Package}\\t${Version}\\t${Architecture}\\n",
                str(path),
            ],
            mutate=False,
            allow_fail=True,
        )
        if result.returncode != 0:
            raise PackagePayloadError(
                f"cannot inspect retained DEB payload {path.name}: {_command_diagnostic(result)}"
            )
        rows = [line.split("\t") for line in result.stdout.splitlines() if line]
        if len(rows) != 1 or len(rows[0]) != 3 or any(not value for value in rows[0]):
            raise PackagePayloadError(
                f"retained DEB payload has an invalid package header: {path.name}"
            )
        name, version, architecture = rows[0]
        return (name, architecture, None, version), []
    query = runner.run(
        [
            "rpm",
            "-qp",
            "--qf",
            "%{NAME}\\t%{EPOCHNUM}\\t%{VERSION}-%{RELEASE}\\t%{ARCH}\\n",
            str(path),
        ],
        mutate=False,
        allow_fail=True,
    )
    if query.returncode != 0:
        raise PackagePayloadError(
            f"cannot inspect retained RPM payload {path.name}: {_command_diagnostic(query)}"
        )
    rows = [line.split("\t") for line in query.stdout.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 4 or any(not value for value in rows[0]):
        raise PackagePayloadError(
            f"retained RPM payload has an invalid package header: {path.name}"
        )
    name, epoch_text, version, architecture = rows[0]
    if re.fullmatch(r"(?:[0-9]+|\(none\))", epoch_text) is None:
        raise PackagePayloadError(
            f"retained RPM payload has an invalid epoch: {path.name}"
        )
    signature = runner.run(
        ["rpmkeys", "--checksig", "--verbose", str(path)],
        mutate=False,
        allow_fail=True,
    )
    signature_text = f"{signature.stdout}\n{signature.stderr}"
    if (
        signature.returncode != 0
        or re.search(r"\b(?:NOT OK|NOKEY|NOTTRUSTED|UNSIGNED)\b", signature_text, re.IGNORECASE)
        or re.search(r"\b(?:signature|gpg|pgp)s?\b.*\bOK\b", signature_text, re.IGNORECASE) is None
    ):
        raise PackagePayloadError(
            f"retained RPM payload lacks a trusted signature: {path.name}: "
            f"{_command_diagnostic(signature)}"
        )
    signer_ids = sorted({value.lower() for value in _SIGNER_ID.findall(signature_text)})
    if not signer_ids:
        raise PackagePayloadError(
            f"retained RPM payload signature has no signer identity: {path.name}"
        )
    epoch = epoch_text if epoch_text not in {"0", "(none)"} else None
    return (name, architecture, epoch, version), signer_ids


def _validate_requested_packages(
    packages: Iterable[PackageInfo], package_manager: str
) -> None:
    if package_manager not in {"apt-get", "dnf", "yum", "zypper"}:
        raise PackagePayloadError(
            f"rollback package payload staging is unsupported for {package_manager}"
        )
    expected_manager = "apt" if package_manager == "apt-get" else "rpm"
    identities: set[PackageIdentity] = set()
    for package in packages:
        identity = package_identity(package)
        if (
            not package.installed
            or package.manager != expected_manager
            or not all((identity[0], identity[1], identity[3]))
        ):
            raise PackagePayloadError(
                "rollback package payload staging requires complete installed package identities"
            )
        if identity in identities:
            raise PackagePayloadError("rollback baseline package identities are not unique")
        identities.add(identity)


def _find_candidates(
    root_fd: int,
    *,
    extension: str,
    required_owner_uid: int,
) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    total_size = 0

    def walk(directory_fd: int, relative: tuple[str, ...]) -> None:
        nonlocal total_size
        for name in sorted(os.listdir(directory_fd)):
            if not _safe_component(name):
                raise PackagePayloadError(
                    "package manager created an unsafe payload staging name"
                )
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = _open_relative_directory(
                    directory_fd,
                    (name,),
                    required_owner_uid,
                )
                try:
                    walk(child_fd, (*relative, name))
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise PackagePayloadError(
                    "package manager created a non-regular payload staging object"
                )
            _validate_candidate_metadata(metadata, required_owner_uid)
            if metadata.st_size > MAX_PACKAGE_PAYLOAD_BYTES:
                raise PackagePayloadError(
                    "package manager created an oversized payload staging object"
                )
            total_size += metadata.st_size
            if name.endswith(extension):
                candidates.append(_Candidate(relative, name, metadata.st_size))
    walk(root_fd, ())
    return candidates, total_size


def _create_space_reserve(parent_fd: int, owner_uid: int) -> int:
    """Allocate reclaimable private bytes that survive every package download."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(_SPACE_RESERVE_NAME, flags, 0o600, dir_fd=parent_fd)
    try:
        os.fchmod(descriptor, 0o600)
        allocated = False
        if hasattr(os, "posix_fallocate"):
            try:
                os.posix_fallocate(
                    descriptor,
                    0,
                    MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES,
                )
                allocated = True
            except OSError as exc:
                if exc.errno not in {
                    errno.EINVAL,
                    errno.ENOSYS,
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }:
                    raise
        if not allocated:
            block = bytes(_READ_CHUNK_BYTES)
            remaining = MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES
            while remaining:
                written = os.write(descriptor, block[: min(len(block), remaining)])
                if written <= 0:
                    raise OSError(errno.EIO, "short write to payload space reserve")
                remaining -= written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != owner_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES
        ):
            raise PackagePayloadError("private payload space reserve is unsafe")
        os.fsync(parent_fd)
        return descriptor
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(_SPACE_RESERVE_NAME, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            pass
        raise


def _hash_bound_file(
    parent_fd: int,
    name: str,
    *,
    required_owner_uid: int,
    require_private_mode: bool,
) -> tuple[str, int]:
    descriptor = os.open(name, _file_read_flags(), dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        _validate_candidate_metadata(before, required_owner_uid)
        if require_private_mode and stat.S_IMODE(before.st_mode) != 0o600:
            raise PackagePayloadError("retained package payload must have mode 0600")
        if before.st_size <= 0 or before.st_size > MAX_PACKAGE_PAYLOAD_BYTES:
            raise PackagePayloadError("retained package payload has an invalid size")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > MAX_PACKAGE_PAYLOAD_BYTES:
                raise PackagePayloadError("retained package payload exceeds the size limit")
        after = os.fstat(descriptor)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise PackagePayloadError("retained package payload changed while hashing")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _validate_candidate_metadata(metadata: os.stat_result, owner_uid: int) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != owner_uid
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise PackagePayloadError(
            "retained package payload must be a singly linked, owner-controlled regular file"
        )


def _create_private_child_directory(parent_fd: int, name: str, owner_uid: int) -> int:
    if not _safe_component(name):
        raise PackagePayloadError("unsafe private payload directory name")
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    os.fsync(parent_fd)
    descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PackagePayloadError("private payload directory metadata is unsafe")
    return descriptor


def _open_relative_directory(
    root_fd: int,
    components: tuple[str, ...],
    owner_uid: int,
) -> int:
    descriptor = os.dup(root_fd)
    try:
        for component in components:
            if not _safe_component(component):
                raise PackagePayloadError("unsafe payload staging path component")
            next_fd = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != owner_uid
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise PackagePayloadError(
                    "payload staging directory is not owner-controlled"
                )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _discard_staging_tree_at(
    parent_fd: int,
    name: str,
    owner_uid: int,
    budget: _CleanupBudget | None = None,
    *,
    _depth: int = 0,
) -> None:
    """Boundedly discard output below one process-created private root."""

    if not _safe_component(name):
        raise PackagePayloadError("unsafe package payload staging artifact name")
    if _depth > MAX_PACKAGE_PAYLOAD_CLEANUP_DEPTH:
        raise PackagePayloadError(
            "package payload staging tree exceeds the cleanup depth limit"
        )
    cleanup_budget = _CleanupBudget() if budget is None else budget
    before = _stat_at(parent_fd, name)
    if before is None:
        return
    cleanup_budget.consume(before)
    if _depth == 0:
        if stat.S_ISDIR(before.st_mode):
            _validate_cleanup_entry(
                before,
                name,
                owner_uid,
                require_directory=True,
                exact_private_directory=True,
            )
        else:
            _validate_cleanup_entry(
                before,
                name,
                owner_uid,
                require_directory=False,
                exact_private_directory=False,
                exact_private_file=True,
            )
    if not stat.S_ISDIR(before.st_mode):
        named = _stat_at(parent_fd, name)
        if named is None or (
            named.st_dev,
            named.st_ino,
            named.st_mode,
        ) != (before.st_dev, before.st_ino, before.st_mode):
            raise PackagePayloadError(
                f"package payload staging artifact changed before cleanup: {name}"
            )
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise PackagePayloadError(
            f"cannot bind package payload staging directory {name}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise PackagePayloadError(
                f"package payload staging directory changed while opening: {name}"
            )
        children: list[str] = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if not _safe_component(entry.name):
                    raise PackagePayloadError(
                        "package payload staging tree has an unsafe name"
                    )
                children.append(entry.name)
                if (
                    cleanup_budget.entries + len(children)
                    > MAX_PACKAGE_PAYLOAD_CLEANUP_ENTRIES
                ):
                    raise PackagePayloadError(
                        "package payload staging tree exceeds the cleanup entry limit"
                    )
        for child in sorted(children):
            _discard_staging_tree_at(
                descriptor,
                child,
                owner_uid,
                cleanup_budget,
                _depth=_depth + 1,
            )
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        named = _stat_at(parent_fd, name)
        if named is None or (
            named.st_dev,
            named.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise PackagePayloadError(
                f"package payload staging directory changed before cleanup: {name}"
            )
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)


def _remove_tree_at(
    parent_fd: int,
    name: str,
    owner_uid: int,
    budget: _CleanupBudget | None = None,
    *,
    _depth: int = 0,
) -> None:
    """Remove one bounded owner-controlled tree without following links."""

    if not _safe_component(name):
        raise PackagePayloadError("unsafe interrupted rollback artifact name")
    if _depth > MAX_PACKAGE_PAYLOAD_CLEANUP_DEPTH:
        raise PackagePayloadError(
            "interrupted rollback artifact tree exceeds the cleanup depth limit"
        )
    cleanup_budget = _CleanupBudget() if budget is None else budget
    metadata = _stat_at(parent_fd, name)
    if metadata is None:
        return
    cleanup_budget.consume(metadata)
    if stat.S_ISREG(metadata.st_mode):
        descriptor = _open_cleanup_regular_at(parent_fd, name, owner_uid)
        try:
            _revalidate_cleanup_entry(
                parent_fd,
                name,
                descriptor,
                owner_uid,
                require_directory=False,
                exact_private_directory=False,
            )
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(descriptor)
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise PackagePayloadError(
            f"interrupted rollback artifact has an unsafe type: {name}"
        )
    descriptor = _open_cleanup_directory_at(
        parent_fd,
        name,
        owner_uid,
        exact_private=_depth == 0,
    )
    try:
        children: list[str] = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if not _safe_component(entry.name):
                    raise PackagePayloadError(
                        "interrupted rollback artifact tree has an unsafe name"
                    )
                children.append(entry.name)
                if (
                    cleanup_budget.entries + len(children)
                    > MAX_PACKAGE_PAYLOAD_CLEANUP_ENTRIES
                ):
                    raise PackagePayloadError(
                        "interrupted rollback artifact tree exceeds the cleanup entry limit"
                    )
        for child in sorted(children):
            _remove_tree_at(
                descriptor,
                child,
                owner_uid,
                cleanup_budget,
                _depth=_depth + 1,
            )
        os.fsync(descriptor)
        _revalidate_cleanup_entry(
            parent_fd,
            name,
            descriptor,
            owner_uid,
            require_directory=True,
            exact_private_directory=_depth == 0,
        )
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)


def _validate_cleanup_entry(
    metadata: os.stat_result,
    name: str,
    owner_uid: int,
    *,
    require_directory: bool,
    exact_private_directory: bool,
    exact_private_file: bool = False,
) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if require_directory:
        trusted = bool(
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == owner_uid
            and (
                mode == 0o700
                if exact_private_directory
                else mode & 0o022 == 0
            )
        )
    else:
        trusted = bool(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and metadata.st_uid == owner_uid
            and (
                mode & 0o077 == 0
                if exact_private_file
                else mode & 0o022 == 0
            )
        )
    if not trusted:
        raise PackagePayloadError(
            f"interrupted rollback artifact metadata is unsafe: {name}"
        )


def _open_cleanup_entry_at(
    parent_fd: int,
    name: str,
    owner_uid: int,
    *,
    require_directory: bool,
    exact_private_directory: bool,
    exact_private_file: bool = False,
) -> int:
    before = _stat_at(parent_fd, name)
    if before is None:
        raise PackagePayloadError(f"bound rollback artifact is missing: {name}")
    _validate_cleanup_entry(
        before,
        name,
        owner_uid,
        require_directory=require_directory,
        exact_private_directory=exact_private_directory,
        exact_private_file=exact_private_file,
    )
    flags = _directory_flags() if require_directory else _file_read_flags()
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise PackagePayloadError(
            f"cannot bind interrupted rollback artifact {name}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        _validate_cleanup_entry(
            opened,
            name,
            owner_uid,
            require_directory=require_directory,
            exact_private_directory=exact_private_directory,
            exact_private_file=exact_private_file,
        )
        if _metadata_fingerprint(before) != _metadata_fingerprint(opened):
            raise PackagePayloadError(
                f"interrupted rollback artifact changed while opening: {name}"
            )
        _revalidate_cleanup_entry(
            parent_fd,
            name,
            descriptor,
            owner_uid,
            require_directory=require_directory,
            exact_private_directory=exact_private_directory,
            exact_private_file=exact_private_file,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_cleanup_regular_at(parent_fd: int, name: str, owner_uid: int) -> int:
    return _open_cleanup_entry_at(
        parent_fd,
        name,
        owner_uid,
        require_directory=False,
        exact_private_directory=False,
    )


def _open_cleanup_directory_at(
    parent_fd: int,
    name: str,
    owner_uid: int,
    *,
    exact_private: bool,
) -> int:
    return _open_cleanup_entry_at(
        parent_fd,
        name,
        owner_uid,
        require_directory=True,
        exact_private_directory=exact_private,
    )


def _open_exact_regular_at(parent_fd: int, name: str, owner_uid: int) -> int:
    return _open_cleanup_entry_at(
        parent_fd,
        name,
        owner_uid,
        require_directory=False,
        exact_private_directory=False,
        exact_private_file=True,
    )


def _open_exact_directory_at(parent_fd: int, name: str, owner_uid: int) -> int:
    return _open_cleanup_entry_at(
        parent_fd,
        name,
        owner_uid,
        require_directory=True,
        exact_private_directory=True,
    )


def _revalidate_cleanup_entry(
    parent_fd: int,
    name: str,
    descriptor: int,
    owner_uid: int,
    *,
    require_directory: bool,
    exact_private_directory: bool,
    exact_private_file: bool = False,
) -> None:
    opened = os.fstat(descriptor)
    named = _stat_at(parent_fd, name)
    if named is None:
        raise PackagePayloadError(
            f"interrupted rollback artifact disappeared while bound: {name}"
        )
    _validate_cleanup_entry(
        opened,
        name,
        owner_uid,
        require_directory=require_directory,
        exact_private_directory=exact_private_directory,
        exact_private_file=exact_private_file,
    )
    _validate_cleanup_entry(
        named,
        name,
        owner_uid,
        require_directory=require_directory,
        exact_private_directory=exact_private_directory,
        exact_private_file=exact_private_file,
    )
    if _metadata_fingerprint(opened) != _metadata_fingerprint(named):
        raise PackagePayloadError(
            f"interrupted rollback artifact changed while bound: {name}"
        )


def _revalidate_exact_entry(
    parent_fd: int,
    name: str,
    descriptor: int,
    owner_uid: int,
    *,
    require_directory: bool,
) -> None:
    _revalidate_cleanup_entry(
        parent_fd,
        name,
        descriptor,
        owner_uid,
        require_directory=require_directory,
        exact_private_directory=require_directory,
        exact_private_file=not require_directory,
    )


def _require_exact_regular_at(parent_fd: int, name: str, owner_uid: int) -> None:
    descriptor = _open_exact_regular_at(parent_fd, name, owner_uid)
    os.close(descriptor)


def _require_exact_directory_at(parent_fd: int, name: str, owner_uid: int) -> None:
    descriptor = _open_exact_directory_at(parent_fd, name, owner_uid)
    os.close(descriptor)


def _remove_exact_regular_at(
    parent_fd: int,
    name: str,
    owner_uid: int,
    budget: _CleanupBudget | None = None,
) -> None:
    metadata = _stat_at(parent_fd, name)
    if metadata is None:
        return
    if budget is not None:
        budget.consume(metadata)
    descriptor = _open_exact_regular_at(parent_fd, name, owner_uid)
    try:
        _revalidate_exact_entry(
            parent_fd,
            name,
            descriptor,
            owner_uid,
            require_directory=False,
        )
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)


def _remove_exact_directory_at(
    parent_fd: int,
    name: str,
    owner_uid: int,
    budget: _CleanupBudget | None = None,
) -> None:
    metadata = _stat_at(parent_fd, name)
    if metadata is None:
        return
    _remove_tree_at(parent_fd, name, owner_uid, budget)


def _verify_parent_binding(path: Path, descriptor: int, owner_uid: int) -> None:
    before = os.fstat(descriptor)
    rebound = open_private_directory(path, required_owner_uid=owner_uid)
    try:
        if _metadata_fingerprint(before) != _metadata_fingerprint(os.fstat(rebound)):
            raise PackagePayloadError(
                "rollback package payload parent changed during publication"
            )
    finally:
        os.close(rebound)


def _require_free_space(path: Path) -> None:
    filesystem = os.statvfs(path)
    available = filesystem.f_bavail * filesystem.f_frsize
    if available < MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES:
        raise PackagePayloadError(
            "insufficient free space to retain rollback package payloads"
        )


def _require_download_capacity(
    path: Path,
    planned_size: int,
    package_count: int,
) -> None:
    """Keep operating headroom beyond the separately allocated safety reserve."""

    if planned_size <= 0 or planned_size > MAX_PACKAGE_PAYLOAD_TOTAL_BYTES:
        raise PackagePayloadError("invalid planned package download size")
    if package_count <= 0 or package_count > MAX_PACKAGE_PAYLOAD_ENTRIES:
        raise PackagePayloadError("invalid planned package download count")
    filesystem = os.statvfs(path)
    available = filesystem.f_bavail * filesystem.f_frsize
    allocation_rounding = package_count * max(filesystem.f_frsize - 1, 0)
    required = (
        planned_size
        + allocation_rounding
        + PACKAGE_DOWNLOAD_FIXED_OVERHEAD_BYTES
        + MIN_FREE_AFTER_PAYLOAD_STAGE_BYTES
    )
    if available < required:
        raise PackagePayloadError(
            "insufficient capacity to bound package downloads while preserving operating headroom"
        )


def _normalized_absolute(path: Path) -> Path:
    if not path.is_absolute() or path.anchor != os.sep:
        raise PackagePayloadError("rollback snapshot path must be absolute")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if normalized != path or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise PackagePayloadError("rollback snapshot path must be normalized")
    return normalized


def _safe_component(name: str) -> bool:
    return bool(
        name
        and name not in {".", ".."}
        and os.sep not in name
        and (os.altsep is None or os.altsep not in name)
        and "\x00" not in name
        and len(os.fsencode(name)) <= 255
    )


def _stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _directory_flags() -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise PackagePayloadError("platform lacks no-follow directory support")
    return flags


def _file_read_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise PackagePayloadError("platform lacks no-follow file support")
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _format_identity(identity: PackageIdentity) -> str:
    name, architecture, epoch, version = identity
    return f"{name}:{architecture}={f'{epoch}:' if epoch else ''}{version}"


def _command_diagnostic(result: CommandResult) -> str:
    detail = result.stderr.strip() or result.stdout.strip()
    return detail or f"command exited with status {result.returncode}"
