from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from nvidia_converge.models import (
    CommandResult,
    PackageInfo,
    PackagePayload,
    PackagePayloadBundle,
)
from nvidia_converge.package_payloads import (
    PackageIdentity,
    PackagePayloadError,
    cleanup_snapshot_payload_artifacts,
    forward_package_command,
    package_identity,
    payload_bundle_directory,
    stage_package_payloads,
    validate_package_payloads,
)
from nvidia_converge.rollback import new_snapshot_path

_SIGNER_ID = "deadbeefcafebabe"


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _private_file(path: Path, payload: bytes = b"private\n") -> Path:
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


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


class _ArtifactRunner:
    """Model authenticated package-manager output without executing packages."""

    def __init__(
        self,
        package_manager: str,
        packages: list[PackageInfo],
        *,
        failure_mode: str | None = None,
        extra_package: PackageInfo | None = None,
    ) -> None:
        self.package_manager = package_manager
        self.identities = {package_identity(package) for package in packages}
        self.failure_mode = failure_mode
        self.extra_identity = (
            package_identity(extra_package) if extra_package is not None else None
        )
        self.calls: list[tuple[list[str], bool]] = []
        self.download_requests: list[tuple[PackageIdentity, ...]] = []
        self._identity_by_bytes = {
            self._artifact_bytes(identity): identity
            for identity in self.identities
            | ({self.extra_identity} if self.extra_identity is not None else set())
        }
        self._failure_injected = False

    def run(
        self,
        command: list[str],
        *,
        mutate: bool = False,
        allow_fail: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        del allow_fail, input_text
        self.calls.append((list(command), mutate))
        if self._is_size_probe(command):
            requested = self._size_probe_identities(command)
            sizes = {
                identity: len(self._artifact_bytes(identity))
                for identity in requested
            }
            if command[0] == "apt-get":
                rows = [
                    (
                        f"'https://packages.example.invalid/{index}.deb' "
                        f"'package-{index}.deb' {sizes[identity]} "
                        f"MD5Sum:{'0' * 32}"
                    )
                    for index, identity in enumerate(requested)
                ]
                return CommandResult(command, 0, stdout="\n".join(rows) + "\n")
            solvables = "".join(
                (
                    '<solvable kind="package" '
                    f'name="{identity[0]}" '
                    f'edition="{self._zypper_edition(identity)}" '
                    f'arch="{identity[1]}"/>'
                )
                for identity in requested
            )
            return CommandResult(
                command,
                0,
                stdout=(
                    '<stream><install-summary '
                    f'download-size="{sum(sizes.values())}">'
                    f"<to-reinstall>{solvables}</to-reinstall>"
                    "</install-summary></stream>"
                ),
            )
        if self._is_download(command):
            if self.failure_mode == "download":
                return CommandResult(command, 1, stderr="download failed")
            destination = self._download_destination(command)
            requested = self._download_identities(command)
            self.download_requests.append(tuple(requested))
            emitted = list(requested)
            if self.failure_mode == "missing" and emitted:
                emitted.pop()
            if self.failure_mode == "extra" and self.extra_identity is not None:
                emitted.append(self.extra_identity)
            for index, identity in enumerate(emitted):
                self._write_artifact(destination, identity, index)
            if self.failure_mode in {"symlink", "hardlink"} and emitted:
                self._inject_unsafe_candidate(destination)
            return CommandResult(command, 0)
        if command[:2] == ["dpkg-deb", "--show"]:
            identity = self._identity_for_path(Path(command[-1]))
            name, architecture, _epoch, version = identity
            return CommandResult(
                command,
                0,
                stdout=f"{name}\t{version}\t{architecture}\n",
            )
        if command and command[0] == "rpm" and "-qp" in command:
            identity = self._identity_for_path(Path(command[-1]))
            name, architecture, epoch, version = identity
            return CommandResult(
                command,
                0,
                stdout=f"{name}\t{epoch or '0'}\t{version}\t{architecture}\n",
            )
        if command[:3] == ["rpmkeys", "--checksig", "--verbose"]:
            return CommandResult(
                command,
                0,
                stdout=(f"Header V4 RSA/SHA256 Signature, key ID {_SIGNER_ID}: OK\n"),
            )
        raise AssertionError(f"unexpected package payload command: {command!r}")

    def _is_download(self, command: list[str]) -> bool:
        return bool(
            (command and command[0] == "apt-get" and "--download-only" in command)
            or (command[:3] == ["python3", "-I", "-c"])
            or (command and command[0] == "zypper" and "download" in command)
        )

    @staticmethod
    def _is_size_probe(command: list[str]) -> bool:
        return bool(
            (command and command[0] == "apt-get" and "--print-uris" in command)
            or (
                command
                and command[0] == "zypper"
                and "--dry-run" in command
                and "install" in command
            )
        )

    def _size_probe_identities(
        self,
        command: list[str],
    ) -> list[PackageIdentity]:
        if command[0] == "apt-get":
            return self._download_identities(command)
        specs = set(command[command.index("--") + 1 :])
        return sorted(
            (
                identity
                for identity in self.identities
                if self._zypper_spec(identity) in specs
            ),
            key=self._identity_sort_key,
        )

    def _download_destination(self, command: list[str]) -> Path:
        if command[0] == "apt-get":
            prefix = "Dir::Cache::archives="
            value = next(part for part in command if part.startswith(prefix))
            return Path(value[len(prefix) :].rstrip("/"))
        if command[0] == "zypper":
            return Path(command[command.index("--pkg-cache-dir") + 1])
        return Path(command[4])

    def _download_identities(self, command: list[str]) -> list[PackageIdentity]:
        if command[0] == "apt-get":
            specs = set(command[command.index("--no-install-recommends") + 1 :])
            return sorted(
                (
                    identity
                    for identity in self.identities
                    if self._apt_spec(identity) in specs
                ),
                key=self._identity_sort_key,
            )
        if command[0] == "zypper":
            specs = set(command[command.index("download") + 1 :])
            return sorted(
                (
                    identity
                    for identity in self.identities
                    if self._zypper_spec(identity) in specs
                ),
                key=self._identity_sort_key,
            )
        fields = command[8:]
        assert len(fields) % 4 == 0
        return [
            (name, architecture, epoch or None, version)
            for name, architecture, epoch, version in (
                fields[index : index + 4] for index in range(0, len(fields), 4)
            )
        ]

    def _write_artifact(
        self,
        destination: Path,
        identity: PackageIdentity,
        index: int,
    ) -> None:
        repository = destination / "repo: CUDA + signed artifacts"
        repository.mkdir(mode=0o700, parents=True, exist_ok=True)
        extension = ".deb" if self.package_manager == "apt-get" else ".rpm"
        filename = f"package+cuda~release%3a-{index}{extension}"
        path = repository / filename
        path.write_bytes(self._artifact_bytes(identity))
        path.chmod(0o600)

    def _inject_unsafe_candidate(self, destination: Path) -> None:
        if self._failure_injected:
            return
        self._failure_injected = True
        repository = destination / "repo: CUDA + signed artifacts"
        extension = ".deb" if self.package_manager == "apt-get" else ".rpm"
        candidate = next(repository.glob(f"*{extension}"))
        if self.failure_mode == "symlink":
            target = repository / "zz-target.bin"
            target.write_bytes(b"not a package")
            target.chmod(0o600)
            candidate.unlink()
            candidate.symlink_to(target.name)
        else:
            os.link(candidate, repository / "zz-hardlink.bin")

    def _identity_for_path(self, path: Path) -> PackageIdentity:
        payload = path.read_bytes()
        try:
            return self._identity_by_bytes[payload]
        except KeyError as exc:
            raise AssertionError(f"unknown mock package bytes at {path}") from exc

    @staticmethod
    def _artifact_bytes(identity: PackageIdentity) -> bytes:
        return ("authenticated-package:" + repr(identity)).encode("utf-8")

    @staticmethod
    def _identity_sort_key(identity: PackageIdentity) -> tuple[str, str, str, str]:
        return (identity[0], identity[1], identity[2] or "", identity[3])

    @staticmethod
    def _apt_spec(identity: PackageIdentity) -> str:
        name, architecture, _epoch, version = identity
        return f"{name}:{architecture}={version}"

    @staticmethod
    def _zypper_spec(identity: PackageIdentity) -> str:
        name, architecture, epoch, version = identity
        return f"{name}.{architecture}={f'{epoch}:' if epoch else ''}{version}"

    @staticmethod
    def _zypper_edition(identity: PackageIdentity) -> str:
        _name, _architecture, epoch, version = identity
        return f"{epoch}:{version}" if epoch else version


def _stage_apt_bundle(
    tmp_path: Path,
) -> tuple[
    Path,
    list[PackageInfo],
    list[PackageInfo],
    _ArtifactRunner,
    PackagePayloadBundle,
]:
    snapshot_path = tmp_path / "snapshots" / "snapshot.json"
    baseline = [
        _package(
            "nvidia-driver",
            "1:580.1+cuda~ubuntu1",
            "apt",
            "amd64",
        )
    ]
    forward = [
        _package(
            "nvidia-driver",
            "1:590.2+cuda~ubuntu1",
            "apt",
            "amd64",
        ),
        _package(
            "libnvidia-dependency",
            "590.2+cuda~ubuntu1",
            "apt",
            "amd64",
        ),
    ]
    runner = _ArtifactRunner("apt-get", [*baseline, *forward])
    bundle = stage_package_payloads(
        snapshot_path,
        baseline,
        "apt-get",
        runner,
        forward_packages=forward,
    )
    return snapshot_path, baseline, forward, runner, bundle


def test_default_applied_snapshot_name_binds_payload_directory() -> None:
    operation_id = "a" * 32
    snapshot_path = new_snapshot_path(operation_id=operation_id)

    assert snapshot_path.is_absolute()
    assert operation_id in snapshot_path.name
    assert payload_bundle_directory(snapshot_path) == snapshot_path.name + ".payloads"


def test_apt_staging_separates_old_baseline_from_forward_dependency_batch(
    tmp_path: Path,
) -> None:
    snapshot_path, baseline, forward, runner, bundle = _stage_apt_bundle(tmp_path)

    requested = {identity for batch in runner.download_requests for identity in batch}
    expected = {package_identity(package) for package in [*baseline, *forward]}
    assert requested == expected
    for batch in runner.download_requests:
        slots = [(identity[0], identity[1]) for identity in batch]
        assert len(slots) == len(set(slots))
    assert any(
        {package_identity(package) for package in forward} == set(batch)
        for batch in runner.download_requests
    )
    assert isinstance(bundle.packages, tuple)
    assert {entry.roles for entry in bundle.packages} == {
        ("baseline",),
        ("forward",),
    }
    paths = validate_package_payloads(
        snapshot_path,
        bundle,
        baseline,
        "apt-get",
        forward_packages=forward,
        runner=runner,
    )
    assert set(paths) == expected
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths.values())


def test_zypper_batch_download_accepts_nested_punctuated_repository_paths(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshots" / "snapshot.json"
    baseline = [
        _package(
            "nvidia-open",
            "580.1+cuda~suse-1",
            "rpm",
            "x86_64",
        )
    ]
    forward = [
        _package(
            "nvidia-open",
            "590.2+cuda~suse-1",
            "rpm",
            "x86_64",
        ),
        _package(
            "nvidia-compute-utils",
            "590.2+cuda~suse-1",
            "rpm",
            "x86_64",
        ),
    ]
    runner = _ArtifactRunner("zypper", [*baseline, *forward])

    bundle = stage_package_payloads(
        snapshot_path,
        baseline,
        "zypper",
        runner,
        forward_packages=forward,
    )

    assert {identity for batch in runner.download_requests for identity in batch} == {
        package_identity(package) for package in [*baseline, *forward]
    }
    assert all(entry.signer_ids == (_SIGNER_ID,) for entry in bundle.packages)
    validate_package_payloads(
        snapshot_path,
        bundle,
        baseline,
        "zypper",
        forward_packages=forward,
        runner=runner,
    )


@pytest.mark.parametrize(
    "failure_mode", ["download", "missing", "extra", "symlink", "hardlink"]
)
def test_staging_failure_is_nonmutating_and_cleans_private_artifacts(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    snapshot_path = tmp_path / "snapshots" / "snapshot.json"
    baseline = [_package("nvidia-driver", "580.1", "apt", "amd64")]
    extra = _package("nvidia-unrequested", "580.1", "apt", "amd64")
    runner = _ArtifactRunner(
        "apt-get",
        baseline,
        failure_mode=failure_mode,
        extra_package=extra,
    )

    with pytest.raises(PackagePayloadError):
        stage_package_payloads(
            snapshot_path,
            baseline,
            "apt-get",
            runner,
        )

    assert runner.calls
    assert all(mutate is False for _command, mutate in runner.calls)
    assert snapshot_path.parent.is_dir()
    assert list(snapshot_path.parent.iterdir()) == []


def test_space_reserve_enospc_cleans_every_private_staging_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "snapshots" / "snapshot.json"
    baseline = [_package("nvidia-driver", "580.1", "apt", "amd64")]
    runner = _ArtifactRunner("apt-get", baseline)
    statvfs_calls: list[Path] = []
    filesystem = os.statvfs(tmp_path)

    def observed_statvfs(path: os.PathLike[str] | str) -> os.statvfs_result:
        statvfs_calls.append(Path(path))
        return filesystem

    def no_reserve_space(_fd: int, _offset: int, _length: int) -> None:
        raise OSError(errno.ENOSPC, "simulated full payload filesystem")

    monkeypatch.setattr(os, "statvfs", observed_statvfs)
    monkeypatch.setattr(
        os,
        "posix_fallocate",
        no_reserve_space,
        raising=False,
    )

    with pytest.raises(PackagePayloadError, match="cannot create"):
        stage_package_payloads(
            snapshot_path,
            baseline,
            "apt-get",
            runner,
        )

    assert statvfs_calls == [snapshot_path.parent]
    assert runner.calls == []
    assert snapshot_path.parent.is_dir()
    assert list(snapshot_path.parent.iterdir()) == []


@pytest.mark.parametrize(
    "tamper", ["content", "extra", "missing", "symlink", "hardlink"]
)
def test_validation_rejects_payload_bundle_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    snapshot_path, baseline, forward, _runner, bundle = _stage_apt_bundle(tmp_path)
    bundle_path = snapshot_path.parent / bundle.directory
    payload_path = bundle_path / bundle.packages[0].filename

    if tamper == "content":
        payload_path.write_bytes(b"tampered")
        payload_path.chmod(0o600)
    elif tamper == "extra":
        extra = bundle_path / "unmanifested"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)
    elif tamper == "missing":
        payload_path.unlink()
    elif tamper == "symlink":
        target = snapshot_path.parent / "outside-payload"
        target.write_bytes(b"outside")
        target.chmod(0o600)
        payload_path.unlink()
        payload_path.symlink_to(target)
    else:
        target = snapshot_path.parent / "outside-payload"
        target.write_bytes(b"outside")
        target.chmod(0o600)
        payload_path.unlink()
        os.link(target, payload_path)

    with pytest.raises(PackagePayloadError):
        validate_package_payloads(
            snapshot_path,
            bundle,
            baseline,
            "apt-get",
            forward_packages=forward,
        )


@pytest.mark.parametrize("package_manager", ["apt-get", "dnf", "yum", "zypper"])
def test_forward_commands_are_exact_local_offline_transactions(
    tmp_path: Path,
    package_manager: str,
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    payload_bytes = b"exact forward payload"
    digest = hashlib.sha256(payload_bytes).hexdigest()
    rpm = package_manager != "apt-get"
    extension = "rpm" if rpm else "deb"
    payload = PackagePayload(
        name="nvidia-driver",
        architecture="x86_64" if rpm else "amd64",
        epoch=None,
        version="590.1-1",
        format=extension,
        filename=f"{digest}.{extension}",
        sha256=digest,
        size_bytes=len(payload_bytes),
        verification="rpm-signature" if rpm else "apt-repository",
        roles=("forward",),
        signer_ids=(_SIGNER_ID,) if rpm else (),
    )
    bundle = PackagePayloadBundle(
        directory=payload_bundle_directory(snapshot_path),
        packages=(payload,),
        total_size_bytes=len(payload_bytes),
    )

    command = forward_package_command(snapshot_path, bundle, package_manager)
    local_path = str(snapshot_path.parent / bundle.directory / payload.filename)

    assert local_path in command
    assert not any(part.startswith(("http://", "https://")) for part in command)
    if package_manager == "apt-get":
        assert "--no-download" in command
    elif package_manager in {"dnf", "yum"}:
        assert "--disablerepo=*" in command
        assert "--setopt=localpkg_gpgcheck=1" in command
    else:
        assert "--disable-repositories" in command
        assert "--no-refresh" in command


def test_payload_directory_rejects_relative_or_non_normalized_snapshot_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(PackagePayloadError, match="absolute"):
        payload_bundle_directory(Path("snapshot.json"))
    with pytest.raises(PackagePayloadError, match="normalized"):
        payload_bundle_directory(tmp_path / "nested" / ".." / "snapshot.json")


@pytest.mark.parametrize(
    "unsafe_kind",
    ["symlink", "fifo", "hardlink", "world-writable"],
)
def test_interrupted_artifact_cleanup_rejects_unsafe_tree_entries(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    snapshot_dir = _private_directory(tmp_path / "snapshots")
    snapshot = snapshot_dir / "snapshot-test.json"
    bundle_temp = _private_directory(
        snapshot_dir / f".{snapshot.name}.payloads.tmp"
    )
    unsafe = bundle_temp / "unsafe"
    if unsafe_kind == "symlink":
        unsafe.symlink_to(tmp_path / "outside")
    elif unsafe_kind == "fifo":
        os.mkfifo(unsafe, 0o600)
    elif unsafe_kind == "hardlink":
        source = _private_file(bundle_temp / "source")
        os.link(source, unsafe)
    else:
        _private_file(unsafe)
        unsafe.chmod(0o622)

    with pytest.raises(PackagePayloadError, match="unsafe"):
        cleanup_snapshot_payload_artifacts(
            snapshot,
            preserve_bound_authority=False,
            required_owner_uid=os.geteuid(),
        )

    assert bundle_temp.is_dir()
    assert os.path.lexists(unsafe)


def test_interrupted_artifact_cleanup_rejects_wrong_owner_authority(
    tmp_path: Path,
) -> None:
    snapshot_dir = _private_directory(tmp_path / "snapshots")
    snapshot = snapshot_dir / "snapshot-test.json"
    snapshot_temp = _private_file(snapshot_dir / f".{snapshot.name}.tmp")

    with pytest.raises(PackagePayloadError, match="cannot bind"):
        cleanup_snapshot_payload_artifacts(
            snapshot,
            preserve_bound_authority=False,
            required_owner_uid=os.geteuid() + 1,
        )

    assert snapshot_temp.is_file()


@pytest.mark.parametrize("limit_kind", ["entries", "bytes", "depth"])
def test_interrupted_artifact_cleanup_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    from nvidia_converge import package_payloads as package_payloads_module

    snapshot_dir = _private_directory(tmp_path / "snapshots")
    snapshot = snapshot_dir / "snapshot-test.json"
    bundle_temp = _private_directory(
        snapshot_dir / f".{snapshot.name}.payloads.tmp"
    )
    if limit_kind == "entries":
        _private_file(bundle_temp / "payload")
        monkeypatch.setattr(
            package_payloads_module,
            "MAX_PACKAGE_PAYLOAD_CLEANUP_ENTRIES",
            1,
        )
    elif limit_kind == "bytes":
        _private_file(bundle_temp / "payload", b"x")
        monkeypatch.setattr(
            package_payloads_module,
            "MAX_PACKAGE_PAYLOAD_CLEANUP_BYTES",
            0,
        )
    else:
        _private_directory(bundle_temp / "nested")
        monkeypatch.setattr(
            package_payloads_module,
            "MAX_PACKAGE_PAYLOAD_CLEANUP_DEPTH",
            0,
        )

    with pytest.raises(PackagePayloadError, match=r"cleanup (?:entry|byte|depth) limit"):
        cleanup_snapshot_payload_artifacts(
            snapshot,
            preserve_bound_authority=False,
            required_owner_uid=os.geteuid(),
        )

    assert bundle_temp.is_dir()


def test_bound_artifact_cleanup_revalidates_preserved_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_converge import package_payloads as package_payloads_module

    snapshot_dir = _private_directory(tmp_path / "snapshots")
    snapshot = _private_file(snapshot_dir / "snapshot-test.json")
    bundle = _private_directory(snapshot_dir / f"{snapshot.name}.payloads")
    bundle_temp = _private_directory(
        snapshot_dir / f".{snapshot.name}.payloads.tmp"
    )
    original_remove = package_payloads_module._remove_exact_directory_at
    authority_changed = False

    def change_authority_after_temp_cleanup(
        parent_fd: int,
        name: str,
        owner_uid: int,
        budget: object = None,
    ) -> None:
        nonlocal authority_changed
        original_remove(parent_fd, name, owner_uid, budget)
        if not authority_changed:
            snapshot.chmod(0o644)
            authority_changed = True

    monkeypatch.setattr(
        package_payloads_module,
        "_remove_exact_directory_at",
        change_authority_after_temp_cleanup,
    )

    with pytest.raises(PackagePayloadError, match="unsafe"):
        cleanup_snapshot_payload_artifacts(
            snapshot,
            preserve_bound_authority=True,
            required_owner_uid=os.geteuid(),
        )

    assert snapshot.is_file()
    assert bundle.is_dir()
    assert not bundle_temp.exists()
