from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import CommandResult, HostAudit, PackageInfo, RollbackSnapshot, utc_now
from .runner import CommandRunner
from .audit import _interesting_package

SNAPSHOT_DIR = Path("/var/lib/nvidia-converge/snapshots")


class RollbackSnapshotError(ValueError):
    pass


def create_snapshot(audit: HostAudit, path: str | None = None, *, persist: bool = True) -> RollbackSnapshot:
    snapshot_path = Path(path) if path else SNAPSHOT_DIR / f"{utc_now().replace(':', '-')}.json"
    commands = _rollback_commands(audit.packages, audit.package_manager)
    snapshot = RollbackSnapshot(
        path=str(snapshot_path) if persist else None,
        packages=[pkg for pkg in audit.packages if pkg.installed],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=commands,
    )
    if not persist:
        return snapshot
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snapshot.__dict__, default=lambda obj: obj.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except PermissionError:
        fallback = Path.cwd() / "nvidia-converge-rollback.json"
        snapshot.path = str(fallback)
        fallback.write_text(json.dumps(snapshot.__dict__, default=lambda obj: obj.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def load_snapshot(path: str) -> RollbackSnapshot:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RollbackSnapshotError(f"cannot read rollback snapshot {path!r}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise RollbackSnapshotError(f"invalid rollback snapshot JSON {path!r}: line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise RollbackSnapshotError("rollback snapshot must be a JSON object")
    missing = [key for key in ("kernel", "packages", "commands") if key not in data]
    if missing:
        raise RollbackSnapshotError(f"rollback snapshot missing required field(s): {', '.join(missing)}")
    snapshot_path = _optional_string(data.get("path"), "path") or path
    kernel = _required_string(data["kernel"], "kernel")
    module_version = _optional_string(data.get("module_version"), "module_version")
    packages = _load_packages(data["packages"])
    commands = _load_commands(data["commands"])
    return RollbackSnapshot(path=snapshot_path, packages=packages, kernel=kernel, module_version=module_version, commands=commands)


def _load_packages(value: Any) -> list[PackageInfo]:
    if not isinstance(value, list):
        raise RollbackSnapshotError("rollback snapshot packages must be an array")
    packages: list[PackageInfo] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise RollbackSnapshotError(f"rollback snapshot packages[{index}] must be an object")
        name = _required_string(entry.get("name"), f"packages[{index}].name")
        version = _optional_string(entry.get("version"), f"packages[{index}].version")
        manager = _optional_string(entry.get("manager"), f"packages[{index}].manager")
        installed = entry.get("installed")
        if not isinstance(installed, bool):
            raise RollbackSnapshotError(f"rollback snapshot packages[{index}].installed must be a boolean")
        packages.append(PackageInfo(name=name, version=version, manager=manager, installed=installed))
    return packages


def _load_commands(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        raise RollbackSnapshotError("rollback snapshot commands must be an array")
    commands: list[list[str]] = []
    for index, command in enumerate(value):
        if not isinstance(command, list) or not command:
            raise RollbackSnapshotError(f"rollback snapshot commands[{index}] must be a non-empty array")
        if not all(isinstance(part, str) and part for part in command):
            raise RollbackSnapshotError(f"rollback snapshot commands[{index}] entries must be non-empty strings")
        if not _allowed_rollback_command(command):
            raise RollbackSnapshotError(f"rollback snapshot commands[{index}] is not a supported rollback command")
        commands.append(command)
    return commands


def _allowed_rollback_command(command: list[str]) -> bool:
    if command[:3] in (["apt-get", "install", "-y"], ["dnf", "downgrade", "-y"], ["yum", "downgrade", "-y"]):
        return _valid_package_specs(command[3:])
    if command[:4] == ["zypper", "--non-interactive", "install", "--oldpackage"]:
        return _valid_package_specs(command[4:])
    return False


def _valid_package_specs(specs: list[str]) -> bool:
    if not specs:
        return False
    return all(_valid_package_spec(spec) for spec in specs)


def _valid_package_spec(spec: str) -> bool:
    if spec.startswith("-"):
        return False
    return re.match(r"^[A-Za-z0-9][A-Za-z0-9.+_:-]*(?:=[A-Za-z0-9][A-Za-z0-9.+:~_-]*)?$", spec) is not None


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RollbackSnapshotError(f"rollback snapshot {name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RollbackSnapshotError(f"rollback snapshot {name} must be null or a non-empty string")
    return value


def apply_rollback(snapshot: RollbackSnapshot, runner: CommandRunner) -> list[CommandResult]:
    return [runner.run(command, mutate=True, allow_fail=True) for command in snapshot.commands]


def _rollback_commands(packages: list[PackageInfo], pm: str | None) -> list[list[str]]:
    installed = [pkg for pkg in packages if pkg.installed and pkg.version and _interesting_package(pkg.name)]
    if pm == "apt-get":
        specs = [f"{pkg.name}={pkg.version}" for pkg in installed if pkg.manager == "apt"]
        return [["apt-get", "install", "-y", *specs]] if specs else []
    if pm in {"dnf", "yum"}:
        specs = [f"{pkg.name}-{pkg.version}" for pkg in installed if pkg.manager == "rpm"]
        return [[pm, "downgrade", "-y", *specs]] if specs else []
    if pm == "zypper":
        specs = [f"{pkg.name}={pkg.version}" for pkg in installed if pkg.manager == "rpm"]
        return [["zypper", "--non-interactive", "install", "--oldpackage", *specs]] if specs else []
    return []
