from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeAlias

from .audit import _interesting_package, _parse_zypper_policy_selectors
from .dnf_module_transaction import (
    DNF_MODULE_FAILSAFE_DIRECTORY,
    dnf_module_enable_command,
    parse_dnf_module_enable_proof,
)
from .dnf_transaction import (
    DNF_LOCAL_TRANSACTION_SCRIPT,
    dnf_local_transaction_command,
)
from .models import (
    CommandResult,
    DesiredState,
    HostAudit,
    PackageInfo,
    PlanAction,
    RollbackSnapshot,
)
from .package_payloads import (
    PackagePayloadError,
    forward_package_command,
    local_payload_paths,
    validate_package_payloads,
)
from .planner import (
    _dnf_module_stream,
    lock_actions,
    package_install_operands,
    package_install_targets,
    package_policy_package_targets,
)
from .rollback import (
    _rpm_package_spec,
    _zypper_package_spec,
    rollback_package_commands,
)
from .runner import CommandRunner
from .xmlsafe import SafeXmlError, parse_bounded_xml

_DIAGNOSTIC_LIMIT = 512

_AptIdentity: TypeAlias = tuple[str, str, str]
_AptInstall: TypeAlias = tuple[str, str, str, str | None]
_AptRemoval: TypeAlias = tuple[str, str | None, str]
_RpmIdentity: TypeAlias = tuple[str, str, str | None, str]

_APT_INSTALL_PATTERN = re.compile(
    r"^Inst\s+(?P<package>\S+)"
    r"(?:\s+\[(?P<old>[^\]\s]+)\])?"
    r"\s+\((?P<version>\S+)(?:\s+.*)?"
    r"\s+\[(?P<architecture>[a-z0-9][a-z0-9-]*)\]\)$"
)
_APT_REMOVE_PATTERN = re.compile(
    r"^Remv\s+(?P<package>\S+)\s+\[(?P<version>[^\]\s]+)\](?:\s+.*)?$"
)
_APT_PACKAGE_TOKEN_PATTERN = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9+.-]*)(?::(?P<architecture>[a-z0-9][a-z0-9-]*))?$"
)

_TOOLKIT_PACKAGE_CLOSURE = {
    "libnvidia-container-tools",
    "libnvidia-container1",
    "nvidia-container-toolkit",
    "nvidia-container-toolkit-base",
}
_DOCKER_PACKAGE_CLOSURE = {
    "containerd.io",
    "docker-buildx-plugin",
    "docker-ce",
    "docker-ce-cli",
    "docker-ce-rootless-extras",
    "docker-compose-plugin",
}

_DNF_FORWARD_TRANSACTION_PROBE = r"""
import json
import sys

try:
    import dnf

    def record(package):
        epoch = str(package.epoch or "")
        version = str(package.version)
        release = str(package.release or "")
        if release:
            version += "-" + release
        return {
            "architecture": str(package.arch),
            "epoch": None if epoch in {"", "0", "None"} else epoch,
            "name": str(package.name),
            "version": version,
        }

    with dnf.Base() as base:
        base.conf.cacheonly = True
        base.conf.clean_requirements_on_remove = False
        base.conf.install_weak_deps = False
        base.read_all_repos()
        base.fill_sack()
        for spec in sys.argv[1:]:
            base.install(spec)
        base.resolve(allow_erasing=False)
        payload = {
            "install": sorted(
                (record(package) for package in base.transaction.install_set),
                key=lambda item: (
                    item["name"],
                    item["architecture"],
                    item["epoch"] or "",
                    item["version"],
                ),
            ),
            "remove": sorted(
                (record(package) for package in base.transaction.remove_set),
                key=lambda item: (
                    item["name"],
                    item["architecture"],
                    item["epoch"] or "",
                    item["version"],
                ),
            ),
        }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2)
""".strip()

class PackagePreflightError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        package_manager: str | None,
        packages: list[str],
        results: list[CommandResult],
    ) -> None:
        super().__init__(message)
        self.package_manager = package_manager
        self.packages = list(packages)
        self.results = list(results)

    def evidence(self) -> dict[str, Any]:
        return {
            "package_manager": self.package_manager,
            "packages": self.packages,
            "checks": [
                {
                    "command": result.command,
                    "returncode": result.returncode,
                    "diagnostic": _diagnostic(result),
                }
                for result in self.results
            ],
        }


def preflight_package_install(
    desired: DesiredState,
    audit: HostAudit,
    runner: CommandRunner,
) -> list[CommandResult]:
    """Resolve planned package targets without refreshing metadata or changing state."""
    package_manager = audit.package_manager
    packages = package_install_targets(desired, audit)
    operands = package_install_operands(desired, audit)
    if not package_manager or not packages:
        raise PackagePreflightError(
            "the package planner produced no repository targets to preflight",
            package_manager=package_manager,
            packages=packages,
            results=[],
        )

    results = (
        preflight_package_lock(desired, audit, runner)
        if desired.kernel_policy == "pin-compatible"
        else []
    )
    if package_manager == "apt-get":
        return [*results, *_preflight_apt(operands, packages, audit, runner)]
    if package_manager == "dnf":
        return [*results, *_preflight_dnf(operands, packages, audit, runner)]
    if package_manager == "zypper":
        return [*results, *_preflight_zypper(operands, packages, audit, runner)]
    raise PackagePreflightError(
        f"package availability preflight is not implemented for {package_manager}",
        package_manager=package_manager,
        packages=packages,
        results=[],
    )


def resolved_forward_payload_packages(
    package_manager: str,
    results: list[CommandResult],
) -> list[PackageInfo]:
    """Return the exact install identities proven by a successful preflight."""

    packages: set[tuple[str, str, str | None, str]] = set()
    if package_manager == "apt-get":
        for result in results:
            if (
                not result.command
                or result.command[0] != "apt-get"
                or "--simulate" not in result.command
            ):
                continue
            transaction = _parse_apt_transaction(result)
            if transaction is None:
                raise PackagePreflightError(
                    "APT forward payload transaction evidence is malformed",
                    package_manager=package_manager,
                    packages=[],
                    results=results,
                )
            installs, _ = transaction
            packages.update(
                (name, architecture, None, version)
                for name, architecture, version, _old in installs.values()
            )
        manager = "apt"
    elif package_manager in {"dnf", "yum"}:
        dnf_transactions = [
            dnf_parsed
            for result in results
            if (dnf_parsed := _parse_dnf_forward_transaction(result)) is not None
        ]
        if len(dnf_transactions) != 1:
            raise PackagePreflightError(
                "DNF forward payload transaction evidence is missing or ambiguous",
                package_manager=package_manager,
                packages=[],
                results=results,
            )
        packages.update(dnf_transactions[0][0])
        manager = "rpm"
    elif package_manager == "zypper":
        zypper_transactions = [
            zypper_parsed
            for result in results
            if (zypper_parsed := _parse_zypper_transaction(result)) is not None
        ]
        if len(zypper_transactions) != 1:
            raise PackagePreflightError(
                "Zypper forward payload transaction evidence is missing or ambiguous",
                package_manager=package_manager,
                packages=[],
                results=results,
            )
        for action, (name, edition, architecture) in zypper_transactions[0]:
            if action == "to-remove":
                continue
            epoch, version = _split_rpm_edition(edition)
            packages.add((name, architecture, epoch, version))
        manager = "rpm"
    else:
        raise PackagePreflightError(
            f"forward payload extraction is not implemented for {package_manager}",
            package_manager=package_manager,
            packages=[],
            results=results,
        )
    if not packages:
        raise PackagePreflightError(
            "forward preflight did not identify any package payloads",
            package_manager=package_manager,
            packages=[],
            results=results,
        )
    return [
        PackageInfo(
            name=name,
            version=version,
            manager=manager,
            installed=True,
            architecture=architecture,
            epoch=epoch,
        )
        for name, architecture, epoch, version in sorted(
            packages,
            key=lambda item: (item[0], item[1], item[2] or "", item[3]),
        )
    ]


def preflight_package_lock(
    desired: DesiredState,
    audit: HostAudit,
    runner: CommandRunner,
    *,
    actions: list[PlanAction] | None = None,
    authorized_failsafe_path: str | None = None,
) -> list[CommandResult]:
    """Prove the policy backend/selector is available before changing it."""
    package_manager = audit.package_manager
    if authorized_failsafe_path is not None and package_manager != "dnf":
        raise PackagePreflightError(
            "DNF module fail-safe authority was supplied for a different backend",
            package_manager=package_manager,
            packages=[],
            results=[],
        )
    planned_actions = lock_actions(desired, audit)
    if actions is not None and actions != planned_actions:
        raise PackagePreflightError(
            "package-policy execution no longer matches the fresh plan",
            package_manager=package_manager,
            packages=[],
            results=[],
        )
    selected_actions = planned_actions if actions is None else actions
    unsupported = [
        action
        for action in selected_actions
        if action.id.startswith("unsupported.")
    ]
    if unsupported:
        reason = unsupported[0].reason or unsupported[0].description
        raise PackagePreflightError(
            reason,
            package_manager=package_manager,
            packages=[],
            results=[],
        )
    if not selected_actions:
        return []
    if package_manager == "apt-get":
        desired_pin = (
            f"nvidia-driver-pinning-{desired.driver if desired.exact_driver_version else desired.driver_major}"
        )
        introduced = package_policy_package_targets(desired, audit)
        expected_removals = {
            selector.name
            for selector in audit.package_policy.selectors
            if selector.name.startswith("nvidia-driver-pinning-")
            and selector.name != desired_pin
        }
        packages = _policy_action_targets(selected_actions)
        if len(selected_actions) != 1 or len(selected_actions[0].commands) != 1:
            raise PackagePreflightError(
                "APT package-policy changes must resolve as one atomic transaction",
                package_manager=package_manager,
                packages=packages,
                results=[],
            )
        if desired_pin not in packages or any(
            f"{package}-" not in packages for package in expected_removals
        ):
            raise PackagePreflightError(
                "APT package-policy target does not match the planned transaction",
                package_manager=package_manager,
                packages=packages,
                results=[],
            )
        architecture_result, native_architecture = _apt_native_architecture(
            runner,
            packages=packages,
            purpose="package-policy",
        )
        command = selected_actions[0].commands[0]
        result = runner.run(
            [command[0], "--simulate", *command[1:]],
            mutate=False,
            allow_fail=True,
        )
        results = [architecture_result, result]
        transaction_error = _apt_policy_transaction_error(
            result,
            audit,
            desired_pin,
            bool(introduced),
            expected_removals,
            native_architecture,
        )
        if result.returncode != 0 or transaction_error is not None:
            raise PackagePreflightError(
                "APT could not safely resolve the package-policy transaction"
                + (
                    _diagnostic_suffix(result)
                    if result.returncode != 0
                    else f": {transaction_error}"
                ),
                package_manager=package_manager,
                packages=packages,
                results=results,
            )
        return results
    if package_manager == "dnf":
        module_stream = _dnf_module_stream(desired)
        target = f"nvidia-driver:{module_stream}"
        expected_apply = dnf_module_enable_command(
            apply=True,
            stream=module_stream,
        )
        if (
            len(selected_actions) != 1
            or selected_actions[0].id != "lock.rpm"
            or selected_actions[0].commands != [expected_apply]
        ):
            raise PackagePreflightError(
                "DNF module policy execution is not bound to the exact proven transaction",
                package_manager=package_manager,
                packages=[target],
                results=[],
            )
        result = runner.run(
            dnf_module_enable_command(apply=False, stream=module_stream),
            mutate=False,
            allow_fail=True,
        )
        proof = parse_dnf_module_enable_proof(
            result,
            applied=False,
            stream=module_stream,
        )
        if proof is None:
            raise PackagePreflightError(
                f"DNF module stream {target!r} lacks an exact dependency-closed cached-metadata proof"
                + (
                    _diagnostic_suffix(result)
                    if result.returncode != 0
                    else ": proof output is malformed, truncated, or expanded beyond the target stream"
                ),
                package_manager=package_manager,
                packages=[target],
                results=[result],
            )
        resolved_failsafe_path = (
            DNF_MODULE_FAILSAFE_DIRECTORY + "/" + proof.failsafe_filename
        )
        if (
            authorized_failsafe_path is not None
            and resolved_failsafe_path != authorized_failsafe_path
        ):
            raise PackagePreflightError(
                "fresh DNF module proof resolved a different rollback fail-safe target",
                package_manager=package_manager,
                packages=[target],
                results=[result],
            )
        if actions is not None:
            actions[0].commands = [
                dnf_module_enable_command(
                    apply=True,
                    stream=module_stream,
                    preflight_sha256=proof.preflight_sha256,
                )
            ]
        return [result]
    if package_manager == "zypper":
        result = runner.run(
            ["zypper", "--xmlout", "--non-interactive", "locks"],
            mutate=False,
            allow_fail=True,
        )
        selectors = _parse_zypper_policy_selectors(result)
        if selectors is None:
            raise PackagePreflightError(
                "Zypper could not inspect its package-lock policy backend"
                + _diagnostic_suffix(result),
                package_manager=package_manager,
                packages=[],
                results=[result],
            )
        if selectors != audit.package_policy.selectors:
            raise PackagePreflightError(
                "Zypper package-lock state changed after audit; rerun before applying",
                package_manager=package_manager,
                packages=[],
                results=[result],
            )
        return [result]
    raise PackagePreflightError(
        f"package-policy preflight is not implemented for {package_manager}",
        package_manager=package_manager,
        packages=[],
        results=[],
    )


def preflight_package_rollback(
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    runner: CommandRunner,
) -> list[CommandResult]:
    """Resolve every exact rollback package before any host mutation starts."""
    retained_payload_results = preflight_snapshot_restore_availability(
        snapshot,
        runner,
    )
    commands = rollback_package_commands(snapshot, audit)
    if not commands:
        return retained_payload_results
    package_manager = snapshot.package_manager
    restore_specs = [
        part
        for command in commands
        for part in _rollback_restore_specs(command)
    ]
    remove_specs = [
        part
        for command in commands
        for part in _rollback_remove_specs(command)
    ]
    packages = [*restore_specs, *remove_specs]
    if package_manager == "apt-get":
        if len(commands) != 1:
            raise PackagePreflightError(
                "APT rollback must resolve as one atomic solver transaction",
                package_manager=package_manager,
                packages=packages,
                results=[],
            )
        architecture_result, native_architecture = _apt_native_architecture(
            runner,
            packages=packages,
            purpose="exact rollback",
        )
        results = [*retained_payload_results, architecture_result]
        for command in commands:
            simulation = [command[0], "--simulate", *command[1:]]
            result = runner.run(simulation, mutate=False, allow_fail=True)
            results.append(result)
            transaction_error = _apt_rollback_transaction_error(
                result,
                snapshot,
                audit,
                restore_specs,
                remove_specs,
                native_architecture,
            )
            if result.returncode != 0 or transaction_error is not None:
                raise PackagePreflightError(
                    "APT could not safely resolve the exact rollback transaction"
                    + (
                        _diagnostic_suffix(result)
                        if result.returncode != 0
                        else f": {transaction_error}"
                    ),
                    package_manager=package_manager,
                    packages=packages,
                    results=results,
                )
        return results
    if package_manager in {"dnf", "yum"}:
        results = list(retained_payload_results)
        results.extend(
            _preflight_dnf_transaction(
                package_manager,
                restore_specs,
                remove_specs,
                commands,
                snapshot,
                audit,
                runner,
                packages,
                results,
            )
        )
        return results
    if package_manager == "zypper":
        if len(commands) != 1:
            raise PackagePreflightError(
                "Zypper rollback must resolve as one atomic solver transaction",
                package_manager=package_manager,
                packages=packages,
                results=[],
            )
        results = list(retained_payload_results)
        for command in commands:
            separator = command.index("--")
            simulation = [
                command[0],
                "--xmlout",
                *command[1:separator],
                "--dry-run",
                *command[separator:],
            ]
            result = runner.run(simulation, mutate=False, allow_fail=True)
            results.append(result)
            transaction_error = _zypper_rollback_transaction_error(
                result,
                snapshot,
                audit,
                restore_specs,
                remove_specs,
            )
            if result.returncode != 0 or transaction_error is not None:
                raise PackagePreflightError(
                    "Zypper could not resolve the exact rollback transaction"
                    + (
                        f": {transaction_error}"
                        if transaction_error
                        else _diagnostic_suffix(result)
                    ),
                    package_manager=package_manager,
                    packages=packages,
                    results=results,
                )
        return results
    raise PackagePreflightError(
        f"rollback preflight is not implemented for {package_manager}",
        package_manager=package_manager,
        packages=packages,
        results=[],
    )


def preflight_snapshot_restore_availability(
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
) -> list[CommandResult]:
    """Prove every exact baseline payload is retained and authenticated."""
    package_manager = snapshot.package_manager
    if (
        snapshot.path is None
        or snapshot.package_payloads is None
        or package_manager is None
    ):
        raise PackagePreflightError(
            "applicable rollback snapshot has no retained package payload bundle",
            package_manager=package_manager,
            packages=[],
            results=[],
        )
    try:
        validate_package_payloads(
            Path(snapshot.path),
            snapshot.package_payloads,
            snapshot.packages,
            package_manager,
            runner=runner,
        )
    except PackagePayloadError as exc:
        raise PackagePreflightError(
            f"retained rollback package payload validation failed: {exc}",
            package_manager=package_manager,
            packages=[
                _format_package_payload_identity(package)
                for package in snapshot.packages
            ],
            results=[],
        ) from exc
    return [CommandResult(["validate-package-payloads"], 0)]


def preflight_staged_forward_transaction(
    desired: DesiredState,
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    actions: list[PlanAction],
    runner: CommandRunner,
) -> list[CommandResult]:
    """Prove the exact staged local transaction that execution will use."""

    package_manager = snapshot.package_manager
    if (
        snapshot.path is None
        or snapshot.package_payloads is None
        or package_manager is None
    ):
        raise PackagePreflightError(
            "applicable install snapshot has no retained forward payload bundle",
            package_manager=package_manager,
            packages=[],
            results=[],
        )
    forward_entries = [
        entry
        for entry in snapshot.package_payloads.packages
        if "forward" in entry.roles
    ]
    forward_packages = [
        PackageInfo(
            name=entry.name,
            version=entry.version,
            manager="apt" if package_manager == "apt-get" else "rpm",
            installed=True,
            architecture=entry.architecture,
            epoch=entry.epoch,
        )
        for entry in forward_entries
    ]
    local_commands = [
        command
        for action in actions
        if action.id in {"install.packages", "lock.apt"}
        for command in action.commands
        if command
    ]
    if len(local_commands) != 1 or not forward_packages:
        raise PackagePreflightError(
            "staged forward payloads are not bound to one execution transaction",
            package_manager=package_manager,
            packages=[
                _format_package_payload_identity(package)
                for package in forward_packages
            ],
            results=[],
        )
    command = local_commands[0]
    try:
        validate_package_payloads(
            Path(snapshot.path),
            snapshot.package_payloads,
            snapshot.packages,
            package_manager,
            forward_packages=forward_packages,
            runner=runner,
        )
    except PackagePayloadError as exc:
        raise PackagePreflightError(
            f"staged forward package payload validation failed: {exc}",
            package_manager=package_manager,
            packages=[
                _format_package_payload_identity(package)
                for package in forward_packages
            ],
            results=[],
        ) from exc
    validation_result = CommandResult(["validate-forward-package-payloads"], 0)
    expected_paths = local_payload_paths(
        Path(snapshot.path),
        snapshot.package_payloads,
        role="forward",
    )
    forward_names = {package.name for package in forward_packages}
    if any(action.id == "install.packages" for action in actions):
        required_names = _forward_target_names(
            package_manager,
            package_install_targets(desired, audit),
            audit,
        )
        missing_names = sorted(required_names - forward_names)
        if missing_names:
            raise PackagePreflightError(
                "staged forward transaction omits fresh direct targets: "
                + ", ".join(missing_names),
                package_manager=package_manager,
                packages=sorted(required_names),
                results=[validation_result],
            )
    if any(action.id == "lock.apt" for action in actions):
        desired_pin = (
            "nvidia-driver-pinning-"
            + (
                desired.driver
                if desired.exact_driver_version
                else desired.driver_major
            )
        )
        if desired_pin not in forward_names:
            raise PackagePreflightError(
                "staged forward transaction omits the desired APT pin payload",
                package_manager=package_manager,
                packages=[desired_pin],
                results=[validation_result],
            )
    if package_manager == "apt-get":
        return [
            validation_result,
            *_preflight_local_apt_forward(
                command,
                forward_packages,
                expected_paths,
                audit,
                runner,
                prior_results=[validation_result],
            ),
        ]
    if package_manager in {"dnf", "yum"}:
        return [
            validation_result,
            *_preflight_local_dnf_forward(
                command,
                forward_packages,
                expected_paths,
                audit,
                runner,
                prior_results=[validation_result],
            ),
        ]
    if package_manager == "zypper":
        return [
            validation_result,
            *_preflight_local_zypper_forward(
                command,
                forward_packages,
                expected_paths,
                snapshot,
                audit,
                runner,
                prior_results=[validation_result],
            ),
        ]
    raise PackagePreflightError(
        f"local forward preflight is unsupported for {package_manager}",
        package_manager=package_manager,
        packages=sorted(forward_names),
        results=[validation_result],
    )


def _preflight_apt(
    operands: list[str],
    packages: list[str],
    audit: HostAudit,
    runner: CommandRunner,
) -> list[CommandResult]:
    architecture_result, native_architecture = _apt_native_architecture(
        runner,
        packages=packages,
        purpose="forward install",
    )
    command = [
        "apt-get",
        "--simulate",
        "install",
        "--allow-downgrades",
        "--no-install-recommends",
        *operands,
    ]
    result = runner.run(command, mutate=False, allow_fail=True)
    transaction_error = _apt_forward_transaction_error(
        result,
        native_architecture,
        packages,
        audit,
    )
    results = [architecture_result, result]
    if result.returncode != 0 or transaction_error is not None:
        raise PackagePreflightError(
            "APT could not resolve every planned package from current repository metadata"
            + (
                _diagnostic_suffix(result)
                if result.returncode != 0
                else f": {transaction_error}"
            ),
            package_manager="apt-get",
            packages=packages,
            results=results,
        )
    return results


def _preflight_dnf(
    operands: list[str],
    packages: list[str],
    audit: HostAudit,
    runner: CommandRunner,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for package in operands:
        result = runner.run(
            [
                "dnf",
                "-C",
                "-q",
                "repoquery",
                "--available",
                "--disable-modular-filtering",
                package,
            ],
            mutate=False,
            allow_fail=True,
        )
        results.append(result)
        if (
            result.returncode != 0
            or not result.stdout.strip()
            or _output_truncated(result)
        ):
            raise PackagePreflightError(
                f"DNF package target {package!r} is not available in cached repository metadata"
                + _diagnostic_suffix(result),
                package_manager="dnf",
                packages=packages,
                results=results,
            )
    result = runner.run(
        ["python3", "-I", "-c", _DNF_FORWARD_TRANSACTION_PROBE, *operands],
        mutate=False,
        allow_fail=True,
    )
    results.append(result)
    transaction_error = _dnf_forward_transaction_error(result, packages, audit)
    if result.returncode != 0 or transaction_error is not None:
        raise PackagePreflightError(
            "DNF could not safely resolve the cached forward package transaction"
            + (
                _diagnostic_suffix(result)
                if result.returncode != 0
                else f": {transaction_error}"
            ),
            package_manager="dnf",
            packages=packages,
            results=results,
        )
    return results


def _preflight_zypper(
    operands: list[str],
    packages: list[str],
    audit: HostAudit,
    runner: CommandRunner,
) -> list[CommandResult]:
    command = [
        "zypper",
        "--xmlout",
        "--non-interactive",
        "--no-refresh",
        "install",
        "--dry-run",
        "--no-recommends",
        *operands,
    ]
    result = runner.run(command, mutate=False, allow_fail=True)
    transaction_error = _zypper_forward_transaction_error(
        result,
        packages,
        audit,
    )
    if result.returncode != 0 or transaction_error is not None:
        raise PackagePreflightError(
            "Zypper could not resolve every planned package from cached repository metadata"
            + (
                _diagnostic_suffix(result)
                if result.returncode != 0
                else f": {transaction_error}"
            ),
            package_manager="zypper",
            packages=packages,
            results=[result],
        )
    return [result]


def _preflight_local_apt_forward(
    command: list[str],
    forward_packages: list[PackageInfo],
    expected_paths: list[str],
    audit: HostAudit,
    runner: CommandRunner,
    *,
    prior_results: list[CommandResult],
) -> list[CommandResult]:
    prefix = [
        "apt-get",
        "install",
        "-y",
        "--allow-change-held-packages",
        "--allow-downgrades",
        "--no-download",
        "--no-install-recommends",
        "--purge",
    ]
    operands = command[len(prefix) :] if command[: len(prefix)] == prefix else []
    restore_paths = [operand for operand in operands if operand.endswith(".deb")]
    remove_specs = [
        operand[:-1]
        for operand in operands
        if operand.endswith("-") and not operand.endswith(".deb-")
    ]
    if (
        not operands
        or len(restore_paths) + len(remove_specs) != len(operands)
        or restore_paths != expected_paths
        or len(remove_specs) != len(set(remove_specs))
    ):
        raise PackagePreflightError(
            "APT execution is not bound to the exact staged local payload transaction",
            package_manager="apt-get",
            packages=[package.name for package in forward_packages],
            results=prior_results,
        )
    architecture_result, native_architecture = _apt_native_architecture(
        runner,
        packages=[package.name for package in forward_packages],
        purpose="staged local forward",
    )
    result = runner.run(
        [command[0], "--simulate", *command[1:]],
        mutate=False,
        allow_fail=True,
    )
    transaction = _parse_apt_transaction(result)
    error: str | None = None
    if result.returncode == 0 and transaction is not None:
        installs, raw_removals = transaction
        expected_installs = {
            (
                package.name,
                package.architecture or native_architecture,
                package.version or "",
            )
            for package in forward_packages
        }
        observed_installs = {
            (name, architecture, version)
            for name, architecture, version, _old in installs.values()
        }
        current = {
            identity
            for package in audit.packages
            if (identity := _apt_identity(package, native_architecture)) is not None
        }
        current_by_slot = {identity[:2]: identity for identity in current}
        if observed_installs != expected_installs:
            error = "solver install set does not exactly match staged DEB identities"
        else:
            for name, architecture, _version, old_version in installs.values():
                current_identity = current_by_slot.get((name, architecture))
                if (current_identity is None) != (old_version is None) or (
                    current_identity is not None
                    and old_version != current_identity[2]
                ):
                    error = (
                        "solver replacement is not bound to the fresh exact APT inventory"
                    )
                    break
        expected_removals: set[_AptIdentity] = set()
        if error is None:
            for spec in remove_specs:
                parsed = _parse_apt_package_token(spec)
                if parsed is None:
                    error = "local APT removal operand is invalid"
                    break
                removal_name, removal_architecture = parsed
                matches = {
                    identity
                    for identity in current
                    if identity[0] == removal_name
                    and (
                        removal_architecture is None
                        or identity[1] == removal_architecture
                    )
                }
                if removal_architecture is None and len(matches) > 1:
                    matches = {
                        identity
                        for identity in matches
                        if identity[1] == native_architecture
                    }
                if len(matches) != 1:
                    error = "local APT removal is not bound to one fresh package identity"
                    break
                expected_removals.update(matches)
        observed_removals: set[_AptIdentity] = set()
        if error is None:
            for removal in raw_removals:
                identity = _resolve_apt_removal(
                    removal,
                    current,
                    native_architecture,
                )
                if identity is None:
                    error = "solver removal is not bound to fresh exact APT inventory"
                    break
                observed_removals.add(identity)
        if error is None and observed_removals != expected_removals:
            error = "solver removal set does not exactly match local APT policy removals"
    elif result.returncode == 0:
        error = "solver emitted malformed or truncated local APT transaction evidence"
    if result.returncode != 0 or error is not None:
        raise PackagePreflightError(
            "APT could not prove the exact staged local transaction"
            + (
                _diagnostic_suffix(result)
                if result.returncode != 0
                else f": {error}"
            ),
            package_manager="apt-get",
            packages=[package.name for package in forward_packages],
            results=[*prior_results, architecture_result, result],
        )
    return [architecture_result, result]


def _preflight_local_dnf_forward(
    command: list[str],
    forward_packages: list[PackageInfo],
    expected_paths: list[str],
    audit: HostAudit,
    runner: CommandRunner,
    *,
    prior_results: list[CommandResult],
) -> list[CommandResult]:
    expected_installs = sorted(
        _rpm_package_spec(package) for package in forward_packages
    )
    forward_slots = {
        (package.name, package.architecture) for package in forward_packages
    }
    forward_identities = {
        (package.name, package.architecture, package.epoch, package.version)
        for package in forward_packages
    }
    expected_removals = sorted(
        _rpm_package_spec(package)
        for package in audit.packages
        if package.installed
        and package.manager == "rpm"
        and package.version
        and (package.name, package.architecture) in forward_slots
        and (package.name, package.architecture, package.epoch, package.version)
        not in forward_identities
    )
    expected_apply = dnf_local_transaction_command(
        apply=True,
        restore_paths=expected_paths,
        remove_specs=[],
        expected_installs=expected_installs,
        expected_removals=expected_removals,
    )
    if command != expected_apply:
        raise PackagePreflightError(
            "DNF execution is not bound to the exact staged local transaction",
            package_manager="dnf",
            packages=expected_installs,
            results=prior_results,
        )
    result = runner.run(
        dnf_local_transaction_command(
            apply=False,
            restore_paths=expected_paths,
            remove_specs=[],
            expected_installs=expected_installs,
            expected_removals=expected_removals,
        ),
        mutate=False,
        allow_fail=True,
    )
    transaction = _parse_dnf_transaction(result)
    if (
        result.returncode != 0
        or transaction != (set(expected_installs), set(expected_removals))
    ):
        raise PackagePreflightError(
            "DNF could not prove the exact staged local transaction"
            + _diagnostic_suffix(result),
            package_manager="dnf",
            packages=expected_installs,
            results=[*prior_results, result],
        )
    return [result]


def _preflight_local_zypper_forward(
    command: list[str],
    forward_packages: list[PackageInfo],
    expected_paths: list[str],
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    runner: CommandRunner,
    *,
    prior_results: list[CommandResult],
) -> list[CommandResult]:
    assert snapshot.path is not None
    assert snapshot.package_payloads is not None
    expected_command = forward_package_command(
        Path(snapshot.path),
        snapshot.package_payloads,
        "zypper",
    )
    if command != expected_command or command[command.index("--") + 1 :] != expected_paths:
        raise PackagePreflightError(
            "Zypper execution is not bound to the exact staged local transaction",
            package_manager="zypper",
            packages=[package.name for package in forward_packages],
            results=prior_results,
        )
    separator = command.index("--")
    result = runner.run(
        [
            command[0],
            "--xmlout",
            *command[1:separator],
            "--dry-run",
            *command[separator:],
        ],
        mutate=False,
        allow_fail=True,
    )
    actions = _parse_zypper_transaction(result)
    expected = {
        _zypper_inventory_identity(package)
        for package in forward_packages
    }
    observed = {
        identity for action, identity in (actions or []) if action != "to-remove"
    }
    error = _zypper_forward_transaction_error(
        result,
        [package.name for package in forward_packages],
        audit,
    )
    if result.returncode != 0 or None in expected or observed != expected or error:
        raise PackagePreflightError(
            "Zypper could not prove the exact staged local transaction"
            + (
                _diagnostic_suffix(result)
                if result.returncode != 0
                else f": {error or 'solver action set differs from retained RPM identities'}"
            ),
            package_manager="zypper",
            packages=[package.name for package in forward_packages],
            results=[*prior_results, result],
        )
    return [result]


def _policy_action_targets(actions: list[PlanAction]) -> list[str]:
    targets: list[str] = []
    for action in actions:
        for command in action.commands:
            targets.extend(_command_packages(command))
    return list(dict.fromkeys(targets))


def _diagnostic_suffix(result: CommandResult) -> str:
    diagnostic = _diagnostic(result)
    return f": {diagnostic}" if diagnostic else f" (exit {result.returncode})"


def _diagnostic(result: CommandResult) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    if not output:
        return ""
    first_line = output.splitlines()[0].strip()
    if len(first_line) <= _DIAGNOSTIC_LIMIT:
        return first_line
    return first_line[: _DIAGNOSTIC_LIMIT - 1] + "…"


def _command_packages(command: list[str]) -> list[str]:
    if not command:
        return []
    if command[0] == "apt-get":
        for index, part in enumerate(command):
            if part in {"install", "remove", "purge"}:
                return [operand for operand in command[index + 1 :] if not operand.startswith("-")]
    if command[0] in {"dnf", "yum"}:
        for index, part in enumerate(command):
            if part in {
                "install",
                "install-nevra",
                "remove",
                "remove-n",
                "remove-nevra",
            }:
                return [operand for operand in command[index + 1 :] if not operand.startswith("-")]
    if command[:2] == ["zypper", "--non-interactive"]:
        for index, part in enumerate(command):
            if part in {"install", "remove"}:
                return [operand for operand in command[index + 1 :] if not operand.startswith("-")]
    return []


def _rollback_restore_specs(command: list[str]) -> list[str]:
    if command[:4] == [
        "python3",
        "-I",
        "-c",
        DNF_LOCAL_TRANSACTION_SCRIPT,
    ]:
        try:
            remove_marker = command.index("--remove", 5)
        except ValueError:
            return []
        return command[5:remove_marker]
    if command[:2] == ["apt-get", "install"]:
        return [
            operand
            for operand in command[2:]
            if not operand.startswith("-") and not operand.endswith("-")
        ]
    if command and command[0] in {"dnf", "yum"}:
        operation_name = (
            "install-nevra"
            if "install-nevra" in command
            else "install"
            if "install" in command
            else None
        )
        if operation_name is None:
            return []
        operation = command.index(operation_name)
        return [
            operand
            for operand in command[operation + 1 :]
            if not operand.startswith("-")
        ]
    if command[:2] == ["zypper", "--non-interactive"] and "--" in command:
        separator = command.index("--")
        return [
            operand
            for operand in command[separator + 1 :]
            if not operand.startswith("-")
        ]
    return []


def _rollback_remove_specs(command: list[str]) -> list[str]:
    if command[:4] == [
        "python3",
        "-I",
        "-c",
        DNF_LOCAL_TRANSACTION_SCRIPT,
    ]:
        try:
            remove_marker = command.index("--remove", 5)
            install_marker = command.index(
                "--expect-install",
                remove_marker + 1,
            )
        except ValueError:
            return []
        return command[remove_marker + 1 : install_marker]
    if command[:2] == ["apt-get", "install"]:
        return [
            operand[:-1]
            for operand in command[2:]
            if not operand.startswith("-") and operand.endswith("-")
        ]
    if command and command[0] in {"dnf", "yum"}:
        operations = {"remove", "remove-n", "remove-nevra"}
        operation = next(
            (index for index, part in enumerate(command) if part in operations),
            None,
        )
        if operation is None:
            return []
        return [
            operand
            for operand in command[operation + 1 :]
            if not operand.startswith("-")
        ]
    if command[:2] == ["zypper", "--non-interactive"] and "--" in command:
        separator = command.index("--")
        return [
            operand[1:]
            for operand in command[separator + 1 :]
            if operand.startswith("-")
        ]
    return []


def _package_name(spec: str) -> str:
    return spec.split("=", 1)[0].split(":", 1)[0]


def _snapshot_payload_operand(
    snapshot: RollbackSnapshot,
    package: PackageInfo,
) -> str:
    if snapshot.path is not None and snapshot.package_payloads is not None:
        identity = (
            package.name,
            package.architecture or "",
            package.epoch,
            package.version or "",
        )
        payload = next(
            (
                entry
                for entry in snapshot.package_payloads.packages
                if "baseline" in entry.roles
                and (
                    entry.name,
                    entry.architecture,
                    entry.epoch,
                    entry.version,
                )
                == identity
            ),
            None,
        )
        if payload is None:
            return ""
        return str(
            Path(snapshot.path).parent
            / snapshot.package_payloads.directory
            / payload.filename
        )
    if snapshot.package_manager == "apt-get":
        return _apt_restore_spec(package) or ""
    if snapshot.package_manager in {"dnf", "yum"}:
        return _rpm_package_spec(package)
    if snapshot.package_manager == "zypper":
        return _zypper_package_spec(package)
    return ""


def _format_package_payload_identity(package: PackageInfo) -> str:
    epoch = f"{package.epoch}:" if package.epoch else ""
    architecture = package.architecture or "unknown"
    return f"{package.name}:{architecture}={epoch}{package.version or 'unknown'}"


def _apt_unapproved_removals(
    result: CommandResult,
    allowed: set[str] | None = None,
) -> list[str]:
    removed = _apt_observed_removals(result)
    if allowed is not None:
        return sorted(removed - allowed)
    return sorted(package for package in removed if not _interesting_package(package))


def _apt_observed_removals(result: CommandResult) -> set[str]:
    return {
        _package_name(match.group(1))
        for line in result.stdout.splitlines()
        if (match := re.match(r"^Remv\s+(\S+)", line))
    }


def _output_truncated(result: CommandResult) -> bool:
    return "[output truncated:" in result.stdout or "[output truncated:" in result.stderr


def _apt_native_architecture(
    runner: CommandRunner,
    *,
    packages: list[str],
    purpose: str,
) -> tuple[CommandResult, str]:
    result = runner.run(
        ["dpkg", "--print-architecture"],
        mutate=False,
        allow_fail=True,
    )
    architecture = result.stdout.strip()
    if (
        result.returncode != 0
        or _output_truncated(result)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]*", architecture) is None
    ):
        raise PackagePreflightError(
            f"APT could not bind the {purpose} transaction to the native architecture"
            + _diagnostic_suffix(result),
            package_manager="apt-get",
            packages=packages,
            results=[result],
        )
    return result, architecture


def _parse_apt_transaction(
    result: CommandResult,
) -> tuple[dict[tuple[str, str], _AptInstall], list[_AptRemoval]] | None:
    if _output_truncated(result):
        return None
    installs: dict[tuple[str, str], _AptInstall] = {}
    removals: list[_AptRemoval] = []
    seen_removals: set[_AptRemoval] = set()
    for line in result.stdout.splitlines():
        if line.startswith("Inst "):
            match = _APT_INSTALL_PATTERN.fullmatch(line)
            if match is None:
                return None
            package = _parse_apt_package_token(match.group("package"))
            if package is None:
                return None
            name, explicit_architecture = package
            architecture = match.group("architecture")
            if (
                explicit_architecture is not None
                and explicit_architecture != architecture
            ):
                return None
            install = (
                name,
                architecture,
                match.group("version"),
                match.group("old"),
            )
            slot = (name, architecture)
            if slot in installs:
                return None
            installs[slot] = install
        elif line.startswith("Remv "):
            match = _APT_REMOVE_PATTERN.fullmatch(line)
            if match is None:
                return None
            package = _parse_apt_package_token(match.group("package"))
            if package is None:
                return None
            removal = (package[0], package[1], match.group("version"))
            if removal in seen_removals:
                return None
            seen_removals.add(removal)
            removals.append(removal)
    return installs, removals


def _parse_apt_package_token(token: str) -> tuple[str, str | None] | None:
    match = _APT_PACKAGE_TOKEN_PATTERN.fullmatch(token)
    if match is None:
        return None
    return match.group("name"), match.group("architecture")


def _resolve_apt_removal(
    removal: _AptRemoval,
    candidates: set[_AptIdentity],
    native_architecture: str,
) -> _AptIdentity | None:
    name, explicit_architecture, version = removal
    matches = {
        identity
        for identity in candidates
        if identity[0] == name
        and identity[2] == version
        and (
            explicit_architecture is None
            or identity[1] == explicit_architecture
        )
    }
    if len(matches) == 1:
        return next(iter(matches))
    if explicit_architecture is not None:
        return None
    native_matches = {
        identity for identity in matches if identity[1] == native_architecture
    }
    return next(iter(native_matches)) if len(native_matches) == 1 else None


def _apt_identity(package: Any, native_architecture: str) -> _AptIdentity | None:
    if not package.installed or package.manager != "apt" or not package.version:
        return None
    return (
        package.name,
        package.architecture or native_architecture,
        package.version,
    )


def _apt_restore_spec(package: Any) -> str | None:
    if not package.installed or package.manager != "apt" or not package.version:
        return None
    architecture = f":{package.architecture}" if package.architecture else ""
    return f"{package.name}{architecture}={package.version}"


def _apt_remove_spec(package: Any) -> str | None:
    if not package.installed or package.manager != "apt":
        return None
    architecture = f":{package.architecture}" if package.architecture else ""
    return f"{package.name}{architecture}"


def _apt_rollback_transaction_error(
    result: CommandResult,
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    restore_specs: list[str],
    remove_specs: list[str],
    native_architecture: str,
) -> str | None:
    transaction = _parse_apt_transaction(result)
    if transaction is None:
        return "machine-readable transaction actions are malformed or truncated"
    installs, raw_removals = transaction

    snapshot_by_spec: dict[str, _AptIdentity] = {}
    for package in snapshot.packages:
        spec = _snapshot_payload_operand(snapshot, package)
        identity = _apt_identity(package, native_architecture)
        if not spec or identity is None:
            continue
        if spec in snapshot_by_spec:
            return "the package baseline contains duplicate APT restore identities"
        snapshot_by_spec[spec] = identity
    audit_by_remove_spec: dict[str, _AptIdentity] = {}
    current_identities: set[_AptIdentity] = set()
    current_by_slot: dict[tuple[str, str], _AptIdentity] = {}
    for package in audit.packages:
        remove_spec = _apt_remove_spec(package)
        identity = _apt_identity(package, native_architecture)
        if remove_spec is None or identity is None:
            continue
        slot = identity[:2]
        if remove_spec in audit_by_remove_spec or slot in current_by_slot:
            return "the current APT inventory contains duplicate package slots"
        audit_by_remove_spec[remove_spec] = identity
        current_identities.add(identity)
        current_by_slot[slot] = identity

    unknown_restores = sorted(set(restore_specs) - set(snapshot_by_spec))
    unknown_removals = sorted(set(remove_specs) - set(audit_by_remove_spec))
    if unknown_restores or unknown_removals:
        return "transaction operands are not bound to exact APT package inventory"
    expected_installs = {snapshot_by_spec[spec] for spec in restore_specs}
    expected_removals = {audit_by_remove_spec[spec] for spec in remove_specs}

    observed_installs: set[_AptIdentity] = set()
    for name, architecture, version, old_version in installs.values():
        identity = (name, architecture, version)
        current = current_by_slot.get((name, architecture))
        if current is None and old_version is not None:
            return (
                "solver reported an unobserved replaced package: "
                + _format_apt_identity((name, architecture, old_version))
            )
        if current is not None and old_version != current[2]:
            return (
                "solver replacement does not match the current exact package: "
                + _format_apt_identity(current)
            )
        observed_installs.add(identity)

    observed_removals: set[_AptIdentity] = set()
    for removal in raw_removals:
        identity = _resolve_apt_removal(
            removal,
            current_identities,
            native_architecture,
        )
        if identity is None:
            return "solver removal is not bound to one current exact package identity"
        observed_removals.add(identity)

    unexpected_installs = observed_installs - expected_installs
    missing_installs = expected_installs - observed_installs
    unexpected_removals = observed_removals - expected_removals
    missing_removals = expected_removals - observed_removals
    if unexpected_installs:
        return "solver would install a package outside the exact baseline: " + ", ".join(
            sorted(_format_apt_identity(identity) for identity in unexpected_installs)
        )
    if missing_installs:
        return "solver omitted an exact baseline restore: " + ", ".join(
            sorted(_format_apt_identity(identity) for identity in missing_installs)
        )
    if unexpected_removals:
        return "solver would remove a package outside the exact rollback set: " + ", ".join(
            sorted(_format_apt_identity(identity) for identity in unexpected_removals)
        )
    if missing_removals:
        return "solver omitted an exact rollback removal: " + ", ".join(
            sorted(_format_apt_identity(identity) for identity in missing_removals)
        )
    return None


def _apt_policy_transaction_error(
    result: CommandResult,
    audit: HostAudit,
    desired_pin: str,
    require_install: bool,
    expected_removal_names: set[str],
    native_architecture: str,
) -> str | None:
    transaction = _parse_apt_transaction(result)
    if transaction is None:
        return "transaction actions are malformed or truncated"
    installs, raw_removals = transaction
    current_identities = {
        identity
        for package in audit.packages
        if (identity := _apt_identity(package, native_architecture)) is not None
    }
    current_by_slot = {identity[:2]: identity for identity in current_identities}
    if len(current_by_slot) != len(current_identities):
        return "the current APT inventory contains duplicate package slots"
    expected_removals = {
        identity
        for identity in current_identities
        if identity[0] in expected_removal_names
    }
    if {identity[0] for identity in expected_removals} != expected_removal_names:
        return "observed pin selectors are not bound to exact package inventory"

    observed_installs: set[_AptIdentity] = set()
    for name, architecture, version, old_version in installs.values():
        identity = (name, architecture, version)
        if name != desired_pin:
            return (
                "solver would install a package outside the desired pin transaction: "
                + _format_apt_identity(identity)
            )
        current = current_by_slot.get((name, architecture))
        if current is None and old_version is not None:
            return "solver reported an unobserved desired-pin replacement"
        if current is not None and old_version != current[2]:
            return "solver desired-pin replacement does not match exact inventory"
        observed_installs.add(identity)

    observed_removals: set[_AptIdentity] = set()
    for removal in raw_removals:
        identity = _resolve_apt_removal(
            removal,
            current_identities,
            native_architecture,
        )
        if identity is None:
            return "solver removal is not bound to one current exact pin identity"
        observed_removals.add(identity)

    if require_install and len(observed_installs) != 1:
        return "solver omitted the desired pin-package installation"
    if not require_install and observed_installs:
        return "solver unexpectedly replaced the already-installed desired pin"
    unexpected_removals = observed_removals - expected_removals
    missing_removals = expected_removals - observed_removals
    if unexpected_removals:
        return "solver would remove a package outside the conflicting pin set: " + ", ".join(
            sorted(_format_apt_identity(identity) for identity in unexpected_removals)
        )
    if missing_removals:
        return "solver omitted an exact conflicting pin removal: " + ", ".join(
            sorted(_format_apt_identity(identity) for identity in missing_removals)
        )
    return None


def _apt_forward_transaction_error(
    result: CommandResult,
    native_architecture: str,
    packages: list[str],
    audit: HostAudit,
) -> str | None:
    transaction = _parse_apt_transaction(result)
    if transaction is None:
        return "transaction actions are malformed or truncated"
    installs, raw_removals = transaction
    current_identities = {
        identity
        for package in audit.packages
        if (identity := _apt_identity(package, native_architecture)) is not None
    }
    current_by_slot: dict[tuple[str, str], _AptIdentity] = {}
    for identity in current_identities:
        slot = identity[:2]
        if slot in current_by_slot:
            return "the current APT inventory contains duplicate package slots"
        current_by_slot[slot] = identity

    install_slots = set(installs)
    observed_names: set[str] = set()
    for name, architecture, version, old_version in installs.values():
        if not _forward_package_allowed(name, packages, "apt-get", audit):
            return (
                "solver dependency expansion is outside the rollback-tracked target closure: "
                + _format_apt_identity((name, architecture, version))
            )
        current = current_by_slot.get((name, architecture))
        if current is None and old_version is not None:
            return (
                "solver reported an unobserved package replacement: "
                + _format_apt_identity((name, architecture, old_version))
            )
        if current is not None and old_version != current[2]:
            return (
                "solver replacement does not match the audited exact package: "
                + _format_apt_identity(current)
            )
        observed_names.add(name)

    for removal in raw_removals:
        identity = _resolve_apt_removal(
            removal,
            current_identities,
            native_architecture,
        )
        if identity is None:
            return "solver removal is not bound to one audited exact package identity"
        if identity[:2] not in install_slots:
            return (
                "solver would remove a package without a planned replacement: "
                + _format_apt_identity(identity)
            )
        if not _forward_package_allowed(identity[0], packages, "apt-get", audit):
            return (
                "solver would replace a package outside the target closure: "
                + _format_apt_identity(identity)
            )

    missing_targets = _forward_target_names("apt-get", packages, audit) - observed_names
    if missing_targets:
        return "solver omitted a direct planned target action: " + ", ".join(
            sorted(missing_targets)
        )
    return None


def _format_apt_identity(identity: _AptIdentity) -> str:
    name, architecture, version = identity
    return f"{name}:{architecture}={version}"


def _forward_target_names(
    package_manager: str,
    packages: list[str],
    audit: HostAudit,
) -> set[str]:
    names: set[str] = set()
    for spec in packages:
        name = re.split(r"[<>=]", spec, maxsplit=1)[0]
        if (
            package_manager in {"dnf", "yum"}
            and name == f"kernel-devel-{audit.kernel.running}"
        ):
            name = "kernel-devel"
        names.add(name)
    return names


def _forward_package_allowed(
    name: str,
    packages: list[str],
    package_manager: str,
    audit: HostAudit,
) -> bool:
    targets = _forward_target_names(package_manager, packages, audit)
    if not _forward_package_observable(name, package_manager, audit, targets):
        return False
    if name in targets:
        return True
    if targets & {"cuda-drivers", "nvidia-open"} and _driver_package_dependency(
        name
    ):
        return True
    if "nvidia-container-toolkit" in targets and name in _TOOLKIT_PACKAGE_CLOSURE:
        return True
    if "docker-ce" in targets and name in _DOCKER_PACKAGE_CLOSURE:
        return True
    if any(target.startswith("linux-headers-") for target in targets):
        return name.startswith("linux-headers-")
    if targets & {"kernel-devel", "kernel-headers"}:
        return name in {"kernel-devel", "kernel-headers"} or name.startswith(
            "kernel-devel-"
        )
    if any(
        re.fullmatch(r"kernel-(?:default|azure|64k)-devel", target)
        for target in targets
    ):
        return re.fullmatch(r"kernel-(?:default|azure|64k)-devel", name) is not None
    return False


def _forward_package_observable(
    name: str,
    package_manager: str,
    audit: HostAudit,
    targets: set[str],
) -> bool:
    if not _interesting_package(name):
        return False
    if package_manager != "apt-get":
        return True
    return bool(
        name.startswith(("nvidia-", "libnvidia-", "cuda-", "docker-ce"))
        or name
        in {
            "build-essential",
            "containerd.io",
            "docker-buildx-plugin",
            "docker-compose-plugin",
            "nvidia-container-toolkit",
        }
        or name == f"linux-headers-{audit.kernel.running}"
        or name in targets
    )


def _driver_package_dependency(name: str) -> bool:
    if (
        name in _TOOLKIT_PACKAGE_CLOSURE
        or name.startswith(
            (
                "cuda-compat-",
                "cuda-drivers-fabricmanager",
                "cuda-toolkit",
                "libnvidia-container",
                "nvidia-container",
                "nvidia-fabricmanager",
            )
        )
    ):
        return False
    return bool(
        name in {"cuda-drivers", "nvidia-open"}
        or name.startswith(("cuda-drivers-", "nvidia-", "libnvidia-"))
    )


def _dnf_forward_transaction_error(
    result: CommandResult,
    packages: list[str],
    audit: HostAudit,
) -> str | None:
    transaction = _parse_dnf_forward_transaction(result)
    if transaction is None:
        return "solver emitted malformed or truncated transaction evidence"
    installs, removals = transaction
    current = {
        identity
        for package in audit.packages
        if (identity := _rpm_identity(package)) is not None
    }
    install_slots = {(identity[0], identity[1]) for identity in installs}
    observed_names: set[str] = set()
    for identity in installs:
        if not _forward_package_allowed(identity[0], packages, "dnf", audit):
            return (
                "solver dependency expansion is outside the rollback-tracked target closure: "
                + _format_rpm_identity(identity)
            )
        observed_names.add(identity[0])
    for identity in removals:
        if identity not in current:
            return (
                "solver removal is not bound to the audited exact package inventory: "
                + _format_rpm_identity(identity)
            )
        if (identity[0], identity[1]) not in install_slots:
            return (
                "solver would remove a package without a planned replacement: "
                + _format_rpm_identity(identity)
            )
        if not _forward_package_allowed(identity[0], packages, "dnf", audit):
            return (
                "solver would replace a package outside the target closure: "
                + _format_rpm_identity(identity)
            )
    missing_targets = _forward_target_names("dnf", packages, audit) - observed_names
    if missing_targets:
        return "solver omitted a direct planned target action: " + ", ".join(
            sorted(missing_targets)
        )
    return None


def _parse_dnf_forward_transaction(
    result: CommandResult,
) -> tuple[set[_RpmIdentity], set[_RpmIdentity]] | None:
    payload = _strict_json_object(result)
    if payload is None or set(payload) != {"install", "remove"}:
        return None
    install = _parse_rpm_identity_records(payload["install"])
    remove = _parse_rpm_identity_records(payload["remove"])
    if install is None or remove is None:
        return None
    return install, remove


def _strict_json_object(result: CommandResult) -> dict[str, Any] | None:
    if _output_truncated(result):
        return None

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(result.stdout, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_rpm_identity_records(value: Any) -> set[_RpmIdentity] | None:
    if not isinstance(value, list):
        return None
    identities: set[_RpmIdentity] = set()
    slots: set[tuple[str, str]] = set()
    for record in value:
        if not isinstance(record, dict) or set(record) != {
            "architecture",
            "epoch",
            "name",
            "version",
        }:
            return None
        name = record["name"]
        architecture = record["architecture"]
        epoch = record["epoch"]
        version = record["version"]
        if (
            not isinstance(name, str)
            or not isinstance(architecture, str)
            or not isinstance(version, str)
            or not all(
                _valid_transaction_field(field)
                for field in (name, architecture, version)
            )
            or (epoch is not None and (not isinstance(epoch, str) or not epoch.isdigit()))
        ):
            return None
        identity = (name, architecture, epoch, version)
        slot = (name, architecture)
        if identity in identities or slot in slots:
            return None
        identities.add(identity)
        slots.add(slot)
    return identities


def _valid_transaction_field(value: str) -> bool:
    return bool(value) and value == value.strip() and not any(
        character.isspace() for character in value
    )


def _rpm_identity(package: Any) -> _RpmIdentity | None:
    if (
        not package.installed
        or package.manager != "rpm"
        or not package.version
        or not package.architecture
    ):
        return None
    return (
        package.name,
        package.architecture,
        package.epoch,
        package.version,
    )


def _format_rpm_identity(identity: _RpmIdentity) -> str:
    name, architecture, epoch, version = identity
    epoch_prefix = f"{epoch}:" if epoch else ""
    return f"{name}-{epoch_prefix}{version}.{architecture}"


def _zypper_forward_transaction_error(
    result: CommandResult,
    packages: list[str],
    audit: HostAudit,
) -> str | None:
    actions = _parse_zypper_transaction(result)
    if actions is None:
        return "machine-readable transaction summary is missing or malformed"
    current: set[tuple[str, str, str]] = set()
    current_by_slot: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for package in audit.packages:
        identity = _zypper_inventory_identity(package)
        if identity is None:
            continue
        current.add(identity)
        current_by_slot.setdefault((identity[0], identity[2]), set()).add(identity)

    changed_slots = {
        (identity[0], identity[2])
        for action, identity in actions
        if action != "to-remove"
    }
    observed_names: set[str] = set()
    for action, identity in actions:
        name, _edition, architecture = identity
        if not _forward_package_allowed(name, packages, "zypper", audit):
            kind = "replace" if action != "to-install" else "install"
            return (
                f"solver would {kind} a package outside the rollback-tracked target closure: "
                + _format_zypper_identity(identity)
            )
        if action == "to-remove":
            if identity not in current:
                return (
                    "solver removal is not bound to the audited exact package inventory: "
                    + _format_zypper_identity(identity)
                )
            if (name, architecture) not in changed_slots:
                return (
                    "solver would remove a package without a planned replacement: "
                    + _format_zypper_identity(identity)
                )
            continue
        if action in {
            "to-change-arch",
            "to-downgrade-change-arch",
            "to-upgrade-change-arch",
        }:
            return (
                "solver architecture replacement does not expose the exact removed identity: "
                + _format_zypper_identity(identity)
            )
        slot_inventory = current_by_slot.get((name, architecture), set())
        if action == "to-reinstall":
            if identity not in current:
                return (
                    "solver reinstall is not bound to the audited exact package inventory: "
                    + _format_zypper_identity(identity)
                )
        elif action in {"to-upgrade", "to-downgrade"} and len(slot_inventory) != 1:
            return (
                "solver replacement is not bound to one audited package slot: "
                + _format_zypper_identity(identity)
            )
        observed_names.add(name)

    missing_targets = _forward_target_names("zypper", packages, audit) - observed_names
    if missing_targets:
        return "solver omitted a direct planned target action: " + ", ".join(
            sorted(missing_targets)
        )
    return None


def _zypper_inventory_identity(package: Any) -> tuple[str, str, str] | None:
    if (
        not package.installed
        or package.manager != "rpm"
        or not package.version
        or not package.architecture
    ):
        return None
    epoch = f"{package.epoch}:" if package.epoch else ""
    return package.name, f"{epoch}{package.version}", package.architecture


def _preflight_dnf_transaction(
    package_manager: str,
    restore_specs: list[str],
    remove_specs: list[str],
    execution_commands: list[list[str]],
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    runner: CommandRunner,
    packages: list[str],
    prior_results: list[CommandResult],
) -> list[CommandResult]:
    current_by_nevra = {
        _rpm_package_spec(package): package
        for package in audit.packages
        if package.installed and package.manager == "rpm"
    }
    unknown_removals = sorted(set(remove_specs) - set(current_by_nevra))
    if unknown_removals:
        raise PackagePreflightError(
            "RPM rollback removal set is not bound to the current exact package inventory: "
            + ", ".join(unknown_removals),
            package_manager=package_manager,
            packages=packages,
            results=prior_results,
        )

    baseline_by_operand = {
        _snapshot_payload_operand(snapshot, package): package
        for package in snapshot.packages
        if package.installed and package.manager == "rpm"
    }
    unknown_restores = sorted(set(restore_specs) - set(baseline_by_operand))
    if unknown_restores:
        raise PackagePreflightError(
            "RPM rollback restore set is not bound to the exact package baseline: "
            + ", ".join(unknown_restores),
            package_manager=package_manager,
            packages=packages,
            results=prior_results,
        )

    restore_slots = {
        (baseline_by_operand[spec].name, baseline_by_operand[spec].architecture)
        for spec in restore_specs
    }
    expected_installs = {
        _rpm_package_spec(baseline_by_operand[spec]) for spec in restore_specs
    }
    replacement_specs = {
        spec
        for spec, package in current_by_nevra.items()
        if (package.name, package.architecture) in restore_slots
        and spec not in expected_installs
    }
    expected_removals = {*remove_specs, *replacement_specs}

    expected_apply_command = dnf_local_transaction_command(
        apply=True,
        restore_paths=restore_specs,
        remove_specs=remove_specs,
        expected_installs=sorted(expected_installs),
        expected_removals=sorted(expected_removals),
    )
    if execution_commands != [expected_apply_command]:
        raise PackagePreflightError(
            "RPM rollback execution is not bound to the exact proven transaction",
            package_manager=package_manager,
            packages=packages,
            results=prior_results,
        )
    result = runner.run(
        dnf_local_transaction_command(
            apply=False,
            restore_paths=restore_specs,
            remove_specs=remove_specs,
            expected_installs=sorted(expected_installs),
            expected_removals=sorted(expected_removals),
        ),
        mutate=False,
        allow_fail=True,
    )
    results = [result]
    transaction = _parse_dnf_transaction(result)
    if result.returncode != 0 or transaction is None:
        detail = (
            _diagnostic_suffix(result)
            if result.returncode != 0
            else ": solver emitted malformed or truncated transaction evidence"
        )
        raise PackagePreflightError(
            "RPM rollback could not resolve the exact retained local transaction"
            + detail,
            package_manager=package_manager,
            packages=packages,
            results=[*prior_results, *results],
        )

    observed_installs, observed_removals = transaction
    unexpected_installs = sorted(observed_installs - expected_installs)
    missing_installs = sorted(expected_installs - observed_installs)
    unexpected_removals = sorted(observed_removals - expected_removals)
    missing_removals = sorted(expected_removals - observed_removals)
    if (
        unexpected_installs
        or missing_installs
        or unexpected_removals
        or missing_removals
    ):
        if unexpected_installs:
            detail = "; solver would install packages outside the exact baseline: " + ", ".join(
                unexpected_installs
            )
        elif missing_installs:
            detail = "; solver omitted exact baseline restores: " + ", ".join(
                missing_installs
            )
        elif unexpected_removals:
            detail = "; solver would remove packages outside the exact rollback set: " + ", ".join(
                unexpected_removals
            )
        else:
            detail = "; solver omitted exact rollback removals: " + ", ".join(
                missing_removals
            )
        raise PackagePreflightError(
            "RPM rollback cannot safely apply the resolved transaction" + detail,
            package_manager=package_manager,
            packages=packages,
            results=[*prior_results, *results],
        )
    return results


def _parse_dnf_transaction(
    result: CommandResult,
) -> tuple[set[str], set[str]] | None:
    if (
        "[output truncated:" in result.stdout
        or "[output truncated:" in result.stderr
    ):
        return None

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            result.stdout,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"install", "remove"}:
        return None
    install = payload["install"]
    remove = payload["remove"]
    if not isinstance(install, list) or not isinstance(remove, list):
        return None
    if not all(isinstance(spec, str) and spec for spec in [*install, *remove]):
        return None
    if len(set(install)) != len(install) or len(set(remove)) != len(remove):
        return None
    return set(install), set(remove)


_ZYPPER_CHANGE_ACTIONS = {
    "to-change-arch",
    "to-downgrade",
    "to-downgrade-change-arch",
    "to-install",
    "to-reinstall",
    "to-upgrade",
    "to-upgrade-change-arch",
}


def _zypper_rollback_transaction_error(
    result: CommandResult,
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    restore_specs: list[str],
    remove_specs: list[str],
) -> str | None:
    actions = _parse_zypper_transaction(result)
    if actions is None:
        return "machine-readable transaction summary is missing or malformed"
    snapshot_by_spec = {
        _snapshot_payload_operand(snapshot, package): package
        for package in snapshot.packages
        if package.installed
        and package.manager == "rpm"
        and package.version
        and package.architecture
    }
    audit_by_spec = {
        _zypper_package_spec(package): package
        for package in audit.packages
        if package.installed
        and package.manager == "rpm"
        and package.version
        and package.architecture
    }
    unknown_restores = sorted(set(restore_specs) - set(snapshot_by_spec))
    unknown_removals = sorted(set(remove_specs) - set(audit_by_spec))
    if unknown_restores or unknown_removals:
        return "transaction operands are not bound to exact package inventory"
    restore_targets = {
        (
            snapshot_by_spec[spec].name,
            (
                f"{snapshot_by_spec[spec].epoch}:"
                if snapshot_by_spec[spec].epoch
                else ""
            )
            + str(snapshot_by_spec[spec].version),
            str(snapshot_by_spec[spec].architecture),
        )
        for spec in restore_specs
    }
    removal_targets = {
        (
            audit_by_spec[spec].name,
            (
                f"{audit_by_spec[spec].epoch}:"
                if audit_by_spec[spec].epoch
                else ""
            )
            + str(audit_by_spec[spec].version),
            str(audit_by_spec[spec].architecture),
        )
        for spec in remove_specs
    }
    restore_slots = {(name, arch) for name, _edition, arch in restore_targets}
    replacement_removals = {
        (
            package.name,
            (f"{package.epoch}:" if package.epoch else "")
            + str(package.version),
            str(package.architecture),
        )
        for package in audit.packages
        if package.installed
        and package.version
        and package.architecture
        and (package.name, package.architecture) in restore_slots
    }
    observed_restores: set[tuple[str, str, str]] = set()
    observed_removals: set[tuple[str, str, str]] = set()
    for action, identity in actions:
        if action == "to-remove":
            if identity not in removal_targets | replacement_removals:
                return (
                    "solver would remove a package outside the exact rollback set: "
                    + _format_zypper_identity(identity)
                )
            observed_removals.add(identity)
        elif action in _ZYPPER_CHANGE_ACTIONS:
            if identity not in restore_targets:
                return (
                    "solver would change a package outside the exact baseline set: "
                    + _format_zypper_identity(identity)
                )
            observed_restores.add(identity)
        else:
            return f"solver reported unsupported action {action!r}"
    missing_restores = restore_targets - observed_restores
    missing_removals = removal_targets - observed_removals
    if missing_restores:
        return "solver omitted exact baseline restore: " + ", ".join(
            sorted(_format_zypper_identity(identity) for identity in missing_restores)
        )
    if missing_removals:
        return "solver omitted exact removal: " + ", ".join(
            sorted(_format_zypper_identity(identity) for identity in missing_removals)
        )
    return None


def _parse_zypper_transaction(
    result: CommandResult,
) -> list[tuple[str, tuple[str, str, str]]] | None:
    if (
        "[output truncated:" in result.stdout
        or "[output truncated:" in result.stderr
    ):
        return None
    try:
        root = parse_bounded_xml(result.stdout)
    except SafeXmlError:
        return None
    if root.tag != "stream":
        return None
    summaries = root.findall("install-summary")
    if len(summaries) != 1:
        return None
    summary = summaries[0]
    try:
        packages_to_change = int(summary.attrib["packages-to-change"])
    except (KeyError, ValueError):
        return None
    actions: list[tuple[str, tuple[str, str, str]]] = []
    seen: set[tuple[str, tuple[str, str, str]]] = set()
    for group in summary:
        if group.tag not in _ZYPPER_CHANGE_ACTIONS | {"to-remove"}:
            return None
        if not list(group):
            return None
        for solvable in group:
            if solvable.tag != "solvable":
                return None
            name = solvable.attrib.get("name", "")
            edition = solvable.attrib.get("edition", "")
            architecture = solvable.attrib.get("arch", "")
            if (
                solvable.attrib.get("kind") != "package"
                or solvable.attrib.get("status")
                not in {"installed", "not-installed", "other-version"}
                or not all(
                    value
                    and value == value.strip()
                    and not any(character.isspace() for character in value)
                    for value in (name, edition, architecture)
                )
            ):
                return None
            action = (group.tag, (name, edition, architecture))
            if action in seen:
                return None
            seen.add(action)
            actions.append(action)
    if packages_to_change != len(actions):
        return None
    return actions


def _format_zypper_identity(identity: tuple[str, str, str]) -> str:
    name, edition, architecture = identity
    return f"{name}.{architecture}={edition}"


def _split_rpm_edition(edition: str) -> tuple[str | None, str]:
    match = re.fullmatch(r"(?:(?P<epoch>[0-9]+):)?(?P<version>.+)", edition)
    if match is None:
        raise PackagePreflightError(
            "Zypper emitted an invalid package edition",
            package_manager="zypper",
            packages=[],
            results=[],
        )
    epoch = match.group("epoch")
    return (None if epoch in {None, "0"} else epoch), match.group("version")
