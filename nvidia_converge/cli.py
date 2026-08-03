from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from types import FrameType

from . import __version__
from .audit import audit_host
from .desired import DesiredConfigError, load_desired
from .dnf_module_transaction import (
    DNF_MODULE_FAILSAFE_DIRECTORY,
    parse_dnf_module_enable_proof,
)
from .dnf_transaction import dnf_local_transaction_command
from .doctor import diagnose
from .files import atomic_write_text
from .gpu_safety import (
    TrustedGpuServiceGuard,
    TrustedGpuServiceIdentity,
    is_workload_probe,
    probe_active_gpu_workloads,
    quiesce_trusted_gpu_services,
    validate_active_trusted_gpu_service_identity,
    validate_trusted_docker_socket_unit,
    validate_trusted_gpu_service_start,
    validate_trusted_gpu_service_unit,
)
from .human import render_human
from .locking import OperationLockError, operation_lock
from .mig import (
    full_mig_geometry_matches,
    mig_geometry_destroy_commands,
    restorable_mig_geometry,
)
from .models import (
    CommandResult,
    DesiredState,
    Finding,
    HostAudit,
    PackageInfo,
    PlanAction,
    Report,
    RollbackSnapshot,
    Severity,
    Verification,
    utc_now,
)
from .package_payloads import (
    PackagePayloadError,
    forward_package_command,
    local_payload_paths,
    stage_package_payloads,
)
from .planner import (
    build_plan,
    lock_actions,
    mig_reconciliation_actions,
    module_reload_required,
)
from .preflight import (
    PackagePreflightError,
    preflight_package_install,
    preflight_package_lock,
    preflight_package_rollback,
    preflight_snapshot_restore_availability,
    preflight_staged_forward_transaction,
    resolved_forward_payload_packages,
)
from .recovery import (
    RecoveryStateError,
    UnresolvedOperation,
    recovery_snapshot_path,
    unresolved_operations,
)
from .report import (
    REPORT_DIR,
    ReportJournalIntegrityError,
    ReportWriteError,
    append_report_journal,
    applied_report_path,
    cleanup_stale_applied_report_reserves,
    release_applied_report_reserve,
    report_json,
    require_applied_state_capacity,
    reserve_applied_report,
    sbom_from_audit,
    write_report,
)
from .rollback import (
    SNAPSHOT_DIR,
    RollbackSnapshotError,
    _host_identity,
    _quarantine_service_for_rollback,
    apply_rollback,
    create_snapshot,
    load_snapshot,
    new_snapshot_path,
    prepare_rollback_service_activity,
    restore_rollback_service_activity,
    restore_rollback_service_enablement,
    validate_snapshot_for_apply,
    verify_rollback,
)
from .runner import CommandRunner
from .schemas import schema_json
from .support import support_human, support_json
from .verify import prepare_stack, verify_stack

_module_reload_required = module_reload_required
_MUTATING_OPERATION_IDS: set[str] = set()


class TerminationRequested(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


class _PayloadStagingRunner:
    def __init__(self, runner: CommandRunner):
        self._runner = runner

    def run(
        self,
        command: list[str],
        *,
        mutate: bool = False,
        allow_fail: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        if mutate:
            raise ValueError("payload staging subcommands cannot mutate host state")
        return self._runner.run_private_state(
            command,
            allow_fail=allow_fail,
            input_text=input_text,
        )


@dataclass
class _MaintenanceGateOutcome:
    guard: TrustedGpuServiceGuard | None
    before_probe_results: list[CommandResult]
    probe: CommandResult | None
    after_probe_results: list[CommandResult]
    findings: list[Finding]

    @property
    def command_results(self) -> list[CommandResult]:
        return [
            *self.before_probe_results,
            *([self.probe] if self.probe is not None else []),
            *self.after_probe_results,
        ]


def main(argv: list[str] | None = None) -> int:
    process_entrypoint = argv is None
    if argv is None:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, _request_termination)
        signal.signal(signal.SIGTERM, _request_termination)
    try:
        return _main(
            argv,
            enforce_process_isolation=process_entrypoint,
        )
    except BrokenPipeError:
        # Keep the replacement stream alive through interpreter shutdown.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        return 1
    except ReportWriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TerminationRequested as exc:
        print(f"error: interrupted by signal {exc.signum}", file=sys.stderr)
        return 128 + exc.signum
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


def _request_termination(signum: int, frame: FrameType | None) -> None:
    del frame
    raise TerminationRequested(signum)


def _main(
    argv: list[str] | None = None,
    *,
    enforce_process_isolation: bool = False,
) -> int:
    parser = argparse.ArgumentParser(
        prog="nvidia-converge",
        description="Converge a node to a desired NVIDIA driver stack.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    _add_common_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "plan"):
        _add_common_args(sub.add_parser(name))
    for name in ("install", "verify", "lock", "snapshot"):
        command_parser = sub.add_parser(name)
        _add_common_args(command_parser, include_apply=True)
        if name in {"install", "lock", "verify"}:
            _add_disruption_args(command_parser)
    validate = sub.add_parser("validate")
    validate.add_argument(
        "--desired", default=argparse.SUPPRESS, help="Desired-state JSON/YAML file."
    )
    validate.add_argument(
        "--out",
        default=argparse.SUPPRESS,
        help="Write machine-readable validation JSON to this path.",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print machine-readable validation details.",
    )
    schema = sub.add_parser("schema")
    schema.add_argument(
        "name",
        choices=("desired", "integration-results", "report", "validation"),
        help="Schema to print.",
    )
    support = sub.add_parser("support")
    support.add_argument(
        "--json", action="store_true", help="Print support matrix as JSON."
    )
    rollback = sub.add_parser("rollback")
    _add_common_args(rollback, include_apply=True)
    _add_disruption_args(rollback)
    rollback.add_argument(
        "--snapshot",
        required=True,
        help="Rollback snapshot JSON created by install or snapshot.",
    )
    args = parser.parse_args(argv)

    if (
        enforce_process_isolation
        and getattr(args, "apply", False)
        and _requires_root(args.command)
        and not sys.flags.isolated
    ):
        print(
            f"error: {args.command} --apply requires CPython isolated mode; "
            "invoke the verified versioned interpreter with "
            "-I -m nvidia_converge",
            file=sys.stderr,
        )
        return 2

    if args.command == "schema":
        print(schema_json(args.name))
        return 0

    if args.command == "support":
        print(support_json() if args.json else support_human())
        return 0

    desired_path = getattr(args, "desired", None)
    out_path = getattr(args, "out", None)
    apply_changes = getattr(args, "apply", False)
    json_stdout = getattr(args, "json", False)
    if (
        apply_changes
        and args.command in {"install", "verify", "lock", "snapshot"}
        and not desired_path
    ):
        print(
            f"error: {args.command} --apply requires an explicit --desired file",
            file=sys.stderr,
        )
        return 2
    root_required = apply_changes and _requires_root(args.command)
    if root_required and hasattr(os, "geteuid") and os.geteuid() != 0:
        print(f"error: {args.command} --apply must be run as root", file=sys.stderr)
        return 2
    try:
        desired = load_desired(
            desired_path,
            require_root_controlled=root_required,
        )
    except DesiredConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "validate":
        emit_validation(
            desired, getattr(args, "out", None), getattr(args, "json", False)
        )
        return 0
    try:
        with operation_lock(root_required):
            operation_report: Report | None = None
            recovery_operations: list[UnresolvedOperation] = []
            if root_required:
                try:
                    # The operation lock proves that no live applied process
                    # can still own an emergency reserve left by a dead peer.
                    cleanup_stale_applied_report_reserves()
                    recovery_operations = unresolved_operations()
                    required_snapshot = recovery_snapshot_path(recovery_operations)
                except (RecoveryStateError, ReportWriteError) as exc:
                    print(
                        f"error: cannot establish crash-recovery authority: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                if required_snapshot is not None and (
                    args.command != "rollback"
                    or Path(args.snapshot) != required_snapshot
                ):
                    print(
                        "error: an interrupted applied operation blocks new "
                        "work; recover it with exactly: nvidia-converge "
                        "rollback --apply --allow-disruption --snapshot "
                        f"{required_snapshot}",
                        file=sys.stderr,
                    )
                    return 2

                # Reserve the crash journal only while holding the same lock
                # that serializes all applied host mutations. A losing process
                # must never leave a false unresolved operation behind.
                operation_report = _provisional_operation_report(
                    args.command,
                    desired,
                )
                secure_path = applied_report_path(
                    args.command,
                    out_path,
                    operation_report.operation_id,
                )
                out_path = str(secure_path)
                operation_report.report_path = out_path
                reserve_applied_report(
                    operation_report,
                    secure_path,
                    capacity_paths=(SNAPSHOT_DIR,),
                )
            try:
                return _execute_command(
                    args,
                    desired,
                    out_path,
                    json_stdout,
                    apply_changes,
                    operation_report,
                    recovery_operations,
                )
            finally:
                if root_required and out_path is not None:
                    release_applied_report_reserve(Path(out_path))
    except OperationLockError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _execute_command(
    args: argparse.Namespace,
    desired: DesiredState,
    out_path: str | None,
    json_stdout: bool,
    apply_changes: bool,
    operation_report: Report | None,
    recovery_operations: list[UnresolvedOperation] | None = None,
) -> int:
    start_callback = None
    result_callback = None
    if operation_report is not None and out_path is not None:
        journal_path = Path(out_path)

        def start_callback(command: list[str], mutate: bool) -> None:
            try:
                if mutate:
                    # Recheck immediately before every state or host mutation;
                    # long preflights and unrelated writers may consume the
                    # admission budget after the operation was reserved.
                    require_applied_state_capacity(REPORT_DIR, SNAPSHOT_DIR)
                append_report_journal(
                    journal_path,
                    operation_report.operation_id,
                    "command-started",
                    command=command,
                    mutating=mutate,
                )
            except ReportWriteError:
                if operation_report.operation_id in _MUTATING_OPERATION_IDS:
                    _emergency_quarantine_launchers()
                raise
            if mutate and tuple(command) not in {
                ("persist-rollback-snapshot",),
                ("stage-package-payloads",),
            }:
                _MUTATING_OPERATION_IDS.add(operation_report.operation_id)

        def result_callback(result: CommandResult, mutate: bool) -> None:
            try:
                append_report_journal(
                    journal_path,
                    operation_report.operation_id,
                    "command-finished",
                    command=result.command,
                    mutating=mutate,
                    returncode=result.returncode,
                    skipped=result.skipped,
                    reason=result.reason,
                )
            except ReportWriteError:
                # The subprocess has already completed. If its durable finish
                # record cannot be appended, leave every launcher persistently
                # quarantined so the next exact-snapshot recovery cannot race a
                # package-script or socket activation.
                if operation_report.operation_id in _MUTATING_OPERATION_IDS:
                    _emergency_quarantine_launchers()
                raise

    runner = CommandRunner(
        apply=apply_changes,
        start_callback=start_callback,
        result_callback=result_callback,
    )

    if args.command == "rollback":
        try:
            snapshot = load_snapshot(args.snapshot)
        except RollbackSnapshotError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        recovery_operations = recovery_operations or []
        if recovery_operations:
            expected_bindings = {
                (
                    operation.snapshot_integrity_sha256,
                    operation.snapshot_operation_id,
                    operation.snapshot_host_id,
                )
                for operation in recovery_operations
            }
            if expected_bindings != {
                (
                    snapshot.integrity_sha256,
                    snapshot.operation_id,
                    snapshot.host_id,
                )
            }:
                print(
                    "error: the recovery snapshot integrity, creator, or host "
                    "does not match the durable interrupted-operation binding",
                    file=sys.stderr,
                )
                return 2
        report = Report("1.2", utc_now(), desired, rollback=snapshot)
        _inherit_operation(report, operation_report)
        rollback_maintenance_results: list[CommandResult] = []
        rollback_service_guard: TrustedGpuServiceGuard | None = None
        rollback_service_result_offset = 0
        pre_rollback_audit: HostAudit | None = None
        if apply_changes:
            pre_rollback_audit = audit_host(runner)
            report.audit = pre_rollback_audit
            report.sbom = sbom_from_audit(pre_rollback_audit)
            if not pre_rollback_audit.package_inventory_complete:
                report.findings.append(
                    Finding(
                        "rollback.package-inventory-incomplete",
                        Severity.ERROR,
                        "Rollback requires a complete package inventory",
                        "The current package inventory is incomplete, so rollback cannot safely derive a state difference.",
                        remediation="Repair package inventory queries before retrying rollback.",
                    )
                )
                emit_report(args.command, report, out_path, json_stdout, apply_changes)
                return 2
            try:
                validate_snapshot_for_apply(snapshot, args.snapshot, pre_rollback_audit)
            except RollbackSnapshotError as exc:
                report.findings.append(
                    Finding(
                        "rollback.snapshot-untrusted",
                        Severity.ERROR,
                        "Rollback snapshot trust validation failed",
                        str(exc),
                        remediation="Use the original private snapshot on the host where it was created.",
                    )
                )
                emit_report(args.command, report, out_path, json_stdout, apply_changes)
                return 2
            try:
                preflight_results = preflight_package_rollback(
                    snapshot,
                    pre_rollback_audit,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "rollback.packages-preflight",
                    preflight_results,
                    "Every exact rollback package resolves before host mutation.",
                )
            except PackagePreflightError as exc:
                _append_preflight_verification(
                    report,
                    "rollback.packages-preflight",
                    exc.results,
                    "An exact rollback package transaction did not resolve safely.",
                    ok=False,
                )
                report.findings.append(
                    Finding(
                        "rollback.packages-preflight-failed",
                        Severity.ERROR,
                        "Rollback package preflight failed",
                        str(exc),
                        evidence=exc.evidence(),
                        remediation=(
                            "Restore the original private snapshot and its paired retained "
                            "payload bundle from trusted backup before retrying rollback."
                        ),
                    )
                )
                emit_report(args.command, report, out_path, json_stdout, apply_changes)
                return 2
            _append_snapshot_binding(
                report,
                out_path,
                snapshot,
                snapshot_path=snapshot.path,
            )
            maintenance = _maintenance_gate(
                args,
                runner,
                pre_rollback_audit,
                operation="rollback",
                maintenance_detail=(
                    "Rollback may change packages, modules, MIG mode, managed "
                    "files, Docker, and Fabric Manager."
                ),
            )
            rollback_maintenance_results = maintenance.command_results
            rollback_service_guard = maintenance.guard
            report.findings.extend(maintenance.findings)
            if maintenance.findings:
                report.command_results = rollback_maintenance_results
                if rollback_service_guard is not None:
                    _, quarantined_audit = _quarantine_failed_maintenance_gate(
                        report,
                        runner,
                        rollback_service_guard,
                    )
                    report.audit = quarantined_audit
                    report.sbom = sbom_from_audit(quarantined_audit)
                emit_report(args.command, report, out_path, json_stdout, apply_changes)
                return 2
            assert rollback_service_guard is not None
            rollback_service_result_offset = len(rollback_service_guard.results)
            rollback_service_guard.mark_mutation_started()
            (
                rollback_boundary_ok,
                fresh_rollback_audit,
                _,
            ) = _pre_gpu_mutation_checkpoint(
                args,
                report,
                runner,
                rollback_service_guard,
            )
            rollback_boundary_ok = bool(
                rollback_boundary_ok
                and _fresh_gpu_boundary_is_safe(
                    report,
                    snapshot,
                    fresh_rollback_audit,
                    boundary="rollback-pre-apply",
                )
            )
            if rollback_boundary_ok:
                try:
                    validate_snapshot_for_apply(
                        snapshot,
                        args.snapshot,
                        fresh_rollback_audit,
                    )
                    fresh_preflight = preflight_package_rollback(
                        snapshot,
                        fresh_rollback_audit,
                        runner,
                    )
                    _append_preflight_verification(
                        report,
                        "rollback.post-quarantine-packages-preflight",
                        fresh_preflight,
                        (
                            "The exact rollback delta resolves from the fresh "
                            "post-quarantine audit."
                        ),
                    )
                except (PackagePreflightError, RollbackSnapshotError) as exc:
                    if isinstance(exc, PackagePreflightError):
                        _append_preflight_verification(
                            report,
                            "rollback.post-quarantine-packages-preflight",
                            exc.results,
                            (
                                "The fresh post-quarantine rollback delta did not "
                                "resolve safely."
                            ),
                            ok=False,
                        )
                    report.findings.append(
                        Finding(
                            "rollback.post-quarantine-preflight-failed",
                            Severity.ERROR,
                            "Fresh rollback preflight failed",
                            str(exc),
                            evidence=(
                                exc.evidence()
                                if isinstance(exc, PackagePreflightError)
                                else {}
                            ),
                            remediation=(
                                "Keep all launchers masked and rerun the exact "
                                "private snapshot rollback after repairing the "
                                "reported precondition."
                            ),
                        )
                    )
                    rollback_boundary_ok = False
            if not rollback_boundary_ok:
                report.findings.append(
                    _rollback_reboot_recovery_finding(
                        snapshot,
                        args.snapshot,
                        operation="rollback",
                    )
                )
                report.command_results = [
                    *rollback_maintenance_results,
                    *report.command_results,
                    *rollback_service_guard.results[rollback_service_result_offset:],
                ]
                report.audit = fresh_rollback_audit
                report.sbom = sbom_from_audit(fresh_rollback_audit)
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2
            pre_rollback_audit = fresh_rollback_audit
        rollback_boundary_results = list(report.command_results)
        results = apply_rollback(
            snapshot,
            runner,
            current_audit=pre_rollback_audit if apply_changes else None,
            restore_service_activity=False,
        )
        report.command_results = [
            *rollback_maintenance_results,
            *rollback_boundary_results,
            *results,
        ]
        rollback_succeeded = _commands_succeeded(results)
        if apply_changes:
            audit = audit_host(runner)
            if rollback_succeeded:
                core_checks = verify_rollback(
                    snapshot,
                    audit,
                    include_service_state=False,
                )
                report.verification.extend(core_checks)
                core_verified = all(check.ok for check in core_checks)
                rollback_reboot_pending = _rollback_state_can_complete_after_reboot(
                    snapshot,
                    audit,
                    core_checks,
                )
                if core_verified:
                    _append_launcher_release_authorization(
                        report,
                        out_path,
                        snapshot,
                        release_target="rollback-baseline",
                    )
                    rollback_succeeded, audit = _commit_rollback_service_activity(
                        args,
                        report,
                        snapshot,
                        runner,
                        audit,
                        rollback_service_guard,
                        operation="rollback",
                    )
                    if rollback_succeeded:
                        full_checks = verify_rollback(snapshot, audit)
                        report.verification.extend(full_checks[len(core_checks) :])
                        rollback_succeeded = all(check.ok for check in full_checks)
                elif rollback_reboot_pending:
                    (
                        launchers_quiesced,
                        audit,
                        _,
                    ) = _quiesce_launchers_for_reboot(
                        args,
                        report,
                        runner,
                        rollback_service_guard,
                    )
                    rollback_succeeded = launchers_quiesced
                    if launchers_quiesced:
                        report.findings.append(
                            _rollback_reboot_recovery_finding(
                                snapshot,
                                args.snapshot,
                                operation="rollback",
                            )
                        )
                else:
                    rollback_succeeded = False
                    _quiesce_launchers_for_reboot(
                        args,
                        report,
                        runner,
                        rollback_service_guard,
                    )
                    report.findings.append(
                        Finding(
                            "rollback.core-verification-failed",
                            Severity.ERROR,
                            "Rollback core state could not be verified",
                            (
                                "All launchers remain quiesced; use the original "
                                "private snapshot after repairing the core stack."
                            ),
                        )
                    )
            else:
                assert pre_rollback_audit is not None
                _quiesce_launchers_for_reboot(
                    args,
                    report,
                    runner,
                    rollback_service_guard,
                )
                audit = audit_host(runner)
                _record_failed_rollback_service_changes(
                    report,
                    pre_rollback_audit,
                    audit,
                )
            if rollback_service_guard is not None:
                if not rollback_succeeded:
                    _record_intentionally_quiesced_services(
                        report,
                        rollback_service_guard,
                    )
                report.command_results.extend(
                    rollback_service_guard.results[rollback_service_result_offset:]
                )
            report.audit = audit
            report.sbom = sbom_from_audit(audit)
        rollback_fully_verified = bool(
            rollback_succeeded
            and report.verification
            and all(check.ok for check in report.verification)
            and all(finding.severity.value != "error" for finding in report.findings)
        )
        if apply_changes and rollback_fully_verified and recovery_operations:
            _mark_operations_recovered(
                recovery_operations,
                recovery_operation_id=report.operation_id,
                snapshot=snapshot,
            )
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _report_status(args.command, report)

    audit = audit_host(runner)
    findings = diagnose(desired, audit)
    report = Report(
        "1.2",
        utc_now(),
        desired,
        audit=audit,
        findings=findings,
        sbom=sbom_from_audit(audit),
    )
    _inherit_operation(report, operation_report)

    if args.command == "doctor":
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0 if all(f.severity.value != "error" for f in findings) else 2

    if args.command == "plan":
        report.plan = build_plan(desired, audit, findings)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _plan_status(report)

    if args.command == "snapshot":
        try:
            report.rollback, snapshot_result = _create_snapshot_with_evidence(
                audit,
                desired,
                runner,
                persist=apply_changes,
                operation_id=report.operation_id,
                journal_report_path=(
                    Path(out_path) if apply_changes and out_path is not None else None
                ),
            )
            report.command_results = [snapshot_result]
        except RollbackSnapshotError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0

    if args.command == "install":
        # Applied planning and both package preflights must share the exact
        # pre-mutation observation that is persisted as rollback authority.
        if apply_changes:
            audit = audit_host(runner)
            findings = diagnose(desired, audit)
            report.audit = audit
            report.findings = findings
            report.sbom = sbom_from_audit(audit)
        report.plan = build_plan(desired, audit, findings)
        unsupported = [
            action for action in report.plan if action.id.startswith("unsupported.")
        ]
        if unsupported:
            report.findings.append(
                Finding(
                    "plan.unsupported",
                    Severity.ERROR,
                    "No safe package recipe is available for this host",
                    "; ".join(
                        action.reason or action.description for action in unsupported
                    ),
                )
            )
            emit_report(args.command, report, out_path, json_stdout, apply_changes)
            return 2
        disruptive = [action for action in report.plan if action.destructive]
        service_guard: TrustedGpuServiceGuard | None = None
        service_result_offset = 0
        maintenance_results: list[CommandResult] = []
        phase_audit = audit
        docker_was_stopped = False
        initial_quarantine_succeeded = True

        # Maintenance authorization and the workload decision are read-only at
        # this point. No systemd stop may precede the private snapshot binding.
        if apply_changes and disruptive:
            authorization = _maintenance_gate(
                args,
                runner,
                audit,
                operation="disruptive convergence",
                maintenance_detail=(
                    "The applied plan contains: "
                    + ", ".join(action.id for action in disruptive)
                ),
                quiesce_services=False,
            )
            maintenance_results = authorization.command_results
            report.findings.extend(authorization.findings)
            if authorization.findings:
                report.command_results = maintenance_results
                emit_report(args.command, report, out_path, json_stdout, apply_changes)
                return 2

            # The read-only workload gate can take time. Rebind every plan and
            # package operand to one fresh audit immediately before snapshot.
            audit = audit_host(runner)
            findings = diagnose(desired, audit)
            report.audit = audit
            report.findings = findings
            report.sbom = sbom_from_audit(audit)
            report.plan = build_plan(desired, audit, findings)
            unsupported = [
                action for action in report.plan if action.id.startswith("unsupported.")
            ]
            if unsupported:
                report.findings.append(
                    Finding(
                        "plan.unsupported",
                        Severity.ERROR,
                        "No safe package recipe is available for this host",
                        "; ".join(
                            action.reason or action.description
                            for action in unsupported
                        ),
                    )
                )
                report.command_results = maintenance_results
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2
            disruptive = [action for action in report.plan if action.destructive]

        forward_payload_packages: list[PackageInfo] = []
        if apply_changes and any(
            action.id == "install.packages" for action in report.plan
        ):
            try:
                preflight_results = preflight_package_install(
                    desired,
                    audit,
                    runner,
                )
                if audit.package_manager is None:
                    raise PackagePreflightError(
                        "package manager disappeared after planning",
                        package_manager=None,
                        packages=[],
                        results=preflight_results,
                    )
                forward_payload_packages = resolved_forward_payload_packages(
                    audit.package_manager,
                    preflight_results,
                )
                report.verification.extend(
                    Verification(
                        "packages.preflight",
                        True,
                        result,
                        "Planned package targets resolve from current trusted repository metadata.",
                    )
                    for result in preflight_results
                )
            except PackagePreflightError as exc:
                report.verification.extend(
                    Verification(
                        "packages.preflight",
                        False,
                        result,
                        "A planned package target did not resolve from current repository metadata.",
                    )
                    for result in exc.results
                )
                report.findings.append(
                    Finding(
                        "packages.preflight.failed",
                        Severity.ERROR,
                        "Package repository preflight failed",
                        str(exc),
                        evidence=exc.evidence(),
                        remediation=(
                            "Refresh package metadata and ensure every planned NVIDIA, "
                            "Docker, and operating-system package is available from a "
                            "configured repository before retrying."
                        ),
                    )
                )
                report.command_results = maintenance_results
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2
        try:
            report.rollback, snapshot_result = _create_snapshot_with_evidence(
                audit,
                desired,
                runner,
                persist=apply_changes,
                operation_id=report.operation_id,
                journal_report_path=(
                    Path(out_path) if apply_changes and out_path is not None else None
                ),
                forward_packages=forward_payload_packages,
            )
        except RollbackSnapshotError as exc:
            report.findings.append(
                Finding(
                    "snapshot.failed",
                    Severity.ERROR,
                    "Pre-install rollback snapshot failed",
                    str(exc),
                )
            )
            report.command_results = maintenance_results
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2
        if apply_changes:
            try:
                report.plan = _bind_forward_package_payloads(
                    report.plan,
                    report.rollback,
                    audit,
                )
            except PackagePayloadError as exc:
                report.findings.append(
                    Finding(
                        "packages.payload-binding-failed",
                        Severity.ERROR,
                        "Forward package payload binding failed",
                        str(exc),
                    )
                )
                report.command_results = [*maintenance_results, snapshot_result]
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2
        if apply_changes:
            try:
                forward_preflight = preflight_staged_forward_transaction(
                    desired,
                    report.rollback,
                    audit,
                    report.plan,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "packages.staged-local-preflight",
                    forward_preflight,
                    "The exact retained forward transaction resolves without repositories.",
                )
            except PackagePreflightError as exc:
                _append_preflight_verification(
                    report,
                    "packages.staged-local-preflight",
                    exc.results,
                    "The retained forward transaction does not resolve exactly offline.",
                    ok=False,
                )
                report.findings.append(
                    Finding(
                        "packages.staged-local-preflight-failed",
                        Severity.ERROR,
                        "Staged local package transaction failed",
                        str(exc),
                        evidence=exc.evidence(),
                    )
                )
                report.command_results = [*maintenance_results, snapshot_result]
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2
            try:
                restore_preflight = preflight_snapshot_restore_availability(
                    report.rollback,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "rollback.baseline-packages-available",
                    restore_preflight,
                    "Every exact baseline NVIDIA-stack package is reinstallable before mutation.",
                )
            except PackagePreflightError as exc:
                _append_preflight_verification(
                    report,
                    "rollback.baseline-packages-available",
                    exc.results,
                    "At least one exact baseline NVIDIA-stack package is unavailable.",
                    ok=False,
                )
                report.findings.append(
                    Finding(
                        "rollback.baseline-preflight.failed",
                        Severity.ERROR,
                        "Exact rollback baseline is unavailable",
                        str(exc),
                        evidence=exc.evidence(),
                        remediation=(
                            "Repair or restore the original private snapshot and its paired "
                            "retained payload bundle, or restart with a fresh snapshot before "
                            "any mutation."
                        ),
                    )
                )
                report.command_results = [
                    *maintenance_results,
                    snapshot_result,
                ]
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2

            maintenance = _maintenance_gate(
                args,
                runner,
                audit,
                operation="disruptive convergence",
                maintenance_detail=(
                    "The applied plan contains: "
                    + ", ".join(action.id for action in disruptive)
                ),
            )
            service_guard = maintenance.guard
            maintenance_results.extend(maintenance.command_results)
            report.findings.extend(maintenance.findings)
            if service_guard is not None:
                service_result_offset = len(service_guard.results)
                service_guard.mark_mutation_started()
            if maintenance.findings or service_guard is None:
                initial_quarantine_succeeded = False
                if service_guard is not None:
                    _, phase_audit = _quarantine_failed_maintenance_gate(
                        report,
                        runner,
                        service_guard,
                    )
            else:
                (
                    initial_quarantine_succeeded,
                    phase_audit,
                    docker_was_stopped,
                ) = _pre_gpu_mutation_checkpoint(
                    args,
                    report,
                    runner,
                    service_guard,
                )
                if initial_quarantine_succeeded:
                    initial_quarantine_succeeded = _fresh_gpu_boundary_is_safe(
                        report,
                        report.rollback,
                        phase_audit,
                        boundary="pre-package",
                    )
                if initial_quarantine_succeeded:
                    fresh_plan = build_plan(
                        desired,
                        phase_audit,
                        diagnose(desired, phase_audit),
                    )
                    fresh_unsupported = [
                        action
                        for action in fresh_plan
                        if action.id.startswith("unsupported.")
                    ]
                    if fresh_unsupported:
                        report.findings.append(
                            Finding(
                                "install.post-quarantine-plan-unsupported",
                                Severity.ERROR,
                                "Fresh transaction plan is unsupported",
                                "; ".join(
                                    action.reason or action.description
                                    for action in fresh_unsupported
                                ),
                            )
                        )
                        initial_quarantine_succeeded = False
                    else:
                        try:
                            report.plan = _bind_forward_package_payloads(
                                fresh_plan,
                                report.rollback,
                                phase_audit,
                            )
                        except PackagePayloadError as exc:
                            report.findings.append(
                                Finding(
                                    "install.post-quarantine-payload-binding-failed",
                                    Severity.ERROR,
                                    "Fresh package payload binding failed",
                                    str(exc),
                                )
                            )
                            initial_quarantine_succeeded = False
                if initial_quarantine_succeeded and any(
                    action.id == "install.packages" for action in report.plan
                ):
                    try:
                        fresh_target_preflight = preflight_staged_forward_transaction(
                            desired,
                            report.rollback,
                            phase_audit,
                            report.plan,
                            runner,
                        )
                        _append_preflight_verification(
                            report,
                            "packages.post-quarantine-preflight",
                            fresh_target_preflight,
                            (
                                "Fresh snapshot-bound local payloads resolve "
                                "exactly after launcher quarantine."
                            ),
                        )
                    except PackagePreflightError as exc:
                        _append_preflight_verification(
                            report,
                            "packages.post-quarantine-preflight",
                            exc.results,
                            (
                                "Fresh snapshot-bound local payloads no longer "
                                "resolve exactly after launcher quarantine."
                            ),
                            ok=False,
                        )
                        report.findings.append(
                            Finding(
                                "install.post-quarantine-preflight-failed",
                                Severity.ERROR,
                                "Fresh target package preflight failed",
                                str(exc),
                                evidence=exc.evidence(),
                            )
                        )
                        initial_quarantine_succeeded = False
                if initial_quarantine_succeeded:
                    try:
                        fresh_rollback_preflight = preflight_package_rollback(
                            report.rollback,
                            phase_audit,
                            runner,
                        )
                        _append_preflight_verification(
                            report,
                            "rollback.post-quarantine-preflight",
                            fresh_rollback_preflight,
                            (
                                "The exact rollback delta remains resolvable from "
                                "the fresh quarantined baseline."
                            ),
                        )
                    except PackagePreflightError as exc:
                        _append_preflight_verification(
                            report,
                            "rollback.post-quarantine-preflight",
                            exc.results,
                            "The exact rollback delta is no longer safely resolvable.",
                            ok=False,
                        )
                        report.findings.append(
                            Finding(
                                "install.post-quarantine-rollback-preflight-failed",
                                Severity.ERROR,
                                "Fresh rollback package preflight failed",
                                str(exc),
                                evidence=exc.evidence(),
                            )
                        )
                        initial_quarantine_succeeded = False
        mig_action_ids = {
            "disable.mig",
            "enable.mig",
            "configure.mig-geometry",
            "reconcile.mig-after-module",
        }
        docker_action_ids = {"service.docker", "configure.docker-runtime"}
        teardown_action_id = "prepare.mig-geometry-teardown"
        deferred_action_ids = {
            "snapshot.current-state",
            "prepare.module",
            "verify.stack",
            teardown_action_id,
            *mig_action_ids,
            *docker_action_ids,
        }
        pre_module_actions = [
            action for action in report.plan if action.id not in deferred_action_ids
        ]
        boundary_results = list(report.command_results)
        report.command_results = [
            *maintenance_results,
            snapshot_result,
            *boundary_results,
        ]
        phase_succeeded = bool(
            initial_quarantine_succeeded and _commands_succeeded(report.command_results)
        )
        module_reset_planned = bool(
            audit.module.loaded
            and any(action.id == "prepare.module" for action in report.plan)
        )
        package_transaction_planned = any(
            action.id
            in {"install.packages", "lock.apt", "lock.rpm", "lock.zypper"}
            and any(action.commands)
            for action in pre_module_actions
        )
        pre_gpu_mutation_planned = any(
            action.id
            in {
                "prepare.module",
                teardown_action_id,
                *mig_action_ids,
            }
            for action in report.plan
        )
        if phase_succeeded and package_transaction_planned and any(
            "forward" in payload.roles
            for payload in (
                report.rollback.package_payloads.packages
                if report.rollback.package_payloads is not None
                else ()
            )
        ):
            try:
                immediate_forward_preflight = preflight_staged_forward_transaction(
                    desired,
                    report.rollback,
                    phase_audit,
                    report.plan,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "packages.immediate-local-preflight",
                    immediate_forward_preflight,
                    "Retained package bytes and the offline solver transaction were revalidated immediately before mutation.",
                )
            except PackagePreflightError as exc:
                _append_preflight_verification(
                    report,
                    "packages.immediate-local-preflight",
                    exc.results,
                    "Retained package bytes or the offline solver transaction changed before mutation.",
                    ok=False,
                )
                report.findings.append(
                    Finding(
                        "install.immediate-local-preflight-failed",
                        Severity.ERROR,
                        "Immediate local package validation failed",
                        str(exc),
                        evidence=exc.evidence(),
                    )
                )
                phase_succeeded = False
        if phase_succeeded and pre_module_actions:
            action_results = _run_plan_actions(pre_module_actions, runner)
            report.command_results.extend(action_results)
            phase_succeeded = _commands_succeeded(action_results)

        if (
            apply_changes
            and phase_succeeded
            and (
                package_transaction_planned
                or pre_gpu_mutation_planned
                or desired.container_runtime == "docker"
            )
        ):
            (
                phase_succeeded,
                phase_audit,
                checkpoint_stopped_docker,
            ) = _pre_gpu_mutation_checkpoint(
                args,
                report,
                runner,
                service_guard,
            )
            docker_was_stopped = docker_was_stopped or checkpoint_stopped_docker
            if phase_succeeded:
                assert report.rollback is not None
                phase_succeeded = _fresh_gpu_boundary_is_safe(
                    report,
                    report.rollback,
                    phase_audit,
                    boundary=(
                        "post-package"
                        if package_transaction_planned
                        else "pre-mutation"
                    ),
                )

        teardown_actions = (
            _fresh_module_reset_mig_teardown(
                phase_audit,
                module_reset_planned=module_reset_planned,
            )
            if apply_changes
            else [action for action in report.plan if action.id == teardown_action_id]
        )
        unsupported_teardown = [
            action
            for action in teardown_actions
            if action.id.startswith("unsupported.")
        ]
        if unsupported_teardown:
            report.findings.append(
                Finding(
                    "install.post-package-mig-unsupported",
                    Severity.ERROR,
                    "Fresh MIG state cannot survive an exact module reset",
                    "; ".join(
                        action.reason or action.description
                        for action in unsupported_teardown
                    ),
                )
            )
            phase_succeeded = False
        elif apply_changes:
            _replace_applied_teardown_plan(report.plan, teardown_actions)
        if phase_succeeded and teardown_actions:
            teardown_results = _run_plan_actions(teardown_actions, runner)
            report.command_results.extend(teardown_results)
            phase_succeeded = phase_succeeded and _commands_succeeded(teardown_results)
            if apply_changes and phase_succeeded:
                phase_audit = audit_host(runner)

        preparation_result_offset = len(runner.results)
        preparation_planned = any(
            action.id == "prepare.module" for action in report.plan
        )
        preparation = (
            prepare_stack(
                runner,
                phase_audit,
                force_reload=apply_changes and module_reset_planned,
            )
            if phase_succeeded and preparation_planned
            else None
        )
        report.command_results.extend(runner.results[preparation_result_offset:])
        preparation_reboot_pending = False
        preparation_audit = phase_audit
        if apply_changes and preparation is not None and not preparation.ok:
            preparation_audit = audit_host(runner)
            preparation_reboot_pending = _preparation_failure_can_complete_after_reboot(
                desired,
                preparation_audit,
                preparation,
            )
            if not preparation_reboot_pending:
                phase_succeeded = False

        mig_actions: list[PlanAction] = []
        mig_reboot_pending = False
        mig_reconciliation_planned = any(
            action.id in mig_action_ids for action in report.plan
        )
        preparation_ready = bool(
            preparation is None or not apply_changes or preparation.ok
        )
        if phase_succeeded and preparation_ready:
            if apply_changes:
                phase_audit = audit_host(runner)
                assert report.rollback is not None
                phase_succeeded = _fresh_gpu_boundary_is_safe(
                    report,
                    report.rollback,
                    phase_audit,
                    boundary="post-module",
                )
                if phase_succeeded and mig_reconciliation_planned:
                    mig_actions = mig_reconciliation_actions(
                        desired,
                        phase_audit,
                    )
                    unsupported_mig = [
                        action
                        for action in mig_actions
                        if action.id.startswith("unsupported.")
                    ]
                    if unsupported_mig:
                        report.findings.append(
                            Finding(
                                "install.post-module-mig-unsupported",
                                Severity.ERROR,
                                "MIG cannot be reconciled after module preparation",
                                "; ".join(
                                    action.reason or action.description
                                    for action in unsupported_mig
                                ),
                            )
                        )
                        phase_succeeded = False
                    _replace_applied_mig_plan(report.plan, mig_actions)
                    mig_reboot_pending = bool(
                        phase_audit.mig_mode != phase_audit.mig_mode_pending
                        and phase_audit.mig_mode_pending == desired.mig
                    )
            elif not apply_changes:
                mig_actions = [
                    action
                    for action in report.plan
                    if action.id in mig_action_ids
                    and action.id != "reconcile.mig-after-module"
                ]

        if phase_succeeded and preparation_ready and mig_actions:
            if apply_changes:
                (
                    phase_succeeded,
                    phase_audit,
                    mig_reboot_pending,
                    mig_results,
                ) = _execute_mig_reconciliation(
                    desired,
                    runner,
                    phase_audit,
                    mig_actions,
                )
            else:
                mig_results = _run_plan_actions(mig_actions, runner)
                phase_succeeded = _commands_succeeded(mig_results)
            report.command_results.extend(mig_results)

        if (
            phase_succeeded
            and preparation_ready
            and not preparation_reboot_pending
            and not mig_reboot_pending
        ):
            docker_configuration = _docker_configuration_actions(report.plan)
            if docker_configuration:
                docker_configuration_results = _run_plan_actions(
                    docker_configuration,
                    runner,
                )
                report.command_results.extend(docker_configuration_results)
                phase_succeeded = _commands_succeeded(docker_configuration_results)
                if apply_changes and phase_succeeded:
                    phase_audit = audit_host(runner)
                    assert report.rollback is not None
                    phase_succeeded = _fresh_gpu_boundary_is_safe(
                        report,
                        report.rollback,
                        phase_audit,
                        boundary="pre-service-commit",
                    )

        if preparation is not None:
            report.verification.append(preparation)

        docker_checkpoint_ok = True
        services_restored = not apply_changes
        transition_reached_verification = False
        target_reboot_pending = bool(preparation_reboot_pending or mig_reboot_pending)

        if not apply_changes and phase_succeeded:
            docker_audit = audit_host(runner) if apply_changes else phase_audit
            docker_actions = _docker_phase_actions(
                report.plan,
                docker_audit,
                docker_was_stopped=docker_was_stopped,
            )
            if docker_actions:
                docker_results = _run_plan_actions(docker_actions, runner)
                report.command_results.extend(docker_results)
                phase_succeeded = _commands_succeeded(docker_results)
                if (
                    apply_changes
                    and phase_succeeded
                    and _plan_starts_or_restarts_docker(docker_actions)
                ):
                    (
                        docker_checkpoint_ok,
                        _,
                    ) = _post_docker_workload_checkpoint(
                        args,
                        report,
                        runner,
                    )
            transition_reached_verification = bool(
                phase_succeeded and docker_checkpoint_ok
            )
        elif apply_changes and phase_succeeded:
            core_audit = audit_host(runner)
            assert report.rollback is not None
            phase_succeeded = _fresh_gpu_boundary_is_safe(
                report,
                report.rollback,
                core_audit,
                boundary="pre-service-commit",
            )
            core_checks = (
                verify_stack(
                    desired,
                    runner,
                    core_audit,
                    include_docker=False,
                    include_fabric_manager=False,
                )
                if phase_succeeded
                else []
            )
            report.verification.extend(core_checks)
            core_verified = bool(core_checks) and all(check.ok for check in core_checks)

            if target_reboot_pending:
                (
                    launchers_quiesced,
                    phase_audit,
                    checkpoint_stopped_docker,
                ) = _pre_gpu_mutation_checkpoint(
                    args,
                    report,
                    runner,
                    service_guard,
                )
                docker_was_stopped = docker_was_stopped or checkpoint_stopped_docker
                transition_reached_verification = False
                services_restored = False
                if phase_succeeded and launchers_quiesced:
                    report.findings.append(
                        Finding(
                            "install.target-reboot-pending-compensation-required",
                            Severity.ERROR,
                            "Unverified reboot-pending target will be compensated",
                            (
                                "The target stack cannot be verified in the current "
                                "boot. The original private snapshot will be restored "
                                "instead of carrying target drift across reboot."
                            ),
                        )
                    )
            elif phase_succeeded and core_verified:
                _append_launcher_release_authorization(
                    report,
                    out_path,
                    report.rollback,
                    release_target="install-target",
                )
                install_target = _install_launcher_target_snapshot(
                    report.rollback,
                    desired,
                )
                services_restored, docker_audit = _commit_rollback_service_activity(
                    args,
                    report,
                    install_target,
                    runner,
                    core_audit,
                    service_guard,
                    operation="install",
                )
                phase_succeeded = bool(phase_succeeded and services_restored)
                fabric_manager_verified = services_restored
                if services_restored and desired.fabric_manager:
                    fabric_checks = _verification_phase_checks(
                        verify_stack(
                            desired,
                            runner,
                            docker_audit,
                            include_docker=False,
                            include_fabric_manager=True,
                        ),
                        phase="fabric-manager",
                    )
                    report.verification.extend(fabric_checks)
                    fabric_manager_verified = bool(fabric_checks) and all(
                        check.ok for check in fabric_checks
                    )
                transition_reached_verification = bool(
                    phase_succeeded and fabric_manager_verified
                )
                if transition_reached_verification:
                    docker_checks = _verification_phase_checks(
                        verify_stack(
                            desired,
                            runner,
                            docker_audit,
                            include_docker=True,
                            include_fabric_manager=False,
                        ),
                        phase="docker",
                    )
                    report.verification.extend(docker_checks)

        compensation_attempted = False
        compensation_applied = False
        if apply_changes and not transition_reached_verification:
            report.findings.append(
                _install_mutation_failure_finding(
                    report.command_results,
                    preparation,
                )
            )
            assert report.rollback is not None
            compensation_attempted = True
            compensation_audit = _prepare_install_compensation(
                report,
                report.rollback,
                runner,
                service_guard,
                allow_active_workloads=args.allow_active_workloads,
            )
            if compensation_audit is not None:
                compensation_applied = _attempt_install_compensation(
                    report,
                    report.rollback,
                    runner,
                    current_audit=compensation_audit,
                )

        if (
            service_guard is not None
            and not transition_reached_verification
            and not compensation_applied
        ):
            _record_intentionally_quiesced_services(report, service_guard)
            services_restored = False

        post_audit = audit_host(runner)
        operation_findings = [
            finding
            for finding in report.findings
            if finding.id.startswith(
                (
                    "gpu-services.",
                    "gpu-workloads.",
                    "install.",
                    "launcher-commit.",
                )
            )
        ]
        report.audit = post_audit
        report.findings = [*diagnose(desired, post_audit), *operation_findings]
        if (
            not compensation_attempted
            and transition_reached_verification
            and not apply_changes
        ):
            report.verification.extend(verify_stack(desired, runner, post_audit))

        if (
            apply_changes
            and transition_reached_verification
            and not compensation_attempted
            and not _install_target_verified(report)
            and not _target_verification_can_complete_after_reboot(report)
        ):
            report.findings.append(_install_verification_failure_finding(report))
            compensation_attempted = True
            assert report.rollback is not None
            compensation_audit = _prepare_install_compensation(
                report,
                report.rollback,
                runner,
                service_guard,
                allow_active_workloads=args.allow_active_workloads,
            )
            if compensation_audit is not None:
                compensation_applied = _attempt_install_compensation(
                    report,
                    report.rollback,
                    runner,
                    current_audit=compensation_audit,
                )
                if service_guard is not None and not compensation_applied:
                    _record_intentionally_quiesced_services(
                        report,
                        service_guard,
                    )
                    services_restored = False
            else:
                _record_intentionally_quiesced_services(
                    report,
                    service_guard,
                )
                services_restored = False
            post_audit = audit_host(runner)

        if compensation_applied:
            assert report.rollback is not None
            rollback_checks = verify_rollback(
                report.rollback,
                post_audit,
                include_service_state=False,
            )
            compensation_checks = [
                Verification(
                    f"install.compensation.{check.name}",
                    check.ok,
                    check.command,
                    check.detail,
                )
                for check in rollback_checks
            ]
            report.verification.extend(compensation_checks)
            compensation_core_verified = all(check.ok for check in rollback_checks)
            compensation_reboot_pending = _rollback_state_can_complete_after_reboot(
                report.rollback,
                post_audit,
                rollback_checks,
            )
            compensation_verified = False
            if compensation_core_verified:
                _append_launcher_release_authorization(
                    report,
                    out_path,
                    report.rollback,
                    release_target="rollback-baseline",
                )
                (
                    services_restored,
                    post_audit,
                ) = _commit_rollback_service_activity(
                    args,
                    report,
                    report.rollback,
                    runner,
                    post_audit,
                    service_guard,
                    operation="install.compensation",
                )
                if services_restored:
                    full_rollback_checks = verify_rollback(
                        report.rollback,
                        post_audit,
                    )
                    service_checks = full_rollback_checks[len(rollback_checks) :]
                    report.verification.extend(
                        Verification(
                            f"install.compensation.{check.name}",
                            check.ok,
                            check.command,
                            check.detail,
                        )
                        for check in service_checks
                    )
                    compensation_verified = all(
                        check.ok for check in full_rollback_checks
                    )
            elif compensation_reboot_pending:
                (
                    launchers_quiesced,
                    post_audit,
                    _,
                ) = _quiesce_launchers_for_reboot(
                    args,
                    report,
                    runner,
                    service_guard,
                )
                services_restored = False
                if launchers_quiesced:
                    snapshot_path = report.rollback.path or ""
                    report.findings.append(
                        _rollback_reboot_recovery_finding(
                            report.rollback,
                            snapshot_path,
                            operation="install.compensation",
                        )
                    )
            else:
                _quiesce_launchers_for_reboot(
                    args,
                    report,
                    runner,
                    service_guard,
                )
                if service_guard is not None:
                    _record_intentionally_quiesced_services(
                        report,
                        service_guard,
                    )
                    services_restored = False
                    post_audit = audit_host(runner)
            if compensation_verified and services_restored:
                report.findings.append(
                    Finding(
                        "install.compensation.succeeded",
                        Severity.WARNING,
                        "Automatic rollback restored the pre-install baseline",
                        (
                            "Convergence failed, but the exact private snapshot "
                            "was reapplied and every modeled rollback invariant passed."
                        ),
                    )
                )
            elif not compensation_reboot_pending:
                failed_checks = [
                    check.name for check in compensation_checks if not check.ok
                ]
                report.findings.append(
                    Finding(
                        "install.compensation.verification-failed",
                        Severity.ERROR,
                        "Automatic rollback could not be verified",
                        (
                            "Failed rollback checks: " + ", ".join(failed_checks)
                            if failed_checks
                            else "Trusted service state was not safely restored."
                        ),
                        remediation=(
                            "Keep the node drained and use the original private "
                            "snapshot to complete and verify rollback."
                        ),
                    )
                )

        if service_guard is not None:
            report.command_results.extend(service_guard.results[service_result_offset:])
        operation_findings = [
            finding
            for finding in report.findings
            if finding.id.startswith(
                (
                    "gpu-services.",
                    "gpu-workloads.",
                    "install.",
                    "launcher-commit.",
                )
            )
        ]
        report.audit = post_audit
        report.findings = [*diagnose(desired, post_audit), *operation_findings]
        report.sbom = sbom_from_audit(post_audit)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _install_status(report)

    if args.command == "verify":
        if not apply_changes:
            runner.results = []
            verify_preparation = prepare_stack(runner, audit)
            report.command_results = list(runner.results)
            report.verification = [
                verify_preparation,
                *verify_stack(desired, runner, audit),
            ]
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return _verification_status(report)

        # Persist and durably bind the exact untouched baseline before the
        # maintenance gate can stop even one trusted service.
        try:
            report.rollback, snapshot_result = _create_snapshot_with_evidence(
                audit,
                desired,
                runner,
                persist=True,
                operation_id=report.operation_id,
                journal_report_path=(Path(out_path) if out_path is not None else None),
            )
        except RollbackSnapshotError as exc:
            report.findings.append(
                Finding(
                    "snapshot.failed",
                    Severity.ERROR,
                    "Pre-verification rollback snapshot failed",
                    str(exc),
                )
            )
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2
        report.command_results = [snapshot_result]
        try:
            restore_preflight = preflight_snapshot_restore_availability(
                report.rollback,
                runner,
            )
            _append_preflight_verification(
                report,
                "rollback.baseline-packages-available",
                restore_preflight,
                "Every exact baseline NVIDIA-stack package is reinstallable before mutation.",
            )
        except PackagePreflightError as exc:
            _append_preflight_verification(
                report,
                "rollback.baseline-packages-available",
                exc.results,
                "At least one exact baseline NVIDIA-stack package is unavailable.",
                ok=False,
            )
            report.findings.append(
                Finding(
                    "rollback.baseline-preflight.failed",
                    Severity.ERROR,
                    "Exact rollback baseline is unavailable",
                    str(exc),
                    evidence=exc.evidence(),
                    remediation=(
                        "Repair or restore the original private snapshot and its paired "
                        "retained payload bundle, or restart with a fresh snapshot before "
                        "any mutation."
                    ),
                )
            )
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2

        maintenance = _maintenance_gate(
            args,
            runner,
            audit,
            operation="applied verification",
        )
        report.command_results.extend(maintenance.command_results)
        report.findings.extend(maintenance.findings)
        service_guard = maintenance.guard
        if maintenance.findings or service_guard is None:
            if service_guard is not None:
                _, report.audit = _quarantine_failed_maintenance_gate(
                    report,
                    runner,
                    service_guard,
                )
            else:
                report.audit = audit_host(runner)
            report.sbom = sbom_from_audit(report.audit)
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2

        service_result_offset = len(service_guard.results)
        service_guard.mark_mutation_started()
        phase_succeeded, phase_audit, _ = _pre_gpu_mutation_checkpoint(
            args,
            report,
            runner,
            service_guard,
        )
        if phase_succeeded:
            baseline_checks = verify_rollback(
                report.rollback,
                phase_audit,
                include_service_state=False,
            )
            report.verification.extend(
                Verification(
                    f"verify.pre-mutation.{check.name}",
                    check.ok,
                    check.command,
                    check.detail,
                )
                for check in baseline_checks
            )
            failed_baseline_checks = [
                check.name for check in baseline_checks if not check.ok
            ]
            if failed_baseline_checks:
                report.findings.append(
                    Finding(
                        "verify.snapshot-baseline-changed",
                        Severity.ERROR,
                        "The verification baseline changed before mutation",
                        "Fresh snapshot checks failed: "
                        + ", ".join(failed_baseline_checks),
                        remediation=(
                            "Keep the node drained, investigate the concurrent "
                            "state change, and restart from a fresh snapshot."
                        ),
                    )
                )
                phase_succeeded = False
        if phase_succeeded:
            phase_succeeded = _fresh_gpu_boundary_is_safe(
                report,
                report.rollback,
                phase_audit,
                boundary="pre-mutation",
            )

        applied_preparation: Verification | None = None
        if phase_succeeded:
            # Re-derive module work from the fresh, fully quarantined audit.
            runner.results = []
            applied_preparation = prepare_stack(
                runner,
                phase_audit,
                force_reload=module_reload_required(phase_audit),
            )
            preparation_results = list(runner.results)
            report.command_results.extend(preparation_results)
            report.verification.append(applied_preparation)
            phase_succeeded = bool(
                _commands_succeeded(preparation_results) and applied_preparation.ok
            )

        core_audit = audit_host(runner)
        if phase_succeeded:
            phase_succeeded = _fresh_gpu_boundary_is_safe(
                report,
                report.rollback,
                core_audit,
                boundary="pre-service-commit",
            )
        core_checks = (
            verify_stack(
                desired,
                runner,
                core_audit,
                include_docker=False,
                include_fabric_manager=False,
            )
            if phase_succeeded
            else []
        )
        report.verification.extend(core_checks)
        core_verified = bool(core_checks) and all(check.ok for check in core_checks)
        services_restored = False
        post_audit = core_audit
        if phase_succeeded and core_verified:
            _append_launcher_release_authorization(
                report,
                out_path,
                report.rollback,
                release_target="operation-target",
            )
            services_restored, post_audit = _commit_rollback_service_activity(
                args,
                report,
                report.rollback,
                runner,
                core_audit,
                service_guard,
                operation="verify",
            )
            if services_restored:
                service_checks = verify_stack(desired, runner, post_audit)
                report.verification.extend(
                    check
                    for check in service_checks
                    if check.name.startswith(
                        ("docker.", "container.", "fabric-manager")
                    )
                )
        else:
            _defer_launcher_enablement(
                report,
                runner,
                audit_host(runner),
            )
            _record_intentionally_quiesced_services(report, service_guard)
            post_audit = audit_host(runner)

        if not services_restored and phase_succeeded and core_verified:
            _record_intentionally_quiesced_services(report, service_guard)
            post_audit = audit_host(runner)
        report.command_results.extend(service_guard.results[service_result_offset:])
        operation_findings = [
            finding
            for finding in report.findings
            if finding.id.startswith(
                (
                    "gpu-services.",
                    "gpu-workloads.",
                    "install.",
                    "launcher-commit.",
                    "verify.",
                )
            )
        ]
        report.audit = post_audit
        report.findings = [
            *diagnose(desired, post_audit),
            *operation_findings,
        ]
        report.sbom = sbom_from_audit(post_audit)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _verification_status(report)

    if args.command == "lock":
        report.plan = lock_actions(desired, audit)
        unsupported = [
            action for action in report.plan if action.id.startswith("unsupported.")
        ]
        if unsupported:
            emit_report(args.command, report, out_path, json_stdout, apply_changes)
            return 2
        if not report.plan:
            report.verification.append(
                Verification(
                    "package-policy.lock",
                    True,
                    detail="The observed package policy already matches desired state.",
                )
            )
            emit_report(args.command, report, out_path, json_stdout, apply_changes)
            return 0
        if not apply_changes:
            action_results = _run_plan_actions(report.plan, runner)
            report.command_results = action_results
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return _report_status(args.command, report)

        # Resolve the exact target policy before any maintenance mutation.
        lock_forward_packages: list[PackageInfo] = []
        try:
            preflight_results = preflight_package_lock(
                desired,
                audit,
                runner,
            )
            _append_preflight_verification(
                report,
                "packages.policy-preflight",
                preflight_results,
                "The requested package-policy selector is available.",
            )
            if audit.package_manager == "apt-get":
                lock_forward_packages = resolved_forward_payload_packages(
                    audit.package_manager,
                    preflight_results,
                )
        except PackagePreflightError as exc:
            _append_preflight_verification(
                report,
                "packages.policy-preflight",
                exc.results,
                "The requested package-policy selector is unavailable.",
                ok=False,
            )
            report.findings.append(
                Finding(
                    "packages.policy-preflight.failed",
                    Severity.ERROR,
                    "Package-policy preflight failed",
                    str(exc),
                    evidence=exc.evidence(),
                    remediation=(
                        "Refresh trusted repository metadata and repair the "
                        "package-policy backend before retrying."
                    ),
                )
            )
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2
        try:
            dnf_module_failsafe_path = _dnf_failsafe_path_from_preflight(
                desired,
                audit,
                preflight_results,
            )
            report.rollback, snapshot_result = _create_snapshot_with_evidence(
                audit,
                desired,
                runner,
                persist=True,
                operation_id=report.operation_id,
                journal_report_path=(Path(out_path) if out_path is not None else None),
                forward_packages=lock_forward_packages,
                dnf_module_failsafe_path=dnf_module_failsafe_path,
            )
        except RollbackSnapshotError as exc:
            report.findings.append(
                Finding(
                    "snapshot.failed",
                    Severity.ERROR,
                    "Pre-lock rollback snapshot failed",
                    str(exc),
                )
            )
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2
        report.command_results = [snapshot_result]
        if audit.package_manager == "apt-get":
            try:
                report.plan = _bind_forward_package_payloads(
                    report.plan,
                    report.rollback,
                    audit,
                )
                staged_lock_preflight = preflight_staged_forward_transaction(
                    desired,
                    report.rollback,
                    audit,
                    report.plan,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "packages.staged-local-policy-preflight",
                    staged_lock_preflight,
                    "The exact retained APT policy transaction resolves offline.",
                )
            except (PackagePayloadError, PackagePreflightError) as exc:
                evidence = (
                    exc.evidence() if isinstance(exc, PackagePreflightError) else {}
                )
                report.findings.append(
                    Finding(
                        "packages.staged-local-policy-preflight-failed",
                        Severity.ERROR,
                        "Staged local package-policy transaction failed",
                        str(exc),
                        evidence=evidence,
                    )
                )
                emit_report(
                    args.command,
                    report,
                    out_path,
                    json_stdout,
                    apply_changes,
                )
                return 2
        try:
            restore_preflight = preflight_snapshot_restore_availability(
                report.rollback,
                runner,
            )
            _append_preflight_verification(
                report,
                "rollback.baseline-packages-available",
                restore_preflight,
                "Every exact baseline NVIDIA-stack package is reinstallable before mutation.",
            )
        except PackagePreflightError as exc:
            _append_preflight_verification(
                report,
                "rollback.baseline-packages-available",
                exc.results,
                "At least one exact baseline NVIDIA-stack package is unavailable.",
                ok=False,
            )
            report.findings.append(
                Finding(
                    "rollback.baseline-preflight.failed",
                    Severity.ERROR,
                    "Exact rollback baseline is unavailable",
                    str(exc),
                    evidence=exc.evidence(),
                    remediation=(
                        "Repair or restore the original private snapshot and its paired "
                        "retained payload bundle, or restart with a fresh snapshot before "
                        "package-policy changes."
                    ),
                )
            )
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2

        maintenance = _maintenance_gate(
            args,
            runner,
            audit,
            operation="package-policy locking",
        )
        report.command_results.extend(maintenance.command_results)
        report.findings.extend(maintenance.findings)
        service_guard = maintenance.guard
        if maintenance.findings or service_guard is None:
            if service_guard is not None:
                _, report.audit = _quarantine_failed_maintenance_gate(
                    report,
                    runner,
                    service_guard,
                )
            else:
                report.audit = audit_host(runner)
            report.sbom = sbom_from_audit(report.audit)
            emit_report(
                args.command,
                report,
                out_path,
                json_stdout,
                apply_changes,
            )
            return 2

        service_result_offset = len(service_guard.results)
        service_guard.mark_mutation_started()
        phase_succeeded, phase_audit, _ = _pre_gpu_mutation_checkpoint(
            args,
            report,
            runner,
            service_guard,
        )
        if phase_succeeded:
            baseline_checks = verify_rollback(
                report.rollback,
                phase_audit,
                include_service_state=False,
            )
            report.verification.extend(
                Verification(
                    f"lock.pre-mutation.{check.name}",
                    check.ok,
                    check.command,
                    check.detail,
                )
                for check in baseline_checks
            )
            failed_baseline_checks = [
                check.name for check in baseline_checks if not check.ok
            ]
            if failed_baseline_checks:
                report.findings.append(
                    Finding(
                        "lock.snapshot-baseline-changed",
                        Severity.ERROR,
                        "The package-policy baseline changed before mutation",
                        "Fresh snapshot checks failed: "
                        + ", ".join(failed_baseline_checks),
                        remediation=(
                            "Keep the node drained, investigate the concurrent "
                            "state change, and restart from a fresh snapshot."
                        ),
                    )
                )
                phase_succeeded = False
        if phase_succeeded:
            phase_succeeded = _fresh_gpu_boundary_is_safe(
                report,
                report.rollback,
                phase_audit,
                boundary="pre-mutation",
            )

        if phase_succeeded:
            fresh_plan = lock_actions(desired, phase_audit)
            fresh_unsupported = [
                action for action in fresh_plan if action.id.startswith("unsupported.")
            ]
            if fresh_unsupported:
                report.findings.append(
                    Finding(
                        "lock.post-quarantine-plan-unsupported",
                        Severity.ERROR,
                        "Fresh package-policy plan is unsupported",
                        "; ".join(
                            action.reason or action.description
                            for action in fresh_unsupported
                        ),
                    )
                )
                phase_succeeded = False
            else:
                try:
                    report.plan = (
                        _bind_forward_package_payloads(
                            fresh_plan,
                            report.rollback,
                            phase_audit,
                        )
                        if phase_audit.package_manager == "apt-get"
                        else fresh_plan
                    )
                except PackagePayloadError as exc:
                    report.findings.append(
                        Finding(
                            "lock.post-quarantine-payload-binding-failed",
                            Severity.ERROR,
                            "Fresh policy payload binding failed",
                            str(exc),
                        )
                    )
                    phase_succeeded = False
        if phase_succeeded and report.plan:
            try:
                fresh_target_preflight = (
                    preflight_staged_forward_transaction(
                        desired,
                        report.rollback,
                        phase_audit,
                        report.plan,
                        runner,
                    )
                    if phase_audit.package_manager == "apt-get"
                    else preflight_package_lock(
                        desired,
                        phase_audit,
                        runner,
                        actions=report.plan,
                        authorized_failsafe_path=dnf_module_failsafe_path,
                    )
                )
                _append_preflight_verification(
                    report,
                    "packages.post-quarantine-policy-preflight",
                    fresh_target_preflight,
                    "Fresh snapshot-bound package-policy targets resolve after quarantine.",
                )
            except PackagePreflightError as exc:
                _append_preflight_verification(
                    report,
                    "packages.post-quarantine-policy-preflight",
                    exc.results,
                    "Fresh snapshot-bound package-policy targets no longer resolve.",
                    ok=False,
                )
                report.findings.append(
                    Finding(
                        "lock.post-quarantine-preflight-failed",
                        Severity.ERROR,
                        "Fresh package-policy preflight failed",
                        str(exc),
                        evidence=exc.evidence(),
                    )
                )
                phase_succeeded = False
        if phase_succeeded:
            try:
                fresh_rollback_preflight = preflight_package_rollback(
                    report.rollback,
                    phase_audit,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "rollback.post-quarantine-preflight",
                    fresh_rollback_preflight,
                    "The exact rollback delta remains resolvable after quarantine.",
                )
            except PackagePreflightError as exc:
                _append_preflight_verification(
                    report,
                    "rollback.post-quarantine-preflight",
                    exc.results,
                    "The exact rollback delta is no longer safely resolvable.",
                    ok=False,
                )
                report.findings.append(
                    Finding(
                        "lock.post-quarantine-rollback-preflight-failed",
                        Severity.ERROR,
                        "Fresh rollback package preflight failed",
                        str(exc),
                        evidence=exc.evidence(),
                    )
                )
                phase_succeeded = False

        if phase_succeeded and phase_audit.package_manager == "apt-get":
            try:
                immediate_lock_preflight = preflight_staged_forward_transaction(
                    desired,
                    report.rollback,
                    phase_audit,
                    report.plan,
                    runner,
                )
                _append_preflight_verification(
                    report,
                    "packages.immediate-local-policy-preflight",
                    immediate_lock_preflight,
                    "Retained APT policy bytes were revalidated immediately before mutation.",
                )
            except PackagePreflightError as exc:
                report.findings.append(
                    Finding(
                        "lock.immediate-local-policy-preflight-failed",
                        Severity.ERROR,
                        "Immediate local package-policy validation failed",
                        str(exc),
                        evidence=exc.evidence(),
                    )
                )
                phase_succeeded = False
        action_results = (
            _run_plan_actions(report.plan, runner) if phase_succeeded else []
        )
        report.command_results.extend(action_results)
        action_succeeded = bool(phase_succeeded and _commands_succeeded(action_results))
        post_audit = audit_host(runner)
        if action_succeeded:
            action_succeeded = _fresh_gpu_boundary_is_safe(
                report,
                report.rollback,
                post_audit,
                boundary="pre-service-commit",
            )
        remaining = lock_actions(desired, post_audit) if action_succeeded else []
        policy_ok = bool(action_succeeded and not remaining)
        detail = (
            "The applied package policy matches desired state."
            if policy_ok
            else (
                "; ".join(action.reason or action.description for action in remaining)
                if remaining
                else "Package-policy mutation did not complete safely."
            )
        )
        report.verification.append(
            Verification("package-policy.lock", policy_ok, detail=detail)
        )

        services_restored = False
        if policy_ok:
            _append_launcher_release_authorization(
                report,
                out_path,
                report.rollback,
                release_target="operation-target",
            )
            services_restored, post_audit = _commit_rollback_service_activity(
                args,
                report,
                report.rollback,
                runner,
                post_audit,
                service_guard,
                operation="lock",
            )
        else:
            _defer_launcher_enablement(
                report,
                runner,
                post_audit,
            )
            _record_intentionally_quiesced_services(report, service_guard)
            post_audit = audit_host(runner)
        if not services_restored and policy_ok:
            _record_intentionally_quiesced_services(report, service_guard)
            post_audit = audit_host(runner)
        report.command_results.extend(service_guard.results[service_result_offset:])
        operation_findings = [
            finding
            for finding in report.findings
            if finding.id.startswith(
                (
                    "gpu-services.",
                    "gpu-workloads.",
                    "install.",
                    "launcher-commit.",
                    "lock.",
                    "rollback.",
                )
            )
        ]
        report.audit = post_audit
        report.findings = [
            *diagnose(desired, post_audit),
            *operation_findings,
        ]
        report.sbom = sbom_from_audit(post_audit)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _report_status(args.command, report)

    return 1


def _add_common_args(
    parser: argparse.ArgumentParser, *, include_apply: bool = False
) -> None:
    parser.add_argument(
        "--desired", default=argparse.SUPPRESS, help="Desired-state JSON/YAML file."
    )
    parser.add_argument(
        "--out",
        default=argparse.SUPPRESS,
        help="Write machine-readable JSON report to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print the full machine-readable JSON report to stdout instead of the human summary.",
    )
    if include_apply:
        parser.add_argument(
            "--apply",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Apply host-mutating actions. Without this, mutating commands are dry-run.",
        )


def _add_disruption_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-disruption",
        action="store_true",
        help="Attest that the node is in a maintenance window and allow disruptive actions.",
    )
    parser.add_argument(
        "--allow-active-workloads",
        action="store_true",
        help="Also allow disruptive actions while active GPU processes, device users, or GPU-assigned Docker containers are observable.",
    )


def emit_report(
    command: str,
    report: Report,
    out_path: str | None,
    json_stdout: bool,
    apply_changes: bool,
) -> None:
    _finalize_report(command, report, out_path, apply_changes)
    if out_path:
        journal_integrity_uncertain = False
        try:
            write_report(report, out_path)
            if apply_changes:
                try:
                    append_report_journal(
                        Path(out_path),
                        report.operation_id,
                        "operation-completed",
                        exit_code=report.exit_code,
                        incomplete=report.incomplete,
                        outcome=report.outcome,
                    )
                    _MUTATING_OPERATION_IDS.discard(report.operation_id)
                except ReportWriteError as journal_error:
                    journal_integrity_uncertain = isinstance(
                        journal_error, ReportJournalIntegrityError
                    )
                    if report.operation_id in _MUTATING_OPERATION_IDS:
                        _emergency_quarantine_launchers()
                    report.exit_code = 2
                    report.outcome = "failed"
                    report.incomplete = True
                    write_report(report, out_path)
                    raise
        except ReportWriteError as exc:
            if report.operation_id in _MUTATING_OPERATION_IDS:
                _emergency_quarantine_launchers()
            report.exit_code = 2
            report.outcome = "failed"
            report.incomplete = apply_changes
            journal_integrity_error: ReportJournalIntegrityError | None = None
            if apply_changes and not journal_integrity_uncertain:
                try:
                    append_report_journal(
                        Path(out_path),
                        report.operation_id,
                        "report-persistence-failed",
                        error=str(exc),
                    )
                except ReportJournalIntegrityError as marker_error:
                    journal_integrity_error = marker_error
                except ReportWriteError:
                    pass
            fallback = (
                report_json(report)
                if json_stdout
                else render_human(command, report, apply=apply_changes)
            )
            print(fallback, file=sys.stderr)
            if journal_integrity_error is not None:
                raise journal_integrity_error from exc
            raise
    if json_stdout:
        print(report_json(report))
    else:
        print(render_human(command, report, apply=apply_changes))


def _finalize_report(
    command: str,
    report: Report,
    out_path: str | None,
    apply_changes: bool,
) -> None:
    completed_at = utc_now()
    report.command = command
    report.mode = "apply" if apply_changes else "dry-run"
    report.tool_version = __version__
    report.started_at = report.started_at or report.generated_at
    report.completed_at = completed_at
    try:
        started = datetime.fromisoformat(report.started_at)
        completed = datetime.fromisoformat(completed_at)
        report.duration_seconds = max(0.0, (completed - started).total_seconds())
    except ValueError:
        report.duration_seconds = None
    try:
        report.host_id = _host_identity()
    except RollbackSnapshotError:
        report.host_id = None
    report.report_path = out_path
    report.exit_code = _report_status(command, report)
    report.outcome = "succeeded" if report.exit_code == 0 else "failed"
    mutation_results = [
        *report.command_results,
        *[
            check.command
            for check in report.verification
            if check.command is not None
            and check.name in {"container.gpu", "module.load", "module.reload"}
        ],
    ]
    mutations_started = any(
        not result.skipped
        and result.returncode is not None
        and not _is_read_only_probe(result.command)
        for result in mutation_results
    )
    uncertain_reason = any(
        result.reason
        in {
            "timeout-process-group-terminated",
            "lingering-process-group-terminated",
        }
        for result in mutation_results
    )
    compensation_verified = any(
        finding.id == "install.compensation.succeeded" for finding in report.findings
    )
    report.incomplete = bool(
        apply_changes
        and report.exit_code != 0
        and (mutations_started or uncertain_reason)
        and not (compensation_verified and not uncertain_reason)
    )
    if command in {"install", "verify"} and report.verification:
        if report.exit_code == 0 and _verification_status(report) == 0:
            report.reboot_required = False
        elif any(
            finding.id == "install.compensation.reboot-required"
            for finding in report.findings
        ) or _verification_can_complete_after_reboot(report):
            report.reboot_required = True
        else:
            report.reboot_required = None
    elif command == "rollback" and report.verification:
        if report.exit_code == 0:
            report.reboot_required = False
        elif _rollback_can_complete_after_reboot(report):
            report.reboot_required = True
        else:
            report.reboot_required = None


def _provisional_operation_report(command: str, desired: DesiredState) -> Report:
    started_at = utc_now()
    report = Report(
        "1.2",
        started_at,
        desired,
        command=command,
        mode="apply",
        tool_version=__version__,
        started_at=started_at,
        completed_at=started_at,
        duration_seconds=0.0,
        outcome="failed",
        exit_code=255,
        incomplete=True,
    )
    try:
        report.host_id = _host_identity()
    except RollbackSnapshotError:
        report.host_id = None
    return report


def _inherit_operation(report: Report, operation_report: Report | None) -> None:
    if operation_report is None:
        return
    report.generated_at = operation_report.generated_at
    report.operation_id = operation_report.operation_id
    report.started_at = operation_report.started_at
    report.host_id = operation_report.host_id
    report.report_path = operation_report.report_path


def _is_workload_probe(command: list[str]) -> bool:
    return is_workload_probe(command)


def _is_read_only_probe(command: list[str]) -> bool:
    return (
        _is_workload_probe(command)
        or command[:2] == ["systemctl", "show"]
        or bool(
            command
            and command[0]
            in {
                "validate-trusted-gpu-service",
                "validate-trusted-gpu-service-start",
                "validate-active-trusted-gpu-service",
            }
        )
        or bool(
            command
            and command[0] in {"inspect-module-dependencies", "rollback-precondition"}
        )
    )


def _report_status(command: str, report: Report) -> int:
    if command == "doctor":
        return 0 if all(f.severity.value != "error" for f in report.findings) else 2
    if command == "plan":
        return _plan_status(report)
    if command == "install":
        return _install_status(report)
    if command == "verify":
        return _verification_status(report)
    if command == "lock":
        if any(action.id.startswith("unsupported.") for action in report.plan):
            return 2
        if any(not check.ok for check in report.verification):
            return 2
        if any(
            finding.id
            in {
                "gpu-workloads.active",
                "gpu-workloads.unknown",
                "maintenance-window.required",
                "snapshot.failed",
            }
            for finding in report.findings
        ):
            return 2
        return _status_from_results(report.command_results)
    if command == "snapshot":
        return 0 if report.rollback is not None else 2
    if command == "rollback":
        checks_ok = all(check.ok for check in report.verification)
        findings_ok = all(
            finding.severity.value != "error" for finding in report.findings
        )
        return (
            0
            if _commands_succeeded(report.command_results) and checks_ok and findings_ok
            else 2
        )
    return 2


def emit_validation(
    desired: DesiredState, out_path: str | None, json_stdout: bool
) -> None:
    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "valid": True,
        "desired": asdict(desired),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out_path:
        try:
            atomic_write_text(Path(out_path), text + "\n")
        except OSError as exc:
            raise ReportWriteError(f"cannot write report {out_path!r}: {exc}") from exc
    if json_stdout:
        print(text)
        return
    print("nvidia-converge validate")
    print("Desired state: valid")
    print(f"Driver: {desired.driver}")
    print(f"CUDA compat: {desired.cuda_compat}")
    print(f"Container runtime: {desired.container_runtime}")
    print("Use --json to print machine-readable validation details.")


def _requires_root(command: str) -> bool:
    return command in {"install", "lock", "rollback", "snapshot", "verify"}


def _bind_forward_package_payloads(
    actions: list[PlanAction],
    snapshot: RollbackSnapshot,
    audit: HostAudit,
) -> list[PlanAction]:
    if snapshot.path is None or snapshot.package_payloads is None:
        raise PackagePayloadError(
            "applicable install snapshot has no retained package payloads"
        )
    has_forward_payloads = any(
        "forward" in payload.roles
        for payload in snapshot.package_payloads.packages
    )
    has_install_action = any(action.id == "install.packages" for action in actions)
    has_apt_lock_action = any(action.id == "lock.apt" for action in actions)
    requires_forward_payloads = has_install_action or has_apt_lock_action
    if requires_forward_payloads != has_forward_payloads:
        raise PackagePayloadError(
            "resolved forward payloads do not match the fresh install plan"
        )
    if not requires_forward_payloads:
        return actions
    if snapshot.package_manager is None:
        raise PackagePayloadError("snapshot has no package manager")
    if snapshot.package_manager == "apt-get":
        remove_specs: list[str] = []
        for action in actions:
            if action.id != "lock.apt":
                continue
            if len(action.commands) != 1 or "--purge" not in action.commands[0]:
                raise PackagePayloadError(
                    "APT policy action is not one recognized atomic transaction"
                )
            remove_specs.extend(
                operand[:-1]
                for operand in action.commands[0][
                    action.commands[0].index("--purge") + 1 :
                ]
                if operand.endswith("-")
            )
        command = forward_package_command(
            Path(snapshot.path),
            snapshot.package_payloads,
            snapshot.package_manager,
            remove_specs=remove_specs,
        )
        anchor = "lock.apt" if has_apt_lock_action else "install.packages"
        return [
            replace(action, commands=[command])
            if action.id == anchor
            else replace(action, commands=[])
            if action.id == "install.packages" and anchor == "lock.apt"
            else action
            for action in actions
        ]
    if snapshot.package_manager in {"dnf", "yum"}:
        forward_entries = [
            entry
            for entry in snapshot.package_payloads.packages
            if "forward" in entry.roles
        ]
        expected_installs = sorted(
            f"{entry.name}-{f'{entry.epoch}:' if entry.epoch else ''}"
            f"{entry.version}.{entry.architecture}"
            for entry in forward_entries
        )
        forward_identities = {
            (entry.name, entry.architecture, entry.epoch, entry.version)
            for entry in forward_entries
        }
        forward_slots = {
            (entry.name, entry.architecture) for entry in forward_entries
        }
        expected_removals = sorted(
            f"{package.name}-{f'{package.epoch}:' if package.epoch else ''}"
            f"{package.version}.{package.architecture}"
            for package in audit.packages
            if package.installed
            and package.manager == "rpm"
            and package.version
            and package.architecture
            and (package.name, package.architecture) in forward_slots
            and (
                package.name,
                package.architecture,
                package.epoch,
                package.version,
            )
            not in forward_identities
        )
        command = dnf_local_transaction_command(
            apply=True,
            restore_paths=local_payload_paths(
                Path(snapshot.path),
                snapshot.package_payloads,
                role="forward",
            ),
            remove_specs=[],
            expected_installs=expected_installs,
            expected_removals=expected_removals,
        )
    else:
        command = forward_package_command(
            Path(snapshot.path),
            snapshot.package_payloads,
            snapshot.package_manager,
        )
    return [
        replace(action, commands=[command])
        if action.id == "install.packages"
        else action
        for action in actions
    ]


def _run_plan_actions(
    actions: list[PlanAction],
    runner: CommandRunner,
    *,
    skip_action_ids: set[str] | None = None,
) -> list[CommandResult]:
    skipped = skip_action_ids or set()
    runner.results = []
    for action in actions:
        if action.id in skipped:
            continue
        for command in action.commands:
            result = runner.run(command, mutate=True, allow_fail=True)
            if result.returncode not in (0, None):
                return list(runner.results)
    return list(runner.results)


def _plan_starts_or_restarts_docker(actions: list[PlanAction]) -> bool:
    for action in actions:
        for command in action.commands:
            if (
                not command
                or command[0] != "systemctl"
                or not any(
                    item in {"docker.service", "docker.socket"} for item in command
                )
            ):
                continue
            if "restart" in command or "start" in command or "--now" in command:
                return True
    return False


def _post_docker_workload_checkpoint(
    args: argparse.Namespace,
    report: Report,
    runner: CommandRunner,
    *,
    operation: str = "install",
) -> tuple[bool, HostAudit]:
    checkpoint_audit = audit_host(runner)
    if checkpoint_audit.docker_service_active is not True:
        report.findings.append(
            Finding(
                f"{operation}.docker-checkpoint-unavailable",
                Severity.ERROR,
                "Docker did not reach a proven active state",
                (
                    "The planned Docker start/restart completed without an "
                    "observable active daemon, so module preparation and "
                    "container verification are blocked."
                ),
                remediation=(
                    "Keep the node drained, repair Docker service state, and "
                    "apply the original private snapshot before retrying."
                ),
            )
        )
        return False, checkpoint_audit
    probe, active_workloads = _probe_active_gpu_workloads(
        runner,
        checkpoint_audit,
    )
    report.command_results.append(probe)
    if active_workloads is None:
        report.findings.append(
            Finding(
                f"{operation}.docker-checkpoint-workloads-unknown",
                Severity.ERROR,
                "GPU workloads became unobservable after Docker started",
                (
                    "The required post-start workload checkpoint could not prove "
                    "that queued GPU containers remained stopped."
                ),
                remediation=(
                    "Keep the node drained and restore complete process/container "
                    "observability before retrying."
                ),
            )
        )
        return False, checkpoint_audit
    if active_workloads and not args.allow_active_workloads:
        report.findings.append(
            Finding(
                f"{operation}.docker-checkpoint-workloads-active",
                Severity.ERROR,
                "Queued GPU workloads started with Docker",
                "Observed GPU workloads: " + ", ".join(active_workloads),
                remediation=(
                    "Drain or disable the queued containers before retrying, or "
                    "explicitly acknowledge them with --allow-active-workloads."
                ),
            )
        )
        return False, checkpoint_audit
    return True, checkpoint_audit


def _quiesce_launchers_for_reboot(
    args: argparse.Namespace,
    report: Report,
    runner: CommandRunner,
    guard: TrustedGpuServiceGuard | None,
) -> tuple[bool, HostAudit, bool]:
    """Keep GPU launchers inactive across a required reboot boundary."""

    return _pre_gpu_mutation_checkpoint(
        args,
        report,
        runner,
        guard,
    )


def _quarantine_failed_maintenance_gate(
    report: Report,
    runner: CommandRunner,
    guard: TrustedGpuServiceGuard,
) -> tuple[bool, HostAudit]:
    """Persistently quarantine after a gate that already stopped a service."""

    del guard
    try:
        return _defer_launcher_enablement(
            report,
            runner,
            audit_host(runner),
        )
    except BaseException:
        _emergency_quarantine_launchers()
        raise


def _defer_launcher_enablement(
    report: Report,
    runner: CommandRunner,
    audit: HostAudit,
) -> tuple[bool, HostAudit]:
    """Persistently stop, disable, and mask every transactional launcher."""

    unit_file_states = (
        ("docker.socket", audit.docker_socket_unit_file_state),
        ("docker.service", audit.docker_service_unit_file_state),
        (
            "nvidia-persistenced.service",
            audit.nvidia_persistenced_unit_file_state,
        ),
        (
            "nvidia-fabricmanager.service",
            audit.fabric_manager_unit_file_state,
        ),
    )
    unknown = [unit for unit, state in unit_file_states if state is None]
    if unknown:
        report.findings.append(
            Finding(
                "install.launcher-enablement-unknown",
                Severity.ERROR,
                "Launcher unit-file state is unknown at a transaction boundary",
                "Unknown units: " + ", ".join(unknown),
            )
        )
        return False, audit
    for unit in (
        "docker.service",
        "nvidia-persistenced.service",
        "nvidia-fabricmanager.service",
    ):
        validation_results, error = validate_trusted_gpu_service_unit(
            runner,
            unit,
            allow_masked=True,
        )
        report.command_results.extend(validation_results)
        if error is not None:
            report.findings.append(
                Finding(
                    "install.launcher-quarantine-service-untrusted",
                    Severity.ERROR,
                    "Active launcher cannot be safely stopped",
                    error,
                    evidence={"unit": unit},
                )
            )
            return False, audit
    socket_results, socket_error = validate_trusted_docker_socket_unit(
        runner,
        allow_masked=True,
    )
    report.command_results.extend(socket_results)
    if socket_error is not None:
        report.findings.append(
            Finding(
                "install.launcher-quarantine-socket-untrusted",
                Severity.ERROR,
                "Docker socket cannot be safely quarantined",
                socket_error,
                evidence={"unit": "docker.socket"},
            )
        )
        return False, audit
    failures: list[CommandResult] = []
    for unit, _ in unit_file_states:
        quarantine_results = _quarantine_service_for_rollback(
            runner,
            unit,
        )
        report.command_results.extend(quarantine_results)
        unit_failures = [
            result
            for result in quarantine_results
            if result.returncode not in (0, None)
        ]
        failures.extend(unit_failures)
        if unit_failures:
            break
    audit = audit_host(runner)
    observed = {
        "docker.socket": (
            audit.docker_socket_active,
            audit.docker_socket_unit_file_state,
        ),
        "docker.service": (
            audit.docker_service_active,
            audit.docker_service_unit_file_state,
        ),
        "nvidia-persistenced.service": (
            audit.nvidia_persistenced_active,
            audit.nvidia_persistenced_unit_file_state,
        ),
        "nvidia-fabricmanager.service": (
            audit.fabric_manager_active,
            audit.fabric_manager_unit_file_state,
        ),
    }
    unsafe = {
        unit: {"active": active, "unit_file_state": state}
        for unit, (active, state) in observed.items()
        if active is not False or state != "masked"
    }
    if unknown or failures or unsafe:
        report.findings.append(
            Finding(
                "install.launcher-quarantine-unverified",
                Severity.ERROR,
                "Launchers are not persistently quarantined",
                (
                    "Every launcher must be observably inactive and masked "
                    "before the transaction can continue."
                ),
                evidence={
                    "unsafe_units": unsafe,
                    "failed_commands": [
                        {
                            "command": result.command,
                            "returncode": result.returncode,
                            "stderr": result.stderr,
                        }
                        for result in failures
                    ],
                },
            )
        )
        return False, audit
    return True, audit


def _pre_gpu_mutation_checkpoint(
    args: argparse.Namespace,
    report: Report,
    runner: CommandRunner,
    guard: TrustedGpuServiceGuard | None,
) -> tuple[bool, HostAudit, bool]:
    """Close package-script/service races before module or MIG mutation."""

    trusted_services_quiesced = _requiesce_services_for_compensation(
        report,
        guard,
    )
    checkpoint_audit = audit_host(runner)
    docker_state_unknown = bool(
        checkpoint_audit.docker_socket_active is None
        or checkpoint_audit.docker_service_active is None
    )
    if docker_state_unknown:
        report.findings.append(
            Finding(
                "install.pre-gpu-docker-state-unknown",
                Severity.ERROR,
                "Docker state is unknown before GPU mutation",
                (
                    "Package maintainer scripts may have changed Docker, but the "
                    "socket/service active states could not both be observed."
                ),
                remediation=(
                    "Keep the node drained and restore systemd observability before retrying."
                ),
            )
        )

    probe, active_workloads = _probe_active_gpu_workloads(
        runner,
        checkpoint_audit,
    )
    report.command_results.append(probe)
    initially_unknown = active_workloads is None
    initially_blocked = bool(active_workloads and not args.allow_active_workloads)
    docker_stopped = bool(
        checkpoint_audit.docker_socket_active is True
        or checkpoint_audit.docker_service_active is True
    )

    enablement_deferred, checkpoint_audit = _defer_launcher_enablement(
        report,
        runner,
        checkpoint_audit,
    )
    reprobe, remaining_workloads = _probe_active_gpu_workloads(
        runner,
        checkpoint_audit,
    )
    report.command_results.append(reprobe)
    if remaining_workloads is None:
        initially_unknown = True
    elif remaining_workloads and not args.allow_active_workloads:
        initially_blocked = True

    if initially_unknown:
        report.findings.append(
            Finding(
                "install.pre-gpu-workloads-unknown",
                Severity.ERROR,
                "GPU workloads are not fully observable before mutation",
                (
                    "The package/module boundary could not prove an idle GPU, so "
                    "MIG and module preparation are refused."
                ),
            )
        )
    if initially_blocked:
        report.findings.append(
            Finding(
                "install.pre-gpu-workloads-active",
                Severity.ERROR,
                "GPU workloads appeared before module/MIG mutation",
                "Observed GPU workloads were not explicitly acknowledged.",
                remediation=(
                    "Drain queued containers/processes or rerun with "
                    "--allow-active-workloads after assessing disruption."
                ),
            )
        )
    return (
        bool(
            trusted_services_quiesced
            and not docker_state_unknown
            and enablement_deferred
            and not initially_unknown
            and not initially_blocked
        ),
        checkpoint_audit,
        docker_stopped,
    )


def _docker_phase_actions(
    actions: list[PlanAction],
    audit: HostAudit,
    *,
    docker_was_stopped: bool,
) -> list[PlanAction]:
    docker_actions: list[PlanAction] = []
    starts_docker = False
    for action in actions:
        if action.id not in {"service.docker", "configure.docker-runtime"}:
            continue
        commands = [
            command
            for command in action.commands
            if not (
                (
                    audit.docker_service_active is False
                    and command[:3] == ["systemctl", "stop", "docker.service"]
                )
                or (
                    command
                    and command[0] == "systemctl"
                    and "docker.service" in command
                    and ("enable" in command or "disable" in command)
                )
            )
        ]
        starts_docker = starts_docker or _commands_start_or_restart_docker(commands)
        docker_actions.append(
            PlanAction(
                action.id,
                action.description,
                commands,
                action.destructive,
                action.reason,
            )
        )
    if (
        docker_was_stopped
        and audit.docker_service_active is False
        and not starts_docker
    ):
        docker_actions.append(
            PlanAction(
                "service.docker-restore",
                "Restore the originally active Docker daemon after GPU mutation.",
                [["systemctl", "start", "docker.service"]],
                destructive=True,
                reason="Restores the captured active service baseline exactly once.",
            )
        )
    return docker_actions


def _docker_configuration_actions(
    actions: list[PlanAction],
) -> list[PlanAction]:
    """Keep Docker configuration mutations while deferring every launcher verb."""

    configured: list[PlanAction] = []
    for action in actions:
        if action.id not in {"service.docker", "configure.docker-runtime"}:
            continue
        commands = [
            command
            for command in action.commands
            if not (command and command[0] == "systemctl")
        ]
        if commands:
            configured.append(replace(action, commands=commands))
    return configured


def _replace_applied_mig_plan(
    plan: list[PlanAction],
    replacements: list[PlanAction],
) -> None:
    mig_ids = {
        "disable.mig",
        "enable.mig",
        "configure.mig-geometry",
        "reconcile.mig-after-module",
    }
    indexes = [index for index, action in enumerate(plan) if action.id in mig_ids]
    insertion = (
        indexes[0]
        if indexes
        else next(
            (
                index
                for index, action in enumerate(plan)
                if action.id in {"service.docker", "configure.docker-runtime"}
            ),
            len(plan),
        )
    )
    plan[:] = [action for action in plan if action.id not in mig_ids]
    for offset, action in enumerate(replacements):
        plan.insert(insertion + offset, action)


def _replace_applied_teardown_plan(
    plan: list[PlanAction],
    replacements: list[PlanAction],
) -> None:
    """Bind the reported teardown to the fresh post-package geometry."""

    action_id = "prepare.mig-geometry-teardown"
    indexes = [index for index, action in enumerate(plan) if action.id == action_id]
    insertion = (
        indexes[0]
        if indexes
        else next(
            (
                index
                for index, action in enumerate(plan)
                if action.id == "prepare.module"
            ),
            len(plan),
        )
    )
    plan[:] = [action for action in plan if action.id != action_id]
    for offset, action in enumerate(replacements):
        plan.insert(insertion + offset, action)


def _fresh_gpu_boundary_is_safe(
    report: Report,
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    *,
    boundary: str,
) -> bool:
    """Require a complete, UUID-stable inventory at every mutation boundary."""

    if not audit.package_inventory_complete:
        report.findings.append(
            Finding(
                f"install.{boundary}.package-inventory-incomplete",
                Severity.ERROR,
                "Fresh package inventory is incomplete",
                (
                    f"The {boundary} audit cannot prove the exact installed "
                    "package state, so the transaction cannot continue."
                ),
            )
        )
        return False
    if boundary == "pre-package":
        baseline_checks = verify_rollback(
            snapshot,
            audit,
            include_service_state=False,
        )
        report.verification.extend(
            Verification(
                f"install.pre-package.{check.name}",
                check.ok,
                check.command,
                check.detail,
            )
            for check in baseline_checks
        )
        failed = [check.name for check in baseline_checks if not check.ok]
        if failed:
            report.findings.append(
                Finding(
                    "install.pre-package.snapshot-baseline-changed",
                    Severity.ERROR,
                    "The rollback baseline changed before host mutation",
                    (
                        "The fresh post-quarantine audit no longer matches the "
                        "private snapshot: " + ", ".join(failed)
                    ),
                    remediation=(
                        "Keep the node drained, investigate the concurrent state "
                        "change, and restart convergence from a fresh snapshot."
                    ),
                )
            )
            return False
    if not snapshot.gpu_uuids or audit.gpu_uuids != snapshot.gpu_uuids:
        report.findings.append(
            Finding(
                f"install.{boundary}.gpu-inventory-changed",
                Severity.ERROR,
                "GPU identity changed during convergence",
                (
                    f"The {boundary} GPU UUID inventory differs from the "
                    "private rollback snapshot."
                ),
                evidence={
                    "snapshot": snapshot.gpu_uuids,
                    "observed": audit.gpu_uuids,
                },
            )
        )
        return False
    if boundary in {
        "pre-package",
        "post-package",
        "pre-mutation",
        "post-module",
        "pre-service-commit",
        "rollback-pre-apply",
    } and (
        audit.docker_socket_active is not False
        or audit.docker_service_active is not False
        or audit.nvidia_persistenced_active is not False
        or audit.fabric_manager_active is not False
        or audit.docker_socket_unit_file_state != "masked"
        or audit.docker_service_unit_file_state != "masked"
        or audit.nvidia_persistenced_unit_file_state != "masked"
        or audit.fabric_manager_unit_file_state != "masked"
    ):
        report.findings.append(
            Finding(
                "install.pre-service-commit.docker-not-quiesced",
                Severity.ERROR,
                "Launchers are not quarantined before core verification",
                (
                    "Every launcher must be observably inactive and masked "
                    "before launcher commit begins."
                ),
            )
        )
        return False
    return True


def _verification_phase_checks(
    checks: list[Verification],
    *,
    phase: str,
) -> list[Verification]:
    if phase == "fabric-manager":
        return [check for check in checks if check.name.startswith("fabric-manager")]
    if phase == "docker":
        return [
            check
            for check in checks
            if check.name.startswith(("docker.", "container."))
        ]
    raise ValueError(f"unknown verification phase: {phase}")


def _execute_mig_reconciliation(
    desired: DesiredState,
    runner: CommandRunner,
    audit: HostAudit,
    actions: list[PlanAction],
) -> tuple[bool, HostAudit, bool, list[CommandResult]]:
    """Execute MIG commands across fresh current/pending-mode checkpoints."""

    results: list[CommandResult] = []
    expected_gpu_uuids = list(audit.gpu_uuids)
    current_audit = audit
    for action in actions:
        for command in action.commands:
            if (
                desired.mig == "enabled"
                and "-cgi" in command
                and current_audit.mig_mode != "enabled"
            ):
                if (
                    current_audit.mig_mode_pending == desired.mig
                    and current_audit.mig_mode != desired.mig
                ):
                    return True, current_audit, True, results
                results.append(
                    CommandResult(
                        ["mig-reconciliation-checkpoint"],
                        1,
                        stderr=(
                            "refusing to create MIG geometry while current MIG "
                            "mode is not enabled"
                        ),
                    )
                )
                return False, current_audit, False, results

            result = runner.run(command, mutate=True, allow_fail=True)
            results.append(result)
            if result.returncode not in (0, None):
                return False, current_audit, False, results

            current_audit = audit_host(runner)
            if (
                not current_audit.package_inventory_complete
                or current_audit.gpu_uuids != expected_gpu_uuids
            ):
                results.append(
                    CommandResult(
                        ["mig-reconciliation-checkpoint"],
                        1,
                        stderr=(
                            "fresh MIG checkpoint lost complete package/GPU "
                            "identity evidence"
                        ),
                    )
                )
                return False, current_audit, False, results

            mode_command = "-mig" in command
            if mode_command:
                if (
                    current_audit.mig_mode == desired.mig
                    and current_audit.mig_mode_pending == desired.mig
                ):
                    continue
                if (
                    current_audit.mig_mode != desired.mig
                    and current_audit.mig_mode_pending == desired.mig
                ):
                    # The mode change is staged but not current. Geometry commands
                    # are invalid until reboot, so stop at this exact boundary.
                    return True, current_audit, True, results
                results.append(
                    CommandResult(
                        ["mig-reconciliation-checkpoint"],
                        1,
                        stderr=(
                            "MIG mode command did not produce the desired current "
                            "or pending state"
                        ),
                    )
                )
                return False, current_audit, False, results

    reboot_pending = bool(
        current_audit.mig_mode != desired.mig
        and current_audit.mig_mode_pending == desired.mig
    )
    target_observed = bool(
        current_audit.mig_mode == desired.mig
        and current_audit.mig_mode_pending == desired.mig
    )
    return target_observed or reboot_pending, current_audit, reboot_pending, results


def _fresh_module_reset_mig_teardown(
    audit: HostAudit,
    *,
    module_reset_planned: bool,
) -> list[PlanAction]:
    if not module_reset_planned or audit.mig_mode != "enabled":
        return []
    if len(audit.gpu_uuids) != 1:
        return [
            PlanAction(
                "unsupported.module-reset-mig-transaction-scope",
                "Cannot reset the module under the fresh MIG topology.",
                [],
                reason="Expected exactly one unchanged GPU UUID after packages.",
            )
        ]
    if not audit.mig_geometry_complete:
        return [
            PlanAction(
                "unsupported.module-reset-mig-geometry-unobservable",
                "Cannot reset the module without fresh exact MIG geometry.",
                [],
                reason="Post-package GI/CI observation is incomplete.",
            )
        ]
    if not restorable_mig_geometry(audit.mig_geometry, audit.gpu_uuids):
        return [
            PlanAction(
                "unsupported.module-reset-mig-rollback-geometry",
                "Cannot restore the fresh MIG geometry after module reset.",
                [],
                reason="Post-package MIG geometry is outside the exact rollback model.",
            )
        ]
    if not audit.mig_geometry:
        return []
    return [
        PlanAction(
            "prepare.mig-geometry-teardown",
            "Destroy fresh UUID-bound MIG geometry before module reset.",
            mig_geometry_destroy_commands(audit.gpu_uuids[0]),
            destructive=True,
            reason="Derived only from the post-package audit used for mutation.",
        )
    ]


def _commands_start_or_restart_docker(commands: list[list[str]]) -> bool:
    return any(
        command
        and command[0] == "systemctl"
        and any(item in {"docker.service", "docker.socket"} for item in command)
        and ("start" in command or "restart" in command or "--now" in command)
        for command in commands
    )


def _preparation_failure_can_complete_after_reboot(
    desired: DesiredState,
    audit: HostAudit,
    preparation: Verification,
) -> bool:
    command = preparation.command
    if preparation.ok or command is None:
        return False
    on_disk_ready = bool(
        audit.package_inventory_complete
        and desired.matches_driver_version(audit.module.installed_version)
        and audit.module.installed_open_module is desired.open_kernel_module
        and (desired.secure_boot != "signed" or audit.module.installed_signed is True)
    )
    mig_mode_ready_after_reboot = bool(
        audit.mig_mode in {"enabled", "disabled"}
        and audit.mig_mode_pending == desired.mig
        and (desired.mig != "enabled" or audit.mig_capable is True)
    )
    mig_geometry_ready = True
    if desired.mig == "enabled" and audit.mig_mode == "enabled":
        # An unload failure after explicit GI/CI teardown is not reboot
        # resolvable: rebooting does not recreate desired MIG geometry.
        mig_geometry_ready = bool(
            audit.mig_geometry_complete
            and full_mig_geometry_matches(
                audit.mig_geometry,
                audit.gpu_uuids,
            )
            and len(audit.mig_device_uuids) == 1
        )
    return bool(
        on_disk_ready
        and mig_mode_ready_after_reboot
        and mig_geometry_ready
        and _reboot_resolvable_module_command_failure(command)
    )


def _dnf_failsafe_path_from_preflight(
    desired: DesiredState,
    audit: HostAudit,
    results: list[CommandResult],
) -> str | None:
    if audit.package_manager != "dnf":
        return None
    stream = (
        f"{desired.driver_major}-"
        f"{'open' if desired.open_kernel_module else 'dkms'}"
    )
    if len(results) != 1:
        raise RollbackSnapshotError(
            "DNF policy preflight did not produce one rollback-target proof"
        )
    proof = parse_dnf_module_enable_proof(
        results[0],
        applied=False,
        stream=stream,
    )
    if proof is None:
        raise RollbackSnapshotError(
            "DNF policy preflight proof cannot authorize a rollback target"
        )
    return DNF_MODULE_FAILSAFE_DIRECTORY + "/" + proof.failsafe_filename


def _create_snapshot_with_evidence(
    audit: HostAudit,
    desired: DesiredState,
    runner: CommandRunner,
    *,
    persist: bool,
    operation_id: str | None = None,
    journal_report_path: Path | None = None,
    forward_packages: list[PackageInfo] | None = None,
    dnf_module_failsafe_path: str | None = None,
) -> tuple[RollbackSnapshot, CommandResult]:
    package_payloads = None
    snapshot_path: Path | None = None
    if persist:
        if operation_id is None or journal_report_path is None:
            raise ReportWriteError(
                "an applied rollback snapshot requires a durable operation binding"
            )
        if audit.package_manager is None:
            raise RollbackSnapshotError(
                "cannot stage package payloads without a supported package manager"
            )
        snapshot_path = new_snapshot_path(operation_id=operation_id)
        stage_command = ["stage-package-payloads"]
        try:
            with runner.private_state_scope(stage_command):
                package_payloads = stage_package_payloads(
                    snapshot_path,
                    [package for package in audit.packages if package.installed],
                    audit.package_manager,
                    _PayloadStagingRunner(runner),
                    forward_packages=forward_packages,
                    required_owner_uid=os.geteuid(),
                )
        except PackagePayloadError as exc:
            raise RollbackSnapshotError(
                f"cannot retain package payloads before snapshot binding: {exc}"
            ) from exc
    command = ["persist-rollback-snapshot"]
    runner.record_external_start(command, persist)
    try:
        snapshot = create_snapshot(
            audit,
            path=str(snapshot_path) if snapshot_path is not None else None,
            desired=desired,
            persist=persist,
            operation_id=operation_id,
            package_payloads=package_payloads,
            dnf_module_failsafe_path=dnf_module_failsafe_path,
        )
    except RollbackSnapshotError as exc:
        result = CommandResult(command, 1, stderr=str(exc))
        runner.record_external_result(result, persist)
        raise
    result = CommandResult(
        command,
        0 if persist else None,
        skipped=not persist,
        reason=None if persist else "dry-run",
    )
    runner.record_external_result(result, persist)
    if persist:
        assert operation_id is not None
        assert journal_report_path is not None
        _append_snapshot_binding(
            None,
            journal_report_path,
            snapshot,
            operation_id=operation_id,
        )
    return snapshot, result


def _append_snapshot_binding(
    report: Report | None,
    report_path: str | Path | None,
    snapshot: RollbackSnapshot,
    *,
    snapshot_path: str | None = None,
    operation_id: str | None = None,
) -> None:
    """Durably bind one exact private snapshot before host mutation."""

    bound_operation_id = operation_id or (
        report.operation_id if report is not None else None
    )
    bound_path = snapshot_path or snapshot.path
    if (
        report_path is None
        or bound_operation_id is None
        or bound_path is None
        or snapshot.integrity_sha256 is None
        or not Path(bound_path).is_absolute()
    ):
        raise ReportWriteError(
            "cannot durably bind an incomplete rollback snapshot authority"
        )
    append_report_journal(
        Path(report_path),
        bound_operation_id,
        "rollback-snapshot-persisted",
        snapshot_path=bound_path,
        snapshot_integrity_sha256=snapshot.integrity_sha256,
        snapshot_operation_id=snapshot.operation_id,
        snapshot_host_id=snapshot.host_id,
    )


def _mark_operations_recovered(
    operations: list[UnresolvedOperation],
    *,
    recovery_operation_id: str,
    snapshot: RollbackSnapshot,
) -> None:
    """Close earlier journals only after exact rollback verification passes."""

    for operation in operations:
        suffix = ".journal.jsonl"
        if not operation.journal_path.name.endswith(suffix):
            raise ReportWriteError("cannot resolve an unsafe recovery journal path")
        report_path = operation.journal_path.with_name(
            operation.journal_path.name[: -len(suffix)]
        )
        append_report_journal(
            report_path,
            operation.operation_id,
            "operation-recovered",
            recovery_operation_id=recovery_operation_id,
            snapshot_path=snapshot.path,
            snapshot_integrity_sha256=snapshot.integrity_sha256,
            snapshot_operation_id=snapshot.operation_id,
            snapshot_host_id=snapshot.host_id,
        )


def _append_launcher_release_authorization(
    report: Report,
    report_path: str | None,
    snapshot: RollbackSnapshot,
    *,
    release_target: str,
) -> None:
    """Seal verified core state before the first launcher is unmasked."""

    # Applied CLI entry always supplies the reserved private report path. Unit
    # callers that exercise the executor directly have no durable journal.
    if report_path is None:
        return
    if snapshot.path is None or snapshot.integrity_sha256 is None:
        raise ReportWriteError(
            "cannot authorize launcher release without exact snapshot evidence"
        )
    append_report_journal(
        Path(report_path),
        report.operation_id,
        "launcher-release-authorized",
        release_target=release_target,
        snapshot_path=snapshot.path,
        snapshot_integrity_sha256=snapshot.integrity_sha256,
        snapshot_operation_id=snapshot.operation_id,
        snapshot_host_id=snapshot.host_id,
    )


def _emergency_quarantine_launchers() -> None:
    """Best-effort persistent fail-closed state after lost journal evidence."""

    emergency_runner = CommandRunner(apply=True)
    for unit in (
        "docker.socket",
        "docker.service",
        "nvidia-persistenced.service",
        "nvidia-fabricmanager.service",
    ):
        emergency_runner.run(
            ["systemctl", "mask", "--now", unit],
            mutate=True,
            allow_fail=True,
        )


def _append_preflight_verification(
    report: Report,
    name: str,
    results: list[CommandResult],
    detail: str,
    *,
    ok: bool = True,
) -> None:
    if not results:
        report.verification.append(Verification(name, ok, detail=detail))
        return
    report.verification.extend(
        Verification(name, ok, result, detail) for result in results
    )


def _install_target_verified(report: Report) -> bool:
    return bool(
        report.verification
        and _commands_succeeded(report.command_results)
        and all(check.ok for check in report.verification)
        and all(finding.severity.value != "error" for finding in report.findings)
    )


def _install_verification_failure_finding(report: Report) -> Finding:
    failed_checks = [check.name for check in report.verification if not check.ok]
    error_findings = [
        finding.id for finding in report.findings if finding.severity.value == "error"
    ]
    return Finding(
        "install.verification-failed",
        Severity.ERROR,
        "Post-convergence verification failed",
        (
            "The applied host did not pass every non-deferrable convergence "
            "invariant; automatic rollback will use the pre-install snapshot."
        ),
        evidence={
            "failed_verifications": failed_checks,
            "error_findings": error_findings,
        },
        remediation=(
            "Keep the node drained until the compensating rollback is fully "
            "verified or the original private snapshot is applied manually."
        ),
    )


def _requiesce_services_for_compensation(
    report: Report,
    guard: TrustedGpuServiceGuard | None,
) -> bool:
    if guard is None or guard.requiesce():
        return True
    if not any(
        finding.id == "gpu-services.requiesce-failed" for finding in report.findings
    ):
        report.findings.append(
            Finding(
                "gpu-services.requiesce-failed",
                Severity.ERROR,
                "Trusted NVIDIA services could not be safely re-quiesced",
                "; ".join(guard.requiesce_errors),
                remediation=(
                    "Isolate the node and inspect only the originally-active "
                    "trusted services before applying the private rollback snapshot."
                ),
            )
        )
    return False


def _prepare_install_compensation(
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    guard: TrustedGpuServiceGuard | None,
    *,
    allow_active_workloads: bool,
) -> HostAudit | None:
    """Re-establish the drained boundary before applying a snapshot."""

    requiesced = _requiesce_services_for_compensation(report, guard)
    if not requiesced:
        # A failed guard qualification must not bypass the persistent all-four
        # quarantine. Do not attempt rollback after this best-effort boundary.
        assert guard is not None
        _quarantine_failed_maintenance_gate(
            report,
            runner,
            guard,
        )
        return None

    try:
        current_audit = audit_host(runner)
    except BaseException:
        _emergency_quarantine_launchers()
        raise
    lingering_probe = not _retry_failed_probe_container_cleanup(
        report,
        runner,
        docker_service_active=current_audit.docker_service_active,
    )
    probe, active_workloads = _probe_active_gpu_workloads(runner, current_audit)
    report.command_results.append(probe)

    try:
        quarantined, current_audit = _defer_launcher_enablement(
            report,
            runner,
            current_audit,
        )
    except BaseException:
        _emergency_quarantine_launchers()
        raise
    if not quarantined:
        return None
    probe, active_workloads = _probe_active_gpu_workloads(
        runner,
        current_audit,
    )
    report.command_results.append(probe)

    if lingering_probe:
        report.findings.append(
            Finding(
                "install.compensation.probe-container-lingering",
                Severity.ERROR,
                "The verification container could not be proven absent",
                (
                    "Automatic rollback is refused because a uniquely named GPU "
                    "probe container remained after a forced-removal retry."
                ),
                remediation=(
                    "Keep Docker and the node drained, remove the named probe "
                    "container, then apply the original private snapshot."
                ),
            )
        )
        return None
    if active_workloads is None:
        report.findings.append(
            Finding(
                "install.compensation.workloads-unknown",
                Severity.ERROR,
                "GPU workloads are not observable before automatic rollback",
                "The post-quiescence GPU workload inventory remained incomplete.",
                remediation=(
                    "Keep the node drained and restore workload observability "
                    "before applying the private rollback snapshot."
                ),
            )
        )
        return None
    if active_workloads and not allow_active_workloads:
        report.findings.append(
            Finding(
                "install.compensation.workloads-active",
                Severity.ERROR,
                "GPU workloads still block automatic rollback",
                "Observed GPU workloads: " + ", ".join(active_workloads),
                remediation=(
                    "Keep the node drained, stop the named workloads, and apply "
                    "the original private snapshot."
                ),
            )
        )
        return None

    # Reuse this exact post-gate observation when deriving the package/service
    # rollback delta; do not open another race with an unrelated audit.
    del snapshot
    return current_audit


def _retry_failed_probe_container_cleanup(
    report: Report,
    runner: CommandRunner,
    *,
    docker_service_active: bool | None,
) -> bool:
    failed_absence = any(
        check.name == "container.probe-absent" and not check.ok
        for check in report.verification
    )
    if not failed_absence:
        return True
    if docker_service_active is not True:
        return False
    probe = next(
        (
            check
            for check in report.verification
            if check.name == "container.gpu"
            and check.command is not None
            and check.command.command[:2] == ["docker", "run"]
        ),
        None,
    )
    if probe is None or probe.command is None:
        return False
    command = probe.command.command
    try:
        name_index = command.index("--name") + 1
        name = command[name_index]
    except (ValueError, IndexError):
        return False
    cleanup = runner.run(
        ["docker", "rm", "--force", name],
        mutate=True,
        allow_fail=True,
    )
    absence = _probe_named_container_absence(runner, name)
    report.verification.extend(
        [
            Verification(
                "install.compensation.container-cleanup-command",
                cleanup.returncode == 0,
                cleanup,
                "Retry forced removal of the failed verification container.",
            ),
            Verification(
                "install.compensation.container-probe-absent",
                absence.returncode == 0 and not absence.stdout.strip(),
                absence,
                "The failed verification container must be absent before rollback.",
            ),
        ]
    )
    return absence.returncode == 0 and not absence.stdout.strip()


def _probe_named_container_absence(
    runner: CommandRunner,
    name: str,
) -> CommandResult:
    return runner.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"name=^/{name}$",
        ],
        allow_fail=True,
    )


def _install_mutation_failure_finding(
    results: list[CommandResult],
    preparation: Verification | None,
) -> Finding:
    failed = next(
        (result for result in results if result.returncode not in (0, None)),
        None,
    )
    evidence: dict[str, object] = {}
    if failed is not None:
        evidence = {
            "command": failed.command,
            "returncode": failed.returncode,
            "reason": failed.reason,
        }
        detail = (
            "Applied convergence stopped after a mutating command failed; "
            "automatic rollback will use the pre-install snapshot."
        )
    else:
        evidence = {
            "verification": preparation.name if preparation is not None else None,
            "ok": preparation.ok if preparation is not None else False,
        }
        detail = (
            "Applied convergence could not prepare a healthy kernel-module "
            "stack; automatic rollback will use the pre-install snapshot."
        )
    return Finding(
        "install.mutation-failed",
        Severity.ERROR,
        "Applied convergence failed",
        detail,
        evidence=evidence,
    )


def _attempt_install_compensation(
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    *,
    current_audit: HostAudit | None = None,
) -> bool:
    """Apply the exact pre-install snapshot after a failed host mutation."""

    if current_audit is None:
        current_audit = audit_host(runner)
    if not current_audit.package_inventory_complete:
        report.findings.append(
            Finding(
                "install.compensation.inventory-incomplete",
                Severity.ERROR,
                "Automatic rollback requires a complete package inventory",
                (
                    "The post-failure package inventory is incomplete, so an "
                    "exact rollback delta cannot be derived safely."
                ),
                remediation=(
                    "Repair the package database while the node remains drained, "
                    "then apply the original private rollback snapshot."
                ),
            )
        )
        return False
    try:
        preflight_results = preflight_package_rollback(
            snapshot,
            current_audit,
            runner,
        )
        _append_preflight_verification(
            report,
            "install.compensation.packages-preflight",
            preflight_results,
            "The exact compensating package transaction resolves solely from the "
            "snapshot-bound retained local payloads.",
        )
    except PackagePreflightError as exc:
        _append_preflight_verification(
            report,
            "install.compensation.packages-preflight",
            exc.results,
            "The exact compensating package transaction did not resolve safely.",
            ok=False,
        )
        report.findings.append(
            Finding(
                "install.compensation.preflight-failed",
                Severity.ERROR,
                "Automatic rollback package preflight failed",
                str(exc),
                evidence=exc.evidence(),
                remediation=(
                    "Restore the original private snapshot and its paired retained "
                    "payload bundle while the node remains drained, then retry rollback."
                ),
            )
        )
        return False

    rollback_results = apply_rollback(
        snapshot,
        runner,
        current_audit=current_audit,
        restore_service_activity=False,
    )
    report.command_results.extend(rollback_results)
    if _commands_succeeded(rollback_results):
        return True
    failed = next(
        (result for result in rollback_results if result.returncode not in (0, None)),
        None,
    )
    report.findings.append(
        Finding(
            "install.compensation.rollback-failed",
            Severity.ERROR,
            "Automatic rollback failed",
            (
                "The compensating rollback command did not complete safely."
                if failed is None
                else (
                    f"Rollback command exited {failed.returncode}: "
                    + " ".join(failed.command)
                )
            ),
            evidence=(
                {}
                if failed is None
                else {
                    "command": failed.command,
                    "returncode": failed.returncode,
                    "reason": failed.reason,
                }
            ),
            remediation=(
                "Keep the node drained and complete rollback from the original "
                "private snapshot before restarting GPU services."
            ),
        )
    )
    return False


def _launcher_state(
    audit: HostAudit,
    unit: str,
) -> tuple[bool | None, str | None]:
    return {
        "nvidia-fabricmanager.service": (
            audit.fabric_manager_active,
            audit.fabric_manager_unit_file_state,
        ),
        "nvidia-persistenced.service": (
            audit.nvidia_persistenced_active,
            audit.nvidia_persistenced_unit_file_state,
        ),
        "docker.service": (
            audit.docker_service_active,
            audit.docker_service_unit_file_state,
        ),
        "docker.socket": (
            audit.docker_socket_active,
            audit.docker_socket_unit_file_state,
        ),
    }[unit]


def _launcher_target(
    snapshot: RollbackSnapshot,
    unit: str,
) -> tuple[bool | None, str | None]:
    return {
        "nvidia-fabricmanager.service": (
            snapshot.fabric_manager_active,
            snapshot.fabric_manager_unit_file_state,
        ),
        "nvidia-persistenced.service": (
            snapshot.nvidia_persistenced_active,
            snapshot.nvidia_persistenced_unit_file_state,
        ),
        "docker.service": (
            snapshot.docker_service_active,
            snapshot.docker_service_unit_file_state,
        ),
        "docker.socket": (
            snapshot.docker_socket_active,
            snapshot.docker_socket_unit_file_state,
        ),
    }[unit]


_DEPENDENCY_ACTIVATABLE_UNIT_FILE_STATES = frozenset({"enabled", "static"})


def _snapshot_with_launcher_activity(
    snapshot: RollbackSnapshot,
    unit: str,
    *,
    active: bool,
) -> RollbackSnapshot:
    if unit == "nvidia-fabricmanager.service":
        return replace(snapshot, fabric_manager_active=active)
    if unit == "nvidia-persistenced.service":
        return replace(snapshot, nvidia_persistenced_active=active)
    if unit == "docker.service":
        return replace(snapshot, docker_service_active=active)
    if unit == "docker.socket":
        return replace(snapshot, docker_socket_active=active)
    raise ValueError(f"unsupported transactional launcher {unit!r}")


def _snapshot_with_docker_socket_target(
    snapshot: RollbackSnapshot,
    *,
    active: bool,
    unit_file_state: str,
) -> RollbackSnapshot:
    return replace(
        snapshot,
        docker_socket_active=active,
        docker_socket_enabled=unit_file_state == "enabled",
        docker_socket_unit_file_state=unit_file_state,
    )


def _inactive_launcher_requires_activation_proof(
    snapshot: RollbackSnapshot,
    unit: str,
) -> bool:
    target_active, target_state = _launcher_target(snapshot, unit)
    return bool(
        target_active is False
        and target_state in _DEPENDENCY_ACTIVATABLE_UNIT_FILE_STATES
    )


def _install_launcher_target_snapshot(
    snapshot: RollbackSnapshot,
    desired: DesiredState,
) -> RollbackSnapshot:
    fabric_state = snapshot.fabric_manager_unit_file_state
    if desired.fabric_manager:
        fabric_state = "enabled"
    elif fabric_state == "enabled":
        fabric_state = "disabled"
    return replace(
        snapshot,
        fabric_manager_active=desired.fabric_manager,
        fabric_manager_enabled=fabric_state == "enabled",
        fabric_manager_unit_file_state=fabric_state,
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
    )


def _prepare_rollback_unit_safely(
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    *,
    unit: str,
    operation: str,
) -> tuple[bool, HostAudit]:
    target_active, target_state = _launcher_target(snapshot, unit)
    current_active, current_state = _launcher_state(audit, unit)
    if (
        target_active is None
        or target_state is None
        or current_active is None
        or current_state is None
    ):
        report.findings.append(
            Finding(
                f"{operation}.service-state-unknown",
                Severity.ERROR,
                "Launcher preparation state is incomplete",
                f"Current or target state is unknown for {unit}.",
            )
        )
        return False, audit
    results = prepare_rollback_service_activity(
        snapshot,
        runner,
        audit,
        units={unit},
    )
    report.command_results.extend(results)
    if not _commands_succeeded(results):
        report.findings.append(
            Finding(
                f"{operation}.service-prepare-failed",
                Severity.ERROR,
                "Launcher activity preparation failed",
                f"Could not safely unmask and prepare {unit}.",
            )
        )
        return False, audit
    audit = audit_host(runner)
    observed_active, observed_state = _launcher_state(audit, unit)
    prepared_state = "disabled" if target_state == "enabled" else target_state
    if observed_active is not False or observed_state != prepared_state:
        report.findings.append(
            Finding(
                f"{operation}.service-prepare-unverified",
                Severity.ERROR,
                "Launcher activity preparation is unverified",
                f"Fresh audit does not show {unit} inactive/{prepared_state}.",
                evidence={
                    "unit": unit,
                    "active": observed_active,
                    "unit_file_state": observed_state,
                },
            )
        )
        return False, audit
    return True, audit


def _restore_rollback_unit_activity_safely(
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    *,
    unit: str,
    operation: str,
    trusted_identities: dict[str, TrustedGpuServiceIdentity] | None = None,
) -> tuple[bool, HostAudit]:
    target_active, _ = _launcher_target(snapshot, unit)
    current_active, _ = _launcher_state(audit, unit)
    trusted_unit = unit in {
        "nvidia-fabricmanager.service",
        "nvidia-persistenced.service",
        "docker.service",
    }
    if target_active is None or current_active is None:
        report.findings.append(
            Finding(
                f"{operation}.service-state-unknown",
                Severity.ERROR,
                "Launcher activity cannot be restored exactly",
                f"Current or target activity is unknown for {unit}.",
            )
        )
        return False, audit
    if trusted_unit and target_active and not current_active:
        validation_results, error = validate_trusted_gpu_service_start(
            runner,
            unit,
        )
        report.command_results.extend(validation_results)
        if error is not None:
            report.findings.append(
                Finding(
                    f"{operation}.trusted-service-start-refused",
                    Severity.ERROR,
                    "Trusted service start validation failed",
                    error,
                    evidence={"unit": unit},
                )
            )
            return False, audit

    if trusted_unit and current_active and not target_active:
        validation_results, _, error = (
            validate_active_trusted_gpu_service_identity(runner, unit)
        )
        report.command_results.extend(validation_results)
        if error is not None:
            report.findings.append(
                Finding(
                    f"{operation}.trusted-service-stop-refused",
                    Severity.ERROR,
                    "Trusted service stop validation failed",
                    error,
                    evidence={"unit": unit},
                )
            )
            return False, audit

    # No observation or mutation may intervene between the trusted pre-start
    # validation and this activity call, which contains the systemctl start.
    results = restore_rollback_service_activity(
        snapshot,
        runner,
        audit,
        units={unit},
    )
    report.command_results.extend(results)
    if not _commands_succeeded(results):
        report.findings.append(
            Finding(
                f"{operation}.service-restore-failed",
                Severity.ERROR,
                "Launcher activity restoration failed",
                f"Could not restore exact activity for {unit}.",
            )
        )
        return False, audit

    if trusted_unit and target_active:
        validation_results, identity, error = (
            validate_active_trusted_gpu_service_identity(runner, unit)
        )
        report.command_results.extend(validation_results)
        if error is not None or identity is None:
            report.findings.append(
                Finding(
                    f"{operation}.trusted-service-active-unverified",
                    Severity.ERROR,
                    "Trusted service binding is unverified after start",
                    error or "trusted service identity was not returned",
                    evidence={"unit": unit},
                )
            )
            return False, audit
        audit = audit_host(runner)
        revalidation_results, _, error = validate_active_trusted_gpu_service_identity(
            runner,
            unit,
            expected_identity=identity,
        )
        report.command_results.extend(revalidation_results)
        if error is not None:
            report.findings.append(
                Finding(
                    f"{operation}.trusted-service-identity-changed",
                    Severity.ERROR,
                    "Trusted service identity changed during validation",
                    error,
                    evidence={"unit": unit},
                )
            )
            return False, audit
        if trusted_identities is not None:
            trusted_identities[unit] = identity
    else:
        audit = audit_host(runner)

    observed_active, _ = _launcher_state(audit, unit)
    if observed_active is not target_active:
        report.findings.append(
            Finding(
                f"{operation}.service-restore-unverified",
                Severity.ERROR,
                "Launcher activity restoration is unverified",
                f"Fresh audit does not match {unit} target activity.",
            )
        )
        return False, audit
    if (
        target_active
        and unit == "nvidia-fabricmanager.service"
        and not (
            audit.fabric_manager_healthy is True
            and audit.fabric_manager_version is not None
            and audit.fabric_manager_version == audit.module.version
        )
    ):
        report.findings.append(
            Finding(
                f"{operation}.fabric-manager-health-failed",
                Severity.ERROR,
                "Fabric Manager is not healthy after restore",
                (
                    "Fabric health and exact loaded-driver version binding must "
                    "pass before another launcher is restored."
                ),
            )
        )
        return False, audit
    return True, audit


def _finalize_rollback_unit_state(
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    *,
    unit: str,
    operation: str,
    require_activity: bool = True,
) -> tuple[bool, HostAudit]:
    results = restore_rollback_service_enablement(
        snapshot,
        runner,
        audit,
        units={unit},
    )
    report.command_results.extend(results)
    if not _commands_succeeded(results):
        report.findings.append(
            Finding(
                f"{operation}.service-enable-restore-failed",
                Severity.ERROR,
                "Launcher persistent state restoration failed",
                f"Could not finalize the exact unit-file state for {unit}.",
            )
        )
        return False, audit
    audit = audit_host(runner)
    observed_active, observed_state = _launcher_state(audit, unit)
    target_active, target_state = _launcher_target(snapshot, unit)
    if (
        require_activity and observed_active is not target_active
    ) or observed_state != target_state:
        report.findings.append(
            Finding(
                f"{operation}.service-final-state-unverified",
                Severity.ERROR,
                "Launcher final state is unverified",
                f"Fresh audit does not match the exact {unit} target.",
                evidence={
                    "expected_active": target_active,
                    "observed_active": observed_active,
                    "expected_unit_file_state": target_state,
                    "observed_unit_file_state": observed_state,
                },
            )
        )
        return False, audit
    return True, audit


def _restore_rollback_unit_staged(
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    *,
    unit: str,
    operation: str,
    trusted_identities: dict[str, TrustedGpuServiceIdentity] | None = None,
) -> tuple[bool, HostAudit]:
    ok, audit = _prepare_rollback_unit_safely(
        report,
        snapshot,
        runner,
        audit,
        unit=unit,
        operation=operation,
    )
    if not ok:
        return False, audit
    if _inactive_launcher_requires_activation_proof(snapshot, unit):
        active_snapshot = _snapshot_with_launcher_activity(
            snapshot,
            unit,
            active=True,
        )
        ok, audit = _restore_rollback_unit_activity_safely(
            report,
            active_snapshot,
            runner,
            audit,
            unit=unit,
            operation=operation,
        )
        if not ok:
            return False, audit
        ok, audit = _restore_rollback_unit_activity_safely(
            report,
            snapshot,
            runner,
            audit,
            unit=unit,
            operation=operation,
        )
        if not ok:
            return False, audit
    else:
        ok, audit = _restore_rollback_unit_activity_safely(
            report,
            snapshot,
            runner,
            audit,
            unit=unit,
            operation=operation,
            trusted_identities=trusted_identities,
        )
        if not ok:
            return False, audit
    return _finalize_rollback_unit_state(
        report,
        snapshot,
        runner,
        audit,
        unit=unit,
        operation=operation,
    )


def _quarantine_failed_launcher_commit(
    args: argparse.Namespace,
    report: Report,
    runner: CommandRunner,
    guard: TrustedGpuServiceGuard | None,
) -> tuple[bool, HostAudit]:
    """Fail closed after any partial launcher release."""

    del args
    _requiesce_services_for_compensation(report, guard)
    audit = audit_host(runner)
    quarantined, audit = _defer_launcher_enablement(
        report,
        runner,
        audit,
    )
    probe, workloads = _probe_active_gpu_workloads(runner, audit)
    report.command_results.append(probe)
    if not quarantined or workloads is None or workloads:
        report.findings.append(
            Finding(
                "launcher-commit.quarantine-unverified",
                Severity.ERROR,
                "Failed launcher commit is not fully quarantined",
                (
                    "All four launchers were re-masked, but the fresh service/"
                    "workload proof did not establish a completely idle boundary."
                    if workloads is None or workloads
                    else "One or more launchers could not be re-masked."
                ),
                evidence={"active_gpu_workloads": workloads},
                remediation=(
                    "Keep the node drained and rerun the exact private snapshot "
                    "rollback before releasing any launcher."
                ),
            )
        )
    return False, audit


_TRANSACTIONAL_LAUNCHERS = (
    "nvidia-fabricmanager.service",
    "nvidia-persistenced.service",
    "docker.service",
    "docker.socket",
)
_TRUSTED_TRANSACTIONAL_LAUNCHERS = (
    "nvidia-fabricmanager.service",
    "nvidia-persistenced.service",
    "docker.service",
)


def _verify_joint_launcher_commit(
    args: argparse.Namespace,
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    *,
    operation: str,
    trusted_identities: dict[str, TrustedGpuServiceIdentity] | None = None,
) -> tuple[bool, HostAudit]:
    """Prove the all-launcher target after the final release mutation."""

    identities: dict[str, TrustedGpuServiceIdentity] = {}
    for unit in _TRUSTED_TRANSACTIONAL_LAUNCHERS:
        target_active, _ = _launcher_target(snapshot, unit)
        if target_active is not True:
            continue
        expected_identity = (
            trusted_identities.get(unit) if trusted_identities is not None else None
        )
        if trusted_identities is not None and expected_identity is None:
            report.findings.append(
                Finding(
                    f"{operation}.launcher-final-identity-unverified",
                    Severity.ERROR,
                    "Final trusted NVIDIA service identity is unverified",
                    "The staged launcher commit did not retain its identity token.",
                    evidence={"unit": unit},
                )
            )
            return False, audit
        validation_results, identity, error = (
            validate_active_trusted_gpu_service_identity(
                runner,
                unit,
                expected_identity=expected_identity,
            )
        )
        report.command_results.extend(validation_results)
        if error is not None or identity is None:
            report.findings.append(
                Finding(
                    f"{operation}.launcher-final-identity-unverified",
                    Severity.ERROR,
                    "Final trusted NVIDIA service identity is unverified",
                    error or "trusted service identity was not returned",
                    evidence={"unit": unit},
                )
            )
            return False, audit
        identities[unit] = identity

    target_docker_active, _ = _launcher_target(snapshot, "docker.service")
    target_socket_active, target_socket_state = _launcher_target(
        snapshot,
        "docker.socket",
    )
    # A masked inactive socket is the one supported exception: the immediate
    # post-start gate already ran while its temporary dependency was live, and
    # the only later socket mutation is stop+mask (strictly reducing release).
    final_workload_checkpoint_required = bool(
        target_docker_active is True
        and not (target_socket_active is False and target_socket_state == "masked")
    )
    if final_workload_checkpoint_required:
        workload_ok, audit = _post_docker_workload_checkpoint(
            args,
            report,
            runner,
            operation=operation,
        )
        if not workload_ok:
            return False, audit

    # This audit is deliberately after socket enablement and the Docker
    # workload probe. The trusted identity observations bracket both checks.
    audit = audit_host(runner)
    for unit, identity in identities.items():
        revalidation_results, _, error = validate_active_trusted_gpu_service_identity(
            runner,
            unit,
            expected_identity=identity,
        )
        report.command_results.extend(revalidation_results)
        if error is not None:
            report.findings.append(
                Finding(
                    f"{operation}.launcher-final-identity-changed",
                    Severity.ERROR,
                    "Trusted NVIDIA service identity changed during final commit",
                    error,
                    evidence={"unit": unit},
                )
            )
            return False, audit

    mismatches: dict[str, dict[str, object]] = {}
    for unit in _TRANSACTIONAL_LAUNCHERS:
        expected_active, expected_state = _launcher_target(snapshot, unit)
        observed_active, observed_state = _launcher_state(audit, unit)
        if (
            expected_active is None
            or expected_state is None
            or observed_active is not expected_active
            or observed_state != expected_state
        ):
            mismatches[unit] = {
                "expected_active": expected_active,
                "observed_active": observed_active,
                "expected_unit_file_state": expected_state,
                "observed_unit_file_state": observed_state,
            }
    if snapshot.fabric_manager_active is True and not (
        audit.fabric_manager_healthy is True
        and audit.fabric_manager_version is not None
        and audit.fabric_manager_version == audit.module.version
    ):
        mismatches.setdefault("nvidia-fabricmanager.service", {}).update(
            {
                "fabric_manager_healthy": audit.fabric_manager_healthy,
                "fabric_manager_version": audit.fabric_manager_version,
                "loaded_module_version": audit.module.version,
            }
        )
    if mismatches:
        report.findings.append(
            Finding(
                f"{operation}.launcher-final-state-unverified",
                Severity.ERROR,
                "Joint launcher commit is unverified",
                (
                    "The final fresh audit does not match the exact all-four "
                    "launcher target after the last release mutation."
                ),
                evidence={"mismatches": mismatches},
            )
        )
        return False, audit
    return True, audit


def _commit_rollback_service_activity_impl(
    args: argparse.Namespace,
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    guard: TrustedGpuServiceGuard | None,
    *,
    operation: str,
) -> tuple[bool, HostAudit]:
    trusted_identities: dict[str, TrustedGpuServiceIdentity] = {}
    for unit in (
        "nvidia-fabricmanager.service",
        "nvidia-persistenced.service",
    ):
        ok, audit = _restore_rollback_unit_staged(
            report,
            snapshot,
            runner,
            audit,
            unit=unit,
            operation=operation,
            trusted_identities=trusted_identities,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
        if guard is not None and unit.startswith("nvidia-"):
            guard.relinquish({unit})

    target_service_active, target_service_state = _launcher_target(
        snapshot,
        "docker.service",
    )
    target_socket_active, target_socket_state = _launcher_target(
        snapshot,
        "docker.socket",
    )
    if (
        target_service_active is None
        or target_service_state is None
        or target_socket_active is None
        or target_socket_state is None
    ):
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )

    # docker.service commonly Requires docker.socket for fd://. A valid
    # baseline can have a running daemon while the socket is persistently
    # masked. Temporarily prepare that dependency while docker.service remains
    # transaction-masked, then restore the exact masked socket as the last
    # release mutation.
    temporary_masked_socket_dependency = bool(
        target_service_active is True
        and target_socket_active is False
        and target_socket_state == "masked"
    )
    socket_dependency_snapshot = (
        _snapshot_with_docker_socket_target(
            snapshot,
            active=True,
            unit_file_state="disabled",
        )
        if temporary_masked_socket_dependency
        else snapshot
    )
    ok, audit = _prepare_rollback_unit_safely(
        report,
        socket_dependency_snapshot,
        runner,
        audit,
        unit="docker.socket",
        operation=operation,
    )
    if not ok:
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )
    ok, audit = _prepare_rollback_unit_safely(
        report,
        snapshot,
        runner,
        audit,
        unit="docker.service",
        operation=operation,
    )
    if not ok:
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )
    prove_socket_activation = bool(
        temporary_masked_socket_dependency
        or target_socket_active
        or _inactive_launcher_requires_activation_proof(
            snapshot,
            "docker.socket",
        )
    )
    if prove_socket_activation:
        active_socket_snapshot = (
            socket_dependency_snapshot
            if temporary_masked_socket_dependency
            else _snapshot_with_launcher_activity(
                snapshot,
                "docker.socket",
                active=True,
            )
        )
        ok, audit = _restore_rollback_unit_activity_safely(
            report,
            active_socket_snapshot,
            runner,
            audit,
            unit="docker.socket",
            operation=operation,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )

    prove_docker_activation = bool(
        target_service_active
        or _inactive_launcher_requires_activation_proof(
            snapshot,
            "docker.service",
        )
        or prove_socket_activation
    )
    service_activity_snapshot = (
        _snapshot_with_launcher_activity(
            snapshot,
            "docker.service",
            active=True,
        )
        if prove_docker_activation
        else snapshot
    )
    ok, audit = _restore_rollback_unit_activity_safely(
        report,
        service_activity_snapshot,
        runner,
        audit,
        unit="docker.service",
        operation=operation,
        trusted_identities=trusted_identities,
    )
    if not ok:
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )
    if prove_docker_activation:
        ok, audit = _post_docker_workload_checkpoint(
            args,
            report,
            runner,
            operation=operation,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
    if target_service_active is False and prove_docker_activation:
        ok, audit = _restore_rollback_unit_activity_safely(
            report,
            snapshot,
            runner,
            audit,
            unit="docker.service",
            operation=operation,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )

    current_socket_active, _ = _launcher_state(audit, "docker.socket")
    if current_socket_active is None:
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )
    if current_socket_active is not target_socket_active:
        ok, audit = _restore_rollback_unit_activity_safely(
            report,
            snapshot,
            runner,
            audit,
            unit="docker.socket",
            operation=operation,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
    observed_socket_active, observed_socket_state = _launcher_state(
        audit,
        "docker.socket",
    )
    observed_service_active, observed_service_state = _launcher_state(
        audit,
        "docker.service",
    )
    prepared_service_state = (
        "disabled" if target_service_state == "enabled" else target_service_state
    )
    prepared_socket_state = (
        "disabled"
        if temporary_masked_socket_dependency or target_socket_state == "enabled"
        else target_socket_state
    )
    if (
        observed_socket_active is not target_socket_active
        or observed_socket_state != prepared_socket_state
        or observed_service_active is not target_service_active
        or observed_service_state != prepared_service_state
    ):
        report.findings.append(
            Finding(
                f"{operation}.docker-activity-commit-unverified",
                Severity.ERROR,
                "Docker activity commit is unverified",
                (
                    "Fresh audit does not match the exact Docker activity target "
                    "and safely prepared unit-file states."
                ),
                evidence={
                    "expected_service_active": target_service_active,
                    "observed_service_active": observed_service_active,
                    "expected_service_prepared_state": prepared_service_state,
                    "observed_service_state": observed_service_state,
                    "expected_socket_active": target_socket_active,
                    "observed_socket_active": observed_socket_active,
                    "expected_socket_prepared_state": prepared_socket_state,
                    "observed_socket_state": observed_socket_state,
                },
            )
        )
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )

    # Persistent activation is the final release step.  In particular, never
    # enable docker.socket until both Docker activity targets and the immediate
    # GPU-workload gate have been proven on a fresh audit.
    for unit in ("docker.service",):
        ok, audit = _finalize_rollback_unit_state(
            report,
            snapshot,
            runner,
            audit,
            unit=unit,
            operation=operation,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
    if temporary_masked_socket_dependency:
        socket_quarantine = _quarantine_service_for_rollback(
            runner,
            "docker.socket",
        )
        report.command_results.extend(socket_quarantine)
        if not _commands_succeeded(socket_quarantine):
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
        audit = audit_host(runner)
        observed_socket_active, observed_socket_state = _launcher_state(
            audit,
            "docker.socket",
        )
        if (
            observed_socket_active is not target_socket_active
            or observed_socket_state != target_socket_state
        ):
            report.findings.append(
                Finding(
                    f"{operation}.docker-socket-final-state-unverified",
                    Severity.ERROR,
                    "Docker socket final state is unverified",
                    "Fresh audit does not prove the exact masked socket target.",
                )
            )
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
    else:
        ok, audit = _finalize_rollback_unit_state(
            report,
            snapshot,
            runner,
            audit,
            unit="docker.socket",
            operation=operation,
        )
        if not ok:
            return _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
    jointly_verified, audit = _verify_joint_launcher_commit(
        args,
        report,
        snapshot,
        runner,
        audit,
        operation=operation,
        trusted_identities=trusted_identities,
    )
    if not jointly_verified:
        return _quarantine_failed_launcher_commit(
            args,
            report,
            runner,
            guard,
        )
    return True, audit


def _commit_rollback_service_activity(
    args: argparse.Namespace,
    report: Report,
    snapshot: RollbackSnapshot,
    runner: CommandRunner,
    audit: HostAudit,
    guard: TrustedGpuServiceGuard | None,
    *,
    operation: str,
) -> tuple[bool, HostAudit]:
    try:
        return _commit_rollback_service_activity_impl(
            args,
            report,
            snapshot,
            runner,
            audit,
            guard,
            operation=operation,
        )
    except BaseException:
        try:
            _quarantine_failed_launcher_commit(
                args,
                report,
                runner,
                guard,
            )
        except BaseException:  # noqa: BLE001 - emergency fail-closed cleanup
            _emergency_quarantine_launchers()
        raise


def _status_from_results(results: list[CommandResult]) -> int:
    failed = [result for result in results if result.returncode not in (0, None)]
    return 2 if failed else 0


def _plan_status(report: Report) -> int:
    return (
        2
        if not report.plan
        or any(action.id.startswith("unsupported.") for action in report.plan)
        else 0
    )


def _commands_succeeded(results: list[CommandResult]) -> bool:
    return _status_from_results(results) == 0


def _verification_can_complete_after_reboot(report: Report) -> bool:
    return bool(
        report.mode == "apply"
        and _target_verification_can_complete_after_reboot(report)
    )


def _target_verification_can_complete_after_reboot(report: Report) -> bool:
    audit = report.audit
    if audit is None or not audit.package_inventory_complete:
        return False

    on_disk_ready = bool(
        report.desired.matches_driver_version(audit.module.installed_version)
        and audit.module.installed_open_module is report.desired.open_kernel_module
        and (
            report.desired.secure_boot != "signed"
            or audit.module.installed_signed is True
        )
    )
    if not on_disk_ready:
        return False

    mig_pending = bool(
        audit.mig_mode in {"enabled", "disabled"}
        and audit.mig_mode_pending == report.desired.mig
        and audit.mig_mode_pending != audit.mig_mode
        and (report.desired.mig != "enabled" or audit.mig_capable is True)
    )
    reload_failed = any(
        check.name == "module.reload" and not check.ok for check in report.verification
    )
    module = audit.module
    loaded_ready = bool(
        module.loaded
        and report.desired.matches_driver_version(module.version)
        and module.open_module is report.desired.open_kernel_module
        and (report.desired.secure_boot != "signed" or module.signed is True)
    )
    module_pending = bool(
        reload_failed
        and module.loaded
        and not loaded_ready
        and (
            mig_pending
            or (
                audit.mig_mode == report.desired.mig
                and audit.mig_mode_pending == audit.mig_mode
                and (report.desired.mig != "enabled" or audit.mig_capable is True)
            )
        )
    )
    if not mig_pending and not module_pending:
        return False

    allowed_failed_checks: set[str] = set()
    allowed_error_findings: set[str] = set()
    if mig_pending:
        allowed_failed_checks.update(
            {
                "mig.mode",
                "mig.no-pending-transition",
                "mig.geometry",
                "mig.device-uuid",
                "container.device-binding",
            }
        )
        allowed_error_findings.update({"mig.pending-reboot", f"mig.{audit.mig_mode}"})
    if module_pending:
        allowed_failed_checks.update(
            {
                "module.reload",
                "module.loaded-version",
                "module.provenance",
                "module.open-variant",
                "module.closed-variant",
                "module.flavor-provenance",
                "secure-boot.module-signed",
                "nvidia-smi",
                "container.cuda-driver-compatibility",
            }
        )
        allowed_error_findings.update(
            {
                "driver.loaded-version-unknown",
                "driver.module-version-mismatch",
                "driver.version-mismatch",
                "driver.closed-module",
                "driver.open-module",
                "driver.module-flavor-unknown",
                "driver.module-flavor-mismatch",
                "secure-boot.unsigned-module",
            }
        )

    for check in report.verification:
        if check.ok or check.name in allowed_failed_checks:
            continue
        if (
            check.name == "container.gpu"
            and check.command is None
            and (mig_pending or module_pending)
        ):
            continue
        return False
    if any(
        finding.severity.value == "error" and finding.id not in allowed_error_findings
        for finding in report.findings
    ):
        return False
    command_results = [
        *report.command_results,
        *[check.command for check in report.verification if check.command is not None],
    ]
    return all(
        result.returncode == 0
        or (module_pending and _reboot_resolvable_module_command_failure(result))
        for result in command_results
    )


def _rollback_can_complete_after_reboot(report: Report) -> bool:
    audit = report.audit
    snapshot = report.rollback
    if report.mode != "apply" or audit is None or snapshot is None:
        return False
    if any(result.returncode != 0 for result in report.command_results):
        return False
    if any(finding.severity.value == "error" for finding in report.findings):
        return False
    return _rollback_state_can_complete_after_reboot(
        snapshot,
        audit,
        report.verification,
    )


def _rollback_state_can_complete_after_reboot(
    snapshot: RollbackSnapshot,
    audit: HostAudit,
    checks: list[Verification],
) -> bool:
    if not (
        audit.package_inventory_complete
        and snapshot.mig_mode is not None
        and audit.mig_mode != snapshot.mig_mode
        and audit.mig_mode_pending == snapshot.mig_mode
    ):
        return False
    failed_checks = {check.name for check in checks if not check.ok}
    return bool(
        failed_checks == {"rollback.mig-mode"}
        and all(
            check.command is None or check.command.returncode == 0 for check in checks
        )
    )


def _rollback_reboot_recovery_finding(
    snapshot: RollbackSnapshot,
    snapshot_path: str,
    *,
    operation: str,
) -> Finding:
    command = [
        "nvidia-converge",
        "rollback",
        "--apply",
        "--allow-disruption",
        "--snapshot",
        snapshot_path,
    ]
    return Finding(
        f"{operation}.reboot-required",
        Severity.WARNING,
        "Rollback is staged and requires a reboot",
        (
            "All GPU launchers remain persistently masked. Reboot, then rerun "
            "the exact rollback command with the same private snapshot before "
            "re-enabling workloads."
        ),
        evidence={
            "snapshot_path": snapshot_path,
            "snapshot_integrity_sha256": snapshot.integrity_sha256,
            "recovery_command": command,
        },
        remediation=" ".join(command),
    )


def _reboot_resolvable_module_command_failure(result: CommandResult) -> bool:
    if (
        result.returncode is None
        or result.returncode == 0
        or result.skipped
        or result.reason is not None
        or not result.command
        or result.command[0] != "modprobe"
        or result.stdout != ""
    ):
        return False
    modules = {
        "nvidia",
        "nvidia_drm",
        "nvidia_fs",
        "nvidia_modeset",
        "nvidia_peermem",
        "nvidia_uvm",
    }
    if not (
        len(result.command) > 2
        and result.command[1] == "-r"
        and all(module in modules for module in result.command[2:])
    ):
        # A reboot cannot repair a missing, unsigned, invalid, or otherwise
        # unloadable on-disk module. Only a recognized busy-unload failure has
        # positive evidence that the current kernel users are the blocker.
        return False
    stderr = result.stderr.strip()
    busy_patterns = (
        r"modprobe: FATAL: Module (?P<module>[A-Za-z0-9_]+) is in use\.?(?: by: [A-Za-z0-9_, -]+)?",
        r"modprobe: ERROR: could not remove ['\"]?(?P<module>[A-Za-z0-9_]+)['\"]?: Device or resource busy",
    )
    for pattern in busy_patterns:
        match = re.fullmatch(pattern, stderr)
        if match is not None and match.group("module") in result.command[2:]:
            return True
    return False


def _probe_active_gpu_workloads(
    runner: CommandRunner,
    audit: HostAudit | None = None,
) -> tuple[CommandResult, list[str] | None]:
    return probe_active_gpu_workloads(runner, audit)


def _maintenance_gate(
    args: argparse.Namespace,
    runner: CommandRunner,
    audit: HostAudit,
    *,
    operation: str,
    maintenance_detail: str | None = None,
    quiesce_services: bool = True,
) -> _MaintenanceGateOutcome:
    if not args.allow_disruption:
        probe, _ = _probe_active_gpu_workloads(runner, audit)
        return _MaintenanceGateOutcome(
            None,
            [],
            probe,
            [],
            [
                Finding(
                    "maintenance-window.required",
                    Severity.ERROR,
                    f"{operation.capitalize()} requires an explicit maintenance window",
                    maintenance_detail
                    or (
                        f"{operation.capitalize()} may load modules, exercise a "
                        "GPU, or change package policy."
                    ),
                    remediation="Drain the node, then rerun with --allow-disruption.",
                )
            ],
        )
    if not quiesce_services:
        probe, active_workloads = _probe_active_gpu_workloads(runner, audit)
        finding = None
        if active_workloads is None:
            finding = Finding(
                "gpu-workloads.unknown",
                Severity.ERROR,
                "Active GPU workloads could not be determined",
                (
                    "The compute-process, GPU-device-user, or Docker GPU-allocation "
                    "inventory was not fully observable."
                ),
                remediation=(
                    f"Repair nvidia-smi workload observability before {operation}."
                ),
            )
        elif active_workloads and not args.allow_active_workloads:
            finding = Finding(
                "gpu-workloads.active",
                Severity.ERROR,
                f"Active GPU workloads block {operation}",
                "Observed GPU workloads: " + ", ".join(active_workloads),
                remediation=(
                    "Drain the GPU workloads or explicitly use "
                    "--allow-active-workloads."
                ),
            )
        return _MaintenanceGateOutcome(
            None,
            [],
            probe,
            [],
            [finding] if finding is not None else [],
        )
    try:
        guard = quiesce_trusted_gpu_services(
            runner,
            restore_on_failure=False,
        )
    except BaseException:
        _emergency_quarantine_launchers()
        raise
    if not guard.ok:
        findings = [
            Finding(
                "gpu-services.quiesce-failed",
                Severity.ERROR,
                "Trusted NVIDIA services could not be safely quiesced",
                guard.error or "Trusted NVIDIA service state was not observable.",
                remediation=(
                    "Repair the exact systemd unit, executable, and cgroup binding "
                    "for Fabric Manager or NVIDIA Persistence Daemon before retrying."
                ),
            )
        ]
        return _MaintenanceGateOutcome(
            guard,
            list(guard.results),
            None,
            [],
            findings,
        )
    before_probe = list(guard.results)
    try:
        probe, active_workloads = _probe_active_gpu_workloads(runner, audit)
    except BaseException:
        _emergency_quarantine_launchers()
        raise
    finding = None
    if active_workloads is None:
        finding = Finding(
            "gpu-workloads.unknown",
            Severity.ERROR,
            "Active GPU workloads could not be determined",
            "The compute-process, GPU-device-user, or Docker GPU-allocation inventory was not fully observable.",
            remediation=(
                f"Repair nvidia-smi workload observability before {operation}."
            ),
        )
    elif active_workloads and not args.allow_active_workloads:
        finding = Finding(
            "gpu-workloads.active",
            Severity.ERROR,
            f"Active GPU workloads block {operation}",
            "Observed GPU workloads: " + ", ".join(active_workloads),
            remediation=(
                "Drain the GPU workloads or explicitly use --allow-active-workloads."
            ),
        )
    findings = [finding] if finding is not None else []
    return _MaintenanceGateOutcome(
        guard,
        before_probe,
        probe,
        [],
        findings,
    )


def _record_intentionally_quiesced_services(
    report: Report,
    guard: TrustedGpuServiceGuard | None,
) -> None:
    if guard is None or not guard.quiesced_service_names:
        return
    if not guard.requiesce():
        report.findings.append(
            Finding(
                "gpu-services.requiesce-failed",
                Severity.ERROR,
                "Trusted NVIDIA services could not be proven quiesced",
                "; ".join(guard.requiesce_errors),
                remediation=(
                    "Isolate the node and inspect only the named trusted systemd "
                    "services before attempting any restart."
                ),
            )
        )
        return
    names = guard.quiesced_service_names
    report.findings.append(
        Finding(
            "gpu-services.intentionally-quiesced",
            Severity.ERROR,
            "Trusted NVIDIA services were not restarted after mutation failure",
            (
                "The stack may be inconsistent, so nvidia-converge intentionally "
                "did not restart: " + ", ".join(names)
            ),
            remediation=(
                "Inspect and repair the failed package/module transaction before "
                "securely validating and restarting the named services."
            ),
        )
    )


def _record_failed_rollback_service_changes(
    report: Report,
    before: HostAudit,
    after: HostAudit,
) -> None:
    fields = (
        "docker_service_active",
        "docker_service_enabled",
        "docker_service_unit_file_state",
        "docker_socket_active",
        "docker_socket_enabled",
        "docker_socket_unit_file_state",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
        "nvidia_persistenced_unit_file_state",
        "fabric_manager_active",
        "fabric_manager_enabled",
        "fabric_manager_unit_file_state",
    )
    changed = {
        field: {
            "before": getattr(before, field),
            "after": getattr(after, field),
        }
        for field in fields
        if getattr(before, field) != getattr(after, field)
    }
    if not changed:
        return
    report.findings.append(
        Finding(
            "rollback.service-state-altered",
            Severity.ERROR,
            "Rollback failed after changing service state",
            (
                "The post-failure audit, rather than the pre-rollback baseline, "
                "records Docker/Fabric Manager state. Changed fields: "
                + ", ".join(sorted(changed))
            ),
            evidence=changed,
            remediation=(
                "Keep the node drained and use the post-failure audit plus the "
                "private snapshot to finish restoring exact service state."
            ),
        )
    )


def _install_status(report: Report) -> int:
    commands_ok = _commands_succeeded(report.command_results)
    verification_ok = all(check.ok for check in report.verification)
    findings_ok = all(finding.severity.value != "error" for finding in report.findings)
    return 0 if commands_ok and verification_ok and findings_ok else 2


def _verification_status(report: Report) -> int:
    verification_ok = bool(report.verification) and all(
        check.ok for check in report.verification
    )
    findings_ok = all(finding.severity.value != "error" for finding in report.findings)
    return 0 if verification_ok and findings_ok else 2


if __name__ == "__main__":
    sys.exit(main())
