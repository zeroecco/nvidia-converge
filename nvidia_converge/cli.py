from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from . import __version__
from .audit import audit_host
from .desired import DesiredConfigError, load_desired
from .doctor import diagnose
from .human import render_human
from .models import Report, utc_now
from .planner import build_plan, lock_actions
from .report import report_json, sbom_from_audit, write_report
from .rollback import RollbackSnapshotError, apply_rollback, create_snapshot, load_snapshot
from .runner import CommandRunner
from .schemas import schema_json
from .support import support_human, support_json
from .verify import verify_stack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nvidia-converge", description="Converge a node to a desired NVIDIA driver stack.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_common_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "plan", "install", "verify", "lock", "snapshot"):
        _add_common_args(sub.add_parser(name))
    validate = sub.add_parser("validate")
    validate.add_argument("--desired", default=argparse.SUPPRESS, help="Desired-state JSON/YAML file.")
    validate.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Print machine-readable validation details.")
    schema = sub.add_parser("schema")
    schema.add_argument("name", choices=("desired", "integration-results", "report"), help="Schema to print.")
    support = sub.add_parser("support")
    support.add_argument("--json", action="store_true", help="Print support matrix as JSON.")
    rollback = sub.add_parser("rollback")
    _add_common_args(rollback)
    rollback.add_argument("--snapshot", required=True, help="Rollback snapshot JSON created by install or snapshot.")
    args = parser.parse_args(argv)

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
    try:
        desired = load_desired(desired_path)
    except DesiredConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "validate":
        emit_validation(desired, getattr(args, "json", False))
        return 0
    if apply_changes and _requires_root(args.command) and hasattr(os, "geteuid") and os.geteuid() != 0:
        print(f"error: {args.command} --apply must be run as root", file=sys.stderr)
        return 2
    runner = CommandRunner(apply=apply_changes)

    if args.command == "rollback":
        try:
            snapshot = load_snapshot(args.snapshot)
        except RollbackSnapshotError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        results = apply_rollback(snapshot, runner)
        report = Report("1.0", utc_now(), desired, command_results=results, rollback=snapshot)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _status_from_results(results)

    audit = audit_host(runner)
    findings = diagnose(desired, audit)
    report = Report("1.0", utc_now(), desired, audit=audit, findings=findings, sbom=sbom_from_audit(audit))

    if args.command == "doctor":
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0 if all(f.severity.value != "error" for f in findings) else 2

    if args.command == "plan":
        report.plan = build_plan(desired, audit, findings)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0

    if args.command == "snapshot":
        report.rollback = create_snapshot(audit)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0

    if args.command == "install":
        report.rollback = create_snapshot(audit, persist=apply_changes)
        report.plan = build_plan(desired, audit, findings)
        runner.results = []
        for action in report.plan:
            if action.id in {"snapshot.current-state", "verify.stack"}:
                continue
            for command in action.commands:
                runner.run(command, mutate=True, allow_fail=True)
        report.command_results = list(runner.results)
        post_audit = audit_host(runner)
        report.audit = post_audit
        report.findings = diagnose(desired, post_audit)
        report.verification = verify_stack(desired, runner, post_audit)
        report.sbom = sbom_from_audit(post_audit)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0 if all(v.ok for v in report.verification) and all(f.severity.value != "error" for f in report.findings) else 2

    if args.command == "verify":
        runner.results = []
        report.verification = verify_stack(desired, runner, audit)
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return 0 if all(v.ok for v in report.verification) else 2

    if args.command == "lock":
        report.plan = lock_actions(desired, audit)
        runner.results = []
        for action in report.plan:
            for command in action.commands:
                runner.run(command, mutate=True, allow_fail=True)
        report.command_results = runner.results
        emit_report(args.command, report, out_path, json_stdout, apply_changes)
        return _status_from_results(runner.results)

    return 1


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--desired", default=argparse.SUPPRESS, help="Desired-state JSON/YAML file.")
    parser.add_argument("--out", default=argparse.SUPPRESS, help="Write machine-readable JSON report to this path.")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Print the full machine-readable JSON report to stdout instead of the human summary.")
    parser.add_argument("--apply", action="store_true", default=argparse.SUPPRESS, help="Apply host-mutating actions. Without this, mutating commands are dry-run.")


def emit_report(command: str, report: Report, out_path: str | None, json_stdout: bool, apply_changes: bool) -> None:
    if out_path:
        write_report(report, out_path)
    if json_stdout:
        print(report_json(report))
    else:
        print(render_human(command, report, apply=apply_changes))


def emit_validation(desired, json_stdout: bool) -> None:
    payload = {"schema_version": "1.0", "valid": True, "desired": asdict(desired)}
    if json_stdout:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("nvidia-converge validate")
    print("Desired state: valid")
    print(f"Driver: {desired.driver}")
    print(f"CUDA compat: {desired.cuda_compat}")
    print(f"Container runtime: {desired.container_runtime}")
    print("Use --json to print machine-readable validation details.")


def _requires_root(command: str) -> bool:
    return command in {"install", "lock", "rollback", "verify"}


def _status_from_results(results: list) -> int:
    failed = [result for result in results if result.returncode not in (0, None)]
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
