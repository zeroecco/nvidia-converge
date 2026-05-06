from __future__ import annotations

import argparse
import sys

from .audit import audit_host
from .desired import load_desired
from .doctor import diagnose
from .models import Report, utc_now
from .planner import build_plan, lock_actions
from .report import sbom_from_audit, write_report
from .rollback import apply_rollback, create_snapshot, load_snapshot
from .runner import CommandRunner
from .verify import verify_stack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nvidia-converge", description="Converge a node to a desired NVIDIA driver stack.")
    _add_common_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "plan", "install", "verify", "lock", "snapshot"):
        _add_common_args(sub.add_parser(name))
    rollback = sub.add_parser("rollback")
    _add_common_args(rollback)
    rollback.add_argument("--snapshot", required=True, help="Rollback snapshot JSON created by install or snapshot.")
    args = parser.parse_args(argv)

    desired_path = getattr(args, "desired", None)
    out_path = getattr(args, "out", None)
    apply_changes = getattr(args, "apply", False)
    desired = load_desired(desired_path)
    runner = CommandRunner(apply=apply_changes)

    if args.command == "rollback":
        snapshot = load_snapshot(args.snapshot)
        results = apply_rollback(snapshot, runner)
        report = Report("1.0", utc_now(), desired, command_results=results, rollback=snapshot)
        write_report(report, out_path)
        return _status_from_results(results)

    audit = audit_host(runner)
    findings = diagnose(desired, audit)
    report = Report("1.0", utc_now(), desired, audit=audit, findings=findings, sbom=sbom_from_audit(audit))

    if args.command == "doctor":
        write_report(report, out_path)
        return 0 if all(f.severity.value != "error" for f in findings) else 2

    if args.command == "plan":
        report.plan = build_plan(desired, audit, findings)
        write_report(report, out_path)
        return 0

    if args.command == "snapshot":
        report.rollback = create_snapshot(audit)
        write_report(report, out_path)
        return 0

    if args.command == "install":
        report.rollback = create_snapshot(audit)
        report.plan = build_plan(desired, audit, findings)
        runner.results = []
        for action in report.plan:
            if action.id in {"snapshot.current-state", "verify.stack"}:
                continue
            for command in action.commands:
                runner.run(command, mutate=True, allow_fail=True)
        report.command_results = runner.results
        post_audit = audit_host(runner)
        report.audit = post_audit
        report.findings = diagnose(desired, post_audit)
        report.verification = verify_stack(desired, runner, post_audit)
        report.sbom = sbom_from_audit(post_audit)
        write_report(report, out_path)
        return 0 if all(v.ok for v in report.verification) and all(f.severity.value != "error" for f in report.findings) else 2

    if args.command == "verify":
        runner.results = []
        report.verification = verify_stack(desired, runner, audit)
        write_report(report, out_path)
        return 0 if all(v.ok for v in report.verification) else 2

    if args.command == "lock":
        report.plan = lock_actions(desired, audit)
        runner.results = []
        for action in report.plan:
            for command in action.commands:
                runner.run(command, mutate=True, allow_fail=True)
        report.command_results = runner.results
        write_report(report, out_path)
        return _status_from_results(runner.results)

    return 1


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--desired", default=argparse.SUPPRESS, help="Desired-state JSON/YAML file.")
    parser.add_argument("--out", default=argparse.SUPPRESS, help="Write machine-readable JSON report to this path. Defaults to stdout.")
    parser.add_argument("--apply", action="store_true", default=argparse.SUPPRESS, help="Apply host-mutating actions. Without this, mutating commands are dry-run.")


def _status_from_results(results: list) -> int:
    failed = [result for result in results if result.returncode not in (0, None)]
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
