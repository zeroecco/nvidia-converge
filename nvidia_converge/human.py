from __future__ import annotations

from .models import CommandResult, Report, Severity


def render_human(command: str, report: Report, *, apply: bool) -> str:
    lines: list[str] = [f"nvidia-converge {command} {'apply' if apply else 'dry-run'}".strip()]
    if report.audit:
        lines.extend(_host_lines(report))
    if report.findings:
        visible = [finding for finding in report.findings if finding.severity != Severity.INFO]
        if visible:
            lines.append("")
            lines.append("Findings:")
            for finding in visible:
                lines.append(f"- {finding.severity.value}: {finding.summary}")
                if finding.remediation:
                    lines.append(f"  fix: {finding.remediation}")
        elif command in {"doctor", "install"}:
            lines.append("")
            lines.append("Findings: no blocking issues found")
    if report.plan:
        lines.append("")
        lines.append("Plan:")
        for action in report.plan:
            lines.append(f"- {action.id}: {action.description}")
            for cmd in action.commands:
                lines.append(f"  $ {_join_command(cmd)}")
    if report.command_results:
        lines.append("")
        lines.append("Commands:")
        for result in report.command_results:
            lines.append(_result_line(result))
            detail = _failure_detail(result)
            if detail:
                lines.append(f"  {detail}")
    if report.verification:
        lines.append("")
        lines.append("Verification:")
        for check in report.verification:
            status = _verification_status(check.command, check.ok)
            lines.append(f"- {status}: {check.name}")
            if check.command and check.command.stderr and not check.ok:
                lines.append(f"  {check.command.stderr.splitlines()[0]}")
    if report.rollback:
        lines.append("")
        if report.rollback.path:
            lines.append(f"Rollback snapshot: {report.rollback.path}")
        else:
            lines.append("Rollback snapshot: preview only; no file written during dry-run")
    lines.append("")
    if not apply and command in {"install", "lock", "rollback", "snapshot", "verify"}:
        lines.append("No host changes made. Re-run with --apply to execute mutating checks/actions.")
    lines.append("Use --out report.json for the full machine-readable report, or --json to print it.")
    return "\n".join(lines)


def _host_lines(report: Report) -> list[str]:
    audit = report.audit
    assert audit is not None
    module = audit.module.version or "missing"
    module_state = "loaded" if audit.module.loaded else "not loaded"
    docker = "ok" if audit.runtime.docker_gpus_usable else "not configured" if audit.runtime.docker_gpus_usable is False else "unknown"
    fabric = "active" if audit.fabric_manager_active else "inactive" if audit.fabric_manager_active is False else "unknown"
    return [
        f"Desired: driver {report.desired.driver}, CUDA compat {report.desired.cuda_compat}, runtime {report.desired.container_runtime}",
        f"Host: {audit.os_id or 'unknown'} {audit.os_version or ''}, kernel {audit.kernel.running}",
        f"GPU stack: NVIDIA module {module} {module_state}, Docker GPU runtime {docker}, Fabric Manager {fabric}",
    ]


def _result_line(result: CommandResult) -> str:
    prefix = "skip" if result.skipped else "ok" if result.returncode == 0 else "fail"
    suffix = f" ({result.reason})" if result.reason else ""
    return f"- {prefix}: {_join_command(result.command)}{suffix}"


def _verification_status(command: CommandResult | None, ok: bool) -> str:
    if command and command.skipped:
        return "skip"
    return "ok" if ok else "fail"


def _failure_detail(result: CommandResult) -> str | None:
    if result.returncode in (0, None):
        return None
    text = result.stderr or result.stdout
    if not text:
        return None
    return text.splitlines()[0]


def _join_command(command: list[str]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(part: str) -> str:
    if not part or any(char.isspace() for char in part):
        return repr(part)
    return part
