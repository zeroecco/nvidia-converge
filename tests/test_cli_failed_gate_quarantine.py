from types import SimpleNamespace

import pytest
from test_planner import _healthy_audit

from nvidia_converge import cli
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    Finding,
    PlanAction,
    Report,
    RollbackSnapshot,
    Severity,
)


class _Runner:
    def __init__(self, **kwargs):
        del kwargs
        self.results = []


class _FailedRequiesceGuard:
    def __init__(self, events):
        self.events = events
        self.requiesce_errors = ["service state is unobservable"]

    def requiesce(self):
        self.events.append("requiesce")
        return False


class _FailedGateGuard:
    def __init__(self):
        self.results = [
            CommandResult(
                ["systemctl", "stop", "nvidia-fabricmanager.service"],
                0,
            )
        ]
        self.error = "trusted service qualification failed"
        self.restore_errors = []
        self.requiesce_errors = []
        self.mutation_started = False

    def mark_mutation_started(self):
        self.mutation_started = True

    @property
    def quiesced_service_names(self):
        return ["nvidia-fabricmanager.service"]

    def requiesce(self):
        return True


def _snapshot(audit):
    return RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        operation_id="test-operation",
        host_id="test-host",
        integrity_sha256="a" * 64,
    )


def _configure_failed_gate_command(
    monkeypatch,
    *,
    command,
    guard,
    maintenance_finding,
):
    audit = _healthy_audit()
    snapshot = _snapshot(audit)
    events = []
    reports = []

    monkeypatch.setattr(cli, "CommandRunner", _Runner)
    monkeypatch.setattr(
        cli,
        "audit_host",
        lambda runner: events.append("audit") or audit,
    )
    monkeypatch.setattr(cli, "diagnose", lambda *args: [])
    monkeypatch.setattr(
        cli,
        "_create_snapshot_with_evidence",
        lambda *args, **kwargs: (
            events.append("snapshot")
            or (snapshot, CommandResult(["persist-rollback-snapshot"], 0))
        ),
    )
    monkeypatch.setattr(
        cli,
        "preflight_snapshot_restore_availability",
        lambda *args: events.append("snapshot-preflight") or [],
    )
    monkeypatch.setattr(
        cli,
        "_maintenance_gate",
        lambda *args, **kwargs: (
            events.append("maintenance")
            or cli._MaintenanceGateOutcome(
                guard,
                [],
                None,
                [],
                [maintenance_finding],
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_defer_launcher_enablement",
        lambda *args: events.append("quarantine") or (True, audit),
    )
    monkeypatch.setattr(
        cli,
        "emit_report",
        lambda command, report, *args, **kwargs: (
            events.append("emit") or reports.append(report)
        ),
    )

    # Any of these events would mean the failed maintenance gate was crossed.
    monkeypatch.setattr(
        cli,
        "prepare_stack",
        lambda *args, **kwargs: pytest.fail(
            "verification mutation ran after a failed maintenance gate"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_plan_actions",
        lambda *args, **kwargs: pytest.fail(
            "lock mutation ran after a failed maintenance gate"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_append_launcher_release_authorization",
        lambda *args, **kwargs: pytest.fail(
            "launcher release was authorized after a failed maintenance gate"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_commit_rollback_service_activity",
        lambda *args, **kwargs: pytest.fail(
            "launcher release ran after a failed maintenance gate"
        ),
    )

    if command == "lock":
        monkeypatch.setattr(
            cli,
            "lock_actions",
            lambda *args: [
                PlanAction(
                    "lock.apt",
                    "Lock packages.",
                    [["apt-mark", "hold", "nvidia-driver"]],
                )
            ],
        )
        monkeypatch.setattr(
            cli,
            "preflight_package_lock",
            lambda *args: events.append("target-preflight") or [],
        )
        monkeypatch.setattr(
            cli,
            "resolved_forward_payload_packages",
            lambda *args: [],
        )
        monkeypatch.setattr(
            cli,
            "_bind_forward_package_payloads",
            lambda actions, _snapshot, _audit: actions,
        )
        monkeypatch.setattr(
            cli,
            "preflight_staged_forward_transaction",
            lambda *args: [],
        )

    return audit, events, reports


@pytest.mark.parametrize("command", ["verify", "lock"])
def test_applied_failed_maintenance_gate_persistently_quarantines_guarded_services(
    monkeypatch,
    command,
):
    guard = object()
    finding = Finding(
        "gpu-services.quiesce-failed",
        Severity.ERROR,
        "Trusted NVIDIA services could not be safely quiesced",
        "A trusted service failed qualification after another service stopped.",
    )
    _, events, reports = _configure_failed_gate_command(
        monkeypatch,
        command=command,
        guard=guard,
        maintenance_finding=finding,
    )

    returncode = cli._execute_command(
        SimpleNamespace(
            command=command,
            allow_disruption=True,
            allow_active_workloads=False,
        ),
        DesiredState(),
        f"/var/lib/nvidia-converge/{command}.json",
        False,
        True,
        None,
    )

    assert returncode == 2
    assert events.count("quarantine") == 1
    assert events.index("maintenance") < events.index("quarantine")
    assert events.index("quarantine") < events.index("emit")
    assert reports[-1].audit is not None
    assert any(item.id == finding.id for item in reports[-1].findings)


@pytest.mark.parametrize("command", ["verify", "lock"])
def test_missing_disruption_authorization_does_not_mutate_quarantine(
    monkeypatch,
    command,
):
    finding = Finding(
        "maintenance-window.required",
        Severity.ERROR,
        "Maintenance authorization is required",
        "No service was quiesced.",
    )
    _, events, _ = _configure_failed_gate_command(
        monkeypatch,
        command=command,
        guard=None,
        maintenance_finding=finding,
    )

    returncode = cli._execute_command(
        SimpleNamespace(
            command=command,
            allow_disruption=False,
            allow_active_workloads=False,
        ),
        DesiredState(),
        f"/var/lib/nvidia-converge/{command}.json",
        False,
        True,
        None,
    )

    assert returncode == 2
    assert "quarantine" not in events
    assert events.index("maintenance") < events.index("emit")


def test_failed_compensation_requiesce_quarantines_without_probe_or_rollback(
    monkeypatch,
):
    audit = _healthy_audit()
    events = []
    guard = _FailedRequiesceGuard(events)
    report = Report("1.2", "2026-08-02T00:00:00+00:00", DesiredState())

    monkeypatch.setattr(
        cli,
        "audit_host",
        lambda runner: events.append("audit") or audit,
    )
    monkeypatch.setattr(
        cli,
        "_defer_launcher_enablement",
        lambda *args: events.append("quarantine") or (True, audit),
    )
    monkeypatch.setattr(
        cli,
        "_retry_failed_probe_container_cleanup",
        lambda *args, **kwargs: pytest.fail(
            "probe cleanup ran after re-quiescence could not be proven"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_probe_active_gpu_workloads",
        lambda *args: pytest.fail(
            "workload probe ran after re-quiescence could not be proven"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_attempt_install_compensation",
        lambda *args, **kwargs: pytest.fail(
            "rollback ran after re-quiescence could not be proven"
        ),
    )

    current_audit = cli._prepare_install_compensation(
        report,
        _snapshot(audit),
        _Runner(),
        guard,
        allow_active_workloads=False,
    )

    assert current_audit is None
    assert events == ["requiesce", "audit", "quarantine"]
    assert any(
        finding.id == "gpu-services.requiesce-failed" for finding in report.findings
    )


def test_failed_gate_quarantine_audit_exception_uses_emergency_mask(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        cli,
        "audit_host",
        lambda runner: (_ for _ in ()).throw(RuntimeError("audit failed")),
    )
    monkeypatch.setattr(
        cli,
        "_defer_launcher_enablement",
        lambda *args: pytest.fail("defer ran without a quarantine audit"),
    )
    monkeypatch.setattr(
        cli,
        "_emergency_quarantine_launchers",
        lambda: events.append("emergency-quarantine"),
    )

    with pytest.raises(RuntimeError, match="audit failed"):
        cli._quarantine_failed_maintenance_gate(
            Report("1.2", "2026-08-02T00:00:00+00:00", DesiredState()),
            _Runner(),
            _FailedGateGuard(),
        )

    assert events == ["emergency-quarantine"]


def test_applied_install_failed_guard_is_quarantined_before_compensation(
    monkeypatch,
):
    audit = _healthy_audit()
    snapshot = _snapshot(audit)
    guard = _FailedGateGuard()
    events = []
    reports = []
    maintenance_calls = 0
    action = PlanAction(
        "verify.stack",
        "Exercise the converged stack.",
        [],
        destructive=True,
    )
    finding = Finding(
        "gpu-services.quiesce-failed",
        Severity.ERROR,
        "Trusted NVIDIA services could not be safely quiesced",
        "A trusted service failed qualification after another service stopped.",
    )

    def maintenance(*args, **kwargs):
        nonlocal maintenance_calls
        del args
        maintenance_calls += 1
        events.append(f"maintenance:{maintenance_calls}")
        if kwargs.get("quiesce_services") is False:
            return cli._MaintenanceGateOutcome(
                None,
                [],
                CommandResult(["probe-active-gpu-workloads"], 0),
                [],
                [],
            )
        return cli._MaintenanceGateOutcome(guard, [], None, [], [finding])

    monkeypatch.setattr(cli, "CommandRunner", _Runner)
    monkeypatch.setattr(cli, "audit_host", lambda runner: audit)
    monkeypatch.setattr(cli, "diagnose", lambda *args: [])
    monkeypatch.setattr(cli, "build_plan", lambda *args: [action])
    monkeypatch.setattr(cli, "_maintenance_gate", maintenance)
    monkeypatch.setattr(
        cli,
        "_create_snapshot_with_evidence",
        lambda *args, **kwargs: (
            events.append("snapshot")
            or (snapshot, CommandResult(["persist-rollback-snapshot"], 0))
        ),
    )
    monkeypatch.setattr(
        cli,
        "preflight_snapshot_restore_availability",
        lambda *args: [],
    )
    monkeypatch.setattr(
        cli,
        "_bind_forward_package_payloads",
        lambda actions, _snapshot, _audit: actions,
    )
    monkeypatch.setattr(
        cli,
        "preflight_staged_forward_transaction",
        lambda *args: [],
    )
    monkeypatch.setattr(
        cli,
        "_defer_launcher_enablement",
        lambda *args: events.append("quarantine") or (True, audit),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_install_compensation",
        lambda *args, **kwargs: events.append("compensation") or None,
    )
    monkeypatch.setattr(
        cli,
        "_record_intentionally_quiesced_services",
        lambda *args: events.append("retained-quiescence"),
    )
    monkeypatch.setattr(
        cli,
        "emit_report",
        lambda command, report, *args, **kwargs: reports.append(report),
    )
    monkeypatch.setattr(
        cli,
        "prepare_stack",
        lambda *args, **kwargs: pytest.fail(
            "stack mutation crossed a failed maintenance gate"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_append_launcher_release_authorization",
        lambda *args, **kwargs: pytest.fail(
            "launcher release crossed a failed maintenance gate"
        ),
    )

    returncode = cli._execute_command(
        SimpleNamespace(
            command="install",
            allow_disruption=True,
            allow_active_workloads=False,
        ),
        DesiredState(),
        "/var/lib/nvidia-converge/install.json",
        False,
        True,
        None,
    )

    assert returncode == 2
    assert guard.mutation_started is True
    assert events.index("snapshot") < events.index("maintenance:2")
    assert events.index("maintenance:2") < events.index("quarantine")
    assert events.index("quarantine") < events.index("compensation")
    assert reports


@pytest.mark.parametrize(
    ("guard", "expected_quarantine"),
    [(_FailedGateGuard(), True), (None, False)],
)
def test_applied_rollback_failed_gate_quarantines_only_after_service_stop(
    monkeypatch,
    guard,
    expected_quarantine,
):
    audit = _healthy_audit()
    snapshot = _snapshot(audit)
    events = []
    bound_paths = []
    finding = Finding(
        (
            "gpu-services.quiesce-failed"
            if guard is not None
            else "maintenance-window.required"
        ),
        Severity.ERROR,
        "Rollback maintenance gate failed",
        "The operation may not cross this gate.",
    )

    monkeypatch.setattr(cli, "load_snapshot", lambda path: snapshot)
    monkeypatch.setattr(cli, "CommandRunner", _Runner)
    monkeypatch.setattr(cli, "audit_host", lambda runner: audit)
    monkeypatch.setattr(cli, "diagnose", lambda *args: [])
    monkeypatch.setattr(cli, "validate_snapshot_for_apply", lambda *args: None)
    monkeypatch.setattr(cli, "preflight_package_rollback", lambda *args: [])
    monkeypatch.setattr(
        cli,
        "_append_snapshot_binding",
        lambda *args, **kwargs: bound_paths.append(kwargs["snapshot_path"]),
    )
    monkeypatch.setattr(
        cli,
        "_maintenance_gate",
        lambda *args, **kwargs: cli._MaintenanceGateOutcome(
            guard,
            [],
            None,
            [],
            [finding],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_defer_launcher_enablement",
        lambda *args: events.append("quarantine") or (True, audit),
    )
    monkeypatch.setattr(cli, "emit_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "apply_rollback",
        lambda *args, **kwargs: pytest.fail(
            "rollback mutation crossed a failed maintenance gate"
        ),
    )

    returncode = cli._execute_command(
        SimpleNamespace(
            command="rollback",
            snapshot=("/var/lib/nvidia-converge/snapshots/subdir/../test.json"),
            allow_disruption=guard is not None,
            allow_active_workloads=False,
        ),
        DesiredState(),
        "/var/lib/nvidia-converge/rollback.json",
        False,
        True,
        None,
    )

    assert returncode == 2
    assert ("quarantine" in events) is expected_quarantine
    assert bound_paths == [snapshot.path]
