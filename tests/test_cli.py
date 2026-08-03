import json
import os
import signal
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import nvidia_converge
from nvidia_converge.cli import (
    _attempt_install_compensation,
    _commands_succeeded,
    _commit_rollback_service_activity,
    _commit_rollback_service_activity_impl,
    _defer_launcher_enablement,
    _execute_command,
    _finalize_report,
    _install_status,
    _maintenance_gate,
    _MaintenanceGateOutcome,
    _module_reload_required,
    _probe_active_gpu_workloads,
    _record_intentionally_quiesced_services,
    _restore_rollback_unit_activity_safely,
    _run_plan_actions,
    _verify_joint_launcher_commit,
    main,
)
from nvidia_converge.human import render_human
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    Finding,
    PlanAction,
    Report,
    RollbackSnapshot,
    Severity,
    Verification,
)
from nvidia_converge.preflight import PackagePreflightError
from nvidia_converge.report import ReportWriteError


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("nvidia-converge ")


def test_process_entry_registers_sighup_for_graceful_termination(monkeypatch, capsys):
    handlers = {}
    monkeypatch.setattr(
        "nvidia_converge.cli.signal.signal",
        lambda signum, handler: handlers.setdefault(signum, handler),
    )
    monkeypatch.setattr(sys, "argv", ["nvidia-converge", "schema", "desired"])

    assert main() == 0

    from nvidia_converge.cli import _request_termination

    assert handlers[signal.SIGHUP] is _request_termination
    assert handlers[signal.SIGTERM] is _request_termination
    capsys.readouterr()


def test_package_version_matches_cli_version():
    assert nvidia_converge.__version__ == _project_version()


def test_applied_state_capacity_is_checked_before_snapshot_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_planner import _healthy_audit

    capacity_checks: list[tuple[Path, ...]] = []
    journal_events: list[str] = []
    snapshot_created = False

    class CallbackRunner:
        def __init__(self, **kwargs: object) -> None:
            self.start_callback = kwargs.get("start_callback")
            self.result_callback = kwargs.get("result_callback")
            self.results: list[CommandResult] = []

        def record_external_start(self, command: list[str], mutate: bool) -> None:
            assert callable(self.start_callback)
            self.start_callback(command, mutate)

        def record_external_result(
            self,
            result: CommandResult,
            mutate: bool,
        ) -> None:
            assert callable(self.result_callback)
            self.result_callback(result, mutate)

        @contextmanager
        def private_state_scope(self, command: list[str]):
            self.record_external_start(command, True)
            yield
            self.record_external_result(CommandResult(command, 0), True)

    def reject_capacity(*paths: Path) -> None:
        capacity_checks.append(paths)
        raise ReportWriteError("injected state-storage exhaustion")

    def create_snapshot_after_gate(*args: object, **kwargs: object) -> None:
        nonlocal snapshot_created
        del args, kwargs
        snapshot_created = True
        raise AssertionError("snapshot persistence must not begin")

    monkeypatch.setattr("nvidia_converge.cli.CommandRunner", CallbackRunner)
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: _healthy_audit(),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.require_applied_state_capacity",
        reject_capacity,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.append_report_journal",
        lambda path, operation_id, event, **details: journal_events.append(event),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.create_snapshot",
        create_snapshot_after_gate,
    )
    report = Report("1.2", "2026-08-02T00:00:00+00:00", DesiredState())

    with pytest.raises(ReportWriteError, match="state-storage exhaustion"):
        _execute_command(
            SimpleNamespace(command="snapshot"),
            DesiredState(),
            "/var/lib/nvidia-converge/reports/snapshot-test.json",
            False,
            True,
            report,
        )

    assert capacity_checks == [
        (
            Path("/var/lib/nvidia-converge/reports"),
            Path("/var/lib/nvidia-converge/snapshots"),
        )
    ]
    assert journal_events == []
    assert snapshot_created is False


def test_module_reload_is_required_for_loaded_on_disk_divergence():
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.module.installed_version = "595.71.05"

    assert _module_reload_required(audit) is True


def test_finalize_marks_observed_pending_mig_transition_as_reboot_required():
    from test_planner import _healthy_audit

    from nvidia_converge.cli import (
        _target_verification_can_complete_after_reboot,
    )

    desired = DesiredState(mig="enabled", mig_profile="full")
    audit = _healthy_audit()
    audit.mig_mode = "disabled"
    audit.mig_mode_pending = "enabled"
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        desired,
        audit=audit,
        findings=[
            Finding(
                "mig.pending-reboot",
                Severity.ERROR,
                "MIG transition pending",
                "Reboot required.",
            )
        ],
        command_results=[CommandResult(["nvidia-smi", "-mig", "1"], 0)],
        verification=[
            Verification("mig.mode", False),
            Verification("mig.geometry", False),
            Verification("mig.device-uuid", False),
            Verification("container.device-binding", False),
            Verification("container.gpu", False),
        ],
    )

    assert _target_verification_can_complete_after_reboot(report) is True
    report.verification[-1].command = CommandResult(["docker", "run"], 1)
    assert _target_verification_can_complete_after_reboot(report) is False
    report.verification[-1].command = None
    _finalize_report("install", report, None, True)

    assert report.exit_code == 2
    assert report.reboot_required is True


def test_finalize_marks_verified_compensation_complete_but_failed():
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
        findings=[
            Finding(
                "install.verification-failed",
                Severity.ERROR,
                "Verification failed",
                "Automatic rollback was required.",
            ),
            Finding(
                "install.compensation.succeeded",
                Severity.WARNING,
                "Baseline restored",
                "The rollback baseline was verified.",
            ),
        ],
        command_results=[CommandResult(["apt-get", "install"], 0)],
        verification=[
            Verification("container.gpu", False),
            Verification("install.compensation.rollback.packages-restored", True),
        ],
    )

    _finalize_report("install", report, None, True)

    assert report.exit_code == 2
    assert report.outcome == "failed"
    assert report.incomplete is False

    report.verification[0].command = CommandResult(
        ["docker", "run"],
        124,
        reason="lingering-process-group-terminated",
    )
    _finalize_report("install", report, None, True)
    assert report.incomplete is True


def test_finalize_does_not_mask_package_failure_as_mig_reboot_required():
    from test_planner import _healthy_audit

    desired = DesiredState(mig="enabled", mig_profile="full")
    audit = _healthy_audit()
    audit.mig_mode = "disabled"
    audit.mig_mode_pending = "enabled"
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        desired,
        audit=audit,
        findings=[
            Finding(
                "mig.pending-reboot",
                Severity.ERROR,
                "MIG transition pending",
                "Reboot required.",
            )
        ],
        command_results=[CommandResult(["apt-get", "install"], 100)],
        verification=[Verification("mig.mode", False)],
    )

    _finalize_report("install", report, None, True)

    assert report.reboot_required is None


def test_finalize_does_not_mask_on_disk_failure_as_mig_reboot_required():
    from test_planner import _healthy_audit

    desired = DesiredState(mig="enabled", mig_profile="full")
    audit = _healthy_audit()
    audit.mig_mode = "disabled"
    audit.mig_mode_pending = "enabled"
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        desired,
        audit=audit,
        findings=[
            Finding(
                "mig.pending-reboot",
                Severity.ERROR,
                "MIG transition pending",
                "Reboot required.",
            )
        ],
        command_results=[CommandResult(["nvidia-smi", "-mig", "1"], 0)],
        verification=[
            Verification("mig.mode", False),
            Verification("module.on-disk-version", False),
        ],
    )

    _finalize_report("install", report, None, True)

    assert report.reboot_required is None


def test_finalize_allows_only_exact_module_reload_failure_before_reboot():
    from test_planner import _healthy_audit

    desired = DesiredState()
    audit = _healthy_audit()
    audit.module.version = "570.172.08"
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        desired,
        audit=audit,
        findings=[
            Finding(
                "driver.module-version-mismatch",
                Severity.ERROR,
                "Loaded module differs",
                "A reboot will select the desired on-disk module.",
            )
        ],
        command_results=[
            CommandResult(
                ["modprobe", "-r", "nvidia_uvm", "nvidia"],
                1,
                stderr="modprobe: FATAL: Module nvidia is in use.",
            )
        ],
        verification=[
            Verification(
                "module.reload",
                False,
                CommandResult(
                    ["modprobe", "-r", "nvidia_uvm", "nvidia"],
                    1,
                    stderr="modprobe: FATAL: Module nvidia is in use.",
                ),
            ),
            Verification("module.loaded-version", False),
            Verification("module.provenance", False),
            Verification("container.cuda-driver-compatibility", False),
            Verification("container.gpu", False),
        ],
    )

    _finalize_report("install", report, None, True)

    assert report.reboot_required is True


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(
            ["modprobe", "nvidia"],
            1,
            stderr="modprobe: ERROR: could not insert 'nvidia': Key was rejected by service",
        ),
        CommandResult(
            ["modprobe", "nvidia"],
            1,
            stderr="modprobe: FATAL: Module nvidia not found in directory /lib/modules/test",
        ),
        CommandResult(
            ["modprobe", "-r", "nvidia_uvm", "nvidia"],
            1,
            stderr="modprobe: ERROR: could not remove 'nvidia': Operation not permitted",
        ),
        CommandResult(
            ["modprobe", "-r", "nvidia_uvm", "nvidia"],
            1,
            stderr="modprobe: FATAL: Module nvidia is in use.",
            reason="timeout-process-group-terminated",
        ),
    ],
)
def test_module_failure_requires_positive_busy_unload_evidence_for_reboot(result):
    from nvidia_converge.cli import _reboot_resolvable_module_command_failure

    assert _reboot_resolvable_module_command_failure(result) is False


def test_busy_module_unload_is_the_only_reboot_resolvable_modprobe_failure():
    from nvidia_converge.cli import _reboot_resolvable_module_command_failure

    result = CommandResult(
        ["modprobe", "-r", "nvidia_uvm", "nvidia"],
        1,
        stderr="modprobe: ERROR: could not remove 'nvidia': Device or resource busy",
    )

    assert _reboot_resolvable_module_command_failure(result) is True


def test_finalize_does_not_mask_service_failure_as_module_reboot_required():
    from test_planner import _healthy_audit

    desired = DesiredState()
    audit = _healthy_audit()
    audit.module.version = "570.172.08"
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        desired,
        audit=audit,
        findings=[
            Finding(
                "driver.module-version-mismatch",
                Severity.ERROR,
                "Loaded module differs",
                "A reboot will select the desired on-disk module.",
            ),
            Finding(
                "fabric-manager.inactive",
                Severity.ERROR,
                "Fabric Manager is inactive",
                "The required service is stopped.",
            ),
        ],
        command_results=[
            CommandResult(["systemctl", "start", "nvidia-fabricmanager"], 1)
        ],
        verification=[Verification("module.reload", False)],
    )

    _finalize_report("install", report, None, True)

    assert report.reboot_required is None


def test_finalize_rollback_requires_only_pending_mig_verification_failure():
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "disabled"
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        mig_mode="disabled",
        gpu_uuids=list(audit.gpu_uuids),
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
        audit=audit,
        rollback=snapshot,
        command_results=[CommandResult(["nvidia-smi", "-mig", "0"], 0)],
        verification=[
            Verification("rollback.mig-mode", False),
            Verification("rollback.mig-pending", True),
            Verification("rollback.packages-restored", True),
        ],
    )

    _finalize_report("rollback", report, None, True)

    assert report.reboot_required is True

    report.verification.append(Verification("rollback.managed-files", False))
    _finalize_report("rollback", report, None, True)

    assert report.reboot_required is None


def test_compensation_reboot_classifier_uses_snapshot_mig_state():
    from test_planner import _healthy_audit

    from nvidia_converge.cli import (
        _rollback_state_can_complete_after_reboot,
    )

    audit = _healthy_audit()
    audit.mig_mode = "enabled"
    audit.mig_mode_pending = "disabled"
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        mig_mode="disabled",
        gpu_uuids=list(audit.gpu_uuids),
    )
    checks = [
        Verification("rollback.mig-mode", False),
        Verification("rollback.mig-pending", True),
        Verification("rollback.packages-restored", True),
    ]

    assert _rollback_state_can_complete_after_reboot(snapshot, audit, checks) is True

    checks.append(Verification("rollback.managed-files", False))
    assert _rollback_state_can_complete_after_reboot(snapshot, audit, checks) is False


def test_schema_command_outputs_json(capsys):
    rc = main(["schema", "desired"])
    captured = capsys.readouterr()
    assert rc == 0
    schema = json.loads(captured.out)
    assert schema["title"] == "nvidia-converge desired state"


def test_broken_stdout_exits_without_traceback(monkeypatch):
    class BrokenStdout:
        def write(self, text):
            raise BrokenPipeError

        def flush(self):
            return None

    monkeypatch.setattr("sys.stdout", BrokenStdout())
    assert main(["schema", "desired"]) == 1


def test_support_command_outputs_human_summary(capsys):
    rc = main(["support"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("nvidia-converge support matrix")
    assert "Root-controlled Python 3.10 or newer" in captured.out
    assert "apt-get" in captured.out
    assert "Known limits:" in captured.out


def test_support_json_outputs_machine_readable_matrix(capsys):
    rc = main(["support", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    support = json.loads(captured.out)
    assert support["package_managers"]["apt-get"]["install"] is True
    assert support["package_managers"]["zypper"]["lock"] is True
    assert support["python_runtime"]["minimum_version"] == "3.10"
    assert support["python_runtime"]["workflow_candidates"] == [
        "python3.12",
        "python3.11",
        "python3.10",
        "python3",
    ]
    assert support["python_runtime"]["root_controlled_for_applied_execution"] is True


def test_validate_command_outputs_human_summary(capsys, tmp_path):
    desired = tmp_path / "desired.yaml"
    desired.write_text(
        """
---
desired:
  driver: 595.71.05
  cuda_compat: none
  container_runtime: docker
  fabric_manager: true
  kernel_policy: pin-compatible
""",
        encoding="utf-8",
    )
    rc = main(["validate", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("nvidia-converge validate")
    assert "Desired state: valid" in captured.out
    assert "595.71.05" in captured.out


def test_validate_json_outputs_machine_readable_payload(capsys):
    rc = main(["validate", "--desired", "examples/compute-580-open.yaml", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0"
    assert "generated_at" in payload
    assert payload["valid"] is True
    assert payload["desired"]["driver"] == "580-open"


def test_validate_writes_machine_readable_payload(capsys, tmp_path):
    out = tmp_path / "validation.json"
    rc = main(
        ["validate", "--desired", "examples/compute-580-open.yaml", "--out", str(out)]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "Desired state: valid" in captured.out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert "generated_at" in payload
    assert payload["valid"] is True
    assert payload["desired"]["driver"] == "580-open"


def test_lock_defaults_to_human_output(capsys, tmp_path, monkeypatch):
    from test_planner import _audit

    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: _audit())
    desired = tmp_path / "desired.yaml"
    desired.write_text(
        """
---
desired:
  driver: 595.71.05
  cuda_compat: none
  container_runtime: docker
  fabric_manager: true
  kernel_policy: pin-compatible
""",
        encoding="utf-8",
    )
    rc = main(["lock", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("nvidia-converge lock dry-run")
    assert '"audit"' not in captured.out
    assert "nvidia-driver-pinning-595.71.05" in captured.out


def test_plan_writes_machine_readable_report(tmp_path, monkeypatch):
    from test_planner import _audit

    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: _audit())
    out = tmp_path / "plan.json"
    rc = main(["plan", "--out", str(out)])
    assert rc == 2
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.2"
    assert report["desired"]["driver"] == "580-open"
    assert "audit" in report
    assert "findings" in report
    assert "plan" in report
    assert "sbom" in report


def test_json_flag_prints_machine_readable_report(capsys, monkeypatch):
    from test_planner import _audit

    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: _audit())
    rc = main(["plan", "--json"])
    captured = capsys.readouterr()
    assert rc == 2
    report = json.loads(captured.out)
    assert report["schema_version"] == "1.2"


def test_unsupported_plan_returns_failure_status(monkeypatch, tmp_path):
    from test_planner import _audit

    audit = _audit()
    audit.package_manager = None
    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)
    out = tmp_path / "unsupported-plan.json"

    rc = main(["plan", "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert report["outcome"] == "failed"
    assert report["plan"][0]["id"] == "unsupported.package-manager"


def test_bad_desired_file_is_clean_error(capsys, tmp_path):
    desired = tmp_path / "desired.yaml"
    desired.write_text("not yaml\n", encoding="utf-8")
    rc = main(["plan", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_bad_json_desired_shape_is_clean_error(capsys, tmp_path):
    desired = tmp_path / "desired.json"
    desired.write_text("[]", encoding="utf-8")
    rc = main(["validate", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "JSON must be an object" in captured.err
    assert "Traceback" not in captured.err


def test_apply_requires_root_when_not_root(capsys):
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        return
    rc = main(
        [
            "lock",
            "--apply",
            "--desired",
            "examples/compute-580-open.yaml",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "must be run as root" in captured.err


@pytest.mark.parametrize("command", ["install", "verify", "lock", "snapshot"])
def test_applied_commands_require_explicit_desired_before_side_effects(
    command, capsys, monkeypatch
):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("desired loading must not begin")

    monkeypatch.setattr("nvidia_converge.cli.load_desired", unexpected_load)
    rc = main([command, "--apply"])
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert captured.err == (
        f"error: {command} --apply requires an explicit --desired file\n"
    )


def test_read_only_commands_reject_apply(capsys):
    try:
        main(["plan", "--apply"])
    except SystemExit as exc:
        assert exc.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: --apply" in captured.err


def test_rollback_help_requires_disruption_acknowledgements(capsys):
    try:
        main(["rollback", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "--allow-disruption" in captured.out
    assert "--allow-active-workloads" in captured.out


def test_missing_nvidia_smi_is_safe_only_with_authoritative_fallback(
    monkeypatch, tmp_path
):
    from test_planner import _audit

    monkeypatch.setattr(
        "nvidia_converge.gpu_safety._PROC_ROOT", tmp_path / "missing-proc"
    )
    audit = _audit()
    runner = _MissingExecutableRunner()

    safe_result, safe_processes = _probe_active_gpu_workloads(runner, audit)
    audit.module.loaded = True
    unknown_result, unknown_processes = _probe_active_gpu_workloads(runner, audit)

    assert safe_result.returncode == 0
    assert safe_processes == []
    assert unknown_result.returncode != 0
    assert unknown_processes is None


def test_maintenance_gate_keeps_services_quiesced_when_final_scan_blocks(
    monkeypatch,
):
    from test_planner import _healthy_audit

    events = []
    guard = _GateGuard(events)

    def quiesced(runner, *, restore_on_failure):
        del runner
        assert restore_on_failure is False
        events.append("quiesce")
        return guard

    monkeypatch.setattr(
        "nvidia_converge.cli.quiesce_trusted_gpu_services",
        quiesced,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._probe_active_gpu_workloads",
        lambda runner, audit: (
            events.append("probe")
            or (CommandResult(["probe-active-gpu-workloads"], 0), ["pid:999"])
        ),
    )
    outcome = _maintenance_gate(
        SimpleNamespace(
            allow_disruption=True,
            allow_active_workloads=False,
        ),
        object(),
        _healthy_audit(),
        operation="test mutation",
    )

    assert events == ["quiesce", "probe"]
    assert [result.command for result in outcome.command_results] == [
        ["systemctl", "stop", "nvidia-fabricmanager.service"],
        ["probe-active-gpu-workloads"],
    ]
    assert [finding.id for finding in outcome.findings] == ["gpu-workloads.active"]
    assert outcome.guard is guard
    assert guard.quiesced_service_names == ["nvidia-fabricmanager.service"]


def test_maintenance_gate_leaves_trusted_service_quiesced_for_mutation(
    monkeypatch,
):
    from test_planner import _healthy_audit

    events = []
    guard = _GateGuard(events)

    def quiesced(runner, *, restore_on_failure):
        del runner
        assert restore_on_failure is False
        events.append("quiesce")
        return guard

    monkeypatch.setattr(
        "nvidia_converge.cli.quiesce_trusted_gpu_services",
        quiesced,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._probe_active_gpu_workloads",
        lambda runner, audit: (
            events.append("probe")
            or (CommandResult(["probe-active-gpu-workloads"], 0), [])
        ),
    )

    outcome = _maintenance_gate(
        SimpleNamespace(
            allow_disruption=True,
            allow_active_workloads=False,
        ),
        object(),
        _healthy_audit(),
        operation="test mutation",
    )

    assert events == ["quiesce", "probe"]
    assert outcome.findings == []
    assert outcome.guard is guard
    assert guard.quiesced_service_names == ["nvidia-fabricmanager.service"]


def test_maintenance_gate_emergency_quarantines_on_pre_mutation_signal(
    monkeypatch,
):
    from test_planner import _healthy_audit

    events = []
    guard = _GateGuard(events)

    def quiesced(runner, *, restore_on_failure):
        del runner
        assert restore_on_failure is False
        events.append("quiesce")
        return guard

    monkeypatch.setattr(
        "nvidia_converge.cli.quiesce_trusted_gpu_services",
        quiesced,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._emergency_quarantine_launchers",
        lambda: events.append("emergency-quarantine"),
    )

    def interrupted_probe(runner, audit):
        del runner, audit
        events.append("probe")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "nvidia_converge.cli._probe_active_gpu_workloads",
        interrupted_probe,
    )

    with pytest.raises(KeyboardInterrupt):
        _maintenance_gate(
            SimpleNamespace(
                allow_disruption=True,
                allow_active_workloads=False,
            ),
            object(),
            _healthy_audit(),
            operation="test mutation",
        )

    assert events == ["quiesce", "probe", "emergency-quarantine"]
    assert guard.quiesced_service_names == ["nvidia-fabricmanager.service"]


def test_converged_apply_install_requires_maintenance_ack_before_gpu_probe(
    monkeypatch,
):
    returncode, report, events = _run_converged_apply_install(
        monkeypatch,
        allow_disruption=False,
        active_workloads=[],
    )

    assert returncode == 2
    assert events == ["audit:1", "audit:2", "workload-probe"]
    assert "verify-stack" not in events
    assert "snapshot" not in events
    assert [action.id for action in report.plan] == ["verify.stack"]
    assert report.plan[0].destructive is True
    assert any(
        finding.id == "maintenance-window.required" for finding in report.findings
    )


def test_converged_apply_install_gates_then_runs_gpu_probe_without_workloads(
    monkeypatch,
):
    returncode, report, events = _run_converged_apply_install(
        monkeypatch,
        allow_disruption=True,
        active_workloads=[],
    )

    assert returncode == 0
    first_probe = events.index("workload-probe")
    snapshot = events.index("snapshot")
    mutating_maintenance = events.index("quiesce")
    assert first_probe < snapshot
    assert snapshot < mutating_maintenance
    assert mutating_maintenance < events.index("quarantine:initial")
    assert events.index("quarantine:initial") < events.index("quarantine:pre-mutation")
    assert events.index("quarantine:pre-mutation") < events.index("verify-stack")
    assert events.index("verify-stack") < events.index("install:commit:fabric-manager")
    assert any(check.name == "container.gpu" for check in report.verification)
    assert not any(
        finding.id.startswith("gpu-workloads.") for finding in report.findings
    )


def test_converged_apply_install_active_workload_blocks_gpu_probe(monkeypatch):
    returncode, report, events = _run_converged_apply_install(
        monkeypatch,
        allow_disruption=True,
        active_workloads=["pid:999"],
    )

    assert returncode == 2
    assert events == ["audit:1", "audit:2", "workload-probe"]
    assert "verify-stack" not in events
    assert "snapshot" not in events
    assert any(finding.id == "gpu-workloads.active" for finding in report.findings)


def test_converged_apply_install_active_workload_requires_explicit_override(
    monkeypatch,
):
    returncode, report, events = _run_converged_apply_install(
        monkeypatch,
        allow_disruption=True,
        active_workloads=["pid:999"],
        allow_active_workloads=True,
    )

    assert returncode == 0
    assert events.index("workload-probe") < events.index("snapshot")
    assert events.index("snapshot") < events.index("quiesce")
    assert events.index("quiesce") < events.index("verify-stack")
    assert not any(finding.id == "gpu-workloads.active" for finding in report.findings)


def test_applied_verify_binds_snapshot_before_quiesce_and_authorizes_release(
    monkeypatch,
):
    returncode, report, events, snapshot_kwargs = _run_applied_verify_or_lock(
        monkeypatch, command="verify"
    )

    assert returncode == 0
    assert snapshot_kwargs == {
        "persist": True,
        "operation_id": report.operation_id,
        "journal_report_path": Path("/var/lib/nvidia-converge/verify.json"),
    }
    assert events.index("snapshot") < events.index("maintenance")
    assert events.index("snapshot-preflight") < events.index("maintenance")
    assert events.index("fresh-quarantined-audit") < events.index(
        "rebuild-verify-operands"
    )
    assert events.index("rebuild-verify-operands") < events.index("mutate:verify")
    assert events.index("verify-core") < events.index("release:operation-target")
    assert events.index("release:operation-target") < events.index("verify:commit")
    assert report.rollback is not None
    assert report.rollback.operation_id == report.operation_id


def test_applied_lock_replans_and_repreflights_before_bound_mutation(
    monkeypatch,
):
    returncode, report, events, snapshot_kwargs = _run_applied_verify_or_lock(
        monkeypatch, command="lock"
    )

    assert returncode == 0
    assert snapshot_kwargs == {
        "persist": True,
        "operation_id": report.operation_id,
        "journal_report_path": Path("/var/lib/nvidia-converge/lock.json"),
        "forward_packages": [],
        "dnf_module_failsafe_path": None,
    }
    assert events.index("target-preflight:1") < events.index("snapshot")
    assert events.index("snapshot") < events.index("staged-target-preflight:1")
    assert events.index("staged-target-preflight:1") < events.index("maintenance")
    assert events.index("snapshot") < events.index("maintenance")
    assert events.index("snapshot-preflight") < events.index("maintenance")
    assert events.index("fresh-quarantined-audit") < events.index("plan:2")
    assert events.index("plan:2") < events.index("staged-target-preflight:2")
    assert events.index("staged-target-preflight:2") < events.index(
        "fresh-rollback-preflight"
    )
    assert events.index("fresh-rollback-preflight") < events.index("mutate:lock")
    assert events.index("release:operation-target") < events.index("lock:commit")
    assert report.rollback is not None
    assert report.rollback.operation_id == report.operation_id


def test_mutation_failure_reports_exact_services_left_quiesced():
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )
    guard = _GateGuard([])

    _record_intentionally_quiesced_services(report, guard)

    assert [finding.id for finding in report.findings] == [
        "gpu-services.intentionally-quiesced"
    ]
    assert "nvidia-fabricmanager.service" in report.findings[0].detail
    assert guard.events == ["requiesce"]


def test_mutation_failure_does_not_claim_services_are_stopped_when_requiesce_fails():
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )
    guard = _GateGuard([])
    guard.requiesce_errors = ["service state is unobservable"]
    guard.requiesce = lambda: False

    _record_intentionally_quiesced_services(report, guard)

    assert [finding.id for finding in report.findings] == [
        "gpu-services.requiesce-failed"
    ]
    assert "unobservable" in report.findings[0].detail


def test_defer_launcher_enablement_masks_exact_unit_order_for_all_states(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.docker_socket_unit_file_state = "enabled"
    audit.docker_socket_enabled = True
    audit.docker_socket_active = True
    audit.docker_service_unit_file_state = "static"
    audit.docker_service_enabled = False
    audit.docker_service_active = True
    audit.nvidia_persistenced_unit_file_state = "not-found"
    audit.nvidia_persistenced_enabled = False
    audit.nvidia_persistenced_active = False
    audit.fabric_manager_unit_file_state = "masked"
    audit.fabric_manager_enabled = False
    audit.fabric_manager_active = False

    quarantined = deepcopy(audit)
    for active_name, enabled_name, state_name in (
        (
            "docker_socket_active",
            "docker_socket_enabled",
            "docker_socket_unit_file_state",
        ),
        (
            "docker_service_active",
            "docker_service_enabled",
            "docker_service_unit_file_state",
        ),
        (
            "nvidia_persistenced_active",
            "nvidia_persistenced_enabled",
            "nvidia_persistenced_unit_file_state",
        ),
        (
            "fabric_manager_active",
            "fabric_manager_enabled",
            "fabric_manager_unit_file_state",
        ),
    ):
        setattr(quarantined, active_name, False)
        setattr(quarantined, enabled_name, False)
        setattr(quarantined, state_name, "masked")

    commands = []

    class RecordingRunner:
        pass

    def quarantine(runner, unit):
        del runner
        command = ["quarantine-launcher", unit]
        commands.append(command)
        return [CommandResult(command, 0)]

    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: quarantined,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_unit",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_docker_socket_unit",
        lambda *args, **kwargs: ([], None),
    )

    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_service_for_rollback",
        quarantine,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _defer_launcher_enablement(
        report,
        RecordingRunner(),
        audit,
    )

    assert ok is True
    assert observed is quarantined
    assert commands == [
        ["quarantine-launcher", "docker.socket"],
        ["quarantine-launcher", "docker.service"],
        ["quarantine-launcher", "nvidia-persistenced.service"],
        ["quarantine-launcher", "nvidia-fabricmanager.service"],
    ]
    assert [result.command for result in report.command_results] == commands
    assert report.findings == []


def test_defer_launcher_enablement_rejects_untrusted_active_docker_before_mutation(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.docker_service_active = True
    audit.nvidia_persistenced_active = False
    audit.fabric_manager_active = False
    validation = CommandResult(["validate", "docker.service"], 1)

    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_unit",
        lambda *args, **kwargs: (
            [validation],
            "cannot trust docker.service: unsafe ExecStop hook",
        ),
    )

    class NoMutationRunner:
        @staticmethod
        def run(*args, **kwargs):
            raise AssertionError("launcher mutation must not run")

    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _defer_launcher_enablement(
        report,
        NoMutationRunner(),
        audit,
    )

    assert ok is False
    assert observed is audit
    assert report.command_results == [validation]
    assert [finding.id for finding in report.findings] == [
        "install.launcher-quarantine-service-untrusted"
    ]
    assert report.findings[0].evidence == {"unit": "docker.service"}


def test_defer_launcher_enablement_rejects_untrusted_socket_without_mutation(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    validation = CommandResult(["validate", "docker.socket"], 1)
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_unit",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_docker_socket_unit",
        lambda *args, **kwargs: (
            [validation],
            "cannot trust docker.socket: unsafe ExecStopPost hook",
        ),
    )

    class NoMutationRunner:
        @staticmethod
        def run(*args, **kwargs):
            raise AssertionError("launcher mutation must not run")

    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _defer_launcher_enablement(
        report,
        NoMutationRunner(),
        audit,
    )

    assert ok is False
    assert observed is audit
    assert report.command_results == [validation]
    assert [finding.id for finding in report.findings] == [
        "install.launcher-quarantine-socket-untrusted"
    ]
    assert report.findings[0].evidence == {"unit": "docker.socket"}


def test_defer_running_socket_masks_rebinds_then_stops(monkeypatch):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.docker_socket_active = True
    audit.docker_socket_enabled = True
    audit.docker_socket_unit_file_state = "enabled"
    for active_name, enabled_name, state_name in (
        (
            "docker_service_active",
            "docker_service_enabled",
            "docker_service_unit_file_state",
        ),
        (
            "nvidia_persistenced_active",
            "nvidia_persistenced_enabled",
            "nvidia_persistenced_unit_file_state",
        ),
        (
            "fabric_manager_active",
            "fabric_manager_enabled",
            "fabric_manager_unit_file_state",
        ),
    ):
        setattr(audit, active_name, False)
        setattr(audit, enabled_name, False)
        setattr(audit, state_name, "masked")
    states = {
        "docker.socket": [True, "enabled", "loaded"],
        "docker.service": [False, "masked", "masked"],
        "nvidia-persistenced.service": [False, "masked", "masked"],
        "nvidia-fabricmanager.service": [False, "masked", "masked"],
    }
    mutations = []
    events = []

    class StateRunner:
        apply = True

        def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
            del allow_fail, input_text
            unit = command[-1]
            state = states[unit]
            if command[:2] == ["systemctl", "show"]:
                return CommandResult(
                    command,
                    0,
                    stdout=(
                        f"Id={unit}\n"
                        f"LoadState={state[2]}\n"
                        f"ActiveState={'active' if state[0] else 'inactive'}\n"
                        f"UnitFileState={state[1]}\n"
                    ),
                )
            assert mutate is True
            mutations.append(command)
            if command[:2] == ["systemctl", "mask"]:
                state[1] = "masked"
                if unit != "docker.socket":
                    state[2] = "masked"
                events.append(f"mask:{unit}")
            elif command[:2] == ["systemctl", "stop"]:
                state[0] = False
                events.append(f"stop:{unit}")
            else:
                raise AssertionError(f"unexpected command: {command}")
            return CommandResult(command, 0)

    socket_identity = object()
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_unit",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_docker_socket_unit",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback.validate_trusted_gpu_service_unit",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.rollback.validate_trusted_docker_socket_unit_identity",
        lambda *args, **kwargs: ([], socket_identity, None),
    )

    def revalidate(runner, identity):
        del runner
        assert identity is socket_identity
        events.append("rebind:docker.socket")
        return [], None

    monkeypatch.setattr(
        "nvidia_converge.rollback.revalidate_trusted_docker_socket_identity",
        revalidate,
    )
    quarantined = deepcopy(audit)
    quarantined.docker_socket_active = False
    quarantined.docker_socket_enabled = False
    quarantined.docker_socket_unit_file_state = "masked"
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: quarantined,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _defer_launcher_enablement(report, StateRunner(), audit)

    assert ok is True
    assert observed is quarantined
    assert mutations[:2] == [
        ["systemctl", "mask", "docker.socket"],
        ["systemctl", "stop", "docker.socket"],
    ]
    assert events[:3] == [
        "mask:docker.socket",
        "rebind:docker.socket",
        "stop:docker.socket",
    ]


def test_defer_launcher_enablement_fails_closed_on_mask_fault(monkeypatch):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.docker_socket_unit_file_state = "static"
    audit.docker_socket_enabled = False
    for active_name, enabled_name, state_name in (
        (
            "docker_service_active",
            "docker_service_enabled",
            "docker_service_unit_file_state",
        ),
        (
            "nvidia_persistenced_active",
            "nvidia_persistenced_enabled",
            "nvidia_persistenced_unit_file_state",
        ),
        (
            "fabric_manager_active",
            "fabric_manager_enabled",
            "fabric_manager_unit_file_state",
        ),
    ):
        setattr(audit, active_name, False)
        setattr(audit, enabled_name, False)
        setattr(audit, state_name, "masked")
    quarantined = deepcopy(audit)
    quarantined.docker_socket_active = False
    quarantined.docker_socket_unit_file_state = "masked"
    class FaultRunner:
        pass

    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: quarantined,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_unit",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_docker_socket_unit",
        lambda *args, **kwargs: ([], None),
    )
    mask_failure = CommandResult(
        ["systemctl", "mask", "docker.socket"],
        1,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_service_for_rollback",
        lambda runner, unit: [mask_failure],
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, _ = _defer_launcher_enablement(report, FaultRunner(), audit)

    assert ok is False
    assert report.command_results == [mask_failure]
    assert [finding.id for finding in report.findings] == [
        "install.launcher-quarantine-unverified"
    ]
    assert report.findings[0].evidence["failed_commands"] == [
        {
            "command": ["systemctl", "mask", "docker.socket"],
            "returncode": 1,
            "stderr": "",
        }
    ]


def test_common_launcher_commit_requarantines_on_interruption(monkeypatch):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=True,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=True,
        nvidia_persistenced_unit_file_state="enabled",
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="disabled",
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )
    events = []

    def interrupted_commit(*args, **kwargs):
        del args, kwargs
        events.append("partially-released")
        raise KeyboardInterrupt

    def quarantined(*args, **kwargs):
        del args, kwargs
        events.append("requarantined")
        return False, audit

    monkeypatch.setattr(
        "nvidia_converge.cli._commit_rollback_service_activity_impl",
        interrupted_commit,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_failed_launcher_commit",
        quarantined,
    )

    with pytest.raises(KeyboardInterrupt):
        _commit_rollback_service_activity(
            SimpleNamespace(),
            report,
            snapshot,
            object(),
            audit,
            None,
            operation="install",
        )

    assert events == ["partially-released", "requarantined"]


@pytest.mark.parametrize(
    ("socket_stop_returncode", "expected_ok"), [(0, True), (1, False)]
)
def test_docker_socket_activity_is_proven_before_final_persistent_commit(
    monkeypatch,
    socket_stop_returncode,
    expected_ok,
):
    from test_planner import _healthy_audit

    initial = _healthy_audit()
    initial.docker_service_active = False
    initial.docker_service_enabled = False
    initial.docker_service_unit_file_state = "masked"
    initial.docker_socket_active = False
    initial.docker_socket_enabled = False
    initial.docker_socket_unit_file_state = "masked"
    initial.nvidia_persistenced_active = False
    initial.nvidia_persistenced_enabled = False
    initial.nvidia_persistenced_unit_file_state = "masked"
    initial.fabric_manager_active = False
    initial.fabric_manager_enabled = False
    initial.fabric_manager_unit_file_state = "masked"
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=initial.kernel.running,
        module_version=initial.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=False,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="disabled",
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="disabled",
    )
    started = deepcopy(initial)
    started.docker_service_active = True
    started.docker_service_unit_file_state = "disabled"
    started.docker_socket_active = True
    started.docker_socket_unit_file_state = "disabled"
    started.nvidia_persistenced_unit_file_state = "disabled"
    started.fabric_manager_unit_file_state = "disabled"
    events = []
    finalized_units = set()
    docker_identity = object()

    def staged(*args, unit, **kwargs):
        del args, kwargs
        events.append(f"staged:{unit}")
        return True, initial

    def prepared(*args, unit, **kwargs):
        del args, kwargs
        events.append(f"prepare:{unit}")
        return True, started

    def restored_activity(*args, unit, **kwargs):
        target = args[1]
        trusted_identities = kwargs.get("trusted_identities")
        if unit == "docker.service" and trusted_identities is not None:
            trusted_identities[unit] = docker_identity
        if unit == "docker.socket" and target.docker_socket_active is False:
            events.append("stop:docker.socket")
            if socket_stop_returncode:
                return False, started
            events.append("proof:docker.socket-inactive")
            stopped = deepcopy(started)
            stopped.docker_socket_active = False
            return True, stopped
        events.append(f"activity:{unit}")
        return True, started

    def validate(runner, unit, *, expected_identity=None):
        del runner
        assert unit == "docker.service"
        assert expected_identity is docker_identity
        return [CommandResult(["validate", unit], 0)], docker_identity, None

    def docker_gate(*args, **kwargs):
        del args, kwargs
        events.append("gate:docker")
        return True, started

    def finalized(*args, unit, **kwargs):
        del args, kwargs
        events.append(f"finalize:{unit}")
        finalized_units.add(unit)
        observed = deepcopy(started)
        observed.docker_socket_active = False
        if "docker.service" in finalized_units:
            observed.docker_service_unit_file_state = "enabled"
            observed.docker_service_enabled = True
        if "docker.socket" in finalized_units:
            observed.docker_socket_unit_file_state = "enabled"
            observed.docker_socket_enabled = True
        return True, observed

    class SocketRunner:
        @staticmethod
        def run(*args, **kwargs):
            raise AssertionError("direct socket mutation must use the safe helper")

    def audited(runner):
        del runner
        events.append("proof:docker.socket-inactive")
        observed = deepcopy(started)
        observed.docker_socket_active = False
        if "docker.service" in finalized_units:
            observed.docker_service_unit_file_state = "enabled"
            observed.docker_service_enabled = True
        if "docker.socket" in finalized_units:
            observed.docker_socket_unit_file_state = "enabled"
            observed.docker_socket_enabled = True
        return observed

    def quarantined(*args, **kwargs):
        del args, kwargs
        events.append("quarantine")
        return False, initial

    monkeypatch.setattr(
        "nvidia_converge.cli._restore_rollback_unit_staged",
        staged,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._prepare_rollback_unit_safely",
        prepared,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._restore_rollback_unit_activity_safely",
        restored_activity,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._post_docker_workload_checkpoint",
        docker_gate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._finalize_rollback_unit_state",
        finalized,
    )
    monkeypatch.setattr("nvidia_converge.cli.audit_host", audited)
    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_failed_launcher_commit",
        quarantined,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, _ = _commit_rollback_service_activity_impl(
        SimpleNamespace(),
        report,
        snapshot,
        SocketRunner(),
        initial,
        None,
        operation="rollback",
    )

    assert ok is expected_ok
    assert events.index("gate:docker") < events.index("stop:docker.socket")
    if expected_ok:
        assert events.index("stop:docker.socket") < events.index(
            "proof:docker.socket-inactive"
        )
        assert events.index("proof:docker.socket-inactive") < events.index(
            "finalize:docker.service"
        )
        final_service = events.index("finalize:docker.service")
        final_socket = events.index("finalize:docker.socket")
        assert final_service < final_socket
        assert final_socket < len(events) - 1
        assert events[final_socket + 1 :] == [
            "gate:docker",
            "proof:docker.socket-inactive",
        ]
    else:
        assert "proof:docker.socket-inactive" not in events
        assert not any(event.startswith("finalize:") for event in events)
        assert events[-1] == "quarantine"


def test_inactive_enabled_launchers_are_proven_active_then_stopped_before_enable(
    monkeypatch,
):
    from test_planner import _healthy_audit

    state = _healthy_audit()
    launcher_attributes = {
        "nvidia-fabricmanager.service": (
            "fabric_manager_active",
            "fabric_manager_enabled",
            "fabric_manager_unit_file_state",
        ),
        "nvidia-persistenced.service": (
            "nvidia_persistenced_active",
            "nvidia_persistenced_enabled",
            "nvidia_persistenced_unit_file_state",
        ),
        "docker.service": (
            "docker_service_active",
            "docker_service_enabled",
            "docker_service_unit_file_state",
        ),
        "docker.socket": (
            "docker_socket_active",
            "docker_socket_enabled",
            "docker_socket_unit_file_state",
        ),
    }
    for active_name, enabled_name, state_name in launcher_attributes.values():
        setattr(state, active_name, False)
        setattr(state, enabled_name, False)
        setattr(state, state_name, "masked")
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=state.kernel.running,
        module_version=state.module.version,
        commands=[],
        docker_service_active=False,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=False,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=True,
        nvidia_persistenced_unit_file_state="enabled",
        fabric_manager_active=False,
        fabric_manager_enabled=True,
        fabric_manager_unit_file_state="enabled",
    )
    events = []
    audit_count = 0
    identities = {}

    def active_target(target, unit):
        active_name, _, _ = launcher_attributes[unit]
        return getattr(target, active_name)

    def prepare_activity(target, runner, current_audit, *, units):
        del target, runner, current_audit
        unit = next(iter(units))
        active_name, enabled_name, state_name = launcher_attributes[unit]
        assert getattr(state, active_name) is False
        events.append(f"prepare:{unit}")
        setattr(state, enabled_name, False)
        setattr(state, state_name, "disabled")
        return [CommandResult(["prepare-service", unit], 0)]

    def restore_activity(target, runner, current_audit, *, units):
        del runner, current_audit
        unit = next(iter(units))
        active_name, _, state_name = launcher_attributes[unit]
        target_active = active_target(target, unit)
        assert getattr(state, state_name) == "disabled"
        if getattr(state, active_name) is target_active:
            return []
        verb = "start" if target_active else "stop"
        events.append(f"{verb}:{unit}")
        setattr(state, active_name, target_active)
        return [CommandResult(["systemctl", verb, unit], 0)]

    def restore_enablement(target, runner, current_audit, *, units):
        del target, runner, current_audit
        unit = next(iter(units))
        active_name, enabled_name, state_name = launcher_attributes[unit]
        assert getattr(state, active_name) is False
        assert getattr(state, state_name) == "disabled"
        events.append(f"enable:{unit}")
        setattr(state, enabled_name, True)
        setattr(state, state_name, "enabled")
        return [CommandResult(["systemctl", "enable", unit], 0)]

    def audited(runner):
        nonlocal audit_count
        del runner
        audit_count += 1
        events.append(f"audit:{audit_count}")
        return deepcopy(state)

    def validate_start(runner, unit):
        del runner
        events.append(f"validate-start:{unit}")
        return [CommandResult(["validate-start", unit], 0)], None

    def validate_active(runner, unit, *, expected_identity=None):
        del runner
        identity = identities.setdefault(unit, object())
        if expected_identity is None:
            events.append(f"validate-active:{unit}")
        else:
            assert expected_identity is identity
            events.append(f"revalidate-active:{unit}")
        return [CommandResult(["validate-active", unit], 0)], identity, None

    def docker_gate(*args, **kwargs):
        del args, kwargs
        assert state.docker_service_active is True
        assert state.docker_socket_active is True
        assert state.docker_service_unit_file_state == "disabled"
        assert state.docker_socket_unit_file_state == "disabled"
        events.append("gate:docker")
        return True, deepcopy(state)

    class StateRunner:
        @staticmethod
        def run(command, *, mutate=False, allow_fail=True):
            assert mutate is True
            assert allow_fail is True
            _, verb, unit = command
            active_name, _, _ = launcher_attributes[unit]
            events.append(f"{verb}:{unit}")
            setattr(state, active_name, verb == "start")
            return CommandResult(command, 0)

    monkeypatch.setattr(
        "nvidia_converge.cli.prepare_rollback_service_activity",
        prepare_activity,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.restore_rollback_service_activity",
        restore_activity,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.restore_rollback_service_enablement",
        restore_enablement,
    )
    monkeypatch.setattr("nvidia_converge.cli.audit_host", audited)
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_start",
        validate_start,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate_active,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._post_docker_workload_checkpoint",
        docker_gate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_failed_launcher_commit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("healthy staged commit must not re-quarantine")
        ),
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _commit_rollback_service_activity_impl(
        SimpleNamespace(),
        report,
        snapshot,
        StateRunner(),
        deepcopy(state),
        None,
        operation="rollback",
    )

    assert ok is True
    for active_name, enabled_name, state_name in launcher_attributes.values():
        assert getattr(observed, active_name) is False
        assert getattr(observed, enabled_name) is True
        assert getattr(observed, state_name) == "enabled"

    def assert_ordered(*ordered_events):
        position = -1
        for expected in ordered_events:
            position = events.index(expected, position + 1)

    for unit in (
        "nvidia-fabricmanager.service",
        "nvidia-persistenced.service",
    ):
        assert_ordered(
            f"prepare:{unit}",
            f"validate-start:{unit}",
            f"start:{unit}",
            f"validate-active:{unit}",
            f"revalidate-active:{unit}",
            f"stop:{unit}",
            f"enable:{unit}",
        )
    assert_ordered(
        "prepare:docker.socket",
        "prepare:docker.service",
        "start:docker.socket",
        "start:docker.service",
        "gate:docker",
        "stop:docker.service",
        "stop:docker.socket",
        "enable:docker.service",
        "enable:docker.socket",
    )
    mutations = [
        event
        for event in events
        if event.startswith(("prepare:", "start:", "stop:", "enable:"))
    ]
    assert mutations[-2:] == [
        "enable:docker.service",
        "enable:docker.socket",
    ]


def test_staged_trusted_service_retains_validated_process_identity(
    monkeypatch,
):
    from test_planner import _healthy_audit

    initial = _healthy_audit()
    initial.nvidia_persistenced_active = False
    initial.nvidia_persistenced_enabled = False
    initial.nvidia_persistenced_unit_file_state = "disabled"
    active = deepcopy(initial)
    active.nvidia_persistenced_active = True
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=initial.kernel.running,
        module_version=initial.module.version,
        commands=[],
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="disabled",
    )
    identity = object()
    identities = {}
    validation_calls = 0

    def validate(runner, unit, *, expected_identity=None):
        nonlocal validation_calls
        del runner
        validation_calls += 1
        assert unit == "nvidia-persistenced.service"
        assert expected_identity is None or expected_identity is identity
        return [CommandResult(["validate", unit], 0)], identity, None

    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_gpu_service_start",
        lambda *args: ([CommandResult(["validate-start"], 0)], None),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.restore_rollback_service_activity",
        lambda *args, **kwargs: [CommandResult(["systemctl", "start"], 0)],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: active,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _restore_rollback_unit_activity_safely(
        report,
        snapshot,
        object(),
        initial,
        unit="nvidia-persistenced.service",
        operation="rollback",
        trusted_identities=identities,
    )

    assert ok is True
    assert observed is active
    assert validation_calls == 2
    assert identities == {"nvidia-persistenced.service": identity}


def test_active_docker_stop_is_refused_when_service_trust_fails(monkeypatch):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.docker_service_active = True
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        docker_service_active=False,
    )
    validation = CommandResult(["validate", "docker.service"], 1)
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        lambda *args, **kwargs: (
            [validation],
            None,
            "cannot trust docker.service: unsafe ExecStopPost hook",
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.restore_rollback_service_activity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("untrusted service must not be stopped")
        ),
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _restore_rollback_unit_activity_safely(
        report,
        snapshot,
        object(),
        audit,
        unit="docker.service",
        operation="rollback",
    )

    assert ok is False
    assert observed is audit
    assert report.command_results == [validation]
    assert [finding.id for finding in report.findings] == [
        "rollback.trusted-service-stop-refused"
    ]
    assert report.findings[0].evidence == {"unit": "docker.service"}


def test_joint_launcher_proof_brackets_final_audit_and_workload_gate(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=True,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=True,
        nvidia_persistenced_unit_file_state="enabled",
        fabric_manager_active=True,
        fabric_manager_enabled=True,
        fabric_manager_unit_file_state="enabled",
    )
    events = []
    identities = {
        "nvidia-fabricmanager.service": object(),
        "nvidia-persistenced.service": object(),
        "docker.service": object(),
    }
    validation_counts = {}

    def validate(runner, unit, *, expected_identity=None):
        del runner
        identity = identities[unit]
        assert expected_identity is identity
        validation_counts[unit] = validation_counts.get(unit, 0) + 1
        if validation_counts[unit] == 1:
            events.append(f"identity-a:{unit}")
        else:
            events.append(f"identity-b:{unit}")
        return [CommandResult(["validate", unit], 0)], identity, None

    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._post_docker_workload_checkpoint",
        lambda *args, **kwargs: events.append("workload-gate") or (True, audit),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: events.append("joint-audit") or audit,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _verify_joint_launcher_commit(
        SimpleNamespace(),
        report,
        snapshot,
        object(),
        audit,
        operation="rollback",
        trusted_identities=identities,
    )

    assert ok is True
    assert observed is audit
    assert events == [
        "identity-a:nvidia-fabricmanager.service",
        "identity-a:nvidia-persistenced.service",
        "identity-a:docker.service",
        "workload-gate",
        "joint-audit",
        "identity-b:nvidia-fabricmanager.service",
        "identity-b:nvidia-persistenced.service",
        "identity-b:docker.service",
    ]


@pytest.mark.parametrize(
    ("unit", "state_field"),
    [
        (
            "nvidia-fabricmanager.service",
            "fabric_manager_unit_file_state",
        ),
        (
            "nvidia-persistenced.service",
            "nvidia_persistenced_unit_file_state",
        ),
        ("docker.service", "docker_service_unit_file_state"),
        ("docker.socket", "docker_socket_unit_file_state"),
    ],
)
def test_joint_launcher_proof_rejects_any_final_unit_drift(
    monkeypatch,
    unit,
    state_field,
):
    from test_planner import _healthy_audit

    expected = _healthy_audit()
    drifted = deepcopy(expected)
    setattr(drifted, state_field, "disabled")
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=expected.kernel.running,
        module_version=expected.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=True,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=True,
        nvidia_persistenced_unit_file_state="enabled",
        fabric_manager_active=True,
        fabric_manager_enabled=True,
        fabric_manager_unit_file_state="enabled",
    )
    identities = {}

    def validate(runner, service, *, expected_identity=None):
        del runner
        identity = identities.setdefault(service, object())
        assert expected_identity is None or expected_identity is identity
        return [CommandResult(["validate", service], 0)], identity, None

    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._post_docker_workload_checkpoint",
        lambda *args, **kwargs: (True, expected),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: drifted,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, _ = _verify_joint_launcher_commit(
        SimpleNamespace(),
        report,
        snapshot,
        object(),
        expected,
        operation="rollback",
    )

    assert ok is False
    finding = next(
        finding
        for finding in report.findings
        if finding.id == "rollback.launcher-final-state-unverified"
    )
    assert unit in finding.evidence["mismatches"]


@pytest.mark.parametrize(
    "changed_unit",
    [
        "nvidia-fabricmanager.service",
        "nvidia-persistenced.service",
    ],
)
def test_joint_launcher_proof_rejects_final_trusted_identity_change(
    monkeypatch,
    changed_unit,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=True,
        docker_socket_enabled=True,
        docker_socket_unit_file_state="enabled",
        nvidia_persistenced_active=True,
        nvidia_persistenced_enabled=True,
        nvidia_persistenced_unit_file_state="enabled",
        fabric_manager_active=True,
        fabric_manager_enabled=True,
        fabric_manager_unit_file_state="enabled",
    )
    events = []
    identities = {}

    def validate(runner, unit, *, expected_identity=None):
        del runner
        identity = identities.setdefault(unit, object())
        if expected_identity is None:
            events.append(f"identity-a:{unit}")
            return [CommandResult(["validate", unit], 0)], identity, None
        events.append(f"identity-b:{unit}")
        error = "service process identity changed" if unit == changed_unit else None
        return (
            [CommandResult(["validate", unit], 1 if error else 0)],
            None if error else identity,
            error,
        )

    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._post_docker_workload_checkpoint",
        lambda *args, **kwargs: events.append("workload-gate") or (True, audit),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: events.append("joint-audit") or audit,
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, _ = _verify_joint_launcher_commit(
        SimpleNamespace(),
        report,
        snapshot,
        object(),
        audit,
        operation="rollback",
    )

    assert ok is False
    assert events.index("workload-gate") < events.index("joint-audit")
    assert events.index("joint-audit") < events.index(f"identity-b:{changed_unit}")
    assert any(
        finding.id == "rollback.launcher-final-identity-changed"
        and finding.evidence["unit"] == changed_unit
        for finding in report.findings
    )


def test_common_launcher_commit_requarantines_after_joint_proof_failure(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    for active_name, enabled_name, state_name in (
        (
            "docker_service_active",
            "docker_service_enabled",
            "docker_service_unit_file_state",
        ),
        (
            "docker_socket_active",
            "docker_socket_enabled",
            "docker_socket_unit_file_state",
        ),
        (
            "nvidia_persistenced_active",
            "nvidia_persistenced_enabled",
            "nvidia_persistenced_unit_file_state",
        ),
        (
            "fabric_manager_active",
            "fabric_manager_enabled",
            "fabric_manager_unit_file_state",
        ),
    ):
        setattr(audit, active_name, False)
        setattr(audit, enabled_name, False)
        setattr(audit, state_name, "disabled")
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        docker_service_active=False,
        docker_service_enabled=False,
        docker_service_unit_file_state="disabled",
        docker_socket_active=False,
        docker_socket_enabled=False,
        docker_socket_unit_file_state="disabled",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="disabled",
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="disabled",
    )
    events = []

    monkeypatch.setattr(
        "nvidia_converge.cli._restore_rollback_unit_staged",
        lambda *args, unit, **kwargs: events.append(f"staged:{unit}") or (True, audit),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._prepare_rollback_unit_safely",
        lambda *args, unit, **kwargs: (
            events.append(f"prepared:{unit}") or (True, audit)
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._restore_rollback_unit_activity_safely",
        lambda *args, unit, **kwargs: (
            events.append(f"activity:{unit}") or (True, audit)
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._finalize_rollback_unit_state",
        lambda *args, unit, **kwargs: (
            events.append(f"finalized:{unit}") or (True, audit)
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._verify_joint_launcher_commit",
        lambda *args, **kwargs: events.append("joint-proof-failed") or (False, audit),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_failed_launcher_commit",
        lambda *args, **kwargs: events.append("requarantine") or (False, audit),
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, _ = _commit_rollback_service_activity_impl(
        SimpleNamespace(),
        report,
        snapshot,
        object(),
        audit,
        None,
        operation="rollback",
    )

    assert ok is False
    assert events[-4:] == [
        "finalized:docker.service",
        "finalized:docker.socket",
        "joint-proof-failed",
        "requarantine",
    ]


def test_active_docker_with_masked_socket_uses_temporary_dependency(
    monkeypatch,
):
    from test_planner import _healthy_audit

    state = _healthy_audit()
    state.fabric_manager_active = False
    state.fabric_manager_enabled = False
    state.fabric_manager_unit_file_state = "disabled"
    state.nvidia_persistenced_active = False
    state.nvidia_persistenced_enabled = False
    state.nvidia_persistenced_unit_file_state = "disabled"
    state.docker_service_active = False
    state.docker_service_enabled = False
    state.docker_service_unit_file_state = "masked"
    state.docker_socket_active = False
    state.docker_socket_enabled = False
    state.docker_socket_unit_file_state = "masked"
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=state.kernel.running,
        module_version=state.module.version,
        commands=[],
        docker_service_active=True,
        docker_service_enabled=True,
        docker_service_unit_file_state="enabled",
        docker_socket_active=False,
        docker_socket_enabled=False,
        docker_socket_unit_file_state="masked",
        nvidia_persistenced_active=False,
        nvidia_persistenced_enabled=False,
        nvidia_persistenced_unit_file_state="disabled",
        fabric_manager_active=False,
        fabric_manager_enabled=False,
        fabric_manager_unit_file_state="disabled",
    )
    events = []
    docker_identity = object()

    def staged(*args, unit, **kwargs):
        del args, kwargs
        events.append(f"staged:{unit}")
        return True, deepcopy(state)

    def prepared(report, target, runner, audit, *, unit, operation):
        del report, runner, audit, operation
        events.append(f"prepare:{unit}")
        if unit == "docker.socket":
            assert target.docker_socket_active is True
            assert target.docker_socket_unit_file_state == "disabled"
            state.docker_socket_unit_file_state = "disabled"
        else:
            assert state.docker_socket_unit_file_state == "disabled"
            state.docker_service_unit_file_state = "disabled"
        return True, deepcopy(state)

    def activity(
        report,
        target,
        runner,
        audit,
        *,
        unit,
        operation,
        trusted_identities=None,
    ):
        del report, runner, audit, operation
        if unit == "docker.socket":
            if target.docker_socket_active:
                events.append("start:docker.socket")
                state.docker_socket_active = True
            else:
                events.append("stop:docker.socket")
                state.docker_socket_active = False
        else:
            events.append(f"start:{unit}")
            assert target.docker_service_active is True
            assert state.docker_socket_active is True
            state.docker_service_active = True
            assert trusted_identities is not None
            trusted_identities[unit] = docker_identity
        return True, deepcopy(state)

    def validate(runner, unit, *, expected_identity=None):
        del runner
        assert unit == "docker.service"
        assert expected_identity is docker_identity
        return [CommandResult(["validate", unit], 0)], docker_identity, None

    def finalized(report, target, runner, audit, *, unit, operation):
        del report, target, runner, audit, operation
        assert unit == "docker.service"
        events.append("enable:docker.service")
        state.docker_service_enabled = True
        state.docker_service_unit_file_state = "enabled"
        return True, deepcopy(state)

    class StateRunner:
        @staticmethod
        def run(command, *, mutate=False, allow_fail=True):
            assert mutate is True
            assert allow_fail is True
            events.append(":".join(command[1:]))
            if command == ["systemctl", "stop", "docker.socket"]:
                state.docker_socket_active = False
            elif command == [
                "systemctl",
                "mask",
                "--now",
                "docker.socket",
            ]:
                state.docker_socket_active = False
                state.docker_socket_enabled = False
                state.docker_socket_unit_file_state = "masked"
            else:
                raise AssertionError(f"unexpected command: {command}")
            return CommandResult(command, 0)

    monkeypatch.setattr(
        "nvidia_converge.cli._restore_rollback_unit_staged",
        staged,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._prepare_rollback_unit_safely",
        prepared,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._restore_rollback_unit_activity_safely",
        activity,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_active_trusted_gpu_service_identity",
        validate,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.validate_trusted_docker_socket_unit",
        lambda *args, **kwargs: ([], None),
    )

    def quarantine_socket(runner, unit):
        del runner
        assert unit == "docker.socket"
        events.append("mask:docker.socket")
        state.docker_socket_active = False
        state.docker_socket_enabled = False
        state.docker_socket_unit_file_state = "masked"
        return [CommandResult(["systemctl", "mask", "docker.socket"], 0)]

    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_service_for_rollback",
        quarantine_socket,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._post_docker_workload_checkpoint",
        lambda *args, **kwargs: (
            events.append("workload-gate") or (True, deepcopy(state))
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._finalize_rollback_unit_state",
        finalized,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.audit_host",
        lambda runner: events.append("audit") or deepcopy(state),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._quarantine_failed_launcher_commit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid masked-socket target must not quarantine")
        ),
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )

    ok, observed = _commit_rollback_service_activity_impl(
        SimpleNamespace(),
        report,
        snapshot,
        StateRunner(),
        deepcopy(state),
        None,
        operation="rollback",
    )

    assert ok is True
    assert observed.docker_service_active is True
    assert observed.docker_service_unit_file_state == "enabled"
    assert observed.docker_socket_active is False
    assert observed.docker_socket_unit_file_state == "masked"
    assert events.index("prepare:docker.socket") < events.index(
        "prepare:docker.service"
    )
    assert events.index("start:docker.socket") < events.index("start:docker.service")
    assert events.index("start:docker.service") < events.index("workload-gate")
    assert events.index("enable:docker.service") < events.index(
        "mask:docker.socket"
    )


@pytest.mark.parametrize(
    (
        "package_returncode",
        "verification_ok",
        "rollback_returncode",
        "compensation_inventory_complete",
        "requiesce_ok",
        "target_reboot_pending",
    ),
    [
        (0, True, 0, True, True, False),
        (100, True, 0, True, True, False),
        (100, True, 1, True, True, False),
        (0, False, 0, True, True, False),
        (0, False, 1, True, True, False),
        (0, False, 0, False, True, False),
        (0, False, 0, True, False, False),
        (0, False, 0, True, True, True),
    ],
)
def test_install_trusted_service_lifecycle_orders_restore_safely(
    monkeypatch,
    package_returncode,
    verification_ok,
    rollback_returncode,
    compensation_inventory_complete,
    requiesce_ok,
    target_reboot_pending,
):
    from test_planner import _healthy_audit

    events = []
    reports = []
    audit_calls = 0
    guard = _GateGuard(events)
    guard.requiesce_ok = requiesce_ok
    runner = _LifecycleRunner(events, package_returncode)
    baseline = _healthy_audit()
    quarantined_audit = deepcopy(baseline)
    quarantined_audit.docker_service_active = False
    quarantined_audit.docker_service_enabled = False
    quarantined_audit.docker_service_unit_file_state = "masked"
    quarantined_audit.docker_socket_active = False
    quarantined_audit.docker_socket_enabled = False
    quarantined_audit.docker_socket_unit_file_state = "masked"
    quarantined_audit.nvidia_persistenced_active = False
    quarantined_audit.nvidia_persistenced_enabled = False
    quarantined_audit.nvidia_persistenced_unit_file_state = "masked"
    quarantined_audit.fabric_manager_active = False
    quarantined_audit.fabric_manager_enabled = False
    quarantined_audit.fabric_manager_unit_file_state = "masked"
    quarantined_audit.runtime.docker_gpus_usable = None
    reboot_pending_audit = deepcopy(quarantined_audit)
    reboot_pending_audit.module.version = "570.172.08"
    compensation_audit = deepcopy(quarantined_audit)
    compensation_audit.package_inventory_complete = compensation_inventory_complete
    install_target_audit = deepcopy(baseline)
    install_target_audit.fabric_manager_active = False
    install_target_audit.fabric_manager_enabled = False
    install_target_audit.fabric_manager_unit_file_state = "disabled"

    def audited(command_runner):
        nonlocal audit_calls
        del command_runner
        audit_calls += 1
        events.append(f"audit:{audit_calls}")
        if "install.compensation:commit:docker-socket" in events:
            return baseline
        if "rollback" in events:
            return baseline
        if "prepare" in events and target_reboot_pending:
            return reboot_pending_audit
        if "install:commit:docker-socket" in events:
            return install_target_audit
        if any(event.startswith("quarantine:") for event in events):
            return quarantined_audit
        return baseline

    monkeypatch.setattr("nvidia_converge.cli.CommandRunner", lambda **kwargs: runner)
    monkeypatch.setattr("nvidia_converge.cli.audit_host", audited)
    monkeypatch.setattr("nvidia_converge.cli.diagnose", lambda desired, audit: [])
    monkeypatch.setattr(
        "nvidia_converge.cli.build_plan",
        lambda desired, audit, findings: [
            PlanAction(
                "install.packages",
                "Install package.",
                [["apt-get", "install", "-y", "nvidia-container-toolkit"]],
                destructive=True,
            ),
            PlanAction(
                "prepare.module",
                "Prepare module.",
                [],
                destructive=True,
            ),
        ],
    )

    def maintenance(*args, **kwargs):
        del args
        probe = CommandResult(["probe-active-gpu-workloads"], 0)
        if kwargs.get("quiesce_services") is False:
            events.extend(["maintenance:read-only", "probe:read-only"])
            return _MaintenanceGateOutcome(None, [], probe, [], [])
        events.extend(["maintenance:mutating", "quiesce", "probe:mutating"])
        return _MaintenanceGateOutcome(
            guard,
            list(guard.results),
            probe,
            [],
            [],
        )

    monkeypatch.setattr("nvidia_converge.cli._maintenance_gate", maintenance)
    monkeypatch.setattr(
        "nvidia_converge.cli._probe_active_gpu_workloads",
        lambda command_runner, host_audit: (
            events.append("compensation-workload-probe")
            or (CommandResult(["probe-active-gpu-workloads"], 0), [])
        ),
    )
    checkpoint_count = 0

    def checkpointed(*args, **kwargs):
        nonlocal checkpoint_count
        del args, kwargs
        checkpoint_count += 1
        boundary = {
            1: "initial",
            2: "post-package",
            3: "target-pending",
        }.get(checkpoint_count, f"extra-{checkpoint_count}")
        events.append(f"quarantine:{boundary}")
        return True, quarantined_audit, True

    monkeypatch.setattr(
        "nvidia_converge.cli._pre_gpu_mutation_checkpoint",
        checkpointed,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._fresh_gpu_boundary_is_safe",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_install",
        lambda desired, audit, command_runner: [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_snapshot_restore_availability",
        lambda snapshot, command_runner: [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_rollback",
        lambda snapshot, audit, command_runner: [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.resolved_forward_payload_packages",
        lambda *args: [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._bind_forward_package_payloads",
        lambda actions, _snapshot, _audit: actions,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_staged_forward_transaction",
        lambda *args: [],
    )
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=baseline.kernel.running,
        module_version=baseline.module.version,
        commands=[],
        gpu_uuids=list(baseline.gpu_uuids),
        docker_service_active=baseline.docker_service_active,
        docker_service_enabled=baseline.docker_service_enabled,
        docker_service_unit_file_state=(baseline.docker_service_unit_file_state),
        docker_socket_active=baseline.docker_socket_active,
        docker_socket_enabled=baseline.docker_socket_enabled,
        docker_socket_unit_file_state=baseline.docker_socket_unit_file_state,
        nvidia_persistenced_active=baseline.nvidia_persistenced_active,
        nvidia_persistenced_enabled=baseline.nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state=(
            baseline.nvidia_persistenced_unit_file_state
        ),
        fabric_manager_active=baseline.fabric_manager_active,
        fabric_manager_enabled=baseline.fabric_manager_enabled,
        fabric_manager_unit_file_state=baseline.fabric_manager_unit_file_state,
    )

    def created_snapshot(*args, **kwargs):
        del args, kwargs
        events.append("snapshot")
        return snapshot, CommandResult(["persist-rollback-snapshot"], 0)

    monkeypatch.setattr(
        "nvidia_converge.cli._create_snapshot_with_evidence",
        created_snapshot,
    )

    def prepared(*args, **kwargs):
        events.append("prepare")
        if target_reboot_pending:
            result = CommandResult(
                ["modprobe", "-r", "nvidia_uvm", "nvidia"],
                1,
                stderr="modprobe: FATAL: Module nvidia is in use.",
            )
            runner.results.append(result)
            return Verification("module.reload", False, result)
        return Verification("module.load", True)

    monkeypatch.setattr("nvidia_converge.cli.prepare_stack", prepared)

    def compensated(*args, **kwargs):
        events.append("rollback")
        return [CommandResult(["rollback"], rollback_returncode)]

    monkeypatch.setattr("nvidia_converge.cli.apply_rollback", compensated)

    def prepared_compensation(
        report,
        snapshot,
        command_runner,
        service_guard,
        *,
        allow_active_workloads,
    ):
        del snapshot, command_runner, service_guard, allow_active_workloads
        events.append("requiesce")
        if requiesce_ok:
            return compensation_audit
        report.findings.append(
            Finding(
                "gpu-services.requiesce-failed",
                Severity.ERROR,
                "Trusted NVIDIA services could not be safely re-quiesced",
                "service state is unobservable",
            )
        )
        return None

    monkeypatch.setattr(
        "nvidia_converge.cli._prepare_install_compensation",
        prepared_compensation,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.verify_rollback",
        lambda *args, **kwargs: (
            events.append("verify-rollback")
            or [Verification("rollback.packages-restored", True)]
        ),
    )

    def committed_launchers(
        args,
        report,
        target,
        command_runner,
        host_audit,
        service_guard,
        *,
        operation,
    ):
        del args, report, command_runner, host_audit, service_guard
        if operation == "install":
            assert target.fabric_manager_active is False
            assert target.fabric_manager_enabled is False
            assert target.fabric_manager_unit_file_state == "disabled"
            assert target.nvidia_persistenced_unit_file_state == "enabled"
            assert target.docker_service_active is True
            assert target.docker_service_enabled is True
            assert target.docker_service_unit_file_state == "enabled"
            assert target.docker_socket_unit_file_state == "enabled"
            resulting_audit = install_target_audit
        else:
            assert operation == "install.compensation"
            assert target.fabric_manager_unit_file_state == "enabled"
            assert target.nvidia_persistenced_unit_file_state == "enabled"
            assert target.docker_service_unit_file_state == "enabled"
            assert target.docker_socket_unit_file_state == "enabled"
            resulting_audit = baseline
        events.extend(
            [
                f"{operation}:commit:fabric-manager",
                f"{operation}:commit:persistenced",
                f"{operation}:commit:docker-service",
                f"{operation}:docker-workload-gate",
                f"{operation}:commit:docker-socket",
            ]
        )
        return True, resulting_audit

    monkeypatch.setattr(
        "nvidia_converge.cli._commit_rollback_service_activity",
        committed_launchers,
    )

    def verified(*args, **kwargs):
        phase = "docker" if kwargs.get("include_docker") is True else "core"
        events.append(f"verify:{phase}")
        if kwargs.get("include_docker") is True:
            return [
                Verification("docker.service-active", verification_ok),
                Verification("container.gpu", verification_ok),
            ]
        if target_reboot_pending:
            return [
                Verification("module.loaded-version", False),
                Verification("module.provenance", False),
            ]
        return [Verification("stack", verification_ok)]

    monkeypatch.setattr("nvidia_converge.cli.verify_stack", verified)
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, *args, **kwargs: reports.append(report),
    )

    returncode = _execute_command(
        SimpleNamespace(
            command="install",
            allow_disruption=True,
            allow_active_workloads=False,
        ),
        DesiredState(),
        None,
        False,
        True,
        None,
    )

    report = reports[-1]
    assert events.index("maintenance:read-only") < events.index("snapshot")
    assert events.index("snapshot") < events.index("maintenance:mutating")
    assert events.index("maintenance:mutating") < events.index("quarantine:initial")
    assert events.index("quarantine:initial") < events.index("mutate:apt-get")
    if target_reboot_pending:
        assert returncode == 2
        assert events.index("mutate:apt-get") < events.index("quarantine:post-package")
        assert events.index("quarantine:post-package") < events.index("prepare")
        assert events.index("prepare") < events.index("verify:core")
        assert events.index("verify:core") < events.index("quarantine:target-pending")
        assert events.index("quarantine:target-pending") < events.index("requiesce")
        assert events.index("requiesce") < events.index("rollback")
        assert events.index("rollback") < events.index("verify-rollback")
        assert not any(event.startswith("install:commit:") for event in events)
        assert any(
            finding.id == "install.target-reboot-pending-compensation-required"
            for finding in report.findings
        )
    elif package_returncode == 0 and verification_ok:
        assert returncode == 0
        assert events.index("mutate:apt-get") < events.index("quarantine:post-package")
        assert events.index("quarantine:post-package") < events.index("prepare")
        assert events.index("prepare") < events.index("verify:core")
        assert events.index("verify:core") < events.index(
            "install:commit:fabric-manager"
        )
        assert events.index("install:commit:fabric-manager") < events.index(
            "install:commit:persistenced"
        )
        assert events.index("install:commit:persistenced") < events.index(
            "install:commit:docker-service"
        )
        assert events.index("install:commit:docker-service") < events.index(
            "install:docker-workload-gate"
        )
        assert events.index("install:docker-workload-gate") < events.index(
            "install:commit:docker-socket"
        )
        assert events.index("install:commit:docker-socket") < events.index(
            "verify:docker"
        )
        assert not any(
            finding.id == "gpu-services.intentionally-quiesced"
            for finding in report.findings
        )
    elif not requiesce_ok:
        assert returncode == 2
        if package_returncode == 0:
            assert events.index("verify:core") < events.index("requiesce")
        assert "rollback" not in events
        assert any(
            finding.id == "gpu-services.requiesce-failed" for finding in report.findings
        )
    elif not compensation_inventory_complete:
        assert returncode == 2
        assert events.index("verify:core") < events.index("requiesce")
        assert "rollback" not in events
        assert any(
            finding.id == "install.compensation.inventory-incomplete"
            for finding in report.findings
        )
        assert any(
            finding.id == "gpu-services.intentionally-quiesced"
            for finding in report.findings
        )
    elif rollback_returncode == 0:
        assert returncode == 2
        if package_returncode == 0:
            assert events.index("verify:core") < events.index("requiesce")
            assert any(
                check.name == "stack" and not check.ok for check in report.verification
            )
            assert any(
                finding.id == "install.mutation-failed" for finding in report.findings
            )
        else:
            assert "restore" not in events
            assert "prepare" not in events
            assert not any(event.startswith("verify:") for event in events)
        assert events.index("mutate:apt-get") < events.index("rollback")
        assert events.index("rollback") < events.index("verify-rollback")
        assert events.index("verify-rollback") < events.index(
            "install.compensation:commit:fabric-manager"
        )
        assert events.index(
            "install.compensation:commit:docker-service"
        ) < events.index("install.compensation:docker-workload-gate")
        assert events.index("install.compensation:docker-workload-gate") < events.index(
            "install.compensation:commit:docker-socket"
        )
        assert any(
            finding.id == "install.compensation.succeeded"
            for finding in report.findings
        )
        assert not any(
            finding.id == "gpu-services.intentionally-quiesced"
            for finding in report.findings
        )
    else:
        assert returncode == 2
        if package_returncode == 0:
            assert events.index("verify:core") < events.index("requiesce")
            assert any(
                finding.id == "install.mutation-failed" for finding in report.findings
            )
        else:
            assert "prepare" not in events
            assert not any(event.startswith("verify:") for event in events)
        assert "verify-rollback" not in events
        assert "requiesce" in events
        assert any(
            finding.id == "install.compensation.rollback-failed"
            for finding in report.findings
        )
        assert any(
            finding.id == "gpu-services.intentionally-quiesced"
            for finding in report.findings
        )


def test_bad_rollback_snapshot_is_clean_error(capsys, tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    rc = main(["rollback", "--snapshot", str(snapshot)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_unwritable_report_is_clean_error(capsys, tmp_path):
    out = tmp_path / "directory"
    out.mkdir()
    rc = main(["validate", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error: cannot write report")
    assert "Traceback" not in captured.err


def test_install_is_dry_run_without_apply(tmp_path):
    out = tmp_path / "install.json"
    rc = main(["install", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc in {0, 2}
    skipped = [result for result in report["command_results"] if result.get("skipped")]
    if report["command_results"]:
        assert skipped
    else:
        assert rc == 2
        assert report["plan"]
        assert all(
            action["id"].startswith("unsupported.")
            and not action["commands"]
            for action in report["plan"]
        )
    assert all(result.get("reason") == "dry-run" for result in skipped)
    if report["rollback"] is not None:
        assert report["rollback"]["path"] is None
    assert not (tmp_path / "nvidia-converge-rollback.json").exists()


def test_snapshot_is_dry_run_without_apply(tmp_path):
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        out = tmp_path / "snapshot.json"
        rc = main(["snapshot", "--out", str(out)])
    finally:
        os.chdir(cwd)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["rollback"]["path"] is None
    assert not (tmp_path / "nvidia-converge-rollback.json").exists()


def test_install_status_fails_on_command_failure():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[CommandResult(["apt-get", "install"], 100, stderr="failed")],
        verification=[Verification("nvidia-smi", True)],
    )
    assert _install_status(report) == 2


def test_install_compensation_fails_closed_when_exact_preflight_fails(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        gpu_uuids=list(audit.gpu_uuids),
    )
    failed_check = CommandResult(["apt-get", "--simulate", "install"], 100)
    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)

    def rejected(*args, **kwargs):
        raise PackagePreflightError(
            "exact baseline package unavailable",
            package_manager="apt-get",
            packages=["nvidia-open:amd64=580.1"],
            results=[failed_check],
        )

    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_rollback",
        rejected,
    )

    assert _attempt_install_compensation(report, snapshot, object()) is False
    assert [finding.id for finding in report.findings] == [
        "install.compensation.preflight-failed"
    ]
    assert len(report.verification) == 1
    assert report.verification[0].ok is False
    assert report.command_results == []


def test_install_compensation_refuses_incomplete_post_failure_inventory(
    monkeypatch,
):
    from test_planner import _healthy_audit

    audit = _healthy_audit()
    audit.package_inventory_complete = False
    audit.package_inventory_result = CommandResult(
        ["dpkg-query", "-W"],
        0,
        stdout="iU \tnvidia-open\t580.126.16-1\tamd64\n",
    )
    report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        DesiredState(),
    )
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=list(audit.packages),
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        gpu_uuids=list(audit.gpu_uuids),
    )
    monkeypatch.setattr("nvidia_converge.cli.audit_host", lambda runner: audit)
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_rollback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("package rollback preflight requires a complete inventory")
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.apply_rollback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("package rollback requires a complete inventory")
        ),
    )

    assert _attempt_install_compensation(report, snapshot, object()) is False
    assert [finding.id for finding in report.findings] == [
        "install.compensation.inventory-incomplete"
    ]
    assert report.verification == []
    assert report.command_results == []


def test_install_status_passes_when_commands_and_checks_pass():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[CommandResult(["apt-get", "install"], 0)],
        verification=[Verification("nvidia-smi", True)],
    )
    assert _install_status(report) == 0


def test_failed_command_results_are_not_safe_for_post_install_verify():
    assert _commands_succeeded([CommandResult(["apt-get", "install"], 100)]) is False
    assert (
        _commands_succeeded(
            [
                CommandResult(
                    ["apt-get", "install"], None, skipped=True, reason="dry-run"
                )
            ]
        )
        is True
    )


def test_plan_execution_stops_after_first_failed_mutating_command():
    runner = _FakeRunner([100, 0])
    actions = [
        PlanAction(
            "install.packages",
            "Install packages.",
            [["apt-get", "install"], ["systemctl", "restart", "docker"]],
        ),
        PlanAction(
            "lock.apt",
            "Lock packages.",
            [["apt-mark", "hold", "nvidia-driver-580-open"]],
        ),
    ]
    results = _run_plan_actions(actions, runner)
    assert [result.command for result in results] == [["apt-get", "install"]]
    assert results[0].returncode == 100


def test_plan_execution_continues_through_dry_run_skips():
    runner = _FakeRunner([None, None])
    actions = [
        PlanAction(
            "configure.docker-runtime",
            "Configure Docker.",
            [["nvidia-ctk"], ["systemctl", "restart", "docker"]],
        )
    ]
    results = _run_plan_actions(actions, runner)
    assert [result.command for result in results] == [
        ["nvidia-ctk"],
        ["systemctl", "restart", "docker"],
    ]


def test_human_output_includes_failed_command_stderr():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[
            CommandResult(
                ["apt-get", "install"], 100, stderr="package not found\nmore detail"
            )
        ],
    )
    output = render_human("install", report, apply=True)
    assert "- fail: apt-get install" in output
    assert "  package not found" in output
    assert "more detail" not in output


def test_human_output_includes_failed_command_stdout_fallback():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[
            CommandResult(["zypper", "install"], 4, stdout="solver failed")
        ],
    )
    output = render_human("install", report, apply=True)
    assert "- fail: zypper install" in output
    assert "  solver failed" in output


def test_human_output_marks_skipped_verification_as_skip():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        verification=[
            Verification(
                "module.load",
                False,
                CommandResult(
                    ["modprobe", "nvidia"], None, skipped=True, reason="dry-run"
                ),
            )
        ],
    )
    output = render_human("verify", report, apply=False)
    assert "- skip: module.load" in output
    assert "- fail: module.load" not in output


def _run_converged_apply_install(
    monkeypatch,
    *,
    allow_disruption,
    active_workloads,
    allow_active_workloads=False,
):
    from test_planner import _healthy_audit, _stage_policy

    desired = DesiredState()
    audit = _healthy_audit()
    _stage_policy(audit, desired)
    quiesced_audit = deepcopy(audit)
    quiesced_audit.docker_socket_active = False
    quiesced_audit.docker_service_active = False
    quiesced_audit.docker_socket_enabled = False
    quiesced_audit.docker_service_enabled = False
    quiesced_audit.docker_socket_unit_file_state = "masked"
    quiesced_audit.docker_service_unit_file_state = "masked"
    quiesced_audit.nvidia_persistenced_active = False
    quiesced_audit.nvidia_persistenced_enabled = False
    quiesced_audit.nvidia_persistenced_unit_file_state = "masked"
    quiesced_audit.fabric_manager_active = False
    quiesced_audit.fabric_manager_enabled = False
    quiesced_audit.fabric_manager_unit_file_state = "masked"
    quiesced_audit.runtime.docker_gpus_usable = None
    events = []
    reports = []
    audit_count = 0
    docker_quiesced = False
    guard = _GateGuard(events)
    runner = _LifecycleRunner(events, 0)
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        packages=[],
        kernel=audit.kernel.running,
        module_version=audit.module.version,
        commands=[],
        gpu_uuids=list(audit.gpu_uuids),
        docker_service_active=audit.docker_service_active,
        docker_service_enabled=audit.docker_service_enabled,
        docker_service_unit_file_state=(audit.docker_service_unit_file_state),
        docker_socket_active=audit.docker_socket_active,
        docker_socket_enabled=audit.docker_socket_enabled,
        docker_socket_unit_file_state=audit.docker_socket_unit_file_state,
        nvidia_persistenced_active=audit.nvidia_persistenced_active,
        nvidia_persistenced_enabled=audit.nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state=(audit.nvidia_persistenced_unit_file_state),
        fabric_manager_active=audit.fabric_manager_active,
        fabric_manager_enabled=audit.fabric_manager_enabled,
        fabric_manager_unit_file_state=audit.fabric_manager_unit_file_state,
    )

    def audited(command_runner):
        nonlocal audit_count
        del command_runner
        audit_count += 1
        events.append(f"audit:{audit_count}")
        return quiesced_audit if docker_quiesced else audit

    checkpoint_count = 0

    def checkpointed(*args, **kwargs):
        nonlocal checkpoint_count, docker_quiesced
        del args, kwargs
        checkpoint_count += 1
        docker_quiesced = True
        events.append(
            "quarantine:initial" if checkpoint_count == 1 else "quarantine:pre-mutation"
        )
        return True, quiesced_audit, True

    def committed_launchers(
        args,
        report,
        target,
        command_runner,
        host_audit,
        guard,
        *,
        operation,
    ):
        nonlocal docker_quiesced
        del args, report, command_runner, host_audit, guard
        assert target.docker_service_unit_file_state == "enabled"
        assert target.docker_socket_unit_file_state == "enabled"
        assert target.nvidia_persistenced_unit_file_state == "enabled"
        assert target.fabric_manager_unit_file_state == "disabled"
        events.extend(
            [
                f"{operation}:commit:fabric-manager",
                f"{operation}:commit:persistenced",
                f"{operation}:commit:docker-service",
                f"{operation}:docker-workload-gate",
                f"{operation}:commit:docker-socket",
            ]
        )
        docker_quiesced = False
        return True, audit

    def quiesced(command_runner, *, restore_on_failure):
        del command_runner
        assert restore_on_failure is False
        events.append("quiesce")
        return guard

    def probed(command_runner, host_audit):
        del command_runner, host_audit
        events.append("workload-probe")
        return (
            CommandResult(["probe-active-gpu-workloads"], 0),
            list(active_workloads),
        )

    def created_snapshot(*args, **kwargs):
        del args, kwargs
        events.append("snapshot")
        return snapshot, CommandResult(["persist-rollback-snapshot"], 0)

    def preflighted_snapshot(*args, **kwargs):
        del args, kwargs
        events.append("snapshot-preflight")
        return []

    def verified(*args, **kwargs):
        del args, kwargs
        events.append("verify-stack")
        return [
            Verification(
                "container.gpu",
                True,
                CommandResult(["docker", "run", "--gpus", "device=GPU-test"], 0),
            )
        ]

    monkeypatch.setattr("nvidia_converge.cli.CommandRunner", lambda **kwargs: runner)
    monkeypatch.setattr("nvidia_converge.cli.audit_host", audited)
    monkeypatch.setattr(
        "nvidia_converge.cli._pre_gpu_mutation_checkpoint",
        checkpointed,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._fresh_gpu_boundary_is_safe",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._commit_rollback_service_activity",
        committed_launchers,
    )
    monkeypatch.setattr("nvidia_converge.cli.quiesce_trusted_gpu_services", quiesced)
    monkeypatch.setattr("nvidia_converge.cli._probe_active_gpu_workloads", probed)
    monkeypatch.setattr(
        "nvidia_converge.cli._create_snapshot_with_evidence", created_snapshot
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_snapshot_restore_availability",
        preflighted_snapshot,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_rollback",
        lambda *args: [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._bind_forward_package_payloads",
        lambda actions, _snapshot, _audit: actions,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_staged_forward_transaction",
        lambda *args: [],
    )
    monkeypatch.setattr("nvidia_converge.cli.verify_stack", verified)
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, *args, **kwargs: reports.append(report),
    )

    returncode = _execute_command(
        SimpleNamespace(
            command="install",
            allow_disruption=allow_disruption,
            allow_active_workloads=allow_active_workloads,
        ),
        desired,
        None,
        False,
        True,
        None,
    )
    return returncode, reports[-1], events


def _run_applied_verify_or_lock(monkeypatch, *, command):
    from test_planner import _healthy_audit

    desired = DesiredState()
    baseline = _healthy_audit()
    quarantined = deepcopy(baseline)
    quarantined.docker_socket_active = False
    quarantined.docker_service_active = False
    quarantined.nvidia_persistenced_active = False
    quarantined.fabric_manager_active = False
    quarantined.docker_socket_enabled = False
    quarantined.docker_service_enabled = False
    quarantined.nvidia_persistenced_enabled = False
    quarantined.fabric_manager_enabled = False
    quarantined.docker_socket_unit_file_state = "masked"
    quarantined.docker_service_unit_file_state = "masked"
    quarantined.nvidia_persistenced_unit_file_state = "masked"
    quarantined.fabric_manager_unit_file_state = "masked"
    quarantined.runtime.docker_gpus_usable = None

    events = []
    reports = []
    snapshot_kwargs = {}
    audit_count = 0
    target_preflight_count = 0
    policy_applied = False
    launchers_quarantined = False
    runner = _LifecycleRunner(events, 0)
    guard = _GateGuard(events)
    operation_report = Report(
        "1.2",
        "2026-08-02T00:00:00+00:00",
        desired,
    )
    snapshot = RollbackSnapshot(
        path="/var/lib/nvidia-converge/snapshots/test.json",
        schema_version="2.6",
        operation_id=operation_report.operation_id,
        packages=list(baseline.packages),
        kernel=baseline.kernel.running,
        module_version=baseline.module.version,
        commands=[],
        gpu_uuids=list(baseline.gpu_uuids),
        docker_service_active=baseline.docker_service_active,
        docker_service_enabled=baseline.docker_service_enabled,
        docker_service_unit_file_state=(baseline.docker_service_unit_file_state),
        docker_socket_active=baseline.docker_socket_active,
        docker_socket_enabled=baseline.docker_socket_enabled,
        docker_socket_unit_file_state=(baseline.docker_socket_unit_file_state),
        nvidia_persistenced_active=baseline.nvidia_persistenced_active,
        nvidia_persistenced_enabled=baseline.nvidia_persistenced_enabled,
        nvidia_persistenced_unit_file_state=(
            baseline.nvidia_persistenced_unit_file_state
        ),
        fabric_manager_active=baseline.fabric_manager_active,
        fabric_manager_enabled=baseline.fabric_manager_enabled,
        fabric_manager_unit_file_state=(baseline.fabric_manager_unit_file_state),
        integrity_sha256="a" * 64,
    )

    def audited(command_runner):
        nonlocal audit_count
        del command_runner
        audit_count += 1
        events.append(f"audit:{audit_count}")
        return quarantined if launchers_quarantined else baseline

    def planned(*args):
        del args
        plan_number = sum(event.startswith("plan:") for event in events) + 1
        events.append(f"plan:{plan_number}")
        if policy_applied:
            return []
        return [
            PlanAction(
                "lock.apt",
                "Lock packages.",
                [["apt-get", "install", "pin-policy"]],
            )
        ]

    def target_preflight(*args):
        nonlocal target_preflight_count
        del args
        target_preflight_count += 1
        events.append(f"target-preflight:{target_preflight_count}")
        return []

    def created_snapshot(*args, **kwargs):
        del args
        events.append("snapshot")
        snapshot_kwargs.update(kwargs)
        return snapshot, CommandResult(["persist-rollback-snapshot"], 0)

    def maintained(*args, **kwargs):
        del args, kwargs
        events.append("maintenance")
        return _MaintenanceGateOutcome(guard, list(guard.results), None, [], [])

    def checkpointed(*args, **kwargs):
        nonlocal launchers_quarantined
        del args, kwargs
        launchers_quarantined = True
        events.append("fresh-quarantined-audit")
        return True, quarantined, True

    def prepared(*args, **kwargs):
        del args, kwargs
        events.append("mutate:verify")
        return Verification("module.load", True)

    def verified(*args, **kwargs):
        del args
        if kwargs.get("include_docker") is False:
            events.append("verify-core")
            return [Verification("module.loaded-version", True)]
        events.append("verify-services")
        return [
            Verification("docker.service-active", True),
            Verification("container.gpu", True),
        ]

    def run_actions(*args, **kwargs):
        nonlocal policy_applied
        del args, kwargs
        events.append("mutate:lock")
        policy_applied = True
        return [CommandResult(["apt-get", "install", "pin-policy"], 0)]

    def committed(*args, **kwargs):
        nonlocal launchers_quarantined
        operation = kwargs["operation"]
        transaction_guard = args[5]
        transaction_guard.relinquish(
            {
                "nvidia-fabricmanager.service",
                "nvidia-persistenced.service",
            }
        )
        events.append(f"{operation}:commit")
        launchers_quarantined = False
        return True, baseline

    monkeypatch.setattr("nvidia_converge.cli.CommandRunner", lambda **kwargs: runner)
    monkeypatch.setattr("nvidia_converge.cli.audit_host", audited)
    monkeypatch.setattr("nvidia_converge.cli.diagnose", lambda *args: [])
    monkeypatch.setattr("nvidia_converge.cli.lock_actions", planned)
    monkeypatch.setattr("nvidia_converge.cli.preflight_package_lock", target_preflight)
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_snapshot_restore_availability",
        lambda *args: events.append("snapshot-preflight") or [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.preflight_package_rollback",
        lambda *args: events.append("fresh-rollback-preflight") or [],
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._create_snapshot_with_evidence",
        created_snapshot,
    )
    if command == "lock":
        monkeypatch.setattr(
            "nvidia_converge.cli.resolved_forward_payload_packages",
            lambda *args: [],
        )
        monkeypatch.setattr(
            "nvidia_converge.cli._bind_forward_package_payloads",
            lambda actions, _snapshot, _audit: actions,
        )

        def staged_target_preflight(*args):
            del args
            staged_count = sum(
                event.startswith("staged-target-preflight:") for event in events
            ) + 1
            events.append(f"staged-target-preflight:{staged_count}")
            return []

        monkeypatch.setattr(
            "nvidia_converge.cli.preflight_staged_forward_transaction",
            staged_target_preflight,
        )
    monkeypatch.setattr("nvidia_converge.cli._maintenance_gate", maintained)
    monkeypatch.setattr(
        "nvidia_converge.cli._pre_gpu_mutation_checkpoint",
        checkpointed,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.verify_rollback",
        lambda *args, **kwargs: (
            events.append("baseline-check")
            or [Verification("rollback.packages-restored", True)]
        ),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._fresh_gpu_boundary_is_safe",
        lambda *args, **kwargs: events.append("fresh-boundary") or True,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.module_reload_required",
        lambda *args: events.append("rebuild-verify-operands") or False,
    )
    monkeypatch.setattr("nvidia_converge.cli.prepare_stack", prepared)
    monkeypatch.setattr("nvidia_converge.cli.verify_stack", verified)
    monkeypatch.setattr("nvidia_converge.cli._run_plan_actions", run_actions)
    monkeypatch.setattr(
        "nvidia_converge.cli._append_launcher_release_authorization",
        lambda *args, **kwargs: events.append(f"release:{kwargs['release_target']}"),
    )
    monkeypatch.setattr(
        "nvidia_converge.cli._commit_rollback_service_activity",
        committed,
    )
    monkeypatch.setattr(
        "nvidia_converge.cli.emit_report",
        lambda command, report, *args, **kwargs: reports.append(report),
    )

    returncode = _execute_command(
        SimpleNamespace(
            command=command,
            allow_disruption=True,
            allow_active_workloads=False,
        ),
        desired,
        f"/var/lib/nvidia-converge/{command}.json",
        False,
        True,
        operation_report,
    )
    return returncode, reports[-1], events, snapshot_kwargs


class _FakeRunner:
    def __init__(self, returncodes):
        self.returncodes = list(returncodes)
        self.results = []

    def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
        del mutate, allow_fail, input_text
        result = CommandResult(command, self.returncodes.pop(0))
        self.results.append(result)
        return result


class _MissingExecutableRunner:
    @staticmethod
    def exists(name):
        del name
        return False


class _GateGuard:
    def __init__(self, events):
        self.events = events
        self.results = [
            CommandResult(
                ["systemctl", "stop", "nvidia-fabricmanager.service"],
                0,
            )
        ]
        self.error = None
        self.restore_errors = []
        self.requiesce_errors = []
        self._pending = {"nvidia-fabricmanager.service"}
        self._originally_active = set(self._pending)
        self.mutation_started = False
        self.requiesce_ok = True

    @property
    def ok(self):
        return True

    @property
    def quiesced_service_names(self):
        return sorted(self._pending)

    def restore(self, *, units=None):
        targets = self._pending if units is None else self._pending & units
        if targets:
            self.events.append("restore")
        for unit in sorted(targets):
            self.results.append(CommandResult(["systemctl", "start", unit], 0))
            self._pending.remove(unit)
        return True

    def mark_mutation_started(self):
        self.mutation_started = True

    def relinquish(self, units):
        self._pending.difference_update(units)
        self._originally_active.difference_update(units)

    def requiesce(self):
        self.events.append("requiesce")
        if not self.requiesce_ok:
            self.requiesce_errors = ["service state is unobservable"]
            return False
        self._pending.update(self._originally_active)
        return True


class _LifecycleRunner:
    def __init__(self, events, package_returncode):
        self.events = events
        self.package_returncode = package_returncode
        self.results = []

    def run(self, command, *, mutate=False, allow_fail=True, input_text=None):
        del allow_fail, input_text
        if mutate:
            self.events.append(f"mutate:{command[0]}")
        returncode = self.package_returncode if command[0] == "apt-get" else 0
        result = CommandResult(command, returncode)
        self.results.append(result)
        return result


def _project_version():
    in_project = False
    for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.strip() == "[project]":
            in_project = True
            continue
        if in_project and line.startswith("["):
            break
        if in_project and line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("pyproject.toml is missing [project] version")
