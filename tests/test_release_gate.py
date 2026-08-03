import copy
import hashlib
import hmac
import io
import json
import os
import stat
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

import scripts.export_integration_attestation as attestation_export
from nvidia_converge.desired import container_cuda_full_version, load_desired
from nvidia_converge.dnf_module_transaction import (
    _combined_state_digest,
    _proof_preflight_sha256,
    dnf_module_enable_command,
)
from nvidia_converge.dnf_transaction import dnf_local_transaction_command
from nvidia_converge.models import (
    PackageInfo,
    PackagePayload,
    PackagePayloadBundle,
)
from nvidia_converge.package_payloads import (
    forward_package_command,
    local_payload_paths,
    payload_bundle_directory,
)
from nvidia_converge.rollback import _rollback_commands
from nvidia_converge.verify import (
    _CUDA_DRIVER_PROBE_SCRIPT,
    _CUDA_DRIVER_PROBE_SHA256,
)
from scripts.check_release_evidence import (
    REQUIRED_SCENARIOS,
    ArtifactEvidenceError,
    _artifact_evidence_members,
    _expected_applied_package_commands,
    _github_bytes,
    _strict_json_document,
    _verify_policy_snapshot_delta,
    _verify_report_journal,
    _verify_rollback_audit,
    _verify_scenario_report,
    _verify_snapshot_command_contract,
    check_commit_provenance,
    check_evidence,
    check_qualification_wheel_binding,
    main,
    verify_github_evidence,
)
from scripts.export_integration_attestation import (
    AttestationExportError,
    build_attestation,
)

SCENARIO_REPORTS = {
    "doctor": "reports/doctor.json",
    "doctor_missing_headers": "reports/doctor-missing-headers.json",
    "doctor_module_unloaded": "reports/doctor-module-unloaded.json",
    "doctor_driver_mismatch": "reports/doctor-driver-mismatch.json",
    "doctor_runtime_missing": "reports/doctor-runtime-missing.json",
    "doctor_fabric_manager_inactive": "reports/doctor-fabric-manager-inactive.json",
    "plan": "reports/plan.json",
    "install_apply": "reports/install.json",
    "verify_apply": "reports/verify.json",
    "lock_apply": "reports/lock.json",
    "snapshot": "reports/snapshot.json",
    "rollback_apply": "reports/rollback.json",
    "rollback_state_verify": "reports/rollback.json",
    "policy_rollback_apply": "reports/policy-rollback.json",
    "policy_rollback_state_verify": "reports/policy-rollback.json",
    "report_schema": "reports/plan.json",
}
FAULT_RESTORATION_REPORTS = {
    "doctor_missing_headers": "restoration/doctor-missing-headers.json",
    "doctor_module_unloaded": "restoration/doctor-module-unloaded.json",
    "doctor_driver_mismatch": "restoration/doctor-driver-mismatch.json",
    "doctor_runtime_missing": "restoration/doctor-runtime-missing.json",
    "doctor_fabric_manager_inactive": "restoration/doctor-fabric-manager-inactive.json",
}
TOOLKIT_PACKAGE = "nvidia-container-toolkit"
QUALIFICATION_WHEEL_NAME = "nvidia_converge-0.1.0-py3-none-any.whl"
QUALIFICATION_WHEEL_SHA256 = "f" * 64
TEST_PSEUDONYM_KEY = b"\xa5" * 32


def _expected_keyed_attestation(
    key: bytes, purpose: str, *parts: str
) -> str:
    mac = hmac.new(key, digestmod=hashlib.sha256)
    for part in (purpose, *parts):
        encoded = part.encode("utf-8")
        mac.update(len(encoded).to_bytes(8, "big"))
        mac.update(encoded)
    return f"attested:{mac.hexdigest()}"
TOOLKIT_PACKAGE_CLOSURE = (
    TOOLKIT_PACKAGE,
    "nvidia-container-toolkit-base",
    "libnvidia-container-tools",
    "libnvidia-container1",
)


def test_example_evidence_cannot_promote_a_release():
    evidence = json.loads(Path("integrations/results.example.json").read_text(encoding="utf-8"))
    errors = check_evidence(evidence, release="v0.1.0")
    assert errors
    assert any("overall_status" in error for error in errors)


def test_complete_multi_distribution_evidence_passes():
    evidence = _passing_evidence()
    assert check_evidence(evidence, release="v0.1.0") == []


def test_old_gpu_workflow_history_cannot_promote_a_release():
    evidence = _passing_evidence()
    evidence["runs"][0]["workflow_path"] = ".github/workflows/gpu-integration.yml"

    errors = check_evidence(evidence, release="v0.1.0")

    assert any("trusted GPU workflow" in error for error in errors)


def test_qualified_platforms_must_exactly_match_observed_applied_tuples():
    evidence = _passing_evidence()
    evidence["qualified_platforms"].append(
        {
            "os_id": "ubuntu",
            "os_version": "22.04",
            "package_manager": "apt-get",
        }
    )

    errors = check_evidence(evidence, release="v0.1.0")

    assert any("canonical exact tuple set" in error for error in errors)


def test_recipe_path_coverage_is_derived_without_family_support_claims():
    evidence = _passing_evidence()
    evidence["recipe_path_coverage"]["dnf"] = False

    errors = check_evidence(evidence, release="v0.1.0")

    assert any("recipe-path coverage" in error for error in errors)
    assert not {
        "ubuntu_lts",
        "rhel_family",
        "suse_family",
    }.intersection(evidence["required_coverage"])


def test_each_run_must_bind_the_common_qualification_wheel():
    evidence = _passing_evidence()
    evidence["runs"][0]["qualification_wheel"]["sha256"] = "e" * 64

    errors = check_evidence(evidence, release="v0.1.0")

    assert any("common tested wheel" in error for error in errors)


def test_release_wheel_bytes_must_match_qualification_binding(tmp_path):
    evidence = _passing_evidence()
    wheel = tmp_path / QUALIFICATION_WHEEL_NAME
    wheel.write_bytes(b"qualified-wheel-bytes")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    evidence["qualification_wheel"]["sha256"] = digest
    for run in evidence["runs"]:
        run["qualification_wheel"]["sha256"] = digest

    assert check_qualification_wheel_binding(evidence, wheel) == []

    wheel.write_bytes(b"different-release-wheel")
    errors = check_qualification_wheel_binding(evidence, wheel)
    assert any("does not match" in error for error in errors)


def test_evidence_requires_full_tested_commit():
    evidence = _passing_evidence()
    evidence["commit"] = "abc1234"
    errors = check_evidence(evidence, release="v0.1.0")
    assert any("full lowercase SHA" in error for error in errors)


def test_each_apply_run_must_pass_every_required_scenario():
    evidence = _passing_evidence()
    failed = next(
        scenario
        for scenario in evidence["runs"][0]["scenarios"]
        if scenario["name"] == "rollback_apply"
    )
    failed["status"] = "failed"
    errors = check_evidence(evidence, release="v0.1.0")
    assert any("did not pass required scenarios" in error for error in errors)


def test_release_evidence_requires_canonical_sanitized_report_paths():
    evidence = _passing_evidence()
    scenario = evidence["runs"][0]["scenarios"][0]
    scenario["report"] = "reports/GPU-raw-identity-1234567890.json"

    errors = check_evidence(evidence, release="v0.1.0")

    assert any("canonical sanitized report path" in error for error in errors)


def test_fabric_manager_fault_is_profile_applicable_only():
    evidence = _passing_evidence()
    non_fm = evidence["runs"][0]
    non_fm_fault = next(
        scenario
        for scenario in non_fm["scenarios"]
        if scenario["name"] == "doctor_fabric_manager_inactive"
    )
    non_fm_fault.update(
        status="passed",
        report=SCENARIO_REPORTS["doctor_fabric_manager_inactive"],
    )
    errors = check_evidence(evidence, release="v0.1.0")
    assert any("must mark the Fabric Manager fault scenario not_applicable" in error for error in errors)

    evidence = _passing_evidence()
    fm_run = next(run for run in evidence["runs"] if run["matrix_id"] == "fabric-manager")
    fm_fault = next(
        scenario
        for scenario in fm_run["scenarios"]
        if scenario["name"] == "doctor_fabric_manager_inactive"
    )
    fm_fault.update(status="not_applicable", report=None)
    errors = check_evidence(evidence, release="v0.1.0")
    assert any("doctor_fabric_manager_inactive" in error for error in errors)


def test_matrix_allows_truthfully_recorded_overlapping_capability_labels():
    evidence = _passing_evidence()
    evidence["runs"][0]["runner_labels"].append("secure-boot")
    evidence["runs"][0]["capabilities"]["secure_boot"] = True
    errors = check_evidence(evidence, release="v0.1.0")
    assert errors == []


def test_non_apply_run_cannot_claim_mutating_scenarios():
    evidence = _passing_evidence()
    evidence["runs"][0]["apply"] = False
    errors = check_evidence(evidence, release="v0.1.0")
    assert any("claims mutating scenarios" in error for error in errors)


def test_os_package_manager_mapping_and_workflow_identity_are_bound():
    evidence = _passing_evidence()
    evidence["runs"][0]["package_manager"] = "zypper"
    evidence["runs"][0]["workflow_run"] = "https://example.invalid/run/1"
    errors = check_evidence(evidence, release="v0.1.0")
    assert any("OS/package-manager mapping" in error for error in errors)
    assert any("workflow_run URL" in error for error in errors)


def test_github_metadata_binds_runs_and_artifact_digests(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    _mock_github(monkeypatch, evidence, archives)
    assert verify_github_evidence(evidence, token="token") == []

    evidence["runs"][0]["artifacts"][0]["sha256"] = "b" * 64
    errors = verify_github_evidence(evidence, token="token")
    assert any("sha256" in error for error in errors)


def test_retained_attestation_binds_the_job_qualification_wheel(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first = evidence["runs"][0]
    artifact_id = first["artifacts"][0]["id"]
    members = _read_zip_members(archives[artifact_id])
    manifest = json.loads(members["attestation.json"])
    manifest["qualification_wheel"]["sha256"] = "e" * 64
    members["attestation.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    archives[artifact_id] = _zip_members(members)
    first["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[artifact_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")

    assert any("attestation context mismatch" in error for error in errors)
    assert any("qualification_wheel" in error for error in errors)


def test_github_verifier_returns_artifacts_only_after_complete_validation(
    monkeypatch,
):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    _mock_github(monkeypatch, evidence, archives)
    verified = {999: b"stale"}

    assert (
        verify_github_evidence(
            evidence,
            token="token",
            verified_artifacts=verified,
        )
        == []
    )
    assert verified == archives

    evidence["runs"][0]["artifacts"][0]["sha256"] = "b" * 64
    assert verify_github_evidence(
        evidence,
        token="token",
        verified_artifacts=verified,
    )
    assert verified == {}


def test_github_verifier_rejects_untrusted_api_origin_before_network():
    verified = {1: b"stale"}

    errors = verify_github_evidence(
        _passing_evidence(),
        token="must-not-be-disclosed",
        api_url="https://attacker.invalid",
        verified_artifacts=verified,
    )

    assert any("exactly https://api.github.com" in error for error in errors)
    assert verified == {}


def test_artifact_download_rejects_untrusted_origin_before_network():
    with pytest.raises(RuntimeError, match="untrusted artifact URL"):
        _github_bytes(
            "https://attacker.invalid/steal",
            "must-not-be-disclosed",
        )


def test_github_artifact_download_digest_is_verified(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_id = evidence["runs"][0]["artifacts"][0]["id"]
    archives[first_id] += b"tampered"
    _mock_github(monkeypatch, evidence, archives, preserve_metadata_digest=True)

    errors = verify_github_evidence(evidence, token="token")
    assert any("downloaded ZIP digest mismatch" in error for error in errors)


def test_github_artifact_rejects_unsafe_zip_paths(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    archives[first_id] = _append_zip_member(archives[first_id], "../escape", b"bad")
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("unsafe path" in error for error in errors)


@pytest.mark.parametrize("metadata", ["archive-comment", "entry-comment", "extra"])
def test_artifact_reader_rejects_uninventoried_zip_metadata(metadata):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        entry = zipfile.ZipInfo("attestation.json")
        if metadata == "archive-comment":
            archive.comment = b"raw host data"
        elif metadata == "entry-comment":
            entry.comment = b"raw host data"
        else:
            entry.extra = b"\x01\x00\x00\x00"
        archive.writestr(entry, b"{}")

    with pytest.raises(ArtifactEvidenceError):
        _artifact_evidence_members(target.getvalue())


def test_artifact_reader_rejects_uninventoried_directory_names():
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("GPU-raw-identity-1234567890/", b"")
        archive.writestr("attestation.json", b"{}")

    with pytest.raises(ArtifactEvidenceError, match="unexpected directory"):
        _artifact_evidence_members(target.getvalue())


def test_github_artifact_requires_referenced_reports_and_journals(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    archives[first_id] = _without_zip_members(
        archives[first_id], {"reports/install.json", "reports/rollback.journal.jsonl"}
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("missing scenario reports" in error and "install.json" in error for error in errors)
    assert any("no retained command journal" in error and "rollback.json" in error for error in errors)


def test_github_artifact_requires_same_host_fault_restoration(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    archives[first_id] = _without_zip_members(
        archives[first_id], {"restoration/doctor-missing-headers.json"}
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("missing fault-restoration reports" in error for error in errors)


def test_github_artifact_rejects_non_attestation_members(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    archives[first_id] = _append_zip_member(
        archives[first_id], "pre/nvidia-smi-q.txt", b"GPU-SECRET"
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("non-attestation member" in error for error in errors)


def test_github_artifact_rejects_extra_sanitized_journal_fields(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def inject_secret(payload):
        entries = [json.loads(line) for line in payload.splitlines()]
        entries[0]["secret"] = "must-not-survive"
        return b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode() for entry in entries
        )

    archives[first_id] = _rewrite_attested_member(
        archives[first_id],
        "reports/install.journal.jsonl",
        inject_secret,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("does not match the sanitized event shape" in error for error in errors)


def test_github_artifact_rejects_unredacted_finding_remediation(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def expose_private_path(report):
        report["findings"][0]["remediation"] = (
            "/var/lib/nvidia-converge/snapshots/private.json"
        )
        return report

    archives[first_id] = _rewrite_attested_json_member(
        archives[first_id],
        "reports/doctor.json",
        expose_private_path,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("exposes finding evidence" in error for error in errors)


def test_attestation_export_rejects_extra_source_journal_fields():
    evidence = _passing_evidence()
    run = evidence["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    authority = report["rollback"]
    entries = [
        {
            "event": "operation-started",
            "operation_id": report["operation_id"],
            "timestamp": report["started_at"],
            "private_injection": "must-not-be-laundered",
        },
        {
            "event": "rollback-snapshot-persisted",
            "operation_id": report["operation_id"],
            "timestamp": report["started_at"],
            "snapshot_path": authority["path"],
            "snapshot_integrity_sha256": authority["integrity_sha256"],
            "snapshot_operation_id": authority["operation_id"],
            "snapshot_host_id": authority["host_id"],
        },
    ]

    with pytest.raises(
        AttestationExportError,
        match="exact source journal event shape",
    ):
        build_attestation(
            {
                "reports/install.json": (json.dumps(report) + "\n").encode(),
                "reports/install.journal.jsonl": b"".join(
                    (json.dumps(entry) + "\n").encode()
                    for entry in entries
                ),
            },
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_attestation_boundaries_reject_non_finite_json_numbers(
    constant: str,
) -> None:
    payload = f'{{"duration_seconds":{constant}}}'.encode()

    with pytest.raises(AttestationExportError, match="strict UTF-8 JSON"):
        attestation_export._strict_json(payload, "source report")
    with pytest.raises(ArtifactEvidenceError, match="invalid JSON"):
        _strict_json_document(payload, "sanitized report")


def test_attestation_export_rejects_unknown_sensitive_report_fields():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    report["audit"]["module"]["private_injection"] = "must-not-be-uploaded"

    with pytest.raises(
        AttestationExportError,
        match=r"source reports/install\.json.*unexpected field.*private_injection",
    ):
        build_attestation(
            {"reports/install.json": (json.dumps(report) + "\n").encode()},
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def test_attestation_export_rejects_unknown_sensitive_snapshot_fields():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    snapshot["private_injection"] = "must-not-be-uploaded"
    _with_snapshot_integrity(snapshot)

    with pytest.raises(
        AttestationExportError,
        match=(
            r"source pre/rollback-snapshot\.json.*unexpected field.*"
            r"private_injection"
        ),
    ):
        build_attestation(
            {
                "pre/rollback-snapshot.json": (
                    json.dumps(snapshot) + "\n"
                ).encode()
            },
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def test_attestation_export_validates_sanitized_report_before_return(monkeypatch):
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    original_sanitizer = attestation_export._sanitize_report

    def inject_sensitive_field(*args, **kwargs):
        sanitized = original_sanitizer(*args, **kwargs)
        sanitized["private_injection"] = "must-not-be-uploaded"
        return sanitized

    monkeypatch.setattr(
        attestation_export,
        "_sanitize_report",
        inject_sensitive_field,
    )

    with pytest.raises(
        AttestationExportError,
        match=r"sanitized reports/install\.json.*unexpected field.*private_injection",
    ):
        build_attestation(
            {"reports/install.json": (json.dumps(report) + "\n").encode()},
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def test_retained_journal_binds_snapshot_and_launcher_release_to_report():
    evidence = _passing_evidence()
    first_run = evidence["runs"][0]
    archive = _archives_for(evidence)[first_run["artifacts"][0]["id"]]
    members = _read_zip_members(archive)
    report = json.loads(members["reports/install.json"])
    journal_path = "reports/install.journal.jsonl"
    entries = [
        json.loads(line) for line in members[journal_path].splitlines()
    ]
    replacements = {
        "snapshot_path": "attested:" + "b" * 64,
        "snapshot_integrity_sha256": "b" * 64,
        "snapshot_operation_id": "b" * 32,
        "snapshot_host_id": "attested:" + "b" * 64,
    }

    for event_name in (
        "rollback-snapshot-persisted",
        "launcher-release-authorized",
    ):
        for field, replacement in replacements.items():
            tampered = copy.deepcopy(entries)
            event = next(
                entry for entry in tampered if entry["event"] == event_name
            )
            event[field] = replacement
            payload = b"".join(
                (json.dumps(entry, sort_keys=True) + "\n").encode()
                for entry in tampered
            )

            errors = _verify_report_journal(
                str(first_run["id"]),
                journal_path,
                payload,
                report,
                require_mutation=True,
                required_mutation_commands=None,
            )

            assert any(
                "does not match report.rollback" in error
                or "invalid launcher release boundary" in error
                for error in errors
            ), (event_name, field, errors)

    wrong_release = copy.deepcopy(entries)
    release = next(
        entry
        for entry in wrong_release
        if entry["event"] == "launcher-release-authorized"
    )
    release["release_target"] = "operation-target"
    errors = _verify_report_journal(
        str(first_run["id"]),
        journal_path,
        b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode()
            for entry in wrong_release
        ),
        report,
        require_mutation=True,
        required_mutation_commands=None,
    )
    assert any("wrong launcher release authorization sequence" in error for error in errors)

    compensated_success = copy.deepcopy(entries)
    rollback_release = copy.deepcopy(release)
    rollback_release["release_target"] = "rollback-baseline"
    compensated_success.insert(-1, rollback_release)
    errors = _verify_report_journal(
        str(first_run["id"]),
        journal_path,
        b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode()
            for entry in compensated_success
        ),
        report,
        require_mutation=True,
        required_mutation_commands=None,
    )
    assert any(
        "wrong launcher release authorization sequence" in error
        for error in errors
    )

    recovered_report = copy.deepcopy(report)
    recovered_report.update(
        {"exit_code": 2, "incomplete": True, "outcome": "failed"}
    )
    recovered_entries = copy.deepcopy(entries)
    recovered_entries[-1].update(
        {"exit_code": 2, "incomplete": True, "outcome": "failed"}
    )
    recovered_entries.append(
        {
            "event": "operation-recovered",
            "operation_id": report["operation_id"],
            "timestamp": report["completed_at"],
            "recovery_operation_id": "b" * 32,
            "snapshot_path": report["rollback"]["path"],
            "snapshot_integrity_sha256": report["rollback"][
                "integrity_sha256"
            ],
            "snapshot_operation_id": report["rollback"]["operation_id"],
            "snapshot_host_id": report["rollback"]["host_id"],
        }
    )
    assert _verify_report_journal(
        str(first_run["id"]),
        journal_path,
        b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode()
            for entry in recovered_entries
        ),
        recovered_report,
        require_mutation=True,
        required_mutation_commands=None,
    ) == []

    nonterminal_recovery = copy.deepcopy(recovered_entries)
    nonterminal_recovery[-2], nonterminal_recovery[-1] = (
        nonterminal_recovery[-1],
        nonterminal_recovery[-2],
    )
    errors = _verify_report_journal(
        str(first_run["id"]),
        journal_path,
        b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode()
            for entry in nonterminal_recovery
        ),
        recovered_report,
        require_mutation=True,
        required_mutation_commands=None,
    )
    assert any("nonterminal recovery event" in error for error in errors)


def test_attestation_export_rejects_mismatched_journal_snapshot_authority():
    evidence = _passing_evidence()
    run = evidence["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    report_snapshot = report["rollback"]
    entries = [
        {
            "event": "operation-started",
            "operation_id": report["operation_id"],
            "timestamp": report["started_at"],
        },
        {
            "event": "rollback-snapshot-persisted",
            "operation_id": report["operation_id"],
            "timestamp": report["started_at"],
            "snapshot_path": report_snapshot["path"],
            "snapshot_integrity_sha256": "b" * 64,
            "snapshot_operation_id": report_snapshot["operation_id"],
            "snapshot_host_id": report_snapshot["host_id"],
        },
    ]

    with pytest.raises(
        AttestationExportError,
        match="unretained or mismatched rollback snapshot authority",
    ):
        build_attestation(
            {
                "reports/install.json": (
                    json.dumps(report) + "\n"
                ).encode(),
                "reports/install.journal.jsonl": b"".join(
                    (json.dumps(entry) + "\n").encode()
                    for entry in entries
                ),
            },
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def test_attestation_export_preserves_pseudonymous_recovery_authority():
    evidence = _passing_evidence()
    run = evidence["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    report.update({"exit_code": 2, "incomplete": True, "outcome": "failed"})
    authority = report["rollback"]
    mutation = report["command_results"][0]["command"]
    common = {
        "operation_id": report["operation_id"],
        "timestamp": report["started_at"],
    }
    binding = {
        "snapshot_path": authority["path"],
        "snapshot_integrity_sha256": authority["integrity_sha256"],
        "snapshot_operation_id": authority["operation_id"],
        "snapshot_host_id": authority["host_id"],
    }
    entries = [
        {"event": "operation-started", **common},
        {"event": "rollback-snapshot-persisted", **common, **binding},
        {
            "event": "command-started",
            **common,
            "command": mutation,
            "mutating": True,
        },
        {
            "event": "command-finished",
            **common,
            "command": mutation,
            "mutating": True,
            "returncode": 0,
            "skipped": False,
            "reason": None,
        },
        {
            "event": "operation-completed",
            **common,
            "exit_code": 2,
            "incomplete": True,
            "outcome": "failed",
        },
        {
            "event": "operation-recovered",
            **common,
            "recovery_operation_id": "b" * 32,
            **binding,
        },
    ]
    exported = build_attestation(
        {
            "reports/install.json": (json.dumps(report) + "\n").encode(),
            "reports/install.journal.jsonl": b"".join(
                (json.dumps(entry) + "\n").encode() for entry in entries
            ),
        },
        matrix_id=run["matrix_id"],
        workflow_run_id=run["workflow_run_id"],
        workflow_run_attempt=run["workflow_run_attempt"],
        qualification_wheel_name=run["qualification_wheel"]["name"],
        qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
    )
    sanitized_report = json.loads(exported["reports/install.json"])
    sanitized_entries = [
        json.loads(line)
        for line in exported["reports/install.journal.jsonl"].splitlines()
    ]
    recovered = sanitized_entries[-1]
    sanitized_authority = sanitized_report["rollback"]

    assert recovered["snapshot_path"] == sanitized_authority["path"]
    assert recovered["snapshot_integrity_sha256"] == sanitized_authority[
        "integrity_sha256"
    ]
    assert recovered["snapshot_operation_id"] == sanitized_authority[
        "operation_id"
    ]
    assert recovered["snapshot_host_id"] == sanitized_authority["host_id"]
    assert authority["path"].encode() not in exported[
        "reports/install.journal.jsonl"
    ]
    attested_payload_prefix = (
        "/attested/"
        + sanitized_authority["path"].removeprefix("attested:")
        + "/"
        + sanitized_authority["package_payloads"]["directory"]
        + "/"
    )
    assert any(
        part.startswith(attested_payload_prefix)
        for part in sanitized_report["command_results"][0]["command"]
    )
    assert _verify_report_journal(
        str(run["id"]),
        "reports/install.journal.jsonl",
        exported["reports/install.journal.jsonl"],
        sanitized_report,
        require_mutation=True,
        required_mutation_commands=None,
    ) == []


def test_attestation_export_rejects_unbound_local_payload_operand():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    command = report["command_results"][0]["command"]
    payload_index = next(
        index for index, part in enumerate(command) if ".payloads/" in part
    )
    payload_root = Path(command[payload_index]).parent
    command[payload_index] = str(payload_root / ("0" * 64 + ".deb"))

    with pytest.raises(
        AttestationExportError,
        match="unbound package payload path",
    ):
        build_attestation(
            {
                "reports/install.json": (
                    json.dumps(report) + "\n"
                ).encode(),
            },
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
            pseudonym_key=TEST_PSEUDONYM_KEY,
        )


def test_github_artifact_binds_install_pre_mutation_snapshot(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def alter_install_baseline(report):
        snapshot = report["rollback"]
        snapshot["module_signed"] = False
        unsigned = dict(snapshot)
        unsigned.pop("integrity_sha256")
        canonical = json.dumps(
            unsigned,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        snapshot["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()
        return report

    archives[first_id] = _rewrite_attested_json_member(
        archives[first_id],
        "reports/install.json",
        alter_install_baseline,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("install pre-mutation baseline" in error for error in errors)


def test_github_artifact_binds_fault_and_rollback_evidence(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    archives[first_id] = _rewrite_json_members(
        archives[first_id],
        {
            "reports/doctor-missing-headers.json": lambda report: {
                **report,
                "findings": [],
            },
            "reports/rollback.json": lambda report: {
                **report,
                "verification": [
                    {**check, "ok": False}
                    if check["name"] == "rollback.kernel"
                    else check
                    for check in report["verification"]
                ],
            },
        },
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("did not record its expected finding" in error for error in errors)
    assert any("failed checks: rollback.kernel" in error for error in errors)


def test_github_artifact_verifies_retained_snapshot_checksum(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    members = _read_zip_members(archives[first_id])
    members["pre/rollback-snapshot.sha256"] = (
        f"{'0' * 64}  artifacts/pre/rollback-snapshot.json\n".encode()
    )
    archives[first_id] = _zip_members(members)
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("snapshot checksum is invalid" in error for error in errors)


def test_github_artifact_requires_pre_policy_snapshot_and_rollback_report(
    monkeypatch,
):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]
    archives[first_id] = _without_zip_members(
        archives[first_id],
        {
            "pre/policy-rollback-snapshot.json",
            "pre/policy-rollback-snapshot.sha256",
            "reports/policy-rollback.json",
            "reports/policy-rollback.journal.jsonl",
        },
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("policy-rollback.json" in error for error in errors)
    assert any(
        "did not retain 'policy-rollback' snapshot" in error for error in errors
    )


def test_two_baseline_delta_compares_full_multiarch_package_records(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def add_shadowed_multiarch_package(snapshot):
        extra = copy.deepcopy(
            next(
                package
                for package in snapshot["packages"]
                if package["name"] == "docker-ce"
            )
        )
        extra["architecture"] = "arm64"
        extra["version"] = "27.5.1-1"
        snapshot["packages"].insert(0, extra)
        return _with_snapshot_integrity(snapshot)

    archive = _rewrite_attested_snapshot(
        archives[first_id],
        "pre/policy-rollback-snapshot.json",
        add_shadowed_multiarch_package,
    )
    for report_path in (
        "reports/lock.json",
        "reports/policy-rollback.json",
    ):
        include_audit = report_path == "reports/policy-rollback.json"

        def alter_report(report, *, include_audit=include_audit):
            report["rollback"] = add_shadowed_multiarch_package(
                report["rollback"]
            )
            if include_audit:
                extra = copy.deepcopy(
                    next(
                        package
                        for package in report["audit"]["packages"]
                        if package["name"] == "docker-ce"
                    )
                )
                extra["architecture"] = "arm64"
                extra["version"] = "27.5.1-1"
                report["audit"]["packages"].insert(0, extra)
            return report

        archive = _rewrite_attested_json_member(
            archive,
            report_path,
            alter_report,
        )
    archives[first_id] = archive
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any(
        "APT policy staging changed package records beyond" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "field",
    [
        "docker_socket_active",
        "docker_socket_enabled",
        "docker_socket_unit_file_state",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
        "nvidia_persistenced_unit_file_state",
        "docker_service_unit_file_state",
        "fabric_manager_unit_file_state",
    ],
)
def test_two_baseline_delta_rejects_transactional_launcher_drift(field):
    run = _passing_evidence()["runs"][0]
    pre_policy = _snapshot_for_run(run, policy_staged=False)
    post_policy = _snapshot_for_run(run, policy_staged=True)
    post_policy[field] = not post_policy[field]
    lock_report = _report_for_scenario(
        run,
        "lock_apply",
        "reports/lock.json",
        post_policy,
        pre_policy,
    )
    plan_report = _report_for_scenario(
        run,
        "plan",
        "reports/plan.json",
        post_policy,
        pre_policy,
    )

    errors = _verify_policy_snapshot_delta(
        run,
        pre_policy,
        post_policy,
        lock_report,
        plan_report,
        asdict(load_desired(run["desired"])),
    )

    assert any(
        "package-policy staging changed non-policy baseline fields" in error
        and field in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("delete", "absent-to-exact-file policy staging"),
        ("path", "architecture-matched fail-safe target"),
        ("mode", "absent-to-exact-file policy staging"),
        ("content", "absent-to-exact-file policy staging"),
    ],
)
def test_dnf_policy_delta_rejects_failsafe_snapshot_tamper(
    tamper,
    expected_error,
):
    run = next(
        item
        for item in _passing_evidence()["runs"]
        if item["package_manager"] == "dnf"
    )
    pre_policy = _snapshot_for_run(run, policy_staged=False)
    post_policy = _snapshot_for_run(run, policy_staged=True)
    expected_path = (
        "/var/lib/dnf/modulefailsafe/"
        "nvidia-driver:580-open:x86_64.yaml"
    )
    post_failsafe = next(
        item
        for item in post_policy["managed_files"]
        if item["path"] == expected_path
    )
    if tamper == "delete":
        post_policy["managed_files"].remove(post_failsafe)
    elif tamper == "path":
        post_failsafe["path"] = (
            "/var/lib/dnf/modulefailsafe/"
            "nvidia-driver:580-open:aarch64.yaml"
        )
    elif tamper == "mode":
        post_failsafe["mode"] = 0o600
    else:
        assert tamper == "content"
        post_failsafe["content_base64"] = None
    lock_report = _report_for_scenario(
        run,
        "lock_apply",
        "reports/lock.json",
        post_policy,
        pre_policy,
    )
    plan_report = _report_for_scenario(
        run,
        "plan",
        "reports/plan.json",
        post_policy,
        pre_policy,
    )

    errors = _verify_policy_snapshot_delta(
        run,
        pre_policy,
        post_policy,
        lock_report,
        plan_report,
        asdict(load_desired(run["desired"])),
    )

    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize(
    "check_name",
    [
        "rollback.docker-socket-active",
        "rollback.docker-socket-enabled",
        "rollback.docker-socket-unit-file-state",
        "rollback.nvidia-persistenced-active",
        "rollback.nvidia-persistenced-enabled",
        "rollback.nvidia-persistenced-unit-file-state",
        "rollback.docker-service-unit-file-state",
        "rollback.fabric-manager-unit-file-state",
    ],
)
def test_rollback_evidence_requires_transactional_launcher_checks(check_name):
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "rollback_state_verify",
        "reports/rollback.json",
        snapshot,
    )
    report["verification"] = [
        check for check in report["verification"] if check["name"] != check_name
    ]

    errors = _verify_scenario_report(
        run,
        "rollback_state_verify",
        "reports/rollback.json",
        report,
    )

    assert any("is missing checks" in error and check_name in error for error in errors)


@pytest.mark.parametrize(
    "field",
    [
        "docker_socket_active",
        "docker_socket_enabled",
        "docker_socket_unit_file_state",
        "nvidia_persistenced_active",
        "nvidia_persistenced_enabled",
        "nvidia_persistenced_unit_file_state",
        "docker_service_unit_file_state",
        "fabric_manager_unit_file_state",
    ],
)
def test_rollback_evidence_rejects_transactional_launcher_audit_drift(field):
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "rollback_state_verify",
        "reports/rollback.json",
        snapshot,
    )
    report["audit"][field] = not report["audit"][field]

    errors = _verify_rollback_audit(run["id"], snapshot, report["audit"])

    assert any("does not match snapshot state" in error and field in error for error in errors)


def test_rollback_evidence_distinguishes_disabled_from_missing_unit():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    snapshot["docker_socket_active"] = False
    snapshot["docker_socket_enabled"] = False
    snapshot["docker_socket_unit_file_state"] = "disabled"
    audit = _audit_for_scenario(run, "rollback_state_verify")
    audit["docker_socket_active"] = False
    audit["docker_socket_enabled"] = False
    audit["docker_socket_unit_file_state"] = "not-found"

    errors = _verify_rollback_audit(run["id"], snapshot, audit)

    assert any(
        "docker_socket_unit_file_state" in error for error in errors
    )


def test_snapshot_creating_report_rejects_different_operation_binding():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "install_apply",
        "reports/install.json",
        snapshot,
    )
    report["rollback"]["operation_id"] = "f" * 32

    errors = _verify_scenario_report(
        run,
        "install_apply",
        "reports/install.json",
        report,
    )

    assert any("different operation ID" in error for error in errors)


def test_rollback_report_may_reference_source_snapshot_operation():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "rollback_apply",
        "reports/rollback.json",
        snapshot,
    )

    assert report["rollback"]["operation_id"] != report["operation_id"]
    errors = _verify_scenario_report(
        run,
        "rollback_apply",
        "reports/rollback.json",
        report,
    )
    assert not any("different operation ID" in error for error in errors)


def test_two_baseline_delta_rejects_unrelated_managed_file_change(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def alter_docker_baseline(snapshot):
        docker = next(
            managed_file
            for managed_file in snapshot["managed_files"]
            if managed_file["path"] == "/etc/docker/daemon.json"
        )
        docker.update(
            existed=True,
            content_base64=f"attested:{'a' * 64}",
            mode=0o640,
        )
        return _with_snapshot_integrity(snapshot)

    archive = _rewrite_attested_snapshot(
        archives[first_id],
        "pre/rollback-snapshot.json",
        alter_docker_baseline,
    )
    for report_path in (
        "reports/snapshot.json",
        "reports/install.json",
        "reports/rollback.json",
    ):
        archive = _rewrite_attested_json_member(
            archive,
            report_path,
            lambda report: {
                **report,
                "rollback": alter_docker_baseline(report["rollback"]),
            },
        )
    archives[first_id] = archive
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("changed an unrelated managed file" in error for error in errors)


def test_retained_snapshot_requires_exact_safe_transaction_contract(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def weaken_apt_transaction(snapshot):
        snapshot["commands"][0].remove("--no-install-recommends")
        return _with_snapshot_integrity(snapshot)

    archive = _rewrite_attested_snapshot(
        archives[first_id],
        "pre/rollback-snapshot.json",
        weaken_apt_transaction,
    )
    for report_path in (
        "reports/snapshot.json",
        "reports/install.json",
        "reports/rollback.json",
    ):
        archive = _rewrite_attested_json_member(
            archive,
            report_path,
            lambda report: {
                **report,
                "rollback": weaken_apt_transaction(report["rollback"]),
            },
        )
    archives[first_id] = archive
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("exact safe rollback transaction contract" in error for error in errors)


@pytest.mark.parametrize(
    ("matrix_id", "mutation"),
    [
        ("ubuntu-24.04", "remote-payload"),
        ("rhel-9", "non-atomic"),
        ("sles-15", "manifest-rebind"),
    ],
)
def test_snapshot_contract_rejects_local_payload_authority_mutations(
    matrix_id,
    mutation,
):
    run = next(
        item
        for item in _passing_evidence()["runs"]
        if item["matrix_id"] == matrix_id
    )
    snapshot = _snapshot_for_run(run)
    assert _verify_snapshot_command_contract(
        matrix_id,
        run["package_manager"],
        snapshot,
        "canonical",
    ) == []

    if mutation == "remote-payload":
        command = snapshot["commands"][0]
        payload_index = next(
            index for index, part in enumerate(command) if ".payloads/" in part
        )
        command[payload_index] = "https://packages.invalid/rollback.deb"
    elif mutation == "non-atomic":
        command = snapshot["commands"][0]
        command[command.index("--apply")] = "--check"
    else:
        snapshot["package_payloads"]["directory"] = "rebound.json.payloads"
    _with_snapshot_integrity(snapshot)

    errors = _verify_snapshot_command_contract(
        matrix_id,
        run["package_manager"],
        snapshot,
        "canonical",
    )
    assert any("exact safe rollback transaction contract" in error for error in errors)


def test_controlled_convergence_requires_hardened_digest_bound_container(
    monkeypatch,
):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def remove_network_isolation(report):
        check = next(
            item
            for item in report["verification"]
            if item["name"] == "container.gpu"
        )
        check["command"]["command"].remove("--network=none")
        return report

    archives[first_id] = _rewrite_attested_json_member(
        archives[first_id],
        "reports/verify.json",
        remove_network_isolation,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("healthy post-install convergence" in error for error in errors)


def test_controlled_convergence_requires_container_pull_to_be_disabled(
    monkeypatch,
):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def allow_implicit_pull(report):
        check = next(
            item
            for item in report["verification"]
            if item["name"] == "container.gpu"
        )
        check["command"]["command"].remove("--pull=never")
        return report

    archives[first_id] = _rewrite_attested_json_member(
        archives[first_id],
        "reports/verify.json",
        allow_implicit_pull,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("healthy post-install convergence" in error for error in errors)


def test_controlled_convergence_requires_frozen_minimal_package_transaction(
    monkeypatch,
):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def weaken_install(report):
        for action in report["plan"]:
            if action["id"] == "install.packages":
                action["commands"][0].remove("--no-install-recommends")
        for result in report["command_results"]:
            if TOOLKIT_PACKAGE in result["command"]:
                result["command"].remove("--no-install-recommends")
        return report

    archive = archives[first_id]
    for report_path in ("reports/plan.json", "reports/install.json"):
        archive = _rewrite_attested_json_member(
            archive,
            report_path,
            weaken_install,
        )
    archives[first_id] = archive
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("plan-bound toolkit/runtime mutation" in error for error in errors)


def test_controlled_convergence_requires_offline_forward_payloads(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def allow_package_download(report):
        for action in report["plan"]:
            if action["id"] == "install.packages":
                action["commands"][0].remove("--no-download")
        for result in report["command_results"]:
            if ".payloads/" in " ".join(result["command"]):
                result["command"].remove("--no-download")
        return report

    archives[first_id] = _rewrite_attested_json_member(
        archives[first_id],
        "reports/install.json",
        allow_package_download,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("plan-bound toolkit/runtime mutation" in error for error in errors)


def test_policy_staging_requires_frozen_metadata_command(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    rhel_run = next(run for run in evidence["runs"] if run["package_manager"] == "dnf")
    artifact_id = rhel_run["artifacts"][0]["id"]
    unsafe_command = [
        "dnf",
        "-C",
        "module",
        "enable",
        "-y",
        "nvidia-driver:580-open",
    ]

    def weaken_lock(report):
        for action in report["plan"]:
            if action["id"].startswith("lock."):
                action["commands"][0] = list(unsafe_command)
        for result in report["command_results"]:
            if result["command"] == _lock_command(rhel_run):
                result["command"] = list(unsafe_command)
        return report

    archive = _rewrite_attested_json_member(
        archives[artifact_id],
        "reports/lock.json",
        weaken_lock,
    )
    archives[artifact_id] = archive
    rhel_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("exact frozen-metadata policy mutation" in error for error in errors)


def test_dnf_policy_staging_rejects_mismatched_check_and_apply_tokens(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    rhel_run = next(run for run in evidence["runs"] if run["package_manager"] == "dnf")
    artifact_id = rhel_run["artifacts"][0]["id"]

    def tamper_check_token(report):
        fresh = next(
            verification
            for verification in report["verification"]
            if verification["name"]
            == "packages.post-quarantine-policy-preflight"
        )
        fresh["command"]["stdout"] = (
            "[verified:dnf-module-proof-v2:check:" + "9" * 64 + "]"
        )
        return report

    archive = _rewrite_attested_json_member(
        archives[artifact_id],
        "reports/lock.json",
        tamper_check_token,
    )
    archives[artifact_id] = archive
    rhel_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("lacks an accepted exact schema-2 check proof" in error for error in errors)


def test_dnf_policy_staging_rejects_report_proof_order_tamper(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    rhel_run = next(run for run in evidence["runs"] if run["package_manager"] == "dnf")
    artifact_id = rhel_run["artifacts"][0]["id"]

    def move_fresh_check_after_policy_verification(report):
        verifications = report["verification"]
        fresh = next(
            verification
            for verification in verifications
            if verification["name"]
            == "packages.post-quarantine-policy-preflight"
        )
        verifications.remove(fresh)
        verifications.append(fresh)
        return report

    archive = _rewrite_attested_json_member(
        archives[artifact_id],
        "reports/lock.json",
        move_fresh_check_after_policy_verification,
    )
    archives[artifact_id] = archive
    rhel_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("proof chronology" in error for error in errors)


def test_dnf_policy_staging_rejects_journal_check_after_apply(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    rhel_run = next(run for run in evidence["runs"] if run["package_manager"] == "dnf")
    artifact_id = rhel_run["artifacts"][0]["id"]

    def move_checks_after_apply(payload):
        entries = [json.loads(line) for line in payload.splitlines()]
        checks = [
            entry
            for entry in entries
            if isinstance(entry.get("command"), list)
            and len(entry["command"]) >= 5
            and entry["command"][4] == "--check"
        ]
        retained = [entry for entry in entries if entry not in checks]
        insertion = next(
            index
            for index, entry in enumerate(retained)
            if entry.get("event") == "launcher-release-authorized"
        )
        retained[insertion:insertion] = checks
        return b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode()
            for entry in retained
        )

    archive = _rewrite_attested_member(
        archives[artifact_id],
        "reports/lock.journal.jsonl",
        move_checks_after_apply,
    )
    archives[artifact_id] = archive
    rhel_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any(
        "initial-check, snapshot, fresh-check, apply chronology" in error
        for error in errors
    )


@pytest.mark.parametrize("tamper", ["delete", "relabel"])
def test_dnf_policy_staging_rejects_missing_fresh_journal_event(
    monkeypatch,
    tamper,
):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    rhel_run = next(
        run for run in evidence["runs"] if run["package_manager"] == "dnf"
    )
    artifact_id = rhel_run["artifacts"][0]["id"]

    def tamper_fresh_check(payload):
        entries = [json.loads(line) for line in payload.splitlines()]
        snapshot_index = next(
            index
            for index, entry in enumerate(entries)
            if entry.get("event") == "rollback-snapshot-persisted"
        )
        fresh_indexes = [
            index
            for index, entry in enumerate(entries)
            if index > snapshot_index
            and isinstance(entry.get("command"), list)
            and len(entry["command"]) >= 5
            and entry["command"][4] == "--check"
        ]
        assert len(fresh_indexes) == 2
        if tamper == "delete":
            entries = [
                entry
                for index, entry in enumerate(entries)
                if index not in fresh_indexes
            ]
        else:
            assert tamper == "relabel"
            entries[fresh_indexes[1]]["event"] = "command-started"
        return b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode()
            for entry in entries
        )

    archive = _rewrite_attested_member(
        archives[artifact_id],
        "reports/lock.journal.jsonl",
        tamper_fresh_check,
    )
    archives[artifact_id] = archive
    rhel_run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any(
        "initial-check, snapshot, fresh-check, apply chronology" in error
        or "unmatched" in error
        or "invalid command-started" in error
        or "retained journal" in error
        for error in errors
    )


def test_controlled_convergence_does_not_claim_host_cuda_compat(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def add_compat_observation(report):
        report["audit"]["cuda_compatibility"] = [
            {
                "version": "13.1",
                "package_name": "cuda-compat-13-1",
                "package_version": "590.1-1",
                "library_path": "/usr/local/cuda-13.1/compat/libcuda.so.1",
                "library_present": True,
                "library_probe": {
                    "command": ["python3", "-I", "-c", "probe"],
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "skipped": False,
                    "reason": None,
                },
            }
        ]
        return report

    archives[first_id] = _rewrite_attested_json_member(
        archives[first_id],
        "reports/doctor.json",
        add_compat_observation,
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[first_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("CUDA forward-compatibility deployment evidence" in error for error in errors)


def test_artifact_member_limit_accepts_boundary_and_rejects_oversize(monkeypatch):
    monkeypatch.setattr("scripts.check_release_evidence.MAX_ARTIFACT_MEMBER_BYTES", 1024)
    accepted = _zip_members({"reports/doctor.json": b"x" * 1024})
    assert _artifact_evidence_members(accepted)["reports/doctor.json"] == b"x" * 1024

    rejected = _zip_members({"reports/doctor.json": b"x" * 1025})
    with pytest.raises(ArtifactEvidenceError, match="exceeds the safety limit"):
        _artifact_evidence_members(rejected)


def test_attestation_export_redacts_host_and_managed_file_secrets(monkeypatch):
    pseudonym_key = bytes(range(32))
    key_draw_sizes = []

    def draw_pseudonym_key(size):
        key_draw_sizes.append(size)
        return pseudonym_key

    monkeypatch.setattr(
        "scripts.export_integration_attestation.secrets.token_bytes",
        draw_pseudonym_key,
    )
    run = next(
        item
        for item in _passing_evidence()["runs"]
        if item["matrix_id"] == "mig-toggle"
    )
    snapshot = _snapshot_for_run(run)
    snapshot["managed_files"][0]["existed"] = True
    snapshot["managed_files"][0]["content_base64"] = "c2VjcmV0"
    snapshot["managed_files"][0]["mode"] = 0o640
    _with_snapshot_integrity(snapshot)
    report = _report_for_scenario(run, "doctor", "reports/doctor.json", snapshot)
    report["audit"]["nvidia_smi"]["stdout"] = "GPU-UUID: secret"
    report["audit"]["package_policy"]["observation"]["stdout"] = "pin: secret"
    report["findings"][0]["evidence"] = {"serial": "secret"}
    report["findings"][0]["remediation"] = (
        f"Recover with the private snapshot at {snapshot['path']}"
    )
    raw_gpu_uuid = snapshot["gpu_uuids"][0]
    raw_mig_uuid = report["audit"]["mig_device_uuids"][0]
    raw_host_id = report["host_id"]
    mig_command = ["nvidia-smi", "-i", raw_gpu_uuid, "-mig", "1"]
    report["plan"] = [
        {
            "id": "enable.mig",
            "description": "enable MIG",
            "commands": [copy.deepcopy(mig_command)],
            "destructive": True,
            "reason": "test",
        }
    ]
    report["command_results"] = [
        {
            "command": copy.deepcopy(mig_command),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "skipped": False,
            "reason": None,
        }
    ]
    journal = "\n".join(
        json.dumps(entry)
        for entry in (
            {
                "event": "command-started",
                "operation_id": report["operation_id"],
                "timestamp": report["started_at"],
                "command": copy.deepcopy(mig_command),
                "mutating": True,
            },
            {
                "event": "command-finished",
                "operation_id": report["operation_id"],
                "timestamp": report["completed_at"],
                "command": copy.deepcopy(mig_command),
                "mutating": True,
                "returncode": 0,
                "skipped": False,
                "reason": None,
            },
        )
    ) + "\n"
    exported = build_attestation(
        {
            "reports/doctor.json": (json.dumps(report) + "\n").encode(),
            "reports/doctor.journal.jsonl": journal.encode(),
            "pre/rollback-snapshot.json": (json.dumps(snapshot) + "\n").encode(),
        },
        matrix_id=run["matrix_id"],
        workflow_run_id=run["workflow_run_id"],
        workflow_run_attempt=run["workflow_run_attempt"],
        qualification_wheel_name=run["qualification_wheel"]["name"],
        qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
    )

    sanitized_report = json.loads(exported["reports/doctor.json"])
    sanitized_snapshot = json.loads(exported["pre/rollback-snapshot.json"])
    attestation_manifest = json.loads(exported["attestation.json"])
    assert key_draw_sizes == [32]
    assert attestation_manifest["schema_version"] == "1.3"
    assert all(
        set(entry) == {"path", "source_path", "sha256"}
        for entry in attestation_manifest["files"]
    )
    assert attestation_manifest["qualification_wheel"] == run[
        "qualification_wheel"
    ]
    expected_host_id = _expected_keyed_attestation(
        pseudonym_key,
        "host-id",
        raw_host_id,
    )
    assert sanitized_report["host_id"] == expected_host_id
    public_context_guess = "attested:" + hashlib.sha256(
        (
            f"{run['matrix_id']}:{run['workflow_run_id']}:"
            f"{run['workflow_run_attempt']}\0{raw_host_id}"
        ).encode()
    ).hexdigest()
    assert sanitized_report["host_id"] != public_context_guess
    assert sanitized_report["audit"]["nvidia_smi"]["stdout"] == "[redacted]"
    assert (
        sanitized_report["audit"]["package_policy"]["observation"]["stdout"]
        == "[redacted]"
    )
    assert sanitized_report["audit"]["module"]["devices"] == []
    assert sanitized_report["audit"]["gpu_uuids"][0].startswith("attested:")
    assert (
        sanitized_report["audit"]["gpu_uuids"]
        == sanitized_snapshot["gpu_uuids"]
    )
    assert sanitized_report["audit"]["mig_device_uuids"][0].startswith(
        "attested:"
    )
    assert sanitized_report["audit"]["mig_geometry"][0][
        "gpu_uuid"
    ] == sanitized_report["audit"]["gpu_uuids"][0]
    assert sanitized_report["findings"][0]["evidence"] == {}
    assert sanitized_report["findings"][0]["remediation"] == "[redacted]"
    assert sanitized_snapshot["path"].startswith("attested:")
    assert sanitized_snapshot["managed_files"][0]["content_base64"].startswith(
        "attested:"
    )
    expected_content = _expected_keyed_attestation(
        pseudonym_key,
        "managed-file-content",
        snapshot["managed_files"][0]["path"],
        "c2VjcmV0",
    )
    assert (
        sanitized_snapshot["managed_files"][0]["content_base64"]
        == expected_content
    )
    attested_gpu_uuid = sanitized_report["audit"]["gpu_uuids"][0]
    assert sanitized_report["plan"][0]["commands"][0][2] == attested_gpu_uuid
    assert sanitized_report["command_results"][0]["command"][2] == attested_gpu_uuid
    sanitized_journal = [
        json.loads(line)
        for line in exported["reports/doctor.journal.jsonl"].splitlines()
    ]
    assert all(entry["command"][2] == attested_gpu_uuid for entry in sanitized_journal)
    assert b"secret" not in b"".join(exported.values())
    assert b"GPU-11111111-2222-3333-4444-555555555555" not in b"".join(
        exported.values()
    )
    assert raw_mig_uuid.encode() not in b"".join(exported.values())
    assert snapshot["path"].encode() not in b"".join(exported.values())
    assert pseudonym_key not in b"".join(exported.values())


def test_attestation_export_uses_unretained_keyed_pseudonyms(monkeypatch):
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    snapshot["managed_files"] = [
        {
            "path": "/etc/docker/daemon.json",
            "existed": True,
            "content_base64": "c2FtZQ==",
            "mode": 0o600,
        },
        {
            "path": "/etc/dnf/modules.d/nvidia-driver.module",
            "existed": True,
            "content_base64": "c2FtZQ==",
            "mode": 0o600,
        },
    ]
    _with_snapshot_integrity(snapshot)
    members = {
        "pre/rollback-snapshot.json": (
            json.dumps(snapshot) + "\n"
        ).encode()
    }
    kwargs = {
        "matrix_id": run["matrix_id"],
        "workflow_run_id": run["workflow_run_id"],
        "workflow_run_attempt": run["workflow_run_attempt"],
        "qualification_wheel_name": run["qualification_wheel"]["name"],
        "qualification_wheel_sha256": run["qualification_wheel"]["sha256"],
    }

    first = build_attestation(
        members, pseudonym_key=b"a" * 32, **kwargs
    )
    repeated = build_attestation(
        members, pseudonym_key=b"a" * 32, **kwargs
    )
    second = build_attestation(
        members, pseudonym_key=b"b" * 32, **kwargs
    )

    assert first == repeated
    assert first != second
    first_snapshot = json.loads(first["pre/rollback-snapshot.json"])
    second_snapshot = json.loads(second["pre/rollback-snapshot.json"])
    assert first_snapshot["host_id"] != second_snapshot["host_id"]
    assert first_snapshot["path"] != second_snapshot["path"]
    first_contents = [
        item["content_base64"] for item in first_snapshot["managed_files"]
    ]
    assert len(set(first_contents)) == 2
    assert b"a" * 32 not in b"".join(first.values())

    default_keys = iter((b"c" * 32, b"d" * 32))
    default_draw_sizes = []

    def draw_default_key(size):
        default_draw_sizes.append(size)
        return next(default_keys)

    monkeypatch.setattr(
        "scripts.export_integration_attestation.secrets.token_bytes",
        draw_default_key,
    )
    default_first = build_attestation(members, **kwargs)
    default_second = build_attestation(members, **kwargs)
    assert default_draw_sizes == [32, 32]
    assert default_first != default_second
    assert b"c" * 32 not in b"".join(default_first.values())
    assert b"d" * 32 not in b"".join(default_second.values())

    for invalid_key in (b"short", "not-bytes"):
        with pytest.raises(
            AttestationExportError,
            match="pseudonym key must be exactly 32 bytes",
        ):
            build_attestation(
                members,
                pseudonym_key=invalid_key,
                **kwargs,
            )


def test_attestation_export_rejects_corrupt_source_snapshot():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    snapshot["kernel"] = "tampered-after-integrity-seal"

    with pytest.raises(
        AttestationExportError,
        match="source rollback snapshot failed its integrity check",
    ):
        build_attestation(
            {
                "pre/rollback-snapshot.json": (
                    json.dumps(snapshot) + "\n"
                ).encode(),
            },
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def _attestation_export_inputs(tmp_path):
    tmp_path.chmod(0o700)
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    source = tmp_path / "source"
    (source / "pre").mkdir(parents=True)
    (source / "pre" / "rollback-snapshot.json").write_text(
        json.dumps(snapshot) + "\n",
        encoding="utf-8",
    )
    return run, source, tmp_path / "export"


def _export_attestation_for_run(run, source, output):
    attestation_export.export_attestation(
        source,
        output,
        matrix_id=run["matrix_id"],
        workflow_run_id=run["workflow_run_id"],
        workflow_run_attempt=run["workflow_run_attempt"],
        qualification_wheel_name=run["qualification_wheel"]["name"],
        qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
    )


def _verify_attestation_for_run(run, output):
    attestation_export.verify_export_directory(
        output,
        matrix_id=run["matrix_id"],
        workflow_run_id=run["workflow_run_id"],
        workflow_run_attempt=run["workflow_run_attempt"],
        qualification_wheel_name=run["qualification_wheel"]["name"],
        qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
    )


def test_attestation_export_creates_and_verifies_exact_private_directory(tmp_path):
    run, source, output = _attestation_export_inputs(tmp_path)
    _export_attestation_for_run(run, source, output)
    _verify_attestation_for_run(run, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "pre").stat().st_mode) == 0o700
    files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert files == {
        "attestation.json",
        "pre/rollback-snapshot.json",
        "pre/rollback-snapshot.sha256",
    }
    for relative in files:
        metadata = (output / relative).stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1

    assert (
        attestation_export.main(
            [
                "--verify-only",
                "--output",
                str(output),
                "--matrix-id",
                run["matrix_id"],
                "--workflow-run-id",
                str(run["workflow_run_id"]),
                "--workflow-run-attempt",
                str(run["workflow_run_attempt"]),
                "--qualification-wheel-name",
                run["qualification_wheel"]["name"],
                "--qualification-wheel-sha256",
                run["qualification_wheel"]["sha256"],
            ]
        )
        == 0
    )


@pytest.mark.parametrize("existing_kind", ["directory", "symlink", "dangling"])
def test_attestation_export_rejects_every_preexisting_output_path(
    tmp_path,
    existing_kind,
):
    run, source, output = _attestation_export_inputs(tmp_path)
    if existing_kind == "directory":
        output.mkdir()
    else:
        target = tmp_path / (
            "existing-target" if existing_kind == "symlink" else "missing-target"
        )
        if existing_kind == "symlink":
            target.mkdir()
        output.symlink_to(target, target_is_directory=True)

    with pytest.raises(AttestationExportError, match="output path already exists"):
        _export_attestation_for_run(run, source, output)


@pytest.mark.parametrize(
    "tamper",
    [
        "extra-member",
        "missing-member",
        "digest",
        "file-mode",
        "directory-mode",
        "symlink-member",
        "hardlink-member",
        "duplicate-manifest",
        "traversal-manifest",
        "context-manifest",
    ],
)
def test_attestation_verifier_rejects_exact_directory_tampering(tmp_path, tamper):
    run, source, output = _attestation_export_inputs(tmp_path)
    _export_attestation_for_run(run, source, output)
    member = output / "pre" / "rollback-snapshot.json"
    manifest_path = output / "attestation.json"

    if tamper == "extra-member":
        extra = output / "unexpected.json"
        extra.write_text("{}\n", encoding="utf-8")
        extra.chmod(0o600)
    elif tamper == "missing-member":
        member.unlink()
    elif tamper == "digest":
        member.write_text("{}\n", encoding="utf-8")
    elif tamper == "file-mode":
        member.chmod(0o644)
    elif tamper == "directory-mode":
        (output / "pre").chmod(0o755)
    elif tamper == "symlink-member":
        member.unlink()
        member.symlink_to("../attestation.json")
    elif tamper == "hardlink-member":
        member.unlink()
        os.link(manifest_path, member)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "duplicate-manifest":
            manifest["files"].append(dict(manifest["files"][0]))
        elif tamper == "traversal-manifest":
            manifest["files"][0]["path"] = "../raw.json"
        elif tamper == "context-manifest":
            manifest["matrix_id"] = "tampered"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)

    with pytest.raises(AttestationExportError):
        _verify_attestation_for_run(run, output)


def test_attestation_verifier_rejects_nested_member_added_after_inventory_scan(
    tmp_path,
    monkeypatch,
):
    run, source, output = _attestation_export_inputs(tmp_path)
    _export_attestation_for_run(run, source, output)
    original_reader = attestation_export._read_export_member
    injected = False

    def inject_after_manifest_read(path, label):
        nonlocal injected
        payload = original_reader(path, label)
        if label == "attestation.json" and not injected:
            late_member = output / "pre" / "late-private.txt"
            late_member.write_text("{}\n", encoding="utf-8")
            late_member.chmod(0o600)
            injected = True
        return payload

    monkeypatch.setattr(
        attestation_export,
        "_read_export_member",
        inject_after_manifest_read,
    )

    with pytest.raises(
        AttestationExportError,
        match="changed during verification",
    ):
        _verify_attestation_for_run(run, output)


def test_attestation_verifier_rejects_same_size_member_rewrite_after_read(
    tmp_path,
    monkeypatch,
):
    run, source, output = _attestation_export_inputs(tmp_path)
    _export_attestation_for_run(run, source, output)
    original_reader = attestation_export._read_export_member
    rewritten = False

    def rewrite_after_member_read(path, label):
        nonlocal rewritten
        payload = original_reader(path, label)
        if label == "pre/rollback-snapshot.json" and not rewritten:
            before = path.stat()
            replacement = bytearray(payload)
            replacement[0] = ord("[")
            path.write_bytes(replacement)
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            rewritten = True
        return payload

    monkeypatch.setattr(
        attestation_export,
        "_read_export_member",
        rewrite_after_member_read,
    )

    with pytest.raises(
        AttestationExportError,
        match="entries changed during verification",
    ):
        _verify_attestation_for_run(run, output)


def test_attestation_export_checks_size_limits_before_creating_output(
    tmp_path,
    monkeypatch,
):
    run, source, output = _attestation_export_inputs(tmp_path)
    monkeypatch.setattr(attestation_export, "MAX_EXPORT_MEMBER_BYTES", 1)

    with pytest.raises(AttestationExportError, match="member size"):
        _export_attestation_for_run(run, source, output)

    assert not output.exists()


def test_attestation_export_rejects_noncanonical_evidence_filename():
    run = _passing_evidence()["runs"][0]

    with pytest.raises(AttestationExportError, match="unsupported attestation path"):
        build_attestation(
            {"reports/GPU-raw-identity-1234567890.json": b"{}\n"},
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def test_attestation_export_rejects_forged_cuda_probe_success():
    run = _passing_evidence()["runs"][0]
    snapshot = _snapshot_for_run(run)
    report = _report_for_scenario(
        run,
        "verify_apply",
        "reports/verify.json",
        snapshot,
    )
    container = next(
        check
        for check in report["verification"]
        if check["name"] == "container.gpu"
    )
    container["command"]["stdout"] = "forged success\n"

    with pytest.raises(
        AttestationExportError,
        match="no CUDA Driver API success marker",
    ):
        build_attestation(
            {"reports/verify.json": (json.dumps(report) + "\n").encode()},
            matrix_id=run["matrix_id"],
            workflow_run_id=run["workflow_run_id"],
            workflow_run_attempt=run["workflow_run_attempt"],
            qualification_wheel_name=run["qualification_wheel"]["name"],
            qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        )


def test_github_job_binds_matrix_identity_and_runner_labels(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    _mock_github(monkeypatch, evidence, archives)
    evidence["runs"][0]["runner_labels"].append("spoofed-rhel-9")

    errors = verify_github_evidence(evidence, token="token")
    assert any("runner_labels" in error for error in errors)

    evidence["runs"][0]["runner_labels"].pop()
    evidence["runs"][0]["workflow_job_id"] += 999
    errors = verify_github_evidence(evidence, token="token")
    assert any("workflow job" in error and "was not found" in error for error in errors)


def test_github_artifact_binds_report_host_and_desired_state(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    first_run = evidence["runs"][0]
    first_id = first_run["artifacts"][0]["id"]

    def relabel_report(report):
        report["audit"]["os_id"] = "rhel"
        report["desired"]["driver"] = "595.71.05"
        return report

    archives[first_id] = _rewrite_json_members(
        archives[first_id], {"reports/plan.json": relabel_report}
    )
    first_run["artifacts"][0]["sha256"] = hashlib.sha256(archives[first_id]).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("host metadata mismatch: os_id" in error for error in errors)
    assert any("does not match its checked-in desired state" in error for error in errors)


def test_matrix_capabilities_must_be_observed_in_reports(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    runs = {run["matrix_id"]: run for run in evidence["runs"]}

    def disable_secure_boot(report):
        report["audit"]["kernel"]["secure_boot_enabled"] = False
        return report

    def stop_fabric_manager(report):
        report["audit"]["fabric_manager_active"] = False
        return report

    def remove_mig_transition(report):
        for result in report["command_results"]:
            if result["command"][-2:] == ["-mig", "1"]:
                result["command"] = ["true", "install"]
        return report

    changes = {
        "secure-boot": {"reports/doctor.json": disable_secure_boot},
        "fabric-manager": {"reports/doctor.json": stop_fabric_manager},
        "mig-toggle": {"reports/install.json": remove_mig_transition},
    }
    for matrix_id, rewrites in changes.items():
        run = runs[matrix_id]
        artifact_id = run["artifacts"][0]["id"]
        archives[artifact_id] = _rewrite_json_members(archives[artifact_id], rewrites)
        run["artifacts"][0]["sha256"] = hashlib.sha256(
            archives[artifact_id]
        ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    errors = verify_github_evidence(evidence, token="token")
    assert any("secure-boot matrix" in error for error in errors)
    assert any("fabric-manager matrix" in error for error in errors)
    assert any("MIG matrix" in error for error in errors)


def test_github_evidence_allows_truthfully_observed_capability_overlap(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    run = next(item for item in evidence["runs"] if item["matrix_id"] == "secure-boot")
    run["runner_labels"].append("mig-capable")
    run["capabilities"]["mig_capable"] = True
    artifact_id = run["artifacts"][0]["id"]

    def observe_mig_capability(report):
        report["audit"]["mig_capable"] = True
        return report

    archives[artifact_id] = _rewrite_attested_json_member(
        archives[artifact_id],
        "reports/doctor.json",
        observe_mig_capability,
    )
    run["artifacts"][0]["sha256"] = hashlib.sha256(
        archives[artifact_id]
    ).hexdigest()
    _mock_github(monkeypatch, evidence, archives)

    assert verify_github_evidence(evidence, token="token") == []


def test_cli_rejects_missing_evidence(capsys, tmp_path):
    rc = main(["--evidence", str(tmp_path / "missing.json"), "--release", "v0.1.0", "--commit", "a" * 40])
    assert rc == 2
    assert "release evidence is invalid" in capsys.readouterr().err


def test_commit_provenance_allows_only_evidence_after_tested_commit(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "code.py").write_text("tested = True\n", encoding="utf-8")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-m", "tested source")
    tested = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    evidence = tmp_path / "integrations" / "results.v0.1.0.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    _git(tmp_path, "add", str(evidence.relative_to(tmp_path)))
    _git(tmp_path, "commit", "-m", "add evidence")
    release = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    assert check_commit_provenance(tested, release, evidence) == []

    (tmp_path / "code.py").write_text("tested = False\n", encoding="utf-8")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-m", "untested change")
    changed_release = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    assert any("source changed" in error for error in check_commit_provenance(tested, changed_release, evidence))


def _passing_evidence():
    evidence = json.loads(Path("integrations/results.example.json").read_text(encoding="utf-8"))
    evidence["release"] = "v0.1.0"
    evidence["commit"] = "a" * 40
    evidence["overall_status"] = "passed"
    evidence["qualification_wheel"] = {
        "name": QUALIFICATION_WHEEL_NAME,
        "sha256": QUALIFICATION_WHEEL_SHA256,
    }
    evidence["required_coverage"] = {name: True for name in evidence["required_coverage"]}
    base = evidence["runs"][0]
    base["status"] = "passed"
    base["head_sha"] = evidence["commit"]
    base["artifacts"][0]["sha256"] = "a" * 64
    base["scenarios"] = [
        {"name": name, "status": "passed", "report": SCENARIO_REPORTS[name]}
        for name in sorted(REQUIRED_SCENARIOS)
    ]
    runs = []
    profiles = (
        (
            "ubuntu-24.04",
            "ubuntu",
            "24.04",
            "apt-get",
            ["self-hosted", "nvidia-gpu", "disposable", "ubuntu-24.04"],
            "examples/compute-580-open.yaml",
            None,
        ),
        (
            "rhel-9",
            "rhel",
            "9.6",
            "dnf",
            ["self-hosted", "nvidia-gpu", "disposable", "rhel-9"],
            "examples/compute-580-open.yaml",
            None,
        ),
        (
            "sles-15",
            "sles",
            "15.6",
            "zypper",
            ["self-hosted", "nvidia-gpu", "disposable", "sles-15"],
            "examples/compute-580-open.yaml",
            None,
        ),
        (
            "secure-boot",
            "ubuntu",
            "24.04",
            "apt-get",
            [
                "self-hosted",
                "nvidia-gpu",
                "disposable",
                "ubuntu-24.04",
                "secure-boot",
            ],
            "examples/compute-580-open.yaml",
            "secure_boot",
        ),
        (
            "fabric-manager",
            "ubuntu",
            "24.04",
            "apt-get",
            [
                "self-hosted",
                "nvidia-gpu",
                "disposable",
                "ubuntu-24.04",
                "fabric-manager",
            ],
            "examples/compute-580-open-fabric-manager.yaml",
            "fabric_manager",
        ),
        (
            "mig-toggle",
            "ubuntu",
            "24.04",
            "apt-get",
            [
                "self-hosted",
                "nvidia-gpu",
                "disposable",
                "ubuntu-24.04",
                "mig-capable",
            ],
            "examples/compute-580-open-mig.yaml",
            "mig_toggle",
        ),
    )
    workflow_run_id = 1
    for index, (
        matrix_id,
        os_id,
        os_version,
        manager,
        labels,
        desired,
        capability,
    ) in enumerate(
        profiles,
        start=1,
    ):
        run = copy.deepcopy(base)
        run["id"] = matrix_id
        run["matrix_id"] = matrix_id
        run["os_id"] = os_id
        run["package_manager"] = manager
        run["qualification_wheel"] = copy.deepcopy(
            evidence["qualification_wheel"]
        )
        run["os_version"] = os_version
        run["runner_labels"] = labels
        run["desired"] = desired
        run["capabilities"] = {
            name: name == capability
            for name in (
                "secure_boot",
                "fabric_manager",
                "mig_capable",
                "mig_toggle",
            )
        }
        if matrix_id == "mig-toggle":
            run["capabilities"]["mig_capable"] = True
        if matrix_id != "fabric-manager":
            fm_fault = next(
                scenario
                for scenario in run["scenarios"]
                if scenario["name"] == "doctor_fabric_manager_inactive"
            )
            fm_fault["status"] = "not_applicable"
            fm_fault["report"] = None
        run["workflow_run_id"] = workflow_run_id
        run["workflow_run"] = (
            "https://github.com/zeroecco/nvidia-converge/actions/runs/"
            f"{workflow_run_id}"
        )
        run["workflow_job_id"] = 100 + index
        run["workflow_job"] = (
            "https://github.com/zeroecco/nvidia-converge/actions/runs/"
            f"{workflow_run_id}/job/{100 + index}"
        )
        run["artifacts"][0]["id"] = index
        run["artifacts"][0]["name"] = (
            f"gpu-integration-{matrix_id}-{workflow_run_id}-1"
        )
        run["artifacts"][0]["uri"] = (
            "https://api.github.com/repos/zeroecco/nvidia-converge/"
            f"actions/artifacts/{index}/zip"
        )
        runs.append(run)
    evidence["runs"] = runs
    evidence["qualified_platforms"] = [
        {
            "os_id": os_id,
            "os_version": os_version,
            "package_manager": package_manager,
        }
        for os_id, os_version, package_manager in sorted(
            {
                (run["os_id"], run["os_version"], run["package_manager"])
                for run in runs
            }
        )
    ]
    evidence["recipe_path_coverage"] = {
        "apt_get": True,
        "dnf": True,
        "zypper": True,
    }
    for run in evidence["runs"]:
        archive = _artifact_zip_for_run(run)
        run["artifacts"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
    return evidence


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


def _archives_for(evidence):
    return {
        run["artifacts"][0]["id"]: _artifact_zip_for_run(run)
        for run in evidence["runs"]
    }


def _mock_github(
    monkeypatch,
    evidence,
    archives,
    *,
    preserve_metadata_digest=False,
    head_branch="main",
    event="repository_dispatch",
):
    runs = copy.deepcopy(evidence["runs"])
    original_digests = {
        run["artifacts"][0]["id"]: run["artifacts"][0]["sha256"]
        for run in runs
    }

    def fake_github_json(url, token):
        assert token == "token"
        run_id = int(url.split("/runs/", 1)[1].split("/", 1)[0])
        matching_runs = [run for run in runs if run["workflow_run_id"] == run_id]
        run = matching_runs[0]
        if "/jobs?per_page=100" in url:
            return {
                "total_count": len(matching_runs),
                "jobs": [
                    {
                        "id": item["workflow_job_id"],
                        "run_id": run_id,
                        "head_sha": item["head_sha"],
                        "html_url": item["workflow_job"],
                        "name": item["matrix_id"],
                        "status": "completed",
                        "conclusion": "success",
                        "workflow_name": "GPU Integration",
                        "labels": item["runner_labels"],
                        "runner_id": item["workflow_job_id"] + 1000,
                        "runner_name": f"runner-{item['matrix_id']}",
                        "started_at": item["started_at"],
                        "completed_at": item["completed_at"],
                    }
                    for item in matching_runs
                ],
            }
        if url.endswith("artifacts?per_page=100"):
            return {
                "total_count": len(matching_runs),
                "artifacts": [
                    {
                        "id": item["artifacts"][0]["id"],
                        "name": item["artifacts"][0]["name"],
                        "archive_download_url": item["artifacts"][0]["uri"],
                        "digest": (
                            "sha256:"
                            + (
                                original_digests[item["artifacts"][0]["id"]]
                                if preserve_metadata_digest
                                else hashlib.sha256(
                                    archives[item["artifacts"][0]["id"]]
                                ).hexdigest()
                            )
                        ),
                        "expired": False,
                        "size_in_bytes": len(archives[item["artifacts"][0]["id"]]),
                        "workflow_run": {
                            "id": run_id,
                            "head_sha": item["head_sha"],
                        },
                    }
                    for item in matching_runs
                ],
            }
        return {
            "id": run_id,
            "run_attempt": run["workflow_run_attempt"],
            "head_branch": head_branch,
            "head_sha": run["head_sha"],
            "path": run["workflow_path"],
            "html_url": run["workflow_run"],
            "event": event,
            "conclusion": "success",
            "repository": {"full_name": evidence["repository"]},
            "head_repository": {"full_name": evidence["repository"]},
        }

    def fake_github_bytes(url, token):
        assert token == "token"
        artifact_id = next(
            run["artifacts"][0]["id"]
            for run in runs
            if run["artifacts"][0]["uri"] == url
        )
        return archives[artifact_id]

    monkeypatch.setattr("scripts.check_release_evidence._github_json", fake_github_json)
    monkeypatch.setattr("scripts.check_release_evidence._github_bytes", fake_github_bytes)


def test_github_verifier_rejects_non_main_workflow_run(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    _mock_github(monkeypatch, evidence, archives, head_branch="feature/untrusted")

    errors = verify_github_evidence(evidence, token="token")

    assert any("head_branch" in error for error in errors)


def test_github_verifier_rejects_old_workflow_dispatch_run(monkeypatch):
    evidence = _passing_evidence()
    archives = _archives_for(evidence)
    _mock_github(monkeypatch, evidence, archives, event="workflow_dispatch")

    errors = verify_github_evidence(evidence, token="token")

    assert any("event" in error for error in errors)


def _append_journal_command_pair(entries, report, result, mutating):
    command = result["command"]
    entries.extend(
        [
            {
                "event": "command-started",
                "operation_id": report["operation_id"],
                "timestamp": report["started_at"],
                "command": command,
                "mutating": mutating,
            },
            {
                "event": "command-finished",
                "operation_id": report["operation_id"],
                "timestamp": report["started_at"],
                "command": command,
                "mutating": mutating,
                "returncode": result["returncode"],
                "skipped": result["skipped"],
                "reason": result["reason"],
            },
        ]
    )


def _artifact_zip_for_run(run):
    snapshot = _snapshot_for_run(run, policy_staged=True)
    policy_snapshot = _snapshot_for_run(run, policy_staged=False)
    report_members = {}
    for scenario in run["scenarios"]:
        if scenario["status"] != "passed":
            continue
        path = scenario["report"]
        if path in report_members:
            continue
        report_members[path] = _report_for_scenario(
            run,
            scenario["name"],
            path,
            snapshot,
            policy_snapshot,
        )

    snapshot_payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    members = {
        path: (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        for path, report in report_members.items()
    }
    members["pre/rollback-snapshot.json"] = snapshot_payload
    members["pre/policy-rollback-snapshot.json"] = (
        json.dumps(policy_snapshot, indent=2, sort_keys=True) + "\n"
    ).encode()
    for scenario in run["scenarios"]:
        restoration_path = FAULT_RESTORATION_REPORTS.get(scenario["name"])
        if scenario["status"] != "passed" or restoration_path is None:
            continue
        restoration = _report_for_scenario(
            run,
            "doctor",
            restoration_path,
            snapshot,
            policy_snapshot,
        )
        members[restoration_path] = (
            json.dumps(restoration, indent=2, sort_keys=True) + "\n"
        ).encode()
    final_cleanup = _report_for_scenario(
        run,
        "doctor",
        "restoration/final-cleanup.json",
        snapshot,
        policy_snapshot,
    )
    members["restoration/final-cleanup.json"] = (
        json.dumps(final_cleanup, indent=2, sort_keys=True) + "\n"
    ).encode()
    for path, report in report_members.items():
        if report["mode"] != "apply":
            continue
        journal_path = f"{path[:-5]}.journal.jsonl"
        report_snapshot = report["rollback"]
        entries = [
            {
                "event": "operation-started",
                "operation_id": report["operation_id"],
                "timestamp": report["started_at"],
            }
        ]
        snapshot_entry = {
            "event": "rollback-snapshot-persisted",
            "operation_id": report["operation_id"],
            "timestamp": report["started_at"],
            "snapshot_path": report_snapshot["path"],
            "snapshot_integrity_sha256": report_snapshot["integrity_sha256"],
            "snapshot_operation_id": report_snapshot["operation_id"],
            "snapshot_host_id": report_snapshot["host_id"],
        }
        dnf_check_verifications = []
        if report["command"] == "lock" and run["package_manager"] == "dnf":
            dnf_check_verifications = [
                verification
                for verification in report["verification"]
                if verification["name"]
                in {
                    "packages.policy-preflight",
                    "packages.post-quarantine-policy-preflight",
                }
            ]
            initial = next(
                verification
                for verification in dnf_check_verifications
                if verification["name"] == "packages.policy-preflight"
            )
            _append_journal_command_pair(entries, report, initial["command"], False)
        entries.append(snapshot_entry)
        if dnf_check_verifications:
            fresh = next(
                verification
                for verification in dnf_check_verifications
                if verification["name"]
                == "packages.post-quarantine-policy-preflight"
            )
            _append_journal_command_pair(entries, report, fresh["command"], False)
        if report["command"] in {"install", "verify", "lock", "rollback"}:
            for command_result in report["command_results"]:
                entries.append(
                    {
                        "event": "command-started",
                        "operation_id": report["operation_id"],
                        "timestamp": report["started_at"],
                        "command": command_result["command"],
                        "mutating": True,
                    }
                )
                entries.append(
                    {
                        "event": "command-finished",
                        "operation_id": report["operation_id"],
                        "timestamp": report["started_at"],
                        "command": command_result["command"],
                        "mutating": True,
                        "returncode": 0,
                        "skipped": False,
                        "reason": None,
                    }
                )
        release_target = {
            "install": "install-target",
            "verify": "operation-target",
            "lock": "operation-target",
            "rollback": "rollback-baseline",
        }.get(report["command"])
        if release_target is not None:
            entries.append(
                {
                    "event": "launcher-release-authorized",
                    "operation_id": report["operation_id"],
                    "timestamp": report["completed_at"],
                    "release_target": release_target,
                    "snapshot_path": report_snapshot["path"],
                    "snapshot_integrity_sha256": report_snapshot[
                        "integrity_sha256"
                    ],
                    "snapshot_operation_id": report_snapshot[
                        "operation_id"
                    ],
                    "snapshot_host_id": report_snapshot["host_id"],
                }
            )
        entries.append(
            {
                "event": "operation-completed",
                "operation_id": report["operation_id"],
                "timestamp": report["completed_at"],
                "exit_code": report["exit_code"],
                "incomplete": report["incomplete"],
                "outcome": report["outcome"],
            }
        )
        members[journal_path] = b"".join(
            (json.dumps(entry, sort_keys=True) + "\n").encode() for entry in entries
        )
    exported = build_attestation(
        members,
        matrix_id=run["matrix_id"],
        workflow_run_id=run["workflow_run_id"],
        workflow_run_attempt=run["workflow_run_attempt"],
        qualification_wheel_name=run["qualification_wheel"]["name"],
        qualification_wheel_sha256=run["qualification_wheel"]["sha256"],
        pseudonym_key=TEST_PSEUDONYM_KEY,
    )
    return _zip_members(exported)


def _snapshot_for_run(run, *, policy_staged=True):
    fabric_manager_required = (
        run["desired"] == "examples/compute-580-open-fabric-manager.yaml"
    )
    packages = [
        _package_record(run, "docker-ce", "28.3.3-1"),
        _package_record(run, "nvidia-open", "580.126.16-1"),
    ]
    if fabric_manager_required:
        packages.append(
            _package_record(
                run,
                "nvidia-fabricmanager-580",
                "580.126.16-1",
            )
        )
    if policy_staged and run["package_manager"] == "apt-get":
        packages.append(
            _package_record(
                run,
                "nvidia-driver-pinning-580",
                "580.126.16-1",
            )
        )
    packages.sort(
        key=lambda package: (
            package["name"],
            package["architecture"] or "",
            package["epoch"] or "",
            package["version"] or "",
        )
    )
    introduced_packages = list(TOOLKIT_PACKAGE_CLOSURE)
    if not policy_staged and run["package_manager"] == "apt-get":
        introduced_packages.append("nvidia-driver-pinning-580")
    snapshot_path = (
        "/var/lib/nvidia-converge/snapshots/integration.json"
        if policy_staged
        else "/var/lib/nvidia-converge/snapshots/policy-integration.json"
    )
    forward_packages = _snapshot_forward_packages(
        run,
        policy_staged=policy_staged,
    )
    package_payloads = _package_payload_bundle(
        run,
        snapshot_path,
        packages,
        forward_packages,
    )
    commands = _snapshot_commands(
        run,
        snapshot_path,
        packages,
        introduced_packages,
        package_payloads,
    )
    managed_files = [
        {
            "path": "/etc/docker/daemon.json",
            "existed": False,
            "content_base64": None,
            "mode": None,
        }
    ]
    policy_path = {
        "dnf": "/etc/dnf/modules.d/nvidia-driver.module",
        "zypper": "/etc/zypp/locks",
    }.get(run["package_manager"])
    if policy_path is not None:
        preexisting_policy_file = run["package_manager"] == "zypper"
        policy_file_exists = policy_staged or preexisting_policy_file
        managed_files.append(
            {
                "path": policy_path,
                "existed": policy_file_exists,
                "content_base64": (
                    "cG9saWN5"
                    if policy_staged
                    else "cHJlLXBvbGljeQ=="
                    if preexisting_policy_file
                    else None
                ),
                "mode": 0o644 if policy_file_exists else None,
            }
        )
    if run["package_manager"] == "dnf":
        managed_files.append(
            {
                "path": (
                    "/var/lib/dnf/modulefailsafe/"
                    "nvidia-driver:580-open:x86_64.yaml"
                ),
                "existed": policy_staged,
                "content_base64": "bW9kdWxlIHlhbWw=" if policy_staged else None,
                "mode": 0o644 if policy_staged else None,
            }
        )
    snapshot = {
        "path": snapshot_path,
        "packages": packages,
        "kernel": "6.8.0-integration",
        "module_version": "580.126.16",
        "commands": commands,
        "schema_version": "2.6",
        "created_at": run["started_at"],
        "operation_id": hashlib.sha256(
            f"{run['workflow_job_id']}:snapshot:{policy_staged}".encode()
        ).hexdigest()[:32],
        "host_id": "machine-id-sha256:"
        + hashlib.sha256(
            f"integration-host:{run['workflow_run_id']}".encode()
        ).hexdigest(),
        "os_id": run["os_id"],
        "os_version": run["os_version"],
        "architecture": "x86_64",
        "package_manager": run["package_manager"],
        "introduced_packages": introduced_packages,
        "module_loaded": True,
        "module_names": ["nvidia"],
        "module_open_module": True,
        "module_signed": True,
        "module_installed_version": "580.126.16",
        "module_installed_open_module": True,
        "module_installed_signed": True,
        "gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
        "mig_mode": "disabled",
        "mig_geometry": [],
        "docker_service_active": True,
        "docker_service_enabled": True,
        "docker_service_unit_file_state": "enabled",
        "docker_socket_active": True,
        "docker_socket_enabled": True,
        "docker_socket_unit_file_state": "enabled",
        "nvidia_persistenced_active": True,
        "nvidia_persistenced_enabled": True,
        "nvidia_persistenced_unit_file_state": "enabled",
        "fabric_manager_active": fabric_manager_required,
        "fabric_manager_enabled": fabric_manager_required,
        "fabric_manager_unit_file_state": (
            "enabled" if fabric_manager_required else "disabled"
        ),
        "managed_files": managed_files,
        "package_payloads": _json_dataclass(package_payloads),
    }
    canonical = json.dumps(
        snapshot, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    snapshot["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return snapshot


def _package_record(run, name, version):
    apt = run["package_manager"] == "apt-get"
    return {
        "name": name,
        "version": version,
        "manager": "apt" if apt else "rpm",
        "installed": True,
        "architecture": "amd64" if apt else "x86_64",
        "epoch": None,
    }


def _snapshot_forward_packages(run, *, policy_staged):
    if policy_staged:
        return [
            _package_record(run, package, "1.19.1-1")
            for package in TOOLKIT_PACKAGE_CLOSURE
        ]
    if run["package_manager"] == "apt-get":
        return [
            _package_record(
                run,
                "nvidia-driver-pinning-580",
                "580.126.16-1",
            )
        ]
    return []


def _package_info(record):
    return PackageInfo(
        record["name"],
        record["version"],
        record["manager"],
        record["installed"],
        architecture=record["architecture"],
        epoch=record["epoch"],
    )


def _package_payload_bundle(
    run,
    snapshot_path,
    baseline_packages,
    forward_packages,
):
    requested = {}
    for role, records in (
        ("baseline", baseline_packages),
        ("forward", forward_packages),
    ):
        for record in records:
            identity = (
                record["name"],
                record["architecture"],
                record["epoch"],
                record["version"],
            )
            requested.setdefault(identity, set()).add(role)
    rpm = run["package_manager"] != "apt-get"
    extension = "rpm" if rpm else "deb"
    payloads = []
    for identity, roles in sorted(requested.items()):
        name, architecture, epoch, version = identity
        content = (
            f"release-evidence-payload:{name}:{architecture}:"
            f"{epoch or ''}:{version}"
        ).encode()
        digest = hashlib.sha256(content).hexdigest()
        payloads.append(
            PackagePayload(
                name=name,
                architecture=architecture,
                epoch=epoch,
                version=version,
                format=extension,
                filename=f"{digest}.{extension}",
                sha256=digest,
                size_bytes=len(content),
                verification=(
                    "rpm-signature" if rpm else "apt-repository"
                ),
                roles=tuple(sorted(roles)),
                signer_ids=("deadbeef",) if rpm else (),
            )
        )
    return PackagePayloadBundle(
        directory=payload_bundle_directory(Path(snapshot_path)),
        packages=tuple(payloads),
        total_size_bytes=sum(payload.size_bytes for payload in payloads),
    )


def _package_payload_bundle_from_snapshot(snapshot):
    manifest = snapshot["package_payloads"]
    return PackagePayloadBundle(
        directory=manifest["directory"],
        packages=tuple(
            PackagePayload(
                name=payload["name"],
                architecture=payload["architecture"],
                epoch=payload["epoch"],
                version=payload["version"],
                format=payload["format"],
                filename=payload["filename"],
                sha256=payload["sha256"],
                size_bytes=payload["size_bytes"],
                verification=payload["verification"],
                roles=tuple(payload["roles"]),
                signer_ids=tuple(payload["signer_ids"]),
            )
            for payload in manifest["packages"]
        ),
        total_size_bytes=manifest["total_size_bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.dumps(asdict(value)))


def _snapshot_commands(
    run,
    snapshot_path,
    packages,
    introduced_packages,
    package_payloads,
):
    return _rollback_commands(
        [_package_info(package) for package in packages],
        run["package_manager"],
        remove_packages=introduced_packages,
        snapshot_path=snapshot_path,
        package_payloads=package_payloads,
    )


def _install_command(run, snapshot=None):
    if snapshot is not None:
        return _local_forward_command(run, snapshot)
    manager = run["package_manager"]
    if manager == "apt-get":
        return [
            manager,
            "install",
            "-y",
            "--allow-downgrades",
            "--no-install-recommends",
            TOOLKIT_PACKAGE,
        ]
    if manager in {"dnf", "yum"}:
        return [
            manager,
            "-C",
            "--setopt=install_weak_deps=False",
            "install",
            "-y",
            TOOLKIT_PACKAGE,
        ]
    return [
        manager,
        "--non-interactive",
        "--no-refresh",
        "install",
        "--no-recommends",
        TOOLKIT_PACKAGE,
    ]


def _lock_command(run, snapshot=None):
    manager = run["package_manager"]
    if manager == "apt-get":
        if snapshot is not None:
            return _local_forward_command(run, snapshot)
        return [
            manager,
            "install",
            "-y",
            "--allow-downgrades",
            "--no-install-recommends",
            "--purge",
            "nvidia-driver-pinning-580",
        ]
    if manager == "dnf":
        return dnf_module_enable_command(
            apply=True,
            stream="580-open",
            preflight_sha256=_dnf_module_preflight_sha256(),
        )
    return [manager, "--non-interactive", "addlock", "*nvidia* >= 590"]


def _dnf_module_proof_payload(*, applied):
    module_state_before = "1" * 64
    module_state_after = ("2" if applied else "1") * 64
    module_failsafe_before = "3" * 64
    module_failsafe_after = ("4" if applied else "3") * 64
    payload = {
        "active_modules_count": 1,
        "active_modules_sha256": "d" * 64,
        "active_target": [
            {
                "architecture": "x86_64",
                "context": "rhel9",
                "name": "nvidia-driver",
                "repository": "cuda-rhel9-x86_64",
                "stream": "580-open",
                "version": "202608020001",
                "yaml_sha256": "c" * 64,
            }
        ],
        "applied": applied,
        "changes": {
            "disable": [],
            "enable": {"nvidia-driver": "580-open"},
            "install_profiles": {},
            "remove_profiles": {},
            "reset": [],
            "switch": {},
        },
        "failsafe_changed_files": (
            ["nvidia-driver:580-open:x86_64.yaml"] if applied else []
        ),
        "failsafe_target": {
            "filename": "nvidia-driver:580-open:x86_64.yaml",
            "yaml_sha256": "c" * 64,
        },
        "module_changed_files": ["nvidia-driver.module"] if applied else [],
        "module_failsafe_after_sha256": module_failsafe_after,
        "module_failsafe_before_sha256": module_failsafe_before,
        "module_platform_id": "platform:el9",
        "module_state_after_sha256": module_state_after,
        "module_state_before_sha256": module_state_before,
        "repositories_count": 1,
        "repositories_sha256": "e" * 64,
        "requirements": [],
        "schema": 2,
        "state_after_sha256": _combined_state_digest(
            module_state_after,
            module_failsafe_after,
        ),
        "state_before_sha256": _combined_state_digest(
            module_state_before,
            module_failsafe_before,
        ),
        "target": {"name": "nvidia-driver", "stream": "580-open"},
    }
    payload["preflight_sha256"] = _proof_preflight_sha256(payload)
    return payload


def _dnf_module_preflight_sha256():
    return _dnf_module_proof_payload(applied=False)["preflight_sha256"]


def _local_forward_command(run, snapshot):
    bundle = _package_payload_bundle_from_snapshot(snapshot)
    snapshot_path = Path(snapshot["path"])
    manager = run["package_manager"]
    if manager not in {"dnf", "yum"}:
        return forward_package_command(snapshot_path, bundle, manager)
    forward_payloads = [
        payload for payload in bundle.packages if "forward" in payload.roles
    ]
    return dnf_local_transaction_command(
        apply=True,
        restore_paths=local_payload_paths(
            snapshot_path,
            bundle,
            role="forward",
        ),
        remove_specs=[],
        expected_installs=sorted(
            f"{payload.name}-"
            f"{f'{payload.epoch}:' if payload.epoch else ''}"
            f"{payload.version}.{payload.architecture}"
            for payload in forward_payloads
        ),
        expected_removals=[],
    )


def _report_for_scenario(
    run,
    scenario,
    path,
    snapshot,
    policy_snapshot=None,
):
    if policy_snapshot is None:
        policy_snapshot = snapshot
    fault_findings = {
        "doctor_missing_headers": "kernel.headers.missing",
        "doctor_module_unloaded": "module.not-loaded",
        "doctor_driver_mismatch": "driver.version-mismatch",
        "doctor_runtime_missing": "docker.nvidia-runtime-missing",
        "doctor_fabric_manager_inactive": "fabric-manager.inactive",
    }
    commands = {
        "install_apply": "install",
        "verify_apply": "verify",
        "lock_apply": "lock",
        "snapshot": "snapshot",
        "rollback_apply": "rollback",
        "rollback_state_verify": "rollback",
        "policy_rollback_apply": "rollback",
        "policy_rollback_state_verify": "rollback",
        "plan": "plan",
    }
    command = commands.get(scenario, "doctor")
    mode = "apply" if scenario in {
        "install_apply",
        "verify_apply",
        "lock_apply",
        "snapshot",
        "rollback_apply",
        "rollback_state_verify",
        "policy_rollback_apply",
        "policy_rollback_state_verify",
    } else "dry-run"
    finding_id = fault_findings.get(scenario)
    prestate_drift = scenario in {"plan", "snapshot", "lock_apply"}
    if finding_id:
        findings = [
            {
            "id": finding_id,
            "severity": "error",
            "summary": "injected integration fault",
            "detail": "doctor detected the injected fault",
            "evidence": {},
            "remediation": "Restore the controlled fixture state.",
            }
        ]
    elif command == "rollback":
        findings = []
    elif prestate_drift:
        findings = [
            {
                "id": "container-toolkit.missing",
                "severity": "error",
                "summary": "NVIDIA container toolkit is missing",
                "detail": "controlled integration fixture drift",
                "evidence": {},
                "remediation": f"Install {TOOLKIT_PACKAGE}.",
            },
            {
                "id": "docker.nvidia-runtime-missing",
                "severity": "error",
                "summary": "Docker NVIDIA runtime is not configured",
                "detail": "controlled integration fixture drift",
                "evidence": {},
                "remediation": "Configure the NVIDIA runtime and restart Docker.",
            },
        ]
    else:
        findings = [
            {
                "id": "stack.healthy",
                "severity": "info",
                "summary": "healthy",
                "detail": "integration fixture",
                "evidence": {},
                "remediation": None,
            }
        ]
    command_results = []
    if command in {"install", "lock", "rollback"}:
        command_values = [["true", command]]
        if command == "install":
            command_values = [
                _install_command(run, snapshot),
                ["nvidia-ctk", "runtime", "configure", "--runtime=docker"],
                ["systemctl", "restart", "docker"],
            ]
        if command == "lock":
            command_values = [_lock_command(run, policy_snapshot)]
        if command == "rollback":
            selected_snapshot = (
                policy_snapshot
                if scenario.startswith("policy_rollback_")
                else snapshot
            )
            current_audit = _audit_for_scenario(
                run,
                (
                    "rollback_state_verify"
                    if scenario.startswith("policy_rollback_")
                    else "doctor"
                ),
            )
            command_values = _expected_applied_package_commands(
                selected_snapshot,
                current_audit,
                run["package_manager"],
            )
            if scenario.startswith("policy_rollback_"):
                pre_files = {
                    item["path"]: item
                    for item in policy_snapshot["managed_files"]
                }
                post_files = {
                    item["path"]: item for item in snapshot["managed_files"]
                }
                command_values.extend(
                    ["restore-file", path]
                    for path in sorted(set(pre_files).intersection(post_files))
                    if pre_files[path] != post_files[path]
                )
            else:
                command_values.append(
                    ["restore-file", "/etc/docker/daemon.json"]
                )
            if (
                run["matrix_id"] == "mig-toggle"
                and scenario == "rollback_apply"
            ):
                command_values.extend(
                    [
                        [
                            "nvidia-smi",
                            "mig",
                            "-i",
                            selected_snapshot["gpu_uuids"][0],
                            "-dci",
                        ],
                        [
                            "nvidia-smi",
                            "mig",
                            "-i",
                            selected_snapshot["gpu_uuids"][0],
                            "-dgi",
                        ],
                        [
                            "nvidia-smi",
                            "-i",
                            selected_snapshot["gpu_uuids"][0],
                            "-mig",
                            "0",
                        ],
                    ]
                )
        command_results = [
            {
                "command": command_value,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "skipped": False,
                "reason": None,
            }
            for command_value in command_values
        ]
        if command == "lock" and run["package_manager"] == "dnf":
            command_results[0]["stdout"] = json.dumps(
                _dnf_module_proof_payload(applied=True),
                separators=(",", ":"),
                sort_keys=True,
            )
        if run["matrix_id"] == "mig-toggle" and command == "install":
            command_results.extend(
                [
                    {
                        "command": [
                            "nvidia-smi",
                            "-i",
                            snapshot["gpu_uuids"][0],
                            "-mig",
                            "1",
                        ],
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                        "skipped": False,
                        "reason": None,
                    },
                    {
                        "command": [
                            "nvidia-smi",
                            "mig",
                            "-i",
                            snapshot["gpu_uuids"][0],
                            "-cgi",
                            "0:0",
                            "-C",
                        ],
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                        "skipped": False,
                        "reason": None,
                    },
                ]
            )
    elif command == "verify":
        command_results = [
            {
                "command": [
                    "systemctl",
                    "mask",
                    "--now",
                    "docker.socket",
                ],
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "skipped": False,
                "reason": None,
            }
        ]
    verification = []
    if command == "verify":
        verification_names = {
            "module.load",
            "module.loaded-version",
            "module.on-disk-version",
            "module.provenance",
            "secure-boot.observable",
            "secure-boot.module-signed",
            "secure-boot.on-disk-module-signed",
            "mig.mode",
            "mig.pending-observable",
            "mig.no-pending-transition",
            "mig.geometry-observable",
            "mig.geometry",
            "gpu.open-module-supported",
            "module.open-variant",
            "module.on-disk-open-variant",
            "module.flavor-provenance",
            "kernel.headers",
            "module.compile-or-loadable",
            "nvidia-smi",
            "nvml",
            "cuda-compat.policy",
            "container.cuda-driver-compatibility",
            "container.device-binding",
            "container.gpu",
            "docker.service-trust",
        }
        if run["matrix_id"] == "mig-toggle":
            verification_names.update({"mig.capable", "mig.device-uuid"})
        if run["matrix_id"] == "fabric-manager":
            verification_names.update(
                {
                    "fabric-manager",
                    "fabric-manager.enabled",
                    "fabric-manager.applicable",
                    "fabric-manager.fabric-health",
                    "fabric-manager.service-trust",
                    "fabric-manager.version",
                }
            )
        verification = [
            {"name": name, "ok": True, "command": None, "detail": None}
            for name in sorted(verification_names)
        ]
        container_check = next(
            check for check in verification if check["name"] == "container.gpu"
        )
        desired = load_desired(run["desired"])
        container_check["command"] = {
            "command": [
                "docker",
                "run",
                "--pull=never",
                "--rm",
                "--name",
                "nvidia-converge-verify-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "--label",
                "io.nvidia-converge.verification=true",
                "--label",
                f"io.nvidia-converge.cuda-probe-sha256={_CUDA_DRIVER_PROBE_SHA256}",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=128",
                "--memory=1g",
                "--cpus=1",
                "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=64m,mode=1777",
                "--user=65534:65534",
                "--gpus",
                (
                    "device=MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                    if run["matrix_id"] == "mig-toggle"
                    else "device=GPU-11111111-2222-3333-4444-555555555555"
                ),
                "--env",
                "NVIDIA_DRIVER_CAPABILITIES=compute",
                "--env",
                f"NVIDIA_CONVERGE_PROBE_SHA256={_CUDA_DRIVER_PROBE_SHA256}",
                "--env",
                f"NVIDIA_CONVERGE_EXPECTED_CUDA_VERSION={container_cuda_full_version(desired.container_test_image)}",
                "--interactive",
                "--entrypoint=/bin/bash",
                desired.container_test_image,
                "-ceu",
                _CUDA_DRIVER_PROBE_SCRIPT,
            ],
            "returncode": 0,
            "stdout": (
                "CUDA_DRIVER_API_OK driver_version=13010 device_count=1\n"
            ),
            "stderr": "",
            "skipped": False,
            "reason": None,
        }
    if command == "rollback":
        verification = [
            {"name": name, "ok": True, "command": None, "detail": None}
            for name in sorted(
                {
                    "rollback.kernel",
                    "rollback.module-version",
                    "rollback.packages-restored",
                    "rollback.added-packages-removed",
                    "rollback.managed-files",
                    "rollback.mig-mode",
                    "rollback.mig-geometry",
                    "rollback.docker-service-active",
                    "rollback.docker-service-enabled",
                    "rollback.docker-service-unit-file-state",
                    "rollback.docker-socket-active",
                    "rollback.docker-socket-enabled",
                    "rollback.docker-socket-unit-file-state",
                    "rollback.nvidia-persistenced-active",
                    "rollback.nvidia-persistenced-enabled",
                    "rollback.nvidia-persistenced-unit-file-state",
                    "rollback.fabric-manager-active",
                    "rollback.fabric-manager-enabled",
                    "rollback.fabric-manager-unit-file-state",
                }
            )
        ]
    if command == "lock":
        verification = []
        if run["package_manager"] == "dnf":
            check_result = {
                "command": dnf_module_enable_command(
                    apply=False,
                    stream="580-open",
                ),
                "returncode": 0,
                "stdout": json.dumps(
                    _dnf_module_proof_payload(applied=False),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "stderr": "",
                "skipped": False,
                "reason": None,
            }
            verification.extend(
                [
                    {
                        "name": "packages.policy-preflight",
                        "ok": True,
                        "command": copy.deepcopy(check_result),
                        "detail": None,
                    },
                    {
                        "name": "packages.post-quarantine-policy-preflight",
                        "ok": True,
                        "command": copy.deepcopy(check_result),
                        "detail": None,
                    },
                ]
            )
        verification.append(
            {
                "name": "package-policy.lock",
                "ok": True,
                "command": None,
                "detail": None,
            }
        )
    outcome = "failed" if finding_id else "succeeded"
    desired = asdict(load_desired(run["desired"]))
    if scenario == "doctor_driver_mismatch":
        desired["driver"] = "595-open"
    operation_id = hashlib.sha256(
        f"{run['workflow_job_id']}:{path}".encode()
    ).hexdigest()[:32]
    rollback_evidence = (
        policy_snapshot
        if scenario
        in {
            "lock_apply",
            "policy_rollback_apply",
            "policy_rollback_state_verify",
        }
        else snapshot
        if command in {"install", "verify", "snapshot", "rollback"}
        else None
    )
    if command in {"install", "verify", "lock", "snapshot"} and isinstance(
        rollback_evidence, dict
    ):
        rollback_evidence = copy.deepcopy(rollback_evidence)
        rollback_evidence["operation_id"] = operation_id
        rollback_evidence.pop("integrity_sha256", None)
        canonical = json.dumps(
            rollback_evidence,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        rollback_evidence["integrity_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
    return {
        "schema_version": "1.2",
        "generated_at": run["started_at"],
        "desired": desired,
        "audit": _audit_for_scenario(run, scenario),
        "findings": findings,
        "plan": (
            [
                {
                    "id": {
                        "apt-get": "lock.apt",
                        "dnf": "lock.rpm",
                        "zypper": "lock.zypper",
                    }[run["package_manager"]],
                    "description": "lock driver",
                    "commands": [_lock_command(run, policy_snapshot)],
                    "destructive": True,
                    "reason": "integration fixture",
                }
            ]
            if command == "lock"
            else (
                [
                    {
                        "id": "install.packages",
                        "description": "install controlled missing package",
                        "commands": [
                            _install_command(
                                run,
                                snapshot if command == "install" else None,
                            )
                        ],
                        "destructive": True,
                        "reason": "integration fixture",
                    }
                    ,
                    {
                        "id": "configure.docker-runtime",
                        "description": "configure NVIDIA Docker runtime",
                        "commands": [
                            [
                                "nvidia-ctk",
                                "runtime",
                                "configure",
                                "--runtime=docker",
                            ],
                            ["systemctl", "restart", "docker"],
                        ],
                        "destructive": True,
                        "reason": "integration fixture",
                    },
                ]
                + (
                    [
                        {
                            "id": "enable.mig",
                            "description": "enable MIG",
                            "commands": [
                                [
                                    "nvidia-smi",
                                    "-i",
                                    snapshot["gpu_uuids"][0],
                                    "-mig",
                                    "1",
                                ],
                                [
                                    "nvidia-smi",
                                    "mig",
                                    "-i",
                                    snapshot["gpu_uuids"][0],
                                    "-cgi",
                                    "0:0",
                                    "-C",
                                ],
                            ],
                            "destructive": True,
                            "reason": "integration fixture",
                        }
                    ]
                    if run["matrix_id"] == "mig-toggle"
                    else []
                )
                if command in {"plan", "install"}
                else []
            )
        ),
        "command_results": command_results,
        "verification": verification,
        "rollback": rollback_evidence,
        "sbom": [],
        "command": command,
        "mode": mode,
        "tool_version": "0.1.0",
        "operation_id": operation_id,
        "host_id": snapshot["host_id"],
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
        "duration_seconds": 0,
        "outcome": outcome,
        "exit_code": 2 if finding_id else 0,
        "reboot_required": None,
        "incomplete": False,
        "report_path": (
            "/var/lib/nvidia-converge/reports/"
            f"{run['matrix_id']}-{run['workflow_run_id']}-"
            f"{run['workflow_run_attempt']}-"
            f"{'policy-rollback' if scenario.startswith('policy_rollback_') else command}.json"
            if mode == "apply"
            else f"artifacts/{path}"
        ),
    }


def _audit_for_scenario(run, scenario):
    fabric_manager_required = (
        run["desired"] == "examples/compute-580-open-fabric-manager.yaml"
    )
    headers_installed = scenario != "doctor_missing_headers"
    module_loaded = scenario != "doctor_module_unloaded"
    module_version = "580.126.16"
    restored_prestate = scenario in {
        "plan",
        "snapshot",
        "lock_apply",
        "rollback_apply",
        "rollback_state_verify",
        "policy_rollback_apply",
        "policy_rollback_state_verify",
    }
    toolkit_installed = not restored_prestate
    runtime_usable = toolkit_installed and scenario != "doctor_runtime_missing"
    policy_staged = scenario not in {
        "policy_rollback_apply",
        "policy_rollback_state_verify",
    }
    fabric_active = (
        fabric_manager_required
        and scenario != "doctor_fabric_manager_inactive"
    )
    mig_mode = (
        "enabled"
        if run["desired"] == "examples/compute-580-open-mig.yaml"
        and not restored_prestate
        else "disabled"
    )
    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    mig_enabled = mig_mode == "enabled"
    mig_geometry = (
        [
            {
                "gpu_uuid": gpu_uuid,
                "profile": "7g.80gb",
                "profile_id": 0,
                "placement_start": 0,
                "placement_size": 8,
                "compute_instances": [
                    {"profile": "7c.7g.80gb", "profile_id": 4}
                ],
            }
        ]
        if mig_enabled
        else []
    )
    command_result = {
        "command": ["true"],
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "skipped": False,
        "reason": None,
    }
    return {
        "timestamp": run["started_at"],
        "os_id": run["os_id"],
        "os_version": run["os_version"],
        "package_manager": run["package_manager"],
        "kernel": {
            "running": "6.8.0-integration",
            "headers_installed": headers_installed,
            "compiler": "gcc",
            "secure_boot_enabled": run["matrix_id"] == "secure-boot",
        },
        "module": {
            "loaded": module_loaded,
            "version": module_version if module_loaded else None,
            "installed_version": module_version,
            "open_module": True,
            "installed_open_module": True,
            "signed": True,
            "installed_signed": True,
            "devices": ["/dev/nvidia0"] if module_loaded else [],
        },
        "runtime": {
            "docker_installed": True,
            "nvidia_container_runtime_installed": runtime_usable,
            "docker_gpus_usable": runtime_usable,
        },
        "packages": _audit_packages_for_scenario(
            run,
            policy_staged=policy_staged,
            toolkit_installed=toolkit_installed,
        ),
        "nvidia_smi": command_result,
        "nvml": command_result,
        "fabric_manager_active": fabric_active,
        "fabric_manager_applicable": fabric_manager_required,
        "fabric_manager_healthy": (
            fabric_manager_required and fabric_active
        ),
        "fabric_manager_health_result": command_result,
        "mig_mode": mig_mode,
        "open_kernel_module_supported": True,
        "mig_capable": run["matrix_id"] == "mig-toggle",
        "mig_mode_pending": mig_mode,
        "gpu_uuids": [gpu_uuid],
        "mig_geometry": mig_geometry,
        "mig_device_uuids": (
            ["MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]
            if mig_enabled
            else []
        ),
        "mig_geometry_complete": True,
        "mig_geometry_results": [],
        "cuda_compatibility": [],
        "package_inventory_complete": True,
        "package_inventory_result": command_result,
        "package_policy": {
            "backend": run["package_manager"],
            "observable": True,
            "selectors": _policy_selectors(run) if policy_staged else [],
            "observation": command_result,
        },
        "docker_service_active": True,
        "docker_service_enabled": True,
        "docker_service_unit_file_state": "enabled",
        "docker_socket_active": True,
        "docker_socket_enabled": True,
        "docker_socket_unit_file_state": "enabled",
        "nvidia_persistenced_active": True,
        "nvidia_persistenced_enabled": True,
        "nvidia_persistenced_unit_file_state": "enabled",
        "fabric_manager_enabled": fabric_manager_required,
        "fabric_manager_unit_file_state": (
            "enabled" if fabric_manager_required else "disabled"
        ),
        "fabric_manager_version": "580.126.16" if fabric_manager_required else None,
    }


def _audit_packages_for_scenario(run, *, policy_staged, toolkit_installed):
    packages = copy.deepcopy(
        _snapshot_for_run(run, policy_staged=policy_staged)["packages"]
    )
    if toolkit_installed:
        packages.extend(
            _package_record(run, package, "1.19.1-1")
            for package in TOOLKIT_PACKAGE_CLOSURE
        )
    return sorted(
        packages,
        key=lambda package: (
            package["name"],
            package["architecture"] or "",
            package["epoch"] or "",
            package["version"] or "",
        ),
    )


def _policy_selectors(run):
    manager = run["package_manager"]
    if manager == "apt-get":
        return [
            {
                "identifier": "nvidia-driver-pinning-580",
                "name": "nvidia-driver-pinning-580",
                "kind": "package",
                "relation": None,
                "version": None,
                "repositories": [],
            }
        ]
    if manager == "dnf":
        return [
            {
                "identifier": "nvidia-driver",
                "name": "nvidia-driver",
                "kind": "module",
                "relation": "stream",
                "version": "580-open",
                "repositories": [],
            }
        ]
    return [
        {
            "identifier": "1",
            "name": "*nvidia*",
            "kind": "package",
            "relation": "ge",
            "version": "590",
            "repositories": [],
        }
    ]


def _zip_members(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)
    return output.getvalue()


def _read_zip_members(archive_payload):
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        return {entry.filename: archive.read(entry) for entry in archive.infolist()}


def _append_zip_member(archive_payload, name, payload):
    members = _read_zip_members(archive_payload)
    members[name] = payload
    return _zip_members(members)


def _without_zip_members(archive_payload, removed):
    members = _read_zip_members(archive_payload)
    for name in removed:
        members.pop(name)
    return _zip_members(members)


def _rewrite_json_members(archive_payload, changes):
    members = _read_zip_members(archive_payload)
    for name, transform in changes.items():
        report = json.loads(members[name])
        members[name] = (json.dumps(transform(report), indent=2, sort_keys=True) + "\n").encode()
    return _zip_members(members)


def _rewrite_attested_json_member(archive_payload, name, transform):
    def rewrite(payload):
        document = transform(json.loads(payload))
        return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()

    return _rewrite_attested_member(archive_payload, name, rewrite)


def _with_snapshot_integrity(snapshot):
    snapshot.pop("integrity_sha256", None)
    canonical = json.dumps(
        snapshot,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    snapshot["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return snapshot


def _rewrite_attested_snapshot(archive_payload, name, transform):
    members = _read_zip_members(archive_payload)
    snapshot = transform(json.loads(members[name]))
    members[name] = (
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    ).encode()
    checksum_name = name.removesuffix(".json") + ".sha256"
    snapshot_digest = hashlib.sha256(members[name]).hexdigest()
    members[checksum_name] = f"{snapshot_digest}  {name}\n".encode()

    manifest = json.loads(members["attestation.json"])
    snapshot_entry = next(
        item for item in manifest["files"] if item["path"] == name
    )
    snapshot_entry["sha256"] = snapshot_digest
    checksum_entry = next(
        item for item in manifest["files"] if item["path"] == checksum_name
    )
    checksum_entry["sha256"] = hashlib.sha256(
        members[checksum_name]
    ).hexdigest()
    members["attestation.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return _zip_members(members)


def _rewrite_attested_member(archive_payload, name, transform):
    members = _read_zip_members(archive_payload)
    members[name] = transform(members[name])
    digest = hashlib.sha256(members[name]).hexdigest()
    manifest = json.loads(members["attestation.json"])
    entry = next(item for item in manifest["files"] if item["path"] == name)
    entry["sha256"] = digest
    members["attestation.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return _zip_members(members)
