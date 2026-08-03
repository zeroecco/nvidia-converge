from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import jsonschema  # type: ignore[import-untyped]

from nvidia_converge import __version__
from nvidia_converge.desired import (
    DesiredConfigError,
    container_cuda_full_version,
    load_desired,
)
from nvidia_converge.dnf_module_transaction import dnf_module_enable_command
from nvidia_converge.dnf_transaction import dnf_local_transaction_command
from nvidia_converge.files import BoundedFileError, read_bounded_utf8
from nvidia_converge.models import (
    PackageInfo,
    PackagePayload,
    PackagePayloadBundle,
)
from nvidia_converge.package_payloads import (
    PackagePayloadError,
    forward_package_command,
    local_payload_paths,
)
from nvidia_converge.rollback import RollbackSnapshotError, _rollback_commands
from nvidia_converge.schemas import load_schema, strict_format_checker
from nvidia_converge.verify import (
    _CUDA_DRIVER_PROBE_SCRIPT,
    _CUDA_DRIVER_PROBE_SHA256,
)

REQUIRED_SCENARIOS = {
    "doctor",
    "doctor_missing_headers",
    "doctor_module_unloaded",
    "doctor_driver_mismatch",
    "doctor_runtime_missing",
    "doctor_fabric_manager_inactive",
    "plan",
    "install_apply",
    "verify_apply",
    "lock_apply",
    "snapshot",
    "rollback_apply",
    "rollback_state_verify",
    "policy_rollback_apply",
    "policy_rollback_state_verify",
    "report_schema",
}
MUTATING_SCENARIOS = {
    "install_apply",
    "verify_apply",
    "lock_apply",
    "snapshot",
    "rollback_apply",
    "rollback_state_verify",
    "policy_rollback_apply",
    "policy_rollback_state_verify",
}
OS_PACKAGE_MANAGERS = {
    "ubuntu": {"apt-get"},
    "debian": {"apt-get"},
    "rhel": {"dnf", "yum"},
    "rocky": {"dnf", "yum"},
    "almalinux": {"dnf", "yum"},
    "centos": {"dnf", "yum"},
    "sles": {"zypper"},
    "opensuse-leap": {"zypper"},
}
EXPECTED_REPOSITORY = "zeroecco/nvidia-converge"
EXPECTED_WORKFLOW = ".github/workflows/production-gpu-qualification.yml"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_ENTRIES = 512
MAX_ARTIFACT_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_MEMBER_NAME_BYTES = 512
MAX_JOURNAL_ENTRIES = 4096
MAX_QUALIFICATION_WHEEL_BYTES = 64 * 1024 * 1024
ATTESTATION_SCHEMA_VERSION = "1.3"
ATTESTATION_KIND = "nvidia-converge-integration-attestation"
CUDA_DRIVER_API_ATTESTATION = "[verified:cuda-driver-api]"
DNF_MODULE_PROOF_ATTESTATION_PREFIX = "[verified:dnf-module-proof-v2:"
ATTESTATION_REDACTIONS = [
    "command-output",
    "finding-detail-and-evidence",
    "gpu-device-topology",
    "host-identity",
    "managed-file-content",
    "private-snapshot-path",
    "verification-detail",
]
_QUALIFICATION_WHEEL_NAME = re.compile(
    r"^nvidia_converge-[0-9][0-9A-Za-z.]*-py3-none-any\.whl$"
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_DNF_MODULE_FAILSAFE_PATH = re.compile(
    r"^/var/lib/dnf/modulefailsafe/"
    r"nvidia-driver:[1-9]\d{2,3}-(?:open|dkms):"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}\.yaml$"
)

_APPLY_SCENARIO_COMMANDS = {
    "install_apply": "install",
    "verify_apply": "verify",
    "lock_apply": "lock",
    "snapshot": "snapshot",
    "rollback_apply": "rollback",
    "rollback_state_verify": "rollback",
    "policy_rollback_apply": "rollback",
    "policy_rollback_state_verify": "rollback",
}
_DRY_RUN_SCENARIO_COMMANDS = {
    "doctor": "doctor",
    "doctor_missing_headers": "doctor",
    "doctor_module_unloaded": "doctor",
    "doctor_driver_mismatch": "doctor",
    "doctor_runtime_missing": "doctor",
    "doctor_fabric_manager_inactive": "doctor",
    "plan": "plan",
}
_EXPECTED_FAULT_FINDINGS = {
    "doctor_missing_headers": {"kernel.headers.missing"},
    "doctor_module_unloaded": {"module.not-loaded"},
    "doctor_driver_mismatch": {"driver.version-mismatch"},
    "doctor_runtime_missing": {
        "docker.missing",
        "container-toolkit.missing",
        "docker.nvidia-runtime-missing",
    },
    "doctor_fabric_manager_inactive": {"fabric-manager.inactive"},
}
_FAULT_RESTORATION_REPORTS = {
    "doctor_missing_headers": "restoration/doctor-missing-headers.json",
    "doctor_module_unloaded": "restoration/doctor-module-unloaded.json",
    "doctor_driver_mismatch": "restoration/doctor-driver-mismatch.json",
    "doctor_runtime_missing": "restoration/doctor-runtime-missing.json",
    "doctor_fabric_manager_inactive": "restoration/doctor-fabric-manager-inactive.json",
}
_SCENARIO_REPORT_PATHS = {
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
_CANONICAL_REPORT_PATHS = set(_SCENARIO_REPORT_PATHS.values())
_CANONICAL_JOURNAL_PATHS = {
    f"{path[:-5]}.journal.jsonl" for path in _CANONICAL_REPORT_PATHS
}
_CANONICAL_RESTORATION_PATHS = {
    *_FAULT_RESTORATION_REPORTS.values(),
    "restoration/final-cleanup.json",
}
_CANONICAL_SNAPSHOT_PATHS = {
    "pre/rollback-snapshot.json",
    "pre/rollback-snapshot.sha256",
    "pre/policy-rollback-snapshot.json",
    "pre/policy-rollback-snapshot.sha256",
}
_CANONICAL_ARTIFACT_PATHS = {
    "attestation.json",
    *_CANONICAL_REPORT_PATHS,
    *_CANONICAL_JOURNAL_PATHS,
    *_CANONICAL_RESTORATION_PATHS,
    *_CANONICAL_SNAPSHOT_PATHS,
}
_MUTATION_EVIDENCE_SCENARIOS = {
    "install_apply",
    "lock_apply",
    "rollback_apply",
    "policy_rollback_apply",
}
_REQUIRED_ROLLBACK_CHECKS = {
    "rollback.kernel",
    "rollback.module-version",
    "rollback.packages-restored",
    "rollback.added-packages-removed",
    "rollback.managed-files",
}
_NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE = {
    "nvidia-container-toolkit",
    "nvidia-container-toolkit-base",
    "libnvidia-container-tools",
    "libnvidia-container1",
}
_BASE_VERIFY_CHECKS = {
    "module.load",
    "module.loaded-version",
    "module.on-disk-version",
    "module.provenance",
    "secure-boot.observable",
    "mig.mode",
    "mig.pending-observable",
    "mig.no-pending-transition",
    "kernel.headers",
    "module.compile-or-loadable",
    "nvidia-smi",
    "nvml",
}
_ROLLBACK_MODULE_PROVENANCE = {
    "module_open_module": "open_module",
    "module_signed": "signed",
    "module_installed_version": "installed_version",
    "module_installed_open_module": "installed_open_module",
    "module_installed_signed": "installed_signed",
}
_JOURNAL_EVENT_KEYS = {
    "operation-started": {"event", "operation_id", "timestamp"},
    "rollback-snapshot-persisted": {
        "event",
        "operation_id",
        "timestamp",
        "snapshot_path",
        "snapshot_integrity_sha256",
        "snapshot_operation_id",
        "snapshot_host_id",
    },
    "launcher-release-authorized": {
        "event",
        "operation_id",
        "timestamp",
        "release_target",
        "snapshot_path",
        "snapshot_integrity_sha256",
        "snapshot_operation_id",
        "snapshot_host_id",
    },
    "command-started": {
        "event",
        "operation_id",
        "timestamp",
        "command",
        "mutating",
    },
    "command-finished": {
        "event",
        "operation_id",
        "timestamp",
        "command",
        "mutating",
        "returncode",
        "skipped",
        "reason",
    },
    "operation-completed": {
        "event",
        "operation_id",
        "timestamp",
        "exit_code",
        "incomplete",
        "outcome",
    },
    # The exporter intentionally removes the raw error string from this event.
    "report-persistence-failed": {"event", "operation_id", "timestamp"},
    "operation-recovered": {
        "event",
        "operation_id",
        "timestamp",
        "recovery_operation_id",
        "snapshot_path",
        "snapshot_integrity_sha256",
        "snapshot_operation_id",
        "snapshot_host_id",
    },
}
_MATRIX_PROFILES: dict[str, dict[str, Any]] = {
    "ubuntu-24.04": {
        "labels": {"self-hosted", "nvidia-gpu", "disposable", "ubuntu-24.04"},
        "desired": "examples/compute-580-open.yaml",
        "package_manager": "apt-get",
        "capability": None,
    },
    "rhel-9": {
        "labels": {"self-hosted", "nvidia-gpu", "disposable", "rhel-9"},
        "desired": "examples/compute-580-open.yaml",
        "package_manager": "dnf",
        "capability": None,
    },
    "sles-15": {
        "labels": {"self-hosted", "nvidia-gpu", "disposable", "sles-15"},
        "desired": "examples/compute-580-open.yaml",
        "package_manager": "zypper",
        "capability": None,
    },
    "secure-boot": {
        "labels": {
            "self-hosted",
            "nvidia-gpu",
            "disposable",
            "ubuntu-24.04",
            "secure-boot",
        },
        "desired": "examples/compute-580-open.yaml",
        "package_manager": "apt-get",
        "capability": "secure_boot",
    },
    "fabric-manager": {
        "labels": {
            "self-hosted",
            "nvidia-gpu",
            "disposable",
            "ubuntu-24.04",
            "fabric-manager",
        },
        "desired": "examples/compute-580-open-fabric-manager.yaml",
        "package_manager": "apt-get",
        "capability": "fabric_manager",
    },
    "mig-toggle": {
        "labels": {
            "self-hosted",
            "nvidia-gpu",
            "disposable",
            "ubuntu-24.04",
            "mig-capable",
        },
        "desired": "examples/compute-580-open-mig.yaml",
        "package_manager": "apt-get",
        "capability": "mig_toggle",
    },
}
_SPECIAL_CAPABILITY_LABELS = {
    "secure-boot": "secure_boot",
    "fabric-manager": "fabric_manager",
    "mig-capable": "mig_capable",
}


class ArtifactEvidenceError(ValueError):
    pass


def check_evidence(
    evidence: dict[str, Any],
    *,
    release: str,
    expected_repository: str = EXPECTED_REPOSITORY,
) -> list[str]:
    errors: list[str] = []
    if release != f"v{__version__}":
        errors.append(f"tag {release!r} does not match package version v{__version__}")
    if evidence.get("release") != release:
        errors.append("evidence release does not match the tag")
    if evidence.get("repository") != expected_repository:
        errors.append("evidence repository does not match the release repository")
    tested_commit = evidence.get("commit")
    if not isinstance(tested_commit, str) or re.fullmatch(r"[0-9a-f]{40}", tested_commit) is None:
        errors.append("evidence commit must be the full lowercase SHA of the tested source commit")
    if evidence.get("overall_status") != "passed":
        errors.append("evidence overall_status must be passed")

    qualification_wheel = evidence.get("qualification_wheel")
    expected_wheel_name = f"nvidia_converge-{__version__}-py3-none-any.whl"
    if not isinstance(qualification_wheel, dict):
        errors.append("evidence has no qualification wheel binding")
    else:
        wheel_name = qualification_wheel.get("name")
        wheel_digest = qualification_wheel.get("sha256")
        if wheel_name != expected_wheel_name:
            errors.append(
                "qualification wheel name does not match the release package version"
            )
        if (
            not isinstance(wheel_digest, str)
            or _SHA256.fullmatch(wheel_digest) is None
            or set(wheel_digest) == {"0"}
        ):
            errors.append("qualification wheel has no non-placeholder SHA256 binding")

    runs = evidence.get("runs", [])
    passed_runs = [run for run in runs if run.get("status") == "passed"]
    if len(passed_runs) != len(runs):
        errors.append("every recorded integration run must have passed")

    managers = {run.get("package_manager") for run in passed_runs if run.get("apply") is True}
    if "apt-get" not in managers:
        errors.append("no passed apply run covers apt-get")
    if "dnf" not in managers:
        errors.append("no passed apply run covers the dnf recipe path")
    if "zypper" not in managers:
        errors.append("no passed apply run covers zypper")

    run_ids: set[str] = set()
    workflow_job_ids: set[int] = set()
    matrix_ids: set[str] = set()
    apply_matrix_ids: set[str] = set()
    artifact_ids: set[int] = set()
    for run in passed_runs:
        run_id = str(run.get("id"))
        if run_id in run_ids:
            errors.append(f"duplicate integration run id {run_id!r}")
        run_ids.add(run_id)
        workflow_run_id = run.get("workflow_run_id")
        workflow_job_id = run.get("workflow_job_id")
        if isinstance(workflow_job_id, int):
            if workflow_job_id in workflow_job_ids:
                errors.append(f"duplicate workflow job id {workflow_job_id}")
            workflow_job_ids.add(workflow_job_id)
        matrix_id = str(run.get("matrix_id"))
        if matrix_id in matrix_ids:
            errors.append(f"duplicate integration matrix id {matrix_id!r}")
        matrix_ids.add(matrix_id)
        if run.get("apply") is True:
            apply_matrix_ids.add(matrix_id)
        expected_url = (
            f"https://github.com/{expected_repository}/actions/runs/{workflow_run_id}"
        )
        if run.get("workflow_run") != expected_url:
            errors.append(f"run {run_id!r} workflow_run URL does not match workflow_run_id")
        if run.get("workflow_path") != EXPECTED_WORKFLOW:
            errors.append(f"run {run_id!r} does not identify the trusted GPU workflow")
        if run.get("head_sha") != tested_commit:
            errors.append(f"run {run_id!r} head_sha does not match the tested commit")
        if run.get("qualification_wheel") != qualification_wheel:
            errors.append(
                f"run {run_id!r} qualification wheel does not match the common tested wheel"
            )
        labels = set(run.get("runner_labels", []))
        missing_labels = {"self-hosted", "nvidia-gpu", "disposable"} - labels
        if missing_labels:
            errors.append(
                f"run {run_id!r} is missing trusted runner label(s): {', '.join(sorted(missing_labels))}"
            )
        profile = _MATRIX_PROFILES.get(matrix_id)
        if profile is None:
            errors.append(f"run {run_id!r} has an unknown integration matrix identity")
        else:
            if not profile["labels"].issubset(labels):
                errors.append(
                    f"run {run_id!r} is missing labels required by matrix identity {matrix_id!r}"
                )
            if run.get("desired") != profile["desired"]:
                errors.append(
                    f"run {run_id!r} desired state does not match matrix identity {matrix_id!r}"
                )
            if run.get("package_manager") != profile["package_manager"]:
                errors.append(
                    f"run {run_id!r} package manager does not match matrix identity {matrix_id!r}"
                )
            capabilities = run.get("capabilities", {})
            for capability_label, capability in _SPECIAL_CAPABILITY_LABELS.items():
                if capabilities.get(capability) is (capability_label in labels):
                    continue
                errors.append(
                    f"run {run_id!r} capability {capability!r} does not match its runner labels"
                )
            if not _matrix_os_matches(matrix_id, run):
                errors.append(
                    f"run {run_id!r} OS identity does not match matrix identity {matrix_id!r}"
                )
        os_id = run.get("os_id")
        package_manager = run.get("package_manager")
        if package_manager not in OS_PACKAGE_MANAGERS.get(str(os_id), set()):
            errors.append(
                f"run {run_id!r} has unsupported OS/package-manager mapping {os_id!r}/{package_manager!r}"
            )
        if not _valid_os_version(str(os_id), str(run.get("os_version"))):
            errors.append(f"run {run_id!r} has unsupported OS version metadata")
        try:
            started = datetime.fromisoformat(str(run.get("started_at")).replace("Z", "+00:00"))
            completed = datetime.fromisoformat(str(run.get("completed_at")).replace("Z", "+00:00"))
            if completed < started:
                errors.append(f"run {run_id!r} completed before it started")
        except ValueError:
            errors.append(f"run {run_id!r} has invalid timestamps")

        scenarios = run.get("scenarios", [])
        scenario_names = [str(scenario.get("name")) for scenario in scenarios]
        duplicates = sorted(
            name for name in set(scenario_names) if scenario_names.count(name) > 1
        )
        if duplicates:
            errors.append(f"run {run_id!r} has duplicate scenarios: {', '.join(duplicates)}")
        scenario_status = {
            str(scenario.get("name")): scenario.get("status") for scenario in scenarios
        }
        if run.get("apply") is True:
            required_scenarios = set(REQUIRED_SCENARIOS)
            if matrix_id != "fabric-manager":
                required_scenarios.remove("doctor_fabric_manager_inactive")
            failed_required = sorted(
                name
                for name in required_scenarios
                if scenario_status.get(name) != "passed"
            )
            if failed_required:
                errors.append(
                    f"apply run {run_id!r} did not pass required scenarios: {', '.join(failed_required)}"
                )
            if (
                matrix_id != "fabric-manager"
                and scenario_status.get("doctor_fabric_manager_inactive")
                != "not_applicable"
            ):
                errors.append(
                    f"apply run {run_id!r} must mark the Fabric Manager fault scenario not_applicable"
                )
        else:
            claimed_mutations = sorted(
                name
                for name in MUTATING_SCENARIOS
                if scenario_status.get(name) == "passed"
            )
            if claimed_mutations:
                errors.append(
                    f"non-apply run {run_id!r} claims mutating scenarios: {', '.join(claimed_mutations)}"
                )
        for scenario in scenarios:
            if scenario.get("status") == "passed":
                report = scenario.get("report")
                if not _safe_report_path(report):
                    errors.append(
                        f"passed scenario {scenario.get('name')!r} in run {run_id!r} has no safe JSON report path"
                    )
                expected_report = _SCENARIO_REPORT_PATHS.get(
                    str(scenario.get("name"))
                )
                if report != expected_report:
                    errors.append(
                        f"passed scenario {scenario.get('name')!r} in run {run_id!r} "
                        "does not use its canonical sanitized report path"
                    )
        if run.get("apply") is True:
            artifacts = run.get("artifacts", [])
            if not artifacts:
                errors.append(f"passed apply run {run.get('id')!r} has no retained artifacts")
            elif len(artifacts) != 1:
                errors.append(
                    f"passed apply run {run.get('id')!r} must identify exactly one matrix artifact"
                )
            for artifact in artifacts:
                digest = str(artifact.get("sha256", ""))
                if not digest or set(digest) == {"0"}:
                    errors.append(f"artifact {artifact.get('name')!r} in run {run_id!r} has a placeholder digest")
                artifact_id = artifact.get("id")
                if isinstance(artifact_id, int):
                    if artifact_id in artifact_ids:
                        errors.append(f"duplicate GitHub artifact id {artifact_id}")
                    artifact_ids.add(artifact_id)
                expected_artifact_uri = (
                    f"https://api.github.com/repos/{expected_repository}/actions/artifacts/{artifact_id}/zip"
                )
                if artifact.get("uri") != expected_artifact_uri:
                    errors.append(
                        f"artifact {artifact.get('name')!r} in run {run_id!r} has an untrusted URI"
                    )
                expected_artifact_name = (
                    f"gpu-integration-{matrix_id}-{workflow_run_id}-"
                    f"{run.get('workflow_run_attempt')}"
                )
                if artifact.get("name") != expected_artifact_name:
                    errors.append(
                        f"artifact {artifact.get('name')!r} in run {run_id!r} does not match its matrix job"
                    )

    missing_matrix_jobs = sorted(set(_MATRIX_PROFILES) - apply_matrix_ids)
    if missing_matrix_jobs:
        errors.append(
            "required applied integration matrix jobs are missing: "
            + ", ".join(missing_matrix_jobs)
        )

    qualified_platforms = _derive_qualified_platforms(passed_runs)
    if evidence.get("qualified_platforms") != qualified_platforms:
        errors.append(
            "qualified_platforms is not the canonical exact tuple set derived "
            "from passed applied runs"
        )

    recipe_path_coverage = evidence.get("recipe_path_coverage", {})
    derived_recipe_path_coverage = _derive_recipe_path_coverage(passed_runs)
    inconsistent_recipe_paths = sorted(
        name
        for name, expected in derived_recipe_path_coverage.items()
        if recipe_path_coverage.get(name) is not expected
    )
    if inconsistent_recipe_paths:
        errors.append(
            "recipe-path coverage is not supported by run observations: "
            + ", ".join(inconsistent_recipe_paths)
        )
    missing_recipe_paths = sorted(
        name
        for name, covered in derived_recipe_path_coverage.items()
        if covered is not True
    )
    if missing_recipe_paths:
        errors.append(
            "required recipe-path coverage is incomplete: "
            + ", ".join(missing_recipe_paths)
        )

    coverage = evidence.get("required_coverage", {})
    derived_coverage = _derive_coverage(passed_runs)
    inconsistent = sorted(
        name
        for name, expected in derived_coverage.items()
        if coverage.get(name) is not expected
    )
    if inconsistent:
        errors.append(
            "required coverage is not supported by run observations: "
            + ", ".join(inconsistent)
        )
    missing_coverage = sorted(
        name for name, covered in derived_coverage.items() if covered is not True
    )
    if missing_coverage:
        errors.append(f"required coverage is incomplete: {', '.join(missing_coverage)}")
    return errors


def _matrix_os_matches(matrix_id: str, run: dict[str, Any]) -> bool:
    os_id = run.get("os_id")
    version = str(run.get("os_version", ""))
    if matrix_id == "rhel-9":
        return os_id in {"rhel", "rocky", "almalinux", "centos"} and version.startswith(
            "9"
        )
    if matrix_id == "sles-15":
        return os_id in {"sles", "opensuse-leap"} and version.startswith("15")
    return os_id == "ubuntu" and version == "24.04"


def _safe_report_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) >= 2
        and path.parts[0] == "reports"
        and path.suffix == ".json"
    )


def _valid_os_version(os_id: str, version: str) -> bool:
    major = version.split(".", 1)[0]
    if os_id == "ubuntu":
        return version in {"22.04", "24.04", "26.04"}
    if os_id == "debian":
        return major in {"12", "13"}
    if os_id in {"rhel", "rocky", "almalinux", "centos"}:
        return major in {"8", "9"}
    if os_id in {"sles", "opensuse-leap"}:
        return major in {"15", "16"}
    return False


def _derive_coverage(runs: list[dict[str, Any]]) -> dict[str, bool]:
    apply_runs = [run for run in runs if run.get("apply") is True]
    capabilities = [run.get("capabilities", {}) for run in apply_runs]
    scenarios = {
        str(scenario.get("name"))
        for run in apply_runs
        for scenario in run.get("scenarios", [])
        if scenario.get("status") == "passed"
    }
    return {
        "secure_boot": any(capability.get("secure_boot") is True for capability in capabilities),
        "fabric_manager": any(
            capability.get("fabric_manager") is True for capability in capabilities
        ),
        "mig_toggle": any(capability.get("mig_toggle") is True for capability in capabilities),
        "install_apply": "install_apply" in scenarios,
        "rollback_apply": "rollback_apply" in scenarios
        and "rollback_state_verify" in scenarios,
        "policy_rollback_apply": "policy_rollback_apply" in scenarios,
        "policy_rollback_state_verify": "policy_rollback_state_verify" in scenarios,
    }


def _derive_qualified_platforms(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    tuples = {
        (
            str(run.get("os_id")),
            str(run.get("os_version")),
            str(run.get("package_manager")),
        )
        for run in runs
        if run.get("apply") is True
    }
    return [
        {
            "os_id": os_id,
            "os_version": os_version,
            "package_manager": package_manager,
        }
        for os_id, os_version, package_manager in sorted(tuples)
    ]


def _derive_recipe_path_coverage(runs: list[dict[str, Any]]) -> dict[str, bool]:
    managers = {
        run.get("package_manager") for run in runs if run.get("apply") is True
    }
    return {
        "apt_get": "apt-get" in managers,
        "dnf": "dnf" in managers,
        "zypper": "zypper" in managers,
    }


def check_qualification_wheel_binding(
    evidence: dict[str, Any], wheel_path: Path
) -> list[str]:
    """Bind release wheel bytes to the exact wheel exercised on every GPU job."""

    binding = evidence.get("qualification_wheel")
    if not isinstance(binding, dict):
        return ["evidence has no qualification wheel binding"]
    name = binding.get("name")
    digest = binding.get("sha256")
    if (
        not isinstance(name, str)
        or _QUALIFICATION_WHEEL_NAME.fullmatch(name) is None
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or set(digest) == {"0"}
    ):
        return ["evidence qualification wheel binding is invalid"]
    mismatched_runs = sorted(
        str(run.get("id"))
        for run in evidence.get("runs", [])
        if not isinstance(run, dict) or run.get("qualification_wheel") != binding
    )
    if mismatched_runs:
        return [
            "integration runs do not share the common qualification wheel: "
            + ", ".join(mismatched_runs)
        ]
    if wheel_path.name != name:
        return [
            f"release wheel name {wheel_path.name!r} does not match qualified wheel {name!r}"
        ]

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(wheel_path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                return ["qualification-bound release wheel is not a regular file"]
            if not 0 < metadata.st_size <= MAX_QUALIFICATION_WHEEL_BYTES:
                return ["qualification-bound release wheel size is outside the safety limit"]
            hasher = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                hasher.update(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        return [f"cannot read qualification-bound release wheel: {exc}"]
    if not hmac.compare_digest(hasher.hexdigest(), digest):
        return [
            "release wheel SHA256 does not match the wheel exercised by GPU qualification"
        ]
    return []


def check_commit_provenance(tested_commit: str, release_commit: str, evidence_path: Path) -> list[str]:
    if re.fullmatch(r"[0-9a-f]{40}", release_commit) is None:
        return ["release commit must be a full lowercase Git SHA"]
    try:
        root_result = subprocess.run(
            ["git", "-C", str(evidence_path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(root_result.stdout.strip()).resolve()
        relative_evidence = evidence_path.resolve().relative_to(root)
        ancestor = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", tested_commit, release_commit],
            check=False,
        )
        changes = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", tested_commit, release_commit, "--"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return [f"cannot verify evidence commit provenance: {exc}"]
    errors: list[str] = []
    if ancestor.returncode != 0:
        errors.append("tested evidence commit is not an ancestor of the release commit")
    changed_paths = {Path(line) for line in changes.stdout.splitlines() if line}
    unexpected = sorted(str(path) for path in changed_paths - {relative_evidence})
    if unexpected:
        errors.append(f"source changed after integration testing: {', '.join(unexpected)}")
    return errors


def verify_github_evidence(
    evidence: dict[str, Any],
    *,
    token: str,
    api_url: str = "https://api.github.com",
    verified_artifacts: dict[int, bytes] | None = None,
) -> list[str]:
    if verified_artifacts is not None:
        verified_artifacts.clear()
    if not token:
        return ["GITHUB_TOKEN is required to verify workflow and artifact provenance"]
    if api_url != "https://api.github.com":
        return [
            "GitHub API URL must be exactly https://api.github.com for this release schema"
        ]
    repository = str(evidence.get("repository", ""))
    errors: list[str] = []
    trusted_runs: dict[int, Any] = {}
    artifact_pages: dict[int, Any] = {}
    job_pages: dict[tuple[int, int], Any] = {}
    seen_operation_ids: dict[str, str] = {}
    downloaded_artifacts: dict[int, bytes] = {}
    for run in evidence.get("runs", []):
        run_id = run.get("workflow_run_id")
        run_attempt = run.get("workflow_run_attempt")
        label = run.get("id")
        if not isinstance(run_id, int) or not isinstance(run_attempt, int):
            errors.append(f"run {label!r} has invalid GitHub run identity")
            continue
        try:
            if run_id not in trusted_runs:
                trusted_runs[run_id] = _github_json(
                    f"{api_url}/repos/{repository}/actions/runs/{run_id}", token
                )
                artifact_pages[run_id] = _github_json(
                    f"{api_url}/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
                    token,
                )
            job_key = (run_id, run_attempt)
            if job_key not in job_pages:
                job_pages[job_key] = _github_json(
                    f"{api_url}/repos/{repository}/actions/runs/{run_id}/attempts/"
                    f"{run_attempt}/jobs?per_page=100",
                    token,
                )
            trusted_run = trusted_runs[run_id]
            artifact_page = artifact_pages[run_id]
            job_page = job_pages[job_key]
        except RuntimeError as exc:
            errors.append(f"cannot verify GitHub run {label!r}: {exc}")
            continue
        if (
            not isinstance(trusted_run, dict)
            or not isinstance(artifact_page, dict)
            or not isinstance(job_page, dict)
        ):
            errors.append(f"GitHub returned invalid provenance metadata for run {label!r}")
            continue
        trusted_repository = trusted_run.get("repository", {})
        trusted_head_repository = trusted_run.get("head_repository", {})
        expected_run_values = {
            "id": run_id,
            "run_attempt": run.get("workflow_run_attempt"),
            "head_branch": "main",
            "head_sha": run.get("head_sha"),
            "path": run.get("workflow_path"),
            "html_url": run.get("workflow_run"),
            "event": "repository_dispatch",
            "conclusion": "success",
        }
        mismatches = sorted(
            key
            for key, expected in expected_run_values.items()
            if trusted_run.get(key) != expected
        )
        if trusted_repository.get("full_name") != repository:
            mismatches.append("repository")
        if trusted_head_repository.get("full_name") != repository:
            mismatches.append("head_repository")
        if mismatches:
            errors.append(
                f"GitHub run {label!r} metadata mismatch: {', '.join(sorted(set(mismatches)))}"
            )
        errors.extend(_verify_github_job(run, job_page))
        artifacts = artifact_page.get("artifacts", [])
        total_count = artifact_page.get("total_count")
        if (
            not isinstance(artifacts, list)
            or not isinstance(total_count, int)
            or total_count != len(artifacts)
            or total_count > 100
        ):
            errors.append(f"GitHub artifact metadata is invalid for run {label!r}")
            continue
        trusted_artifacts = {
            artifact.get("id"): artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
        }
        retained_contents: list[tuple[str, bytes]] = []
        for claimed in run.get("artifacts", []):
            artifact_id = claimed.get("id")
            trusted = trusted_artifacts.get(artifact_id)
            if not isinstance(trusted, dict):
                errors.append(
                    f"artifact {claimed.get('name')!r} in run {label!r} was not retained by GitHub"
                )
                continue
            expected_digest = f"sha256:{claimed.get('sha256')}"
            artifact_mismatches = []
            if trusted.get("name") != claimed.get("name"):
                artifact_mismatches.append("name")
            if trusted.get("archive_download_url") != claimed.get("uri"):
                artifact_mismatches.append("uri")
            if trusted.get("digest") != expected_digest:
                artifact_mismatches.append("sha256")
            if trusted.get("expired") is not False:
                artifact_mismatches.append("expired")
            size_in_bytes = trusted.get("size_in_bytes")
            if (
                not isinstance(size_in_bytes, int)
                or isinstance(size_in_bytes, bool)
                or not 0 < size_in_bytes <= MAX_ARTIFACT_ARCHIVE_BYTES
            ):
                artifact_mismatches.append("size_in_bytes")
            trusted_workflow = trusted.get("workflow_run", {})
            if trusted_workflow.get("id") != run_id:
                artifact_mismatches.append("workflow_run_id")
            if trusted_workflow.get("head_sha") != run.get("head_sha"):
                artifact_mismatches.append("head_sha")
            if artifact_mismatches:
                errors.append(
                    f"artifact {claimed.get('name')!r} in run {label!r} metadata mismatch: "
                    + ", ".join(artifact_mismatches)
                )
                continue
            try:
                archive = _github_bytes(str(trusted["archive_download_url"]), token)
            except RuntimeError as exc:
                errors.append(
                    f"cannot download artifact {claimed.get('name')!r} in run {label!r}: {exc}"
                )
                continue
            actual_digest = hashlib.sha256(archive).hexdigest()
            if not hmac.compare_digest(actual_digest, str(claimed.get("sha256"))):
                errors.append(
                    f"artifact {claimed.get('name')!r} in run {label!r} downloaded ZIP digest mismatch"
                )
                continue
            if isinstance(artifact_id, int) and not isinstance(artifact_id, bool):
                downloaded_artifacts[artifact_id] = archive
            retained_contents.append((str(claimed.get("name")), archive))
        errors.extend(
            _verify_run_artifact_contents(
                run,
                retained_contents,
                seen_operation_ids=seen_operation_ids,
            )
        )
    if not errors and verified_artifacts is not None:
        verified_artifacts.update(downloaded_artifacts)
    return errors


def _verify_github_job(run: dict[str, Any], job_page: dict[str, Any]) -> list[str]:
    label = str(run.get("id"))
    jobs = job_page.get("jobs")
    total_count = job_page.get("total_count")
    if (
        not isinstance(jobs, list)
        or not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count != len(jobs)
        or total_count > 100
    ):
        return [f"GitHub workflow job metadata is invalid for run {label!r}"]
    workflow_job_id = run.get("workflow_job_id")
    matching = [
        job
        for job in jobs
        if isinstance(job, dict) and job.get("id") == workflow_job_id
    ]
    if len(matching) != 1:
        return [f"GitHub workflow job {workflow_job_id!r} was not found for run {label!r}"]
    job = matching[0]
    expected = {
        "run_id": run.get("workflow_run_id"),
        "head_sha": run.get("head_sha"),
        "html_url": run.get("workflow_job"),
        "name": run.get("matrix_id"),
        "status": "completed",
        "conclusion": "success",
        "workflow_name": "GPU Integration",
    }
    mismatches = [
        field for field, value in expected.items() if job.get(field) != value
    ]
    trusted_labels = job.get("labels")
    claimed_labels = run.get("runner_labels")
    if (
        not isinstance(trusted_labels, list)
        or len(trusted_labels) != len(set(trusted_labels))
        or set(trusted_labels) != set(claimed_labels or [])
    ):
        mismatches.append("runner_labels")
    if not isinstance(job.get("runner_id"), int) or job.get("runner_id", 0) <= 0:
        mismatches.append("runner_id")
    if not isinstance(job.get("runner_name"), str) or not job.get("runner_name"):
        mismatches.append("runner_name")
    try:
        if _parse_datetime(str(job.get("started_at"))) != _parse_datetime(
            str(run.get("started_at"))
        ):
            mismatches.append("started_at")
        if _parse_datetime(str(job.get("completed_at"))) != _parse_datetime(
            str(run.get("completed_at"))
        ):
            mismatches.append("completed_at")
    except ValueError:
        mismatches.append("timestamps")
    if mismatches:
        return [
            f"GitHub workflow job metadata mismatch for run {label!r}: "
            + ", ".join(sorted(set(mismatches)))
        ]
    return []


def _verify_run_artifact_contents(
    run: dict[str, Any],
    archives: list[tuple[str, bytes]],
    *,
    seen_operation_ids: dict[str, str] | None = None,
) -> list[str]:
    label = str(run.get("id"))
    errors: list[str] = []
    members: dict[str, bytes] = {}
    for artifact_name, archive in archives:
        try:
            artifact_members = _artifact_evidence_members(archive)
        except ArtifactEvidenceError as exc:
            errors.append(
                f"artifact {artifact_name!r} in run {label!r} has invalid ZIP contents: {exc}"
            )
            continue
        duplicates = sorted(set(members).intersection(artifact_members))
        if duplicates:
            errors.append(
                f"run {label!r} has duplicate retained evidence paths across artifacts: "
                + ", ".join(duplicates)
            )
        for name, payload in artifact_members.items():
            members.setdefault(name, payload)

    errors.extend(_verify_attestation_manifest(run, members))

    passed_scenarios = [
        scenario
        for scenario in run.get("scenarios", [])
        if scenario.get("status") == "passed"
    ]
    report_paths = {
        str(scenario.get("report"))
        for scenario in passed_scenarios
        if _safe_report_path(scenario.get("report"))
    }
    retained_report_paths = {
        name
        for name in members
        if name.startswith("reports/") and name.endswith(".json")
    }
    missing_reports = sorted(report_paths - retained_report_paths)
    if missing_reports:
        errors.append(
            f"run {label!r} retained artifact is missing scenario reports: "
            + ", ".join(missing_reports)
        )
    unclaimed_reports = sorted(retained_report_paths - report_paths)
    if unclaimed_reports:
        errors.append(
            f"run {label!r} retained artifact has unclaimed reports: "
            + ", ".join(unclaimed_reports)
        )
    expected_journal_paths = {
        f"{report_path[:-5]}.journal.jsonl"
        for scenario in passed_scenarios
        if scenario.get("name") in _APPLY_SCENARIO_COMMANDS
        and isinstance((report_path := scenario.get("report")), str)
        and report_path.endswith(".json")
    }
    retained_journal_paths = {
        name
        for name in members
        if name.startswith("reports/") and name.endswith(".journal.jsonl")
    }
    missing_journals = sorted(expected_journal_paths - retained_journal_paths)
    if missing_journals:
        errors.append(
            f"run {label!r} retained artifact is missing command journals: "
            + ", ".join(missing_journals)
        )
    unclaimed_journals = sorted(retained_journal_paths - expected_journal_paths)
    if unclaimed_journals:
        errors.append(
            f"run {label!r} retained artifact has unclaimed command journals: "
            + ", ".join(unclaimed_journals)
        )
    passed_fault_scenarios = {
        str(scenario.get("name"))
        for scenario in passed_scenarios
        if scenario.get("name") in _FAULT_RESTORATION_REPORTS
    }
    expected_restoration_paths = {
        _FAULT_RESTORATION_REPORTS[name] for name in passed_fault_scenarios
    }
    if run.get("apply") is True:
        expected_restoration_paths.add("restoration/final-cleanup.json")
    retained_restoration_paths = {
        name
        for name in members
        if name.startswith("restoration/") and name.endswith(".json")
    }
    missing_restoration = sorted(
        expected_restoration_paths - retained_restoration_paths
    )
    if missing_restoration:
        errors.append(
            f"run {label!r} retained artifact is missing fault-restoration reports: "
            + ", ".join(missing_restoration)
        )
    unclaimed_restoration = sorted(
        retained_restoration_paths - expected_restoration_paths
    )
    if unclaimed_restoration:
        errors.append(
            f"run {label!r} retained artifact has unclaimed fault-restoration reports: "
            + ", ".join(unclaimed_restoration)
        )

    report_schema = load_schema("report")
    expected_desired: dict[str, Any] | None = None
    desired_path = REPOSITORY_ROOT / str(run.get("desired", ""))
    try:
        resolved_desired = desired_path.resolve(strict=True)
        resolved_desired.relative_to(REPOSITORY_ROOT)
        expected_desired = asdict(load_desired(str(resolved_desired)))
    except (OSError, ValueError, DesiredConfigError) as exc:
        errors.append(
            f"run {label!r} does not reference a trusted checked-in desired state: {exc}"
        )
    scenario_names_by_path: dict[str, set[str]] = {}
    for scenario in passed_scenarios:
        report_path = scenario.get("report")
        if isinstance(report_path, str):
            scenario_names_by_path.setdefault(report_path, set()).add(
                str(scenario.get("name"))
            )
    reports: dict[str, dict[str, Any]] = {}
    retained_json_paths = retained_report_paths | retained_restoration_paths
    for path in sorted(retained_json_paths):
        try:
            document = _strict_json_document(members[path], path)
            jsonschema.validate(
                document,
                report_schema,
                format_checker=strict_format_checker(),
            )
        except (ArtifactEvidenceError, jsonschema.ValidationError) as exc:
            errors.append(f"run {label!r} retained report {path!r} is invalid: {exc}")
            continue
        if not isinstance(document, dict):
            errors.append(f"run {label!r} retained report {path!r} is not a JSON object")
            continue
        reports[path] = document
        operation_id = str(document.get("operation_id"))
        operation_owner = f"{label}:{path}"
        if seen_operation_ids is not None:
            previous_owner = seen_operation_ids.setdefault(
                operation_id, operation_owner
            )
            if previous_owner != operation_owner:
                errors.append(
                    f"operation ID {operation_id!r} is reused by {previous_owner!r} and {operation_owner!r}"
                )
        errors.extend(
            _verify_report_run_binding(
                run,
                path,
                document,
                expected_desired,
                allow_driver_fault_desired=(
                    scenario_names_by_path.get(path) == {"doctor_driver_mismatch"}
                ),
            )
        )
        errors.extend(_verify_embedded_snapshot_integrity(document, label, path))

    run_host_ids = {
        report.get("host_id")
        for report in reports.values()
        if isinstance(report.get("host_id"), str) and report.get("host_id")
    }
    if reports and (
        len(run_host_ids) != 1
        or any(not report.get("host_id") for report in reports.values())
    ):
        errors.append(f"run {label!r} retained reports do not bind one observable host identity")
    run_gpu_bindings = {
        tuple(report.get("audit", {}).get("gpu_uuids", []))
        for report in reports.values()
        if isinstance(report.get("audit"), dict)
    }
    if reports and (
        len(run_gpu_bindings) != 1
        or any(not binding for binding in run_gpu_bindings)
    ):
        errors.append(
            f"run {label!r} retained reports do not bind one observable GPU inventory"
        )

    journal_checks: set[str] = set()
    for scenario in passed_scenarios:
        scenario_name = str(scenario.get("name"))
        report_path = scenario.get("report")
        if not isinstance(report_path, str) or report_path not in reports:
            continue
        report = reports[report_path]
        errors.extend(
            _verify_scenario_report(
                run,
                scenario_name,
                report_path,
                report,
            )
        )
        if scenario_name in _APPLY_SCENARIO_COMMANDS and report_path not in journal_checks:
            journal_checks.add(report_path)
            journal_path = f"{report_path[:-5]}.journal.jsonl"
            journal = members.get(journal_path)
            if journal is None:
                errors.append(
                    f"run {label!r} applied report {report_path!r} has no retained command journal"
                )
            else:
                errors.extend(
                    _verify_report_journal(
                        label,
                        journal_path,
                        journal,
                        report,
                        require_mutation=any(
                            item.get("name") in _MUTATION_EVIDENCE_SCENARIOS
                            and item.get("report") == report_path
                            and item.get("status") == "passed"
                            for item in passed_scenarios
                        ),
                        required_mutation_commands=(
                            _planned_install_mutations(report)
                            if scenario_name == "install_apply"
                            else (
                                _planned_lock_mutations(report)
                                if scenario_name == "lock_apply"
                                else None
                            )
                        ),
                    )
                )

    errors.extend(_verify_scenario_chronology(run, passed_scenarios, reports))
    errors.extend(
        _verify_fault_restoration(
            run,
            passed_scenarios,
            reports,
            expected_desired,
        )
    )
    errors.extend(_verify_matrix_capability_evidence(run, passed_scenarios, reports))
    errors.extend(
        _verify_controlled_convergence(
            run,
            passed_scenarios,
            reports,
            expected_desired,
        )
    )
    if run.get("apply") is True:
        errors.extend(
            _verify_retained_snapshot(
                run,
                members,
                reports,
                expected_desired,
            )
        )
    return errors


def _verify_attestation_manifest(
    run: dict[str, Any], members: dict[str, bytes]
) -> list[str]:
    label = str(run.get("id"))
    payload = members.get("attestation.json")
    if payload is None:
        return [f"run {label!r} artifact has no sanitized attestation manifest"]
    try:
        manifest = _strict_json_document(payload, "attestation.json")
    except ArtifactEvidenceError as exc:
        return [f"run {label!r} attestation manifest is invalid: {exc}"]
    if not isinstance(manifest, dict):
        return [f"run {label!r} attestation manifest is not a JSON object"]
    errors: list[str] = []
    expected_keys = {
        "schema_version",
        "kind",
        "matrix_id",
        "workflow_run_id",
        "workflow_run_attempt",
        "qualification_wheel",
        "redactions",
        "files",
    }
    if set(manifest) != expected_keys:
        errors.append(f"run {label!r} attestation manifest has unexpected fields")
    expected_values = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "matrix_id": run.get("matrix_id"),
        "workflow_run_id": run.get("workflow_run_id"),
        "workflow_run_attempt": run.get("workflow_run_attempt"),
        "qualification_wheel": run.get("qualification_wheel"),
        "redactions": ATTESTATION_REDACTIONS,
    }
    mismatches = sorted(
        key for key, value in expected_values.items() if manifest.get(key) != value
    )
    if mismatches:
        errors.append(
            f"run {label!r} attestation context mismatch: "
            + ", ".join(mismatches)
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        return [*errors, f"run {label!r} attestation file inventory is invalid"]
    indexed: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "source_path",
            "sha256",
        }:
            errors.append(f"run {label!r} attestation has an invalid file entry")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or path in indexed:
            errors.append(
                f"run {label!r} attestation has a duplicate or invalid file path"
            )
            continue
        indexed[path] = entry
    expected_paths = set(members) - {"attestation.json"}
    if set(indexed) != expected_paths:
        errors.append(
            f"run {label!r} attestation inventory does not exactly cover its artifact"
        )
    for path, entry in indexed.items():
        content = members.get(path)
        expected_source_path = (
            path.removesuffix(".sha256") + ".json"
            if path
            in {
                "pre/rollback-snapshot.sha256",
                "pre/policy-rollback-snapshot.sha256",
            }
            else path
        )
        if entry.get("source_path") != expected_source_path:
            errors.append(
                f"run {label!r} attestation source mapping is invalid for {path!r}"
            )
        exported_digest = entry.get("sha256")
        if content is None or exported_digest != hashlib.sha256(content).hexdigest():
            errors.append(
                f"run {label!r} attestation digest mismatch for {path!r}"
            )
    errors.extend(_verify_sanitized_members(label, members))
    return errors


def _verify_sanitized_members(label: str, members: dict[str, bytes]) -> list[str]:
    errors: list[str] = []
    for path, payload in members.items():
        if path == "attestation.json" or path.endswith(".sha256"):
            continue
        if re.search(
            rb"(?:GPU|MIG)-[A-Za-z0-9][A-Za-z0-9/.-]{15,}", payload
        ) is not None:
            errors.append(
                f"run {label!r} retained evidence {path!r} exposes raw GPU identity"
            )
        if path.endswith(".journal.jsonl"):
            _, journal_errors = _load_sanitized_journal(label, path, payload)
            errors.extend(journal_errors)
            continue
        try:
            document = _strict_json_document(payload, path)
        except ArtifactEvidenceError:
            continue
        if not isinstance(document, dict):
            continue
        if path in {
            "pre/rollback-snapshot.json",
            "pre/policy-rollback-snapshot.json",
        }:
            errors.extend(_verify_sanitized_snapshot(label, path, document))
            continue
        host_id = document.get("host_id")
        if not isinstance(host_id, str) or re.fullmatch(
            r"attested:[a-f0-9]{64}", host_id
        ) is None:
            errors.append(f"run {label!r} retained report {path!r} exposes host identity")
        audit = document.get("audit")
        if isinstance(audit, dict):
            if not _has_sanitized_gpu_uuids(audit.get("gpu_uuids")):
                errors.append(
                    f"run {label!r} retained report {path!r} exposes raw GPU identity"
                )
            if not _has_sanitized_optional_gpu_uuids(
                audit.get("mig_device_uuids")
            ) or not _has_sanitized_mig_geometry(
                audit.get("mig_geometry"), audit.get("gpu_uuids")
            ):
                errors.append(
                    f"run {label!r} retained report {path!r} exposes raw MIG identity"
                )
            module = audit.get("module")
            if not isinstance(module, dict) or module.get("devices") != []:
                errors.append(
                    f"run {label!r} retained report {path!r} exposes GPU topology"
                )
            for key in (
                "nvidia_smi",
                "nvml",
                "package_inventory_result",
                "fabric_manager_health_result",
            ):
                errors.extend(
                    _verify_sanitized_command_result(label, path, audit.get(key))
                )
            for result in audit.get("mig_geometry_results") or []:
                errors.extend(
                    _verify_sanitized_command_result(label, path, result)
                )
            package_policy = audit.get("package_policy")
            if isinstance(package_policy, dict):
                errors.extend(
                    _verify_sanitized_command_result(
                        label, path, package_policy.get("observation")
                    )
                )
            for compatibility in audit.get("cuda_compatibility") or []:
                if isinstance(compatibility, dict):
                    errors.extend(
                        _verify_sanitized_command_result(
                            label, path, compatibility.get("library_probe")
                        )
                    )
        for finding in document.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            if (
                finding.get("detail") != "[redacted]"
                or finding.get("evidence") != {}
                or finding.get("remediation") not in {None, "[redacted]"}
            ):
                errors.append(
                    f"run {label!r} retained report {path!r} exposes finding evidence"
                )
        for result in document.get("command_results") or []:
            errors.extend(_verify_sanitized_command_result(label, path, result))
        for verification in document.get("verification") or []:
            if not isinstance(verification, dict):
                continue
            if verification.get("detail") is not None:
                errors.append(
                    f"run {label!r} retained report {path!r} exposes verification detail"
                )
            errors.extend(
                _verify_sanitized_command_result(
                    label,
                    path,
                    verification.get("command"),
                    allowed_stdout=(
                        {CUDA_DRIVER_API_ATTESTATION}
                        if verification.get("name") == "container.gpu"
                        and verification.get("ok") is True
                        else None
                    ),
                )
            )
        snapshot = document.get("rollback")
        if isinstance(snapshot, dict):
            errors.extend(_verify_sanitized_snapshot(label, path, snapshot))
    return errors


def _verify_sanitized_command_result(
    label: str,
    path: str,
    result: Any,
    *,
    allowed_stdout: set[str] | None = None,
) -> list[str]:
    if not isinstance(result, dict):
        return []
    allowed_stdout_values = set(allowed_stdout or {"", "[redacted]"})
    if _sanitized_dnf_module_marker(result) is not None:
        allowed_stdout_values.add(result["stdout"])
    allowed_stderr_values = {"", "[redacted]"}
    if (
        result.get("stdout") not in allowed_stdout_values
        or result.get("stderr") not in allowed_stderr_values
        or result.get("reason") is not None
    ):
        return [f"run {label!r} retained report {path!r} exposes command output"]
    return []


def _sanitized_dnf_module_marker(
    result: Any,
) -> tuple[str, str] | None:
    if not isinstance(result, dict):
        return None
    stdout = result.get("stdout")
    if not isinstance(stdout, str):
        return None
    match = re.fullmatch(
        re.escape(DNF_MODULE_PROOF_ATTESTATION_PREFIX)
        + r"(check|apply):([a-f0-9]{64})\]",
        stdout,
    )
    if match is None:
        return None
    mode, binding = match.groups()
    command = result.get("command")
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        return None
    try:
        expected = dnf_module_enable_command(
            apply=mode == "apply",
            stream=command[6] if len(command) >= 7 else "",
            preflight_sha256=binding if mode == "apply" else None,
        )
    except (IndexError, ValueError):
        return None
    if command != expected:
        return None
    return mode, binding


def _verify_sanitized_snapshot(
    label: str, path: str, snapshot: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    snapshot_path = snapshot.get("path")
    if snapshot_path is not None and (
        not isinstance(snapshot_path, str)
        or re.fullmatch(r"attested:[a-f0-9]{64}", snapshot_path) is None
    ):
        errors.append(
            f"run {label!r} retained snapshot {path!r} exposes its private path"
        )
    host_id = snapshot.get("host_id")
    if not isinstance(host_id, str) or re.fullmatch(
        r"attested:[a-f0-9]{64}", host_id
    ) is None:
        errors.append(f"run {label!r} retained snapshot {path!r} exposes host identity")
    managed_files_are_private = all(
        isinstance(item, dict)
        and (
            (
                item.get("existed") is False
                and item.get("content_base64") is None
                and item.get("mode") is None
            )
            or (
                item.get("existed") is True
                and isinstance(item.get("mode"), int)
                and not isinstance(item.get("mode"), bool)
                and isinstance(item.get("content_base64"), str)
                and re.fullmatch(
                    r"attested:[a-f0-9]{64}",
                    item["content_base64"],
                )
                is not None
            )
        )
        for item in snapshot.get("managed_files") or []
    )
    if not managed_files_are_private:
        errors.append(
            f"run {label!r} retained snapshot {path!r} exposes managed-file content"
        )
    if not _has_sanitized_gpu_uuids(snapshot.get("gpu_uuids")):
        errors.append(
            f"run {label!r} retained snapshot {path!r} exposes raw GPU identity"
        )
    if not _has_sanitized_mig_geometry(
        snapshot.get("mig_geometry"), snapshot.get("gpu_uuids")
    ):
        errors.append(
            f"run {label!r} retained snapshot {path!r} exposes raw MIG identity"
        )
    return errors


def _has_sanitized_gpu_uuids(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and len(value) == len(set(value))
        and all(
            isinstance(item, str)
            and re.fullmatch(r"attested:[a-f0-9]{64}", item) is not None
            for item in value
        )
    )


def _has_sanitized_optional_gpu_uuids(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(set(value))
        and all(
            isinstance(item, str)
            and re.fullmatch(r"attested:[a-f0-9]{64}", item) is not None
            for item in value
        )
    )


def _has_sanitized_mig_geometry(value: Any, gpu_uuids: Any) -> bool:
    if not isinstance(value, list) or not isinstance(gpu_uuids, list):
        return False
    return all(
        isinstance(instance, dict)
        and isinstance(instance.get("gpu_uuid"), str)
        and instance["gpu_uuid"] in gpu_uuids
        and re.fullmatch(r"attested:[a-f0-9]{64}", instance["gpu_uuid"])
        is not None
        for instance in value
    )


def _verify_scenario_chronology(
    run: dict[str, Any],
    passed_scenarios: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
) -> list[str]:
    label = str(run.get("id"))
    by_scenario = {
        str(scenario.get("name")): reports.get(str(scenario.get("report")))
        for scenario in passed_scenarios
    }
    sequence = (
        "lock_apply",
        "snapshot",
        "plan",
        "install_apply",
        "verify_apply",
        "doctor",
        "rollback_apply",
        "policy_rollback_apply",
    )
    previous_name: str | None = None
    previous_completed: datetime | None = None
    for name in sequence:
        report = by_scenario.get(name)
        if not isinstance(report, dict):
            continue
        try:
            started = _parse_datetime(str(report.get("started_at")))
            completed = _parse_datetime(str(report.get("completed_at")))
        except ValueError:
            continue
        if previous_completed is not None and started < previous_completed:
            return [
                f"run {label!r} scenario chronology is invalid: {name!r} started before {previous_name!r} completed"
            ]
        previous_name = name
        previous_completed = completed
    rollback = by_scenario.get("rollback_apply")
    healthy_doctor = by_scenario.get("doctor")
    healthy_completed: datetime | None = None
    if isinstance(healthy_doctor, dict):
        try:
            healthy_completed = _parse_datetime(
                str(healthy_doctor.get("completed_at"))
            )
        except ValueError:
            pass
    if isinstance(rollback, dict):
        try:
            rollback_started = _parse_datetime(str(rollback.get("started_at")))
        except ValueError:
            return []
        for scenario in passed_scenarios:
            name = str(scenario.get("name"))
            if not name.startswith("doctor_"):
                continue
            report = reports.get(str(scenario.get("report")))
            if not isinstance(report, dict):
                continue
            try:
                started = _parse_datetime(str(report.get("started_at")))
                completed = _parse_datetime(str(report.get("completed_at")))
            except ValueError:
                continue
            if healthy_completed is not None and started < healthy_completed:
                return [
                    f"run {label!r} fault scenario {name!r} started before the healthy doctor completed"
                ]
            if completed > rollback_started:
                return [
                    f"run {label!r} fault scenario {name!r} completed after rollback started"
                ]
    return []


def _verify_matrix_capability_evidence(
    run: dict[str, Any],
    passed_scenarios: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
) -> list[str]:
    label = str(run.get("id"))
    matrix_id = str(run.get("matrix_id"))
    errors: list[str] = []
    scenario_reports = {
        str(scenario.get("name")): reports.get(str(scenario.get("report")))
        for scenario in passed_scenarios
    }
    doctor = scenario_reports.get("doctor")
    verify = scenario_reports.get("verify_apply")
    install = scenario_reports.get("install_apply")
    rollback = scenario_reports.get("rollback_state_verify")
    snapshot_report = scenario_reports.get("snapshot")
    doctor_audit = doctor.get("audit") if isinstance(doctor, dict) else None
    doctor_kernel = (
        doctor_audit.get("kernel") if isinstance(doctor_audit, dict) else None
    )
    doctor_module = (
        doctor_audit.get("module") if isinstance(doctor_audit, dict) else None
    )
    install_commands = _successful_commands(install)
    rollback_commands = _successful_commands(rollback)
    rollback_checks = _verification_results(rollback)
    verify_checks = _verification_results(verify)
    observed_mig_geometry = (
        doctor_audit.get("mig_geometry")
        if isinstance(doctor_audit, dict)
        else None
    )
    observed_capabilities = {
        "secure_boot": bool(
            isinstance(doctor_kernel, dict)
            and doctor_kernel.get("secure_boot_enabled") is True
            and isinstance(doctor_module, dict)
            and doctor_module.get("signed") is True
            and doctor_module.get("installed_signed") is True
            and doctor_module.get("version")
            == doctor_module.get("installed_version")
        ),
        "fabric_manager": bool(
            isinstance(doctor_audit, dict)
            and isinstance(doctor_module, dict)
            and doctor_audit.get("fabric_manager_active") is True
            and doctor_audit.get("fabric_manager_enabled") is True
            and doctor_audit.get("fabric_manager_applicable") is True
            and doctor_audit.get("fabric_manager_healthy") is True
            and doctor_audit.get("fabric_manager_version")
            == doctor_module.get("version")
        ),
        "mig_capable": bool(
            isinstance(doctor_audit, dict)
            and doctor_audit.get("mig_capable") is True
        ),
        "mig_toggle": bool(
            isinstance(doctor_audit, dict)
            and doctor_audit.get("mig_capable") is True
            and doctor_audit.get("mig_mode") == "enabled"
            and doctor_audit.get("mig_mode_pending") == "enabled"
            and doctor_audit.get("mig_geometry_complete") is True
            and _is_single_full_mig_geometry(observed_mig_geometry)
            and _has_single_attested_uuid(
                doctor_audit.get("mig_device_uuids")
            )
            and _has_uuid_bound_mig_command(
                install_commands, doctor_audit, "1"
            )
            and _has_uuid_bound_mig_create_command(
                install_commands, doctor_audit
            )
            and _has_uuid_bound_mig_command(
                rollback_commands, doctor_audit, "0"
            )
            and _has_uuid_bound_mig_destroy_commands(
                rollback_commands, doctor_audit
            )
            and rollback_checks.get("rollback.mig-mode") is True
            and rollback_checks.get("rollback.mig-geometry") is True
            and verify_checks.get("mig.geometry-observable") is True
            and verify_checks.get("mig.geometry") is True
            and verify_checks.get("mig.device-uuid") is True
            and verify_checks.get("container.device-binding") is True
            and verify_checks.get("container.gpu") is True
        ),
    }
    claimed_capabilities = run.get("capabilities", {})
    mismatched_capabilities = sorted(
        capability
        for capability, observed in observed_capabilities.items()
        if not isinstance(claimed_capabilities, dict)
        or claimed_capabilities.get(capability) is not observed
    )
    if mismatched_capabilities:
        errors.append(
            f"run {label!r} capability claims do not match retained observations: "
            + ", ".join(mismatched_capabilities)
        )
    labels = set(run.get("runner_labels", []))
    label_mismatches = sorted(
        capability_label
        for capability_label, capability in _SPECIAL_CAPABILITY_LABELS.items()
        if (capability_label in labels) is not observed_capabilities[capability]
    )
    if label_mismatches:
        errors.append(
            f"run {label!r} capability labels do not match retained observations: "
            + ", ".join(label_mismatches)
        )

    if matrix_id == "secure-boot" and not (
            observed_capabilities["secure_boot"]
            and verify_checks.get("secure-boot.observable") is True
            and verify_checks.get("secure-boot.module-signed") is True
            and verify_checks.get("secure-boot.on-disk-module-signed") is True
    ):
        errors.append(
            f"secure-boot matrix run {label!r} does not prove enabled firmware and a signed module"
        )

    if matrix_id == "fabric-manager" and not (
            observed_capabilities["fabric_manager"]
            and verify_checks.get("fabric-manager") is True
            and verify_checks.get("fabric-manager.enabled") is True
            and verify_checks.get("fabric-manager.applicable") is True
            and verify_checks.get("fabric-manager.fabric-health") is True
            and verify_checks.get("fabric-manager.version") is True
    ):
        errors.append(
            f"fabric-manager matrix run {label!r} does not prove active, enabled, version-matched service state"
        )

    if matrix_id == "mig-toggle":
        snapshot = (
            snapshot_report.get("rollback")
            if isinstance(snapshot_report, dict)
            else None
        )
        rollback_audit = rollback.get("audit") if isinstance(rollback, dict) else None
        if not (
            isinstance(snapshot, dict)
            and snapshot.get("mig_mode") == "disabled"
            and snapshot.get("mig_geometry") == []
            and isinstance(doctor_audit, dict)
            and doctor_audit.get("mig_mode") == "enabled"
            and doctor_audit.get("mig_capable") is True
            and doctor_audit.get("mig_mode_pending") == "enabled"
            and doctor_audit.get("mig_geometry_complete") is True
            and _is_single_full_mig_geometry(
                doctor_audit.get("mig_geometry")
            )
            and _has_single_attested_uuid(
                doctor_audit.get("mig_device_uuids")
            )
            and _has_uuid_bound_mig_command(
                install_commands, doctor_audit, "1"
            )
            and _has_uuid_bound_mig_create_command(
                install_commands, doctor_audit
            )
            and isinstance(rollback_audit, dict)
            and rollback_audit.get("mig_mode") == "disabled"
            and rollback_audit.get("mig_mode_pending") == "disabled"
            and rollback_audit.get("mig_geometry_complete") is True
            and rollback_audit.get("mig_geometry") == []
            and rollback_audit.get("mig_device_uuids") == []
            and _has_uuid_bound_mig_command(
                rollback_commands, doctor_audit, "0"
            )
            and _has_uuid_bound_mig_destroy_commands(
                rollback_commands, doctor_audit
            )
            and rollback_checks.get("rollback.mig-mode") is True
            and rollback_checks.get("rollback.mig-geometry") is True
            and verify_checks.get("mig.geometry") is True
            and verify_checks.get("mig.device-uuid") is True
            and verify_checks.get("container.device-binding") is True
            and verify_checks.get("container.gpu") is True
        ):
            errors.append(
                f"MIG matrix run {label!r} does not prove enable, observe, and rollback transitions"
            )
    return errors


def _verification_results(report: Any) -> dict[str, bool]:
    if not isinstance(report, dict):
        return {}
    return {
        str(check.get("name")): check.get("ok") is True
        for check in report.get("verification", [])
        if isinstance(check, dict)
    }


def _has_uuid_bound_mig_command(
    commands: set[tuple[str, ...]], audit: dict[str, Any], mode: str
) -> bool:
    gpu_uuids = audit.get("gpu_uuids")
    return bool(
        isinstance(gpu_uuids, list)
        and gpu_uuids
        and isinstance(gpu_uuids[0], str)
        and (
            "nvidia-smi",
            "-i",
            gpu_uuids[0],
            "-mig",
            mode,
        )
        in commands
    )


def _has_uuid_bound_mig_create_command(
    commands: set[tuple[str, ...]], audit: dict[str, Any]
) -> bool:
    gpu_uuids = audit.get("gpu_uuids")
    return bool(
        isinstance(gpu_uuids, list)
        and gpu_uuids
        and isinstance(gpu_uuids[0], str)
        and (
            "nvidia-smi",
            "mig",
            "-i",
            gpu_uuids[0],
            "-cgi",
            "0:0",
            "-C",
        )
        in commands
    )


def _has_uuid_bound_mig_destroy_commands(
    commands: set[tuple[str, ...]], audit: dict[str, Any]
) -> bool:
    gpu_uuids = audit.get("gpu_uuids")
    if (
        not isinstance(gpu_uuids, list)
        or not gpu_uuids
        or not isinstance(gpu_uuids[0], str)
    ):
        return False
    prefix = ("nvidia-smi", "mig", "-i", gpu_uuids[0])
    return (*prefix, "-dci") in commands and (*prefix, "-dgi") in commands


def _has_single_attested_uuid(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], str)
        and re.fullmatch(r"attested:[a-f0-9]{64}", value[0]) is not None
    )


def _is_single_full_mig_geometry(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    instance = value[0]
    if (
        not isinstance(instance, dict)
        or instance.get("profile_id") != 0
        or instance.get("placement_start") != 0
        or not isinstance(instance.get("profile"), str)
    ):
        return False
    compute_instances = instance.get("compute_instances")
    if not isinstance(compute_instances, list) or len(compute_instances) != 1:
        return False
    compute = compute_instances[0]
    if not isinstance(compute, dict) or not isinstance(
        compute.get("profile"), str
    ):
        return False
    gpu_profile = instance["profile"].removeprefix("MIG ").lower()
    compute_profile = compute["profile"].removeprefix("MIG ").lower()
    if compute_profile == gpu_profile:
        return True
    gpu_match = re.fullmatch(r"([1-9]\d*)g\..+", gpu_profile)
    compute_match = re.fullmatch(
        r"([1-9]\d*)c\.([1-9]\d*g\..+)", compute_profile
    )
    return bool(
        gpu_match
        and compute_match
        and compute_match.group(1) == gpu_match.group(1)
        and compute_match.group(2) == gpu_profile
    )


def _successful_commands(report: Any) -> set[tuple[str, ...]]:
    if not isinstance(report, dict):
        return set()
    return {
        tuple(result["command"])
        for result in report.get("command_results", [])
        if isinstance(result, dict)
        and isinstance(result.get("command"), list)
        and all(isinstance(part, str) for part in result["command"])
        and result.get("returncode") == 0
        and result.get("skipped") is False
    }


def _planned_install_commands(report: Any) -> set[tuple[str, ...]]:
    if not isinstance(report, dict):
        return set()
    return {
        tuple(command)
        for action in report.get("plan", [])
        if isinstance(action, dict) and action.get("id") == "install.packages"
        for command in action.get("commands", [])
        if isinstance(command, list)
        and command
        and all(isinstance(part, str) for part in command)
    }


def _planned_install_mutations(report: Any) -> set[tuple[str, ...]]:
    return _planned_install_commands(report).intersection(
        _successful_commands(report)
    )


def _expected_forward_install_command(
    package_manager: str,
    packages: list[str],
) -> tuple[str, ...] | None:
    if not packages:
        return None
    if package_manager == "apt-get":
        return (
            "apt-get",
            "install",
            "-y",
            "--allow-downgrades",
            "--no-install-recommends",
            *packages,
        )
    if package_manager in {"dnf", "yum"}:
        return (
            package_manager,
            "-C",
            "--setopt=install_weak_deps=False",
            "install",
            "-y",
            *packages,
        )
    if package_manager == "zypper":
        return (
            "zypper",
            "--non-interactive",
            "--no-refresh",
            "install",
            "--no-recommends",
            *packages,
        )
    return None


def _expected_local_forward_command(
    snapshot: dict[str, Any],
    package_manager: str,
    *,
    remove_specs: list[str] | None = None,
) -> tuple[str, ...] | None:
    authority = _snapshot_package_authority(snapshot, package_manager)
    if authority is None:
        return None
    forward_payloads = [
        payload
        for payload in authority.bundle.packages
        if "forward" in payload.roles
    ]
    if not forward_payloads:
        return None
    try:
        if package_manager not in {"dnf", "yum"}:
            return tuple(
                forward_package_command(
                    authority.snapshot_path,
                    authority.bundle,
                    package_manager,
                    remove_specs=remove_specs,
                )
            )
        forward_identities = {
            (
                payload.name,
                payload.architecture,
                payload.epoch,
                payload.version,
            )
            for payload in forward_payloads
        }
        forward_slots = {
            (payload.name, payload.architecture)
            for payload in forward_payloads
        }
        expected_removals = sorted(
            _rpm_package_info_spec(package)
            for package in authority.packages
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
        return tuple(
            dnf_local_transaction_command(
                apply=True,
                restore_paths=local_payload_paths(
                    authority.snapshot_path,
                    authority.bundle,
                    role="forward",
                ),
                remove_specs=[],
                expected_installs=sorted(
                    _rpm_payload_spec(payload) for payload in forward_payloads
                ),
                expected_removals=expected_removals,
            )
        )
    except (PackagePayloadError, ValueError):
        return None


def _rpm_package_info_spec(package: PackageInfo) -> str:
    epoch = f"{package.epoch}:" if package.epoch else ""
    architecture = f".{package.architecture}" if package.architecture else ""
    return f"{package.name}-{epoch}{package.version}{architecture}"


def _rpm_payload_spec(payload: PackagePayload) -> str:
    epoch = f"{payload.epoch}:" if payload.epoch else ""
    return (
        f"{payload.name}-{epoch}{payload.version}.{payload.architecture}"
    )


def _planned_lock_mutations(report: Any) -> set[tuple[str, ...]]:
    if not isinstance(report, dict):
        return set()
    planned = {
        tuple(command)
        for action in report.get("plan", [])
        if isinstance(action, dict)
        and isinstance(action.get("id"), str)
        and action["id"].startswith("lock.")
        for command in action.get("commands", [])
        if isinstance(command, list)
        and command
        and all(isinstance(part, str) for part in command)
    }
    return planned.intersection(_successful_commands(report))


def _verify_controlled_convergence(
    run: dict[str, Any],
    passed_scenarios: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    expected_desired: dict[str, Any] | None,
) -> list[str]:
    if run.get("apply") is not True or expected_desired is None:
        return []
    label = str(run.get("id"))
    scenario_reports = {
        str(scenario.get("name")): reports.get(str(scenario.get("report")))
        for scenario in passed_scenarios
    }
    plan = scenario_reports.get("plan")
    snapshot_report = scenario_reports.get("snapshot")
    install = scenario_reports.get("install_apply")
    verify = scenario_reports.get("verify_apply")
    rollback = scenario_reports.get("rollback_state_verify")
    policy_rollback = scenario_reports.get("policy_rollback_state_verify")
    if not all(
        isinstance(report, dict)
        for report in (
            plan,
            snapshot_report,
            install,
            verify,
            rollback,
            policy_rollback,
        )
    ):
        return []
    assert isinstance(plan, dict)
    assert isinstance(snapshot_report, dict)
    assert isinstance(install, dict)
    assert isinstance(verify, dict)
    assert isinstance(rollback, dict)
    assert isinstance(policy_rollback, dict)
    errors: list[str] = []
    expected_package = "nvidia-container-toolkit"
    if expected_desired.get("cuda_compat") != "none":
        errors.append(
            f"apply run {label!r} requests out-of-scope CUDA forward compatibility"
        )
    for scenario_name in (
        "plan",
        "install_apply",
        "verify_apply",
        "doctor",
        "rollback_state_verify",
        "policy_rollback_state_verify",
    ):
        scenario_report = scenario_reports.get(scenario_name)
        scenario_audit = (
            scenario_report.get("audit")
            if isinstance(scenario_report, dict)
            else None
        )
        compatibility = (
            scenario_audit.get("cuda_compatibility")
            if isinstance(scenario_audit, dict)
            else None
        )
        packages = (
            scenario_audit.get("packages")
            if isinstance(scenario_audit, dict)
            else None
        )
        if compatibility != [] or not isinstance(packages, list) or any(
            isinstance(package, dict)
            and str(package.get("name", "")).startswith("cuda-compat-")
            and package.get("installed") is True
            for package in packages or []
        ):
            errors.append(
                f"apply run {label!r} retains CUDA forward-compatibility deployment evidence in {scenario_name!r}"
            )
    plan_audit = plan.get("audit")
    plan_packages = (
        (plan_audit.get("packages") or []) if isinstance(plan_audit, dict) else []
    )
    installed_before = {
        package.get("name")
        for package in plan_packages
        if isinstance(package, dict) and package.get("installed") is True
    }
    plan_runtime = (
        plan_audit.get("runtime") if isinstance(plan_audit, dict) else None
    )
    error_findings = {
        finding.get("id")
        for finding in plan.get("findings", [])
        if isinstance(finding, dict) and finding.get("severity") == "error"
    }
    if (
        expected_package in installed_before
        or not isinstance(plan_runtime, dict)
        or plan_runtime.get("docker_installed") is not True
        or plan_runtime.get("nvidia_container_runtime_installed") is not False
        or plan_runtime.get("docker_gpus_usable") is not False
        or error_findings
        != {"container-toolkit.missing", "docker.nvidia-runtime-missing"}
    ):
        errors.append(
            f"apply run {label!r} does not prove the controlled missing-container-toolkit prestate"
        )
    planned_mutations = _planned_install_mutations(install)
    expected_plan_command = _expected_forward_install_command(
        str(run.get("package_manager")),
        [expected_package],
    )
    snapshot = snapshot_report.get("rollback")
    expected_install_command = (
        _expected_local_forward_command(
            snapshot,
            str(run.get("package_manager")),
        )
        if isinstance(snapshot, dict)
        else None
    )
    configure_commands = {
        tuple(command)
        for action in install.get("plan", [])
        if isinstance(action, dict) and action.get("id") == "configure.docker-runtime"
        for command in action.get("commands", [])
        if isinstance(command, list)
        and command
        and all(isinstance(part, str) for part in command)
    }
    required_configure_commands = {
        ("nvidia-ctk", "runtime", "configure", "--runtime=docker"),
        ("systemctl", "restart", "docker"),
    }
    if (
        expected_plan_command is None
        or _planned_install_commands(plan) != {expected_plan_command}
        or expected_install_command is None
        or _planned_install_commands(install) != {expected_install_command}
        or planned_mutations != {expected_install_command}
        or configure_commands != required_configure_commands
        or not required_configure_commands.issubset(_successful_commands(install))
    ):
        errors.append(
            f"apply run {label!r} has no complete successful plan-bound toolkit/runtime mutation"
        )
    install_audit = install.get("audit")
    installed_after = {
        package.get("name")
        for package in (
            (install_audit.get("packages") or [])
            if isinstance(install_audit, dict)
            else []
        )
        if isinstance(package, dict) and package.get("installed") is True
    }
    container_checks = [
        check
        for check in verify.get("verification", [])
        if isinstance(check, dict) and check.get("name") == "container.gpu"
    ]
    container_result = (
        container_checks[0].get("command") if len(container_checks) == 1 else None
    )
    container_command = (
        container_result.get("command")
        if isinstance(container_result, dict)
        else None
    )
    container_verified = bool(
        isinstance(container_result, dict)
        and container_result.get("returncode") == 0
        and container_result.get("skipped") is False
        and container_result.get("stdout") == CUDA_DRIVER_API_ATTESTATION
        and _is_hardened_container_verification(
            container_command,
            expected_desired.get("container_test_image"),
            _expected_container_device(install_audit, expected_desired),
        )
    )
    if (
        not _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE.issubset(installed_after)
        or not _audit_observes_healthy_stack(
            install_audit if isinstance(install_audit, dict) else {},
            expected_desired,
        )
        or _verification_results(verify).get(
            "container.cuda-driver-compatibility"
        )
        is not True
        or _verification_results(verify).get("container.gpu") is not True
        or not container_verified
    ):
        errors.append(
            f"apply run {label!r} does not prove healthy post-install convergence"
        )
    expected_introduced_packages = (
        _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE - installed_before
    )
    install_packages = (
        install_audit.get("packages")
        if isinstance(install_audit, dict)
        else []
    )
    expected_forward_identities = {
        _package_state_identity(package)
        for package in install_packages or []
        if isinstance(package, dict)
        and package.get("installed") is True
        and package.get("name") in _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE
        and package.get("name") not in installed_before
    }
    snapshot_authority = (
        _snapshot_package_authority(
            snapshot,
            str(run.get("package_manager")),
        )
        if isinstance(snapshot, dict)
        else None
    )
    observed_forward_identities = (
        {
            (
                payload.name,
                payload.architecture,
                payload.epoch,
                payload.version,
            )
            for payload in snapshot_authority.bundle.packages
            if "forward" in payload.roles
        }
        if snapshot_authority is not None
        else set()
    )
    if not isinstance(snapshot, dict) or (
        set(snapshot.get("introduced_packages") or [])
        != expected_introduced_packages
        or observed_forward_identities != expected_forward_identities
        or any(
            isinstance(package, dict)
            and package.get("name")
            in _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE
            and package.get("installed") is True
            for package in snapshot.get("packages") or []
        )
    ):
        errors.append(
            f"apply run {label!r} rollback snapshot does not bind the controlled package drift"
        )
    for rollback_name, rollback_report in (
        ("canonical", rollback),
        ("pre-policy", policy_rollback),
    ):
        rollback_audit = rollback_report.get("audit")
        rollback_packages = (
            (rollback_audit.get("packages") or [])
            if isinstance(rollback_audit, dict)
            else []
        )
        rollback_runtime = (
            rollback_audit.get("runtime")
            if isinstance(rollback_audit, dict)
            else None
        )
        rollback_installed = {
            package.get("name")
            for package in rollback_packages
            if isinstance(package, dict) and package.get("installed") is True
        }
        if (
            rollback_installed.intersection(
                _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE
            )
            != installed_before.intersection(
                _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE
            )
            or not isinstance(rollback_runtime, dict)
            or rollback_runtime.get("docker_installed") is not True
            or rollback_runtime.get("nvidia_container_runtime_installed")
            is not False
            or rollback_runtime.get("docker_gpus_usable") is not False
        ):
            errors.append(
                f"apply run {label!r} {rollback_name} rollback did not restore the controlled toolkit prestate"
            )
    return errors


def _is_hardened_container_verification(
    command: Any,
    expected_image: Any,
    expected_device: Any,
) -> bool:
    expected_cuda_version = (
        container_cuda_full_version(expected_image)
        if isinstance(expected_image, str)
        else None
    )
    if (
        not isinstance(command, list)
        or len(command) != 32
        or not all(isinstance(part, str) for part in command)
        or not isinstance(expected_image, str)
        or expected_cuda_version is None
        or not isinstance(expected_device, str)
        or re.fullmatch(r"attested:[a-f0-9]{64}", expected_device) is None
    ):
        return False
    return bool(
        command[:5] == ["docker", "run", "--pull=never", "--rm", "--name"]
        and re.fullmatch(r"nvidia-converge-verify-[a-f0-9]{32}", command[5])
        is not None
        and command[6:]
        == [
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
            f"device={expected_device}",
            "--env",
            "NVIDIA_DRIVER_CAPABILITIES=compute",
            "--env",
            f"NVIDIA_CONVERGE_PROBE_SHA256={_CUDA_DRIVER_PROBE_SHA256}",
            "--env",
            f"NVIDIA_CONVERGE_EXPECTED_CUDA_VERSION={expected_cuda_version}",
            "--interactive",
            "--entrypoint=/bin/bash",
            expected_image,
            "-ceu",
            _CUDA_DRIVER_PROBE_SCRIPT,
        ]
    )


def _verify_report_run_binding(
    run: dict[str, Any],
    report_path: str,
    report: dict[str, Any],
    expected_desired: dict[str, Any] | None,
    *,
    allow_driver_fault_desired: bool = False,
) -> list[str]:
    label = str(run.get("id"))
    errors: list[str] = []
    audit = report.get("audit")
    if not isinstance(audit, dict):
        errors.append(
            f"run {label!r} retained report {report_path!r} has no host audit"
        )
    else:
        mismatches = [
            field
            for field in ("os_id", "os_version", "package_manager")
            if audit.get(field) != run.get(field)
        ]
        if mismatches:
            errors.append(
                f"run {label!r} retained report {report_path!r} host metadata mismatch: "
                + ", ".join(mismatches)
            )
    if (
        expected_desired is not None
        and report.get("desired") != expected_desired
        and not (
            allow_driver_fault_desired
            and _valid_driver_fault_desired(report, expected_desired)
        )
    ):
        errors.append(
            f"run {label!r} retained report {report_path!r} does not match its checked-in desired state"
        )
    snapshot = report.get("rollback")
    if isinstance(snapshot, dict):
        snapshot_mismatches = [
            field
            for field in ("os_id", "os_version", "package_manager")
            if snapshot.get(field) != run.get(field)
        ]
        if snapshot_mismatches:
            errors.append(
                f"run {label!r} snapshot in {report_path!r} host metadata mismatch: "
                + ", ".join(snapshot_mismatches)
            )
        if snapshot.get("host_id") != report.get("host_id"):
            errors.append(
                f"run {label!r} snapshot in {report_path!r} does not bind the report host"
            )
        if isinstance(audit, dict) and snapshot.get("gpu_uuids") != audit.get(
            "gpu_uuids"
        ):
            errors.append(
                f"run {label!r} snapshot in {report_path!r} does not bind the report GPU inventory"
            )
    return errors


def _valid_driver_fault_desired(
    report: dict[str, Any], expected_desired: dict[str, Any]
) -> bool:
    fault_desired = report.get("desired")
    audit = report.get("audit")
    if not isinstance(fault_desired, dict) or not isinstance(audit, dict):
        return False
    expected_without_driver = dict(expected_desired)
    fault_without_driver = dict(fault_desired)
    expected_driver = expected_without_driver.pop("driver", None)
    fault_driver = fault_without_driver.pop("driver", None)
    module = audit.get("module")
    module_version = module.get("version") if isinstance(module, dict) else None
    return bool(
        fault_without_driver == expected_without_driver
        and isinstance(expected_driver, str)
        and isinstance(fault_driver, str)
        and fault_driver != expected_driver
        and _driver_version_matches(expected_driver, module_version)
        and not _driver_version_matches(fault_driver, module_version)
    )


def _verify_fault_restoration(
    run: dict[str, Any],
    passed_scenarios: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    expected_desired: dict[str, Any] | None,
) -> list[str]:
    label = str(run.get("id"))
    errors: list[str] = []
    scenario_reports = {
        str(scenario.get("name")): reports.get(str(scenario.get("report")))
        for scenario in passed_scenarios
    }
    rollback = scenario_reports.get("rollback_apply")
    rollback_started: datetime | None = None
    if isinstance(rollback, dict):
        try:
            rollback_started = _parse_datetime(str(rollback.get("started_at")))
        except ValueError:
            pass

    restoration_completions: list[datetime] = []
    for scenario, restoration_path in _FAULT_RESTORATION_REPORTS.items():
        fault = scenario_reports.get(scenario)
        if not isinstance(fault, dict):
            continue
        restoration = reports.get(restoration_path)
        if not isinstance(restoration, dict):
            continue
        if restoration.get("report_path") != f"artifacts/{restoration_path}":
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} has an unexpected original report path"
            )
        if (
            restoration.get("command") != "doctor"
            or restoration.get("mode") != "dry-run"
            or restoration.get("outcome") != "succeeded"
            or restoration.get("exit_code") != 0
            or restoration.get("incomplete") is not False
            or restoration.get("tool_version") != __version__
        ):
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} is not a successful doctor observation"
            )
        if any(
            finding.get("severity") == "error"
            for finding in restoration.get("findings", [])
            if isinstance(finding, dict)
        ):
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} retained error findings"
            )
        audit = restoration.get("audit")
        if (
            expected_desired is None
            or not isinstance(audit, dict)
            or not _audit_observes_healthy_stack(audit, expected_desired)
        ):
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} does not prove the matrix desired state"
            )
        if restoration.get("host_id") != fault.get("host_id"):
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} does not bind the faulted host"
            )
        errors.extend(
            _verify_report_time_bounds(
                run,
                f"{scenario}_restoration",
                restoration,
            )
        )
        try:
            fault_completed = _parse_datetime(str(fault.get("completed_at")))
            restoration_started = _parse_datetime(
                str(restoration.get("started_at"))
            )
            restoration_completed = _parse_datetime(
                str(restoration.get("completed_at"))
            )
        except ValueError:
            continue
        restoration_completions.append(restoration_completed)
        if restoration_started < fault_completed:
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} started before the fault observation completed"
            )
        if rollback_started is not None and restoration_completed > rollback_started:
            errors.append(
                f"fault restoration for {scenario!r} in run {label!r} completed after rollback started"
            )
    if run.get("apply") is True:
        final_path = "restoration/final-cleanup.json"
        final_cleanup = reports.get(final_path)
        if isinstance(final_cleanup, dict):
            if final_cleanup.get("report_path") != f"artifacts/{final_path}":
                errors.append(
                    f"final fault cleanup in run {label!r} has an unexpected original report path"
                )
            if (
                final_cleanup.get("command") != "doctor"
                or final_cleanup.get("mode") != "dry-run"
                or final_cleanup.get("outcome") != "succeeded"
                or final_cleanup.get("exit_code") != 0
                or final_cleanup.get("incomplete") is not False
                or final_cleanup.get("tool_version") != __version__
            ):
                errors.append(
                    f"final fault cleanup in run {label!r} is not a successful doctor observation"
                )
            final_audit = final_cleanup.get("audit")
            if (
                expected_desired is None
                or not isinstance(final_audit, dict)
                or not _audit_observes_healthy_stack(final_audit, expected_desired)
            ):
                errors.append(
                    f"final fault cleanup in run {label!r} does not prove the matrix desired state"
                )
            healthy = scenario_reports.get("doctor")
            if isinstance(healthy, dict) and final_cleanup.get("host_id") != healthy.get(
                "host_id"
            ):
                errors.append(
                    f"final fault cleanup in run {label!r} does not bind the integration host"
                )
            errors.extend(
                _verify_report_time_bounds(run, "fault_final_cleanup", final_cleanup)
            )
            try:
                final_started = _parse_datetime(str(final_cleanup.get("started_at")))
                final_completed = _parse_datetime(
                    str(final_cleanup.get("completed_at"))
                )
            except ValueError:
                pass
            else:
                if restoration_completions and final_started < max(
                    restoration_completions
                ):
                    errors.append(
                        f"final fault cleanup in run {label!r} started before scenario restoration completed"
                    )
                if rollback_started is not None and final_completed > rollback_started:
                    errors.append(
                        f"final fault cleanup in run {label!r} completed after rollback started"
                    )
    return errors


def _artifact_evidence_members(archive: bytes) -> dict[str, bytes]:
    if not archive or len(archive) > MAX_ARTIFACT_ARCHIVE_BYTES:
        raise ArtifactEvidenceError("archive size is outside the safety limit")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            if handle.comment:
                raise ArtifactEvidenceError("archive comment is not permitted")
            entries = handle.infolist()
            if not entries or len(entries) > MAX_ARTIFACT_ENTRIES:
                raise ArtifactEvidenceError("archive entry count is outside the safety limit")
            total_size = 0
            names: set[str] = set()
            evidence: dict[str, bytes] = {}
            for entry in entries:
                _validate_zip_entry(entry)
                if entry.filename in names:
                    raise ArtifactEvidenceError(
                        f"archive contains duplicate path {entry.filename!r}"
                    )
                names.add(entry.filename)
                total_size += entry.file_size
                if total_size > MAX_ARTIFACT_TOTAL_UNCOMPRESSED_BYTES:
                    raise ArtifactEvidenceError(
                        "archive uncompressed size exceeds the safety limit"
                    )
                if entry.is_dir():
                    if entry.filename not in {
                        "pre/",
                        "reports/",
                        "restoration/",
                    }:
                        raise ArtifactEvidenceError(
                            f"archive contains unexpected directory {entry.filename!r}"
                        )
                    continue
                if not _is_evidence_member(entry.filename):
                    raise ArtifactEvidenceError(
                        f"archive contains non-attestation member {entry.filename!r}"
                    )
                evidence[entry.filename] = _read_zip_member(handle, entry)
            return evidence
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, RuntimeError) as exc:
        raise ArtifactEvidenceError(f"cannot read ZIP archive: {exc}") from exc


def _validate_zip_entry(entry: zipfile.ZipInfo) -> None:
    name = entry.filename
    try:
        encoded_name = name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ArtifactEvidenceError("archive path is not valid UTF-8") from exc
    parts = name.split("/")
    if (
        not name
        or len(encoded_name) > MAX_ARTIFACT_MEMBER_NAME_BYTES
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or any(part in {"", ".", ".."} for part in parts[:-1])
        or parts[-1] in {".", ".."}
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ArtifactEvidenceError(f"archive has unsafe path {name!r}")
    if entry.flag_bits & 0x1:
        raise ArtifactEvidenceError(f"archive member {name!r} is encrypted")
    if entry.comment or entry.extra:
        raise ArtifactEvidenceError(
            f"archive member {name!r} contains unsupported ZIP metadata"
        )
    if entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ArtifactEvidenceError(
            f"archive member {name!r} uses an unsupported compression method"
        )
    mode = entry.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise ArtifactEvidenceError(f"archive member {name!r} is a symbolic link")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ArtifactEvidenceError(f"archive member {name!r} is not a regular file")
    if entry.file_size < 0 or entry.compress_size < 0:
        raise ArtifactEvidenceError(f"archive member {name!r} has an invalid size")


def _is_evidence_member(name: str) -> bool:
    return name in _CANONICAL_ARTIFACT_PATHS


def _read_zip_member(handle: zipfile.ZipFile, entry: zipfile.ZipInfo) -> bytes:
    if entry.file_size > MAX_ARTIFACT_MEMBER_BYTES:
        raise ArtifactEvidenceError(
            f"evidence member {entry.filename!r} exceeds the safety limit"
        )
    try:
        with handle.open(entry, "r") as source:
            payload = source.read(MAX_ARTIFACT_MEMBER_BYTES + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArtifactEvidenceError(
            f"cannot read evidence member {entry.filename!r}: {exc}"
        ) from exc
    if len(payload) > MAX_ARTIFACT_MEMBER_BYTES or len(payload) != entry.file_size:
        raise ArtifactEvidenceError(
            f"evidence member {entry.filename!r} has an invalid expanded size"
        )
    return payload


def _strict_json_document(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ArtifactEvidenceError(f"{label} is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactEvidenceError(
            f"{label} is invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise ArtifactEvidenceError(f"{label} is invalid JSON: {exc}") from exc


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite number {value!r}")


def _verify_scenario_report(
    run: dict[str, Any],
    scenario: str,
    path: str,
    report: dict[str, Any],
) -> list[str]:
    label = str(run.get("id"))
    errors: list[str] = []
    if scenario == "report_schema":
        return errors
    expected_command = _APPLY_SCENARIO_COMMANDS.get(
        scenario, _DRY_RUN_SCENARIO_COMMANDS.get(scenario)
    )
    expected_mode = "apply" if scenario in _APPLY_SCENARIO_COMMANDS else "dry-run"
    if report.get("command") != expected_command:
        errors.append(
            f"scenario {scenario!r} in run {label!r} references {path!r} with the wrong command"
        )
    if report.get("mode") != expected_mode:
        errors.append(
            f"scenario {scenario!r} in run {label!r} references {path!r} with the wrong mode"
        )
    rollback = report.get("rollback")
    if (
        report.get("command") in {"install", "verify", "lock", "snapshot"}
        and isinstance(rollback, dict)
        and rollback.get("operation_id") != report.get("operation_id")
    ):
        errors.append(
            f"scenario {scenario!r} in run {label!r} has a rollback snapshot "
            "bound to a different operation ID"
        )
    expected_report_path = (
        "/var/lib/nvidia-converge/reports/"
        f"{run.get('matrix_id')}-{run.get('workflow_run_id')}-"
        f"{run.get('workflow_run_attempt')}-"
        f"{'policy-rollback' if scenario.startswith('policy_rollback_') else expected_command}.json"
        if expected_mode == "apply"
        else f"artifacts/{path}"
    )
    if report.get("report_path") != expected_report_path:
        errors.append(
            f"scenario {scenario!r} in run {label!r} has an unexpected original report path"
        )
    if report.get("tool_version") != __version__:
        errors.append(
            f"scenario {scenario!r} in run {label!r} was produced by a different tool version"
        )
    if report.get("incomplete") is not False:
        errors.append(
            f"scenario {scenario!r} in run {label!r} has incomplete operation evidence"
        )

    expected_faults = _EXPECTED_FAULT_FINDINGS.get(scenario)
    findings = {
        finding.get("id")
        for finding in report.get("findings", [])
        if isinstance(finding, dict)
    }
    if expected_faults is not None:
        if not findings.intersection(expected_faults):
            errors.append(
                f"fault scenario {scenario!r} in run {label!r} did not record its expected finding"
            )
        if not any(
            finding.get("id") in expected_faults
            and isinstance(finding.get("remediation"), str)
            and bool(finding["remediation"].strip())
            for finding in report.get("findings", [])
            if isinstance(finding, dict)
        ):
            errors.append(
                f"fault scenario {scenario!r} in run {label!r} has no actionable remediation"
            )
        if report.get("outcome") != "failed" or report.get("exit_code") == 0:
            errors.append(
                f"fault scenario {scenario!r} in run {label!r} did not record an expected doctor failure"
            )
    elif report.get("outcome") != "succeeded" or report.get("exit_code") != 0:
        errors.append(f"scenario {scenario!r} in run {label!r} did not succeed")

    if scenario == "doctor" and any(
        finding.get("severity") == "error"
        for finding in report.get("findings", [])
        if isinstance(finding, dict)
    ):
        errors.append(f"healthy doctor scenario in run {label!r} retained error findings")
    audit = report.get("audit")
    if (
        expected_faults is not None
        and isinstance(audit, dict)
        and not _audit_observes_fault(scenario, audit, report.get("desired", {}))
    ):
        errors.append(
            f"fault scenario {scenario!r} in run {label!r} is inconsistent with its host audit"
        )
    if (
        scenario == "doctor"
        and isinstance(audit, dict)
        and not _audit_observes_healthy_stack(audit, report.get("desired", {}))
    ):
        errors.append(
            f"healthy doctor scenario in run {label!r} is inconsistent with its host audit"
        )
    if scenario in {
        "snapshot",
        "rollback_apply",
        "rollback_state_verify",
        "policy_rollback_apply",
        "policy_rollback_state_verify",
    } and not isinstance(report.get("rollback"), dict):
        errors.append(
            f"scenario {scenario!r} in run {label!r} has no rollback snapshot evidence"
        )
    if scenario in {"rollback_state_verify", "policy_rollback_state_verify"}:
        checks: dict[str, bool] = {
            str(check.get("name")): check.get("ok") is True
            for check in report.get("verification", [])
            if isinstance(check, dict)
        }
        required_checks = set(_REQUIRED_ROLLBACK_CHECKS)
        snapshot = report.get("rollback", {})
        if isinstance(snapshot, dict):
            if snapshot.get("mig_mode") is not None:
                required_checks.add("rollback.mig-mode")
            for field, check_name in (
                ("docker_service_active", "rollback.docker-service-active"),
                ("docker_service_enabled", "rollback.docker-service-enabled"),
                (
                    "docker_service_unit_file_state",
                    "rollback.docker-service-unit-file-state",
                ),
                ("docker_socket_active", "rollback.docker-socket-active"),
                ("docker_socket_enabled", "rollback.docker-socket-enabled"),
                (
                    "docker_socket_unit_file_state",
                    "rollback.docker-socket-unit-file-state",
                ),
                (
                    "nvidia_persistenced_active",
                    "rollback.nvidia-persistenced-active",
                ),
                (
                    "nvidia_persistenced_enabled",
                    "rollback.nvidia-persistenced-enabled",
                ),
                (
                    "nvidia_persistenced_unit_file_state",
                    "rollback.nvidia-persistenced-unit-file-state",
                ),
                ("fabric_manager_active", "rollback.fabric-manager-active"),
                ("fabric_manager_enabled", "rollback.fabric-manager-enabled"),
                (
                    "fabric_manager_unit_file_state",
                    "rollback.fabric-manager-unit-file-state",
                ),
            ):
                if snapshot.get(field) is not None:
                    required_checks.add(check_name)
        missing_checks = sorted(required_checks - checks.keys())
        failed_checks = sorted(name for name, ok in checks.items() if not ok)
        if missing_checks:
            errors.append(
                f"rollback verification in run {label!r} is missing checks: "
                + ", ".join(missing_checks)
            )
        if failed_checks:
            errors.append(
                f"rollback verification in run {label!r} has failed checks: "
                + ", ".join(failed_checks)
            )
        if isinstance(snapshot, dict) and isinstance(audit, dict):
            errors.extend(_verify_rollback_audit(label, snapshot, audit))
    if scenario == "verify_apply":
        check_names = [
            str(check.get("name"))
            for check in report.get("verification", [])
            if isinstance(check, dict)
        ]
        if len(check_names) != len(set(check_names)):
            errors.append(
                f"applied verification in run {label!r} has duplicate check names"
            )
        required_checks = _required_verify_checks(report.get("desired"))
        missing_checks = sorted(required_checks - set(check_names))
        if missing_checks:
            errors.append(
                f"applied verification in run {label!r} is missing checks: "
                + ", ".join(missing_checks)
            )
    errors.extend(_verify_report_internal_status(label, scenario, report))
    errors.extend(_verify_report_time_bounds(run, scenario, report))
    return errors


def _required_verify_checks(desired: Any) -> set[str]:
    checks = set(_BASE_VERIFY_CHECKS)
    if not isinstance(desired, dict):
        return checks
    if desired.get("cuda_compat") == "none":
        checks.add("cuda-compat.policy")
    else:
        checks.update(
            {
                "cuda-compat.hardware-eligibility",
                "cuda-compat.package",
                "cuda-compat.library",
            }
        )
    if desired.get("secure_boot") == "signed":
        checks.update(
            {
                "secure-boot.module-signed",
                "secure-boot.on-disk-module-signed",
            }
        )
    elif desired.get("secure_boot") == "disabled":
        checks.add("secure-boot.policy")
    if str(desired.get("driver", "")).endswith("-open"):
        checks.update(
            {
                "gpu.open-module-supported",
                "module.open-variant",
                "module.on-disk-open-variant",
                "module.flavor-provenance",
            }
        )
    else:
        checks.update(
            {
                "module.closed-variant",
                "module.on-disk-closed-variant",
                "module.flavor-provenance",
            }
        )
    if desired.get("mig") == "enabled":
        checks.add("mig.capable")
    if desired.get("container_runtime") == "docker":
        checks.update(
            {
                "container.cuda-driver-compatibility",
                "container.gpu",
                "docker.service-trust",
            }
        )
    if desired.get("fabric_manager") is True:
        checks.update(
            {
                "fabric-manager",
                "fabric-manager.enabled",
                "fabric-manager.applicable",
                "fabric-manager.fabric-health",
                "fabric-manager.service-trust",
                "fabric-manager.version",
            }
        )
    return checks


def _verify_rollback_audit(
    run_label: str, snapshot: dict[str, Any], audit: dict[str, Any]
) -> list[str]:
    module = audit.get("module", {})
    kernel = audit.get("kernel", {})
    if not isinstance(module, dict) or not isinstance(kernel, dict):
        return [f"rollback verification in run {run_label!r} has an invalid host audit"]
    mismatches: list[str] = []
    if kernel.get("running") != snapshot.get("kernel"):
        mismatches.append("kernel")
    if (
        module.get("loaded") != snapshot.get("module_loaded")
        or module.get("version") != snapshot.get("module_version")
    ):
        mismatches.append("module")
    mismatches.extend(
        snapshot_field
        for snapshot_field, audit_field in _ROLLBACK_MODULE_PROVENANCE.items()
        if module.get(audit_field) != snapshot.get(snapshot_field)
    )
    snapshot_packages, snapshot_duplicates = _snapshot_package_multiset(snapshot)
    audit_packages, audit_duplicates = _snapshot_package_multiset(
        {
            "packages": [
                package
                for package in audit.get("packages") or []
                if isinstance(package, dict) and package.get("installed") is True
            ]
        }
    )
    if (
        snapshot_duplicates
        or audit_duplicates
        or snapshot_packages != audit_packages
    ):
        mismatches.append("packages")
    for field in (
        "gpu_uuids",
        "mig_mode",
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
    ):
        if snapshot.get(field) is not None and audit.get(field) != snapshot.get(field):
            mismatches.append(field)
    if mismatches:
        return [
            f"rollback verification in run {run_label!r} does not match snapshot state: "
            + ", ".join(mismatches)
        ]
    return []


def _audit_observes_fault(
    scenario: str, audit: dict[str, Any], desired: Any
) -> bool:
    kernel = audit.get("kernel", {})
    module = audit.get("module", {})
    runtime = audit.get("runtime", {})
    if scenario == "doctor_missing_headers":
        return isinstance(kernel, dict) and kernel.get("headers_installed") is False
    if scenario == "doctor_module_unloaded":
        return isinstance(module, dict) and module.get("loaded") is False
    if scenario == "doctor_driver_mismatch":
        return (
            isinstance(module, dict)
            and isinstance(desired, dict)
            and not _driver_version_matches(
                str(desired.get("driver", "")), module.get("version")
            )
        )
    if scenario == "doctor_runtime_missing":
        return isinstance(runtime, dict) and any(
            runtime.get(field) is False
            for field in (
                "docker_installed",
                "nvidia_container_runtime_installed",
                "docker_gpus_usable",
            )
        )
    if scenario == "doctor_fabric_manager_inactive":
        return audit.get("fabric_manager_active") is False
    return False


def _audit_observes_healthy_stack(audit: dict[str, Any], desired: Any) -> bool:
    if not isinstance(desired, dict):
        return False
    kernel = audit.get("kernel", {})
    module = audit.get("module", {})
    runtime = audit.get("runtime", {})
    nvidia_smi = audit.get("nvidia_smi", {})
    nvml = audit.get("nvml", {})
    if not all(
        isinstance(item, dict)
        for item in (kernel, module, runtime, nvidia_smi, nvml)
    ):
        return False
    if (
        kernel.get("headers_installed") is not True
        or not kernel.get("compiler")
        or module.get("loaded") is not True
        or not _driver_version_matches(
            str(desired.get("driver", "")), module.get("version")
        )
        or not _driver_version_matches(
            str(desired.get("driver", "")), module.get("installed_version")
        )
        or module.get("version") != module.get("installed_version")
        or nvidia_smi.get("returncode") != 0
        or nvml.get("returncode") != 0
        or audit.get("package_inventory_complete") is not True
    ):
        return False
    if desired.get("driver", "").endswith("-open") and (
        module.get("open_module") is not True
        or module.get("installed_open_module") is not True
        or module.get("open_module") != module.get("installed_open_module")
        or audit.get("open_kernel_module_supported") is not True
    ):
        return False
    secure_boot = desired.get("secure_boot")
    if secure_boot == "signed" and (
        module.get("signed") is not True
        or module.get("installed_signed") is not True
        or module.get("signed") != module.get("installed_signed")
    ):
        return False
    if secure_boot == "disabled" and kernel.get("secure_boot_enabled") is not False:
        return False
    if desired.get("container_runtime") == "docker" and any(
        runtime.get(field) is not True
        for field in (
            "docker_installed",
            "nvidia_container_runtime_installed",
            "docker_gpus_usable",
        )
    ):
        return False
    if desired.get("fabric_manager") is True and (
        audit.get("fabric_manager_active") is not True
        or audit.get("fabric_manager_enabled") is not True
        or audit.get("fabric_manager_applicable") is not True
        or audit.get("fabric_manager_healthy") is not True
        or not _driver_version_matches(
            str(desired.get("driver", "")), audit.get("fabric_manager_version")
        )
    ):
        return False
    compatibility = audit.get("cuda_compatibility")
    if not isinstance(compatibility, list):
        return False
    if desired.get("cuda_compat") != "none" and not any(
        isinstance(item, dict)
        and item.get("version") == desired.get("cuda_compat")
        and bool(item.get("package_version"))
        and item.get("library_present") is True
        and isinstance(item.get("library_probe"), dict)
        and item["library_probe"].get("returncode") == 0
        for item in compatibility
    ):
        return False
    if (
        audit.get("mig_mode") != desired.get("mig")
        or audit.get("mig_mode_pending") != desired.get("mig")
        or audit.get("mig_geometry_complete") is not True
    ):
        return False
    if desired.get("mig") == "enabled":
        return bool(
            audit.get("mig_capable") is True
            and _is_single_full_mig_geometry(audit.get("mig_geometry"))
            and _has_single_attested_uuid(audit.get("mig_device_uuids"))
        )
    return bool(
        audit.get("mig_geometry") == []
        and audit.get("mig_device_uuids") == []
    )


def _expected_container_device(audit: Any, desired: Any) -> str | None:
    if not isinstance(audit, dict) or not isinstance(desired, dict):
        return None
    if desired.get("mig") == "enabled":
        devices = audit.get("mig_device_uuids")
    else:
        devices = audit.get("gpu_uuids")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], str)
    ):
        return None
    return devices[0]


def _driver_version_matches(driver: str, version: Any) -> bool:
    if not isinstance(version, str) or not version:
        return False
    if "." in driver:
        return version == driver
    desired_major = driver.split("-", 1)[0]
    match = re.match(r"^(\d+)(?:\.|$)", version)
    return bool(match and match.group(1) == desired_major)


def _verify_report_internal_status(
    run_label: str, scenario: str, report: dict[str, Any]
) -> list[str]:
    if scenario in _EXPECTED_FAULT_FINDINGS or scenario in {"doctor", "plan", "report_schema"}:
        return []
    errors: list[str] = []
    failed_commands = [
        result
        for result in report.get("command_results", [])
        if isinstance(result, dict) and result.get("returncode") not in {0, None}
    ]
    failed_verification = [
        check
        for check in report.get("verification", [])
        if isinstance(check, dict) and check.get("ok") is not True
    ]
    error_findings = [
        finding
        for finding in report.get("findings", [])
        if isinstance(finding, dict) and finding.get("severity") == "error"
    ]
    if scenario in {"snapshot", "lock_apply"}:
        error_findings = [
            finding
            for finding in error_findings
            if finding.get("id")
            not in {
                "container-toolkit.missing",
                "docker.nvidia-runtime-missing",
            }
        ]
    if failed_commands or failed_verification or error_findings:
        errors.append(
            f"scenario {scenario!r} in run {run_label!r} has internally failed report results"
        )
    if scenario == "verify_apply" and not report.get("verification"):
        errors.append(
            f"scenario {scenario!r} in run {run_label!r} has no verification checks"
        )
    if scenario == "lock_apply" and not report.get("plan"):
        errors.append(f"scenario {scenario!r} in run {run_label!r} has no lock plan")
    return errors


def _verify_report_time_bounds(
    run: dict[str, Any], scenario: str, report: dict[str, Any]
) -> list[str]:
    label = str(run.get("id"))
    try:
        run_started = _parse_datetime(str(run.get("started_at")))
        run_completed = _parse_datetime(str(run.get("completed_at")))
        report_started = _parse_datetime(str(report.get("started_at")))
        report_completed = _parse_datetime(str(report.get("completed_at")))
    except ValueError:
        return [f"scenario {scenario!r} in run {label!r} has invalid report timestamps"]
    if not run_started <= report_started <= report_completed <= run_completed:
        return [
            f"scenario {scenario!r} in run {label!r} has report timestamps outside the workflow run"
        ]
    return []


def _parse_datetime(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return result


def _verify_report_journal(
    run_label: str,
    path: str,
    payload: bytes,
    report: dict[str, Any],
    *,
    require_mutation: bool,
    required_mutation_commands: set[tuple[str, ...]] | None = None,
) -> list[str]:
    entries, errors = _load_sanitized_journal(run_label, path, payload)
    if errors or not entries:
        return errors
    operation_id = report.get("operation_id")
    if any(entry.get("operation_id") != operation_id for entry in entries):
        errors.append(
            f"run {run_label!r} retained journal {path!r} does not bind the report operation ID"
        )
    if entries[0].get("event") != "operation-started":
        errors.append(
            f"run {run_label!r} retained journal {path!r} does not start with operation-started"
        )
    if any(
        entry.get("event") == "operation-started" for entry in entries[1:]
    ):
        errors.append(
            f"run {run_label!r} retained journal {path!r} repeats operation-started"
        )
    report_snapshot = report.get("rollback")
    expected_snapshot_binding = (
        _sanitized_snapshot_binding(report_snapshot)
        if isinstance(report_snapshot, dict)
        else None
    )
    snapshot_events = [
        entry
        for entry in entries
        if entry.get("event") == "rollback-snapshot-persisted"
    ]
    if expected_snapshot_binding is None:
        errors.append(
            f"run {run_label!r} applied report {path!r} has no retained rollback snapshot authority"
        )
    elif len(snapshot_events) != 1:
        errors.append(
            f"run {run_label!r} retained journal {path!r} does not contain exactly one rollback snapshot binding"
        )
    elif _sanitized_journal_snapshot_binding(
        snapshot_events[0]
    ) != expected_snapshot_binding:
        errors.append(
            f"run {run_label!r} retained journal {path!r} snapshot path/hash/creator/host does not match report.rollback"
        )

    pending_command: tuple[tuple[str, ...], bool] | None = None
    snapshot_seen = False
    successful_host_mutation_seen = False
    release_targets: list[str] = []
    for entry_index, entry in enumerate(entries):
        event = entry.get("event")
        if event == "command-started":
            command = entry["command"]
            mutating = entry["mutating"]
            if pending_command is not None:
                errors.append(
                    f"run {run_label!r} retained journal {path!r} has an invalid command-started event"
                )
                continue
            if (
                mutating is True
                and tuple(command) != ("persist-rollback-snapshot",)
                and not snapshot_seen
            ):
                errors.append(
                    f"run {run_label!r} retained journal {path!r} mutates before its rollback snapshot binding"
                )
            pending_command = (tuple(command), mutating)
        elif event == "command-finished":
            observed = (tuple(entry["command"]), entry["mutating"])
            if pending_command is None or observed != pending_command:
                errors.append(
                    f"run {run_label!r} retained journal {path!r} has an unmatched command-finished event"
                )
            if (
                observed[1] is True
                and observed[0] != ("persist-rollback-snapshot",)
                and entry.get("returncode") == 0
                and entry.get("skipped") is False
            ):
                successful_host_mutation_seen = True
            pending_command = None
        elif event == "rollback-snapshot-persisted":
            if pending_command is not None or snapshot_seen:
                errors.append(
                    f"run {run_label!r} retained journal {path!r} has a misplaced or duplicate snapshot binding"
                )
            snapshot_seen = True
        elif event == "launcher-release-authorized":
            target = entry["release_target"]
            if (
                pending_command is not None
                or not snapshot_seen
                or not successful_host_mutation_seen
                or expected_snapshot_binding is None
                or _sanitized_journal_snapshot_binding(entry)
                != expected_snapshot_binding
            ):
                errors.append(
                    f"run {run_label!r} retained journal {path!r} has an invalid launcher release boundary"
                )
            release_targets.append(target)
        elif event == "operation-recovered":
            if entry_index != len(entries) - 1:
                errors.append(
                    f"run {run_label!r} retained journal {path!r} has a nonterminal recovery event"
                )
            if (
                pending_command is not None
                or not snapshot_seen
                or not successful_host_mutation_seen
                or expected_snapshot_binding is None
                or _sanitized_journal_snapshot_binding(entry)
                != expected_snapshot_binding
            ):
                errors.append(
                    f"run {run_label!r} retained journal {path!r} has an invalid recovery snapshot binding"
                )
    initial_dnf_checks = [
        verification.get("command")
        for verification in report.get("verification") or []
        if isinstance(verification, dict)
        and verification.get("name") == "packages.policy-preflight"
        and verification.get("ok") is True
        and _sanitized_dnf_module_marker(verification.get("command"))
        is not None
    ]
    fresh_dnf_checks = [
        verification.get("command")
        for verification in report.get("verification") or []
        if isinstance(verification, dict)
        and verification.get("name")
        == "packages.post-quarantine-policy-preflight"
        and verification.get("ok") is True
        and _sanitized_dnf_module_marker(verification.get("command"))
        is not None
    ]
    applied_dnf_results = []
    for result in report.get("command_results") or []:
        marker = _sanitized_dnf_module_marker(result)
        if marker is not None and marker[0] == "apply":
            applied_dnf_results.append(result)
    if initial_dnf_checks or fresh_dnf_checks or applied_dnf_results:
        if (
            len(initial_dnf_checks) != 1
            or len(fresh_dnf_checks) != 1
            or len(applied_dnf_results) != 1
        ):
            errors.append(
                f"run {run_label!r} retained journal {path!r} lacks its exact initial-check, snapshot, fresh-check, apply sequence"
            )
        else:
            initial_result = initial_dnf_checks[0]
            check_result = fresh_dnf_checks[0]
            apply_result = applied_dnf_results[0]
            assert isinstance(initial_result, dict)
            assert isinstance(check_result, dict)
            assert isinstance(apply_result, dict)
            initial_marker = _sanitized_dnf_module_marker(initial_result)
            check_marker = _sanitized_dnf_module_marker(check_result)
            apply_marker = _sanitized_dnf_module_marker(apply_result)
            assert (
                initial_marker is not None
                and check_marker is not None
                and apply_marker is not None
            )
            check_command = tuple(check_result["command"])
            apply_command = tuple(apply_result["command"])
            check_finished = [
                index
                for index, entry in enumerate(entries)
                if entry.get("event") == "command-finished"
                and entry.get("mutating") is False
                and entry.get("returncode") == 0
                and entry.get("skipped") is False
                and tuple(entry.get("command", [])) == check_command
            ]
            apply_started = [
                index
                for index, entry in enumerate(entries)
                if entry.get("event") == "command-started"
                and entry.get("mutating") is True
                and tuple(entry.get("command", [])) == apply_command
            ]
            apply_finished = [
                index
                for index, entry in enumerate(entries)
                if entry.get("event") == "command-finished"
                and entry.get("mutating") is True
                and entry.get("returncode") == 0
                and entry.get("skipped") is False
                and tuple(entry.get("command", [])) == apply_command
            ]
            snapshot_indexes = [
                index
                for index, entry in enumerate(entries)
                if entry.get("event") == "rollback-snapshot-persisted"
            ]
            if (
                initial_marker[1] != check_marker[1]
                or check_marker[1] != apply_marker[1]
                or initial_result.get("command") != check_result.get("command")
                or len(check_finished) != 2
                or len(snapshot_indexes) != 1
                or len(apply_started) != 1
                or len(apply_finished) != 1
                or not (
                    check_finished[0]
                    < snapshot_indexes[0]
                    < check_finished[1]
                    < apply_started[0]
                )
                or apply_started[0] >= apply_finished[0]
            ):
                errors.append(
                    f"run {run_label!r} retained journal {path!r} does not prove DNF initial-check, snapshot, fresh-check, apply chronology with one token"
                )
    if pending_command is not None:
        errors.append(
            f"run {run_label!r} retained journal {path!r} has an unfinished command"
        )
    expected_release_sequences: dict[str, set[tuple[str, ...]]] = {
        "install": {("install-target",)},
        "verify": {("operation-target",)},
        "lock": {("operation-target",)},
        "rollback": {("rollback-baseline",)},
        "snapshot": {()},
    }
    release_sequence = tuple(release_targets)
    command = report.get("command")
    if (
        report.get("outcome") == "succeeded"
        and report.get("incomplete") is False
        and isinstance(command, str)
        and command in expected_release_sequences
        and release_sequence not in expected_release_sequences[command]
    ):
        errors.append(
            f"run {run_label!r} retained journal {path!r} has the wrong launcher release authorization sequence"
        )
    if any(entry.get("event") == "report-persistence-failed" for entry in entries):
        errors.append(
            f"run {run_label!r} retained journal {path!r} records report persistence failure"
        )
    terminal = entries[-1]
    completion_events = [
        entry for entry in entries if entry.get("event") == "operation-completed"
    ]
    if terminal.get("event") == "operation-completed":
        completion = terminal
        if len(completion_events) != 1:
            errors.append(
                f"run {run_label!r} retained journal {path!r} has duplicate completion events"
            )
        for field in ("exit_code", "incomplete", "outcome"):
            if completion.get(field) != report.get(field):
                errors.append(
                    f"run {run_label!r} retained journal {path!r} completion {field} does not match its report"
                )
    elif terminal.get("event") == "operation-recovered":
        if (
            terminal.get("recovery_operation_id") == operation_id
            or report.get("incomplete") is not True
            or report.get("outcome") != "failed"
            or len(completion_events) > 1
            or (
                completion_events
                and completion_events[0].get("incomplete") is not True
            )
            or (
                completion_events
                and entries[-2].get("event") != "operation-completed"
            )
        ):
            errors.append(
                f"run {run_label!r} retained journal {path!r} has invalid terminal recovery evidence"
            )
        if completion_events:
            completion = completion_events[0]
            for field in ("exit_code", "incomplete", "outcome"):
                if completion.get(field) != report.get(field):
                    errors.append(
                        f"run {run_label!r} retained journal {path!r} completion {field} does not match its report"
                    )
    else:
        errors.append(
            f"run {run_label!r} retained journal {path!r} has no durable completion event"
        )
    if terminal.get("event") != "operation-recovered" and any(
        entry.get("event") == "operation-completed" for entry in entries[:-1]
    ):
        errors.append(
            f"run {run_label!r} retained journal {path!r} completes before its final event"
        )
    timestamps = [_parse_datetime(entry["timestamp"]) for entry in entries]
    if timestamps != sorted(timestamps):
        errors.append(
            f"run {run_label!r} retained journal {path!r} timestamps are not monotonic"
        )
    if require_mutation:
        successful_report_commands = {
            tuple(result.get("command", []))
            for result in report.get("command_results", [])
            if isinstance(result, dict)
            and result.get("returncode") == 0
            and result.get("skipped") is False
            and isinstance(result.get("command"), list)
        }
        acceptable_mutations = (
            successful_report_commands.intersection(required_mutation_commands)
            if required_mutation_commands is not None
            else successful_report_commands
        )
        has_bound_mutation = any(
            entry.get("event") == "command-finished"
            and entry.get("mutating") is True
            and entry.get("skipped") is False
            and entry.get("returncode") == 0
            and tuple(entry["command"]) in acceptable_mutations
            for entry in entries
        )
        if not has_bound_mutation:
            errors.append(
                f"run {run_label!r} retained journal {path!r} has no report-bound successful mutation evidence"
            )
    return errors


def _sanitized_snapshot_binding(
    snapshot: dict[str, Any],
) -> tuple[str, str, str, str] | None:
    snapshot_path = snapshot.get("path")
    integrity = snapshot.get("integrity_sha256")
    operation_id = snapshot.get("operation_id")
    host_id = snapshot.get("host_id")
    if (
        not isinstance(snapshot_path, str)
        or re.fullmatch(r"attested:[a-f0-9]{64}", snapshot_path) is None
        or not isinstance(integrity, str)
        or re.fullmatch(r"[a-f0-9]{64}", integrity) is None
        or not isinstance(operation_id, str)
        or re.fullmatch(r"[a-f0-9]{32}", operation_id) is None
        or not isinstance(host_id, str)
        or re.fullmatch(r"attested:[a-f0-9]{64}", host_id) is None
    ):
        return None
    return snapshot_path, integrity, operation_id, host_id


def _sanitized_journal_snapshot_binding(
    entry: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    return (
        entry.get("snapshot_path"),
        entry.get("snapshot_integrity_sha256"),
        entry.get("snapshot_operation_id"),
        entry.get("snapshot_host_id"),
    )


def _load_sanitized_journal(
    run_label: str, path: str, payload: bytes
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    lines = payload.splitlines()
    if not lines or len(lines) > MAX_JOURNAL_ENTRIES or any(not line.strip() for line in lines):
        return [], [
            f"run {run_label!r} retained journal {path!r} has an invalid entry count"
        ]
    entries: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            entry = _strict_json_document(line, f"{path} line {index}")
        except ArtifactEvidenceError as exc:
            errors.append(f"run {run_label!r} retained journal is invalid: {exc}")
            continue
        if not isinstance(entry, dict):
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} is not an object"
            )
            continue
        entries.append(entry)
    for index, entry in enumerate(entries, start=1):
        event = entry.get("event")
        if not isinstance(event, str):
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has an unknown event"
            )
            continue
        expected_keys = _JOURNAL_EVENT_KEYS.get(event)
        if expected_keys is None:
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has an unknown event"
            )
            continue
        if set(entry) != expected_keys:
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} does not match the sanitized event shape"
            )
        if (
            not isinstance(entry.get("operation_id"), str)
            or re.fullmatch(r"[a-f0-9]{32}", entry["operation_id"]) is None
            or not isinstance(entry.get("timestamp"), str)
        ):
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has invalid common fields"
            )
        else:
            try:
                _parse_datetime(entry["timestamp"])
            except ValueError:
                errors.append(
                    f"run {run_label!r} retained journal {path!r} line {index} has an invalid timestamp"
                )
        if event in {"command-started", "command-finished"}:
            command = entry.get("command")
            mutating = entry.get("mutating")
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(part, str) for part in command)
                or not isinstance(mutating, bool)
            ):
                errors.append(
                    f"run {run_label!r} retained journal {path!r} line {index} has invalid command fields"
                )
        if event == "command-finished" and (
            (
                entry.get("returncode") is not None
                and (
                    not isinstance(entry.get("returncode"), int)
                    or isinstance(entry.get("returncode"), bool)
                )
            )
            or not isinstance(entry.get("skipped"), bool)
            or entry.get("reason") is not None
        ):
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has invalid sanitized result fields"
            )
        if event in {
            "rollback-snapshot-persisted",
            "launcher-release-authorized",
            "operation-recovered",
        } and _sanitized_snapshot_binding(
            {
                "path": entry.get("snapshot_path"),
                "integrity_sha256": entry.get(
                    "snapshot_integrity_sha256"
                ),
                "operation_id": entry.get("snapshot_operation_id"),
                "host_id": entry.get("snapshot_host_id"),
            }
        ) is None:
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has an invalid snapshot binding"
            )
        if event == "launcher-release-authorized" and entry.get(
            "release_target"
        ) not in {
            "install-target",
            "operation-target",
            "rollback-baseline",
        }:
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has an invalid release target"
            )
        if event == "operation-recovered" and (
            not isinstance(entry.get("recovery_operation_id"), str)
            or re.fullmatch(
                r"[a-f0-9]{32}", entry["recovery_operation_id"]
            )
            is None
            or entry.get("recovery_operation_id")
            == entry.get("operation_id")
        ):
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has an invalid recovery operation ID"
            )
        if event == "operation-completed" and (
            not isinstance(entry.get("exit_code"), int)
            or isinstance(entry.get("exit_code"), bool)
            or not 0 <= entry["exit_code"] <= 255
            or not isinstance(entry.get("incomplete"), bool)
            or entry.get("outcome") not in {"succeeded", "failed"}
        ):
            errors.append(
                f"run {run_label!r} retained journal {path!r} line {index} has invalid completion fields"
            )
    return entries, errors


def _verify_retained_snapshot(
    run: dict[str, Any],
    members: dict[str, bytes],
    reports: dict[str, dict[str, Any]],
    expected_desired: dict[str, Any] | None,
) -> list[str]:
    label = str(run.get("id"))
    snapshot, errors = _load_retained_snapshot(run, members, "rollback")
    policy_snapshot, policy_errors = _load_retained_snapshot(
        run, members, "policy-rollback"
    )
    errors.extend(policy_errors)
    if snapshot is None or policy_snapshot is None:
        return errors
    manager = str(run.get("package_manager"))
    errors.extend(
        _verify_snapshot_command_contract(
            label,
            manager,
            snapshot,
            "canonical",
        )
    )
    errors.extend(
        _verify_snapshot_command_contract(
            label,
            manager,
            policy_snapshot,
            "pre-policy",
        )
    )

    scenario_reports = {
        str(scenario.get("name")): reports.get(str(scenario.get("report")))
        for scenario in run.get("scenarios", [])
        if scenario.get("status") == "passed"
    }
    lock_report = scenario_reports.get("lock_apply")
    snapshot_report = scenario_reports.get("snapshot")
    plan_report = scenario_reports.get("plan")
    install_report = scenario_reports.get("install_apply")
    lock_snapshot = (
        lock_report.get("rollback") if isinstance(lock_report, dict) else None
    )
    install_snapshot = (
        install_report.get("rollback")
        if isinstance(install_report, dict)
        else None
    )
    if not isinstance(lock_snapshot, dict):
        errors.append(
            f"apply run {label!r} lock report has no pre-policy rollback baseline"
        )
    elif _snapshot_baseline(lock_snapshot) != _snapshot_baseline(policy_snapshot):
        errors.append(
            f"apply run {label!r} lock pre-policy baseline does not match its retained rollback snapshot"
        )
    if not isinstance(install_snapshot, dict):
        errors.append(
            f"apply run {label!r} install report has no pre-mutation rollback baseline"
        )
    elif _snapshot_baseline(install_snapshot) != _snapshot_baseline(snapshot):
        errors.append(
            f"apply run {label!r} install pre-mutation baseline does not match the canonical post-policy snapshot"
        )
    lock_audit = lock_report.get("audit") if isinstance(lock_report, dict) else None
    snapshot_audit = (
        snapshot_report.get("audit") if isinstance(snapshot_report, dict) else None
    )
    plan_audit = plan_report.get("audit") if isinstance(plan_report, dict) else None
    if not all(isinstance(audit, dict) for audit in (lock_audit, snapshot_audit, plan_audit)):
        errors.append(
            f"apply run {label!r} does not retain the complete post-policy audit chain"
        )
    else:
        assert isinstance(lock_audit, dict)
        assert isinstance(snapshot_audit, dict)
        assert isinstance(plan_audit, dict)
        if not (
            _audit_baseline(lock_audit) == _audit_baseline(snapshot_audit)
            == _audit_baseline(plan_audit)
        ):
            errors.append(
                f"apply run {label!r} post-policy lock, snapshot, and plan audits do not describe one stable state"
            )
        errors.extend(
            _verify_policy_snapshot_delta(
                run,
                policy_snapshot,
                snapshot,
                lock_report if isinstance(lock_report, dict) else {},
                plan_report if isinstance(plan_report, dict) else {},
                expected_desired,
            )
        )
    snapshot_creation_report = scenario_reports.get("snapshot")
    if (
        isinstance(snapshot_creation_report, dict)
        and isinstance(snapshot_creation_report.get("rollback"), dict)
        and _snapshot_baseline(snapshot_creation_report["rollback"])
        != _snapshot_baseline(snapshot)
    ):
        errors.append(
            f"scenario 'snapshot' in run {label!r} does not reference the retained rollback baseline"
        )
    for scenario_name in ("rollback_apply", "rollback_state_verify"):
        report = scenario_reports.get(scenario_name)
        if isinstance(report, dict) and report.get("rollback") != snapshot:
            errors.append(
                f"scenario {scenario_name!r} in run {label!r} does not reference the retained rollback snapshot"
            )
    for scenario_name in ("policy_rollback_apply", "policy_rollback_state_verify"):
        report = scenario_reports.get(scenario_name)
        if isinstance(report, dict) and report.get("rollback") != policy_snapshot:
            errors.append(
                f"scenario {scenario_name!r} in run {label!r} does not reference the retained pre-policy snapshot"
            )
    doctor_report = scenario_reports.get("doctor")
    rollback_report = scenario_reports.get("rollback_apply")
    policy_rollback_report = scenario_reports.get("policy_rollback_apply")
    doctor_audit = (
        doctor_report.get("audit") if isinstance(doctor_report, dict) else None
    )
    rollback_audit = (
        rollback_report.get("audit") if isinstance(rollback_report, dict) else None
    )
    if isinstance(doctor_audit, dict) and isinstance(rollback_report, dict):
        errors.extend(
            _verify_applied_rollback_transaction(
                label,
                manager,
                "canonical",
                snapshot,
                doctor_audit,
                rollback_report,
                required_file_paths={"/etc/docker/daemon.json"},
            )
        )
    if (
        isinstance(rollback_audit, dict)
        and isinstance(policy_rollback_report, dict)
    ):
        pre_files = _managed_files_by_path(policy_snapshot)
        post_files = _managed_files_by_path(snapshot)
        changed_policy_paths = {
            path
            for path in set(pre_files) | set(post_files)
            if pre_files.get(path) != post_files.get(path)
        }
        errors.extend(
            _verify_applied_rollback_transaction(
                label,
                manager,
                "pre-policy",
                policy_snapshot,
                rollback_audit,
                policy_rollback_report,
                required_file_paths=changed_policy_paths,
            )
        )
    return errors


def _load_retained_snapshot(
    run: dict[str, Any], members: dict[str, bytes], stem: str
) -> tuple[dict[str, Any] | None, list[str]]:
    label = str(run.get("id"))
    snapshot_path = f"pre/{stem}-snapshot.json"
    checksum_path = f"pre/{stem}-snapshot.sha256"
    snapshot_payload = members.get(snapshot_path)
    checksum_payload = members.get(checksum_path)
    if snapshot_payload is None or checksum_payload is None:
        return None, [
            f"apply run {label!r} did not retain {stem!r} snapshot and checksum"
        ]
    errors: list[str] = []
    actual_digest = hashlib.sha256(snapshot_payload).hexdigest()
    try:
        checksum_text = checksum_payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        checksum_text = ""
    checksum_match = re.fullmatch(
        rf"([a-f0-9]{{64}}) [ *](?:artifacts/)?{re.escape(snapshot_path)}",
        checksum_text,
    )
    if checksum_match is None or not hmac.compare_digest(
        checksum_match.group(1), actual_digest
    ):
        errors.append(
            f"apply run {label!r} retained {stem!r} snapshot checksum is invalid"
        )
    try:
        snapshot = _strict_json_document(snapshot_payload, snapshot_path)
    except ArtifactEvidenceError as exc:
        errors.append(
            f"apply run {label!r} retained {stem!r} snapshot is invalid: {exc}"
        )
        return None, errors
    if not isinstance(snapshot, dict):
        errors.append(
            f"apply run {label!r} retained {stem!r} snapshot is not an object"
        )
        return None, errors
    errors.extend(
        _verify_snapshot_integrity(snapshot, label, f"retained {stem!r} snapshot")
    )
    return snapshot, errors


def _snapshot_baseline(snapshot: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(snapshot)
    for field in ("path", "created_at", "operation_id", "integrity_sha256"):
        baseline.pop(field, None)
    return baseline


@dataclass(frozen=True)
class _SnapshotPackageAuthority:
    snapshot_path: Path
    packages: tuple[PackageInfo, ...]
    bundle: PackagePayloadBundle


def _snapshot_package_authority(
    snapshot: dict[str, Any],
    manager: str,
) -> _SnapshotPackageAuthority | None:
    """Bind a sanitized snapshot to one canonical local payload tree."""

    if snapshot.get("schema_version") != "2.6" or snapshot.get(
        "package_manager"
    ) != manager:
        return None
    raw_packages = snapshot.get("packages")
    raw_bundle = snapshot.get("package_payloads")
    if not isinstance(raw_packages, list) or not isinstance(raw_bundle, dict):
        return None
    expected_package_manager = "apt" if manager == "apt-get" else "rpm"
    packages: list[PackageInfo] = []
    package_identities: list[tuple[str, str, str | None, str]] = []
    for raw in raw_packages:
        if not isinstance(raw, dict) or set(raw) != {
            "architecture",
            "epoch",
            "installed",
            "manager",
            "name",
            "version",
        }:
            return None
        name = raw.get("name")
        version = raw.get("version")
        architecture = raw.get("architecture")
        epoch = raw.get("epoch")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or not isinstance(architecture, str)
            or not architecture
            or not (epoch is None or isinstance(epoch, str))
            or raw.get("manager") != expected_package_manager
            or raw.get("installed") is not True
        ):
            return None
        packages.append(
            PackageInfo(
                name,
                version,
                expected_package_manager,
                True,
                architecture=architecture,
                epoch=epoch,
            )
        )
        package_identities.append((name, architecture, epoch, version))
    if len(package_identities) != len(set(package_identities)):
        return None
    if set(raw_bundle) != {"directory", "packages", "total_size_bytes"}:
        return None
    directory = raw_bundle.get("directory")
    raw_payloads = raw_bundle.get("packages")
    total_size = raw_bundle.get("total_size_bytes")
    if (
        not isinstance(directory, str)
        or not 10 <= len(directory) <= 255
        or re.fullmatch(r"[^/\x00]+\.payloads", directory) is None
        or not isinstance(raw_payloads, list)
        or not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or total_size < 0
    ):
        return None
    expected_format = "deb" if manager == "apt-get" else "rpm"
    expected_verification = (
        "apt-repository" if manager == "apt-get" else "rpm-signature"
    )
    payloads: list[PackagePayload] = []
    payload_identities: list[tuple[str, str, str | None, str]] = []
    filenames: list[str] = []
    for raw in raw_payloads:
        if not isinstance(raw, dict) or set(raw) != {
            "architecture",
            "epoch",
            "filename",
            "format",
            "name",
            "roles",
            "sha256",
            "signer_ids",
            "size_bytes",
            "verification",
            "version",
        }:
            return None
        name = raw.get("name")
        architecture = raw.get("architecture")
        epoch = raw.get("epoch")
        version = raw.get("version")
        filename = raw.get("filename")
        digest = raw.get("sha256")
        size_bytes = raw.get("size_bytes")
        roles = raw.get("roles")
        signer_ids = raw.get("signer_ids")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(architecture, str)
            or not architecture
            or not (epoch is None or isinstance(epoch, str))
            or not isinstance(version, str)
            or not version
            or raw.get("format") != expected_format
            or raw.get("verification") != expected_verification
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or filename != f"{digest}.{expected_format}"
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
            or not isinstance(roles, list)
            or not roles
            or roles != sorted(set(roles))
            or not set(roles).issubset({"baseline", "forward"})
            or not isinstance(signer_ids, list)
            or signer_ids != sorted(set(signer_ids))
            or not all(
                isinstance(signer, str)
                and re.fullmatch(r"[0-9a-f]{8,40}", signer) is not None
                for signer in signer_ids
            )
            or (expected_format == "rpm" and not signer_ids)
            or (expected_format == "deb" and bool(signer_ids))
        ):
            return None
        payloads.append(
            PackagePayload(
                name=name,
                architecture=architecture,
                epoch=epoch,
                version=version,
                format=expected_format,
                filename=filename,
                sha256=digest,
                size_bytes=size_bytes,
                verification=expected_verification,
                roles=tuple(roles),
                signer_ids=tuple(signer_ids),
            )
        )
        payload_identities.append((name, architecture, epoch, version))
        filenames.append(filename)
    if (
        len(payload_identities) != len(set(payload_identities))
        or len(filenames) != len(set(filenames))
        or total_size != sum(payload.size_bytes for payload in payloads)
    ):
        return None
    baseline_identities = {
        identity
        for identity, payload in zip(payload_identities, payloads, strict=True)
        if "baseline" in payload.roles
    }
    if baseline_identities != set(package_identities):
        return None
    bundle = PackagePayloadBundle(
        directory=directory,
        packages=tuple(payloads),
        total_size_bytes=total_size,
    )
    commands = snapshot.get("commands")
    if not isinstance(commands, list) or not commands:
        return None
    command_parts: list[str] = []
    for command in commands:
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) for part in command)
        ):
            return None
        command_parts.extend(command)
    filename_set = set(filenames)
    observed_payload_paths: list[Path] = []
    for part in command_parts:
        path = Path(part)
        if (
            path.is_absolute()
            and str(path) == part
            and path.parent.name == directory
            and path.name in filename_set
        ):
            observed_payload_paths.append(path)
    payload_roots = {path.parent for path in observed_payload_paths}
    if len(payload_roots) != 1:
        return None
    payload_root = next(iter(payload_roots))
    expected_baseline_paths = {
        payload_root / payload.filename
        for payload in payloads
        if "baseline" in payload.roles
    }
    if (
        set(observed_payload_paths) != expected_baseline_paths
        or len(observed_payload_paths) != len(expected_baseline_paths)
    ):
        return None
    snapshot_name = directory.removesuffix(".payloads")
    snapshot_path = payload_root.parent / snapshot_name
    if not snapshot_path.is_absolute() or str(snapshot_path.parent / directory) != str(
        payload_root
    ):
        return None
    declared_path = snapshot.get("path")
    if not isinstance(declared_path, str):
        return None
    attested_match = re.fullmatch(r"attested:([a-f0-9]{64})", declared_path)
    if attested_match is not None:
        if payload_root.parent != Path("/attested") / attested_match.group(1):
            return None
    elif (
        not Path(declared_path).is_absolute()
        or str(Path(declared_path)) != declared_path
        or Path(declared_path) != snapshot_path
    ):
        return None
    return _SnapshotPackageAuthority(
        snapshot_path=snapshot_path,
        packages=tuple(packages),
        bundle=bundle,
    )


def _verify_snapshot_command_contract(
    run_label: str,
    manager: str,
    snapshot: dict[str, Any],
    baseline_name: str,
) -> list[str]:
    authority = _snapshot_package_authority(snapshot, manager)
    introduced = sorted(
        {
            package
            for package in snapshot.get("introduced_packages") or []
            if isinstance(package, str) and package
        }
    )
    expected: list[list[str]] | None = None
    if authority is not None:
        try:
            expected = _rollback_commands(
                list(authority.packages),
                manager,
                remove_packages=introduced,
                snapshot_path=str(authority.snapshot_path),
                package_payloads=authority.bundle,
            )
        except (PackagePayloadError, RollbackSnapshotError, ValueError):
            expected = None
    if expected is None or snapshot.get("commands") != expected:
        return [
            f"apply run {run_label!r} {baseline_name} snapshot does not use the exact safe rollback transaction contract"
        ]
    return []


def _verify_applied_rollback_transaction(
    run_label: str,
    manager: str,
    baseline_name: str,
    snapshot: dict[str, Any],
    current_audit: dict[str, Any],
    rollback_report: dict[str, Any],
    *,
    required_file_paths: set[str],
) -> list[str]:
    if _snapshot_package_authority(snapshot, manager) is None:
        return [
            f"apply run {run_label!r} {baseline_name} rollback has no valid snapshot-bound local package authority"
        ]
    expected_commands = _expected_applied_package_commands(
        snapshot,
        current_audit,
        manager,
    )
    successful = _successful_commands(rollback_report)
    missing_package_commands = [
        command for command in expected_commands if tuple(command) not in successful
    ]
    missing_file_restores = sorted(
        path
        for path in required_file_paths
        if ("restore-file", path) not in successful
    )
    if missing_package_commands or missing_file_restores:
        return [
            f"apply run {run_label!r} {baseline_name} rollback does not retain its exact safe package/file transaction evidence"
        ]
    return []


def _expected_applied_package_commands(
    snapshot: dict[str, Any],
    current_audit: dict[str, Any],
    manager: str,
) -> list[list[str]]:
    authority = _snapshot_package_authority(snapshot, manager)
    if authority is None:
        return []
    baseline_packages = [
        package
        for package in snapshot.get("packages") or []
        if isinstance(package, dict) and package.get("installed") is True
    ]
    current_packages = [
        package
        for package in current_audit.get("packages") or []
        if isinstance(package, dict) and package.get("installed") is True
    ]
    baseline_states = {
        _package_state_identity(package) for package in baseline_packages
    }
    current_states = {
        _package_state_identity(package) for package in current_packages
    }
    restore_packages = [
        package
        for package in baseline_packages
        if _package_state_identity(package) not in current_states
    ]
    baseline_slots = {
        (package.get("name"), package.get("architecture"))
        for package in baseline_packages
    }
    current_baseline_slots = {
        (package.get("name"), package.get("architecture"))
        for package in current_packages
        if _package_state_identity(package) in baseline_states
    }
    added_packages = [
        package
        for package in current_packages
        if _package_state_identity(package) not in baseline_states
        and (
            (package.get("name"), package.get("architecture"))
            not in baseline_slots
            or (package.get("name"), package.get("architecture"))
            in current_baseline_slots
        )
    ]
    removal_specs = sorted(
        {
            _exact_removal_spec(package, manager)
            for package in added_packages
        }
    )
    restore_infos = [
        PackageInfo(
            str(package.get("name")),
            str(package.get("version")),
            str(package.get("manager")),
            True,
            architecture=str(package.get("architecture")),
            epoch=(
                str(package["epoch"])
                if package.get("epoch") is not None
                else None
            ),
        )
        for package in restore_packages
    ]
    expected_rpm_installs: list[str] | None = None
    expected_rpm_removals: list[str] | None = None
    if manager in {"dnf", "yum"}:
        expected_rpm_installs = sorted(
            _rpm_snapshot_package_spec(package)
            for package in restore_packages
            if package.get("manager") == "rpm"
        )
        restore_slots = {
            (package.get("name"), package.get("architecture"))
            for package in restore_packages
            if package.get("manager") == "rpm"
        }
        expected_rpm_removals = sorted(
            {
                *removal_specs,
                *(
                    _rpm_snapshot_package_spec(package)
                    for package in current_packages
                    if package.get("manager") == "rpm"
                    and (package.get("name"), package.get("architecture"))
                    in restore_slots
                    and _package_state_identity(package) not in baseline_states
                ),
            }
        )
    try:
        return _rollback_commands(
            restore_infos,
            manager,
            remove_packages=removal_specs,
            exact_removals=True,
            snapshot_path=str(authority.snapshot_path),
            package_payloads=authority.bundle,
            expected_rpm_installs=expected_rpm_installs,
            expected_rpm_removals=expected_rpm_removals,
        )
    except (PackagePayloadError, RollbackSnapshotError, ValueError):
        return []


def _package_state_identity(package: dict[str, Any]) -> tuple[Any, ...]:
    return (
        package.get("name"),
        package.get("architecture"),
        package.get("epoch"),
        package.get("version"),
    )


def _exact_removal_spec(package: dict[str, Any], manager: str) -> str:
    if manager == "apt-get":
        architecture = (
            f":{package['architecture']}" if package.get("architecture") else ""
        )
        return f"{package['name']}{architecture}"
    if manager in {"dnf", "yum"}:
        return _rpm_snapshot_package_spec(package)
    return _zypper_snapshot_package_spec(package)


def _rpm_snapshot_package_spec(package: dict[str, Any]) -> str:
    epoch = f"{package['epoch']}:" if package.get("epoch") else ""
    architecture = (
        f".{package['architecture']}" if package.get("architecture") else ""
    )
    return f"{package['name']}-{epoch}{package['version']}{architecture}"


def _zypper_snapshot_package_spec(package: dict[str, Any]) -> str:
    epoch = f"{package['epoch']}:" if package.get("epoch") else ""
    architecture = (
        f".{package['architecture']}" if package.get("architecture") else ""
    )
    return f"{package['name']}{architecture}={epoch}{package['version']}"


def _audit_baseline(audit: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(audit)
    baseline.pop("timestamp", None)
    return baseline


def _dnf_policy_proof_binding(
    lock_report: dict[str, Any],
    stream: str,
    run_label: str,
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    verifications = lock_report.get("verification")
    if not isinstance(verifications, list):
        return None, [
            f"apply run {run_label!r} DNF lock report has no verification sequence"
        ]
    fresh = [
        (index, verification)
        for index, verification in enumerate(verifications)
        if isinstance(verification, dict)
        and verification.get("name")
        == "packages.post-quarantine-policy-preflight"
    ]
    initial = [
        (index, verification)
        for index, verification in enumerate(verifications)
        if isinstance(verification, dict)
        and verification.get("name") == "packages.policy-preflight"
    ]
    binding: str | None = None
    if len(initial) != 1 or len(fresh) != 1:
        errors.append(
            f"apply run {run_label!r} DNF lock report does not have exactly one initial and fresh module preflight proof"
        )
    else:
        initial_index, initial_verification = initial[0]
        fresh_index, fresh_verification = fresh[0]
        initial_marker = _sanitized_dnf_module_marker(
            initial_verification.get("command")
        )
        marker = _sanitized_dnf_module_marker(
            fresh_verification.get("command")
        )
        expected_check = dnf_module_enable_command(
            apply=False,
            stream=stream,
        )
        initial_result = initial_verification.get("command")
        check_result = fresh_verification.get("command")
        if (
            initial_verification.get("ok") is not True
            or fresh_verification.get("ok") is not True
            or initial_marker is None
            or marker is None
            or initial_marker[0] != "check"
            or marker[0] != "check"
            or initial_marker[1] != marker[1]
            or not isinstance(initial_result, dict)
            or not isinstance(check_result, dict)
            or initial_result.get("command") != expected_check
            or check_result.get("command") != expected_check
            or initial_result.get("returncode") != 0
            or check_result.get("returncode") != 0
            or initial_result.get("skipped") is not False
            or check_result.get("skipped") is not False
        ):
            errors.append(
                f"apply run {run_label!r} DNF lock report lacks an accepted exact schema-2 check proof"
            )
        else:
            binding = marker[1]
        lock_indexes = [
            index
            for index, verification in enumerate(verifications)
            if isinstance(verification, dict)
            and verification.get("name") == "package-policy.lock"
            and verification.get("ok") is True
        ]
        if len(lock_indexes) != 1 or fresh_index >= lock_indexes[0]:
            errors.append(
                f"apply run {run_label!r} DNF proof chronology does not place fresh check before policy verification"
            )
        if initial_index >= fresh_index:
            errors.append(
                f"apply run {run_label!r} DNF proof chronology does not place initial check before fresh check"
            )

    applied = []
    for result in lock_report.get("command_results") or []:
        marker = _sanitized_dnf_module_marker(result)
        if marker is not None and marker[0] == "apply":
            applied.append((result, marker[1]))
    if binding is None or len(applied) != 1:
        errors.append(
            f"apply run {run_label!r} DNF lock report does not have exactly one schema-2 applied proof"
        )
    else:
        apply_result, apply_binding = applied[0]
        expected_apply = dnf_module_enable_command(
            apply=True,
            stream=stream,
            preflight_sha256=binding,
        )
        if (
            apply_binding != binding
            or apply_result.get("command") != expected_apply
            or apply_result.get("returncode") != 0
            or apply_result.get("skipped") is not False
        ):
            errors.append(
                f"apply run {run_label!r} DNF apply command/proof is not bound to its accepted check token"
            )
    return binding, errors


def _verify_policy_snapshot_delta(
    run: dict[str, Any],
    pre_policy: dict[str, Any],
    post_policy: dict[str, Any],
    lock_report: dict[str, Any],
    plan_report: dict[str, Any],
    expected_desired: dict[str, Any] | None,
) -> list[str]:
    label = str(run.get("id"))
    manager = str(run.get("package_manager"))
    errors: list[str] = []
    invariant_fields = {
        "schema_version",
        "host_id",
        "os_id",
        "os_version",
        "architecture",
        "package_manager",
        "kernel",
        "module_version",
        "module_loaded",
        "module_names",
        "gpu_uuids",
        *_ROLLBACK_MODULE_PROVENANCE,
        "mig_mode",
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
    }
    changed_invariants = sorted(
        field
        for field in invariant_fields
        if pre_policy.get(field) != post_policy.get(field)
    )
    if changed_invariants:
        errors.append(
            f"apply run {label!r} package-policy staging changed non-policy baseline fields: "
            + ", ".join(changed_invariants)
        )

    driver = (
        str(expected_desired.get("driver"))
        if isinstance(expected_desired, dict)
        else ""
    )
    driver_major_match = re.match(r"^(\d+)", driver)
    driver_major = driver_major_match.group(1) if driver_major_match else ""
    pin_selector = driver if "." in driver else driver_major
    expected_pin = f"nvidia-driver-pinning-{pin_selector}"
    dnf_preflight_sha256: str | None = None
    if manager == "dnf" and driver_major:
        suffix = "open" if driver.endswith("-open") else "dkms"
        dnf_preflight_sha256, proof_errors = _dnf_policy_proof_binding(
            lock_report,
            f"{driver_major}-{suffix}",
            label,
        )
        errors.extend(proof_errors)
    pre_packages, pre_duplicates = _snapshot_package_multiset(pre_policy)
    post_packages, post_duplicates = _snapshot_package_multiset(post_policy)
    if pre_duplicates or post_duplicates:
        errors.append(
            f"apply run {label!r} package-policy snapshots contain duplicate package identities"
        )
    if manager == "apt-get":
        pre_pin_records = [
            package
            for package in pre_policy.get("packages") or []
            if isinstance(package, dict)
            and str(package.get("name", "")).startswith(
                "nvidia-driver-pinning-"
            )
        ]
        post_pin_records = [
            package
            for package in post_policy.get("packages") or []
            if isinstance(package, dict)
            and str(package.get("name", "")).startswith(
                "nvidia-driver-pinning-"
            )
        ]
        if pre_pin_records:
            errors.append(
                f"apply run {label!r} APT pre-policy baseline already contains a driver pin"
            )
        if (
            len(post_pin_records) != 1
            or post_pin_records[0].get("name") != expected_pin
        ):
            errors.append(
                f"apply run {label!r} APT policy staging did not establish only {expected_pin}"
            )
        else:
            expected_post_packages = Counter(pre_packages)
            expected_post_packages.update(
                [_canonical_json(post_pin_records[0])]
            )
            if post_packages != expected_post_packages:
                errors.append(
                    f"apply run {label!r} APT policy staging changed package records beyond {expected_pin}"
                )
    elif pre_packages != post_packages:
        errors.append(
            f"apply run {label!r} package-policy staging changed package inventory"
        )

    pre_introduced = set(pre_policy.get("introduced_packages") or [])
    post_introduced = set(post_policy.get("introduced_packages") or [])
    baseline_package_names = {
        package.get("name")
        for package in post_policy.get("packages") or []
        if isinstance(package, dict) and package.get("installed") is True
    }
    expected_post_introduced = (
        _NVIDIA_CONTAINER_TOOLKIT_PACKAGE_CLOSURE - baseline_package_names
    )
    expected_pre_introduced = set(expected_post_introduced)
    if manager == "apt-get":
        expected_pre_introduced.add(expected_pin)
    if (
        post_introduced != expected_post_introduced
        or pre_introduced != expected_pre_introduced
    ):
        errors.append(
            f"apply run {label!r} package-policy snapshots do not exactly track their introduced package targets"
        )

    pre_files = _managed_files_by_path(pre_policy)
    post_files = _managed_files_by_path(post_policy)
    if (
        len(pre_files) != len(pre_policy.get("managed_files") or [])
        or len(post_files) != len(post_policy.get("managed_files") or [])
    ):
        errors.append(
            f"apply run {label!r} package-policy snapshots contain duplicate managed-file paths"
        )
    policy_paths: set[str] = set()
    if manager == "dnf":
        policy_paths.add("/etc/dnf/modules.d/nvidia-driver.module")
        stream = (
            f"{driver_major}-"
            f"{'open' if driver.endswith('-open') else 'dkms'}"
        )
        architecture = pre_policy.get("architecture")
        expected_failsafe_path = (
            "/var/lib/dnf/modulefailsafe/"
            f"nvidia-driver:{stream}:{architecture}.yaml"
            if isinstance(architecture, str)
            else ""
        )
        dynamic_paths = {
            path
            for path in set(pre_files) | set(post_files)
            if _DNF_MODULE_FAILSAFE_PATH.fullmatch(path) is not None
        }
        if dynamic_paths != {expected_failsafe_path}:
            errors.append(
                f"apply run {label!r} DNF snapshots do not bind one architecture-matched fail-safe target"
            )
        else:
            policy_paths.add(expected_failsafe_path)
            expected_absent = {
                "content_base64": None,
                "existed": False,
                "mode": None,
                "path": expected_failsafe_path,
            }
            post_failsafe = post_files.get(expected_failsafe_path)
            if (
                pre_files.get(expected_failsafe_path) != expected_absent
                or not isinstance(post_failsafe, dict)
                or post_failsafe.get("existed") is not True
                or post_failsafe.get("mode") != 0o644
                or not isinstance(post_failsafe.get("content_base64"), str)
                or not post_failsafe["content_base64"]
            ):
                errors.append(
                    f"apply run {label!r} DNF fail-safe target does not prove absent-to-exact-file policy staging"
                )
    elif manager == "zypper":
        policy_paths.add("/etc/zypp/locks")

    all_file_paths = set(pre_files) | set(post_files)
    if set(pre_files) != set(post_files) or any(
        pre_files.get(path) != post_files.get(path)
        for path in all_file_paths - policy_paths
    ):
        errors.append(
            f"apply run {label!r} package-policy staging changed an unrelated managed file"
        )
    if any(pre_files.get(path) == post_files.get(path) for path in policy_paths):
        errors.append(
            f"apply run {label!r} package-policy staging did not change every managed backend file"
        )

    lock_checks = _verification_results(lock_report)
    if lock_checks.get("package-policy.lock") is not True:
        errors.append(
            f"apply run {label!r} lock report does not verify its applied package policy"
        )
    expected_policy_command = _expected_policy_command(
        manager,
        driver,
        expected_desired,
        snapshot=pre_policy,
        preflight_sha256=dnf_preflight_sha256,
    )
    if (
        expected_policy_command is None
        or _planned_lock_mutations(lock_report) != {expected_policy_command}
    ):
        errors.append(
            f"apply run {label!r} lock report does not use the exact frozen-metadata policy mutation"
        )
    if any(
        isinstance(action, dict) and str(action.get("id", "")).startswith("lock.")
        for action in plan_report.get("plan", [])
    ):
        errors.append(
            f"apply run {label!r} post-policy plan still requests package-policy mutation"
        )
    plan_audit = plan_report.get("audit")
    if not isinstance(plan_audit, dict) or not _package_policy_matches_desired(
        manager, driver, plan_audit.get("package_policy")
    ):
        errors.append(
            f"apply run {label!r} post-policy plan does not observe the desired backend selector"
        )
    return errors


def _expected_policy_command(
    manager: str,
    driver: str,
    expected_desired: dict[str, Any] | None,
    *,
    snapshot: dict[str, Any] | None = None,
    preflight_sha256: str | None = None,
) -> tuple[str, ...] | None:
    major_match = re.match(r"^(\d+)", driver)
    if major_match is None:
        return None
    major = major_match.group(1)
    if manager == "apt-get":
        if snapshot is None:
            return None
        selector = driver if "." in driver else major
        target = f"nvidia-driver-pinning-{selector}"
        old_pins = sorted(
            {
                str(package.get("name"))
                for package in snapshot.get("packages") or []
                if isinstance(package, dict)
                and package.get("installed") is True
                and isinstance(package.get("name"), str)
                and package["name"].startswith("nvidia-driver-pinning-")
                and package["name"] != target
            }
        )
        return _expected_local_forward_command(
            snapshot,
            manager,
            remove_specs=old_pins,
        )
    if manager == "dnf":
        if preflight_sha256 is None:
            return None
        suffix = (
            "open"
            if isinstance(expected_desired, dict)
            and str(expected_desired.get("driver", "")).endswith("-open")
            else "dkms"
        )
        return tuple(
            dnf_module_enable_command(
                apply=True,
                stream=f"{major}-{suffix}",
                preflight_sha256=preflight_sha256,
            )
        )
    if manager == "zypper":
        upper_bound = str((int(major) // 10 + 1) * 10)
        return (
            "zypper",
            "--non-interactive",
            "addlock",
            f"*nvidia* >= {upper_bound}",
        )
    return None


def _snapshot_package_multiset(
    snapshot: dict[str, Any],
) -> tuple[Counter[str], bool]:
    records: Counter[str] = Counter()
    identities: set[tuple[Any, ...]] = set()
    duplicate_identity = False
    for package in snapshot.get("packages") or []:
        if not isinstance(package, dict):
            continue
        records.update([_canonical_json(package)])
        identity = (
            package.get("name"),
            package.get("architecture"),
            package.get("epoch"),
            package.get("version"),
        )
        if identity in identities:
            duplicate_identity = True
        identities.add(identity)
    return records, duplicate_identity


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _managed_files_by_path(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(managed_file.get("path")): managed_file
        for managed_file in snapshot.get("managed_files") or []
        if isinstance(managed_file, dict)
        and isinstance(managed_file.get("path"), str)
    }


def _package_policy_matches_desired(
    manager: str, driver: str, policy: Any
) -> bool:
    if not isinstance(policy, dict):
        return False
    if policy.get("backend") != manager or policy.get("observable") is not True:
        return False
    selectors = policy.get("selectors")
    if not isinstance(selectors, list) or not all(
        isinstance(selector, dict) for selector in selectors
    ):
        return False
    major_match = re.match(r"^(\d+)", driver)
    if major_match is None:
        return False
    major = major_match.group(1)
    if manager == "apt-get":
        pin = f"nvidia-driver-pinning-{driver if '.' in driver else major}"
        return selectors == [
            {
                "identifier": pin,
                "name": pin,
                "kind": "package",
                "relation": None,
                "version": None,
                "repositories": [],
            }
        ]
    if manager == "dnf":
        suffix = "open" if driver.endswith("-open") else "dkms"
        return selectors == [
            {
                "identifier": "nvidia-driver",
                "name": "nvidia-driver",
                "kind": "module",
                "relation": "stream",
                "version": f"{major}-{suffix}",
                "repositories": [],
            }
        ]
    if manager == "zypper":
        relevant = [
            selector
            for selector in selectors
            if "nvidia" in str(selector.get("name", "")).lower()
        ]
        upper = str((int(major) // 10 + 1) * 10)
        return bool(
            len(relevant) == 1
            and str(relevant[0].get("identifier", "")).isdigit()
            and int(relevant[0]["identifier"]) > 0
            and relevant[0].get("name") == "*nvidia*"
            and relevant[0].get("kind") == "package"
            and relevant[0].get("relation") == "ge"
            and relevant[0].get("version") == upper
            and relevant[0].get("repositories") == []
        )
    return False


def _verify_embedded_snapshot_integrity(
    report: dict[str, Any], run_label: str, report_path: str
) -> list[str]:
    snapshot = report.get("rollback")
    if not isinstance(snapshot, dict):
        return []
    return _verify_snapshot_integrity(
        snapshot,
        run_label,
        f"snapshot embedded in {report_path}",
    )


def _verify_snapshot_integrity(
    snapshot: dict[str, Any], run_label: str, description: str
) -> list[str]:
    claimed = snapshot.get("integrity_sha256")
    if not isinstance(claimed, str) or re.fullmatch(r"[a-f0-9]{64}", claimed) is None:
        return [f"run {run_label!r} {description} has no valid integrity digest"]
    payload = dict(snapshot)
    payload.pop("integrity_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(actual, claimed):
        return [f"run {run_label!r} {description} failed its integrity check"]
    return []


class _SafeArtifactRedirectHandler(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        destination = urlsplit(newurl)
        if destination.scheme != "https" or destination.username or destination.password:
            raise HTTPError(newurl, code, "unsafe artifact redirect", headers, fp)
        source = urlsplit(req.full_url)
        if (source.scheme, source.hostname, source.port) != (
            destination.scheme,
            destination.hostname,
            destination.port,
        ):
            redirected.remove_header("Authorization")
        return redirected


class _RejectGithubJsonRedirectHandler(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        del req, fp, code, msg, headers, newurl


_GITHUB_JSON_OPENER = build_opener(_RejectGithubJsonRedirectHandler())


def _github_bytes(url: str, token: str) -> bytes:
    if re.fullmatch(
        r"https://api\.github\.com/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"
        r"actions/artifacts/[1-9][0-9]*/zip",
        url,
    ) is None:
        raise RuntimeError("refusing to send credentials to an untrusted artifact URL")
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "nvidia-converge-release-gate",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = build_opener(_SafeArtifactRedirectHandler())
    try:
        with opener.open(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > MAX_ARTIFACT_ARCHIVE_BYTES:
                raise RuntimeError("GitHub artifact ZIP exceeded the safety limit")
            payload = bytes(response.read(MAX_ARTIFACT_ARCHIVE_BYTES + 1))
            final_url = urlsplit(response.geturl())
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"artifact request failed ({type(exc).__name__})"
        ) from exc
    if final_url.scheme != "https":
        raise RuntimeError("GitHub artifact download resolved to an unsafe URL")
    if len(payload) > MAX_ARTIFACT_ARCHIVE_BYTES:
        raise RuntimeError("GitHub artifact ZIP exceeded the safety limit")
    return payload


def _github_json(url: str, token: str) -> Any:
    destination = urlsplit(url)
    if (
        destination.scheme != "https"
        or destination.netloc != "api.github.com"
        or destination.username is not None
        or destination.password is not None
        or destination.fragment
        or not destination.path.startswith("/repos/")
    ):
        raise RuntimeError("refusing to send credentials to an untrusted GitHub API URL")
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "nvidia-converge-release-gate",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with _GITHUB_JSON_OPENER.open(request, timeout=30) as response:
            if getattr(response, "status", None) != 200:
                raise RuntimeError("GitHub API returned an unexpected HTTP status")
            if response.geturl() != url:
                raise RuntimeError("GitHub API metadata request was redirected")
            if response.headers.get_content_type() != "application/json":
                raise RuntimeError("GitHub API returned an unexpected content type")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length, 10)
                except ValueError as exc:
                    raise RuntimeError(
                        "GitHub API returned an invalid Content-Length"
                    ) from exc
                if declared_length < 0 or declared_length > MAX_GITHUB_RESPONSE_BYTES:
                    raise RuntimeError("GitHub API response exceeded the safety limit")
            payload = bytes(response.read(MAX_GITHUB_RESPONSE_BYTES + 1))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"GitHub API request failed ({type(exc).__name__})"
        ) from exc
    if len(payload) > MAX_GITHUB_RESPONSE_BYTES:
        raise RuntimeError("GitHub API response exceeded the safety limit")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("GitHub API returned invalid JSON") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate release promotion evidence.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", EXPECTED_REPOSITORY),
    )
    parser.add_argument("--verify-github", action="store_true")
    parser.add_argument(
        "--qualification-wheel",
        type=Path,
        help="release wheel whose bytes must match the GPU qualification binding",
    )
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--github-api-url",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    args = parser.parse_args(argv)

    path = Path(args.evidence)
    try:
        evidence = json.loads(
            read_bounded_utf8(path, max_bytes=MAX_EVIDENCE_BYTES),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
        jsonschema.validate(
            evidence,
            load_schema("integration-results"),
            format_checker=strict_format_checker(),
        )
    except (
        OSError,
        BoundedFileError,
        json.JSONDecodeError,
        jsonschema.ValidationError,
        ValueError,
        RecursionError,
    ) as exc:
        print(f"release evidence is invalid: {exc}", file=sys.stderr)
        return 2

    errors = check_evidence(
        evidence,
        release=args.release,
        expected_repository=args.repository,
    )
    tested_commit = evidence.get("commit")
    if isinstance(tested_commit, str) and re.fullmatch(r"[0-9a-f]{40}", tested_commit):
        errors.extend(check_commit_provenance(tested_commit, args.commit, path))
    if args.verify_github:
        errors.extend(
            verify_github_evidence(
                evidence,
                token=args.github_token,
                api_url=args.github_api_url,
            )
        )
    if args.qualification_wheel is not None:
        errors.extend(
            check_qualification_wheel_binding(evidence, args.qualification_wheel)
        )
    if errors:
        for error in errors:
            print(f"release evidence is invalid: {error}", file=sys.stderr)
        return 2
    print(f"release evidence passed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
