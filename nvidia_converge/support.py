from __future__ import annotations

import json
from typing import Literal, TypedDict


class PackageManagerSupport(TypedDict):
    audit: bool
    plan: bool
    install: bool
    lock: bool
    rollback: str
    notes: str


class PythonRuntimeSupport(TypedDict):
    minimum_version: str
    workflow_candidates: list[str]
    virtual_environment_required: bool
    root_controlled_for_applied_execution: bool
    notes: str


FeatureName = Literal["audit", "plan", "install", "lock"]
FEATURES: tuple[FeatureName, ...] = ("audit", "plan", "install", "lock")


PACKAGE_MANAGERS: dict[str, PackageManagerSupport] = {
    "apt-get": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "authenticated exact-identity local restore",
        "notes": "Primary supported path for Ubuntu/Debian hosts; retained DEBs are applied in one offline transaction.",
    },
    "dnf": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "authenticated exact-identity local restore",
        "notes": "Uses NVIDIA DNF module streams for branch selection and one repository-disabled retained-RPM transaction.",
    },
    "yum": {
        "audit": True,
        "plan": False,
        "install": False,
        "lock": False,
        "rollback": "authenticated exact-identity local restore",
        "notes": "Audit and rollback only; current RHEL convergence recipes require DNF module streams.",
    },
    "zypper": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "authenticated exact-identity local restore",
        "notes": "Uses RPM inventory, zypper addlock, and one repository-disabled retained-RPM rollback transaction.",
    },
}
PYTHON_RUNTIME: PythonRuntimeSupport = {
    "minimum_version": "3.10",
    "workflow_candidates": ["python3.12", "python3.11", "python3.10", "python3"],
    "virtual_environment_required": True,
    "root_controlled_for_applied_execution": True,
    "notes": (
        "Provision a root-owned, non-group/world-writable Python 3.10 or newer "
        "with venv/ensurepip support; the base "
        "distribution's unversioned python3 is not assumed to be sufficient on "
        "RHEL 8/9 or SLES 15."
    ),
}
CONTAINER_RUNTIMES = ["docker"]
DESIRED_STATE_FIELDS = [
    "role",
    "driver",
    "cuda_compat",
    "secure_boot",
    "container_runtime",
    "container_test_image",
    "fabric_manager",
    "mig",
    "mig_profile",
    "kernel_policy",
]
KNOWN_LIMITS = [
    "release promotion remains blocked until disposable-node GPU evidence is committed for the release tag",
    "implemented package recipes are not production qualification; only exact OS IDs, versions, desired states, and scenarios in a passing release evidence manifest are qualified",
    "first-driver bootstrap and recovery from an unenumerable GPU stack are out of scope; every GPU must be observable through nvidia-smi before mutation",
    "in-place driver branch, exact-version, and open/closed flavor transitions are not qualified and fail closed",
    "exact driver versions are supported only on the APT path",
    "signed NVIDIA and Docker repositories must be provisioned before convergence",
    "package rollback proves authenticated retained bytes for exact package identities, not byte identity with the archives that originally installed those packages",
    "Oracle Linux UEK and 64K kernel dependency recipes are not yet qualified and fail closed",
    "MIG changes are qualified on one GPU with an empty baseline or one full compute instance; mixed, multi-GPU, and more complex geometries fail closed",
]

SUPPORT_MATRIX: dict[str, object] = {
    "schema_version": "1.0",
    "qualification_boundary": {
        "authority": "passing tag-specific integration evidence",
        "scope": "exact observed OS ID/version, desired state, and scenarios only",
        "workflow_targets": [
            "ubuntu-24.04",
            "rhel-family-9-exact-observed-release",
            "sles-or-leap-15-exact-observed-release",
        ],
    },
    "python_runtime": PYTHON_RUNTIME,
    "package_managers": PACKAGE_MANAGERS,
    "container_runtimes": CONTAINER_RUNTIMES,
    "desired_state_fields": DESIRED_STATE_FIELDS,
    "known_limits": KNOWN_LIMITS,
}


def support_json() -> str:
    return json.dumps(SUPPORT_MATRIX, indent=2, sort_keys=True)


def support_human() -> str:
    lines = [
        "nvidia-converge support matrix",
        "Implemented recipes are not production qualification.",
        "A passing tag-specific evidence manifest qualifies only its exact observed targets.",
        "",
        "Python runtime:",
        "- Root-controlled Python 3.10 or newer with venv/ensurepip support",
        "  " + str(PYTHON_RUNTIME["notes"]),
        "",
        "Package managers:",
    ]
    for name, data in PACKAGE_MANAGERS.items():
        features = ", ".join(feature for feature in FEATURES if data[feature])
        lines.append(f"- {name}: {features}; rollback: {data['rollback']}")
        lines.append(f"  {data['notes']}")
    lines.append("")
    lines.append("Container runtimes: " + ", ".join(CONTAINER_RUNTIMES))
    lines.append("")
    lines.append("Known limits:")
    for limit in KNOWN_LIMITS:
        lines.append(f"- {limit}")
    lines.append("")
    lines.append("Use `nvidia-converge support --json` for machine-readable output.")
    return "\n".join(lines)
