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


FeatureName = Literal["audit", "plan", "install", "lock"]
FEATURES: tuple[FeatureName, ...] = ("audit", "plan", "install", "lock")


PACKAGE_MANAGERS: dict[str, PackageManagerSupport] = {
    "apt-get": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "best-effort package version restore",
        "notes": "Primary supported path for Ubuntu/Debian hosts.",
    },
    "dnf": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "best-effort package downgrade",
        "notes": "Requires dnf versionlock plugin for lock enforcement.",
    },
    "yum": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "best-effort package downgrade",
        "notes": "Requires yum versionlock plugin for lock enforcement.",
    },
    "zypper": {
        "audit": True,
        "plan": True,
        "install": True,
        "lock": True,
        "rollback": "best-effort package version restore",
        "notes": "Uses rpm inventory, zypper addlock, and zypper install --oldpackage for rollback.",
    },
}
CONTAINER_RUNTIMES = ["docker"]
DESIRED_STATE_FIELDS = ["role", "driver", "cuda_compat", "secure_boot", "container_runtime", "fabric_manager", "mig", "kernel_policy"]
KNOWN_LIMITS = [
    "apply and rollback are not yet integration-tested on disposable GPU nodes",
    "host-mutation promotion criteria are documented in docs/integration-testing.md",
    "package names may need distro-specific tuning outside Ubuntu/RHEL/SUSE-family defaults",
]

SUPPORT_MATRIX: dict[str, object] = {
    "schema_version": "1.0",
    "package_managers": PACKAGE_MANAGERS,
    "container_runtimes": CONTAINER_RUNTIMES,
    "desired_state_fields": DESIRED_STATE_FIELDS,
    "known_limits": KNOWN_LIMITS,
}


def support_json() -> str:
    return json.dumps(SUPPORT_MATRIX, indent=2, sort_keys=True)


def support_human() -> str:
    lines = ["nvidia-converge support matrix", "", "Package managers:"]
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
