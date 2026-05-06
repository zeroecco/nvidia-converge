from __future__ import annotations

import json


SUPPORT_MATRIX = {
    "schema_version": "1.0",
    "package_managers": {
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
            "rollback": "not implemented",
            "notes": "Uses rpm inventory and zypper addlock; rollback command generation is not implemented.",
        },
    },
    "container_runtimes": ["docker"],
    "desired_state_fields": ["role", "driver", "cuda_compat", "secure_boot", "container_runtime", "fabric_manager", "mig", "kernel_policy"],
    "known_limits": [
        "apply and rollback are not yet integration-tested on disposable GPU nodes",
        "package names may need distro-specific tuning outside Ubuntu/RHEL/SUSE-family defaults",
        "release artifacts include checksums but no signed provenance attestations yet",
    ],
}


def support_json() -> str:
    return json.dumps(SUPPORT_MATRIX, indent=2, sort_keys=True)


def support_human() -> str:
    lines = ["nvidia-converge support matrix", "", "Package managers:"]
    for name, data in SUPPORT_MATRIX["package_managers"].items():
        features = ", ".join(feature for feature in ("audit", "plan", "install", "lock") if data[feature])
        lines.append(f"- {name}: {features}; rollback: {data['rollback']}")
        lines.append(f"  {data['notes']}")
    lines.append("")
    lines.append("Container runtimes: " + ", ".join(SUPPORT_MATRIX["container_runtimes"]))
    lines.append("")
    lines.append("Known limits:")
    for limit in SUPPORT_MATRIX["known_limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    lines.append("Use `nvidia-converge support --json` for machine-readable output.")
    return "\n".join(lines)
