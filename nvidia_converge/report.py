from __future__ import annotations

import json
from pathlib import Path

from .models import HostAudit, PackageInfo, Report


def write_report(report: Report, path: str | None) -> None:
    text = report_json(report)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def report_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def sbom_from_audit(audit: HostAudit) -> list[PackageInfo]:
    sbom = list(audit.packages)
    if audit.module.version:
        sbom.append(PackageInfo(name="nvidia-kernel-module", version=audit.module.version, manager="kernel", installed=audit.module.loaded))
    sbom.append(PackageInfo(name="linux-kernel", version=audit.kernel.running, manager="kernel", installed=True))
    return sbom
