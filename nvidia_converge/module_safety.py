from __future__ import annotations

import os
import stat
from pathlib import Path

_SYS_MODULE_ROOT = Path("/sys/module")
_KNOWN_NVIDIA_MODULES = frozenset(
    {
        "nvidia",
        "nvidia_drm",
        "nvidia_fs",
        "nvidia_modeset",
        "nvidia_peermem",
        "nvidia_uvm",
    }
)


class ModuleDependencyError(RuntimeError):
    """Raised when the loaded NVIDIA module dependency graph is not safe to unload."""


def nvidia_module_unload_order(
    *, sys_module_root: Path = _SYS_MODULE_ROOT
) -> list[str]:
    """Return holders before dependencies and reject unrecognized dependents."""

    try:
        module_names = {entry.name for entry in sys_module_root.iterdir()}
    except OSError as exc:
        raise ModuleDependencyError(
            f"cannot inspect loaded kernel modules: {exc}"
        ) from exc
    loaded = set(_KNOWN_NVIDIA_MODULES & module_names)
    for module in loaded:
        _require_module_directory(sys_module_root, module)
    if not loaded:
        return []

    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visited or module not in loaded:
            return
        if module in visiting:
            raise ModuleDependencyError(
                f"loaded NVIDIA module dependency cycle includes {module}"
            )
        visiting.add(module)
        holders_path = sys_module_root / module / "holders"
        try:
            holders = sorted(entry.name for entry in holders_path.iterdir())
        except OSError as exc:
            raise ModuleDependencyError(
                f"cannot inspect loaded module holders for {module}: {exc}"
            ) from exc
        unknown = [holder for holder in holders if holder not in _KNOWN_NVIDIA_MODULES]
        if unknown:
            raise ModuleDependencyError(
                f"loaded module {module} has unsupported dependents: "
                + ", ".join(unknown)
            )
        for holder in holders:
            _require_module_directory(sys_module_root, holder)
            loaded.add(holder)
            visit(holder)
        visiting.remove(module)
        visited.add(module)
        ordered.append(module)

    # Start at the base driver to follow the actual kernel holder graph. Visit any
    # known disconnected module as well, since leaving it loaded makes a base-driver
    # replacement ambiguous and modprobe -r will otherwise fail partway through.
    visit("nvidia")
    for module in sorted(loaded):
        visit(module)
    return ordered


def _require_module_directory(sys_module_root: Path, module: str) -> None:
    try:
        metadata = os.stat(sys_module_root / module)
    except OSError as exc:
        raise ModuleDependencyError(
            f"loaded module holder {module} disappeared during inspection: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ModuleDependencyError(
            f"loaded module entry {module} is not a directory"
        )
