from pathlib import Path

import pytest

from nvidia_converge.models import CommandResult
from nvidia_converge.module_safety import (
    ModuleDependencyError,
    nvidia_module_unload_order,
)
from nvidia_converge.verify import prepare_stack


def test_unload_order_follows_loaded_holder_graph(tmp_path):
    root = _module_graph(tmp_path)

    assert nvidia_module_unload_order(sys_module_root=root) == [
        "nvidia_drm",
        "nvidia_modeset",
        "nvidia_peermem",
        "nvidia_fs",
        "nvidia_uvm",
        "nvidia",
    ]


def test_unknown_loaded_nvidia_dependent_fails_closed(tmp_path):
    root = _module_graph(tmp_path)
    (root / "nvidia" / "holders" / "third_party_gpu_client").mkdir()

    with pytest.raises(ModuleDependencyError, match="unsupported dependents"):
        nvidia_module_unload_order(sys_module_root=root)


def test_unobservable_module_inventory_fails_closed(tmp_path):
    with pytest.raises(ModuleDependencyError, match="cannot inspect loaded kernel modules"):
        nvidia_module_unload_order(sys_module_root=tmp_path / "missing")


def test_forced_reload_unloads_all_known_dependents_before_base(tmp_path):
    root = _module_graph(tmp_path)
    runner = _ModuleRunner([0, 0, 0, 0, 0, 0, 0])

    check = prepare_stack(runner, force_reload=True, sys_module_root=root)

    assert check.name == "module.reload"
    assert check.ok is True
    assert runner.commands == [
        [
            "modprobe",
            "-r",
            "nvidia_drm",
            "nvidia_modeset",
            "nvidia_peermem",
            "nvidia_fs",
            "nvidia_uvm",
            "nvidia",
        ],
        ["modprobe", "nvidia"],
        ["modprobe", "nvidia_uvm"],
        ["modprobe", "nvidia_fs"],
        ["modprobe", "nvidia_peermem"],
        ["modprobe", "nvidia_modeset"],
        ["modprobe", "nvidia_drm"],
    ]


def test_forced_reload_does_not_load_after_unload_failure(tmp_path):
    root = _module_graph(tmp_path)
    runner = _ModuleRunner([1])

    check = prepare_stack(runner, force_reload=True, sys_module_root=root)

    assert check.ok is False
    assert len(runner.commands) == 1
    assert runner.commands[0][:2] == ["modprobe", "-r"]


def test_forced_reload_reports_partial_dependent_restore(tmp_path):
    root = _module_graph(tmp_path)
    runner = _ModuleRunner([0, 0, 0, 1])

    check = prepare_stack(runner, force_reload=True, sys_module_root=root)

    assert check.ok is False
    assert runner.commands[-1] == ["modprobe", "nvidia_fs"]
    assert ["modprobe", "nvidia_drm"] not in runner.commands


def test_forced_reload_refuses_unknown_dependent_without_mutating(tmp_path):
    root = _module_graph(tmp_path)
    (root / "nvidia_uvm" / "holders" / "unrecognized").mkdir()
    runner = _ModuleRunner([])

    check = prepare_stack(runner, force_reload=True, sys_module_root=root)

    assert check.ok is False
    assert check.command is not None
    assert "unsupported dependents" in check.command.stderr
    assert runner.commands == []


def _module_graph(tmp_path: Path) -> Path:
    root = tmp_path / "sys-module"
    modules = (
        "nvidia",
        "nvidia_drm",
        "nvidia_fs",
        "nvidia_modeset",
        "nvidia_peermem",
        "nvidia_uvm",
    )
    for module in modules:
        (root / module / "holders").mkdir(parents=True)
    for dependency, holder in (
        ("nvidia", "nvidia_modeset"),
        ("nvidia", "nvidia_peermem"),
        ("nvidia", "nvidia_uvm"),
        ("nvidia_modeset", "nvidia_drm"),
        ("nvidia_uvm", "nvidia_fs"),
    ):
        (root / dependency / "holders" / holder).mkdir()
    return root


class _ModuleRunner:
    def __init__(self, returncodes: list[int]):
        self.returncodes = returncodes
        self.commands: list[list[str]] = []

    def exists(self, name: str) -> bool:
        return name == "modprobe"

    def run(self, command, *, mutate=False, allow_fail=True):
        del mutate, allow_fail
        self.commands.append(command)
        return CommandResult(command, self.returncodes.pop(0))
