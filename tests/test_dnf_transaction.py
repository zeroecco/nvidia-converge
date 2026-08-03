from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nvidia_converge.dnf_transaction import (
    DNF_LOCAL_TRANSACTION_SCRIPT,
    dnf_local_transaction_command,
)
from nvidia_converge.models import CommandResult


def test_local_dnf_command_uses_the_fixed_script():
    command = dnf_local_transaction_command(
        apply=True,
        restore_paths=["/payload/driver.rpm"],
        remove_specs=[],
        expected_installs=["driver-1.0-1.x86_64"],
        expected_removals=[],
    )

    assert command[:5] == [
        "python3",
        "-I",
        "-c",
        DNF_LOCAL_TRANSACTION_SCRIPT,
        "--apply",
    ]
    compile(DNF_LOCAL_TRANSACTION_SCRIPT, "<dnf-local-transaction>", "exec")


def test_local_dnf_apply_suppresses_one_failsafe_update_and_restores_method(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state, container_class, original = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        updater_calls=1,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "install": ["driver-1.0-1.x86_64"],
        "remove": [],
    }
    assert state == {"real_updater_calls": 0, "transaction_calls": 1}
    assert container_class.updateFailSafeData is original


def test_local_dnf_rejects_pending_module_delta_before_transaction(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state, _container_class, _original = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        pending_enable={"nodejs": "20"},
    )

    assert result.returncode == 2
    assert "pending DNF module-state delta" in result.stderr
    assert state["transaction_calls"] == 0


def test_local_dnf_rejects_authoritative_changed_flag_with_empty_getters(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state, _container_class, _original = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        authoritative_changed=True,
    )

    assert result.returncode == 2
    assert "authoritative DNF module-state delta" in result.stderr
    assert state["transaction_calls"] == 0


def test_local_dnf_refuses_when_swig_class_cannot_be_patched(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state, _container_class, _original = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        patchable=False,
    )

    assert result.returncode == 2
    assert "cannot suppress module fail-safe updates" in result.stderr
    assert state["transaction_calls"] == 0


@pytest.mark.parametrize("updater_calls", [0, 2])
def test_local_dnf_requires_exactly_one_suppressed_updater_call(
    tmp_path,
    monkeypatch,
    capsys,
    updater_calls,
):
    result, state, container_class, original = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        updater_calls=updater_calls,
    )

    assert result.returncode == 2
    assert "exactly once" in result.stderr
    assert state["transaction_calls"] == 1
    assert state["real_updater_calls"] == 0
    assert container_class.updateFailSafeData is original


@pytest.mark.parametrize(
    ("changed_tree", "message"),
    [
        ("modules", "changed DNF module state"),
        ("failsafe", "changed DNF module fail-safe state"),
        ("modules-transient", "changed DNF module state"),
        ("failsafe-transient", "changed DNF module fail-safe state"),
    ],
)
def test_local_dnf_rejects_any_module_tree_change(
    tmp_path,
    monkeypatch,
    capsys,
    changed_tree,
    message,
):
    result, state, container_class, original = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        changed_tree=changed_tree,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert state["transaction_calls"] == 1
    assert state["real_updater_calls"] == 0
    assert container_class.updateFailSafeData is original


def _run_fixed_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    updater_calls: int = 1,
    pending_enable: dict[str, str] | None = None,
    patchable: bool = True,
    changed_tree: str | None = None,
    authoritative_changed: bool = False,
) -> tuple[CommandResult, dict[str, int], type, object]:
    modules_directory = tmp_path / "modules.d"
    modules_directory.mkdir()
    persist_directory = tmp_path / "persist"
    failsafe_directory = persist_directory / "modulefailsafe"
    failsafe_directory.mkdir(parents=True)
    state = {"real_updater_calls": 0, "transaction_calls": 0}

    class ContainerMeta(type):
        def __setattr__(cls, name, value):
            if name == "updateFailSafeData" and not patchable:
                raise TypeError("read-only SWIG wrapper")
            super().__setattr__(name, value)

    class FakeContainer(metaclass=ContainerMeta):
        def getDisabledModules(self):
            return []

        def getEnabledStreams(self):
            return dict(pending_enable or {})

        def getInstalledProfiles(self):
            return {}

        def getRemovedProfiles(self):
            return {}

        def getResetModules(self):
            return []

        def getSwitchedStreams(self):
            return {}

        def isChanged(self):
            return authoritative_changed or bool(pending_enable)

        def updateFailSafeData(self):
            state["real_updater_calls"] += 1
            (failsafe_directory / "unexpected:stable:x86_64.yaml").write_text(
                "unexpected\n",
                encoding="utf-8",
            )

    original_updater = FakeContainer.updateFailSafeData

    package = SimpleNamespace(
        name="driver",
        epoch=None,
        version="1.0",
        release="1",
        arch="x86_64",
    )

    class FakeBase:
        def __init__(self):
            self.conf = SimpleNamespace(persistdir=str(persist_directory))
            self._moduleContainer = FakeContainer()
            self.transaction = SimpleNamespace(install_set=[], remove_set=[])
            self.sack = SimpleNamespace(
                query=lambda: SimpleNamespace(installed=list)
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def fill_sack(self, *, load_system_repo, load_available_repos):
            assert load_system_repo is True
            assert load_available_repos is False

        def add_remote_rpms(self, paths, *, strict):
            assert paths == [str(tmp_path / "driver.rpm")]
            assert strict is True
            return [package]

        def package_signature_check(self, observed_package):
            assert observed_package is package
            return 0, ""

        def package_install(self, observed_package):
            assert observed_package is package
            self.transaction.install_set = [package]

        def remove(self, spec, *, forms):
            raise AssertionError((spec, forms))

        def resolve(self, *, allow_erasing):
            assert allow_erasing is False

        def do_transaction(self):
            state["transaction_calls"] += 1
            for _ in range(updater_calls):
                self._moduleContainer.updateFailSafeData()
            if changed_tree == "modules":
                (modules_directory / "unexpected.module").write_text(
                    "unexpected\n",
                    encoding="utf-8",
                )
            if changed_tree == "failsafe":
                (failsafe_directory / "unexpected:stable:x86_64.yaml").write_text(
                    "unexpected\n",
                    encoding="utf-8",
                )
            if changed_tree == "modules-transient":
                transient = modules_directory / "transient.module"
                transient.write_text("transient\n", encoding="utf-8")
                transient.unlink()
            if changed_tree == "failsafe-transient":
                transient = failsafe_directory / "transient:stable:x86_64.yaml"
                transient.write_text("transient\n", encoding="utf-8")
                transient.unlink()

    dnf = ModuleType("dnf")
    dnf.Base = FakeBase
    hawkey = ModuleType("hawkey")
    hawkey.FORM_NEVRA = object()
    rpm = ModuleType("rpm")
    rpm.labelCompare = lambda left, right: (left > right) - (left < right)
    monkeypatch.setitem(sys.modules, "dnf", dnf)
    monkeypatch.setitem(sys.modules, "hawkey", hawkey)
    monkeypatch.setitem(sys.modules, "rpm", rpm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dnf-local-transaction",
            "--apply",
            str(tmp_path / "driver.rpm"),
            "--remove",
            "--expect-install",
            "driver-1.0-1.x86_64",
            "--expect-remove",
        ],
    )
    script = DNF_LOCAL_TRANSACTION_SCRIPT.replace(
        'MODULE_STATE_DIRECTORY = "/etc/dnf/modules.d"',
        f"MODULE_STATE_DIRECTORY = {str(modules_directory)!r}",
    ).replace(
        "metadata.st_uid != 0",
        "metadata.st_uid not in {0, os.geteuid()}",
    ).replace(
        "opened.st_uid != 0",
        "opened.st_uid not in {0, os.geteuid()}",
    )
    returncode = 0
    try:
        exec(  # noqa: S102  # nosec B102 - fixed script in a fake DNF runtime
            compile(script, "<dnf-local-transaction-test>", "exec"),
            {},
        )
    except SystemExit as exc:
        returncode = int(exc.code or 0)
    captured = capsys.readouterr()
    return (
        CommandResult(
            ["dnf-local-transaction"],
            returncode,
            stdout=captured.out,
            stderr=captured.err,
        ),
        state,
        FakeContainer,
        original_updater,
    )
