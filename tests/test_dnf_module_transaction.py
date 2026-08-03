from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nvidia_converge.dnf_module_transaction import (
    DNF_MODULE_ENABLE_SCRIPT,
    DNF_MODULE_UNBOUND_PREFLIGHT,
    DnfModuleEnableProof,
    DnfModuleIdentity,
    DnfModuleRequirement,
    _combined_state_digest,
    _proof_preflight_sha256,
    dnf_module_enable_command,
    parse_dnf_module_enable_proof,
)
from nvidia_converge.models import CommandResult


def test_module_enable_command_uses_one_fixed_check_or_apply_script():
    check = dnf_module_enable_command(apply=False, stream="580-open")
    apply = dnf_module_enable_command(apply=True, stream="580-open")

    assert check == [
        "python3",
        "-I",
        "-c",
        DNF_MODULE_ENABLE_SCRIPT,
        "--check",
        "nvidia-driver",
        "580-open",
    ]
    assert apply[:4] == check[:4]
    assert apply[4:] == [
        "--apply",
        "nvidia-driver",
        "580-open",
        DNF_MODULE_UNBOUND_PREFLIGHT,
    ]
    assert dnf_module_enable_command(
        apply=True,
        stream="580-open",
        preflight_sha256="d" * 64,
    )[4:] == ["--apply", "nvidia-driver", "580-open", "d" * 64]
    compile(DNF_MODULE_ENABLE_SCRIPT, "<dnf-module-enable>", "exec")


@pytest.mark.parametrize(
    "stream",
    [
        "",
        "latest-open",
        "580",
        "580-open/evil",
        "580-open\nother",
        "0580-open",
    ],
)
def test_module_enable_command_rejects_noncanonical_streams(stream):
    with pytest.raises(ValueError, match="invalid exact"):
        dnf_module_enable_command(apply=False, stream=stream)


def test_module_enable_command_rejects_invalid_or_misplaced_binding():
    with pytest.raises(ValueError, match="preflight binding"):
        dnf_module_enable_command(
            apply=True,
            stream="580-open",
            preflight_sha256="not-a-digest",
        )
    with pytest.raises(ValueError, match="check cannot accept"):
        dnf_module_enable_command(
            apply=False,
            stream="580-open",
            preflight_sha256="f" * 64,
        )


@pytest.mark.parametrize("applied", [False, True])
def test_exact_module_enable_proof_is_accepted(applied):
    payload = _proof_payload(applied=applied)
    result = CommandResult(
        ["module-proof"],
        0,
        stdout=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )

    assert parse_dnf_module_enable_proof(
        result,
        applied=applied,
        stream="580-open",
    ) == DnfModuleEnableProof(
        applied=applied,
        target=DnfModuleIdentity(
            name="nvidia-driver",
            stream="580-open",
            version="202608020001",
            context="rhel9",
            architecture="x86_64",
            repository="cuda-rhel9-x86_64",
            yaml_sha256="c" * 64,
        ),
        requirements=(
            DnfModuleRequirement(name="platform", streams=("el9",)),
        ),
        preflight_sha256=payload["preflight_sha256"],
        active_modules_count=1,
        active_modules_sha256="d" * 64,
        repositories_count=1,
        repositories_sha256="e" * 64,
        module_platform_id="platform:el9",
        failsafe_filename="nvidia-driver:580-open:x86_64.yaml",
        failsafe_yaml_sha256="c" * 64,
        module_state_before_sha256="1" * 64,
        module_state_after_sha256=("2" if applied else "1") * 64,
        module_failsafe_before_sha256="3" * 64,
        module_failsafe_after_sha256=("4" if applied else "3") * 64,
        state_before_sha256=_combined_state_digest("1" * 64, "3" * 64),
        state_after_sha256=_combined_state_digest(
            ("2" if applied else "1") * 64,
            ("4" if applied else "3") * 64,
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["changes"]["enable"].update(
            {"dependency": "stable"}
        ),
        lambda payload: payload["changes"].update(
            {"install_profiles": {"nvidia-driver": ["default"]}}
        ),
        lambda payload: payload.update(
            {
                "module_changed_files": [
                    "dependency.module",
                    "nvidia-driver.module",
                ]
            }
        ),
        lambda payload: payload["active_target"].append(
            dict(payload["active_target"][0], context="other")
        ),
        lambda payload: payload["active_target"][0].update(
            {"context": "../../escape"}
        ),
        lambda payload: payload["requirements"].append(
            {"name": "platform", "streams": ["el9"]}
        ),
        lambda payload: payload["requirements"][0].update(
            {"streams": ["el9", "el9"]}
        ),
        lambda payload: payload.update({"schema": True}),
        lambda payload: payload.update({"unexpected": True}),
    ],
)
def test_module_enable_proof_rejects_ambiguous_or_expanded_state(mutation):
    payload = _proof_payload(applied=True)
    mutation(payload)
    result = CommandResult(["module-proof"], 0, stdout=json.dumps(payload))

    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=True,
            stream="580-open",
        )
        is None
    )


def test_module_enable_proof_rejects_wrong_mode_or_digest_relation():
    payload = _proof_payload(applied=True)
    payload["state_after_sha256"] = payload["state_before_sha256"]
    result = CommandResult(["module-proof"], 0, stdout=json.dumps(payload))
    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=True,
            stream="580-open",
        )
        is None
    )

    payload = _proof_payload(applied=False)
    payload["state_after_sha256"] = "b" * 64
    result.stdout = json.dumps(payload)
    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=False,
            stream="580-open",
        )
        is None
    )


def test_module_enable_proof_rejects_tampered_or_wrong_expected_binding():
    payload = _proof_payload(applied=False)
    payload["active_modules_sha256"] = "9" * 64
    result = CommandResult(["module-proof"], 0, stdout=json.dumps(payload))
    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=False,
            stream="580-open",
        )
        is None
    )

    payload = _proof_payload(applied=True)
    result.stdout = json.dumps(payload)
    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=True,
            stream="580-open",
            preflight_sha256="9" * 64,
        )
        is None
    )


def test_module_enable_proof_rejects_duplicate_json_keys_and_truncation():
    duplicate = json.dumps(_proof_payload(applied=False)).replace(
        '"schema": 2',
        '"schema": 2, "schema": 2',
    )
    assert (
        parse_dnf_module_enable_proof(
            CommandResult(["module-proof"], 0, stdout=duplicate),
            applied=False,
            stream="580-open",
        )
        is None
    )
    truncated = json.dumps(_proof_payload(applied=False)) + (
        "\n[output truncated: retained first 10 of 20 bytes]"
    )
    assert (
        parse_dnf_module_enable_proof(
            CommandResult(["module-proof"], 0, stdout=truncated),
            applied=False,
            stream="580-open",
        )
        is None
    )


def test_fixed_script_check_proves_exact_target_without_persisting(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
    )

    assert result.returncode == 0
    assert not list(state_directory.iterdir())
    proof = parse_dnf_module_enable_proof(
        result,
        applied=False,
        stream="580-open",
    )
    assert proof is not None
    assert proof.module_platform_id == "platform:el9"
    assert proof.failsafe_filename == "nvidia-driver:580-open:x86_64.yaml"


def test_fixed_script_check_does_not_create_native_dnf_rpmdb_lock(
    tmp_path,
    monkeypatch,
    capsys,
):
    events: list[str] = []
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        lock_events=events,
    )

    assert result.returncode == 0
    assert events == []
    assert not list(state_directory.iterdir())


def test_fixed_script_apply_holds_and_releases_native_dnf_rpmdb_lock(
    tmp_path,
    monkeypatch,
    capsys,
):
    events: list[str] = []
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        lock_events=events,
    )

    assert result.returncode == 0
    assert events == ["built", "entered", "exited"]
    assert (state_directory / "nvidia-driver.module").exists()


def test_fixed_script_refuses_busy_native_dnf_rpmdb_lock_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
):
    events: list[str] = []
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        lock_events=events,
        refuse_rpmdb_lock=True,
    )

    assert result.returncode == 2
    assert "RPMDB already locked" in result.stderr
    assert events == ["built", "entered"]
    assert not list(state_directory.iterdir())


def test_fixed_script_apply_persists_and_reobserves_only_target_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
    )

    assert result.returncode == 0
    assert [path.name for path in state_directory.iterdir()] == [
        "nvidia-driver.module"
    ]
    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=True,
            stream="580-open",
        )
        is not None
    )


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        (DNF_MODULE_UNBOUND_PREFLIGHT, "lacks a bound preflight proof"),
        ("f" * 64, "does not match the accepted preflight proof"),
    ],
)
def test_fixed_script_apply_refuses_unbound_or_stale_proof_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
    binding,
    message,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        preflight_sha256=binding,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_refuses_a_repository_missing_from_cached_universe(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        drop_repository_from_cache=True,
    )

    assert result.returncode == 2
    assert "every configured enabled repository" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_refuses_unrelated_active_identity_shift(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        active_unrelated=True,
        unrelated_context_shift=True,
    )

    assert result.returncode == 2
    assert "unrelated active module identity" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_detects_identical_unrelated_state_file_rewrite(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        rewrite_unrelated_identically=True,
    )

    assert result.returncode == 2
    assert "outside the exact target file" in result.stderr
    assert (state_directory / "nodejs.module").exists()


def test_fixed_script_detects_transient_state_file_during_check(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        transient_during_enable=True,
    )

    assert result.returncode == 2
    assert "changed during dependency preflight" in result.stderr
    assert not (state_directory / "transient.module").exists()


def test_fixed_script_refuses_recursive_dependency_enable_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        extra_enable={"dependency": "stable"},
    )

    assert result.returncode == 2
    assert "exact target-only" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_preserves_unrelated_persisted_state_view(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        active_unrelated=True,
    )

    assert result.returncode == 0
    assert [path.name for path in state_directory.iterdir()] == [
        "nvidia-driver.module"
    ]
    assert (
        parse_dnf_module_enable_proof(
            result,
            applied=True,
            stream="580-open",
        )
        is not None
    )


def test_fixed_script_refuses_dependency_state_removal_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        baseline_pending_enable={"nodejs": "20"},
    )

    assert result.returncode == 2
    assert "pre-existing pending state change" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_refuses_already_enabled_target_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        baseline_pending_enable={"nvidia-driver": "580-open"},
    )

    assert result.returncode == 2
    assert "pre-existing pending state change" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_accepts_an_exact_explicit_platform_override(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, _state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        module_platform_override="platform:el9",
    )

    proof = parse_dnf_module_enable_proof(
        result,
        applied=False,
        stream="580-open",
    )
    assert proof is not None
    assert proof.module_platform_id == "platform:el9"


def test_fixed_script_uses_unique_installed_platform_when_available_is_ambiguous(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, _state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        available_platform_ids=("platform:el8", "platform:el9"),
        installed_platform_ids=("platform:el9",),
    )

    proof = parse_dnf_module_enable_proof(
        result,
        applied=False,
        stream="580-open",
    )
    assert proof is not None
    assert proof.module_platform_id == "platform:el9"


def test_fixed_script_rejects_unobservable_platform_identity(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        available_platform_ids=(),
        installed_platform_ids=(),
    )

    assert result.returncode == 2
    assert "platform ID is not uniquely inspectable" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_stale_platform_binding_refuses_apply_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        change_platform_after_check=True,
    )

    assert result.returncode == 2
    assert "does not match the accepted preflight proof" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_restores_umask_and_persists_world_readable_state(
    tmp_path,
    monkeypatch,
    capsys,
):
    original_umask = os.umask(0o077)
    try:
        result, state_directory = _run_fixed_script(
            tmp_path,
            monkeypatch,
            capsys,
            applied=True,
        )
        observed_umask = os.umask(0o077)
    finally:
        os.umask(original_umask)

    assert result.returncode == 0
    assert observed_umask == 0o077
    assert stat.S_IMODE(
        (state_directory / "nvidia-driver.module").stat().st_mode
    ) == 0o644


def test_fixed_script_rejects_module_state_directory_metadata_drift(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, _state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=True,
        change_state_directory_mode_during_save=True,
    )

    assert result.returncode == 2
    assert "replaced or changed the module state directory" in result.stderr


def test_fixed_script_requires_preexisting_trusted_failsafe_directory(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        missing_failsafe_directory=True,
    )

    assert result.returncode == 2
    assert "modulefailsafe" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_rejects_noncanonical_persistdir(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        wrong_persistdir=True,
    )

    assert result.returncode == 2
    assert "persistdir is not the supported exact path" in result.stderr
    assert not list(state_directory.iterdir())


def test_fixed_script_rejects_noncanonical_installroot_from_main_config(
    tmp_path,
    monkeypatch,
    capsys,
):
    result, state_directory = _run_fixed_script(
        tmp_path,
        monkeypatch,
        capsys,
        applied=False,
        wrong_installroot=True,
    )

    assert result.returncode == 2
    assert "installroot is not the supported exact path" in result.stderr
    assert not list(state_directory.iterdir())


def _proof_payload(*, applied: bool) -> dict[str, object]:
    module_state_before = "1" * 64
    module_state_after = ("2" if applied else "1") * 64
    module_failsafe_before = "3" * 64
    module_failsafe_after = ("4" if applied else "3") * 64
    payload: dict[str, object] = {
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
        "requirements": [{"name": "platform", "streams": ["el9"]}],
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


def _run_fixed_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    applied: bool,
    extra_enable: dict[str, str] | None = None,
    baseline_pending_enable: dict[str, str] | None = None,
    active_unrelated: bool = False,
    unrelated_context_shift: bool = False,
    drop_repository_from_cache: bool = False,
    preflight_sha256: str | None = None,
    rewrite_unrelated_identically: bool = False,
    transient_during_enable: bool = False,
    module_platform_override: str | None = None,
    change_platform_after_check: bool = False,
    available_platform_ids: tuple[str, ...] = ("platform:el9",),
    installed_platform_ids: tuple[str, ...] = ("platform:el9",),
    missing_failsafe_directory: bool = False,
    wrong_persistdir: bool = False,
    wrong_installroot: bool = False,
    change_state_directory_mode_during_save: bool = False,
    lock_events: list[str] | None = None,
    refuse_rpmdb_lock: bool = False,
) -> tuple[CommandResult, Path]:
    state_directory = tmp_path / "modules.d"
    state_directory.mkdir()
    persist_directory = tmp_path / "persist"
    failsafe_directory = persist_directory / "modulefailsafe"
    if missing_failsafe_directory:
        persist_directory.mkdir()
    else:
        failsafe_directory.mkdir(parents=True)
    if active_unrelated:
        (failsafe_directory / "nodejs:20:x86_64.yaml").write_text(
            "document: modulemd\nname: nodejs\n",
            encoding="utf-8",
        )
    if rewrite_unrelated_identically:
        (state_directory / "nodejs.module").write_text(
            "[nodejs]\nname=nodejs\nprofiles=\nstate=enabled\nstream=20\n",
            encoding="utf-8",
        )
    state = {"base_instances": 0, "persisted": False}
    observed_lock_events = lock_events if lock_events is not None else []
    state_enabled = object()

    class FakeModule:
        def __init__(
            self,
            *,
            module_id,
            name,
            stream,
            context,
            repository="cuda-rhel9-x86_64",
        ):
            self.module_id = module_id
            self.name = name
            self.stream = stream
            self.context = context
            self.repository = repository

        def getId(self):
            return self.module_id

        def getName(self):
            return self.name

        def getStream(self):
            return self.stream

        def getVersion(self):
            return "202608020001"

        def getContext(self):
            return self.context

        def getArch(self):
            return "x86_64"

        def getRepoID(self):
            return self.repository

        def getYaml(self):
            return f"document: modulemd\nname: {self.name}\n"

        def getModuleDependencies(self):
            return []

    class FakePlatformModule(FakeModule):
        def __init__(self):
            super().__init__(
                module_id=3,
                name="platform",
                stream="el9",
                context="00000000",
                repository="@System",
            )

        def getVersion(self):
            return "0"

        def getArch(self):
            return "noarch"

        def getYaml(self):
            raise AssertionError("synthetic platform YAML must not be inspected")

    class FakeContainer:
        def __init__(self):
            self.enable: dict[str, str] = dict(baseline_pending_enable or {})
            self.target_active = state["persisted"]
            self.target_module = FakeModule(
                module_id=1,
                name="nvidia-driver",
                stream="580-open",
                context="rhel9",
            )
            self.unrelated_module = FakeModule(
                module_id=2,
                name="nodejs",
                stream="20",
                context="rhel9",
                repository="appstream",
            )
            self.platform_module = FakePlatformModule()

        def getDisabledModules(self):
            return []

        def getEnabledStreams(self):
            return self.enable

        def getInstalledProfiles(self, name=None):
            return {} if name is None else []

        def getRemovedProfiles(self):
            return {}

        def getResetModules(self):
            return []

        def getSwitchedStreams(self):
            return {}

        def isModuleActive(self, module_id):
            return (module_id == 1 and self.target_active) or (
                module_id == 2 and active_unrelated
            ) or module_id == 3

        def isEnabled(self, name, stream):
            if name == "nvidia-driver" and stream == "580-open":
                return self.target_active and (
                    state["persisted"]
                    or self.enable.get("nvidia-driver") == "580-open"
                )
            return name == "nodejs" and stream == "20" and active_unrelated

        def getModulePackages(self):
            modules = [self.target_module, self.platform_module]
            if active_unrelated:
                modules.append(self.unrelated_module)
            return modules

        def getModuleState(self, name):
            assert name == "nvidia-driver"
            return state_enabled if state["persisted"] else object()

        def getEnabledStream(self, name):
            assert name == "nvidia-driver"
            return "580-open" if state["persisted"] else ""

        def save(self):
            if rewrite_unrelated_identically:
                unrelated = state_directory / "nodejs.module"
                replacement = state_directory / "nodejs.tmp"
                replacement.write_text(
                    unrelated.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                replacement.replace(unrelated)
            (state_directory / "nvidia-driver.module").write_text(
                "[nvidia-driver]\n"
                "name=nvidia-driver\n"
                "profiles=\n"
                "state=enabled\n"
                "stream=580-open\n",
                encoding="utf-8",
            )
            if change_state_directory_mode_during_save:
                state_directory.chmod(0o700)
            state["persisted"] = True

        def updateFailSafeData(self):
            target = failsafe_directory / "nvidia-driver:580-open:x86_64.yaml"
            target.write_text(
                self.target_module.getYaml(),
                encoding="utf-8",
            )
            target.chmod(0o644)

    release_packages = [
        SimpleNamespace(
            provides=["system-release", f"base-module({platform_id})"],
        )
        for platform_id in available_platform_ids
    ]
    installed_release_packages = [
        SimpleNamespace(
            provides=["system-release", f"base-module({platform_id})"],
        )
        for platform_id in installed_platform_ids
    ]
    changed_release_package = SimpleNamespace(
        provides=["system-release", "base-module(platform:el8)"],
    )

    class FakeQuery:
        def __init__(self, base):
            self.base = base

        def filter(self, **filters):
            assert filters == {"provides": "system-release", "latest": 1}
            return self

        def available(self):
            if change_platform_after_check and self.base.instance > 1:
                return [changed_release_package]
            return release_packages

        def installed(self):
            return installed_release_packages

    class FakeBase:
        def __init__(self):
            state["base_instances"] += 1
            self.instance = state["base_instances"]
            self._moduleContainer = FakeContainer()
            self.repos = SimpleNamespace(iter_enabled=self._iter_enabled)
            self.cache_loaded = False

            class FakeConf:
                def __init__(self):
                    self.loaded = False
                    self.module_platform_id = None
                    self.persistdir = str(persist_directory)
                    self.installroot = "/"

                def read(self):
                    self.loaded = True
                    self.module_platform_id = module_platform_override
                    if wrong_persistdir:
                        self.persistdir = str(tmp_path / "wrong-persist")
                    if wrong_installroot:
                        self.installroot = str(tmp_path / "installroot")

            self.conf = FakeConf()
            self.sack = SimpleNamespace(query=lambda: FakeQuery(self))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def read_all_repos(self):
            assert self.conf.loaded is True

        def _iter_enabled(self):
            if self.cache_loaded and drop_repository_from_cache:
                return iter(())
            return iter([SimpleNamespace(id="cuda-rhel9-x86_64")])

        def fill_sack_from_repos_in_cache(self, *, load_system_repo):
            assert load_system_repo is True
            self.cache_loaded = True

        def do_transaction(self):
            raise AssertionError("module helper must not call Base.do_transaction")

    class FakeModuleBase:
        def __init__(self, base):
            self.base = base

        def enable(self, specs):
            assert specs == ["nvidia-driver:580-open"]
            container = self.base._moduleContainer
            container.enable = {
                **container.enable,
                "nvidia-driver": "580-open",
                **(extra_enable or {}),
            }
            container.target_active = True
            if unrelated_context_shift:
                container.unrelated_module.context = "shifted"
            if transient_during_enable:
                transient = state_directory / "transient.module"
                transient.write_text("transient\n", encoding="utf-8")
                transient.unlink()

        def get_modules(self, spec):
            assert spec == "nvidia-driver:580-open"
            return (
                [self.base._moduleContainer.target_module],
                SimpleNamespace(
                    name="nvidia-driver",
                    stream="580-open",
                    version=None,
                    context=None,
                    arch=None,
                    profile=None,
                ),
            )

    dnf = ModuleType("dnf")
    dnf.Base = FakeBase
    dnf_lock = ModuleType("dnf.lock")

    class FakeRpmdbLock:
        def __enter__(self):
            observed_lock_events.append("entered")
            if refuse_rpmdb_lock:
                raise RuntimeError("RPMDB already locked")

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            observed_lock_events.append("exited")

    def build_rpmdb_lock(persistdir, exit_on_lock):
        assert persistdir == str(persist_directory)
        assert exit_on_lock is True
        observed_lock_events.append("built")
        return FakeRpmdbLock()

    dnf_lock.build_rpmdb_lock = build_rpmdb_lock
    dnf.lock = dnf_lock
    dnf_module = ModuleType("dnf.module")
    module_base = ModuleType("dnf.module.module_base")
    module_base.ModuleBase = FakeModuleBase
    module_base.STATE_ENABLED = state_enabled
    monkeypatch.setitem(sys.modules, "dnf", dnf)
    monkeypatch.setitem(sys.modules, "dnf.lock", dnf_lock)
    monkeypatch.setitem(sys.modules, "dnf.module", dnf_module)
    monkeypatch.setitem(sys.modules, "dnf.module.module_base", module_base)
    script = DNF_MODULE_ENABLE_SCRIPT.replace(
        'STATE_DIRECTORY = "/etc/dnf/modules.d"',
        f"STATE_DIRECTORY = {str(state_directory)!r}",
    ).replace(
        'PERSIST_DIRECTORY = "/var/lib/dnf"',
        f"PERSIST_DIRECTORY = {str(persist_directory)!r}",
    ).replace(
        'FAILSAFE_DIRECTORY = "/var/lib/dnf/modulefailsafe"',
        f"FAILSAFE_DIRECTORY = {str(failsafe_directory)!r}",
    ).replace(
        "metadata.st_uid != 0",
        "metadata.st_uid not in {0, os.geteuid()}",
    ).replace(
        "opened.st_uid != 0",
        "opened.st_uid not in {0, os.geteuid()}",
    ).replace(
        "if os.geteuid() != 0:",
        "if False:",
    )
    def execute(arguments):
        monkeypatch.setattr(sys, "argv", arguments)
        returncode = 0
        try:
            exec(  # noqa: S102  # nosec B102 - fixed script in a fake DNF runtime
                compile(script, "<dnf-module-enable-test>", "exec"),
                {},
            )
        except SystemExit as exc:
            returncode = int(exc.code or 0)
        captured = capsys.readouterr()
        return CommandResult(
            ["module-proof"],
            returncode,
            stdout=captured.out,
            stderr=captured.err,
        )

    if applied and preflight_sha256 is None:
        check_result = execute(
            [
                "dnf-module-proof",
                "--check",
                "nvidia-driver",
                "580-open",
            ]
        )
        check_proof = parse_dnf_module_enable_proof(
            check_result,
            applied=False,
            stream="580-open",
        )
        preflight_sha256 = (
            check_proof.preflight_sha256 if check_proof is not None else "0" * 64
        )
    arguments = [
        "dnf-module-proof",
        "--apply" if applied else "--check",
        "nvidia-driver",
        "580-open",
    ]
    if applied:
        arguments.append(preflight_sha256 or DNF_MODULE_UNBOUND_PREFLIGHT)
    result = execute(arguments)
    return (
        result,
        state_directory,
    )
