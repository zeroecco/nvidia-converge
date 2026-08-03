import io
import os
import subprocess
from pathlib import Path

import pytest

import nvidia_converge.runner as runner_module
from nvidia_converge.runner import CommandRunner, _BoundedCapture

_TEST_SYSTEM_EXECUTABLE = os.path.realpath("/usr/bin/true")


def _fake_popen(captured, *, stdout=b"", stderr=b"", timeout=False):
    simulate_timeout = timeout

    class FakeProcess:
        pid = 424242

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            self.returncode = 0
            self.stdout = io.BytesIO(stdout)
            self.stderr = io.BytesIO(stderr)
            self.stdin = io.BytesIO() if kwargs["stdin"] == subprocess.PIPE else None
            self._timed_out = False

        def wait(self, timeout=None):
            captured["timeout"] = timeout
            if simulate_timeout and not self._timed_out:
                self._timed_out = True
                raise subprocess.TimeoutExpired(captured["command"], timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    return FakeProcess


def _patch_process(monkeypatch, captured, **kwargs):
    monkeypatch.setattr(
        "nvidia_converge.runner.shutil.which",
        lambda name, path: _TEST_SYSTEM_EXECUTABLE,
    )
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(captured, **kwargs))
    monkeypatch.setattr("nvidia_converge.runner._process_group_exists", lambda group_id: False)


def test_mutating_commands_use_extended_timeout_and_noninteractive_environment(monkeypatch):
    captured = {}
    _patch_process(monkeypatch, captured, stdout=b"ok\n")

    result = CommandRunner(apply=True, timeout=10, mutation_timeout=900).run(
        ["apt-get", "install", "-y", "driver"], mutate=True
    )

    assert result.returncode == 0
    assert result.stdout == "ok"
    assert captured["timeout"] == 900
    assert captured["env"]["LC_ALL"] == "C"
    assert captured["env"]["DEBIAN_FRONTEND"] == "noninteractive"
    assert captured["env"]["NEEDRESTART_MODE"] == "l"
    assert captured["executable"] == _TEST_SYSTEM_EXECUTABLE
    assert captured["cwd"] == "/"
    assert captured["start_new_session"] is True


def test_read_only_commands_keep_short_timeout(monkeypatch):
    captured = {}
    _patch_process(monkeypatch, captured)
    monkeypatch.delenv("DEBIAN_FRONTEND", raising=False)

    CommandRunner(timeout=15, mutation_timeout=900).run(["nvidia-smi"])

    assert captured["timeout"] == 15
    assert "DEBIAN_FRONTEND" not in captured["env"]


def test_timeout_is_explicit_in_command_result(monkeypatch):
    captured = {}
    _patch_process(monkeypatch, captured, stderr=b"partial error\n", timeout=True)

    def fake_terminate(process):
        process.returncode = -15

    monkeypatch.setattr("nvidia_converge.runner._terminate_process_group", fake_terminate)
    result = CommandRunner(timeout=7).run(["nvidia-smi"])

    assert result.returncode == 124
    assert result.reason == "timeout-process-group-terminated"
    assert result.stderr == "partial error\ncommand process group timed out after 7 seconds and was terminated"


def test_package_timeout_warns_that_recovery_may_be_required(monkeypatch):
    captured = {}
    _patch_process(monkeypatch, captured, timeout=True)
    monkeypatch.setattr("nvidia_converge.runner._terminate_process_group", lambda process: None)

    result = CommandRunner(apply=True, mutation_timeout=1).run(["apt-get", "install", "-y", "driver"], mutate=True)

    assert "package database recovery may be required" in result.stderr


def test_runner_drops_dynamic_loader_and_python_injection_environment(monkeypatch):
    captured = {}
    _patch_process(monkeypatch, captured)
    monkeypatch.setenv("PATH", "/tmp/poison")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/poison.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/poison-python")
    monkeypatch.setenv("HOME", "/tmp/poison-home")

    CommandRunner().run(["nvidia-smi"])

    assert captured["env"]["PATH"] != "/tmp/poison"
    assert "LD_PRELOAD" not in captured["env"]
    assert "PYTHONPATH" not in captured["env"]
    assert captured["env"]["HOME"] != "/tmp/poison-home"


def test_runner_refuses_relative_executable_paths():
    result = CommandRunner(apply=True).run(["./apt-get", "install"], mutate=True)
    assert result.returncode == 127
    assert "trusted executable not found" in result.stderr


def test_runner_refuses_absolute_executable_outside_trusted_path(tmp_path):
    executable = tmp_path / "nvidia-smi"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    result = CommandRunner().run([str(executable)])

    assert result.returncode == 127
    assert "trusted executable not found" in result.stderr


@pytest.fixture
def trusted_executable_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    owner_uid = os.geteuid()
    monkeypatch.setattr(
        runner_module,
        "_TRUSTED_EXECUTABLE_OWNER_UID",
        owner_uid,
    )
    monkeypatch.setattr(
        runner_module,
        "_TRUSTED_EXECUTABLE_ANCESTOR_UIDS",
        frozenset({0, owner_uid}),
    )
    executable_root = tmp_path / "trusted-bin"
    executable_root.mkdir(mode=0o755)
    executable = executable_root / "trusted-tool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable_root, executable


def test_resolver_accepts_an_owned_nonwritable_executable_tree(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, executable = trusted_executable_tree

    resolved = CommandRunner(executable_path=str(executable_root)).resolve_executable(
        executable.name
    )

    assert resolved == str(executable.resolve(strict=True))


def test_resolver_allows_a_trusted_symlink_within_the_same_search_root(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, executable = trusted_executable_tree
    alias = executable_root / "trusted-alias"
    alias.symlink_to(executable.name)

    resolved = CommandRunner(executable_path=str(executable_root)).resolve_executable(
        alias.name
    )

    assert resolved == str(executable.resolve(strict=True))


def test_resolver_allows_a_trusted_search_root_alias(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, executable = trusted_executable_tree
    alias_root = executable_root.parent / "trusted-bin-alias"
    alias_root.symlink_to(executable_root, target_is_directory=True)

    resolved = CommandRunner(executable_path=str(alias_root)).resolve_executable(
        executable.name
    )

    assert resolved == str(executable.resolve(strict=True))


def test_resolver_rejects_a_symlink_that_escapes_the_search_root(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, _ = trusted_executable_tree
    outside_root = executable_root.parent / "outside-bin"
    outside_root.mkdir(mode=0o755)
    outside = outside_root / "outside-tool"
    outside.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    outside.chmod(0o755)
    alias = executable_root / "escaped-tool"
    alias.symlink_to(outside)

    assert (
        CommandRunner(executable_path=str(executable_root)).resolve_executable(
            alias.name
        )
        is None
    )


def test_resolver_rejects_an_absolute_alias_outside_the_search_root(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, executable = trusted_executable_tree
    alias_root = executable_root.parent / "alias-bin"
    alias_root.mkdir(mode=0o755)
    alias = alias_root / "trusted-tool"
    alias.symlink_to(executable)

    assert (
        CommandRunner(executable_path=str(executable_root)).resolve_executable(
            str(alias)
        )
        is None
    )


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_resolver_rejects_group_or_world_writable_executable_modes(
    trusted_executable_tree: tuple[Path, Path],
    mode: int,
) -> None:
    executable_root, executable = trusted_executable_tree
    executable.chmod(mode)

    assert (
        CommandRunner(executable_path=str(executable_root)).resolve_executable(
            executable.name
        )
        is None
    )


def test_resolver_rejects_a_non_executable_regular_file(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, executable = trusted_executable_tree
    executable.chmod(0o644)

    assert (
        CommandRunner(executable_path=str(executable_root)).resolve_executable(
            str(executable)
        )
        is None
    )


def test_resolver_rejects_an_untrusted_executable_owner(
    trusted_executable_tree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_root, executable = trusted_executable_tree
    monkeypatch.setattr(
        runner_module,
        "_TRUSTED_EXECUTABLE_OWNER_UID",
        os.geteuid() + 1,
    )

    assert (
        CommandRunner(executable_path=str(executable_root)).resolve_executable(
            executable.name
        )
        is None
    )


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_resolver_rejects_group_or_world_writable_search_roots(
    trusted_executable_tree: tuple[Path, Path],
    mode: int,
) -> None:
    executable_root, executable = trusted_executable_tree
    executable_root.chmod(mode)

    assert (
        CommandRunner(executable_path=str(executable_root)).resolve_executable(
            executable.name
        )
        is None
    )


def test_resolver_rejects_a_search_root_below_an_unsafe_ancestor(
    trusted_executable_tree: tuple[Path, Path],
) -> None:
    executable_root, executable = trusted_executable_tree
    unsafe_ancestor = executable_root.parent / "unsafe-ancestor"
    unsafe_ancestor.mkdir(mode=0o777)
    unsafe_ancestor.chmod(0o777)
    alias = unsafe_ancestor / "bin"
    alias.symlink_to(executable_root, target_is_directory=True)

    assert (
        CommandRunner(executable_path=str(alias)).resolve_executable(executable.name)
        is None
    )


@pytest.mark.parametrize("path_value", ["", ":/usr/bin", "relative/bin"])
def test_resolver_rejects_malformed_search_paths(path_value: str) -> None:
    assert CommandRunner(executable_path=path_value).resolve_executable("true") is None


def test_output_capture_is_bounded():
    output = _BoundedCapture(limit=16)
    output.add(b"a" * 64)
    captured = output.text()
    assert len(output.data) == 16
    assert captured.startswith("a" * 16)
    assert "retained first 16 of 64 bytes" in captured


def test_result_callback_receives_every_recorded_result():
    observed = []
    runner = CommandRunner(
        apply=False,
        result_callback=lambda result, mutate: observed.append((result, mutate)),
    )

    result = runner.run(["apt-get", "install"], mutate=True)

    assert observed == [(result, True)]
    assert result.skipped is True


def test_private_state_subcommand_is_rejected_outside_exact_scope() -> None:
    with pytest.raises(RuntimeError, match="exact package staging scope"):
        CommandRunner().run_private_state(["apt-get", "--download-only", "install"])


@pytest.mark.parametrize(
    "scope",
    [
        [],
        ["stage-package-payloads", "--unsafe"],
        ["persist-rollback-snapshot"],
    ],
)
def test_private_state_scope_rejects_every_near_miss(scope: list[str]) -> None:
    with (
        pytest.raises(ValueError, match="unsupported private-state"),
        CommandRunner().private_state_scope(scope),
    ):
        pass


def test_private_state_scope_rejects_nesting() -> None:
    runner = CommandRunner()

    with (
        runner.private_state_scope(["stage-package-payloads"]),
        pytest.raises(RuntimeError, match="already active"),
        runner.private_state_scope(["stage-package-payloads"]),
    ):
        pass


def test_private_state_scope_emits_only_outer_callbacks_and_uses_long_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    _patch_process(monkeypatch, captured)
    starts: list[tuple[list[str], bool]] = []
    results: list[tuple[list[str], bool]] = []
    runner = CommandRunner(
        apply=True,
        timeout=5,
        mutation_timeout=901,
        start_callback=lambda command, mutate: starts.append((command, mutate)),
        result_callback=lambda result, mutate: results.append(
            (result.command, mutate)
        ),
    )

    with runner.private_state_scope(["stage-package-payloads"]):
        nested = runner.run_private_state(["apt-get", "--download-only", "install"])

    assert nested.returncode == 0
    assert captured["timeout"] == 901
    assert starts == [(["stage-package-payloads"], True)]
    assert results == [(["stage-package-payloads"], True)]
    assert [result.command for result in runner.results] == [
        ["apt-get", "--download-only", "install"],
        ["stage-package-payloads"],
    ]


def test_private_state_subcommand_exception_restores_runner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[tuple[list[str], bool]] = []
    results: list[tuple[list[str], bool]] = []
    start_callback = lambda command, mutate: starts.append((command, mutate))
    result_callback = lambda result, mutate: results.append((result.command, mutate))
    runner = CommandRunner(
        timeout=7,
        mutation_timeout=902,
        start_callback=start_callback,
        result_callback=result_callback,
    )

    def fail_run(*_args, **_kwargs):
        assert runner.timeout == 902
        assert runner.start_callback is None
        assert runner.result_callback is None
        raise RuntimeError("subcommand failed before a result")

    monkeypatch.setattr(runner, "run", fail_run)
    with (
        pytest.raises(RuntimeError, match="subcommand failed"),
        runner.private_state_scope(["stage-package-payloads"]),
    ):
        runner.run_private_state(["apt-get", "--download-only", "install"])

    assert runner.timeout == 7
    assert runner.start_callback is start_callback
    assert runner.result_callback is result_callback
    assert starts == [(["stage-package-payloads"], True)]
    assert results == [(["stage-package-payloads"], True)]
