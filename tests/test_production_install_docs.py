import subprocess
import sys
from pathlib import Path

import pytest


def _module_cli(
    *arguments: str,
    isolated: bool = False,
    interpreter: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [str(interpreter or sys.executable)]
    if isolated:
        command.append("-I")
    command.extend(("-m", "nvidia_converge", *arguments))
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def installed_module_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    install_root = tmp_path_factory.mktemp("installed-module")
    dist = install_root / "dist"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist),
        ],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = list(dist.glob("nvidia_converge-*.whl"))
    assert len(wheels) == 1

    environment = install_root / "venv"
    create = subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert create.returncode == 0, create.stdout + create.stderr
    interpreter = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    install = subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            str(wheels[0]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    return interpreter


def test_production_commands_use_isolated_versioned_interpreter() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "do not elevate the setuptools-generated console script" in readme
    assert "NVIDIA_CONVERGE=(" in readme
    assert "/bin/python -I -m nvidia_converge" in readme
    assert 'install -o root -g root -m 0644 desired.yaml "$DESIRED"' in readme
    assert 'install -o root -g root -m 0600 desired.yaml "$DESIRED"' not in readme
    assert 'sudo "${NVIDIA_CONVERGE[@]}" install' in readme
    assert 'sudo "${NVIDIA_CONVERGE[@]}" verify' in readme
    assert 'sudo "${NVIDIA_CONVERGE[@]}" lock' in readme
    assert 'sudo "${NVIDIA_CONVERGE[@]}" snapshot' in readme
    assert 'sudo "$BIN"' not in readme


@pytest.mark.parametrize(
    ("command", "extra_arguments"),
    [
        ("install", ()),
        ("verify", ()),
        ("lock", ()),
        ("snapshot", ()),
        ("rollback", ("--snapshot", "/definitely/missing-snapshot.json")),
    ],
)
def test_nonisolated_module_entry_rejects_applied_commands_before_desired_read(
    command: str,
    extra_arguments: tuple[str, ...],
) -> None:
    result = _module_cli(
        command,
        *extra_arguments,
        "--apply",
        "--desired",
        "/definitely/missing-desired.yaml",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "requires CPython isolated mode" in result.stderr
    assert "cannot read desired" not in result.stderr
    assert "must be run as root" not in result.stderr


def test_isolated_module_entry_reaches_normal_applied_validation(
    installed_module_python: Path,
) -> None:
    result = _module_cli(
        "snapshot",
        "--apply",
        isolated=True,
        interpreter=installed_module_python,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "error: snapshot --apply requires an explicit --desired file\n"
    )
    assert "isolated mode" not in result.stderr


def test_nonisolated_dry_run_remains_available() -> None:
    result = _module_cli(
        "snapshot",
        "--desired",
        "/definitely/missing-desired.yaml",
    )

    assert result.returncode == 2
    assert "cannot read desired" in result.stderr
    assert "isolated mode" not in result.stderr


def test_console_script_entrypoint_cannot_bypass_isolated_mode() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'nvidia-converge = "nvidia_converge.cli:main"' in pyproject

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from nvidia_converge.cli import main; "
                "sys.argv = ['nvidia-converge', 'snapshot', '--apply']; "
                "raise SystemExit(main())"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires CPython isolated mode" in result.stderr


def test_production_install_validates_complete_release_ancestry() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "for directory in /opt/nvidia-converge /opt/nvidia-converge/releases" in readme
    assert 'validate_nvidia_python_path component "$RELEASE_DIR"' in readme
    assert 'validate_nvidia_python_path directory "$RELEASE_DIR"' in readme
    assert 'validate_nvidia_python_path directory "$RELEASE_DIR/bin"' in readme
    assert 'test "$(readlink -f -- "$RELEASE_DIR/bin/python")" = "$PYTHON_BIN"' in readme
    assert '"$RELEASE_DIR/bin/python" -I -m compileall' in readme


def test_upgrade_and_uninstall_preserve_recovery_authority() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "never run `pip install --upgrade`" in readme
    assert "new, verified versioned virtual environment beside" in readme
    assert "paired package-payload bundle" in readme
    assert "throughout every unresolved-operation and rollback window" in readme
    assert "rollback snapshot schema is exactly 2.6" in readme
    assert "do not assume that a newer release can consume an older snapshot" in readme
    assert "exact baseline and forward package identities" in readme
    assert "does not claim that a freshly acquired archive is byte-for-byte identical" in readme
    assert "digest-addressed directory bound to the snapshot path" in readme
    assert "remote repositories or downloads disabled" in readme
    assert "a reachable mutable repository is not rollback authority" in readme
    assert "no journal represents unresolved mutation" in readme
    assert "do not automatically delete `/var/lib/nvidia-converge`" in readme


def test_active_workload_example_directly_follows_its_introduction() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    introduction = readme.index(
        "the stronger `--allow-active-workloads` acknowledgement are required:"
    )
    example = readme.index('  --out "$REPORT_DIR/install-active-$RUN_ID.json"')
    acknowledgement = readme.index("Prefer draining the node instead.", example)
    compensation = readme.index("If an install mutation fails", introduction)

    assert introduction < example < acknowledgement < compensation
