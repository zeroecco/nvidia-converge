import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.bundle_release_evidence import (
    ReleaseEvidenceBundleError,
    _write_release_evidence_bundle,
    main,
)


def test_release_evidence_bundle_retains_manifest_archives_and_checksums(
    tmp_path,
):
    payload = _artifact_zip(b"sanitized evidence")
    evidence = _evidence(payload)
    output = tmp_path / "gpu-evidence.zip"

    _write_release_evidence_bundle(
        evidence,
        output,
        verified_artifacts={42: payload},
    )

    with zipfile.ZipFile(output) as bundle:
        assert bundle.namelist() == [
            "integration-results.json",
            "artifacts/ubuntu-24.04/42-gpu-integration-ubuntu.zip",
            "bundle-manifest.json",
            "SHA256SUMS",
        ]
        retained = bundle.read(
            "artifacts/ubuntu-24.04/42-gpu-integration-ubuntu.zip"
        )
        with zipfile.ZipFile(io.BytesIO(retained)) as canonical:
            assert canonical.namelist() == ["attestation.json"]
            assert canonical.read("attestation.json") == b"sanitized evidence"
        manifest = json.loads(bundle.read("integration-results.json"))
        assert manifest == evidence
        bundle_manifest = json.loads(bundle.read("bundle-manifest.json"))
        assert bundle_manifest["qualification_wheel"] == evidence[
            "qualification_wheel"
        ]
        retained_metadata = bundle_manifest["artifacts"][0]
        assert retained_metadata["source_sha256"] == hashlib.sha256(payload).hexdigest()
        assert retained_metadata["retained_sha256"] == hashlib.sha256(retained).hexdigest()
        checksums = bundle.read("SHA256SUMS").decode("ascii")
        assert hashlib.sha256(retained).hexdigest() in checksums
        assert "integration-results.json" in checksums
        assert "bundle-manifest.json" in checksums


def test_release_evidence_bundle_rejects_digest_change(tmp_path):
    payload = _artifact_zip(b"sanitized evidence")
    evidence = _evidence(payload)

    with pytest.raises(ReleaseEvidenceBundleError, match="source digest"):
        _write_release_evidence_bundle(
            evidence,
            tmp_path / "gpu-evidence.zip",
            verified_artifacts={42: _artifact_zip(b"changed")},
        )

    assert not (tmp_path / "gpu-evidence.zip").exists()


def test_release_evidence_bundle_refuses_overwrite(tmp_path):
    payload = _artifact_zip(b"sanitized evidence")
    output = tmp_path / "gpu-evidence.zip"
    output.write_bytes(b"existing")

    with pytest.raises(ReleaseEvidenceBundleError, match="already exists"):
        _write_release_evidence_bundle(
            _evidence(payload),
            output,
            verified_artifacts={42: payload},
        )

    assert output.read_bytes() == b"existing"


def test_release_evidence_bundle_is_byte_reproducible(tmp_path):
    payload = _artifact_zip(b"sanitized evidence")
    evidence = _evidence(payload)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    for output in (first, second):
        _write_release_evidence_bundle(
            evidence,
            output,
            verified_artifacts={42: payload},
        )

    assert first.read_bytes() == second.read_bytes()


def test_release_evidence_bundle_strips_opaque_source_zip_bytes(tmp_path):
    secret = b"RAW-HOST-IDENTITY-MUST-NOT-SURVIVE"
    opaque = secret + _artifact_zip(b"sanitized evidence") + secret
    output = tmp_path / "gpu-evidence.zip"

    _write_release_evidence_bundle(
        _evidence(opaque),
        output,
        verified_artifacts={42: opaque},
    )

    retained_bundle = output.read_bytes()
    assert secret not in retained_bundle


def test_release_evidence_bundle_rejects_concurrent_target_creation(
    monkeypatch, tmp_path
):
    payload = _artifact_zip(b"sanitized evidence")
    output = tmp_path / "gpu-evidence.zip"

    def race_link(_source, destination, *, follow_symlinks):
        assert follow_symlinks is False
        output.write_bytes(b"racer")
        raise FileExistsError(destination)

    monkeypatch.setattr("scripts.bundle_release_evidence.os.link", race_link)

    with pytest.raises(ReleaseEvidenceBundleError, match="already exists"):
        _write_release_evidence_bundle(
            _evidence(payload),
            output,
            verified_artifacts={42: payload},
        )

    assert output.read_bytes() == b"racer"
    assert not list(tmp_path.glob(".*.tmp"))


def test_release_evidence_bundle_cli_uses_only_verified_artifact_bytes(
    monkeypatch, tmp_path
):
    payload = _artifact_zip(b"sanitized evidence")
    evidence = json.loads(
        Path("integrations/results.example.json").read_text(encoding="utf-8")
    )
    evidence["release"] = "v0.1.0"
    evidence["commit"] = "a" * 40
    evidence["runs"][0]["artifacts"][0]["sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    wheel = tmp_path / "nvidia_converge-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"qualified-wheel")
    wheel_binding = {
        "name": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    evidence["qualification_wheel"] = wheel_binding
    evidence["runs"][0]["qualification_wheel"] = wheel_binding
    evidence_path = tmp_path / "results.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output = tmp_path / "gpu-evidence.zip"

    monkeypatch.setattr(
        "scripts.bundle_release_evidence.check_evidence",
        lambda evidence, release, expected_repository: [],
    )
    monkeypatch.setattr(
        "scripts.bundle_release_evidence.check_commit_provenance",
        lambda tested_commit, release_commit, evidence_path: [],
    )

    def verify(evidence, *, token, api_url, verified_artifacts):
        assert token == "token"
        assert api_url == "https://api.github.com"
        verified_artifacts[1] = payload
        return []

    monkeypatch.setattr(
        "scripts.bundle_release_evidence.verify_github_evidence",
        verify,
    )

    assert (
        main(
            [
                "--evidence",
                str(evidence_path),
                "--release",
                "v0.1.0",
                "--commit",
                "b" * 40,
                "--repository",
                "zeroecco/nvidia-converge",
                    "--output",
                    str(output),
                    "--qualification-wheel",
                    str(wheel),
                    "--github-token",
                "token",
            ]
        )
        == 0
    )
    assert output.is_file()


def test_release_evidence_bundle_module_entrypoint_is_runnable():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.bundle_release_evidence", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Bundle validated sanitized GPU evidence" in result.stdout


def _artifact_zip(payload: bytes) -> bytes:
    from io import BytesIO

    target = BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("attestation.json", payload)
    return target.getvalue()


def _evidence(payload: bytes):
    return {
        "schema_version": "1.5",
        "repository": "zeroecco/nvidia-converge",
        "release": "v0.1.0",
        "commit": "a" * 40,
        "qualification_wheel": {
            "name": "nvidia_converge-0.1.0-py3-none-any.whl",
            "sha256": "f" * 64,
        },
        "runs": [
            {
                "matrix_id": "ubuntu-24.04",
                "qualification_wheel": {
                    "name": "nvidia_converge-0.1.0-py3-none-any.whl",
                    "sha256": "f" * 64,
                },
                "artifacts": [
                    {
                        "id": 42,
                        "name": "gpu-integration-ubuntu",
                        "uri": (
                            "https://api.github.com/repos/zeroecco/"
                            "nvidia-converge/actions/artifacts/42/zip"
                        ),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ],
    }
