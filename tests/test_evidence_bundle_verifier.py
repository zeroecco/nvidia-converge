import hashlib
import io
import json
import stat
import struct
import subprocess
import sys
import zipfile

import pytest

import scripts.verify_release_evidence_bundle as verifier
from scripts.bundle_release_evidence import (
    _canonical_artifact_archive,
    _write_release_evidence_bundle,
    _zip_info,
)
from scripts.verify_release_evidence_bundle import (
    main,
    verify_release_evidence_bundle,
)
from tests.test_release_gate import (
    _archives_for,
    _passing_evidence,
    _rewrite_attested_json_member,
)

RELEASE = "v0.1.0"
REPOSITORY = "zeroecco/nvidia-converge"
QUALIFICATION_WHEEL_NAME = "nvidia_converge-0.1.0-py3-none-any.whl"
QUALIFICATION_WHEEL_BYTES = b"qualified-wheel-bytes"


@pytest.fixture(scope="module")
def production_bundle(tmp_path_factory):
    evidence = _passing_evidence()
    qualification_digest = hashlib.sha256(QUALIFICATION_WHEEL_BYTES).hexdigest()
    evidence["qualification_wheel"]["sha256"] = qualification_digest
    for run in evidence["runs"]:
        run["qualification_wheel"]["sha256"] = qualification_digest
    archives = _archives_for(evidence)
    for run in evidence["runs"]:
        artifact = run["artifacts"][0]
        artifact["sha256"] = hashlib.sha256(archives[artifact["id"]]).hexdigest()
    output = tmp_path_factory.mktemp("offline-evidence") / "evidence.zip"
    _write_release_evidence_bundle(
        evidence,
        output,
        verified_artifacts=archives,
    )
    return output.read_bytes()


def test_offline_verifier_accepts_production_shaped_canonical_bundle(
    production_bundle,
    tmp_path,
    capsys,
):
    path = _write_bundle(tmp_path, production_bundle)
    wheel = tmp_path / QUALIFICATION_WHEEL_NAME
    wheel.write_bytes(QUALIFICATION_WHEEL_BYTES)

    assert verify_release_evidence_bundle(
        path,
        release=RELEASE,
        repository=REPOSITORY,
    ) == []
    assert main(
        [
            "--bundle",
            str(path),
            "--release",
            RELEASE,
            "--repository",
            REPOSITORY,
            "--qualification-wheel",
            str(wheel),
        ]
    ) == 0
    assert "GitHub API provenance was not re-queried" in capsys.readouterr().out


def test_offline_verifier_cli_help_states_network_provenance_limit():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.verify_release_evidence_bundle", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "without network access" in result.stdout
    assert "does not query GitHub" in result.stdout


def test_offline_verifier_cli_requires_release_wheel_binding(
    production_bundle,
    tmp_path,
):
    path = _write_bundle(tmp_path, production_bundle)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--bundle",
                str(path),
                "--release",
                RELEASE,
                "--repository",
                REPOSITORY,
            ]
        )

    assert exc_info.value.code == 2


def test_offline_verifier_checks_a_supplied_release_wheel(
    production_bundle,
    tmp_path,
    monkeypatch,
):
    path = _write_bundle(tmp_path, production_bundle)
    wheel = tmp_path / "nvidia_converge-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"release-wheel")
    observed = []

    def check(evidence, wheel_path):
        observed.append((evidence["qualification_wheel"], wheel_path))
        return []

    monkeypatch.setattr(verifier, "check_qualification_wheel_binding", check)

    assert verify_release_evidence_bundle(
        path,
        release=RELEASE,
        repository=REPOSITORY,
        qualification_wheel_path=wheel,
    ) == []
    assert len(observed) == 1
    assert observed[0][0]["name"] == wheel.name
    assert observed[0][1] == wheel


def test_offline_verifier_returns_exit_two_for_invalid_bundle(tmp_path, capsys):
    path = tmp_path / "invalid.zip"
    path.write_bytes(b"not a zip")
    wheel = tmp_path / QUALIFICATION_WHEEL_NAME
    wheel.write_bytes(QUALIFICATION_WHEEL_BYTES)

    assert main(
        [
            "--bundle",
            str(path),
            "--release",
            RELEASE,
            "--repository",
            REPOSITORY,
            "--qualification-wheel",
            str(wheel),
        ]
    ) == 2
    assert "offline release evidence bundle is invalid" in capsys.readouterr().err


def test_offline_verifier_rejects_retained_artifact_tampering(
    production_bundle,
    tmp_path,
):
    members = _member_map(production_bundle)
    manifest = json.loads(members["bundle-manifest.json"])
    retained = manifest["artifacts"][0]["retained_path"]
    members[retained] += b"tampered"
    payload = _canonical_outer(_ordered_members(members, manifest))

    errors = _verify_bytes(payload, tmp_path)

    assert any("digest does not match its bundle manifest" in error for error in errors)


def test_offline_verifier_rejects_bundle_qualification_wheel_rebinding(
    production_bundle,
    tmp_path,
):
    members = _member_map(production_bundle)
    manifest = json.loads(members["bundle-manifest.json"])
    manifest["qualification_wheel"]["sha256"] = "e" * 64
    payload = _reseal_outer(members, manifest)

    errors = _verify_bytes(payload, tmp_path)

    assert any("qualification_wheel" in error for error in errors)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_offline_verifier_rejects_extra_or_missing_outer_members(
    production_bundle,
    tmp_path,
    mutation,
):
    entries = _members_in_order(production_bundle)
    if mutation == "extra":
        entries.insert(
            -2,
            (
                "artifacts/ubuntu-24.04/999-unclaimed.zip",
                _canonical_zip([("attestation.json", b"{}\n")]),
            ),
        )
    else:
        del entries[1]

    errors = _verify_bytes(_canonical_outer(entries), tmp_path)

    assert any("inventory or order" in error for error in errors)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "artifacts\\escape.zip",
        "artifacts/control\x01/file.zip",
    ],
)
def test_offline_verifier_rejects_unsafe_outer_paths(tmp_path, unsafe_name):
    payload = _canonical_outer([(unsafe_name, b"x")])

    errors = _verify_bytes(payload, tmp_path)

    assert any("unsafe ZIP path" in error for error in errors)


def test_offline_verifier_rejects_duplicate_outer_paths(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        payload = _canonical_outer(
            [
                ("integration-results.json", b"{}\n"),
                ("integration-results.json", b"{}\n"),
            ]
        )

    errors = _verify_bytes(payload, tmp_path)

    assert any("duplicate path" in error for error in errors)


@pytest.mark.parametrize(
    "metadata", ["deflate", "symlink", "fifo", "extra", "comment"]
)
def test_offline_verifier_rejects_noncanonical_outer_metadata(
    production_bundle,
    tmp_path,
    metadata,
):
    entries = _members_in_order(production_bundle)
    payload = _outer_with_metadata(entries, metadata)

    errors = _verify_bytes(payload, tmp_path)

    expected = {
        "deflate": "unsupported compression",
        "symlink": "symbolic link",
        "fifo": "not a regular file",
        "extra": "canonical",
        "comment": "canonical",
    }[metadata]
    assert any(expected in error for error in errors)


def test_offline_verifier_rejects_encrypted_member_flag(
    production_bundle,
    tmp_path,
):
    payload = bytearray(production_bundle)
    local = payload.find(b"PK\x03\x04")
    with zipfile.ZipFile(io.BytesIO(production_bundle)) as archive:
        central = payload.find(b"PK\x01\x02", archive.start_dir)
    assert local >= 0 and central >= 0
    struct.pack_into("<H", payload, local + 6, 1)
    struct.pack_into("<H", payload, central + 8, 1)

    errors = _verify_bytes(bytes(payload), tmp_path)

    assert any("is encrypted" in error for error in errors)


@pytest.mark.parametrize("suffix", [b"trailing", b"PK\x05\x06covert"])
def test_offline_verifier_rejects_trailing_bytes(
    production_bundle,
    tmp_path,
    suffix,
):
    errors = _verify_bytes(production_bundle + suffix, tmp_path)

    assert any(
        "trailing bytes" in error or "end record is not canonical" in error
        or "not a zip file" in error
        for error in errors
    )


def test_offline_verifier_rejects_zip_preamble(production_bundle, tmp_path):
    errors = _verify_bytes(b"covert-preamble" + production_bundle, tmp_path)

    assert any("noncanonical offset" in error for error in errors)


def test_offline_verifier_rejects_entry_count_bomb(tmp_path):
    entries = [
        (f"artifacts/ubuntu-24.04/{index}-artifact-{index}.zip", b"x")
        for index in range(1, verifier.MAX_BUNDLE_ARTIFACTS + 5)
    ]

    errors = _verify_bytes(_canonical_outer(entries), tmp_path)

    assert any("entry count is outside" in error for error in errors)


def test_offline_verifier_rejects_member_size_bomb(
    production_bundle,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(verifier, "MAX_EVIDENCE_BYTES", 32)

    errors = _verify_bytes(production_bundle, tmp_path)

    assert any("exceeds the safety limit" in error for error in errors)


@pytest.mark.parametrize("checksum_mutation", ["uppercase", "one-space", "duplicate", "self", "nul"])
def test_offline_verifier_strictly_parses_sha256sums(
    production_bundle,
    tmp_path,
    checksum_mutation,
):
    entries = _members_in_order(production_bundle)
    checksum_index = next(
        index for index, (name, _payload) in enumerate(entries) if name == "SHA256SUMS"
    )
    checksum = entries[checksum_index][1].decode("ascii")
    first = checksum.splitlines()[0]
    if checksum_mutation == "uppercase":
        checksum = first[:64].upper() + first[64:] + "\n" + "\n".join(
            checksum.splitlines()[1:]
        ) + "\n"
    elif checksum_mutation == "one-space":
        checksum = checksum.replace("  ", " ", 1)
    elif checksum_mutation == "duplicate":
        checksum += first + "\n"
    elif checksum_mutation == "self":
        checksum += f"{'0' * 64}  SHA256SUMS\n"
    else:
        checksum = checksum.replace("integration-results.json", "integration\x00-results.json", 1)
    entries[checksum_index] = ("SHA256SUMS", checksum.encode("ascii"))

    errors = _verify_bytes(_canonical_outer(entries), tmp_path)

    assert any("SHA256SUMS" in error or "unsafe ZIP path" in error for error in errors)


def test_offline_verifier_rejects_duplicate_json_keys(
    production_bundle,
    tmp_path,
):
    entries = _members_in_order(production_bundle)
    evidence_index = next(
        index
        for index, (name, _payload) in enumerate(entries)
        if name == "integration-results.json"
    )
    original = entries[evidence_index][1]
    duplicate = b'{\n  "release": "v0.1.0",' + original[1:]
    entries[evidence_index] = ("integration-results.json", duplicate)

    errors = _verify_bytes(_canonical_outer(entries), tmp_path)

    assert any("duplicate object key 'release'" in error for error in errors)


def test_offline_verifier_binds_bundle_manifest_to_integration_identity(
    production_bundle,
    tmp_path,
):
    members = _member_map(production_bundle)
    manifest = json.loads(members["bundle-manifest.json"])
    manifest["artifacts"][0]["artifact_name"] = "substituted"
    members["bundle-manifest.json"] = _json_bytes(manifest)
    payload = _reseal_outer(members, manifest)

    errors = _verify_bytes(payload, tmp_path)

    assert any("artifact entry 0 mismatch" in error for error in errors)


@pytest.mark.parametrize("metadata", ["deflate", "comment", "directory"])
def test_offline_verifier_rejects_noncanonical_inner_archive(
    production_bundle,
    tmp_path,
    metadata,
):
    members = _member_map(production_bundle)
    manifest = json.loads(members["bundle-manifest.json"])
    retained = manifest["artifacts"][0]["retained_path"]
    inner_entries = _members_in_order(members[retained])
    if metadata == "directory":
        inner_entries.append(("reports/", b""))
        replacement = _canonical_zip(inner_entries)
    else:
        replacement = _outer_with_metadata(inner_entries, metadata)
    members[retained] = replacement
    payload = _reseal_outer(members, manifest)

    errors = _verify_bytes(payload, tmp_path)

    assert any("is not canonical" in error for error in errors)


def test_offline_verifier_deeply_validates_retained_reports(
    production_bundle,
    tmp_path,
):
    members = _member_map(production_bundle)
    manifest = json.loads(members["bundle-manifest.json"])
    retained = manifest["artifacts"][0]["retained_path"]

    def change_command(report):
        report["command"] = "plan"
        return report

    mutated_source = _rewrite_attested_json_member(
        members[retained],
        "reports/doctor.json",
        change_command,
    )
    members[retained] = _canonical_artifact_archive(mutated_source)
    payload = _reseal_outer(members, manifest)

    errors = _verify_bytes(payload, tmp_path)

    assert any("references 'reports/doctor.json' with the wrong command" in error for error in errors)


def test_offline_verifier_rejects_cross_run_operation_id_reuse(
    production_bundle,
    tmp_path,
):
    members = _member_map(production_bundle)
    manifest = json.loads(members["bundle-manifest.json"])
    first_path = manifest["artifacts"][0]["retained_path"]
    second_path = manifest["artifacts"][1]["retained_path"]
    first_operation = json.loads(
        _member_map(members[first_path])["reports/doctor.json"]
    )["operation_id"]

    def reuse_operation_id(report):
        report["operation_id"] = first_operation
        return report

    mutated_source = _rewrite_attested_json_member(
        members[second_path],
        "reports/doctor.json",
        reuse_operation_id,
    )
    members[second_path] = _canonical_artifact_archive(mutated_source)
    payload = _reseal_outer(members, manifest)

    errors = _verify_bytes(payload, tmp_path)

    assert any("operation ID" in error and "is reused by" in error for error in errors)


def test_offline_verifier_rejects_symlink_bundle_path(
    production_bundle,
    tmp_path,
):
    target = _write_bundle(tmp_path, production_bundle, name="target.zip")
    link = tmp_path / "link.zip"
    link.symlink_to(target)

    errors = verify_release_evidence_bundle(
        link,
        release=RELEASE,
        repository=REPOSITORY,
    )

    assert errors


def _verify_bytes(payload, tmp_path):
    path = _write_bundle(tmp_path, payload)
    return verify_release_evidence_bundle(
        path,
        release=RELEASE,
        repository=REPOSITORY,
    )


def _write_bundle(tmp_path, payload, *, name="bundle.zip"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _members_in_order(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return [(entry.filename, archive.read(entry)) for entry in archive.infolist()]


def _member_map(payload):
    return dict(_members_in_order(payload))


def _canonical_zip(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, payload in entries:
            archive.writestr(_zip_info(name), payload)
    return output.getvalue()


def _canonical_outer(entries):
    return _canonical_zip(entries)


def _outer_with_metadata(entries, metadata):
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for index, (name, payload) in enumerate(entries):
            info = _zip_info(name)
            if index == 0 and metadata == "deflate":
                info.compress_type = zipfile.ZIP_DEFLATED
            elif index == 0 and metadata == "symlink":
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            elif index == 0 and metadata == "fifo":
                info.external_attr = (stat.S_IFIFO | 0o644) << 16
            elif index == 0 and metadata == "extra":
                info.extra = b"\x01\x00\x00\x00"
            archive.writestr(info, payload)
        if metadata == "comment":
            archive.comment = b"covert"
    return output.getvalue()


def _ordered_members(members, manifest):
    return [
        ("integration-results.json", members["integration-results.json"]),
        *[
            (item["retained_path"], members[item["retained_path"]])
            for item in manifest["artifacts"]
        ],
        ("bundle-manifest.json", members["bundle-manifest.json"]),
        ("SHA256SUMS", members["SHA256SUMS"]),
    ]


def _reseal_outer(members, manifest):
    for item in manifest["artifacts"]:
        item["retained_sha256"] = hashlib.sha256(
            members[item["retained_path"]]
        ).hexdigest()
    members["bundle-manifest.json"] = _json_bytes(manifest)
    checksum_paths = [
        "integration-results.json",
        *[item["retained_path"] for item in manifest["artifacts"]],
        "bundle-manifest.json",
    ]
    members["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(members[path]).hexdigest()}  {path}\n"
        for path in checksum_paths
    ).encode("ascii")
    return _canonical_outer(_ordered_members(members, manifest))


def _json_bytes(value):
    return (
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
