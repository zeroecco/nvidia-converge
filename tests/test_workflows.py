import subprocess
import sys
from pathlib import Path

from nvidia_converge.desired import load_desired


def test_production_gpu_qualification_has_new_current_main_dispatch_history():
    workflow_path = Path(
        ".github/workflows/production-gpu-qualification.yml"
    )
    old_workflow_path = Path(".github/workflows/gpu-integration.yml")

    assert workflow_path.is_file()
    assert not old_workflow_path.exists()
    workflow = workflow_path.read_text(encoding="utf-8")
    assert "repository_dispatch:" in workflow
    assert "types: [production-gpu-qualification]" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "client_payload" not in workflow
    assert 'DISPATCH_ACTION: ${{ github.event.action }}' in workflow
    assert '"$GITHUB_EVENT_NAME" != repository_dispatch' in workflow
    assert '"$DISPATCH_ACTION" != production-gpu-qualification' in workflow
    assert '"$GITHUB_REF" != refs/heads/main' in workflow
    assert '.default_branch | select(. == "main")' in workflow
    assert '"$live_main_sha" != "$GITHUB_SHA"' in workflow

    controls = workflow[
        workflow.index("  repository-controls:") :
        workflow.index("  qualification-build:")
    ]
    assert controls.index("Bind the dispatch to the live default-branch head") < (
        controls.index("Check out repository-control checker")
    )
    assert workflow.count("uses: actions/checkout@") == 4
    assert workflow.count("ref: ${{ github.sha }}") == 4
    assert "ref: ${{ github.ref }}" not in workflow
    assert "ref: main" not in workflow


def test_gpu_integration_uses_virtualenv_for_python_tooling():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "python3 -m pip install --user" not in workflow
    assert "for candidate in python3.12 python3.11 python3.10 python3" in workflow
    assert "sys.version_info >= (3, 10)" in workflow
    assert "import ensurepip" in workflow
    assert "validate_trusted_path()" in workflow
    assert "PATH: /usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin" in workflow
    assert "export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin" in workflow
    assert "sys.base_prefix" in workflow
    assert "sys.exec_prefix" in workflow
    assert 'sysconfig.get_path("stdlib")' in workflow
    assert 'sysconfig.get_path("platstdlib")' in workflow
    assert 'for path in "${python_trust_paths[@]:4}"' in workflow
    assert 'getattr(module, "__file__", None)' in workflow
    assert 'for package in (ensurepip, venv)' in workflow
    assert 'root.rglob("*")' in workflow
    assert "Python trust path is empty, relative, or malformed" in workflow
    assert "Python trust path has a symlinked or noncanonical resolution" in workflow
    assert "8#$mode & 8#022" in workflow
    assert 'sudo "$PYTHON_BIN" -I -S -m venv' in workflow
    stage = workflow[workflow.index("- name: Stage trusted qualification runtime") :]
    assert stage.index('python_now="$(readlink -f -- "$PYTHON_BIN")"') < stage.index(
        'if sudo test -e "$QUALIFICATION_ROOT"'
    )
    assert '"$PYTHON_BIN" -m venv .venv' in workflow
    assert "no Python >=3.10 interpreter with venv/ensurepip support found" in workflow
    assert "python3 -m venv .venv" not in workflow
    assert '"$QUALIFICATION_PYTHON" -I -m nvidia_converge plan' in workflow
    assert "Stage trusted qualification runtime" in workflow
    assert ".venv/bin/python -m build --wheel" not in workflow
    assert '"$QUALIFICATION_PYTHON" -I -m nvidia_converge install' in workflow
    assert '--desired "$QUALIFICATION_DESIRED"' in workflow
    assert "--no-index --no-deps" in workflow
    assert "qualification-inputs.sha256" in workflow
    assert 'git rev-parse --verify HEAD)" != "$GITHUB_SHA"' in workflow
    assert 'git diff --quiet HEAD -- "$DESIRED_FILE"' in workflow
    assert 'git ls-files --error-unmatch -- "$DESIRED_FILE"' in workflow
    assert 'git archive --format=tar "$GITHUB_SHA"' in workflow
    assert 'archived_desired="$desired_source/$DESIRED_FILE"' in workflow
    assert 'artifacts/pre/qualification-desired.yaml "$QUALIFICATION_DESIRED"' in workflow
    assert '"$DESIRED_PATH" "$QUALIFICATION_DESIRED"' not in workflow
    assert "validate_trusted_directory /opt" in workflow
    assert 'PYTHON_BIN="$PWD/.venv/bin/python"' not in workflow
    assert 'PYTHONPATH="$PWD"' not in workflow


def test_readme_documents_python_floor_and_private_applied_lock_reports():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "All implemented recipe targets require Python 3.10 or newer" in readme
    assert (
        "/usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 "
        "/usr/bin/python3"
    ) in readme
    assert "--out lock.json" not in readme
    assert '--out "$REPORT_DIR/lock-$RUN_ID.json"' in readme
    assert "export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin" in readme
    assert 'sudo "$PYTHON_BIN" -I -S -m venv' in readme
    assert "Interrupted-operation recovery" in readme
    assert "operation-recovered" in readme
    assert "Do not manually start, enable, unmask, delete, rename, or edit" in readme
    assert "There is no force/bypass flag" in readme


def test_gpu_integration_rejects_writable_privileged_python_modules():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(
        encoding="utf-8"
    )
    selection = workflow[
        workflow.index("- name: Select supported Python interpreter") :
        workflow.index("- name: Install downloaded qualification wheel")
    ]

    assert 'getattr(module, "__file__", None)' in selection
    assert 'for package in (ensurepip, venv)' in selection
    assert '*module_paths,' in selection
    assert '*package_paths,' in selection
    assert 'validate_trusted_path component "$path"' in selection
    assert "(( (8#$mode & 8#022) != 0 ))" in selection


def test_gpu_integration_validates_every_generated_report():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    gpu_job = workflow[workflow.index("  gpu:") : workflow.index("  validate-attestations:")]
    assert "jsonschema" not in gpu_job
    assert "Validate retained report schemas" in workflow
    assert 'Path("attestation-artifacts")' in workflow
    assert "jsonschema.validate(" in workflow
    assert "strict_format_checker()" in workflow
    assert "Require every GPU matrix job to pass" in workflow


def test_gpu_integration_uses_hosted_hashed_clean_source_build():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    build_job = workflow[
        workflow.index("  qualification-build:") : workflow.index("  gpu:")
    ]
    gpu_job = workflow[workflow.index("  gpu:") : workflow.index("  validate-attestations:")]

    assert "runs-on: ubuntu-latest" in build_job
    assert 'if [ "$GITHUB_REF" != refs/heads/main' in build_job
    assert 'python-version: "3.12.13"' in build_job
    assert "--require-hashes --no-deps" in build_job
    assert "requirements/build.lock" in build_job
    assert 'git archive --format=tar "$GITHUB_SHA"' in build_job
    assert "python -m build --wheel --no-isolation" in build_job
    assert "environment: gpu-qualification" in gpu_job
    assert "actions/download-artifact@" in gpu_job
    assert "QUALIFICATION_WHEEL_SHA256" in gpu_job
    assert '--qualification-wheel-name "$QUALIFICATION_WHEEL_NAME"' in gpu_job
    assert '--qualification-wheel-sha256 "$QUALIFICATION_WHEEL_SHA256"' in gpu_job
    assert "qualification_wheel_name: $QUALIFICATION_WHEEL_NAME" in gpu_job
    assert "qualification_wheel_sha256: $QUALIFICATION_WHEEL_SHA256" in gpu_job
    assert "qualification artifact inventory is not exact" in gpu_job
    assert "checksum manifest does not match hosted outputs" in gpu_job
    assert "--no-index --no-deps" in gpu_job
    assert "--requirement" not in gpu_job
    assert "--upgrade pip" not in gpu_job
    assert "python -m build" not in gpu_job


def test_sensitive_workflows_fail_closed_on_live_repository_controls():
    gpu_workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(
        encoding="utf-8"
    )
    gpu_controls = gpu_workflow[
        gpu_workflow.index("  repository-controls:") :
        gpu_workflow.index("  qualification-build:")
    ]
    gpu_job = gpu_workflow[
        gpu_workflow.index("  gpu:") :
        gpu_workflow.index("  validate-attestations:")
    ]

    assert "runs-on: ubuntu-latest" in gpu_controls
    assert "actions: read" in gpu_controls
    assert "actions: write" not in gpu_controls
    assert "contents: read" in gpu_controls
    assert "GITHUB_TOKEN: ${{ github.token }}" in gpu_controls
    assert "environment:" not in gpu_controls
    assert "self-hosted" not in gpu_controls
    assert "sudo " not in gpu_controls
    assert "continue-on-error" not in gpu_controls
    assert "always()" not in gpu_controls
    assert "scripts/check_repository_controls.py" in gpu_controls
    assert "--scope gpu" in gpu_controls
    assert "needs: repository-controls" in gpu_workflow[
        gpu_workflow.index("  qualification-build:") :
        gpu_workflow.index("  gpu:")
    ]
    assert "needs: [repository-controls, qualification-build]" in gpu_job
    assert "group: nvidia-converge-gpu" in gpu_job
    assert "labels: ${{ matrix.labels }}" in gpu_job
    assert "environment: gpu-qualification" in gpu_job

    release_workflow = Path(".github/workflows/production-release.yml").read_text(
        encoding="utf-8"
    )
    release_controls = release_workflow[
        release_workflow.index("  repository-controls:") :
        release_workflow.index("  gates:")
    ]
    gates_job = release_workflow[
        release_workflow.index("  gates:") :
        release_workflow.index("  artifacts:")
    ]
    assert "runs-on: ubuntu-latest" in release_controls
    assert "actions: read" in release_controls
    assert "actions: write" not in release_controls
    assert "contents: read" in release_controls
    assert "GITHUB_TOKEN: ${{ github.token }}" in release_controls
    assert "GH_TOKEN: ${{ github.token }}" in release_controls
    assert "environment:" not in release_controls
    assert "contents: write" not in release_controls
    assert "RELEASE_CREATOR_PRIVATE_KEY" not in release_controls
    assert "scripts/check_repository_controls.py" in release_controls
    assert "--scope release" in release_controls
    assert release_controls.index(
        "Bind the dispatch to the live default-branch head"
    ) < release_controls.index("Check out repository-control checker")
    assert "needs: repository-controls" in gates_job
    assert "secrets.REPOSITORY_AUDIT_TOKEN" in release_workflow
    assert "--require-immutable-releases" in release_workflow


def test_gpu_integration_sanitizes_dispatch_path_and_always_rolls_back():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "DESIRED_FILE: ${{ matrix.desired }}" in workflow
    assert '--desired "${{ inputs.desired }}"' not in workflow
    assert '"$GITHUB_WORKSPACE"/examples/*.yaml' in workflow
    assert "QUALIFICATION_APPLY: \"true\"" in workflow
    assert "env.QUALIFICATION_APPLY == 'true'" in workflow
    assert "inputs.apply" not in workflow
    assert "Plan dry-run dispatch" not in workflow
    assert "steps.retain_snapshot.outcome == 'success'" in workflow
    assert "Snapshot after package-policy staging" in workflow
    assert "rollback-snapshot.sha256" in workflow
    assert "policy-rollback-snapshot.sha256" in workflow
    assert "Restore pre-policy baseline" in workflow


def test_gpu_integration_uses_snapshot_as_rollback_oracle():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "Verify after rollback" not in workflow
    assert 'rollback --desired "$QUALIFICATION_DESIRED"' in workflow


def test_gpu_integration_profiles_are_capability_specific():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "desired: examples/compute-580-open-fabric-manager.yaml" in workflow
    assert load_desired("examples/compute-580-open.yaml").fabric_manager is False
    assert (
        load_desired("examples/compute-580-open-fabric-manager.yaml").fabric_manager
        is True
    )
    assert load_desired("examples/compute-580-open-mig.yaml").fabric_manager is False
    assert "runner does not expose an observable Secure Boot state" in workflow
    assert "mig-toggle runner does not expose MIG capability" in workflow


def test_gpu_integration_faults_use_live_probes_and_independent_cleanup():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    for finding in (
        "kernel.headers.missing",
        "module.not-loaded",
        "driver.version-mismatch",
        "container-toolkit.missing",
        "fabric-manager.inactive",
    ):
        assert finding in workflow
    assert "cli.audit_host =" not in workflow
    assert "operation.lock" in workflow
    assert "flock --exclusive --nonblock" in workflow
    assert "unshare --mount --propagation private" in workflow
    assert "Independently restore controlled-fault pre-state" in workflow
    assert "fault-cleanup-required" in workflow
    assert 'test "${{ steps.fault_cleanup.outcome }}" = success' in workflow


def test_gpu_integration_requires_real_package_drift_and_uploads_only_attestation():
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "Validate staged policy and controlled pre-drift" in workflow
    assert 'expected_package = "nvidia-container-toolkit"' in workflow
    assert '"container-toolkit.missing"' in workflow
    assert '"docker.nvidia-runtime-missing"' in workflow
    assert 'desired.cuda_compat != "none"' in workflow
    assert 'actions.get("install.packages")' in workflow
    assert "scripts/export_integration_attestation.py" in workflow
    assert "path: artifacts/export/" in workflow
    assert "path: artifacts/\n" not in workflow
    assert "id: export-attestation" in workflow
    assert "id: verify-attestation" in workflow
    assert "--verify-only" in workflow
    assert 'if [[ ! -d artifacts || -L artifacts ]]; then' in workflow
    assert "chmod 0700 artifacts" in workflow
    assert (
        "if: ${{ always() && steps.export-attestation.outcome == 'success' }}"
        in workflow
    )
    assert (
        "if: ${{ always() && steps.verify-attestation.outcome == 'success' }}"
        in workflow
    )
    export_step = workflow.index(
        "- name: Export privacy-minimized integration attestation"
    )
    verify_step = workflow.index(
        "- name: Verify exact sanitized upload handoff"
    )
    upload_step = workflow.index("- name: Upload integration artifacts")
    assert export_step < verify_step < upload_step
    assert "--verify-only" not in workflow[export_step:verify_step]
    assert "--verify-only" in workflow[verify_step:upload_step]


def test_release_requires_tag_specific_integration_evidence():
    workflow = Path(".github/workflows/production-release.yml").read_text(encoding="utf-8")
    assert not Path(".github/workflows/release.yml").exists()
    assert "repository_dispatch:" in workflow
    assert "types: [production-release]" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "client_payload" not in workflow
    assert 'release_tag="v${release_version}"' in workflow
    assert "release_tag: ${{ needs.repository-controls.outputs.release_tag }}" in workflow
    assert "Build and bind release artifacts" in workflow
    assert "python -m scripts.bundle_release_evidence" in workflow
    assert 'integrations/results.${RELEASE_TAG}.json' in workflow
    assert '--commit "${GITHUB_SHA}"' in workflow
    assert '--qualification-wheel "$build_root/dist-a/$wheel"' in workflow
    assert '--output "${RUNNER_TEMP}/gpu-integration-evidence.zip"' in workflow
    assert '"dist/gpu-integration-evidence-${RELEASE_TAG}.zip"' in workflow
    assert "GITHUB_REF_NAME" not in workflow
    assert 'python -m twine check "dist/$wheel" "dist/$sdist"' in workflow
    assert 'subject-path: "dist/*"' in workflow
    assert "files: dist/*" not in workflow
    assert "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/" in workflow
    assert "fetch-depth: 0" in workflow
    assert "scripts/verify_release_evidence_bundle.py" in workflow


def test_release_build_is_locked_mainline_clean_source_and_reproducible():
    workflow = Path(".github/workflows/production-release.yml").read_text(encoding="utf-8")
    gates_job = workflow[workflow.index("  gates:") : workflow.index("  artifacts:")]
    artifacts_job = workflow[
        workflow.index("  artifacts:") : workflow.index("  publish:")
    ]
    publish_job = workflow[workflow.index("  publish:") :]

    assert 'python-version: "3.12.13"' in gates_job
    assert 'python-version: "3.12.13"' in artifacts_job
    assert workflow.count("ref: ${{ github.sha }}") == 3
    assert "--require-hashes --no-deps" in gates_job
    assert "--require-hashes --no-deps" in artifacts_job
    assert "requirements/release.lock" in gates_job
    assert "requirements/release.lock" in artifacts_job
    assert 'git rev-parse --verify refs/remotes/origin/main)' in gates_job
    assert '!= "$GITHUB_SHA"' in gates_job
    assert "git merge-base --is-ancestor" in artifacts_job
    assert 'refs/remotes/origin/main' in artifacts_job
    assert "Recheck immutable checkout after gates" in gates_job
    assert "python -m pytest -q" in gates_job
    assert "python -m pytest -q" not in artifacts_job
    assert "python -m build" not in gates_job
    assert "python -m scripts.bundle_release_evidence" not in gates_job
    assert "actions/upload-artifact@" not in gates_job
    assert "needs: gates" in artifacts_job
    assert "Check out repository on fresh artifact runner" in artifacts_job
    assert "Prove clean immutable artifact checkout" in artifacts_job
    assert "Recheck immutable checkout before artifact handoff" in artifacts_job
    assert "git diff --quiet HEAD --" in artifacts_job
    assert "git diff --cached --quiet HEAD --" in artifacts_job
    assert "git ls-files --others --exclude-standard" in artifacts_job
    assert artifacts_job.count('git archive --format=tar "$GITHUB_SHA"') == 3
    assert artifacts_job.count('git archive --format=tar "$tested_commit"') == 2
    assert artifacts_job.count("python -m build --wheel --no-isolation") == 2
    assert artifacts_job.count("python -m build --sdist --no-isolation") == 2
    assert 'SOURCE_DATE_EPOCH="$(git show -s --format=%ct "$tested_commit")"' in (
        artifacts_job
    )
    assert 'cd "$build_root/evidence-source"' in artifacts_job
    assert "evidence binding modified the verified release wheel" in artifacts_job
    assert "scripts/canonicalize_sdist.py" in artifacts_job
    assert "independent clean-source builds differ" in artifacts_job
    assert '["git", "ls-files", "-z"]' in artifacts_job
    assert "source distribution omits required source files" in artifacts_job
    assert "archive.extractall(extract_root, filter=\"data\")" in artifacts_job
    assert "python -m scripts.verify_release_evidence_bundle --help" in artifacts_job
    assert "needs: artifacts" in publish_job
    assert "needs: gates" not in publish_job
    assert workflow.count("runs-on: ubuntu-latest") == 4
    assert "python -m build\n" not in workflow
    assert "python -m pip install --upgrade" not in workflow
    assert "python -m pip install build twine" not in workflow
    assert "--no-build-isolation" in workflow


def test_release_handoff_is_exact_checksums_first_and_immutable():
    workflow = Path(".github/workflows/production-release.yml").read_text(encoding="utf-8")

    assert 'wheel="nvidia_converge-${release_version}-py3-none-any.whl"' in workflow
    assert 'sdist="nvidia_converge-${release_version}.tar.gz"' in workflow
    assert 'evidence="gpu-integration-evidence-${RELEASE_TAG}.zip"' in workflow
    assert 'release_assets=("$evidence" "$wheel" "$sdist")' in workflow
    assert "LC_ALL=C sort --check" in workflow
    assert 'sha256sum -- "${release_assets[@]}" > SHA256SUMS' in workflow
    assert "sha256sum *" not in workflow
    assert "find dist -mindepth 1 -maxdepth 1 -printf . | wc -c" in workflow
    assert '[[ ! -f "dist/$name" || -L "dist/$name" ]]' in workflow
    assert 'validate_dist_inventory "${release_assets[@]}" SHA256SUMS' in workflow
    assert "SHA256SUMS is not the exact checksum manifest" in workflow
    assert workflow.count("sha256sum --check --strict -- SHA256SUMS") == 5
    assert "retention-days: 30" in workflow
    assert "retention-days: 7" not in workflow
    assert "group: production-release" in workflow
    assert "cancel-in-progress: false" in workflow

    download = workflow.index("- name: Download built release artifacts")
    verify = workflow.index("- name: Verify exact release handoff")
    controls = workflow.index(
        "- name: Recheck current main, release controls, and immutable configuration"
    )
    mint = workflow.index("- name: Mint a short-lived Release Creator token")
    create_tag = workflow.index("- name: Create and bind the exact release tag")
    attest = workflow.index("- name: Attest release artifacts")
    verify_tag = workflow.index(
        "- name: Recheck controls and tag immediately before draft creation"
    )
    draft = workflow.index("- name: Create draft GitHub release")
    verify_draft = workflow.index("- name: Verify draft release inventory")
    controls_before_publish = workflow.index(
        "- name: Recheck controls immediately before publication"
    )
    reverify_draft = workflow.index(
        "- name: Reverify the bound draft immediately before publication"
    )
    publish = workflow.index("- name: Publish the bound release ID")
    verify_immutable = workflow.index("- name: Verify published release is immutable")
    assert (
        download
        < verify
        < controls
        < mint
        < create_tag
        < attest
        < verify_tag
        < draft
        < verify_draft
        < controls_before_publish
        < reverify_draft
        < publish
        < verify_immutable
    )
    assert "tag_ref_object_sha" in workflow
    assert workflow.count(
        '"repos/${GITHUB_REPOSITORY}/git/ref/tags/${RELEASE_TAG}"'
    ) == 3
    assert workflow.count(
        '"repos/${GITHUB_REPOSITORY}/commits/tags/${RELEASE_TAG}"'
    ) == 4
    assert workflow.count('!= "$GITHUB_SHA"') >= 5
    assert "release source or tag changed before draft creation" in workflow
    assert "release source or tag changed before immutable publication" in workflow
    assert "immutable release tag is not the exact gated tag object" in workflow
    assert '[[ "$tag_status" != 201' in workflow
    assert '[[ "$status" != 201' in workflow
    assert "draft: true" in workflow
    assert "softprops/action-gh-release" not in workflow
    assert "--clobber" not in workflow
    assert '"repos/${GITHUB_REPOSITORY}/releases/${RELEASE_ID}"' in workflow
    assert "and (.assets | length == 4)" in workflow
    assert 'and (.digest | test("^sha256:[0-9a-f]{64}$"))' in workflow
    assert "draft release is not the exact complete release inventory" in workflow
    assert ".digest == $digest and .size == $size" in workflow
    assert '"repos/${GITHUB_REPOSITORY}/releases/assets/${asset_id}"' in workflow
    assert 'cmp --silent -- "dist/$name" "$remote_dir/$name"' in workflow
    assert "--rawfile body \"$release_body_path\"" in workflow
    assert "body: $body" in workflow
    assert "draft: false" in workflow
    assert workflow.count("and .prerelease == false") >= 5
    assert workflow.count("and .name == $tag") >= 5
    assert "release_body_sha256=$release_body_sha256" in workflow
    assert "EXPECTED_RELEASE_BODY_SHA256" in workflow
    assert "immutable release body differs from the bound draft" in workflow
    assert ".immutable == true" in workflow
    assert "contents: read" in workflow[workflow.index("  publish:") :]
    assert "permission-contents: write" in workflow
    assert "secrets.RELEASE_CREATOR_PRIVATE_KEY" in workflow


def test_release_docs_require_immutable_release_and_ordered_consumer_verification():
    readme = Path("README.md").read_text(encoding="utf-8")
    integration = Path("docs/integration-testing.md").read_text(encoding="utf-8")

    for document in (readme, integration):
        assert "release immutability" in document.lower() or (
            "immutable release" in document.lower()
        )
        assert "scripts.verify_release_evidence_bundle" in document
        assert "--qualification-wheel" in document
        assert "gh release verify" in document
        assert "--signer-workflow" in document
        assert "--source-ref" in document
        assert "--source-digest" in document
        assert "30 days" in document
        assert "fresh" in document
    assert "sha256sum --check --strict -- SHA256SUMS" in readme
    assert "Canonicalization strips ZIP metadata" in readme
    assert "original GitHub archive bytes" in integration


def test_docs_separate_recipe_recognition_from_exact_release_qualification():
    readme = Path("README.md").read_text(encoding="utf-8")
    integration = Path("docs/integration-testing.md").read_text(encoding="utf-8")

    assert "Recipe recognition" in readme
    assert "exact OS ID/version" in readme
    assert "recognized releases outside those observed tuples" in readme
    assert "qualified_platforms" in readme
    assert "recipe_path_coverage" in readme
    assert "does not qualify" in integration
    assert "install-failure compensation" in integration
    assert "second-pass idempotence" in integration


def test_docs_require_repository_and_runner_protection_controls():
    integration = Path("docs/integration-testing.md").read_text(encoding="utf-8")

    assert "Protect `main`" in integration
    assert "has an empty bypass inventory" in integration
    assert "Do not make the Release Creator App a main-branch bypass actor" in integration
    assert "every effective `main` ruleset has no bypass actors" in integration
    assert "Protect `v*` tags" in integration
    assert "gpu-qualification" in integration
    assert "Do not permit self-approval" in integration
    assert "nvidia-converge-gpu` runner group" in integration
    assert "Transfer the repository to an organization" in integration
    assert "REPOSITORY_AUDIT_TOKEN" in integration
    assert "checker success never claims" in integration
    assert "head_branch: main" in integration


def test_ci_exercises_each_supported_cpython_minor():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]' in workflow
    assert "requirements/release.lock" in workflow
    assert "--require-hashes --no-deps" in workflow
    assert "if: matrix.python-version == '3.12'" in workflow
    assert 'pip install --upgrade pip ".[test]"' not in workflow
    assert "python -m build --no-isolation" in workflow
    assert "--isolated --no-index --no-deps --force-reinstall dist/*.whl" in workflow
    assert "--no-build-isolation" in workflow


def test_stdlib_suite_runs_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-I", "-S", "tests/run_tests.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "all tests passed\n"
    assert result.stderr == ""


def test_ci_and_release_gate_the_evidence_tooling():
    for path in (
        ".github/workflows/ci.yml",
        ".github/workflows/production-release.yml",
    ):
        workflow = Path(path).read_text(encoding="utf-8")
        assert "python -m compileall -q nvidia_converge scripts tests" in workflow
        assert "scripts/canonicalize_sdist.py" in workflow
        assert "scripts/check_release_evidence.py" in workflow
        assert "scripts/export_integration_attestation.py" in workflow
        assert "scripts/bundle_release_evidence.py" in workflow
        assert "scripts/verify_release_evidence_bundle.py" in workflow


def test_ci_and_release_gate_workflow_semantics_with_pinned_actionlint():
    expected_hash = (
        "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
    )
    expected_url = (
        "https://github.com/rhysd/actionlint/releases/download/"
        "v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz"
    )

    for path in (
        ".github/workflows/ci.yml",
        ".github/workflows/production-release.yml",
    ):
        workflow = Path(path).read_text(encoding="utf-8")
        assert 'ACTIONLINT_VERSION: "1.7.12"' in workflow
        assert f"ACTIONLINT_ARCHIVE_SHA256: {expected_hash}" in workflow
        assert expected_url in workflow
        assert "sha256sum --check --strict -" in workflow
        assert 'test ! -L "$tool_root/actionlint"' in workflow
        assert 'test "${version_output%%$\'\\n\'*}" = "$ACTIONLINT_VERSION"' in workflow
        assert (
            '"$tool_root/actionlint" -color=false -shellcheck= -pyflakes='
            in workflow
        )


def test_source_distribution_retains_offline_verification_inputs():
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    assert "include nvidia-converge" in manifest
    assert "recursive-include scripts *.py" in manifest
    assert "recursive-include examples *.yaml" in manifest
    assert "recursive-include docs *.md" in manifest
    assert "recursive-include integrations *.json" in manifest
    assert "recursive-include requirements *.lock" in manifest
    assert "recursive-include schemas *.json" in manifest
    assert "recursive-include tests *.py" in manifest
    assert "recursive-include .github/workflows *.yml" in manifest


def test_package_metadata_matches_distributed_license():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    license_text = Path("LICENSE").read_text(encoding="utf-8")

    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'test = ["build>=1.2",' in pyproject
    assert "Apache License" in license_text[:200]
    assert "Version 2.0" in license_text[:200]


def test_release_lockfiles_are_exact_and_hashed():
    for path in ("requirements/build.lock", "requirements/release.lock"):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        assert "--only-binary=:all:" in lines
        requirements = [
            line for line in lines if line and not line.startswith(("#", "--"))
        ]
        assert requirements
        for requirement in requirements:
            assert "==" in requirement
            assert " --hash=sha256:" in requirement
            assert ">=" not in requirement
            assert " @ " not in requirement

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools==83.0.0"]' in pyproject


def test_workflows_pin_node24_actions_and_drop_checkout_credentials():
    workflows = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            ".github/workflows/ci.yml",
            ".github/workflows/production-gpu-qualification.yml",
            ".github/workflows/production-release.yml",
        )
    }
    all_workflows = "\n".join(workflows.values())

    expected_pins = (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "actions/attest-build-provenance@0f67c3f4856b2e3261c31976d6725780e5e4c373",
        "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1",
    )
    for pin in expected_pins:
        assert pin in all_workflows
    for workflow in workflows.values():
        if "actions/checkout@" in workflow:
            assert "persist-credentials: false" in workflow

    integration_docs = Path("docs/integration-testing.md").read_text(
        encoding="utf-8"
    )
    assert "GitHub Actions Runner 2.327.1 or newer" in integration_docs
