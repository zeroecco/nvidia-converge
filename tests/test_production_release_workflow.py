from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/production-release.yml")


def _workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _workflow() -> dict[str, object]:
    payload = yaml.load(_workflow_text(), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def test_production_release_has_a_new_default_branch_only_trigger() -> None:
    assert WORKFLOW_PATH.is_file()
    assert not Path(".github/workflows/release.yml").exists()
    workflow = _workflow()
    trigger = workflow["on"]
    assert trigger == {
        "repository_dispatch": {"types": ["production-release"]}
    }
    text = _workflow_text()
    assert "workflow_dispatch:" not in text
    assert "push:" not in text
    assert "client_payload" not in text
    assert '"$GITHUB_REF" != refs/heads/main' in text
    assert '"$live_main_sha" != "$GITHUB_SHA"' in text
    assert "commits/main" in text
    assert "group: production-release" in text


def test_release_identity_is_derived_from_trusted_source() -> None:
    text = _workflow_text()
    assert "Bind the release identity to trusted source" in text
    assert 'release_tag="v${release_version}"' in text
    assert 'evidence_path="integrations/results.${release_tag}.json"' in text
    assert 'git ls-files --error-unmatch -- "$evidence_path"' in text
    assert "github.event.client_payload" not in text
    assert text.count("release_tag: ${{") == 3
    assert 'RELEASE_TAG: ${{ needs.artifacts.outputs.release_tag }}' in text


def test_release_app_is_the_only_non_owner_release_writer() -> None:
    text = _workflow_text()
    publish = text[text.index("  publish:") :]
    assert "environment: release" in publish
    assert "contents: read" in publish
    assert "\n      contents: write\n" not in publish
    assert (
        "actions/create-github-app-token@"
        "bcd2ba49218906704ab6c1aa796996da409d3eb1"
    ) in publish
    assert "permission-contents: write" in publish
    assert "secrets.RELEASE_CREATOR_PRIVATE_KEY" in publish
    assert "vars.RELEASE_CREATOR_APP_ID" in publish
    assert publish.index("Recheck current main") < publish.index(
        "Mint a short-lived Release Creator token"
    )
    assert publish.index("Mint a short-lived Release Creator token") < publish.index(
        "Prove Release Creator token has exact repository scope"
    ) < publish.index("Create and bind the exact release tag")


def test_release_app_token_is_explicitly_repo_scoped_and_proven() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    publish = jobs["publish"]
    assert isinstance(publish, dict)
    steps = publish["steps"]
    assert isinstance(steps, list)

    token_step = next(
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("name") == "Mint a short-lived Release Creator token"
    )
    assert token_step["with"] == {
        "app-id": "${{ vars.RELEASE_CREATOR_APP_ID }}",
        "private-key": "${{ secrets.RELEASE_CREATOR_PRIVATE_KEY }}",
        "owner": "${{ github.repository_owner }}",
        "repositories": "${{ github.event.repository.name }}",
        "permission-contents": "write",
    }

    proof_step = next(
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("name")
        == "Prove Release Creator token has exact repository scope"
    )
    proof_env = proof_step["env"]
    assert isinstance(proof_env, dict)
    assert proof_env["EXPECTED_INSTALLATION_ID"] == (
        "${{ steps.release-token.outputs.installation-id }}"
    )
    assert proof_env["EXPECTED_APP_SLUG"] == (
        "${{ steps.release-token.outputs.app-slug }}"
    )
    assert proof_env["GH_TOKEN"] == "${{ steps.release-token.outputs.token }}"
    proof = proof_step["run"]
    assert isinstance(proof, str)
    assert (
        "https://api.github.com/installation/repositories?per_page=100&page=1"
        in proof
    )
    assert '.total_count == 1' in proof
    assert '(.repositories | type == "array" and length == 1)' in proof
    assert ".repositories[0].id == $repository_id" in proof
    assert ".repositories[0].full_name == $repository" in proof
    assert ".repositories[0].owner.login == $repository_owner" in proof
    assert '"$status" != 200' in proof
    assert "-gt 2097152" in proof


def test_tag_and_release_are_created_once_and_bound_by_id() -> None:
    text = _workflow_text()
    publish = text[text.index("  publish:") :]
    assert '"https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs"' in publish
    assert '[[ "$tag_status" != 201' in publish
    assert "created release tag is not bound to the gated commit" in publish
    assert '"https://api.github.com/repos/${GITHUB_REPOSITORY}/releases"' in publish
    assert '[[ "$status" != 201' in publish
    assert 'echo "release_id=$release_id" >> "$GITHUB_OUTPUT"' in publish
    assert (
        '"https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/'
        '${release_id}/assets?name=${name}"'
    ) in publish
    assert (
        '"https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/'
        '${RELEASE_ID}"'
    ) in publish
    assert "softprops/action-gh-release" not in publish
    assert "gh release edit" not in publish
    assert "--clobber" not in publish


def test_publication_rechecks_controls_tag_and_bytes() -> None:
    text = _workflow_text()
    publish = text[text.index("  publish:") :]
    assert publish.count("--require-immutable-releases") == 3
    assert publish.count("--release-creator-app-id") == 3
    assert publish.count("--require-release-writer-isolation") == 3
    assert publish.count("--release-workflow-run-id") == 3
    assert publish.count("--release-workflow-run-attempt") == 3
    assert "Recheck controls immediately before publication" in publish
    assert "Reverify the bound draft immediately before publication" in publish
    assert "Publish the bound release ID" in publish
    assert "Verify published release is immutable" in publish
    assert publish.count('sha256sum --check --strict -- SHA256SUMS') == 4
    assert publish.count('cmp --silent -- "dist/$name" "$remote_dir/$name"') == 3
    assert '.immutable == false' in publish
    assert '.immutable == true' in publish
    assert publish.count("and .prerelease == false") >= 5
    assert publish.count("and .name == $tag") >= 5
    assert "EXPECTED_RELEASE_BODY_SHA256" in publish
    assert "--rawfile body \"$release_body_path\"" in publish
    assert "immutable release body differs from the bound draft" in publish
    assert '[[ "$status" != 200' in publish
