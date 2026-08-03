from __future__ import annotations

import copy
from email.message import Message
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request

import pytest
from typing_extensions import Self

import nvidia_converge.repository_controls as controls
import scripts.check_repository_controls as checker_cli

API_URL = "https://api.github.com"
REPOSITORY = "example-org/nvidia-converge"
TOKEN = "test-token-that-must-not-be-printed"
RELEASE_CREATOR_APP_ID = 987654
RELEASE_WORKFLOW_RUN_ID = 7654321
RELEASE_WORKFLOW_RUN_ATTEMPT = 1
MAIN_SHA = "a" * 40
EXPECTED_CONTEXTS = tuple(f"Python 3.{minor}" for minor in range(10, 15))


def _effective_rule(
    rule_type: str,
    *,
    ruleset_id: int,
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    rule: dict[str, object] = {
        "type": rule_type,
        "ruleset_source_type": "Repository",
        "ruleset_source": REPOSITORY,
        "ruleset_id": ruleset_id,
    }
    if parameters is not None:
        rule["parameters"] = parameters
    return rule


def _passing_branch_rules() -> list[dict[str, object]]:
    return [
        _effective_rule(
            "pull_request",
            ruleset_id=11,
            parameters={
                "allowed_merge_methods": ["merge", "squash", "rebase"],
                "dismiss_stale_reviews_on_push": True,
                "require_code_owner_review": False,
                "require_last_push_approval": True,
                "required_approving_review_count": 1,
                "required_review_thread_resolution": True,
            },
        ),
        _effective_rule(
            "required_status_checks",
            ruleset_id=11,
            parameters={
                "do_not_enforce_on_create": False,
                "strict_required_status_checks_policy": True,
                "required_status_checks": [
                    {"context": context, "integration_id": 15368}
                    for context in EXPECTED_CONTEXTS
                ],
            },
        ),
        _effective_rule("deletion", ruleset_id=11),
        _effective_rule("non_fast_forward", ruleset_id=11),
    ]


def _branch_ruleset(
    identifier: int = 11,
    *,
    enforcement: str = "active",
    rule_types: tuple[str, ...] = (
        "pull_request",
        "required_status_checks",
        "deletion",
        "non_fast_forward",
    ),
) -> dict[str, object]:
    return {
        "id": identifier,
        "name": f"protected-main-{identifier}",
        "source_type": "Repository",
        "source": REPOSITORY,
        "enforcement": enforcement,
        "target": "branch",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "include": ["~DEFAULT_BRANCH"],
                "exclude": [],
            }
        },
        "rules": [{"type": rule_type} for rule_type in rule_types],
    }


def _environment(name: str, *, identifier: int) -> dict[str, object]:
    return {
        "id": identifier,
        "node_id": f"ENV_{identifier}",
        "name": name,
        "url": f"{API_URL}/repos/{REPOSITORY}/environments/{name}",
        "html_url": f"https://github.com/{REPOSITORY}/settings/environments/{name}",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "protection_rules": [
            {
                "id": identifier * 10,
                "node_id": f"RULE_{identifier}",
                "type": "required_reviewers",
                "prevent_self_review": True,
                "reviewers": [
                    {
                        "type": "Team",
                        "reviewer": {
                            "id": 700,
                            "name": "release-reviewers",
                            "slug": "release-reviewers",
                        },
                    }
                ],
            }
        ],
        "deployment_branch_policy": {
            "protected_branches": False,
            "custom_branch_policies": True,
        },
    }


def _deployment_branch_policy(
    identifier: int,
    *,
    name: str = "main",
    policy_type: str = "branch",
) -> dict[str, object]:
    return {
        "id": identifier,
        "node_id": f"DEPLOYMENT_POLICY_{identifier}",
        "name": name,
        "type": policy_type,
    }


def _deployment_branch_policy_page(
    policies: list[dict[str, object]],
    *,
    total_count: int | None = None,
) -> dict[str, object]:
    return {
        "total_count": len(policies) if total_count is None else total_count,
        "branch_policies": policies,
    }


def _workflow(
    identifier: int,
    path: str,
    *,
    state: str = "active",
) -> dict[str, object]:
    return {
        "id": identifier,
        "node_id": f"WORKFLOW_{identifier}",
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "state": state,
    }


def _workflow_page(
    workflows: list[dict[str, object]],
    *,
    total_count: int | None = None,
) -> dict[str, object]:
    return {
        "total_count": len(workflows) if total_count is None else total_count,
        "workflows": workflows,
    }


def _ruleset_summary(
    identifier: int,
    *,
    enforcement: str = "active",
) -> dict[str, object]:
    return {
        "id": identifier,
        "name": f"release-tags-{identifier}",
        "source_type": "Repository",
        "source": REPOSITORY,
        "enforcement": enforcement,
    }


def _tag_ruleset(
    identifier: int = 71,
    *,
    enforcement: str = "active",
) -> dict[str, object]:
    return {
        **_ruleset_summary(identifier, enforcement=enforcement),
        "target": "tag",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "include": ["refs/tags/v*"],
                "exclude": [],
            }
        },
        "rules": [
            {"type": "creation"},
            {"type": "update"},
            {"type": "deletion"},
        ],
    }


def _privileged_tag_rulesets() -> tuple[
    list[dict[str, object]], dict[int, object]
]:
    creation = _tag_ruleset(71)
    creation["rules"] = [{"type": "creation"}]
    creation["bypass_actors"] = [
        {
            "actor_id": RELEASE_CREATOR_APP_ID,
            "actor_type": "Integration",
            "bypass_mode": "always",
        }
    ]
    immutable = _tag_ruleset(72)
    immutable["rules"] = [{"type": "update"}, {"type": "deletion"}]
    return (
        [_ruleset_summary(71), _ruleset_summary(72)],
        {71: creation, 72: immutable},
    )


class GithubStub:
    def __init__(self) -> None:
        self.repository: object = {
            "id": 101,
            "name": "nvidia-converge",
            "full_name": REPOSITORY,
            "default_branch": "main",
            "archived": False,
            "owner": {
                "id": 7,
                "login": "example-org",
                "type": "Organization",
            },
        }
        self.branch: object = {
            "name": "main",
            "protected": True,
            "commit": {"sha": MAIN_SHA},
            "protection_url": f"{API_URL}/repos/{REPOSITORY}/branches/main/protection",
        }
        self.organization: object = {
            "id": 7,
            "login": "example-org",
            "type": "Organization",
            "default_repository_permission": "read",
        }
        self.branch_rule_pages: dict[int, object] = {1: _passing_branch_rules()}
        self.workflow_pages: dict[int, object] = {
            1: _workflow_page(
                [
                    _workflow(41, ".github/workflows/ci.yml"),
                    _workflow(
                        42,
                        ".github/workflows/production-gpu-qualification.yml",
                    ),
                    _workflow(43, ".github/workflows/production-release.yml"),
                    _workflow(
                        44,
                        ".github/workflows/gpu-integration.yml",
                        state="disabled_manually",
                    ),
                    _workflow(
                        45,
                        ".github/workflows/release.yml",
                        state="disabled_manually",
                    ),
                ]
            )
        }
        self.environments: dict[str, object] = {
            "gpu-qualification": _environment("gpu-qualification", identifier=21),
            "release": _environment("release", identifier=22),
        }
        self.environment_branch_policy_pages: dict[str, dict[int, object]] = {
            "gpu-qualification": {
                1: _deployment_branch_policy_page(
                    [_deployment_branch_policy(31)]
                )
            },
            "release": {
                1: _deployment_branch_policy_page(
                    [_deployment_branch_policy(32)]
                )
            },
        }
        self.ruleset_pages: dict[int, object] = {1: [_ruleset_summary(71)]}
        self.ruleset_details: dict[int, object] = {
            11: _branch_ruleset(),
            71: _tag_ruleset(),
        }
        self.immutable_releases: object = {
            "enabled": True,
            "enforced_by_owner": True,
        }
        self.authenticated_user: object = {
            "id": 701,
            "login": "release-auditor",
            "type": "User",
        }
        self.organization_owner_pages: dict[int, object] = {
            1: [copy.deepcopy(self.authenticated_user)]
        }
        self.organization_invitation_pages: dict[int, object] = {1: []}
        self.collaborator_pages: dict[int, object] = {
            1: [
                {
                    **copy.deepcopy(self.authenticated_user),
                    "role_name": "admin",
                    "permissions": {
                        "admin": True,
                        "maintain": True,
                        "pull": True,
                        "push": True,
                        "triage": True,
                    },
                }
            ]
        }
        self.team_pages: dict[int, object] = {1: []}
        self.invitation_pages: dict[int, object] = {1: []}
        self.deploy_key_pages: dict[int, object] = {1: []}
        self.workflow_permissions: object = {
            "default_workflow_permissions": "read",
            "can_approve_pull_request_reviews": False,
        }
        self.installation_pages: dict[int, object] = {
            1: {
                "total_count": 1,
                "installations": [
                    {
                        "id": 801,
                        "app_id": RELEASE_CREATOR_APP_ID,
                        "app_slug": "nvidia-converge-release-creator",
                        "target_type": "Organization",
                        "repository_selection": "selected",
                        "permissions": {
                            "contents": "write",
                            "metadata": "read",
                        },
                        "account": {
                            "id": 7,
                            "login": "example-org",
                            "type": "Organization",
                        },
                    }
                ],
            }
        }
        self.workflow_run_pages: dict[str, dict[int, object]] = {
            status: {
                1: {
                    "total_count": 0,
                    "workflow_runs": [],
                }
            }
            for status in controls.NONTERMINAL_WORKFLOW_RUN_STATUSES
        }
        self.workflow_run_pages["in_progress"][1] = {
            "total_count": 1,
            "workflow_runs": [
                {
                    "id": RELEASE_WORKFLOW_RUN_ID,
                    "workflow_id": 43,
                    "run_attempt": RELEASE_WORKFLOW_RUN_ATTEMPT,
                    "event": "repository_dispatch",
                    "status": "in_progress",
                    "conclusion": None,
                    "head_branch": "main",
                    "head_sha": MAIN_SHA,
                    "path": ".github/workflows/production-release.yml@main",
                }
            ],
        }
        self.calls: list[str] = []

    def __call__(self, url: str, token: str) -> object:
        assert token == TOKEN
        parsed = urlsplit(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "api.github.com"
        self.calls.append(url)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        base = f"/repos/{REPOSITORY}"

        if path == "/user":
            return copy.deepcopy(self.authenticated_user)
        if path == "/orgs/example-org":
            return copy.deepcopy(self.organization)
        if path == "/orgs/example-org/members":
            assert query.get("role") == ["admin"]
            return copy.deepcopy(self.organization_owner_pages.get(_page(query), []))
        if path == "/orgs/example-org/invitations":
            return copy.deepcopy(
                self.organization_invitation_pages.get(_page(query), [])
            )
        if path == "/orgs/example-org/installations":
            return copy.deepcopy(self.installation_pages.get(_page(query), {}))
        if path == base:
            return copy.deepcopy(self.repository)
        if path == f"{base}/branches/main":
            return copy.deepcopy(self.branch)
        if path == f"{base}/rules/branches/main":
            return copy.deepcopy(self.branch_rule_pages.get(_page(query), []))
        if path == f"{base}/actions/workflows":
            return copy.deepcopy(self.workflow_pages.get(_page(query), []))
        if path == f"{base}/actions/runs":
            statuses = query.get("status", [])
            assert len(statuses) == 1
            return copy.deepcopy(
                self.workflow_run_pages[statuses[0]].get(_page(query), {})
            )
        if path == f"{base}/actions/permissions/workflow":
            return copy.deepcopy(self.workflow_permissions)
        if path == f"{base}/collaborators":
            assert query.get("affiliation") == ["all"]
            return copy.deepcopy(self.collaborator_pages.get(_page(query), []))
        if path == f"{base}/teams":
            return copy.deepcopy(self.team_pages.get(_page(query), []))
        if path == f"{base}/invitations":
            return copy.deepcopy(self.invitation_pages.get(_page(query), []))
        if path == f"{base}/keys":
            return copy.deepcopy(self.deploy_key_pages.get(_page(query), []))
        environment_prefix = f"{base}/environments/"
        branch_policy_suffix = "/deployment-branch-policies"
        if path.startswith(environment_prefix) and path.endswith(branch_policy_suffix):
            name = path.removeprefix(environment_prefix).removesuffix(
                branch_policy_suffix
            )
            pages = self.environment_branch_policy_pages.get(name)
            if pages is None:
                raise AssertionError(f"unexpected environment request: {name}")
            return copy.deepcopy(pages.get(_page(query), []))
        if path.startswith(environment_prefix):
            name = path.removeprefix(f"{base}/environments/")
            if name not in self.environments:
                raise AssertionError(f"unexpected environment request: {name}")
            return copy.deepcopy(self.environments[name])
        if path == f"{base}/rulesets":
            return copy.deepcopy(self.ruleset_pages.get(_page(query), []))
        if path.startswith(f"{base}/rulesets/"):
            identifier = int(path.removeprefix(f"{base}/rulesets/"))
            if identifier not in self.ruleset_details:
                raise controls.RepositoryControlError(
                    f"missing ruleset detail: {identifier}"
                )
            return copy.deepcopy(self.ruleset_details[identifier])
        if path == f"{base}/immutable-releases":
            return copy.deepcopy(self.immutable_releases)
        raise AssertionError(f"unexpected GitHub request: {url}")


def _page(query: dict[str, list[str]]) -> int:
    values = query.get("page", ["1"])
    assert len(values) == 1
    return int(values[0])


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> GithubStub:
    stub = GithubStub()
    monkeypatch.setattr(controls, "_github_json", stub)
    return stub


def _check(
    github: GithubStub,
    *,
    scope: controls.ControlScope = "release",
    require_immutable_releases: bool = False,
    release_creator_app_id: int | None = None,
    require_release_writer_isolation: bool = False,
    release_workflow_run_id: int | None = None,
    release_workflow_run_attempt: int | None = None,
) -> list[str]:
    del github
    if require_release_writer_isolation:
        release_workflow_run_id = (
            RELEASE_WORKFLOW_RUN_ID
            if release_workflow_run_id is None
            else release_workflow_run_id
        )
        release_workflow_run_attempt = (
            RELEASE_WORKFLOW_RUN_ATTEMPT
            if release_workflow_run_attempt is None
            else release_workflow_run_attempt
        )
    return controls.check_repository_controls(
        REPOSITORY,
        TOKEN,
        scope=scope,
        require_immutable_releases=require_immutable_releases,
        release_creator_app_id=release_creator_app_id,
        require_release_writer_isolation=require_release_writer_isolation,
        release_workflow_run_id=release_workflow_run_id,
        release_workflow_run_attempt=release_workflow_run_attempt,
    )


@pytest.mark.parametrize("scope", ["gpu", "release"])
def test_passing_repository_controls_use_scope_specific_endpoints(
    github: GithubStub,
    scope: controls.ControlScope,
) -> None:
    assert _check(github, scope=scope) == []

    paths = [urlsplit(url).path for url in github.calls]
    assert f"/repos/{REPOSITORY}" in paths
    assert f"/repos/{REPOSITORY}/branches/main" in paths
    assert f"/repos/{REPOSITORY}/rules/branches/main" in paths
    assert f"/repos/{REPOSITORY}/rulesets/11" in paths
    assert f"/repos/{REPOSITORY}/actions/workflows" in paths
    assert f"/repos/{REPOSITORY}/environments/gpu-qualification" in paths
    assert (
        f"/repos/{REPOSITORY}/environments/gpu-qualification/"
        "deployment-branch-policies"
    ) in paths
    if scope == "gpu":
        assert f"/repos/{REPOSITORY}/rulesets" not in paths
        assert f"/repos/{REPOSITORY}/environments/release" not in paths
    else:
        assert f"/repos/{REPOSITORY}/rulesets" in paths
        assert f"/repos/{REPOSITORY}/rulesets/71" in paths
        assert f"/repos/{REPOSITORY}/environments/release" in paths
        assert (
            f"/repos/{REPOSITORY}/environments/release/"
            "deployment-branch-policies"
        ) in paths


@pytest.mark.parametrize(
    "case",
    [
        "personal-owner",
        "wrong-default",
        "unprotected-main",
        "wrong-branch-name",
        "malformed-repository",
        "malformed-branch",
    ],
)
def test_repository_and_main_branch_metadata_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    if case == "personal-owner":
        assert isinstance(github.repository, dict)
        github.repository["owner"]["type"] = "User"
    elif case == "wrong-default":
        assert isinstance(github.repository, dict)
        github.repository["default_branch"] = "develop"
    elif case == "unprotected-main":
        assert isinstance(github.branch, dict)
        github.branch["protected"] = False
    elif case == "wrong-branch-name":
        assert isinstance(github.branch, dict)
        github.branch["name"] = "Main"
    elif case == "malformed-repository":
        github.repository = []
    else:
        github.branch = []

    assert _check(github, scope="gpu")


@pytest.mark.parametrize(
    "case",
    [
        "missing-ci",
        "missing-production-gpu",
        "missing-production-release",
        "legacy-gpu-active",
        "legacy-release-active",
        "legacy-disabled-inactivity",
        "unexpected-active",
        "duplicate-id",
        "duplicate-node-id",
        "duplicate-path",
        "boolean-id",
        "malformed-path",
        "malformed-state",
        "malformed-entry",
        "malformed-page",
        "missing-total-count",
        "boolean-total-count",
        "inconsistent-total-count",
        "incomplete-page",
    ],
)
def test_workflow_inventory_mutations_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    page = github.workflow_pages[1]
    assert isinstance(page, dict)
    workflows = page["workflows"]
    assert isinstance(workflows, list)
    ci = workflows[0]
    production_gpu = workflows[1]
    production_release = workflows[2]
    legacy_gpu = workflows[3]
    legacy_release = workflows[4]
    assert all(
        isinstance(item, dict)
        for item in (ci, production_gpu, production_release, legacy_gpu, legacy_release)
    )

    if case == "missing-ci":
        workflows.remove(ci)
        page["total_count"] = 4
    elif case == "missing-production-gpu":
        workflows.remove(production_gpu)
        page["total_count"] = 4
    elif case == "missing-production-release":
        workflows.remove(production_release)
        page["total_count"] = 4
    elif case == "legacy-gpu-active":
        legacy_gpu["state"] = "active"
    elif case == "legacy-release-active":
        legacy_release["state"] = "active"
    elif case == "legacy-disabled-inactivity":
        legacy_release["state"] = "disabled_inactivity"
    elif case == "unexpected-active":
        workflows.append(_workflow(99, ".github/workflows/other.yml"))
        page["total_count"] = 6
    elif case == "duplicate-id":
        production_gpu["id"] = ci["id"]
    elif case == "duplicate-node-id":
        production_gpu["node_id"] = ci["node_id"]
    elif case == "duplicate-path":
        production_gpu["path"] = ci["path"]
    elif case == "boolean-id":
        ci["id"] = True
    elif case == "malformed-path":
        ci["path"] = "../ci.yml"
    elif case == "malformed-state":
        ci["state"] = None
    elif case == "malformed-entry":
        workflows[0] = []
    elif case == "malformed-page":
        github.workflow_pages[1] = []
    elif case == "missing-total-count":
        del page["total_count"]
    elif case == "boolean-total-count":
        page["total_count"] = True
    elif case == "inconsistent-total-count":
        first_page = [
            _workflow(1000 + index, f".github/workflows/disabled-{index}.yml", state="disabled_manually")
            for index in range(100)
        ]
        github.workflow_pages[1] = _workflow_page(first_page, total_count=105)
        github.workflow_pages[2] = _workflow_page(
            workflows, total_count=106
        )
    else:
        page["total_count"] = 6

    assert _check(github, scope="gpu")


def test_workflow_inventory_pagination_reaches_later_page(
    github: GithubStub,
) -> None:
    disabled = [
        _workflow(
            1000 + index,
            f".github/workflows/disabled-{index}.yml",
            state="disabled_manually",
        )
        for index in range(100)
    ]
    first_page = github.workflow_pages[1]
    assert isinstance(first_page, dict)
    required = first_page["workflows"]
    assert isinstance(required, list)
    github.workflow_pages = {
        1: _workflow_page(disabled, total_count=105),
        2: _workflow_page(required, total_count=105),
    }

    assert _check(github, scope="gpu") == []
    calls = [
        url
        for url in github.calls
        if urlsplit(url).path.endswith("/actions/workflows")
    ]
    assert {_page(parse_qs(urlsplit(url).query)) for url in calls} == {1, 2}


@pytest.mark.parametrize(
    "case",
    [
        "missing-pr",
        "duplicate-pr",
        "zero-approvals",
        "boolean-approvals",
        "stale-reviews-survive",
        "last-pusher-can-approve",
        "threads-unresolved",
        "missing-status",
        "duplicate-status",
        "non-strict-status",
        "missing-context",
        "duplicate-context",
        "unbound-context",
        "missing-deletion",
        "missing-non-fast-forward",
    ],
)
def test_effective_main_rule_mutations_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    rules = github.branch_rule_pages[1]
    assert isinstance(rules, list)
    pull_request = next(rule for rule in rules if rule["type"] == "pull_request")
    status = next(rule for rule in rules if rule["type"] == "required_status_checks")
    pull_parameters = pull_request["parameters"]
    status_parameters = status["parameters"]
    assert isinstance(pull_parameters, dict)
    assert isinstance(status_parameters, dict)
    checks = status_parameters["required_status_checks"]
    assert isinstance(checks, list)

    if case == "missing-pr":
        rules.remove(pull_request)
    elif case == "duplicate-pr":
        rules.append(copy.deepcopy(pull_request))
    elif case == "zero-approvals":
        pull_parameters["required_approving_review_count"] = 0
    elif case == "boolean-approvals":
        pull_parameters["required_approving_review_count"] = True
    elif case == "stale-reviews-survive":
        pull_parameters["dismiss_stale_reviews_on_push"] = False
    elif case == "last-pusher-can-approve":
        pull_parameters["require_last_push_approval"] = False
    elif case == "threads-unresolved":
        pull_parameters["required_review_thread_resolution"] = False
    elif case == "missing-status":
        rules.remove(status)
    elif case == "duplicate-status":
        rules.append(copy.deepcopy(status))
    elif case == "non-strict-status":
        status_parameters["strict_required_status_checks_policy"] = False
    elif case == "missing-context":
        checks.pop()
    elif case == "duplicate-context":
        checks.append(copy.deepcopy(checks[0]))
    elif case == "unbound-context":
        checks[0]["integration_id"] = None
    elif case == "missing-deletion":
        rules[:] = [rule for rule in rules if rule["type"] != "deletion"]
    else:
        rules[:] = [rule for rule in rules if rule["type"] != "non_fast_forward"]

    assert _check(github, scope="gpu")


@pytest.mark.parametrize(
    "case",
    [
        "missing-detail",
        "malformed-detail",
        "wrong-id",
        "wrong-target",
        "inactive",
        "conflicting-source",
        "missing-conditions",
        "does-not-select-main",
        "excludes-main",
        "missing-effective-rule",
        "duplicate-detail-rule",
    ],
)
def test_effective_main_ruleset_detail_mutations_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    detail = github.ruleset_details[11]
    assert isinstance(detail, dict)

    if case == "missing-detail":
        del github.ruleset_details[11]
    elif case == "malformed-detail":
        github.ruleset_details[11] = []
    elif case == "wrong-id":
        detail["id"] = 12
    elif case == "wrong-target":
        detail["target"] = "tag"
    elif case == "inactive":
        detail["enforcement"] = "evaluate"
    elif case == "conflicting-source":
        rules = github.branch_rule_pages[1]
        assert isinstance(rules, list)
        rules[0]["ruleset_source"] = "another-org/repository"
    elif case == "missing-conditions":
        del detail["conditions"]
    elif case == "does-not-select-main":
        conditions = detail["conditions"]
        assert isinstance(conditions, dict)
        ref_name = conditions["ref_name"]
        assert isinstance(ref_name, dict)
        ref_name["include"] = ["refs/heads/release"]
    elif case == "excludes-main":
        conditions = detail["conditions"]
        assert isinstance(conditions, dict)
        ref_name = conditions["ref_name"]
        assert isinstance(ref_name, dict)
        ref_name["exclude"] = ["refs/heads/main"]
    elif case == "missing-effective-rule":
        rules = detail["rules"]
        assert isinstance(rules, list)
        rules[:] = [rule for rule in rules if rule["type"] != "pull_request"]
    else:
        rules = detail["rules"]
        assert isinstance(rules, list)
        rules.append({"type": "deletion"})

    assert _check(github, scope="gpu")


def test_unprivileged_main_check_reports_rules_when_bypass_inventory_is_hidden(
    github: GithubStub,
) -> None:
    detail = github.ruleset_details[11]
    assert isinstance(detail, dict)
    del detail["bypass_actors"]

    assert _check(github, scope="gpu") == []


@pytest.mark.parametrize(
    "case",
    [
        "hidden",
        "release-creator-app",
        "another-actor",
        "malformed",
        "duplicate",
    ],
)
def test_privileged_release_check_requires_empty_visible_main_bypass_inventory(
    github: GithubStub,
    case: str,
) -> None:
    summaries, tag_details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details.update(tag_details)
    main = github.ruleset_details[11]
    assert isinstance(main, dict)
    actor = {
        "actor_id": RELEASE_CREATOR_APP_ID,
        "actor_type": "Integration",
        "bypass_mode": "always",
    }
    if case == "hidden":
        del main["bypass_actors"]
    elif case == "release-creator-app":
        main["bypass_actors"] = [actor]
    elif case == "another-actor":
        main["bypass_actors"] = [
            {
                "actor_id": 1234,
                "actor_type": "Team",
                "bypass_mode": "pull_request",
            }
        ]
    elif case == "malformed":
        main["bypass_actors"] = [{"actor_type": "Integration"}]
    else:
        main["bypass_actors"] = [actor, copy.deepcopy(actor)]

    errors = _check(
        github,
        scope="release",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
    )

    assert any("main ruleset" in error and "bypass" in error for error in errors)


@pytest.mark.parametrize("environment_name", ["gpu-qualification", "release"])
@pytest.mark.parametrize(
    "case",
    [
        "missing-review-rule",
        "duplicate-review-rule",
        "self-review",
        "empty-reviewers",
        "malformed-reviewer",
        "wrong-name",
        "malformed-environment",
    ],
)
def test_environment_reviewer_gate_mutations_fail_closed(
    github: GithubStub,
    environment_name: str,
    case: str,
) -> None:
    environment = github.environments[environment_name]
    assert isinstance(environment, dict)
    protection_rules = environment["protection_rules"]
    assert isinstance(protection_rules, list)
    review_rule = protection_rules[0]
    assert isinstance(review_rule, dict)

    if case == "missing-review-rule":
        protection_rules.clear()
    elif case == "duplicate-review-rule":
        protection_rules.append(copy.deepcopy(review_rule))
    elif case == "self-review":
        review_rule["prevent_self_review"] = False
    elif case == "empty-reviewers":
        review_rule["reviewers"] = []
    elif case == "malformed-reviewer":
        review_rule["reviewers"] = [{"type": "Robot", "reviewer": None}]
    elif case == "wrong-name":
        environment["name"] = f"{environment_name}-lookalike"
    else:
        github.environments[environment_name] = []

    scope: controls.ControlScope = (
        "gpu" if environment_name == "gpu-qualification" else "release"
    )
    assert _check(github, scope=scope)


@pytest.mark.parametrize("environment_name", ["gpu-qualification", "release"])
@pytest.mark.parametrize(
    "case",
    [
        "missing-config",
        "malformed-config",
        "protected-branches",
        "custom-disabled",
        "no-policies",
        "multiple-policies",
        "wrong-name",
        "wrong-type",
        "boolean-id",
        "zero-id",
        "missing-node-id",
        "duplicate-id",
        "duplicate-node-id",
        "malformed-policy",
        "malformed-page",
        "missing-total-count",
        "boolean-total-count",
        "inconsistent-total-count",
        "incomplete-page",
    ],
)
def test_environment_deployment_branch_policy_mutations_fail_closed(
    github: GithubStub,
    environment_name: str,
    case: str,
) -> None:
    environment = github.environments[environment_name]
    assert isinstance(environment, dict)
    config = environment["deployment_branch_policy"]
    assert isinstance(config, dict)
    pages = github.environment_branch_policy_pages[environment_name]
    page = pages[1]
    assert isinstance(page, dict)
    policies = page["branch_policies"]
    assert isinstance(policies, list)
    policy = policies[0]
    assert isinstance(policy, dict)

    if case == "missing-config":
        del environment["deployment_branch_policy"]
    elif case == "malformed-config":
        environment["deployment_branch_policy"] = []
    elif case == "protected-branches":
        config["protected_branches"] = True
    elif case == "custom-disabled":
        config["custom_branch_policies"] = False
    elif case == "no-policies":
        page["total_count"] = 0
        policies.clear()
    elif case == "multiple-policies":
        policies.append(_deployment_branch_policy(999, name="release/*"))
        page["total_count"] = 2
    elif case == "wrong-name":
        policy["name"] = "release/*"
    elif case == "wrong-type":
        policy["type"] = "tag"
    elif case == "boolean-id":
        policy["id"] = True
    elif case == "zero-id":
        policy["id"] = 0
    elif case == "missing-node-id":
        del policy["node_id"]
    elif case == "duplicate-id":
        duplicate = _deployment_branch_policy(999, name="release/*")
        duplicate["id"] = policy["id"]
        policies.append(duplicate)
        page["total_count"] = 2
    elif case == "duplicate-node-id":
        duplicate = _deployment_branch_policy(999, name="release/*")
        duplicate["node_id"] = policy["node_id"]
        policies.append(duplicate)
        page["total_count"] = 2
    elif case == "malformed-policy":
        policies[0] = []
    elif case == "malformed-page":
        pages[1] = []
    elif case == "missing-total-count":
        del page["total_count"]
    elif case == "boolean-total-count":
        page["total_count"] = True
    elif case == "inconsistent-total-count":
        first_page = [
            _deployment_branch_policy(1000 + index, name=f"branch-{index}")
            for index in range(100)
        ]
        pages[1] = _deployment_branch_policy_page(first_page, total_count=101)
        pages[2] = _deployment_branch_policy_page(
            [_deployment_branch_policy(2000)], total_count=102
        )
    else:
        page["total_count"] = 2

    scope: controls.ControlScope = (
        "gpu" if environment_name == "gpu-qualification" else "release"
    )
    assert _check(github, scope=scope)


def test_environment_deployment_branch_policy_pagination_is_bounded(
    github: GithubStub,
) -> None:
    pages = github.environment_branch_policy_pages["gpu-qualification"]
    policies = [
        _deployment_branch_policy(1000 + index, name=f"branch-{index}")
        for index in range(100)
    ]
    pages[1] = _deployment_branch_policy_page(policies, total_count=101)
    pages[2] = _deployment_branch_policy_page(
        [_deployment_branch_policy(2000)], total_count=101
    )

    assert _check(github, scope="gpu")
    calls = [
        url
        for url in github.calls
        if urlsplit(url).path.endswith("/deployment-branch-policies")
    ]
    assert {_page(parse_qs(urlsplit(url).query)) for url in calls} == {1, 2}


@pytest.mark.parametrize(
    "case",
    [
        "missing-ruleset",
        "summary-detail-id-mismatch",
        "wrong-target",
        "inactive",
        "wrong-include",
        "excluded-tag",
        "missing-creation",
        "missing-update",
        "missing-deletion",
        "duplicate-rule",
        "malformed-summary-page",
        "malformed-detail",
    ],
)
def test_release_tag_ruleset_mutations_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    detail = github.ruleset_details[71]
    assert isinstance(detail, dict)

    if case == "missing-ruleset":
        github.ruleset_pages[1] = []
    elif case == "summary-detail-id-mismatch":
        detail["id"] = 72
    elif case == "wrong-target":
        detail["target"] = "branch"
    elif case == "inactive":
        detail["enforcement"] = "evaluate"
    elif case == "wrong-include":
        detail["conditions"]["ref_name"]["include"] = ["refs/tags/release-*"]
    elif case == "excluded-tag":
        detail["conditions"]["ref_name"]["exclude"] = ["refs/tags/v0.*"]
    elif case.startswith("missing-"):
        missing = case.removeprefix("missing-")
        detail["rules"] = [
            rule for rule in detail["rules"] if rule["type"] != missing
        ]
    elif case == "duplicate-rule":
        detail["rules"].append({"type": "deletion"})
    elif case == "malformed-summary-page":
        github.ruleset_pages[1] = {}
    else:
        github.ruleset_details[71] = []

    assert _check(github, scope="release")


def test_effective_branch_rule_pagination_reaches_later_page(
    github: GithubStub,
) -> None:
    filler_types = tuple(f"required_signature_{index}" for index in range(100))
    github.branch_rule_pages = {
        1: [
            _effective_rule(rule_type, ruleset_id=12)
            for rule_type in filler_types
        ],
        2: _passing_branch_rules(),
    }
    github.ruleset_details[12] = _branch_ruleset(
        12,
        rule_types=filler_types,
    )

    assert _check(github, scope="gpu") == []
    rule_calls = [
        url
        for url in github.calls
        if urlsplit(url).path.endswith("/rules/branches/main")
    ]
    assert {_page(parse_qs(urlsplit(url).query)) for url in rule_calls} == {1, 2}
    detail_paths = [
        urlsplit(url).path
        for url in github.calls
        if "/rulesets/" in urlsplit(url).path
    ]
    assert detail_paths.count(f"/repos/{REPOSITORY}/rulesets/11") == 1
    assert detail_paths.count(f"/repos/{REPOSITORY}/rulesets/12") == 1


def test_paginated_branch_rules_reject_duplicate_items(
    github: GithubStub,
) -> None:
    filler_types = tuple(f"required_signature_{index}" for index in range(100))
    first_page = [
        _effective_rule(rule_type, ruleset_id=12)
        for rule_type in filler_types
    ]
    github.branch_rule_pages = {
        1: first_page,
        2: [copy.deepcopy(first_page[-1]), *_passing_branch_rules()],
    }
    github.ruleset_details[12] = _branch_ruleset(
        12,
        rule_types=filler_types,
    )

    assert _check(github, scope="gpu")


def test_effective_main_ruleset_detail_fetch_is_bounded_across_pages(
    github: GithubStub,
) -> None:
    filler_ids = tuple(1000 + index for index in range(100))
    github.branch_rule_pages = {
        1: [
            _effective_rule(f"auxiliary-{identifier}", ruleset_id=identifier)
            for identifier in filler_ids
        ],
        2: _passing_branch_rules(),
    }
    github.ruleset_details.update(
        {
            identifier: _branch_ruleset(
                identifier,
                rule_types=(f"auxiliary-{identifier}",),
            )
            for identifier in filler_ids
        }
    )

    errors = _check(github, scope="gpu")

    assert any("detail verification limit" in error for error in errors)
    detail_calls = [
        url
        for url in github.calls
        if "/rulesets/" in urlsplit(url).path
    ]
    assert len(detail_calls) == controls.MAX_EFFECTIVE_MAIN_RULESETS


def test_paginated_tag_summaries_reach_later_page_and_fetch_every_detail(
    github: GithubStub,
) -> None:
    summaries = [
        _ruleset_summary(1000 + index, enforcement="disabled")
        for index in range(100)
    ]
    github.ruleset_pages = {1: summaries, 2: [_ruleset_summary(71)]}
    github.ruleset_details.update(
        {
            1000 + index: _tag_ruleset(1000 + index, enforcement="disabled")
            for index in range(100)
        }
    )

    assert _check(github, scope="release") == []
    paths = [urlsplit(url).path for url in github.calls]
    assert f"/repos/{REPOSITORY}/rulesets/1000" in paths
    assert f"/repos/{REPOSITORY}/rulesets/71" in paths
    summary_calls = [url for url in github.calls if urlsplit(url).path.endswith("/rulesets")]
    assert {_page(parse_qs(urlsplit(url).query)) for url in summary_calls} == {1, 2}


def test_separate_tag_creation_and_immutable_update_delete_rules_pass(
    github: GithubStub,
) -> None:
    github.ruleset_pages[1] = [_ruleset_summary(71), _ruleset_summary(72)]
    creation = _tag_ruleset(71)
    creation["rules"] = [{"type": "creation"}]
    immutable = _tag_ruleset(72)
    immutable["rules"] = [{"type": "update"}, {"type": "deletion"}]
    github.ruleset_details = {
        11: _branch_ruleset(),
        71: creation,
        72: immutable,
    }

    assert _check(github, scope="release") == []


def test_privileged_release_tag_topology_passes(
    github: GithubStub,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}

    assert _check(
        github,
        scope="release",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
    ) == []


def test_release_writer_isolation_passes_for_owner_and_exact_creator_app(
    github: GithubStub,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}

    assert _check(
        github,
        scope="release",
        require_immutable_releases=True,
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
        require_release_writer_isolation=True,
    ) == []

    paths = [urlsplit(url).path for url in github.calls]
    assert "/user" in paths
    assert "/orgs/example-org" in paths
    assert "/orgs/example-org/members" in paths
    assert "/orgs/example-org/invitations" in paths
    assert f"/repos/{REPOSITORY}/collaborators" in paths
    assert f"/repos/{REPOSITORY}/teams" in paths
    assert f"/repos/{REPOSITORY}/invitations" in paths
    assert f"/repos/{REPOSITORY}/keys" in paths
    assert f"/repos/{REPOSITORY}/actions/permissions/workflow" in paths
    assert paths.count(f"/repos/{REPOSITORY}/actions/runs") == 2 * len(
        controls.NONTERMINAL_WORKFLOW_RUN_STATUSES
    )
    assert "/orgs/example-org/installations" in paths


def test_release_writer_isolation_rejects_writer_inventory_churn(
    github: GithubStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}
    collaborator_scans = 0

    def changing_inventory(url: str, token: str) -> object:
        nonlocal collaborator_scans
        payload = github(url, token)
        if urlsplit(url).path == f"/repos/{REPOSITORY}/collaborators":
            collaborator_scans += 1
            if collaborator_scans == 2:
                assert isinstance(payload, list)
                payload.append(
                    {
                        "id": 704,
                        "login": "racing-reader",
                        "type": "User",
                        "role_name": "read",
                        "permissions": {
                            "admin": False,
                            "maintain": False,
                            "pull": True,
                            "push": False,
                            "triage": False,
                        },
                    }
                )
        return payload

    monkeypatch.setattr(controls, "_github_json", changing_inventory)
    errors = _check(
        github,
        scope="release",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
        require_release_writer_isolation=True,
    )

    assert any("changed between scans" in error for error in errors)


def test_release_writer_isolation_rejects_workflow_run_inventory_churn(
    github: GithubStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}
    workflow_run_scans = 0

    def changing_inventory(url: str, token: str) -> object:
        nonlocal workflow_run_scans
        payload = github(url, token)
        if urlsplit(url).path == f"/repos/{REPOSITORY}/actions/runs":
            workflow_run_scans += 1
            if workflow_run_scans == len(
                controls.NONTERMINAL_WORKFLOW_RUN_STATUSES
            ) + 1:
                assert isinstance(payload, dict)
                workflow_runs = payload.get("workflow_runs")
                assert isinstance(workflow_runs, list)
                workflow_runs.append(
                    {
                        "id": RELEASE_WORKFLOW_RUN_ID + 1,
                        "workflow_id": 44,
                        "run_attempt": 1,
                        "event": "push",
                        "status": "in_progress",
                        "conclusion": None,
                        "head_branch": "main",
                        "head_sha": MAIN_SHA,
                        "path": ".github/workflows/ci.yml@main",
                    }
                )
                payload["total_count"] = len(workflow_runs)
        return payload

    monkeypatch.setattr(controls, "_github_json", changing_inventory)
    errors = _check(
        github,
        scope="release",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
        require_release_writer_isolation=True,
    )

    assert any("workflow-run inventory changed between scans" in error for error in errors)


@pytest.mark.parametrize(
    "case",
    [
        "audit-user-not-owner",
        "organization-default-write",
        "organization-pending-invitation",
        "non-owner-writer",
        "non-owner-custom-role",
        "write-team",
        "pending-invitation",
        "write-deploy-key",
        "actions-default-write",
        "other-active-workflow",
        "wrong-release-run-attempt",
        "creator-installed-everywhere",
        "creator-extra-write-permission",
        "missing-creator",
        "malformed-installation-page",
    ],
)
def test_release_writer_isolation_mutations_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}
    installation_page = github.installation_pages[1]
    assert isinstance(installation_page, dict)
    installations = installation_page["installations"]
    assert isinstance(installations, list)
    creator = installations[0]
    assert isinstance(creator, dict)

    if case == "audit-user-not-owner":
        github.organization_owner_pages[1] = []
    elif case == "organization-default-write":
        assert isinstance(github.organization, dict)
        github.organization["default_repository_permission"] = "write"
    elif case == "organization-pending-invitation":
        github.organization_invitation_pages[1] = [
            {"id": 990, "role": "direct_member"}
        ]
    elif case == "non-owner-writer":
        github.collaborator_pages[1].append(  # type: ignore[union-attr]
            {
                "id": 702,
                "login": "untrusted-writer",
                "type": "User",
                "role_name": "write",
                "permissions": {
                    "admin": False,
                    "maintain": False,
                    "pull": True,
                    "push": True,
                    "triage": True,
                },
            }
        )
    elif case == "non-owner-custom-role":
        github.collaborator_pages[1].append(  # type: ignore[union-attr]
            {
                "id": 703,
                "login": "custom-reader",
                "type": "User",
                "role_name": "release-manager",
                "permissions": {
                    "admin": False,
                    "maintain": False,
                    "pull": True,
                    "push": False,
                    "triage": False,
                },
            }
        )
    elif case == "write-team":
        github.team_pages[1] = [
            {"id": 901, "slug": "writers", "permission": "push"}
        ]
    elif case == "pending-invitation":
        github.invitation_pages[1] = [{"id": 902}]
    elif case == "write-deploy-key":
        github.deploy_key_pages[1] = [{"id": 903, "read_only": False}]
    elif case == "actions-default-write":
        assert isinstance(github.workflow_permissions, dict)
        github.workflow_permissions["default_workflow_permissions"] = "write"
    elif case == "other-active-workflow":
        github.workflow_run_pages["queued"][1] = {
            "total_count": 1,
            "workflow_runs": [
                {
                    "id": 7654322,
                    "workflow_id": 45,
                    "run_attempt": 1,
                    "event": "workflow_dispatch",
                    "status": "queued",
                    "conclusion": None,
                    "head_branch": "main",
                    "head_sha": MAIN_SHA,
                    "path": ".github/workflows/release.yml@main",
                }
            ],
        }
    elif case == "wrong-release-run-attempt":
        page = github.workflow_run_pages["in_progress"][1]
        assert isinstance(page, dict)
        runs = page["workflow_runs"]
        assert isinstance(runs, list)
        assert isinstance(runs[0], dict)
        runs[0]["run_attempt"] = 2
    elif case == "creator-installed-everywhere":
        creator["repository_selection"] = "all"
    elif case == "creator-extra-write-permission":
        permissions = creator["permissions"]
        assert isinstance(permissions, dict)
        permissions["issues"] = "write"
    elif case == "missing-creator":
        installations.clear()
        installation_page["total_count"] = 0
    else:
        github.installation_pages[1] = []

    assert _check(
        github,
        scope="release",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
        require_release_writer_isolation=True,
    )


@pytest.mark.parametrize(
    "permission",
    [
        "actions",
        "administration",
        "contents",
        "environments",
        "issues",
        "members",
        "organization_secrets",
        "secrets",
        "unknown_future_permission",
        "variables",
        "workflows",
    ],
)
def test_release_writer_isolation_rejects_other_apps_with_release_authority(
    github: GithubStub,
    permission: str,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}
    installation_page = github.installation_pages[1]
    assert isinstance(installation_page, dict)
    installations = installation_page["installations"]
    assert isinstance(installations, list)
    installations.append(
        {
            "id": 802,
            "app_id": 123456,
            "app_slug": "alternate-writer",
            "target_type": "Organization",
            "repository_selection": "selected",
            "permissions": {permission: "write", "metadata": "read"},
            "account": {
                "id": 7,
                "login": "example-org",
                "type": "Organization",
            },
        }
    )
    installation_page["total_count"] = 2

    assert (
        "another GitHub App has write or admin permission"
        in _check(
            github,
            scope="release",
            release_creator_app_id=RELEASE_CREATOR_APP_ID,
            require_release_writer_isolation=True,
        )
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing-inventory",
        "malformed-inventory",
        "malformed-actor",
        "boolean-actor-id",
        "wrong-app-id",
        "wrong-actor-type",
        "wrong-bypass-mode",
        "extra-actor",
        "duplicate-actor",
        "extra-creation-lock",
        "missing-immutable-inventory",
        "immutable-bypass",
        "same-ruleset",
        "missing-update",
        "missing-deletion",
    ],
)
def test_privileged_release_tag_bypass_mutations_fail_closed(
    github: GithubStub,
    case: str,
) -> None:
    summaries, details = _privileged_tag_rulesets()
    github.ruleset_pages[1] = summaries
    github.ruleset_details = {11: _branch_ruleset(), **details}
    creation = details[71]
    immutable = details[72]
    assert isinstance(creation, dict)
    assert isinstance(immutable, dict)
    creation_actors = creation["bypass_actors"]
    assert isinstance(creation_actors, list)
    creator = creation_actors[0]
    assert isinstance(creator, dict)

    if case == "missing-inventory":
        del creation["bypass_actors"]
    elif case == "malformed-inventory":
        creation["bypass_actors"] = {}
    elif case == "malformed-actor":
        creation_actors[0] = []
    elif case == "boolean-actor-id":
        creator["actor_id"] = True
    elif case == "wrong-app-id":
        creator["actor_id"] = RELEASE_CREATOR_APP_ID + 1
    elif case == "wrong-actor-type":
        creator["actor_type"] = "Team"
    elif case == "wrong-bypass-mode":
        creator["bypass_mode"] = "pull_request"
    elif case == "extra-actor":
        creation_actors.append(
            {
                "actor_id": 123,
                "actor_type": "Team",
                "bypass_mode": "always",
            }
        )
    elif case == "duplicate-actor":
        creation_actors.append(copy.deepcopy(creator))
    elif case == "extra-creation-lock":
        extra = _tag_ruleset(73)
        extra["rules"] = [{"type": "creation"}]
        summaries.append(_ruleset_summary(73))
        details[73] = extra
    elif case == "missing-immutable-inventory":
        del immutable["bypass_actors"]
    elif case == "immutable-bypass":
        immutable["bypass_actors"] = copy.deepcopy(creation_actors)
    elif case == "same-ruleset":
        creation["rules"] = [
            {"type": "creation"},
            {"type": "update"},
            {"type": "deletion"},
        ]
        del details[72]
        summaries.pop()
    elif case in {"missing-update", "missing-deletion"}:
        missing = case.removeprefix("missing-")
        rules = immutable["rules"]
        assert isinstance(rules, list)
        immutable["rules"] = [rule for rule in rules if rule["type"] != missing]

    assert _check(
        github,
        scope="release",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
    )


def test_structural_release_check_does_not_require_visible_bypass_inventory(
    github: GithubStub,
) -> None:
    detail = github.ruleset_details[71]
    assert isinstance(detail, dict)
    del detail["bypass_actors"]

    assert _check(github, scope="release") == []


def test_paginated_tag_summaries_reject_duplicate_ids(
    github: GithubStub,
) -> None:
    summaries = [
        _ruleset_summary(1000 + index, enforcement="disabled")
        for index in range(100)
    ]
    github.ruleset_pages = {
        1: summaries,
        2: [copy.deepcopy(summaries[-1]), _ruleset_summary(71)],
    }
    github.ruleset_details.update(
        {
            1000 + index: _tag_ruleset(1000 + index, enforcement="disabled")
            for index in range(100)
        }
    )

    assert _check(github, scope="release")


def test_owner_enforced_immutable_releases_pass(
    github: GithubStub,
) -> None:
    assert _check(
        github,
        scope="release",
        require_immutable_releases=True,
    ) == []
    assert any(
        urlsplit(url).path == f"/repos/{REPOSITORY}/immutable-releases"
        for url in github.calls
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"enabled": False, "enforced_by_owner": True},
        {"enabled": True},
        {"enabled": True, "enforced_by_owner": False},
        {"enabled": True, "enforced_by_owner": 1},
    ],
)
def test_immutable_release_mutations_fail_closed(
    github: GithubStub,
    payload: object,
) -> None:
    github.immutable_releases = payload

    assert _check(
        github,
        scope="release",
        require_immutable_releases=True,
    )


@pytest.mark.parametrize("value", [0, -1, True, "123"])
def test_invalid_release_creator_app_ids_make_no_requests(
    github: GithubStub,
    value: object,
) -> None:
    errors = controls.check_repository_controls(
        REPOSITORY,
        TOKEN,
        scope="release",
        release_creator_app_id=value,  # type: ignore[arg-type]
    )

    assert errors == ["release creator app ID must be a positive integer"]
    assert github.calls == []


def test_release_creator_app_verification_rejects_gpu_scope_without_requests(
    github: GithubStub,
) -> None:
    errors = controls.check_repository_controls(
        REPOSITORY,
        TOKEN,
        scope="gpu",
        release_creator_app_id=RELEASE_CREATOR_APP_ID,
    )

    assert errors == ["release creator app ID verification requires release scope"]
    assert github.calls == []


@pytest.mark.parametrize(
    ("scope", "app_id", "token"),
    [
        ("gpu", RELEASE_CREATOR_APP_ID, TOKEN),
        ("release", None, TOKEN),
        ("release", RELEASE_CREATOR_APP_ID, ""),
    ],
)
def test_release_writer_isolation_requires_privileged_release_context(
    github: GithubStub,
    scope: controls.ControlScope,
    app_id: int | None,
    token: str,
) -> None:
    errors = controls.check_repository_controls(
        REPOSITORY,
        token,
        scope=scope,
        release_creator_app_id=app_id,
        require_release_writer_isolation=True,
    )

    assert errors
    assert github.calls == []


def test_malformed_token_and_untrusted_api_origin_do_not_make_requests(
    github: GithubStub,
) -> None:
    assert controls.check_repository_controls(REPOSITORY, "   ", scope="gpu")
    assert controls.check_repository_controls(
        REPOSITORY,
        TOKEN,
        scope="gpu",
        api_url="https://attacker.invalid",
    )
    assert github.calls == []


def test_api_failure_is_returned_as_a_redacted_control_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_request(url: str, token: str) -> object:
        del url, token
        raise controls.RepositoryControlError("HTTP 403")

    monkeypatch.setattr(controls, "_github_json", fail_request)
    errors = controls.check_repository_controls(REPOSITORY, TOKEN, scope="gpu")

    assert errors
    assert TOKEN not in "\n".join(errors)


class _HttpResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str,
        status: int = 200,
        content_type: str = "application/json; charset=utf-8",
        content_length: str | None = None,
    ) -> None:
        self.payload = payload
        self.url = url
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = (
            str(len(payload)) if content_length is None else content_length
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def geturl(self) -> str:
        return self.url

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class _HttpOpener:
    def __init__(self, response: _HttpResponse) -> None:
        self.response = response
        self.request: Request | None = None
        self.timeout: int | None = None

    def open(self, request: Request, *, timeout: int) -> _HttpResponse:
        self.request = request
        self.timeout = timeout
        return self.response


def test_github_transport_binds_origin_headers_timeout_and_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"{API_URL}/repos/{REPOSITORY}"
    opener = _HttpOpener(_HttpResponse(b'{"ok":true}', url=url))
    monkeypatch.setattr(controls, "_OPENER", opener)

    assert controls._github_json(url, TOKEN) == {"ok": True}
    assert opener.timeout == 20
    assert opener.request is not None
    assert opener.request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert opener.request.get_header("X-github-api-version") == "2026-03-10"
    assert opener.request.get_header("Accept") == "application/vnd.github+json"


@pytest.mark.parametrize(
    ("payload", "response_kwargs"),
    [
        (b'{"key":1,"key":2}', {}),
        (b'{"value":NaN}', {}),
        (b"\xff", {}),
        (b"not-json", {}),
        (b"{}", {"status": 201}),
        (b"{}", {"content_type": "text/plain"}),
        (b"{}", {"content_length": "invalid"}),
        (b"{}", {"content_length": "-1"}),
        (b"{}", {"url": "https://attacker.invalid/redirect"}),
    ],
)
def test_github_transport_rejects_malformed_or_redirected_responses(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    response_kwargs: dict[str, object],
) -> None:
    url = f"{API_URL}/repos/{REPOSITORY}"
    kwargs = {"url": url, **response_kwargs}
    response = _HttpResponse(payload, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(controls, "_OPENER", _HttpOpener(response))

    with pytest.raises(controls.RepositoryControlError):
        controls._github_json(url, TOKEN)


def test_github_transport_rejects_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"{API_URL}/repos/{REPOSITORY}"
    payload = b" " * (controls.MAX_GITHUB_RESPONSE_BYTES + 1)
    response = _HttpResponse(
        payload,
        url=url,
        content_length=str(controls.MAX_GITHUB_RESPONSE_BYTES),
    )
    monkeypatch.setattr(controls, "_OPENER", _HttpOpener(response))

    with pytest.raises(controls.RepositoryControlError, match="safety limit"):
        controls._github_json(url, TOKEN)


@pytest.mark.parametrize("passing", [True, False])
def test_cli_returns_zero_or_two_and_reports_manual_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    passing: bool,
) -> None:
    observed: list[
        tuple[str, str, str, str, bool, int | None, bool, int | None, int | None]
    ] = []

    def check(
        repository: str,
        token: str,
        *,
        scope: str,
        api_url: str = API_URL,
        require_immutable_releases: bool = False,
        release_creator_app_id: int | None = None,
        require_release_writer_isolation: bool = False,
        release_workflow_run_id: int | None = None,
        release_workflow_run_attempt: int | None = None,
    ) -> list[str]:
        observed.append(
            (
                repository,
                token,
                scope,
                api_url,
                require_immutable_releases,
                release_creator_app_id,
                require_release_writer_isolation,
                release_workflow_run_id,
                release_workflow_run_attempt,
            )
        )
        return [] if passing else ["main is not protected"]

    monkeypatch.setattr(controls, "check_repository_controls", check)
    rc = checker_cli.main(
        [
            "--repository",
            REPOSITORY,
            "--scope",
            "release",
            "--github-token",
            TOKEN,
        ]
    )
    output = capsys.readouterr()

    assert rc == (0 if passing else 2)
    assert observed == [
        (REPOSITORY, TOKEN, "release", API_URL, False, None, False, None, None)
    ]
    combined = f"{output.out}\n{output.err}".lower()
    if passing:
        assert "verification remains required" in combined
        assert "immutable" in combined
        assert "runner" in combined
    assert TOKEN not in combined


def test_cli_forwards_privileged_release_verification_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[tuple[bool, int | None, bool, int | None, int | None]] = []

    def check(
        repository: str,
        token: str,
        *,
        scope: str,
        api_url: str = API_URL,
        require_immutable_releases: bool = False,
        release_creator_app_id: int | None = None,
        require_release_writer_isolation: bool = False,
        release_workflow_run_id: int | None = None,
        release_workflow_run_attempt: int | None = None,
    ) -> list[str]:
        del repository, token, scope, api_url
        observed.append(
            (
                require_immutable_releases,
                release_creator_app_id,
                require_release_writer_isolation,
                release_workflow_run_id,
                release_workflow_run_attempt,
            )
        )
        return []

    monkeypatch.setattr(controls, "check_repository_controls", check)
    rc = checker_cli.main(
        [
            "--repository",
            REPOSITORY,
            "--scope",
            "release",
            "--github-token",
            TOKEN,
            "--require-immutable-releases",
            "--release-creator-app-id",
            str(RELEASE_CREATOR_APP_ID),
            "--require-release-writer-isolation",
            "--release-workflow-run-id",
            str(RELEASE_WORKFLOW_RUN_ID),
            "--release-workflow-run-attempt",
            str(RELEASE_WORKFLOW_RUN_ATTEMPT),
        ]
    )
    output = capsys.readouterr()

    assert rc == 0
    assert observed == [
        (
            True,
            RELEASE_CREATOR_APP_ID,
            True,
            RELEASE_WORKFLOW_RUN_ID,
            RELEASE_WORKFLOW_RUN_ATTEMPT,
        )
    ]
    assert "ruleset bypass actors" not in output.out
    assert "immutable-release configuration" not in output.out
    assert "release-writer isolation" not in output.out
    assert "runner" in output.out
    assert TOKEN not in f"{output.out}\n{output.err}"


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_cli_rejects_non_positive_release_creator_app_id(
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        checker_cli.main(
            [
                "--repository",
                REPOSITORY,
                "--scope",
                "release",
                "--release-creator-app-id",
                value,
            ]
        )

    assert exc_info.value.code == 2
