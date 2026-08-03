from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from typing import Any, Literal, TypeGuard
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_ACTIONS_INTEGRATION_ID = 15368
MAX_GITHUB_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PAGES = 10
PER_PAGE = 100
MAX_EFFECTIVE_MAIN_RULESETS = 100
MAX_RELEASE_WRITER_IDENTITIES = 1000
NONTERMINAL_WORKFLOW_RUN_STATUSES = (
    "in_progress",
    "queued",
    "requested",
    "waiting",
    "pending",
)
PRODUCTION_RELEASE_WORKFLOW_PATH = ".github/workflows/production-release.yml"
REQUIRED_CI_CONTEXTS = frozenset(
    {
        "Python 3.10",
        "Python 3.11",
        "Python 3.12",
        "Python 3.13",
        "Python 3.14",
    }
)
REQUIRED_ACTIVE_WORKFLOWS = frozenset(
    {
        ".github/workflows/ci.yml",
        ".github/workflows/production-gpu-qualification.yml",
        ".github/workflows/production-release.yml",
    }
)
ALLOWED_MAIN_BYPASS_ACTORS: frozenset[tuple[str, int, str]] = frozenset()
RETIRED_PRIVILEGED_WORKFLOWS = frozenset(
    {
        ".github/workflows/gpu-integration.yml",
        ".github/workflows/release.yml",
    }
)
_REPOSITORY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$"
)
_WORKFLOW_PATH = re.compile(
    r"^\.github/workflows/[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.ya?ml$"
)

ControlScope = Literal["gpu", "release"]


class RepositoryControlError(RuntimeError):
    pass


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl


_OPENER = build_opener(_RejectRedirects())


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RepositoryControlError(
                f"GitHub API JSON contains duplicate object key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise RepositoryControlError(
        f"GitHub API JSON contains non-finite number {value!r}"
    )


def _github_json(url: str, token: str) -> Any:
    if not url.startswith(f"{GITHUB_API_URL}/"):
        raise RepositoryControlError("refusing an untrusted GitHub API URL")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "nvidia-converge-repository-control-check",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with _OPENER.open(request, timeout=20) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise RepositoryControlError(
                    f"GitHub API returned unexpected HTTP status {status!r}"
                )
            if response.geturl() != url:
                raise RepositoryControlError("GitHub API request was redirected")
            content_type = response.headers.get_content_type()
            if content_type != "application/json":
                raise RepositoryControlError(
                    "GitHub API returned an unexpected content type"
                )
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length, 10)
                except ValueError as exc:
                    raise RepositoryControlError(
                        "GitHub API returned an invalid Content-Length"
                    ) from exc
                if declared_length < 0 or declared_length > MAX_GITHUB_RESPONSE_BYTES:
                    raise RepositoryControlError(
                        "GitHub API response exceeded the safety limit"
                    )
            payload = bytes(response.read(MAX_GITHUB_RESPONSE_BYTES + 1))
    except HTTPError as exc:
        raise RepositoryControlError(
            f"GitHub API request failed with HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RepositoryControlError(
            f"GitHub API request failed ({type(exc).__name__})"
        ) from exc
    if len(payload) > MAX_GITHUB_RESPONSE_BYTES:
        raise RepositoryControlError("GitHub API response exceeded the safety limit")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise RepositoryControlError("GitHub API returned invalid UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise RepositoryControlError("GitHub API returned invalid JSON") from exc


def _paged_list(base_url: str, token: str) -> list[Any]:
    separator = "&" if "?" in base_url else "?"
    values: list[Any] = []
    for page in range(1, MAX_PAGES + 1):
        payload = _github_json(
            f"{base_url}{separator}per_page={PER_PAGE}&page={page}", token
        )
        if not isinstance(payload, list):
            raise RepositoryControlError("GitHub API paginated response is not a list")
        if len(payload) > PER_PAGE:
            raise RepositoryControlError(
                "GitHub API page contains more entries than requested"
            )
        values.extend(payload)
        if len(payload) < PER_PAGE:
            return values
    raise RepositoryControlError("GitHub API pagination exceeded the safety limit")


def _paged_deployment_branch_policies(base_url: str, token: str) -> list[Any]:
    separator = "&" if "?" in base_url else "?"
    values: list[Any] = []
    expected_total: int | None = None
    for page in range(1, MAX_PAGES + 1):
        payload = _github_json(
            f"{base_url}{separator}per_page={PER_PAGE}&page={page}", token
        )
        if not isinstance(payload, dict):
            raise RepositoryControlError(
                "deployment branch-policy response is not an object"
            )
        total_count = payload.get("total_count")
        policies = payload.get("branch_policies")
        if not _is_int(total_count) or total_count < 0:
            raise RepositoryControlError(
                "deployment branch-policy response has an invalid total count"
            )
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise RepositoryControlError(
                "deployment branch-policy total count changed during pagination"
            )
        if not isinstance(policies, list):
            raise RepositoryControlError(
                "deployment branch-policy page is not a list"
            )
        if len(policies) > PER_PAGE:
            raise RepositoryControlError(
                "deployment branch-policy page contains more entries than requested"
            )
        values.extend(policies)
        if len(values) > total_count:
            raise RepositoryControlError(
                "deployment branch-policy inventory exceeds its total count"
            )
        if len(values) == total_count:
            return values
        if len(policies) < PER_PAGE:
            raise RepositoryControlError(
                "deployment branch-policy inventory is incomplete"
            )
    raise RepositoryControlError(
        "deployment branch-policy pagination exceeded the safety limit"
    )


def _paged_workflows(base_url: str, token: str) -> list[Any]:
    separator = "&" if "?" in base_url else "?"
    values: list[Any] = []
    expected_total: int | None = None
    for page in range(1, MAX_PAGES + 1):
        payload = _github_json(
            f"{base_url}{separator}per_page={PER_PAGE}&page={page}", token
        )
        if not isinstance(payload, dict):
            raise RepositoryControlError("workflow inventory response is not an object")
        total_count = payload.get("total_count")
        workflows = payload.get("workflows")
        if not _is_int(total_count) or total_count < 0:
            raise RepositoryControlError(
                "workflow inventory response has an invalid total count"
            )
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise RepositoryControlError(
                "workflow inventory total count changed during pagination"
            )
        if not isinstance(workflows, list):
            raise RepositoryControlError("workflow inventory page is not a list")
        if len(workflows) > PER_PAGE:
            raise RepositoryControlError(
                "workflow inventory page contains more entries than requested"
            )
        values.extend(workflows)
        if len(values) > total_count:
            raise RepositoryControlError(
                "workflow inventory exceeds its total count"
            )
        if len(values) == total_count:
            return values
        if len(workflows) < PER_PAGE:
            raise RepositoryControlError("workflow inventory is incomplete")
    raise RepositoryControlError("workflow inventory pagination exceeded the safety limit")


def _paged_installations(base_url: str, token: str) -> list[Any]:
    separator = "&" if "?" in base_url else "?"
    values: list[Any] = []
    expected_total: int | None = None
    for page in range(1, MAX_PAGES + 1):
        payload = _github_json(
            f"{base_url}{separator}per_page={PER_PAGE}&page={page}", token
        )
        if not isinstance(payload, dict):
            raise RepositoryControlError(
                "GitHub App installation response is not an object"
            )
        total_count = payload.get("total_count")
        installations = payload.get("installations")
        if (
            not _is_int(total_count)
            or not 0 <= total_count <= MAX_RELEASE_WRITER_IDENTITIES
        ):
            raise RepositoryControlError(
                "GitHub App installation response has an invalid total count"
            )
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise RepositoryControlError(
                "GitHub App installation total count changed during pagination"
            )
        if not isinstance(installations, list) or len(installations) > PER_PAGE:
            raise RepositoryControlError(
                "GitHub App installation page has an invalid inventory"
            )
        values.extend(installations)
        if len(values) > total_count:
            raise RepositoryControlError(
                "GitHub App installation inventory exceeds its total count"
            )
        if len(values) == total_count:
            return values
        if len(installations) < PER_PAGE:
            raise RepositoryControlError(
                "GitHub App installation inventory is incomplete"
            )
    raise RepositoryControlError(
        "GitHub App installation pagination exceeded the safety limit"
    )


def _paged_workflow_runs(base_url: str, token: str) -> list[Any]:
    separator = "&" if "?" in base_url else "?"
    values: list[Any] = []
    expected_total: int | None = None
    for page in range(1, MAX_PAGES + 1):
        payload = _github_json(
            f"{base_url}{separator}per_page={PER_PAGE}&page={page}", token
        )
        if not isinstance(payload, dict):
            raise RepositoryControlError("workflow-run response is not an object")
        total_count = payload.get("total_count")
        runs = payload.get("workflow_runs")
        if (
            not _is_int(total_count)
            or not 0 <= total_count <= MAX_RELEASE_WRITER_IDENTITIES
        ):
            raise RepositoryControlError(
                "workflow-run response has an invalid total count"
            )
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise RepositoryControlError(
                "workflow-run total count changed during pagination"
            )
        if not isinstance(runs, list) or len(runs) > PER_PAGE:
            raise RepositoryControlError(
                "workflow-run page has an invalid inventory"
            )
        values.extend(runs)
        if len(values) > total_count:
            raise RepositoryControlError(
                "workflow-run inventory exceeds its total count"
            )
        if len(values) == total_count:
            return values
        if len(runs) < PER_PAGE:
            raise RepositoryControlError("workflow-run inventory is incomplete")
    raise RepositoryControlError("workflow-run pagination exceeded the safety limit")


def _inventory_fingerprint(values: list[Any]) -> str:
    members = sorted(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for value in values
    )
    return json.dumps(members, ensure_ascii=True, separators=(",", ":"))


def _stable_paged_list(base_url: str, token: str) -> list[Any]:
    first = _paged_list(base_url, token)
    second = _paged_list(base_url, token)
    if _inventory_fingerprint(first) != _inventory_fingerprint(second):
        raise RepositoryControlError("GitHub API inventory changed between scans")
    return second


def _stable_paged_workflows(base_url: str, token: str) -> list[Any]:
    first = _paged_workflows(base_url, token)
    second = _paged_workflows(base_url, token)
    if _inventory_fingerprint(first) != _inventory_fingerprint(second):
        raise RepositoryControlError("workflow inventory changed between scans")
    return second


def _stable_paged_installations(base_url: str, token: str) -> list[Any]:
    first = _paged_installations(base_url, token)
    second = _paged_installations(base_url, token)
    if _inventory_fingerprint(first) != _inventory_fingerprint(second):
        raise RepositoryControlError(
            "GitHub App installation inventory changed between scans"
        )
    return second


def _workflow_run_authority_fingerprint(values: list[Any]) -> str:
    authority = []
    for value in values:
        if not isinstance(value, dict):
            authority.append(value)
            continue
        authority.append(
            {
                key: value.get(key)
                for key in (
                    "conclusion",
                    "event",
                    "head_branch",
                    "head_sha",
                    "id",
                    "path",
                    "run_attempt",
                    "status",
                    "workflow_id",
                )
            }
        )
    return _inventory_fingerprint(authority)


def _nonterminal_workflow_runs(base_url: str, token: str) -> list[Any]:
    values: list[Any] = []
    for status in NONTERMINAL_WORKFLOW_RUN_STATUSES:
        values.extend(
            _paged_workflow_runs(f"{base_url}?status={status}", token)
        )
    return values


def _stable_nonterminal_workflow_runs(base_url: str, token: str) -> list[Any]:
    first = _nonterminal_workflow_runs(base_url, token)
    second = _nonterminal_workflow_runs(base_url, token)
    if _workflow_run_authority_fingerprint(
        first
    ) != _workflow_run_authority_fingerprint(second):
        raise RepositoryControlError(
            "nonterminal workflow-run inventory changed between scans"
        )
    return second


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _check_repository(payload: Any, repository: str) -> list[str]:
    if not isinstance(payload, dict):
        return ["repository metadata is not an object"]
    errors: list[str] = []
    if payload.get("full_name") != repository:
        errors.append("repository metadata does not match the requested repository")
    owner = payload.get("owner")
    if not isinstance(owner, dict) or owner.get("type") != "Organization":
        errors.append(
            "repository must be organization-owned for the required GPU runner group"
        )
    if payload.get("default_branch") != "main":
        errors.append("repository default branch must be main")
    if payload.get("archived") is not False:
        errors.append("repository must be active and not archived")
    return errors


def _check_main_branch(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["main branch metadata is not an object"]
    errors: list[str] = []
    if payload.get("name") != "main":
        errors.append("main branch metadata has the wrong name")
    if payload.get("protected") is not True:
        errors.append("main branch is not protected")
    commit = payload.get("commit")
    if (
        not isinstance(commit, dict)
        or not isinstance(commit.get("sha"), str)
        or re.fullmatch(r"[a-f0-9]{40}", commit["sha"]) is None
    ):
        errors.append("main branch metadata has an invalid commit identity")
    return errors


def _valid_pull_request_rule(rule: dict[str, Any]) -> bool:
    parameters = rule.get("parameters")
    if not isinstance(parameters, dict):
        return False
    approvals = parameters.get("required_approving_review_count")
    return (
        _is_int(approvals)
        and approvals >= 1
        and parameters.get("dismiss_stale_reviews_on_push") is True
        and parameters.get("require_last_push_approval") is True
        and parameters.get("required_review_thread_resolution") is True
    )


def _valid_status_check_rule(rule: dict[str, Any]) -> bool:
    parameters = rule.get("parameters")
    if not isinstance(parameters, dict):
        return False
    if parameters.get("strict_required_status_checks_policy") is not True:
        return False
    checks = parameters.get("required_status_checks")
    if not isinstance(checks, list) or not checks:
        return False
    contexts: dict[str, int] = {}
    for check in checks:
        if not isinstance(check, dict):
            return False
        context = check.get("context")
        integration_id = check.get("integration_id")
        if (
            not isinstance(context, str)
            or not context
            or context in contexts
            or not _is_int(integration_id)
        ):
            return False
        contexts[context] = integration_id
    return all(
        contexts.get(context) == GITHUB_ACTIONS_INTEGRATION_ID
        for context in REQUIRED_CI_CONTEXTS
    )


def _check_main_rules(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return ["effective main-branch rules are not a list"]
    errors: list[str] = []
    rules: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            errors.append("effective main-branch rule is not an object")
            continue
        rule_type = item.get("type")
        ruleset_id = item.get("ruleset_id")
        if (
            not isinstance(rule_type, str)
            or not rule_type
            or not _is_int(ruleset_id)
            or ruleset_id <= 0
        ):
            errors.append("effective main-branch rule has an invalid identity")
            continue
        identity = (ruleset_id, rule_type)
        if identity in seen:
            errors.append("effective main-branch rules contain a duplicate rule")
            continue
        seen.add(identity)
        rules.append(item)
    if not any(
        rule.get("type") == "pull_request" and _valid_pull_request_rule(rule)
        for rule in rules
    ):
        errors.append("main lacks a canonical required pull-request review rule")
    if not any(
        rule.get("type") == "required_status_checks" and _valid_status_check_rule(rule)
        for rule in rules
    ):
        errors.append("main lacks strict, app-bound required CI checks")
    rule_types = {rule.get("type") for rule in rules}
    if "deletion" not in rule_types:
        errors.append("main branch deletion is not blocked")
    if "non_fast_forward" not in rule_types:
        errors.append("main branch force pushes are not blocked")
    return errors


def _check_workflow_inventory(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return ["repository workflow inventory is not a list"]
    errors: list[str] = []
    seen_ids: set[int] = set()
    seen_node_ids: set[str] = set()
    seen_paths: set[str] = set()
    states: dict[str, str] = {}
    for workflow in payload:
        if not isinstance(workflow, dict):
            errors.append("repository workflow metadata is not an object")
            continue
        identifier = workflow.get("id")
        node_id = workflow.get("node_id")
        path = workflow.get("path")
        state = workflow.get("state")
        if (
            not _is_int(identifier)
            or identifier <= 0
            or not isinstance(node_id, str)
            or not node_id
            or not isinstance(path, str)
            or _WORKFLOW_PATH.fullmatch(path) is None
            or not isinstance(state, str)
            or not state
        ):
            errors.append("repository workflow metadata has an invalid identity")
            continue
        if identifier in seen_ids or node_id in seen_node_ids or path in seen_paths:
            errors.append("repository workflow inventory contains duplicate metadata")
            continue
        seen_ids.add(identifier)
        seen_node_ids.add(node_id)
        seen_paths.add(path)
        states[path] = state

    active_paths = {path for path, state in states.items() if state == "active"}
    missing = sorted(REQUIRED_ACTIVE_WORKFLOWS - active_paths)
    unexpected = sorted(active_paths - REQUIRED_ACTIVE_WORKFLOWS)
    if missing:
        errors.append(
            "required production workflows are not active: " + ", ".join(missing)
        )
    if unexpected:
        errors.append(
            "unexpected repository workflows remain active: "
            + ", ".join(unexpected)
        )
    improperly_retired = sorted(
        path
        for path in RETIRED_PRIVILEGED_WORKFLOWS
        if path in states and states[path] != "disabled_manually"
    )
    if improperly_retired:
        errors.append(
            "legacy privileged workflows are not disabled manually: "
            + ", ".join(improperly_retired)
        )
    return errors


def _check_environment(payload: Any, name: str) -> list[str]:
    if not isinstance(payload, dict):
        return [f"environment {name!r} metadata is not an object"]
    errors: list[str] = []
    if payload.get("name") != name:
        errors.append(f"environment {name!r} metadata has the wrong name")
    deployment_branch_policy = payload.get("deployment_branch_policy")
    if not isinstance(deployment_branch_policy, dict):
        errors.append(
            f"environment {name!r} has invalid deployment branch-policy metadata"
        )
    elif (
        deployment_branch_policy.get("protected_branches") is not False
        or deployment_branch_policy.get("custom_branch_policies") is not True
    ):
        errors.append(
            f"environment {name!r} must use custom deployment branch policies only"
        )
    protection_rules = payload.get("protection_rules")
    if not isinstance(protection_rules, list):
        return errors + [f"environment {name!r} has invalid protection rules"]
    reviewer_rules = [
        rule
        for rule in protection_rules
        if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
    ]
    if len(reviewer_rules) != 1:
        errors.append(
            f"environment {name!r} must have exactly one required-reviewers rule"
        )
        return errors
    reviewer_rule = reviewer_rules[0]
    if reviewer_rule.get("prevent_self_review") is not True:
        errors.append(f"environment {name!r} permits self-review")
    reviewers = reviewer_rule.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers:
        errors.append(f"environment {name!r} has no required reviewer")
        return errors
    seen_reviewers: set[tuple[str, int]] = set()
    for reviewer_entry in reviewers:
        if not isinstance(reviewer_entry, dict):
            errors.append(f"environment {name!r} has malformed reviewer metadata")
            continue
        reviewer_type = reviewer_entry.get("type")
        reviewer = reviewer_entry.get("reviewer")
        reviewer_id = reviewer.get("id") if isinstance(reviewer, dict) else None
        if reviewer_type not in {"User", "Team"} or not _is_int(reviewer_id):
            errors.append(f"environment {name!r} has malformed reviewer metadata")
            continue
        identity = (reviewer_type, reviewer_id)
        if identity in seen_reviewers:
            errors.append(f"environment {name!r} has duplicate reviewer metadata")
            continue
        seen_reviewers.add(identity)
    return errors


def _check_deployment_branch_policies(payload: Any, name: str) -> list[str]:
    if not isinstance(payload, list):
        return [f"environment {name!r} deployment branch policies are not a list"]
    errors: list[str] = []
    seen_ids: set[int] = set()
    seen_node_ids: set[str] = set()
    valid_main_policies = 0
    for policy in payload:
        if not isinstance(policy, dict):
            errors.append(
                f"environment {name!r} has malformed deployment branch-policy metadata"
            )
            continue
        identifier = policy.get("id")
        node_id = policy.get("node_id")
        policy_name = policy.get("name")
        policy_type = policy.get("type")
        if (
            not _is_int(identifier)
            or identifier <= 0
            or not isinstance(node_id, str)
            or not node_id
            or not isinstance(policy_name, str)
            or not policy_name
            or not isinstance(policy_type, str)
            or not policy_type
        ):
            errors.append(
                f"environment {name!r} has malformed deployment branch-policy metadata"
            )
            continue
        if identifier in seen_ids or node_id in seen_node_ids:
            errors.append(
                f"environment {name!r} has duplicate deployment branch-policy metadata"
            )
            continue
        seen_ids.add(identifier)
        seen_node_ids.add(node_id)
        if policy_name == "main" and policy_type == "branch":
            valid_main_policies += 1
    if len(payload) != 1 or valid_main_policies != 1:
        errors.append(
            f"environment {name!r} must allow deployments from exactly main as a branch"
        )
    return errors


def _active_tag_ruleset_types(
    payload: Any, expected_id: int
) -> frozenset[str] | None:
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("id") != expected_id
        or payload.get("target") != "tag"
        or payload.get("enforcement") != "active"
    ):
        return None
    conditions = payload.get("conditions")
    ref_name = conditions.get("ref_name") if isinstance(conditions, dict) else None
    if not isinstance(ref_name, dict):
        return None
    include = ref_name.get("include")
    exclude = ref_name.get("exclude")
    if (
        not isinstance(include, list)
        or "refs/tags/v*" not in include
        or exclude != []
        or not all(isinstance(pattern, str) for pattern in include)
    ):
        return None
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return None
    types: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
            return None
        types.append(rule["type"])
    if len(types) != len(set(types)):
        return None
    return frozenset(types)


def _bypass_actor_inventory(
    payload: dict[str, Any],
) -> tuple[tuple[str, int, str], ...] | None:
    actors = payload.get("bypass_actors")
    if not isinstance(actors, list):
        return None
    inventory: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for actor in actors:
        if not isinstance(actor, dict):
            return None
        actor_type = actor.get("actor_type")
        actor_id = actor.get("actor_id")
        bypass_mode = actor.get("bypass_mode")
        if (
            not isinstance(actor_type, str)
            or not actor_type
            or not _is_int(actor_id)
            or actor_id <= 0
            or not isinstance(bypass_mode, str)
            or not bypass_mode
        ):
            return None
        identity = (actor_type, actor_id, bypass_mode)
        if identity in seen:
            return None
        seen.add(identity)
        inventory.append(identity)
    return tuple(inventory)


def _main_ref_pattern_matches(pattern: str) -> bool:
    if pattern in {"~ALL", "~DEFAULT_BRANCH"}:
        return True
    return pattern.startswith("refs/heads/") and fnmatch.fnmatchcase(
        "refs/heads/main", pattern
    )


def _effective_main_ruleset_ids(payload: Any) -> tuple[int, ...]:
    if not isinstance(payload, list):
        return ()
    return tuple(
        sorted(
            {
                ruleset_id
                for rule in payload
                if isinstance(rule, dict)
                and _is_int(ruleset_id := rule.get("ruleset_id"))
                and ruleset_id > 0
            }
        )
    )


def _check_main_ruleset_details(
    effective_rules: Any,
    details: dict[int, Any],
    *,
    require_bypass_inventory: bool,
) -> list[str]:
    if not isinstance(effective_rules, list):
        return ["effective main-branch rules are not a list"]
    errors: list[str] = []
    expected_ids = _effective_main_ruleset_ids(effective_rules)
    if len(expected_ids) > MAX_EFFECTIVE_MAIN_RULESETS:
        errors.append(
            "effective main-branch rules reference more rulesets than the "
            "detail verification limit"
        )
    fetched_ids = tuple(sorted(details))
    if fetched_ids != expected_ids[:MAX_EFFECTIVE_MAIN_RULESETS]:
        errors.append("effective main ruleset detail inventory is incomplete")

    for ruleset_id in expected_ids[:MAX_EFFECTIVE_MAIN_RULESETS]:
        detail = details.get(ruleset_id)
        if not isinstance(detail, dict):
            errors.append(
                f"effective main ruleset {ruleset_id} detail is not an object"
            )
            continue
        attributed = [
            rule
            for rule in effective_rules
            if isinstance(rule, dict) and rule.get("ruleset_id") == ruleset_id
        ]
        sources = {
            (rule.get("ruleset_source_type"), rule.get("ruleset_source"))
            for rule in attributed
        }
        source_identity = (detail.get("source_type"), detail.get("source"))
        if (
            detail.get("id") != ruleset_id
            or detail.get("target") != "branch"
            or detail.get("enforcement") != "active"
            or len(sources) != 1
            or source_identity not in sources
        ):
            errors.append(
                f"effective main ruleset {ruleset_id} has a conflicting identity"
            )

        conditions = detail.get("conditions")
        ref_name = (
            conditions.get("ref_name") if isinstance(conditions, dict) else None
        )
        includes = ref_name.get("include") if isinstance(ref_name, dict) else None
        excludes = ref_name.get("exclude") if isinstance(ref_name, dict) else None
        if (
            not isinstance(includes, list)
            or not includes
            or not all(isinstance(pattern, str) for pattern in includes)
            or not isinstance(excludes, list)
            or not all(isinstance(pattern, str) for pattern in excludes)
            or not any(_main_ref_pattern_matches(pattern) for pattern in includes)
            or any(_main_ref_pattern_matches(pattern) for pattern in excludes)
        ):
            errors.append(
                f"effective main ruleset {ruleset_id} no longer selects main"
            )

        rules = detail.get("rules")
        detail_types: list[str] = []
        if isinstance(rules, list):
            for rule in rules:
                rule_type = rule.get("type") if isinstance(rule, dict) else None
                if not isinstance(rule_type, str) or not rule_type:
                    detail_types = []
                    break
                detail_types.append(rule_type)
        attributed_types = {
            rule.get("type")
            for rule in attributed
            if isinstance(rule.get("type"), str)
        }
        if (
            not detail_types
            or len(detail_types) != len(set(detail_types))
            or not attributed_types.issubset(detail_types)
        ):
            errors.append(
                f"effective main ruleset {ruleset_id} detail conflicts with its rules"
            )

        bypass_present = "bypass_actors" in detail
        bypass_inventory = _bypass_actor_inventory(detail)
        if bypass_present and bypass_inventory is None:
            errors.append(
                f"effective main ruleset {ruleset_id} has an invalid bypass inventory"
            )
        elif not bypass_present:
            if require_bypass_inventory:
                errors.append(
                    f"effective main ruleset {ruleset_id} bypass inventory is not visible"
                )
        elif frozenset(bypass_inventory or ()) != ALLOWED_MAIN_BYPASS_ACTORS:
            errors.append(
                f"effective main ruleset {ruleset_id} permits an unauthorized bypass actor"
            )
    return errors


def _check_tag_rulesets(
    summaries: Any,
    details: dict[int, Any],
    *,
    release_creator_app_id: int | None = None,
) -> list[str]:
    if not isinstance(summaries, list):
        return ["tag ruleset summaries are not a list"]
    errors: list[str] = []
    ids: list[int] = []
    for summary in summaries:
        ruleset_id = summary.get("id") if isinstance(summary, dict) else None
        if not _is_int(ruleset_id) or ruleset_id <= 0:
            errors.append("tag ruleset summary has an invalid identity")
            continue
        if ruleset_id in ids:
            errors.append("tag ruleset summaries contain a duplicate identity")
            continue
        ids.append(ruleset_id)
    if set(details) != set(ids):
        errors.append("tag ruleset detail inventory is incomplete")
    active_types = [
        rule_types
        for ruleset_id in ids
        if (
            rule_types := _active_tag_ruleset_types(
                details.get(ruleset_id), ruleset_id
            )
        )
        is not None
    ]
    if not any("creation" in rule_types for rule_types in active_types):
        errors.append("v* tags lack an active creation restriction")
    if not any(
        {"update", "deletion"}.issubset(rule_types)
        for rule_types in active_types
    ):
        errors.append(
            "v* tags lack one active update/deletion lock with no exclusions"
        )
    if release_creator_app_id is not None:
        expected_creator = (
            ("Integration", release_creator_app_id, "always"),
        )
        creation_ids: list[int] = []
        immutable_ids: list[int] = []
        for ruleset_id in ids:
            detail = details.get(ruleset_id)
            rule_types = _active_tag_ruleset_types(detail, ruleset_id)
            if rule_types is None:
                continue
            assert isinstance(detail, dict)
            inventory = _bypass_actor_inventory(detail)
            if inventory is None:
                errors.append(
                    f"active v* tag ruleset {ruleset_id} has an invalid bypass inventory"
                )
                continue
            if "creation" in rule_types:
                if inventory == expected_creator:
                    creation_ids.append(ruleset_id)
                else:
                    errors.append(
                        "every active v* creation restriction must be bypassable "
                        "only by the configured release creator app"
                    )
            if {"update", "deletion"}.issubset(rule_types) and inventory == ():
                immutable_ids.append(ruleset_id)
        if not creation_ids:
            errors.append(
                "v* tag creation lacks the configured release creator app bypass"
            )
        if not immutable_ids:
            errors.append(
                "v* tag updates and deletion lack an active no-bypass lock"
            )
        elif creation_ids and not any(
            creation_id != immutable_id
            for creation_id in creation_ids
            for immutable_id in immutable_ids
        ):
            errors.append(
                "v* tag creation and immutable update/deletion controls must be separate"
            )
    return errors


def _check_immutable_releases(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["immutable-release metadata is not an object"]
    if payload.get("enabled") is not True:
        return ["immutable releases are not enabled"]
    if payload.get("enforced_by_owner") is not True:
        return ["immutable releases are not enforced by the repository owner"]
    return []


def _github_user_identity(value: Any) -> tuple[int, str] | None:
    if not isinstance(value, dict):
        return None
    identifier = value.get("id")
    login = value.get("login")
    if (
        not _is_int(identifier)
        or identifier <= 0
        or not isinstance(login, str)
        or not login
        or value.get("type") != "User"
    ):
        return None
    return identifier, login


def _check_nonterminal_workflow_runs(
    workflow_runs: Any,
    release_workflow_run_id: int,
    release_workflow_run_attempt: int,
    repository_main_sha: str | None,
) -> list[str]:
    if (
        not isinstance(workflow_runs, list)
        or len(workflow_runs) > MAX_RELEASE_WRITER_IDENTITIES
    ):
        return ["nonterminal workflow-run inventory is invalid or unbounded"]
    errors: list[str] = []
    seen_ids: set[int] = set()
    release_run_count = 0
    allowed_paths = {
        PRODUCTION_RELEASE_WORKFLOW_PATH,
        f"{PRODUCTION_RELEASE_WORKFLOW_PATH}@main",
        f"{PRODUCTION_RELEASE_WORKFLOW_PATH}@refs/heads/main",
    }
    for run in workflow_runs:
        run_id = run.get("id") if isinstance(run, dict) else None
        workflow_id = run.get("workflow_id") if isinstance(run, dict) else None
        run_attempt = run.get("run_attempt") if isinstance(run, dict) else None
        status = run.get("status") if isinstance(run, dict) else None
        if (
            not _is_int(run_id)
            or run_id <= 0
            or run_id in seen_ids
            or not _is_int(workflow_id)
            or workflow_id <= 0
            or not _is_int(run_attempt)
            or run_attempt <= 0
            or status not in NONTERMINAL_WORKFLOW_RUN_STATUSES
            or run.get("conclusion") is not None
        ):
            errors.append("nonterminal workflow-run inventory contains an invalid run")
            continue
        seen_ids.add(run_id)
        if run_id != release_workflow_run_id:
            errors.append(
                "another queued or running GitHub Actions workflow retains authority"
            )
            continue
        release_run_count += 1
        if (
            run.get("event") != "repository_dispatch"
            or run.get("head_branch") != "main"
            or run.get("path") not in allowed_paths
            or run.get("head_sha") != repository_main_sha
            or run_attempt != release_workflow_run_attempt
        ):
            errors.append(
                "the allowed release workflow run has an invalid source identity"
            )
    if release_run_count != 1:
        errors.append(
            "the current release workflow is not the sole visible nonterminal run"
        )
    return errors


def _check_release_writer_isolation(
    *,
    organization: Any,
    authenticated_user: Any,
    organization_owners: Any,
    organization_invitations: Any,
    collaborators: Any,
    teams: Any,
    invitations: Any,
    deploy_keys: Any,
    workflow_permissions: Any,
    installations: Any,
    workflow_runs: Any,
    repository_owner_id: int | None,
    repository_owner_login: str,
    release_creator_app_id: int,
    release_workflow_run_id: int,
    release_workflow_run_attempt: int,
    repository_main_sha: str | None,
) -> list[str]:
    """Reject every visible release writer except trusted org owners and the App."""

    errors: list[str] = []
    if (
        not isinstance(organization, dict)
        or organization.get("id") != repository_owner_id
        or organization.get("login") != repository_owner_login
        or organization.get("type") != "Organization"
        or organization.get("default_repository_permission") not in {"none", "read"}
    ):
        errors.append(
            "organization metadata does not prove read-only or no default repository access"
        )
    if not isinstance(organization_invitations, list):
        errors.append("organization invitation inventory is invalid")
    elif organization_invitations:
        errors.append("organization has a pending membership invitation")

    if (
        not isinstance(organization_owners, list)
        or len(organization_owners) > MAX_RELEASE_WRITER_IDENTITIES
    ):
        return ["organization-owner inventory is invalid or unbounded"]
    owner_ids: set[int] = set()
    owner_logins: set[str] = set()
    for owner in organization_owners:
        identity = _github_user_identity(owner)
        if identity is None:
            errors.append("organization-owner inventory contains an invalid identity")
            continue
        owner_id, login = identity
        if owner_id in owner_ids or login.casefold() in owner_logins:
            errors.append("organization-owner inventory contains a duplicate identity")
            continue
        owner_ids.add(owner_id)
        owner_logins.add(login.casefold())
    caller = _github_user_identity(authenticated_user)
    if caller is None or caller[0] not in owner_ids:
        errors.append(
            "repository audit credential is not visibly owned by an organization owner"
        )

    if (
        not isinstance(collaborators, list)
        or len(collaborators) > MAX_RELEASE_WRITER_IDENTITIES
    ):
        errors.append("collaborator inventory is invalid or unbounded")
    else:
        seen_collaborators: set[int] = set()
        for collaborator in collaborators:
            identity = _github_user_identity(collaborator)
            permissions = (
                collaborator.get("permissions")
                if isinstance(collaborator, dict)
                else None
            )
            role_name = (
                collaborator.get("role_name")
                if isinstance(collaborator, dict)
                else None
            )
            expected_permissions = {
                "read": {
                    "admin": False,
                    "maintain": False,
                    "pull": True,
                    "push": False,
                    "triage": False,
                },
                "triage": {
                    "admin": False,
                    "maintain": False,
                    "pull": True,
                    "push": False,
                    "triage": True,
                },
                "write": {
                    "admin": False,
                    "maintain": False,
                    "pull": True,
                    "push": True,
                    "triage": True,
                },
                "maintain": {
                    "admin": False,
                    "maintain": True,
                    "pull": True,
                    "push": True,
                    "triage": True,
                },
                "admin": {
                    "admin": True,
                    "maintain": True,
                    "pull": True,
                    "push": True,
                    "triage": True,
                },
            }
            if (
                identity is None
                or identity[0] in seen_collaborators
                or not isinstance(permissions, dict)
                or set(permissions)
                != {"admin", "maintain", "pull", "push", "triage"}
                or not all(isinstance(value, bool) for value in permissions.values())
                or role_name not in expected_permissions
                or permissions != expected_permissions.get(role_name)
            ):
                errors.append("collaborator inventory contains an invalid identity or role")
                continue
            seen_collaborators.add(identity[0])
            if any(permissions[key] for key in ("push", "maintain", "admin")) and (
                identity[0] not in owner_ids
            ):
                errors.append(
                    "a non-owner collaborator has effective release-write access"
                )

    if not isinstance(teams, list) or len(teams) > MAX_RELEASE_WRITER_IDENTITIES:
        errors.append("repository-team inventory is invalid or unbounded")
    else:
        seen_teams: set[int] = set()
        for team in teams:
            team_id = team.get("id") if isinstance(team, dict) else None
            slug = team.get("slug") if isinstance(team, dict) else None
            permission = team.get("permission") if isinstance(team, dict) else None
            if (
                not _is_int(team_id)
                or team_id <= 0
                or team_id in seen_teams
                or not isinstance(slug, str)
                or not slug
                or permission not in {"pull", "triage", "push", "maintain", "admin"}
            ):
                errors.append("repository-team inventory contains an invalid identity or role")
                continue
            seen_teams.add(team_id)
            if permission in {"push", "maintain", "admin"}:
                errors.append("a repository team has release-write access")

    if not isinstance(invitations, list):
        errors.append("repository invitation inventory is invalid")
    elif invitations:
        errors.append("repository has a pending collaborator invitation")

    if (
        not isinstance(deploy_keys, list)
        or len(deploy_keys) > MAX_RELEASE_WRITER_IDENTITIES
    ):
        errors.append("deploy-key inventory is invalid or unbounded")
    else:
        seen_keys: set[int] = set()
        for key in deploy_keys:
            key_id = key.get("id") if isinstance(key, dict) else None
            read_only = key.get("read_only") if isinstance(key, dict) else None
            if (
                not _is_int(key_id)
                or key_id <= 0
                or key_id in seen_keys
                or not isinstance(read_only, bool)
            ):
                errors.append("deploy-key inventory contains an invalid key")
                continue
            seen_keys.add(key_id)
            if read_only is not True:
                errors.append("repository has a write-enabled deploy key")

    if not isinstance(workflow_permissions, dict) or (
        workflow_permissions.get("default_workflow_permissions") != "read"
        or workflow_permissions.get("can_approve_pull_request_reviews") is not False
    ):
        errors.append("GitHub Actions default workflow authority is not read-only")

    if (
        not isinstance(installations, list)
        or len(installations) > MAX_RELEASE_WRITER_IDENTITIES
    ):
        errors.append("GitHub App installation inventory is invalid or unbounded")
        return errors
    seen_installations: set[int] = set()
    seen_apps: set[int] = set()
    release_installations = 0
    for installation in installations:
        installation_id = (
            installation.get("id") if isinstance(installation, dict) else None
        )
        app_id = installation.get("app_id") if isinstance(installation, dict) else None
        app_slug = installation.get("app_slug") if isinstance(installation, dict) else None
        account = installation.get("account") if isinstance(installation, dict) else None
        selection = (
            installation.get("repository_selection")
            if isinstance(installation, dict)
            else None
        )
        permissions = (
            installation.get("permissions")
            if isinstance(installation, dict)
            else None
        )
        account_id = account.get("id") if isinstance(account, dict) else None
        account_login = account.get("login") if isinstance(account, dict) else None
        valid_permissions = (
            isinstance(permissions, dict)
            and all(
                isinstance(name, str)
                and bool(name)
                and level in {"read", "write", "admin"}
                for name, level in permissions.items()
            )
        )
        if (
            not _is_int(installation_id)
            or installation_id <= 0
            or installation_id in seen_installations
            or not _is_int(app_id)
            or app_id <= 0
            or app_id in seen_apps
            or not isinstance(app_slug, str)
            or not app_slug
            or installation.get("target_type") != "Organization"
            or not _is_int(account_id)
            or account_id != repository_owner_id
            or account_login != repository_owner_login
            or selection not in {"all", "selected"}
            or not valid_permissions
        ):
            errors.append("GitHub App installation inventory contains an invalid entry")
            continue
        seen_installations.add(installation_id)
        seen_apps.add(app_id)
        assert isinstance(permissions, dict)
        privileged_permissions = {
            name
            for name, level in permissions.items()
            if level in {"write", "admin"}
        }
        if app_id == release_creator_app_id:
            release_installations += 1
            if (
                app_id == GITHUB_ACTIONS_INTEGRATION_ID
                or selection != "selected"
                or permissions.get("contents") != "write"
                or privileged_permissions != {"contents"}
            ):
                errors.append(
                    "release creator App installation is not selected-repository, Contents-write-only"
                )
        elif app_id != GITHUB_ACTIONS_INTEGRATION_ID and privileged_permissions:
            errors.append(
                "another GitHub App has write or admin permission"
            )
    if release_installations != 1:
        errors.append(
            "organization does not have exactly one visible release creator App installation"
        )
    errors.extend(
        _check_nonterminal_workflow_runs(
            workflow_runs,
            release_workflow_run_id,
            release_workflow_run_attempt,
            repository_main_sha,
        )
    )
    return errors


def _fetch(label: str, url: str, token: str, errors: list[str]) -> tuple[bool, Any]:
    try:
        return True, _github_json(url, token)
    except RepositoryControlError as exc:
        errors.append(f"{label}: {exc}")
        return False, None


def check_repository_controls(
    repository: str,
    token: str,
    *,
    scope: ControlScope,
    api_url: str = GITHUB_API_URL,
    require_immutable_releases: bool = False,
    release_creator_app_id: int | None = None,
    require_release_writer_isolation: bool = False,
    release_workflow_run_id: int | None = None,
    release_workflow_run_attempt: int | None = None,
) -> list[str]:
    if api_url != GITHUB_API_URL:
        return [f"GitHub API URL must be exactly {GITHUB_API_URL}"]
    if _REPOSITORY.fullmatch(repository) is None:
        return ["repository must have the form OWNER/REPOSITORY"]
    if any(character.isspace() or ord(character) < 0x20 for character in token):
        return ["GitHub token is malformed"]
    if scope not in {"gpu", "release"}:
        return ["control scope must be gpu or release"]
    if release_creator_app_id is not None and (
        not _is_int(release_creator_app_id) or release_creator_app_id <= 0
    ):
        return ["release creator app ID must be a positive integer"]
    if release_creator_app_id is not None and scope != "release":
        return ["release creator app ID verification requires release scope"]
    if require_release_writer_isolation and (
        scope != "release"
        or release_creator_app_id is None
        or release_workflow_run_id is None
        or release_workflow_run_attempt is None
    ):
        return [
            (
                "release-writer isolation requires release scope, a release creator "
                "app ID, and the current workflow run ID/attempt"
            )
        ]
    for value, label in (
        (release_workflow_run_id, "release workflow run ID"),
        (release_workflow_run_attempt, "release workflow run attempt"),
    ):
        if value is not None and (not _is_int(value) or value <= 0):
            return [f"{label} must be a positive integer"]
        if value is not None and scope != "release":
            return [f"{label} verification requires release scope"]
    if require_release_writer_isolation and not token:
        return ["release-writer isolation requires an organization-owner audit token"]

    root = f"{api_url}/repos/{repository}"
    repository_owner_login = repository.split("/", 1)[0]
    errors: list[str] = []
    ok, metadata = _fetch("repository metadata", root, token, errors)
    if ok:
        errors.extend(_check_repository(metadata, repository))
    ok, branch = _fetch("main branch", f"{root}/branches/main", token, errors)
    if ok:
        errors.extend(_check_main_branch(branch))
    try:
        branch_rules = _paged_list(
            f"{root}/rules/branches/main?includes_parents=true", token
        )
    except RepositoryControlError as exc:
        errors.append(f"effective main-branch rules: {exc}")
    else:
        errors.extend(_check_main_rules(branch_rules))
        main_ruleset_ids = _effective_main_ruleset_ids(branch_rules)
        main_details: dict[int, Any] = {}
        for ruleset_id in main_ruleset_ids[:MAX_EFFECTIVE_MAIN_RULESETS]:
            ok, payload = _fetch(
                f"effective main ruleset {ruleset_id}",
                f"{root}/rulesets/{ruleset_id}?includes_parents=true",
                token,
                errors,
            )
            if ok:
                main_details[ruleset_id] = payload
        errors.extend(
            _check_main_ruleset_details(
                branch_rules,
                main_details,
                require_bypass_inventory=release_creator_app_id is not None,
            )
        )
    try:
        workflows = _stable_paged_workflows(f"{root}/actions/workflows", token)
    except RepositoryControlError as exc:
        errors.append(f"repository workflows: {exc}")
    else:
        errors.extend(_check_workflow_inventory(workflows))

    environments: tuple[str, ...] = ("gpu-qualification",)
    if scope == "release":
        environments = ("gpu-qualification", "release")
    for environment in environments:
        ok, payload = _fetch(
            f"environment {environment!r}",
            f"{root}/environments/{environment}",
            token,
            errors,
        )
        if ok:
            errors.extend(_check_environment(payload, environment))
        try:
            policies = _paged_deployment_branch_policies(
                f"{root}/environments/{environment}/deployment-branch-policies",
                token,
            )
        except RepositoryControlError as exc:
            errors.append(
                f"environment {environment!r} deployment branch policies: {exc}"
            )
        else:
            errors.extend(_check_deployment_branch_policies(policies, environment))

    if scope == "release":
        try:
            summaries = _paged_list(
                f"{root}/rulesets?includes_parents=true&targets=tag", token
            )
        except RepositoryControlError as exc:
            errors.append(f"tag rulesets: {exc}")
        else:
            details: dict[int, Any] = {}
            summary_ids: list[int] = []
            for item in summaries:
                summary_id = item.get("id") if isinstance(item, dict) else None
                if _is_int(summary_id):
                    summary_ids.append(summary_id)
            for ruleset_id in summary_ids:
                ok, payload = _fetch(
                    f"tag ruleset {ruleset_id}",
                    f"{root}/rulesets/{ruleset_id}?includes_parents=true",
                    token,
                    errors,
                )
                if ok:
                    details[ruleset_id] = payload
            errors.extend(
                _check_tag_rulesets(
                    summaries,
                    details,
                    release_creator_app_id=release_creator_app_id,
                )
            )

    if require_immutable_releases:
        if not token:
            errors.append(
                "an Administration-read token is required to verify immutable releases"
            )
        else:
            ok, payload = _fetch(
                "immutable releases", f"{root}/immutable-releases", token, errors
            )
            if ok:
                errors.extend(_check_immutable_releases(payload))

    if require_release_writer_isolation:
        ok_organization, organization = _fetch(
            "organization metadata",
            f"{api_url}/orgs/{repository_owner_login}",
            token,
            errors,
        )
        ok_user, authenticated_user = _fetch(
            "authenticated audit user", f"{api_url}/user", token, errors
        )
        try:
            organization_owners = _stable_paged_list(
                f"{api_url}/orgs/{repository_owner_login}/members?role=admin",
                token,
            )
        except RepositoryControlError as exc:
            errors.append(f"organization owners: {exc}")
            organization_owners = None
        try:
            organization_invitations = _stable_paged_list(
                f"{api_url}/orgs/{repository_owner_login}/invitations",
                token,
            )
        except RepositoryControlError as exc:
            errors.append(f"organization invitations: {exc}")
            organization_invitations = None
        try:
            collaborators = _stable_paged_list(
                f"{root}/collaborators?affiliation=all", token
            )
        except RepositoryControlError as exc:
            errors.append(f"repository collaborators: {exc}")
            collaborators = None
        try:
            teams = _stable_paged_list(f"{root}/teams", token)
        except RepositoryControlError as exc:
            errors.append(f"repository teams: {exc}")
            teams = None
        try:
            invitations = _stable_paged_list(f"{root}/invitations", token)
        except RepositoryControlError as exc:
            errors.append(f"repository invitations: {exc}")
            invitations = None
        try:
            deploy_keys = _stable_paged_list(f"{root}/keys", token)
        except RepositoryControlError as exc:
            errors.append(f"repository deploy keys: {exc}")
            deploy_keys = None
        ok_workflow_permissions, workflow_permissions = _fetch(
            "default workflow permissions",
            f"{root}/actions/permissions/workflow",
            token,
            errors,
        )
        try:
            installations = _stable_paged_installations(
                f"{api_url}/orgs/{repository_owner_login}/installations",
                token,
            )
        except RepositoryControlError as exc:
            errors.append(f"organization GitHub App installations: {exc}")
            installations = None
        try:
            workflow_runs = _stable_nonterminal_workflow_runs(
                f"{root}/actions/runs",
                token,
            )
        except RepositoryControlError as exc:
            errors.append(f"nonterminal workflow runs: {exc}")
            workflow_runs = None
        repository_owner = (
            metadata.get("owner") if isinstance(metadata, dict) else None
        )
        repository_owner_id = (
            repository_owner.get("id") if isinstance(repository_owner, dict) else None
        )
        main_commit = branch.get("commit") if isinstance(branch, dict) else None
        repository_main_sha = (
            main_commit.get("sha") if isinstance(main_commit, dict) else None
        )
        assert release_creator_app_id is not None
        assert release_workflow_run_id is not None
        assert release_workflow_run_attempt is not None
        errors.extend(
            _check_release_writer_isolation(
                organization=organization if ok_organization else None,
                authenticated_user=authenticated_user if ok_user else None,
                organization_owners=organization_owners,
                organization_invitations=organization_invitations,
                collaborators=collaborators,
                teams=teams,
                invitations=invitations,
                deploy_keys=deploy_keys,
                workflow_permissions=(
                    workflow_permissions if ok_workflow_permissions else None
                ),
                installations=installations,
                workflow_runs=workflow_runs,
                repository_owner_id=(
                    repository_owner_id if _is_int(repository_owner_id) else None
                ),
                repository_owner_login=repository_owner_login,
                release_creator_app_id=release_creator_app_id,
                release_workflow_run_id=release_workflow_run_id,
                release_workflow_run_attempt=release_workflow_run_attempt,
                repository_main_sha=(
                    repository_main_sha
                    if isinstance(repository_main_sha, str)
                    else None
                ),
            )
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed unless required GitHub repository controls are active."
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in OWNER/REPOSITORY form.",
    )
    parser.add_argument("--scope", choices=("gpu", "release"), required=True)
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--github-api-url",
        default=os.environ.get("GITHUB_API_URL", GITHUB_API_URL),
    )
    parser.add_argument("--require-immutable-releases", action="store_true")
    parser.add_argument(
        "--release-creator-app-id",
        type=_positive_int,
        help="Verify the exact GitHub App bypass used to create release tags.",
    )
    parser.add_argument(
        "--require-release-writer-isolation",
        action="store_true",
        help=(
            "Require an organization-owner-visible inventory with no release writer "
            "besides trusted organization owners and the dedicated creator App."
        ),
    )
    parser.add_argument(
        "--release-workflow-run-id",
        type=_positive_int,
        help="Bind release-writer isolation to the current GitHub Actions run.",
    )
    parser.add_argument(
        "--release-workflow-run-attempt",
        type=_positive_int,
        help="Bind release-writer isolation to the current run attempt.",
    )
    args = parser.parse_args(argv)
    errors = check_repository_controls(
        args.repository,
        args.github_token,
        scope=args.scope,
        api_url=args.github_api_url,
        require_immutable_releases=args.require_immutable_releases,
        release_creator_app_id=args.release_creator_app_id,
        require_release_writer_isolation=args.require_release_writer_isolation,
        release_workflow_run_id=args.release_workflow_run_id,
        release_workflow_run_attempt=args.release_workflow_run_attempt,
    )
    if errors:
        for error in errors:
            print(f"repository control check failed: {error}", file=sys.stderr)
        return 2
    print(f"{args.scope} repository controls visible to this token passed")
    unverified = [
        "environment administrator bypass",
        "GPU runner-group repository/workflow scope",
    ]
    if args.release_creator_app_id is None:
        unverified.insert(0, "ruleset bypass actors")
    if not args.require_immutable_releases:
        unverified.append("immutable-release configuration")
    if not args.require_release_writer_isolation:
        unverified.append("release-writer isolation")
    print("administrator verification remains required for: " + ", ".join(unverified))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
