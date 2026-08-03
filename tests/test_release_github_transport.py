from __future__ import annotations

from email.message import Message
from typing import Any
from urllib.request import Request

import pytest
from typing_extensions import Self

import scripts.check_release_evidence as release_gate

URL = "https://api.github.com/repos/example/project/actions/runs/1"
TOKEN = "secret-token-that-must-not-leak"


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = URL,
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


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.request: Request | None = None
        self.timeout: int | None = None

    def open(self, request: Request, *, timeout: int) -> _Response:
        self.request = request
        self.timeout = timeout
        return self.response


def test_release_github_json_transport_binds_origin_headers_and_timeout(monkeypatch):
    opener = _Opener(_Response(b'{"trusted":true}'))
    monkeypatch.setattr(release_gate, "_GITHUB_JSON_OPENER", opener)

    assert release_gate._github_json(URL, TOKEN) == {"trusted": True}
    assert opener.timeout == 30
    assert opener.request is not None
    assert opener.request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert opener.request.get_header("X-github-api-version") == "2026-03-10"


def test_release_github_json_rejects_untrusted_origin_before_network(monkeypatch):
    class _ForbiddenOpener:
        def open(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise AssertionError("network must not be called")

    monkeypatch.setattr(release_gate, "_GITHUB_JSON_OPENER", _ForbiddenOpener())

    with pytest.raises(RuntimeError, match="untrusted GitHub API URL"):
        release_gate._github_json("https://attacker.invalid/steal", TOKEN)


@pytest.mark.parametrize(
    "response",
    [
        _Response(b'{"key":1,"key":2}'),
        _Response(b"\xff"),
        _Response(b"not-json"),
        _Response(b"{}", status=201),
        _Response(b"{}", content_type="text/plain"),
        _Response(b"{}", content_length="invalid"),
        _Response(b"{}", content_length="-1"),
        _Response(b"{}", url="https://attacker.invalid/redirect"),
    ],
)
def test_release_github_json_rejects_malformed_or_redirected_responses(
    monkeypatch,
    response,
):
    monkeypatch.setattr(release_gate, "_GITHUB_JSON_OPENER", _Opener(response))

    with pytest.raises(RuntimeError) as failure:
        release_gate._github_json(URL, TOKEN)

    assert TOKEN not in str(failure.value)


def test_release_github_json_rejects_oversized_body(monkeypatch):
    payload = b" " * (release_gate.MAX_GITHUB_RESPONSE_BYTES + 1)
    response = _Response(
        payload,
        content_length=str(release_gate.MAX_GITHUB_RESPONSE_BYTES),
    )
    monkeypatch.setattr(release_gate, "_GITHUB_JSON_OPENER", _Opener(response))

    with pytest.raises(RuntimeError, match="safety limit"):
        release_gate._github_json(URL, TOKEN)
