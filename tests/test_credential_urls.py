"""assert_no_credential_urls: no URL the model was sent carries a key, token, or signature."""

from __future__ import annotations

import pytest

from agentverify import CredentialURLError, assert_no_credential_urls
from agentverify.cassette.adapters.base import NormalizedRequest

PRESIGNED = "https://download-ring.amazon.dev/v1/download?X-Amz-Date=20261105&X-Amz-Expires=900&security_token=SECRET1&X-Amz-Signature=SECRET2"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl"


def _request(*tool_results, tools=None):
    messages = [{"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "history", "arguments": "{}"}}]}]
    messages += [{"role": "tool", "tool_call_id": "c1", "content": r} for r in tool_results]
    return NormalizedRequest(model="m", messages=messages, tools=tools)


def test_clean_requests_pass():
    assert_no_credential_urls(_request('{"events": [{"at": "now", "link": "https://example.com/events?page=2"}]}'))


def test_presigned_and_token_urls_fail_without_showing_values():
    request = _request(f'{{"visualizations": "{PRESIGNED}"}}', f"see https://api.example.com/v1/{JWT}/x and https://h.example/cb?Access_Token=abc")
    with pytest.raises(CredentialURLError) as error:
        assert_no_credential_urls(request)
    message = str(error.value)
    assert message.startswith("3 URLs with a credential sent to the model")
    assert "request[0].message[1](tool).tool_result[0](history).visualizations: download-ring.amazon.dev (X-Amz-Signature, security_token)" in message
    assert "api.example.com (JWT)" in message and "h.example (Access_Token)" in message
    assert "SECRET1" not in message and "SECRET2" not in message and JWT not in message and "abc" not in message


def test_tool_definitions_are_searched_and_options_apply():
    tools = [{"name": "f", "description": "Docs at https://docs.example/?key=public-value"}]
    with pytest.raises(CredentialURLError, match=r"tool\[f\]\.description: docs\.example \(key\)"):
        assert_no_credential_urls(_request(tools=tools))
    assert_no_credential_urls(_request(tools=tools), ignore=["KEY"])
    with pytest.raises(CredentialURLError, match=r"pattern '\[\?&\]code='"):
        assert_no_credential_urls(_request("https://app.example/callback?code=xyz"), extra_patterns=[r"[?&]code="])
    assert_no_credential_urls(_request("https://app.example/callback?code=xyz"))  # code is not in the defaults
    with pytest.raises(CredentialURLError, match=r"\?"):
        assert_no_credential_urls(_request("http://?token=1"))  # no host
