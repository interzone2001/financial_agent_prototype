"""Regression: AnthropicLLM() must construct with the installed SDK (Phase 0 bug)."""
from app.llm import AnthropicLLM


def test_anthropic_llm_constructs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert AnthropicLLM()._client is not None


def _status_error(code):
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError("boom", response=httpx.Response(code, request=req), body=None)


class _Boom:
    def __init__(self, exc):
        self.exc = exc

    def create(self, **_):
        raise self.exc


class _Client:
    def __init__(self, exc):
        self.messages = _Boom(exc)


def test_transient_errors_become_llm_error():
    import pytest

    from app.llm import LLMError

    for code in (429, 529, 500):
        with pytest.raises(LLMError):
            AnthropicLLM(client=_Client(_status_error(code))).text("fast", "s", [])


def test_request_bugs_still_raise():
    import anthropic
    import pytest

    with pytest.raises(anthropic.APIStatusError):
        AnthropicLLM(client=_Client(_status_error(400))).text("fast", "s", [])
