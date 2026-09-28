"""Regression: AnthropicLLM() must construct with the installed SDK (Phase 0 bug)."""
from app.llm import AnthropicLLM


def test_anthropic_llm_constructs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert AnthropicLLM()._client is not None
