"""LLM wrapper — the only module that imports `anthropic`. See spec §3.3.

Agents take an `LLM` so tests can inject `tests.fakes.FakeLLM`. Every method is
`@traceable`, so each call shows up in LangSmith when LANGSMITH_TRACING=true (the
client is also `wrap_anthropic`-ed for token usage on the calls it can see).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from app.tracing import traceable

ModelRole = Literal["agent", "fast"]
T = TypeVar("T", bound=BaseModel)

MODELS: dict[ModelRole, str] = {
    "agent": "claude-sonnet-5",
    "fast": "claude-haiku-4-5",
}
MAX_TOKENS: dict[ModelRole, int] = {"agent": 16000, "fast": 1024}


class LLM(Protocol):
    def parse(
        self, role: ModelRole, system: str, messages: list[dict], schema: type[T]
    ) -> T:
        """Structured output: returns a validated instance of `schema`."""
        ...

    def text(self, role: ModelRole, system: str, messages: list[dict]) -> str:
        """Plain completion: returns the concatenated text blocks."""
        ...

    def run_tools(
        self, role: ModelRole, system: str, messages: list[dict], tools: list[Any]
    ) -> str:
        """Agentic loop over `@beta_tool` functions; returns the final text."""
        ...


class LLMError(Exception):
    """The model returned no usable output (refusal, max_tokens, unparsable)."""


class AnthropicLLM:
    def __init__(self, client: Any | None = None):
        if client is None:
            import anthropic
            from langsmith.wrappers import wrap_anthropic

            client = wrap_anthropic(anthropic.Anthropic())
        self._client = client

    @traceable(run_type="llm", name="llm.parse")
    def parse(self, role: ModelRole, system: str, messages: list[dict], schema: type[T]) -> T:
        resp = self._client.messages.parse(
            model=MODELS[role],
            max_tokens=MAX_TOKENS[role],
            system=system,
            messages=messages,
            output_format=schema,
        )
        if resp.stop_reason in ("refusal", "max_tokens") or resp.parsed_output is None:
            raise LLMError(f"parse failed: stop_reason={resp.stop_reason}")
        return resp.parsed_output

    @traceable(run_type="llm", name="llm.text")
    def text(self, role: ModelRole, system: str, messages: list[dict]) -> str:
        resp = self._client.messages.create(
            model=MODELS[role],
            max_tokens=MAX_TOKENS[role],
            system=system,
            messages=messages,
        )
        if resp.stop_reason == "refusal":
            raise LLMError("model refused")
        return "".join(b.text for b in resp.content if b.type == "text")

    @traceable(run_type="chain", name="llm.run_tools")
    def run_tools(
        self, role: ModelRole, system: str, messages: list[dict], tools: list[Any]
    ) -> str:
        runner = self._client.beta.messages.tool_runner(
            model=MODELS[role],
            max_tokens=MAX_TOKENS[role],
            system=system,
            messages=messages,
            tools=tools,
        )
        final = None
        for message in runner:
            final = message
        if final is None or final.stop_reason == "refusal":
            raise LLMError("tool run produced no answer")
        return "".join(b.text for b in final.content if b.type == "text")
