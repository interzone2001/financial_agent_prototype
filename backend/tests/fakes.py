"""FakeLLM — satisfies app.llm.LLM without network. Queue canned returns per method.

    llm = FakeLLM(parse=[RouteDecision(route="market", reason="price q")], text=["hi"])
    ...
    assert llm.calls[0].method == "parse"
    assert "AAPL" in llm.calls[0].system

`parse` items may be a model instance or a callable(schema, messages) -> instance.
`run_tools` items may be a string or a callable(tools, messages) -> str; a callable
can invoke the tools itself to simulate the model searching.
Running out of queued responses raises AssertionError (a test bug, loudly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMCall:
    method: str
    role: str
    system: str
    messages: list[dict]
    extra: Any = None


class FakeLLM:
    def __init__(self, parse=None, text=None, run_tools=None):
        self._q = {"parse": list(parse or []), "text": list(text or []),
                   "run_tools": list(run_tools or [])}
        self.calls: list[LLMCall] = []

    def queue(self, method: str, *items) -> FakeLLM:
        self._q[method].extend(items)
        return self

    def _next(self, method: str):
        if not self._q[method]:
            raise AssertionError(f"FakeLLM: no queued response for {method}()")
        return self._q[method].pop(0)

    def parse(self, role, system, messages, schema):
        self.calls.append(LLMCall("parse", role, system, messages, schema))
        item = self._next("parse")
        out = item(schema, messages) if callable(item) else item
        assert isinstance(out, schema), f"FakeLLM.parse: queued {type(out).__name__}, wanted {schema.__name__}"
        return out

    def text(self, role, system, messages):
        self.calls.append(LLMCall("text", role, system, messages))
        item = self._next("text")
        return item(messages) if callable(item) else item

    def run_tools(self, role, system, messages, tools):
        self.calls.append(LLMCall("run_tools", role, system, messages, tools))
        item = self._next("run_tools")
        return item(tools, messages) if callable(item) else item
