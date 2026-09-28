from langsmith.run_helpers import is_traceable_function

from app.agents.filings import CHUNK_ID_RE, answer_filings_question, summarize_filings
from app.contracts import fixture_retriever
from app.models import ChatTurn
from tests.fakes import FakeLLM

K_RISK = "0000320193-25-000079:item-1a:0"
NEVER_RETRIEVED = "0000320193-26-000018:8k-items:1"
FABRICATED = "0000320193-99-999999:item-1a:0"


def _searching_model(query, form_type, answer):
    """FakeLLM run_tools callable: invokes the tool the way the SDK tool runner does."""

    def run(tools, messages):
        (tool,) = tools
        assert tool.name == "search_filings"
        result = tool.call({"query": query, "form_type": form_type})
        assert K_RISK in result  # the model 'sees' ids in the tool output
        return answer

    return run


def test_regex_matches_real_chunk_id_shapes():
    text = "a [0000320193-26-000020:part1-item-2:2] b [0000320193-26-000018:8k-items:0, x]"
    assert CHUNK_ID_RE.findall(text) == [
        "0000320193-26-000020:part1-item-2:2",
        "0000320193-26-000018:8k-items:0",
    ]


def test_answer_cites_only_retrieved_ids():
    answer_text = (
        f"Apple flags global economic conditions [{K_RISK}]. "
        f"Also [{NEVER_RETRIEVED}] and [{FABRICATED}]. Again [{K_RISK}]."
    )
    llm = FakeLLM(run_tools=[_searching_model("risk factors economic", "10-K", answer_text)])
    ans = answer_filings_question("AAPL", "What are the main risks?", [], fixture_retriever, llm)
    assert ans.text == answer_text
    assert [c.chunk_id for c in ans.citations] == [K_RISK]
    assert ans.citations[0].section == "Item 1A. Risk Factors"


def test_tool_passes_form_filter_and_ignores_unknown_form():
    calls = []

    def spy(ticker, query, k=6, form_type=None):
        calls.append(form_type)
        return fixture_retriever(ticker, query, k, form_type)

    def run(tools, messages):
        tools[0].call({"query": "results", "form_type": "8-K"})
        tools[0].call({"query": "risk", "form_type": "10K"})
        tools[0].call({"query": "risk"})
        return "done"

    answer_filings_question("AAPL", "q", [], spy, FakeLLM(run_tools=[run]))
    assert calls == ["8-K", None, None]


def _turns(n):
    return [ChatTurn(role=("user", "assistant")[i % 2], content=f"turn {i}") for i in range(n)]


def test_history_truncated_to_last_six_turns_then_question():
    llm = FakeLLM(run_tools=["ok", "ok"])
    answer_filings_question("AAPL", "latest q?", _turns(10), fixture_retriever, llm)
    msgs = llm.calls[0].messages
    assert [m["content"] for m in msgs] == [f"turn {i}" for i in range(4, 10)] + ["latest q?"]
    assert msgs[-1] == {"role": "user", "content": "latest q?"}
    assert "AAPL" in llm.calls[0].system
    # 7 turns -> last 6 would open with assistant 'turn 1'; it must be dropped
    answer_filings_question("AAPL", "q", _turns(7), fixture_retriever, llm)
    assert llm.calls[1].messages[0] == {"role": "user", "content": "turn 2"}


def test_public_functions_are_traced():
    assert is_traceable_function(summarize_filings)
    assert is_traceable_function(answer_filings_question)
