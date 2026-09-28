"""Phase 2: router sees the last two turns so pronoun follow-ups stay on-topic."""
from app.prompts.router import router_messages


def test_router_includes_recent_turns():
    recent = [{"role": "user", "content": "What are AAPL's risk factors?"},
              {"role": "assistant", "content": "Supply chain concentration in Asia..."}]
    content = router_messages("AAPL", "Apple Inc.", "what about their debt?", recent)[0]["content"]
    assert "Supply chain" in content and content.endswith("<question>what about their debt?</question>")


def test_router_without_history_unchanged_shape():
    content = router_messages("AAPL", "Apple Inc.", "price?")[0]["content"]
    assert "Recent conversation" not in content


def test_router_prompt_keeps_conversation_followups_on_topic():
    from app.prompts.router import ROUTER_SYSTEM

    assert "summarize what we discussed" in ROUTER_SYSTEM and "on-topic" in ROUTER_SYSTEM
