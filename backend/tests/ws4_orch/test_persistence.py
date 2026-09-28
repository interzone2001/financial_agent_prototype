from fastapi.testclient import TestClient

from app.api import create_app
from app.contracts import stub_answer_filings_question
from app.graph import get_report
from tests.ws4_orch.helpers import chat_llm, deps_with


def test_report_and_history_survive_restart(tmp_data_dir):
    c1 = TestClient(create_app(deps_with(chat_llm("filings"))))  # default SqliteSaver
    assert c1.post("/api/report", json={"thread_id": "tp", "ticker": "AAPL"}).status_code == 200
    assert c1.post("/api/chat", json={"thread_id": "tp", "message": "Risks?"}).status_code == 200
    assert (tmp_data_dir / "checkpoints.sqlite").exists()

    seen = []

    def spy(ticker, question, history, retriever, llm):
        seen.append(history)
        return stub_answer_filings_question(ticker, question, history, retriever, llm)

    app2 = create_app(deps_with(chat_llm("filings"), answer_filings_question=spy))  # "restart"
    assert get_report(app2.state.graph, "tp").ticker == "AAPL"
    r = TestClient(app2).post("/api/chat", json={"thread_id": "tp", "message": "More on China?"})
    assert r.status_code == 200 and r.json()["route"] == "filings"
    assert seen[0][0].content == "Risks?"  # prior turn loaded from disk
