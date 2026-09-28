import json

import httpx
import pytest
import respx

from tests.ws2_ingest.helpers import (
    SUBMISSIONS_URL,
    TEN_Q_HTML,
    TICKERS_URL,
    UA,
    URL_8K,
    URL_10K,
    URL_10Q,
    HashEmbedding,
    fixture_text,
)


@pytest.fixture
def edgar_api():
    """All EDGAR endpoints mocked from fixtures. Unmocked URLs fail loudly."""
    with respx.mock(assert_all_called=False) as mock:
        tickers = json.loads(fixture_text("sec_company_tickers.json"))
        mock.get(TICKERS_URL, name="tickers").respond(json=tickers)
        subs = json.loads(fixture_text("sec_submissions_AAPL.json"))
        mock.get(SUBMISSIONS_URL, name="submissions").respond(json=subs)
        mock.get(URL_10K, name="doc_10k").respond(text=fixture_text("sec_10k_AAPL_excerpt.html"))
        mock.get(URL_10Q, name="doc_10q").respond(text=TEN_Q_HTML)
        mock.get(URL_8K, name="doc_8k").respond(text=fixture_text("sec_8k_AAPL.html"))
        yield mock


@pytest.fixture
def client():
    from app.sources.edgar import EdgarClient

    with httpx.Client() as http:
        yield EdgarClient(user_agent=UA, http=http, min_interval=0)


@pytest.fixture
def collection(tmp_data_dir):
    from app import vectorstore

    return vectorstore.get_collection(embedding_function=HashEmbedding())
