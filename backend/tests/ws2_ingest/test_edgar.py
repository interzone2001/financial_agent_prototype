import httpx
import pytest

from app.models import DataSourceError, RateLimitedError, TickerNotFoundError
from app.sources.edgar import EdgarClient, resolve_company
from tests.ws2_ingest.helpers import UA


def test_resolve_known_ticker(edgar_api, client):
    assert resolve_company("AAPL", client=client) == ("0000320193", "Apple Inc.")


def test_resolve_normalises_case_space_and_share_class_dot(edgar_api, client):
    assert resolve_company(" brk.b ", client=client) == ("0001067983", "BERKSHIRE HATHAWAY INC")


def test_resolve_unknown_raises(edgar_api, client):
    with pytest.raises(TickerNotFoundError):
        resolve_company("ZZZZZ", client=client)


def test_ticker_map_fetched_once_per_client(edgar_api, client):
    resolve_company("AAPL", client=client)
    resolve_company("MSFT", client=client)
    assert edgar_api.routes["tickers"].call_count == 1


def test_requests_carry_sec_user_agent(edgar_api, client):
    client.get_submissions("0000320193")
    req = edgar_api.routes["submissions"].calls.last.request
    assert req.headers["User-Agent"] == UA and "gzip" in req.headers["Accept-Encoding"]


@pytest.mark.parametrize(("status", "exc"), [(503, DataSourceError), (429, RateLimitedError)])
def test_http_errors_map_to_datasource_errors(edgar_api, client, status, exc):
    edgar_api.routes["submissions"].respond(status)
    with pytest.raises(exc) as info:
        client.get_submissions("0000320193")
    assert info.value.provider == "sec_edgar"


def test_throttle_spaces_consecutive_requests(edgar_api):
    sleeps: list[float] = []
    c = EdgarClient(user_agent=UA, http=httpx.Client(), min_interval=0.5, sleep=sleeps.append)
    c.get_submissions("0000320193")
    c.get_submissions("0000320193")
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 0.5
