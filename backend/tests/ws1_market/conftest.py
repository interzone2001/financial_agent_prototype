import httpx
import pytest

from tests.ws1_market.helpers import KEY, FakeAV, FakeClock


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_client(tmp_path, clock):
    """make_client(GLOBAL_QUOTE=body, OVERVIEW=body) -> (AlphaVantageClient, FakeAV)."""
    from app.cache import SqliteCache
    from app.sources.alpha_vantage import AlphaVantageClient

    def _make(**bodies):
        av = FakeAV(**bodies)
        client = AlphaVantageClient(
            api_key=KEY,
            http=httpx.Client(transport=httpx.MockTransport(av)),
            cache=SqliteCache(tmp_path / "cache.sqlite", clock=clock),
            clock=clock,
        )
        return client, av

    return _make
