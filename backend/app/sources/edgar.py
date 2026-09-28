"""SEC EDGAR client: ticker -> CIK, submissions, filing selection, documents (spec §6.2)."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import lru_cache

import httpx

from app import config
from app.models import DataSourceError, RateLimitedError, TickerNotFoundError
from app.tracing import traceable

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _size(out) -> dict:  # keep multi-MB payloads out of LangSmith
    return {"chars": len(out)}


class EdgarClient:
    """Thin httpx wrapper: SEC User-Agent, <=10 req/s throttle, errors -> DataSourceError."""

    def __init__(self, user_agent: str | None = None, http: httpx.Client | None = None,
                 min_interval: float = 0.11, sleep: Callable[[float], None] = time.sleep):
        self.http = http or httpx.Client(timeout=30.0, follow_redirects=True)
        self.headers = {"User-Agent": user_agent or config.sec_user_agent(),
                        "Accept-Encoding": "gzip, deflate"}
        self.min_interval = min_interval
        self._sleep = sleep
        self._last = float("-inf")
        self._tickers: dict[str, tuple[str, str]] | None = None

    def _get(self, url: str) -> httpx.Response:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        try:
            resp = self.http.get(url, headers=self.headers)
        except httpx.HTTPError as e:
            raise DataSourceError("sec_edgar", f"EDGAR request failed: {url}: {e}") from e
        if resp.status_code == 429:
            raise RateLimitedError("sec_edgar", f"EDGAR rate limited: {url}")
        if resp.status_code != 200:
            raise DataSourceError("sec_edgar", f"EDGAR HTTP {resp.status_code}: {url}")
        return resp

    @traceable(run_type="tool", name="edgar.get_company_tickers", process_outputs=_size)
    def get_company_tickers(self) -> dict[str, tuple[str, str]]:
        """TICKER -> (10-digit cik, title). Fetched once per client (process lifetime)."""
        if self._tickers is None:
            raw = self._get(TICKERS_URL).json()
            self._tickers = {row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"])
                             for row in raw.values()}
        return self._tickers

    @traceable(run_type="tool", name="edgar.get_submissions", process_outputs=_size)
    def get_submissions(self, cik: str) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    @traceable(run_type="tool", name="edgar.get_document", process_outputs=_size)
    def get_document(self, url: str) -> str:
        return self._get(url).text


@lru_cache(maxsize=1)
def default_client() -> EdgarClient:
    return EdgarClient()


def normalize_ticker(ticker: str) -> str:
    """SEC uses '-' for share classes: 'brk.b ' -> 'BRK-B'."""
    return ticker.strip().upper().replace(".", "-")


@traceable(run_type="tool", name="edgar.resolve_company")
def resolve_company(ticker: str, *, client: EdgarClient | None = None) -> tuple[str, str]:
    """(10-digit cik, company title). Raises TickerNotFoundError."""
    hit = (client or default_client()).get_company_tickers().get(normalize_ticker(ticker))
    if hit is None:
        raise TickerNotFoundError(ticker)
    return hit
