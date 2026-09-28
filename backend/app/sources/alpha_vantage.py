"""Alpha Vantage source: parsers + cached client (spec §6.1).

The API key never leaves AlphaVantageClient: SourceRef URLs say apikey=REDACTED,
cache keys are "{function}:{symbol}", and error messages are scrubbed.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import httpx

from app.cache import Clock, SqliteCache, utc_now
from app.models import (
    DataSourceError,
    Overview,
    Quote,
    RateLimitedError,
    SourceRef,
    TickerNotFoundError,
)
from app.tracing import traceable

BASE_URL = "https://www.alphavantage.co/query"
PROVIDER = "alpha_vantage"
_MISSING = {"", "none", "-", "null", "n/a"}

# httpx logs every request URL at INFO, and ours carries the API key.
logging.getLogger("httpx").setLevel(logging.WARNING)


def source_url(function: str, symbol: str) -> str:
    query = urlencode({"function": function, "symbol": symbol, "apikey": "REDACTED"})
    return f"{BASE_URL}?{query}"


def _source(function: str, symbol: str, retrieved_at: datetime) -> SourceRef:
    return SourceRef(provider=PROVIDER, url=source_url(function, symbol), retrieved_at=retrieved_at)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING else text


def _opt_float(value: object) -> float | None:
    text = _clean(value)
    try:
        value = None if text is None else float(text)
    except ValueError:
        return None
    return value if value is None or math.isfinite(value) else None


def _opt_int(value: object) -> int | None:
    text = _clean(value)
    try:
        return None if text is None else int(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError):  # junk, NaN, Infinity
        return None


def parse_quote(body: dict, retrieved_at: datetime) -> Quote:
    q = body.get("Global Quote") or {}
    try:
        symbol = q["01. symbol"]
        return Quote(
            symbol=symbol,
            price=float(q["05. price"]),
            change=float(q["09. change"]),
            change_pct=float(q["10. change percent"].strip().rstrip("%")),
            volume=int(q["06. volume"]),
            latest_trading_day=date.fromisoformat(q["07. latest trading day"]),
            source=_source("GLOBAL_QUOTE", symbol, retrieved_at),
        )
    except (KeyError, ValueError, TypeError, AttributeError) as e:  # pydantic ValidationError is a ValueError
        raise DataSourceError(PROVIDER, f"malformed GLOBAL_QUOTE response: {e!r}") from None


def parse_overview(body: dict, retrieved_at: datetime) -> Overview:
    symbol, name = _clean(body.get("Symbol")), _clean(body.get("Name"))
    if symbol is None or name is None:
        raise DataSourceError(PROVIDER, "malformed OVERVIEW response: missing Symbol/Name")
    return Overview(
        symbol=symbol,
        name=name,
        sector=_clean(body.get("Sector")),
        industry=_clean(body.get("Industry")),
        market_cap=_opt_int(body.get("MarketCapitalization")),
        pe_ratio=_opt_float(body.get("PERatio")),
        week52_high=_opt_float(body.get("52WeekHigh")),
        week52_low=_opt_float(body.get("52WeekLow")),
        description=_clean(body.get("Description")),
        source=_source("OVERVIEW", symbol, retrieved_at),
    )


_PARSERS = {"GLOBAL_QUOTE": parse_quote, "OVERVIEW": parse_overview}

TTL: dict[str, timedelta] = {
    "GLOBAL_QUOTE": timedelta(seconds=60),
    "OVERVIEW": timedelta(hours=24),
}


class AlphaVantageClient:
    def __init__(
        self,
        api_key: str,
        http: httpx.Client | None = None,
        cache: SqliteCache | None = None,
        clock: Clock = utc_now,
    ):
        self._api_key = api_key
        self._http = http or httpx.Client(timeout=10.0)
        self._cache = cache
        self._clock = clock

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "REDACTED") if self._api_key else text

    @traceable(run_type="tool", name="alpha_vantage.fetch")
    def fetch(self, function: str, symbol: str) -> tuple[dict, datetime]:
        """Return (validated body, retrieved_at). Traced inputs are function+symbol only."""
        key = f"{function}:{symbol}"
        if self._cache is not None and (hit := self._cache.get(key, TTL[function])):
            return hit
        params = {"function": function, "symbol": symbol, "apikey": self._api_key}
        try:
            resp = self._http.get(BASE_URL, params=params)
        except httpx.HTTPError as e:  # message/request carry the keyed URL: drop both
            raise DataSourceError(
                PROVIDER, f"{function} {symbol}: request failed ({type(e).__name__})"
            ) from None
        if resp.status_code != 200:
            raise DataSourceError(PROVIDER, f"{function} {symbol}: HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError:
            raise DataSourceError(PROVIDER, f"{function} {symbol}: non-JSON response") from None
        self._check(function, symbol, body)
        now = self._clock()
        _PARSERS[function](body, now)  # raises on junk, so only parseable bodies are cached
        if self._cache is not None:
            self._cache.set(key, body, now)
        return body, now

    def _check(self, function: str, symbol: str, body: object) -> None:
        if not isinstance(body, dict):
            raise DataSourceError(PROVIDER, f"{function} {symbol}: unexpected JSON shape")
        for k in ("Note", "Information"):
            if k in body:
                raise RateLimitedError(PROVIDER, self._redact(f"Alpha Vantage limit: {body[k]}"))
        if "Error Message" in body:
            raise DataSourceError(
                PROVIDER, self._redact(f"{function} {symbol}: {body['Error Message']}")
            )
        if function == "GLOBAL_QUOTE" and not body.get("Global Quote"):
            raise TickerNotFoundError(symbol)
        if function == "OVERVIEW" and not body:
            raise TickerNotFoundError(symbol)

    def quote(self, symbol: str) -> Quote:
        return parse_quote(*self.fetch("GLOBAL_QUOTE", symbol))

    def overview(self, symbol: str) -> Overview:
        return parse_overview(*self.fetch("OVERVIEW", symbol))
