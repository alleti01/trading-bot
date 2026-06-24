"""Alpaca historical bars fetcher → per-symbol OHLCV CSVs.

Read-only market data. Pulls 1-minute bars from Alpaca's market-data
API and writes them in the repo convention
``data/historical/<SYMBOL>/<timeframe>.csv`` so the signal engine,
backtester, and trainer can consume real equity data.

This module never places orders. It uses the same paper API keys as the
broker adapter but only hits the data endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger

_log = get_logger("data.alpaca_bars")

_DATA_URL = "https://data.alpaca.markets/v2"
_TIMEFRAME_MAP = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}


class AlpacaBarsError(RuntimeError):
    """Raised when bars cannot be fetched."""


class AlpacaBarsFetcher:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        secret_key: Optional[str],
        data_url: str = _DATA_URL,
        timeout_seconds: float = 30.0,
        http_client: Any = None,
    ) -> None:
        if not api_key or not secret_key:
            raise AlpacaBarsError("Missing Alpaca API key/secret for bars fetch.")
        self._api_key = api_key
        self._secret_key = secret_key
        self.data_url = data_url.rstrip("/")
        self._timeout = timeout_seconds
        self._http = http_client
        self.log = _log

    def _client(self):  # noqa: ANN202
        if self._http is not None:
            return self._http
        import httpx

        self._http = httpx.Client(timeout=self._timeout)
        return self._http

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }

    def fetch_bars(
        self,
        symbol: str,
        *,
        timeframe: str = "1m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Return a list of bar dicts (t, o, h, l, c, v) for ``symbol``."""
        import httpx

        sym = symbol.upper()
        tf = _TIMEFRAME_MAP.get(timeframe, "1Min")
        end = end or datetime.now(tz=timezone.utc)
        start = start or (end - timedelta(days=30))
        url = f"{self.data_url}/stocks/{sym}/bars"
        params: dict[str, Any] = {
            "timeframe": tf,
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "limit": limit,
            "adjustment": "raw",
            "feed": "iex",
        }
        bars: list[dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                resp = self._client().get(url, headers=self._headers(), params=params)
            except httpx.HTTPError as e:
                raise AlpacaBarsError(f"Alpaca bars network error: {e}") from e
            if resp.status_code >= 400:
                raise AlpacaBarsError(
                    f"Alpaca bars failed status={resp.status_code} for {sym}"
                )
            data = resp.json()
            page = data.get("bars") or []
            bars.extend(page)
            page_token = data.get("next_page_token")
            if not page_token:
                break
        if not bars:
            raise AlpacaBarsError(f"No bars returned for {sym}")
        return bars

    def write_csv(
        self,
        symbol: str,
        *,
        historical_dir: Path,
        timeframe: str = "1m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Path:
        """Fetch bars and write them to ``<historical_dir>/<SYM>/<tf>.csv``."""
        import pandas as pd

        sym = symbol.upper()
        bars = self.fetch_bars(sym, timeframe=timeframe, start=start, end=end)
        rows = [
            {
                "timestamp": b["t"],
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
            }
            for b in bars
        ]
        df = pd.DataFrame(rows)
        out_dir = Path(historical_dir) / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{timeframe}.csv"
        df.to_csv(out_path, index=False)
        self.log.info(
            "alpaca_bars.written",
            symbol=sym,
            rows=len(df),
            path=str(out_path),
        )
        return out_path


def download_symbols(
    settings,
    *,
    symbols: Optional[list[str]] = None,
    timeframe: str = "1m",
    days: int = 30,
    http_client: Any = None,
) -> dict[str, str]:
    """Download bars for each symbol into the historical data dir.

    Returns a map of symbol → result ("ok" or an error string). One
    symbol failing never aborts the others.
    """
    api_key = (
        settings.ALPACA_API_KEY.get_secret_value() if settings.ALPACA_API_KEY else None
    )
    secret = (
        settings.ALPACA_SECRET_KEY.get_secret_value()
        if settings.ALPACA_SECRET_KEY
        else None
    )
    fetcher = AlpacaBarsFetcher(
        api_key=api_key,
        secret_key=secret,
        timeout_seconds=float(settings.BROKER_REQUEST_TIMEOUT_SECONDS),
        http_client=http_client,
    )
    syms = symbols or list(settings.ENABLED_SYMBOLS)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    results: dict[str, str] = {}
    for sym in syms:
        try:
            fetcher.write_csv(
                sym,
                historical_dir=Path(settings.HISTORICAL_DATA_DIR),
                timeframe=timeframe,
                start=start,
                end=end,
            )
            results[sym.upper()] = "ok"
        except AlpacaBarsError as e:
            _log.warning("alpaca_bars.symbol_failed", symbol=sym, error=str(e))
            results[sym.upper()] = str(e)
    return results
