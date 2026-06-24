"""Alpaca historical bars fetcher tests (mocked httpx, no network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data.alpaca_bars import AlpacaBarsError, AlpacaBarsFetcher


def _resp(status_code: int, payload):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = payload
    return r


def _bars(n: int) -> list[dict]:
    return [
        {
            "t": f"2026-06-10T13:3{i % 10}:00Z",
            "o": 100.0 + i,
            "h": 101.0 + i,
            "l": 99.0 + i,
            "c": 100.5 + i,
            "v": 1000 + i,
        }
        for i in range(n)
    ]


def test_fetch_bars_single_page() -> None:
    http = MagicMock()
    http.get.return_value = _resp(200, {"bars": _bars(5), "next_page_token": None})
    fetcher = AlpacaBarsFetcher(api_key="k", secret_key="s", http_client=http)
    bars = fetcher.fetch_bars("SPY")
    assert len(bars) == 5


def test_fetch_bars_paginates() -> None:
    http = MagicMock()
    http.get.side_effect = [
        _resp(200, {"bars": _bars(3), "next_page_token": "tok"}),
        _resp(200, {"bars": _bars(2), "next_page_token": None}),
    ]
    fetcher = AlpacaBarsFetcher(api_key="k", secret_key="s", http_client=http)
    bars = fetcher.fetch_bars("SPY")
    assert len(bars) == 5


def test_fetch_bars_raises_on_error_status() -> None:
    http = MagicMock()
    http.get.return_value = _resp(403, {})
    fetcher = AlpacaBarsFetcher(api_key="k", secret_key="s", http_client=http)
    with pytest.raises(AlpacaBarsError):
        fetcher.fetch_bars("SPY")


def test_fetch_bars_raises_when_empty() -> None:
    http = MagicMock()
    http.get.return_value = _resp(200, {"bars": [], "next_page_token": None})
    fetcher = AlpacaBarsFetcher(api_key="k", secret_key="s", http_client=http)
    with pytest.raises(AlpacaBarsError):
        fetcher.fetch_bars("SPY")


def test_missing_credentials_raises() -> None:
    with pytest.raises(AlpacaBarsError):
        AlpacaBarsFetcher(api_key=None, secret_key="s")


def test_write_csv_creates_loadable_file(tmp_path: Path) -> None:
    http = MagicMock()
    http.get.return_value = _resp(200, {"bars": _bars(10), "next_page_token": None})
    fetcher = AlpacaBarsFetcher(api_key="k", secret_key="s", http_client=http)
    out = fetcher.write_csv("SPY", historical_dir=tmp_path, timeframe="1m")
    assert out.exists()
    assert out.name == "1m.csv"
    assert out.parent.name == "SPY"

    # The written file must be loadable by the repo's CSV loader.
    from data.csv_loader import load_ohlcv_csv

    df = load_ohlcv_csv(out, "SPY", "1m", "America/New_York")
    assert len(df) > 0
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
