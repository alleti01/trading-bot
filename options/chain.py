"""Options chain providers (Alpaca + deterministic synthetic for tests)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Optional

from app.logging_config import get_logger
from options.contract import OptionContract, OptionType, build_occ_symbol, parse_occ_symbol
from options.greeks import black_scholes_greeks


class OptionsChainError(RuntimeError):
    """Raised when a chain cannot be fetched."""


class BaseChainProvider(ABC):
    """Fetches the option chain for an underlying."""

    @abstractmethod
    def get_chain(
        self,
        underlying: str,
        *,
        expiry: Optional[date] = None,
        option_type: Optional[OptionType] = None,
    ) -> list[OptionContract]:
        ...

    @abstractmethod
    def list_expirations(self, underlying: str) -> list[date]:
        ...


class SyntheticChainProvider(BaseChainProvider):
    """Deterministic synthetic chain for backtests + tests (no network).

    Builds strikes around a spot price at fixed intervals across a few
    weekly/monthly expirations and prices them with Black-Scholes so the
    selection + risk layers have realistic greeks to work with.
    """

    def __init__(
        self,
        *,
        spot_by_symbol: Optional[dict[str, float]] = None,
        strike_step: float = 5.0,
        n_strikes: int = 20,
        volatility: float = 0.20,
        expirations_days: Optional[list[int]] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.spot_by_symbol = {k.upper(): v for k, v in (spot_by_symbol or {}).items()}
        self.strike_step = strike_step
        self.n_strikes = n_strikes
        self.volatility = volatility
        self.expirations_days = expirations_days or [7, 14, 30, 60]
        self._now = now or datetime.now()
        self.log = get_logger("options.chain.synthetic")

    def _spot(self, underlying: str) -> float:
        return self.spot_by_symbol.get(underlying.upper(), 100.0)

    def list_expirations(self, underlying: str) -> list[date]:
        base = self._now.date()
        return [base + timedelta(days=d) for d in self.expirations_days]

    def get_chain(
        self,
        underlying: str,
        *,
        expiry: Optional[date] = None,
        option_type: Optional[OptionType] = None,
    ) -> list[OptionContract]:
        sym = underlying.upper()
        spot = self._spot(sym)
        atm = round(spot / self.strike_step) * self.strike_step
        strikes = [
            atm + (i - self.n_strikes // 2) * self.strike_step
            for i in range(self.n_strikes)
        ]
        strikes = [s for s in strikes if s > 0]
        expiries = [expiry] if expiry else self.list_expirations(sym)
        types: list[OptionType] = [option_type] if option_type else ["call", "put"]

        contracts: list[OptionContract] = []
        for exp in expiries:
            dte = max((exp - self._now.date()).days, 0)
            for strike in strikes:
                for opt_type in types:
                    contracts.append(
                        self._price_contract(sym, exp, strike, opt_type, spot, dte)
                    )
        return contracts

    def _price_contract(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: OptionType,
        spot: float,
        dte: int,
    ) -> OptionContract:
        greeks = black_scholes_greeks(
            spot=spot,
            strike=strike,
            days_to_expiry=max(dte, 1),
            volatility=self.volatility,
            option_type=option_type,
        )
        price = greeks.price if greeks else max(0.01, spot - strike)
        spread = max(0.02, price * 0.02)
        return OptionContract(
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            occ_symbol=build_occ_symbol(underlying, expiry, strike, option_type),
            bid=round(max(price - spread / 2, 0.01), 2),
            ask=round(price + spread / 2, 2),
            last=round(price, 2),
            delta=greeks.delta if greeks else None,
            gamma=greeks.gamma if greeks else None,
            theta=greeks.theta if greeks else None,
            vega=greeks.vega if greeks else None,
            implied_volatility=self.volatility,
        )


class AlpacaChainProvider(BaseChainProvider):
    """Fetches the live options chain from Alpaca's market-data API.

    Read-only — placing orders goes through the options execution
    adapter, not here. Falls back gracefully (raises OptionsChainError)
    when the network/credentials are unavailable.
    """

    DEFAULT_DATA_URL = "https://data.alpaca.markets/v1beta1"

    def __init__(
        self,
        *,
        api_key: Optional[str],
        secret_key: Optional[str],
        data_url: Optional[str] = None,
        timeout_seconds: float = 15.0,
        http_client: Any = None,
    ) -> None:
        self.log = get_logger("options.chain.alpaca")
        self._api_key = api_key
        self._secret_key = secret_key
        self.data_url = (data_url or self.DEFAULT_DATA_URL).rstrip("/")
        self._timeout = timeout_seconds
        self._http = http_client

    def _client(self):  # noqa: ANN202
        if self._http is not None:
            return self._http
        import httpx

        self._http = httpx.Client(timeout=self._timeout)
        return self._http

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key or "",
            "APCA-API-SECRET-KEY": self._secret_key or "",
            "Accept": "application/json",
        }

    def list_expirations(self, underlying: str) -> list[date]:
        chain = self.get_chain(underlying)
        return sorted({c.expiry for c in chain})

    def get_chain(
        self,
        underlying: str,
        *,
        expiry: Optional[date] = None,
        option_type: Optional[OptionType] = None,
    ) -> list[OptionContract]:
        import httpx

        sym = underlying.upper()
        url = f"{self.data_url}/options/snapshots/{sym}"
        params: dict[str, Any] = {"feed": "indicative", "limit": 1000}
        try:
            resp = self._client().get(url, headers=self._headers(), params=params)
        except httpx.HTTPError as e:
            raise OptionsChainError(f"Alpaca chain network error: {e}") from e
        if resp.status_code >= 400:
            raise OptionsChainError(
                f"Alpaca chain failed status={resp.status_code}"
            )
        data = resp.json()
        snapshots = data.get("snapshots", {}) if isinstance(data, dict) else {}
        contracts: list[OptionContract] = []
        for occ, snap in snapshots.items():
            try:
                base = parse_occ_symbol(occ)
            except ValueError:
                continue
            if expiry and base.expiry != expiry:
                continue
            if option_type and base.option_type != option_type:
                continue
            quote = snap.get("latestQuote", {}) if isinstance(snap, dict) else {}
            greeks = snap.get("greeks", {}) if isinstance(snap, dict) else {}
            contracts.append(
                OptionContract(
                    underlying=sym,
                    expiry=base.expiry,
                    strike=base.strike,
                    option_type=base.option_type,
                    occ_symbol=occ,
                    bid=quote.get("bp"),
                    ask=quote.get("ap"),
                    last=(snap.get("latestTrade", {}) or {}).get("p"),
                    delta=greeks.get("delta"),
                    gamma=greeks.get("gamma"),
                    theta=greeks.get("theta"),
                    vega=greeks.get("vega"),
                    implied_volatility=snap.get("impliedVolatility"),
                    raw=snap if isinstance(snap, dict) else {},
                )
            )
        if not contracts:
            raise OptionsChainError(f"No option contracts returned for {sym}")
        return contracts
