from __future__ import annotations

import random
from typing import Optional


class MarketDataUnavailable(Exception):
    pass


def fetch_ohlcv(symbol: str, days: int = 90) -> dict:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise MarketDataUnavailable(
            f"yfinance is not installed: {exc}"
        ) from exc

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=f"{days}d")
    except Exception as exc:
        raise MarketDataUnavailable(
            f"Failed to fetch data for {symbol!r}: {exc}"
        ) from exc

    if history is None or len(history) == 0:
        raise MarketDataUnavailable(f"No data returned for {symbol!r}")

    try:
        closes = [float(x) for x in history["Close"].tolist()]
        highs = [float(x) for x in history["High"].tolist()]
        lows = [float(x) for x in history["Low"].tolist()]
        volumes = [float(x) for x in history["Volume"].tolist()]
    except Exception as exc:
        raise MarketDataUnavailable(
            f"Failed to parse data for {symbol!r}: {exc}"
        ) from exc

    if not closes:
        raise MarketDataUnavailable(f"Empty close series for {symbol!r}")

    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
    }


def _synthetic_ohlcv(
    symbol: str, days: int, seed_config: Optional[dict]
) -> dict:
    cfg = seed_config or {"start_price": 100.0, "base_volume": 1_000_000.0}
    start_price = float(cfg.get("start_price", 100.0))
    base_volume = float(cfg.get("base_volume", 1_000_000.0))

    rng = random.Random(hash(symbol) & 0xFFFFFFFF)

    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []

    price = start_price
    for _ in range(days):
        daily_return = rng.gauss(0.0005, 0.015)
        open_price = price
        close_price = max(1.0, price * (1.0 + daily_return))
        intraday_range = abs(rng.gauss(0.0, 0.012)) * close_price + 0.001 * close_price
        high = max(open_price, close_price) + intraday_range * rng.random()
        low = min(open_price, close_price) - intraday_range * rng.random()
        low = max(0.5, low)
        vol_noise = rng.gauss(1.0, 0.25)
        volume = max(1.0, base_volume * vol_noise)

        closes.append(round(close_price, 2))
        highs.append(round(high, 2))
        lows.append(round(low, 2))
        volumes.append(round(volume, 0))

        price = close_price

    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
    }


def fetch_or_synthetic(
    symbol: str,
    days: int = 90,
    seed_config: dict | None = None,
) -> tuple[dict, str]:
    try:
        data = fetch_ohlcv(symbol, days=days)
        return data, "yfinance"
    except MarketDataUnavailable:
        return _synthetic_ohlcv(symbol, days, seed_config), "synthetic"
