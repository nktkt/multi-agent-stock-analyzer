"""Technical analyst agent for stock market pattern analysis."""

from __future__ import annotations

import json
from typing import List, Optional

import anthropic


MODEL = "claude-opus-4-7"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a senior technical analyst with deep expertise in equity market "
    "microstructure, chart patterns, and quantitative indicators. You are given "
    "a snapshot of computed technical indicators (SMA, EMA, RSI, MACD, Bollinger "
    "Bands) for a single symbol. Produce a concise read (strictly under 200 words) "
    "covering: (1) trend direction and strength, (2) momentum posture, and "
    "(3) overbought/oversold conditions with any notable divergence or "
    "mean-reversion signal. Be direct, use precise terminology, avoid hedging "
    "filler, and do not restate the raw numbers verbatim. Do not give financial "
    "advice or price targets."
)


def sma(values: List[float], window: int = 20) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema_series(values: List[float], window: int) -> List[float]:
    if len(values) < window:
        return []
    k = 2.0 / (window + 1)
    seed = sum(values[:window]) / window
    out = [seed]
    for v in values[window:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema(values: List[float], window: int) -> Optional[float]:
    series = ema_series(values, window)
    return series[-1] if series else None


def rsi(values: List[float], window: int = 14) -> Optional[float]:
    if len(values) < window + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, window + 1):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / window
    avg_loss = losses / window
    for i in range(window + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    fast_series = ema_series(values, fast)
    slow_series = ema_series(values, slow)
    if not fast_series or not slow_series:
        return None, None, None
    offset = (slow - fast)
    aligned_fast = fast_series[offset:] if offset > 0 else fast_series
    length = min(len(aligned_fast), len(slow_series))
    if length == 0:
        return None, None, None
    macd_line_series = [aligned_fast[i] - slow_series[i] for i in range(length)]
    if len(macd_line_series) < signal:
        macd_line = macd_line_series[-1]
        return macd_line, None, None
    signal_series = ema_series(macd_line_series, signal)
    if not signal_series:
        return macd_line_series[-1], None, None
    macd_line = macd_line_series[-1]
    signal_line = signal_series[-1]
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(values: List[float], window: int = 20, num_std: float = 2.0):
    if len(values) < window:
        return None, None, None
    window_slice = values[-window:]
    mean = sum(window_slice) / window
    variance = sum((x - mean) ** 2 for x in window_slice) / window
    std = variance ** 0.5
    upper = mean + num_std * std
    lower = mean - num_std * std
    return upper, mean, lower


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


class TechnicalAnalystAgent:
    def __init__(self, client: anthropic.Anthropic, model: str = MODEL):
        self.client = client
        self.model = model
        self.async_client = anthropic.AsyncAnthropic(api_key=getattr(client, "api_key", None))

    def _compute_indicators(self, closes: List[float]) -> dict:
        macd_line, signal_line, histogram = macd(closes, 12, 26, 9)
        upper, middle, lower = bollinger_bands(closes, 20, 2.0)
        latest_close = closes[-1] if closes else None
        indicators = {
            "latest_close": _round(latest_close),
            "sma_20": _round(sma(closes, 20)),
            "ema_12": _round(ema(closes, 12)),
            "ema_26": _round(ema(closes, 26)),
            "rsi_14": _round(rsi(closes, 14), 2),
            "macd": {
                "macd_line": _round(macd_line),
                "signal_line": _round(signal_line),
                "histogram": _round(histogram),
            },
            "bollinger": {
                "upper": _round(upper),
                "middle": _round(middle),
                "lower": _round(lower),
            },
        }
        return indicators

    async def analyze(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> dict:
        indicators = self._compute_indicators(closes)

        if len(closes) < 30:
            return {
                "agent": "technical",
                "indicators": indicators,
                "analysis": (
                    f"Insufficient data for {symbol}: {len(closes)} data points "
                    "provided; at least 30 are required for a reliable technical read."
                ),
            }

        n = len(closes)
        recent_high = max(highs[-20:]) if len(highs) >= 20 else (max(highs) if highs else None)
        recent_low = min(lows[-20:]) if len(lows) >= 20 else (min(lows) if lows else None)
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
        latest_volume = volumes[-1] if volumes else None

        payload = {
            "symbol": symbol,
            "data_points": n,
            "latest_close": indicators["latest_close"],
            "recent_20d_high": _round(recent_high),
            "recent_20d_low": _round(recent_low),
            "latest_volume": _round(latest_volume, 0) if latest_volume is not None else None,
            "avg_volume_20d": _round(avg_volume, 0) if avg_volume is not None else None,
            "indicators": indicators,
        }

        user_content = (
            f"Symbol: {symbol}\n"
            f"Technical snapshot (JSON):\n{json.dumps(payload, indent=2)}\n\n"
            "Provide your concise technical read now."
        )

        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        analysis_text = ""
        for block in response.content:
            if block.type == "text":
                analysis_text += block.text

        return {
            "agent": "technical",
            "indicators": indicators,
            "analysis": analysis_text.strip(),
        }
