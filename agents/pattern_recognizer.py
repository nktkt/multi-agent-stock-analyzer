from __future__ import annotations

import anthropic


def find_local_extrema(
    prices: list[float], window: int = 5
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    peaks: list[tuple[int, float]] = []
    troughs: list[tuple[int, float]] = []
    n = len(prices)
    if n < 2 * window + 1:
        return peaks, troughs
    for i in range(window, n - window):
        left = prices[i - window : i]
        right = prices[i + 1 : i + 1 + window]
        center = prices[i]
        if center > max(left) and center > max(right):
            peaks.append((i, center))
        elif center < min(left) and center < min(right):
            troughs.append((i, center))
    return peaks, troughs


def detect_double_top(peaks: list[tuple[int, float]]) -> list[dict]:
    results: list[dict] = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            idx1, p1 = peaks[i]
            idx2, p2 = peaks[j]
            if idx2 - idx1 < 5:
                continue
            avg = (p1 + p2) / 2.0
            if avg == 0:
                continue
            diff = abs(p1 - p2) / avg
            if diff <= 0.03:
                results.append(
                    {
                        "pattern": "double_top",
                        "indices": [idx1, idx2],
                        "prices": [p1, p2],
                        "similarity": round(1.0 - diff, 4),
                    }
                )
    return results


def detect_double_bottom(troughs: list[tuple[int, float]]) -> list[dict]:
    results: list[dict] = []
    for i in range(len(troughs)):
        for j in range(i + 1, len(troughs)):
            idx1, t1 = troughs[i]
            idx2, t2 = troughs[j]
            if idx2 - idx1 < 5:
                continue
            avg = (t1 + t2) / 2.0
            if avg == 0:
                continue
            diff = abs(t1 - t2) / avg
            if diff <= 0.03:
                results.append(
                    {
                        "pattern": "double_bottom",
                        "indices": [idx1, idx2],
                        "prices": [t1, t2],
                        "similarity": round(1.0 - diff, 4),
                    }
                )
    return results


def detect_head_and_shoulders(peaks: list[tuple[int, float]]) -> list[dict]:
    results: list[dict] = []
    for i in range(len(peaks) - 2):
        left_idx, left_p = peaks[i]
        head_idx, head_p = peaks[i + 1]
        right_idx, right_p = peaks[i + 2]
        if head_p <= left_p or head_p <= right_p:
            continue
        avg_shoulders = (left_p + right_p) / 2.0
        if avg_shoulders == 0:
            continue
        shoulder_diff = abs(left_p - right_p) / avg_shoulders
        if shoulder_diff <= 0.05:
            results.append(
                {
                    "pattern": "head_and_shoulders",
                    "indices": [left_idx, head_idx, right_idx],
                    "prices": [left_p, head_p, right_p],
                    "shoulder_symmetry": round(1.0 - shoulder_diff, 4),
                }
            )
    return results


def detect_trend(closes: list[float]) -> str:
    n = len(closes)
    if n < 6:
        return "sideways"
    third = n // 3
    start_avg = sum(closes[:third]) / third
    mid_avg = sum(closes[third : 2 * third]) / third
    end_avg = sum(closes[2 * third :]) / (n - 2 * third)
    if start_avg == 0:
        return "sideways"
    total_change = (end_avg - start_avg) / start_avg
    x_mean = (n - 1) / 2.0
    y_mean = sum(closes) / n
    num = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    normalized_slope = slope / y_mean if y_mean != 0 else 0.0
    if total_change > 0.03 and normalized_slope > 0 and end_avg > mid_avg > start_avg * 0.98:
        return "uptrend"
    if total_change < -0.03 and normalized_slope < 0 and end_avg < mid_avg < start_avg * 1.02:
        return "downtrend"
    if total_change > 0.05 and normalized_slope > 0:
        return "uptrend"
    if total_change < -0.05 and normalized_slope < 0:
        return "downtrend"
    return "sideways"


def _line_slope(points: list[tuple[int, float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    x_mean = sum(p[0] for p in points) / n
    y_mean = sum(p[1] for p in points) / n
    num = sum((p[0] - x_mean) * (p[1] - y_mean) for p in points)
    den = sum((p[0] - x_mean) ** 2 for p in points)
    if den == 0:
        return 0.0
    slope = num / den
    return slope / y_mean if y_mean != 0 else 0.0


def detect_triangle(
    peaks: list[tuple[int, float]], troughs: list[tuple[int, float]]
) -> list[dict]:
    if len(peaks) < 2 or len(troughs) < 2:
        return []
    peak_slope = _line_slope(peaks)
    trough_slope = _line_slope(troughs)
    flat_threshold = 0.0005
    results: list[dict] = []
    if peak_slope < -flat_threshold and trough_slope > flat_threshold:
        results.append(
            {
                "pattern": "symmetrical_triangle",
                "peak_slope": round(peak_slope, 6),
                "trough_slope": round(trough_slope, 6),
            }
        )
    elif abs(peak_slope) <= flat_threshold and trough_slope > flat_threshold:
        results.append(
            {
                "pattern": "ascending_triangle",
                "peak_slope": round(peak_slope, 6),
                "trough_slope": round(trough_slope, 6),
            }
        )
    elif peak_slope < -flat_threshold and abs(trough_slope) <= flat_threshold:
        results.append(
            {
                "pattern": "descending_triangle",
                "peak_slope": round(peak_slope, 6),
                "trough_slope": round(trough_slope, 6),
            }
        )
    return results


class PatternRecognizerAgent:
    SYSTEM_PROMPT = (
        "You are a veteran technical chartist with decades of experience reading "
        "equity price action. You specialize in classical chart patterns: double "
        "tops and bottoms, head-and-shoulders formations, and triangle "
        "consolidations (symmetrical, ascending, descending). For each pattern "
        "presented to you, explain concisely what it historically implies for "
        "future price direction, how reliable it tends to be, and what typical "
        "targets or invalidation levels traders look for. Weight your outlook by "
        "the confidence implied by the geometry (symmetry, duration, volume "
        "context when available) and by the current trend. Be disciplined: if "
        "signals conflict, say so. Always produce a final confidence-weighted "
        "outlook (bullish / bearish / neutral with a qualitative confidence "
        "level). Keep the full analysis under 200 words. Never give financial "
        "advice — frame everything as pattern-based historical tendencies."
    )

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str = "claude-opus-4-7",
    ) -> None:
        self.client = client
        self.model = model

    async def analyze(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
    ) -> dict:
        peaks, troughs = find_local_extrema(highs)
        _, low_troughs = find_local_extrema(lows)
        combined_troughs = troughs + low_troughs
        combined_troughs.sort(key=lambda t: t[0])

        patterns: list[dict] = []
        patterns.extend(detect_double_top(peaks))
        patterns.extend(detect_double_bottom(combined_troughs))
        patterns.extend(detect_head_and_shoulders(peaks))
        patterns.extend(detect_triangle(peaks, combined_troughs))

        trend = detect_trend(closes)

        latest_price = closes[-1] if closes else None
        first_price = closes[0] if closes else None
        pct_change = None
        if latest_price is not None and first_price not in (None, 0):
            pct_change = round((latest_price - first_price) / first_price * 100, 2)

        if patterns:
            pattern_summary = "\n".join(
                f"- {p['pattern']}: {p}" for p in patterns
            )
            user_content = (
                f"Symbol: {symbol}\n"
                f"Bars analyzed: {len(closes)}\n"
                f"Current trend (derived): {trend}\n"
                f"Latest close: {latest_price}\n"
                f"Period change: {pct_change}%\n\n"
                f"Detected patterns:\n{pattern_summary}\n\n"
                "Interpret each pattern's historical implications and provide "
                "a confidence-weighted outlook for the near-term."
            )
        else:
            user_content = (
                f"Symbol: {symbol}\n"
                f"Bars analyzed: {len(closes)}\n"
                f"Current trend (derived): {trend}\n"
                f"Latest close: {latest_price}\n"
                f"Period change: {pct_change}%\n\n"
                "No classical chart patterns were detected in this window. "
                "Provide a concise read on the prevailing trend, typical "
                "behavior to expect while no pattern is active, and what "
                "developments would constitute a meaningful signal."
            )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": self.SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        analysis_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return {
            "agent": "pattern",
            "patterns_detected": patterns,
            "trend": trend,
            "analysis": analysis_text,
        }
