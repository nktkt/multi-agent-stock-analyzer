from __future__ import annotations

import argparse
from typing import Optional

from agents.technical_analyst import sma, ema, rsi, macd


def _sma_series(closes: list[float], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(len(closes)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(sma(closes[: i + 1], window))
    return out


def _ema_series_aligned(closes: list[float], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(len(closes)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(ema(closes[: i + 1], window))
    return out


def _rsi_series(closes: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(len(closes)):
        if i + 1 < period + 1:
            out.append(None)
        else:
            out.append(rsi(closes[: i + 1], period))
    return out


def _macd_series(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    macd_line: list[Optional[float]] = []
    signal_line: list[Optional[float]] = []
    for i in range(len(closes)):
        m, s, _h = macd(closes[: i + 1], fast, slow, signal)
        macd_line.append(m)
        signal_line.append(s)
    return macd_line, signal_line


def signal_sma_cross(closes: list[float], fast: int = 20, slow: int = 50) -> list[int]:
    n = len(closes)
    fast_s = _sma_series(closes, fast)
    slow_s = _sma_series(closes, slow)
    signals = [0] * n
    prev_diff: Optional[float] = None
    for i in range(n):
        if fast_s[i] is None or slow_s[i] is None:
            prev_diff = None
            continue
        diff = fast_s[i] - slow_s[i]
        if prev_diff is not None:
            if prev_diff <= 0 and diff > 0:
                signals[i] = 1
            elif prev_diff >= 0 and diff < 0:
                signals[i] = -1
        prev_diff = diff
    return signals


def signal_rsi(
    closes: list[float], period: int = 14, oversold: float = 30, overbought: float = 70
) -> list[int]:
    n = len(closes)
    rsi_s = _rsi_series(closes, period)
    signals = [0] * n
    prev: Optional[float] = None
    for i in range(n):
        cur = rsi_s[i]
        if cur is None:
            prev = None
            continue
        if prev is not None:
            if prev <= oversold and cur > oversold:
                signals[i] = 1
            elif prev >= overbought and cur < overbought:
                signals[i] = -1
        prev = cur
    return signals


def signal_macd(closes: list[float]) -> list[int]:
    n = len(closes)
    macd_line, signal_line = _macd_series(closes)
    signals = [0] * n
    prev_diff: Optional[float] = None
    for i in range(n):
        if macd_line[i] is None or signal_line[i] is None:
            prev_diff = None
            continue
        diff = macd_line[i] - signal_line[i]
        if prev_diff is not None:
            if prev_diff <= 0 and diff > 0:
                signals[i] = 1
            elif prev_diff >= 0 and diff < 0:
                signals[i] = -1
        prev_diff = diff
    return signals


def backtest(closes: list[float], signals: list[int]) -> dict:
    n = min(len(closes), len(signals))
    in_position = False
    entry_price = 0.0
    equity = 1.0
    equity_curve = [1.0]
    trade_returns: list[float] = []

    for i in range(n):
        price = closes[i]
        sig = signals[i]
        if not in_position and sig == 1:
            in_position = True
            entry_price = price
        elif in_position and sig == -1:
            ret = (price - entry_price) / entry_price if entry_price else 0.0
            trade_returns.append(ret)
            equity *= 1.0 + ret
            in_position = False
        if in_position and entry_price:
            mark = equity * (price / entry_price)
        else:
            mark = equity
        equity_curve.append(mark)

    if in_position and entry_price:
        final_price = closes[n - 1]
        ret = (final_price - entry_price) / entry_price
        trade_returns.append(ret)
        equity *= 1.0 + ret

    total_return = equity - 1.0
    num_trades = len(trade_returns)
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = (wins / num_trades) if num_trades else 0.0
    avg_trade_return = (sum(trade_returns) / num_trades) if num_trades else 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

    return {
        "total_return": total_return,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "max_drawdown": max_dd,
    }


def _buy_and_hold(closes: list[float]) -> dict:
    if not closes:
        return {
            "total_return": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "max_drawdown": 0.0,
        }
    start = closes[0]
    total_return = (closes[-1] - start) / start if start else 0.0
    peak = start
    max_dd = 0.0
    for p in closes:
        if p > peak:
            peak = p
        if peak > 0:
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd
    return {
        "total_return": total_return,
        "num_trades": 1,
        "win_rate": 1.0 if total_return > 0 else 0.0,
        "avg_trade_return": total_return,
        "max_drawdown": max_dd,
    }


def compare_strategies(closes: list[float]) -> dict:
    return {
        "sma_cross": backtest(closes, signal_sma_cross(closes)),
        "rsi": backtest(closes, signal_rsi(closes)),
        "macd": backtest(closes, signal_macd(closes)),
        "buy_and_hold": _buy_and_hold(closes),
    }


def _format_table(results: dict) -> str:
    headers = ["Strategy", "TotalRet", "Trades", "WinRate", "AvgTrade", "MaxDD"]
    widths = [16, 10, 8, 10, 10, 10]
    lines = []
    header = "".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header)
    lines.append("-" * sum(widths))
    for name, r in results.items():
        row = [
            name,
            f"{r['total_return'] * 100:+.2f}%",
            f"{r['num_trades']}",
            f"{r['win_rate'] * 100:.1f}%",
            f"{r['avg_trade_return'] * 100:+.2f}%",
            f"{r['max_drawdown'] * 100:.2f}%",
        ]
        lines.append("".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest technical signals.")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    from data.market_data import fetch_or_synthetic

    data, source = fetch_or_synthetic(args.symbol, args.days)
    closes = [float(c) for c in data["closes"]]

    results = compare_strategies(closes)
    print(f"Backtest: {args.symbol} over {len(closes)} bars ({args.days} days requested) [source: {source}]")
    print()
    print(_format_table(results))


if __name__ == "__main__":
    main()
