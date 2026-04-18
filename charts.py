from __future__ import annotations

import argparse
import io
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from agents.technical_analyst import bollinger_bands, rsi, sma


def _rolling(
    func: Callable,
    closes: list[float],
    window: int,
    min_len: int,
) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(len(closes)):
        if i + 1 < min_len:
            out.append(None)
        else:
            value = func(closes[: i + 1], window)
            out.append(value)
    return out


def _bollinger_series(
    closes: list[float], window: int = 20, num_std: float = 2.0
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    upper: list[Optional[float]] = []
    middle: list[Optional[float]] = []
    lower: list[Optional[float]] = []
    for i in range(len(closes)):
        if i + 1 < window:
            upper.append(None)
            middle.append(None)
            lower.append(None)
        else:
            u, m, l = bollinger_bands(closes[: i + 1], window, num_std)
            upper.append(u)
            middle.append(m)
            lower.append(l)
    return middle, upper, lower


def _mask_none(values: list[Optional[float]]) -> list[float]:
    return [float("nan") if v is None else float(v) for v in values]


def render_chart(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    dates: list[str] | None = None,
    signals: list[int] | None = None,
    title: str = "",
) -> bytes:
    n = len(closes)
    x = list(range(n)) if dates is None else list(range(n))
    xtick_labels = dates if dates is not None else None

    sma20 = _rolling(sma, closes, 20, 20)
    sma50 = _rolling(sma, closes, 50, 50)
    rsi_series = _rolling(rsi, closes, 14, 15)
    bb_mid, bb_up, bb_low = _bollinger_series(closes, 20, 2.0)

    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(x, closes, color="#1f77b4", linewidth=1.4, label="Close")
    ax_price.plot(x, _mask_none(sma20), color="#ff7f0e", linewidth=1.0, label="SMA 20")
    ax_price.plot(x, _mask_none(sma50), color="#2ca02c", linewidth=1.0, label="SMA 50")

    bb_up_f = _mask_none(bb_up)
    bb_low_f = _mask_none(bb_low)
    ax_price.fill_between(
        x,
        bb_low_f,
        bb_up_f,
        color="#9467bd",
        alpha=0.15,
        label="Bollinger (20, 2)",
    )
    ax_price.plot(x, _mask_none(bb_mid), color="#9467bd", linewidth=0.7, alpha=0.6)

    if signals is not None:
        buy_x = [i for i, s in enumerate(signals) if s == 1 and i < n]
        buy_y = [closes[i] for i in buy_x]
        sell_x = [i for i, s in enumerate(signals) if s == -1 and i < n]
        sell_y = [closes[i] for i in sell_x]
        if buy_x:
            ax_price.scatter(
                buy_x, buy_y, marker="^", color="green", s=60, zorder=5, label="BUY"
            )
        if sell_x:
            ax_price.scatter(
                sell_x, sell_y, marker="v", color="red", s=60, zorder=5, label="SELL"
            )

    ax_price.set_ylabel("Price")
    ax_price.set_title(title)
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(True, linestyle="--", alpha=0.3)

    ax_rsi.plot(x, _mask_none(rsi_series), color="#8c564b", linewidth=1.0, label="RSI 14")
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.7, alpha=0.7)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.7, alpha=0.7)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(True, linestyle="--", alpha=0.3)
    ax_rsi.legend(loc="upper left", fontsize=8)

    if xtick_labels is not None and n > 0:
        step = max(1, n // 8)
        ticks = list(range(0, n, step))
        labels = [xtick_labels[i] for i in ticks]
        ax_rsi.set_xticks(ticks)
        ax_rsi.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    else:
        ax_rsi.set_xlabel("Index")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def render_comparison_chart(results: dict[str, dict]) -> bytes:
    names = list(results.keys())
    returns = [float(results[k].get("total_return", 0.0)) * 100.0 for k in names]
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in returns]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, returns, color=colors, edgecolor="black", linewidth=0.5)

    for bar, value in zip(bars, returns):
        height = bar.get_height()
        offset = 0.5 if height >= 0 else -1.0
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{value:+.2f}%",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Strategy Comparison: Total Return")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a technical chart PNG.")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--out", type=str, default="chart.png")
    args = parser.parse_args()

    from data.market_data import fetch_or_synthetic
    from backtest import signal_macd

    data, source = fetch_or_synthetic(args.symbol, args.days)
    closes = [float(c) for c in data["closes"]]
    highs = [float(h) for h in data["highs"]]
    lows = [float(l) for l in data["lows"]]

    signals = signal_macd(closes)
    title = f"{args.symbol} ({len(closes)} bars, source: {source})"
    png = render_chart(closes, highs, lows, dates=None, signals=signals, title=title)

    with open(args.out, "wb") as f:
        f.write(png)

    print(f"Wrote {args.out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
