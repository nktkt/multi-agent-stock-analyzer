import argparse
import asyncio
import json
import os
import random
import sys
from typing import Any

from data.market_data import fetch_or_synthetic
from data.news_fetcher import fetch_with_fallback
from orchestrator import StockMarketAnalyzer


SYMBOL_SEEDS = {
    "AAPL": {"start_price": 225.0, "base_volume": 55_000_000},
    "MSFT": {"start_price": 420.0, "base_volume": 22_000_000},
    "NVDA": {"start_price": 135.0, "base_volume": 240_000_000},
}

HEADLINES = {
    "AAPL": [
        "Apple reports record services revenue as iPhone demand stabilizes",
        "Analysts raise Apple price targets on AI-enabled iPhone upgrade cycle",
        "Apple faces renewed antitrust scrutiny in European App Store probe",
        "Supply chain checks suggest softer Vision Pro sell-through than expected",
        "Apple boosts buyback authorization, signaling confidence in cash generation",
    ],
    "MSFT": [
        "Microsoft Azure growth reaccelerates, fueled by enterprise AI workloads",
        "Copilot enterprise seats cross milestone as CIO adoption broadens",
        "Regulators open inquiry into Microsoft's OpenAI investment structure",
        "Datacenter capex outlook raises margin questions for fiscal year ahead",
        "Microsoft announces expanded partnership with leading chip manufacturer",
    ],
    "NVDA": [
        "Nvidia Blackwell chips enter volume production, demand outpaces supply",
        "Hyperscaler capex guidance lifts Nvidia forward revenue expectations",
        "Short-seller report questions sustainability of Nvidia gross margins",
        "Export-control updates may restrict Nvidia shipments to select markets",
        "Nvidia unveils new networking silicon to extend datacenter platform moat",
    ],
}


def generate_ohlcv(symbol: str, days: int = 60) -> dict[str, list[float]]:
    cfg = SYMBOL_SEEDS[symbol]
    rng = random.Random(hash(symbol) & 0xFFFFFFFF)

    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []

    price = cfg["start_price"]
    for _ in range(days):
        daily_return = rng.gauss(0.0005, 0.015)
        open_price = price
        close_price = max(1.0, price * (1.0 + daily_return))
        intraday_range = abs(rng.gauss(0.0, 0.012)) * close_price + 0.001 * close_price
        high = max(open_price, close_price) + intraday_range * rng.random()
        low = min(open_price, close_price) - intraday_range * rng.random()
        low = max(0.5, low)
        vol_noise = rng.gauss(1.0, 0.25)
        volume = max(1.0, cfg["base_volume"] * vol_noise)

        closes.append(round(close_price, 2))
        highs.append(round(high, 2))
        lows.append(round(low, 2))
        volumes.append(round(volume, 0))

        price = close_price

    return {"closes": closes, "highs": highs, "lows": lows, "volumes": volumes}


def build_sample_data(symbol: str, live: bool = False) -> dict[str, Any]:
    if symbol not in SYMBOL_SEEDS:
        raise ValueError(
            f"Unknown symbol {symbol!r}. Supported: {', '.join(sorted(SYMBOL_SEEDS))}"
        )
    if live:
        ohlcv, source = fetch_or_synthetic(
            symbol, days=90, seed_config=SYMBOL_SEEDS[symbol]
        )
        headlines, news_source = fetch_with_fallback(symbol, HEADLINES[symbol])
    else:
        ohlcv = generate_ohlcv(symbol)
        source = "synthetic"
        headlines = HEADLINES[symbol]
        news_source = "hardcoded"
    return {
        "symbol": symbol,
        "source": source,
        "news_source": news_source,
        **ohlcv,
        "headlines": headlines,
    }


def print_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def print_subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def render_agent_block(agent_result: dict[str, Any]) -> None:
    name = agent_result.get("agent", "unknown")
    print_subheader(name.replace("_", " ").title())
    if "error" in agent_result:
        print(f"[ERROR] {agent_result['error']}")
        return
    body = {k: v for k, v in agent_result.items() if k != "agent"}
    print(json.dumps(body, indent=2, default=str))


async def run(symbol: str, live: bool = False) -> int:
    data = build_sample_data(symbol, live=live)

    analyzer = StockMarketAnalyzer()

    print_header(f"Stock Market Pattern Analysis: {symbol}")
    print(f"Data source: {data['source']}")
    print(
        f"Loaded {len(data['closes'])} days of OHLCV | "
        f"{len(data['headlines'])} headlines"
    )
    print(f"Latest close: {data['closes'][-1]:.2f}")

    result = await analyzer.analyze(
        symbol=data["symbol"],
        closes=data["closes"],
        highs=data["highs"],
        lows=data["lows"],
        volumes=data["volumes"],
        headlines=data["headlines"],
    )

    print_header("Specialist Agent Reports")
    for agent_result in result["agents"]:
        render_agent_block(agent_result)

    print_header("Chief Strategist Synthesis")
    print(result["synthesis"])
    print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-agent stock market pattern analysis (Claude-powered)."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        choices=sorted(SYMBOL_SEEDS.keys()),
        help="Ticker symbol to analyze.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch real market data via yfinance (falls back to synthetic).",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it before running, e.g.:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(run(args.symbol, live=args.live))


if __name__ == "__main__":
    raise SystemExit(main())
