import asyncio
import os
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backtest import compare_strategies, signal_macd
from charts import render_chart
from data.market_data import fetch_or_synthetic
from data.news_fetcher import fetch_with_fallback
from orchestrator import StockMarketAnalyzer
from portfolio import PortfolioAgent


app = FastAPI(title="Stock AI Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

FALLBACK_HEADLINES = {
    "AAPL": [
        "Apple unveils new product lineup with strong analyst reception",
        "Apple services revenue continues to expand quarter over quarter",
        "Supply chain checks suggest steady iPhone demand",
    ],
    "MSFT": [
        "Microsoft cloud growth remains robust amid AI investments",
        "Azure wins new enterprise contracts across regions",
        "Copilot adoption accelerates among Fortune 500 customers",
    ],
    "NVDA": [
        "NVIDIA data center demand outpaces supply for latest GPUs",
        "Hyperscaler capex guidance points to sustained AI spending",
        "New chip architecture roadmap receives positive industry response",
    ],
    "GENERIC": [
        "Broader market sentiment remains mixed amid macro uncertainty",
        "Sector rotation continues as investors reposition portfolios",
        "Earnings season sets the tone for near-term price action",
    ],
}

SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,6}$")


def _validate_symbol(symbol: str) -> str:
    s = symbol.upper()
    if not SYMBOL_RE.match(s):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    return s


def _headlines_for(symbol: str) -> list[str]:
    return FALLBACK_HEADLINES.get(symbol, FALLBACK_HEADLINES["GENERIC"])


def _require_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set; agent analysis unavailable",
        )


class PortfolioRequest(BaseModel):
    weights: dict[str, float]
    days: int = 180


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index)


@app.get("/static/{path:path}")
async def static_asset(path: str):
    target = (STATIC_DIR / path).resolve()
    try:
        target.relative_to(STATIC_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target)


@app.get("/api/analyze/{symbol}")
async def analyze(symbol: str, live: bool = True):
    _require_api_key()
    sym = _validate_symbol(symbol)
    market, _ = await asyncio.to_thread(fetch_or_synthetic, sym, 180)
    headlines, _ = await asyncio.to_thread(
        fetch_with_fallback, sym, _headlines_for(sym), 10
    )
    analyzer = StockMarketAnalyzer()
    result = await analyzer.analyze(
        sym,
        market["closes"],
        market["highs"],
        market["lows"],
        market["volumes"],
        headlines,
    )
    return result


@app.get("/api/backtest/{symbol}")
async def backtest(symbol: str, days: int = 180):
    sym = _validate_symbol(symbol)
    market, _ = await asyncio.to_thread(fetch_or_synthetic, sym, days)
    result = await asyncio.to_thread(compare_strategies, market["closes"])
    return result


@app.get("/api/chart/{symbol}")
async def chart(symbol: str, days: int = 180):
    sym = _validate_symbol(symbol)
    market, _ = await asyncio.to_thread(fetch_or_synthetic, sym, days)
    signals = await asyncio.to_thread(signal_macd, market["closes"])
    png = await asyncio.to_thread(
        render_chart,
        market["closes"],
        market["highs"],
        market["lows"],
        market.get("dates") or None,
        signals,
        f"{sym} price with MACD signals",
    )
    return Response(content=png, media_type="image/png")


@app.post("/api/portfolio")
async def portfolio(req: PortfolioRequest):
    _require_api_key()
    if not req.weights:
        raise HTTPException(status_code=400, detail="weights required")
    closes_by_symbol: dict[str, list[float]] = {}
    normalized_weights: dict[str, float] = {}
    for raw_symbol, weight in req.weights.items():
        sym = _validate_symbol(raw_symbol)
        market, _ = await asyncio.to_thread(fetch_or_synthetic, sym, req.days)
        closes_by_symbol[sym] = market["closes"]
        normalized_weights[sym] = float(weight)
    agent = PortfolioAgent()
    result = await agent.analyze(normalized_weights, closes_by_symbol)
    return result


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
