# Multi-Agent Stock Market Pattern Analyzer

A Claude-powered multi-agent AI that analyzes stock market patterns using four specialist agents running in parallel, with a FastAPI web dashboard, real market data, backtesting, and chart generation.

## Architecture

Four specialist agents run concurrently and a Chief Strategist synthesizes their reports:

| Agent | Role |
| --- | --- |
| **Technical Analyst** | SMA, EMA, RSI, MACD, Bollinger Bands — trend and momentum |
| **Pattern Recognizer** | Double top/bottom, head & shoulders, triangles, trend classification |
| **Sentiment Analyst** | Scores news headlines and produces a financial sentiment read |
| **Risk Assessor** | Volatility (annualized), max drawdown, VaR 95%, Sharpe ratio |
| **Portfolio Agent** | Correlation matrix, diversification score, portfolio-level metrics |

All agents use the Anthropic SDK (`AsyncAnthropic`) with the `claude-opus-4-7` model. Specialist calls run in parallel via `asyncio.gather`, then their outputs are fed back to Claude for a unified Chief Strategist recommendation. Prompt caching is enabled on every system prompt.

## Features

- **Parallel multi-agent pipeline** — four specialists run concurrently, one synthesis pass at the end
- **Real market data** via `yfinance` with graceful fallback to seeded synthetic data
- **Real news headlines** via Yahoo Finance RSS with fallback to curated sample headlines
- **Backtesting** of SMA cross, RSI mean-reversion, and MACD strategies vs. buy-and-hold
- **Chart generation** — price + SMA 20/50 + Bollinger bands + buy/sell markers + RSI subplot as PNG
- **Portfolio analysis** — correlation matrix, diversification score, portfolio volatility / VaR / Sharpe
- **FastAPI web dashboard** — single-page vanilla-JS UI with tabs for Analyze / Backtest / Chart / Portfolio
- **Unit tests** — 27 tests covering all indicator math and pattern detectors

## Requirements

- Python 3.10+
- An Anthropic API key (set `ANTHROPIC_API_KEY`)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install matplotlib fastapi 'uvicorn[standard]'
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### CLI — full multi-agent analysis

```bash
# Synthetic data, built-in sample headlines
.venv/bin/python main.py --symbol AAPL

# Live Yahoo Finance data + Yahoo RSS headlines
.venv/bin/python main.py --symbol AAPL --live
```

Supported sample symbols: `AAPL`, `MSFT`, `NVDA`. Any ticker works with `--live`.

### CLI — backtest technical strategies

```bash
.venv/bin/python backtest.py --symbol AAPL --days 180
```

Example output:

```
Backtest: AAPL over 180 bars (180 days requested) [source: yfinance]

Strategy        TotalRet  Trades  WinRate   AvgTrade  MaxDD
----------------------------------------------------------------
sma_cross       -6.10%    1       0.0%      -6.10%    8.86%
rsi             +3.93%    1       100.0%    +3.93%    11.24%
macd            +5.43%    6       50.0%     +0.92%    10.64%
buy_and_hold    +30.58%   1       100.0%    +30.58%   13.80%
```

### CLI — render a chart

```bash
.venv/bin/python charts.py --symbol AAPL --days 180 --out chart.png
```

### Web dashboard

```bash
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ in a browser. The dashboard has four tabs:

- **Analyze** — full multi-agent analysis with synthesis
- **Backtest** — strategy comparison table
- **Chart** — PNG chart with signals
- **Portfolio** — editable weight grid, correlation matrix, portfolio metrics

### REST API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/analyze/{symbol}?live=true` | Full multi-agent analysis (requires API key) |
| GET | `/api/backtest/{symbol}?days=180` | Strategy comparison JSON |
| GET | `/api/chart/{symbol}?days=180` | PNG chart (image/png) |
| POST | `/api/portfolio` | Body: `{"weights": {"AAPL": 0.5, ...}, "days": 180}` |

## Project Structure

```
.
├── agents/
│   ├── technical_analyst.py    # SMA/EMA/RSI/MACD/Bollinger + LLM agent
│   ├── pattern_recognizer.py   # Chart pattern detection + LLM agent
│   ├── sentiment_analyst.py    # Headline sentiment scoring
│   └── risk_assessor.py        # Volatility / VaR / drawdown / Sharpe
├── data/
│   ├── market_data.py          # yfinance fetch + synthetic fallback
│   └── news_fetcher.py         # Yahoo Finance RSS + fallback
├── tests/
│   ├── test_indicators.py      # 18 indicator tests
│   └── test_patterns.py        # 9 pattern-detector tests
├── static/
│   └── index.html              # Single-page dashboard
├── orchestrator.py             # StockMarketAnalyzer — parallel pipeline
├── portfolio.py                # Correlation + portfolio metrics + agent
├── backtest.py                 # Signal generators + strategy backtest
├── charts.py                   # matplotlib PNG renderer
├── server.py                   # FastAPI web service
├── main.py                     # CLI entry point
└── requirements.txt
```

## Testing

```bash
.venv/bin/python -m unittest discover tests -v
```

27 tests pass, covering:

- SMA / EMA / RSI / MACD / Bollinger Bands correctness on known inputs
- Edge cases (insufficient data, constant series, monotonic series)
- `find_local_extrema`, `detect_double_top`, `detect_trend`

## Notes

- Indicator functions return scalar "latest value" results; `ema_series` returns the full aligned list. Callers that need full series (e.g. `backtest.py`) wrap scalar calls over progressively-extended slices.
- Agents gracefully degrade when inputs are insufficient (e.g. `TechnicalAnalystAgent` skips the LLM call with fewer than 30 data points).
- All LLM calls use prompt caching on the system prompt.
- `fetch_or_synthetic` and `fetch_with_fallback` never raise on network failure — they fall back to local defaults so the pipeline always runs.

## License

MIT
