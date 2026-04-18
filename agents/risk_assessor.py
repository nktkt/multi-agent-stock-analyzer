import math
import anthropic


SYSTEM_PROMPT = """You are a risk officer at a long/short equity fund.
You receive precomputed risk metrics for a single security and produce a concise,
decision-useful interpretation for the portfolio manager.
Always address:
  - absolute level of each metric (is it high, moderate, low vs. typical equities)
  - the story the metrics tell together (e.g. high vol with deep drawdown and weak Sharpe)
  - position-sizing or hedging implications
Tone: measured, specific, no hedging filler. Hard cap: under 180 words.
Do not output JSON, markdown headers, or bullet symbols unless they add clarity."""


def returns(closes: list[float]) -> list[float]:
    out = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0:
            continue
        out.append((closes[i] - prev) / prev)
    return out


def volatility(rets: list[float]) -> float:
    n = len(rets)
    if n < 2:
        return 0.0
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252)


def max_drawdown(closes: list[float]) -> float:
    if not closes:
        return 0.0
    peak = closes[0]
    worst = 0.0
    for price in closes:
        if price > peak:
            peak = price
        if peak > 0:
            dd = (peak - price) / peak
            if dd > worst:
                worst = dd
    return worst


def value_at_risk(rets: list[float], confidence: float = 0.95) -> float:
    if not rets:
        return 0.0
    sorted_rets = sorted(rets)
    alpha = 1.0 - confidence
    idx = int(math.floor(alpha * len(sorted_rets)))
    if idx >= len(sorted_rets):
        idx = len(sorted_rets) - 1
    if idx < 0:
        idx = 0
    quantile = sorted_rets[idx]
    return -quantile if quantile < 0 else 0.0


def sharpe_ratio(rets: list[float], risk_free: float = 0.0) -> float:
    n = len(rets)
    if n < 2:
        return 0.0
    daily_rf = risk_free / 252
    excess = [r - daily_rf for r in rets]
    mean = sum(excess) / n
    var = sum((e - mean) ** 2 for e in excess) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return (mean / sd) * math.sqrt(252)


class RiskAssessorAgent:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-opus-4-7"):
        self.client = client
        self.model = model

    async def analyze(self, symbol: str, closes: list[float]) -> dict:
        rets = returns(closes)
        metrics = {
            "volatility": volatility(rets),
            "max_drawdown": max_drawdown(closes),
            "var_95": value_at_risk(rets, 0.95),
            "sharpe": sharpe_ratio(rets),
        }

        if len(closes) < 2:
            return {
                "agent": "risk",
                "metrics": metrics,
                "analysis": "Insufficient price history to form a risk view.",
            }

        user_prompt = (
            f"Ticker: {symbol}\n"
            f"Observations: {len(closes)} closes, {len(rets)} daily returns\n"
            f"Annualized volatility: {metrics['volatility']:.4f}\n"
            f"Max drawdown: {metrics['max_drawdown']:.4f}\n"
            f"Historical 95% VaR (1-day, positive fraction): {metrics['var_95']:.4f}\n"
            f"Annualized Sharpe (rf=0): {metrics['sharpe']:.4f}\n\n"
            "Write the risk interpretation now, under 180 words."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        analysis = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        return {
            "agent": "risk",
            "metrics": metrics,
            "analysis": analysis,
        }
