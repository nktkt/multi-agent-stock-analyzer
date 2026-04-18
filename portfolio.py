import math
import anthropic

from agents.risk_assessor import (
    returns,
    volatility,
    max_drawdown,
    value_at_risk,
    sharpe_ratio,
)


SYSTEM_PROMPT = """You are a portfolio manager overseeing a multi-asset book.
You receive precomputed portfolio-level statistics: weights, a pairwise return
correlation matrix, a diversification score, and risk metrics on the weighted
portfolio return series.
Assess, in order:
  - concentration risk in the weight vector (single-name or cluster exposure)
  - the correlation regime across holdings (tight, dispersed, regime-shifted)
  - the resulting diversification quality and whether portfolio vol / drawdown /
    VaR look commensurate with the weight and correlation picture
  - concrete rebalancing or hedging implications
Tone: direct, specific, no filler. Hard cap: under 250 words.
Do not output JSON, markdown headers, or bullet symbols unless they add clarity."""


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs = xs[:n]
    ys = ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def correlation_matrix(returns_by_symbol: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    symbols = list(returns_by_symbol.keys())
    if not symbols:
        return {}
    min_len = min(len(returns_by_symbol[s]) for s in symbols)
    truncated = {s: returns_by_symbol[s][-min_len:] for s in symbols}
    out: dict[str, dict[str, float]] = {s: {} for s in symbols}
    for i, a in enumerate(symbols):
        out[a][a] = 1.0
        for j in range(i + 1, len(symbols)):
            b = symbols[j]
            r = pearson_correlation(truncated[a], truncated[b])
            out[a][b] = r
            out[b][a] = r
    return out


def portfolio_returns(
    weights: dict[str, float],
    returns_by_symbol: dict[str, list[float]],
) -> list[float]:
    symbols = [s for s in weights if s in returns_by_symbol]
    if not symbols:
        return []
    min_len = min(len(returns_by_symbol[s]) for s in symbols)
    if min_len == 0:
        return []
    aligned = {s: returns_by_symbol[s][-min_len:] for s in symbols}
    out = []
    for i in range(min_len):
        out.append(sum(weights[s] * aligned[s][i] for s in symbols))
    return out


def diversification_score(corr: dict[str, dict[str, float]]) -> float:
    symbols = list(corr.keys())
    n = len(symbols)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += corr[symbols[i]][symbols[j]]
            count += 1
    if count == 0:
        return 1.0
    avg = total / count
    score = 1.0 - avg
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _portfolio_closes_from_returns(port_rets: list[float], base: float = 100.0) -> list[float]:
    closes = [base]
    for r in port_rets:
        closes.append(closes[-1] * (1.0 + r))
    return closes


def portfolio_metrics(
    weights: dict[str, float],
    closes_by_symbol: dict[str, list[float]],
) -> dict:
    returns_by_symbol = {s: returns(closes_by_symbol[s]) for s in closes_by_symbol}
    port_rets = portfolio_returns(weights, returns_by_symbol)
    port_closes = _portfolio_closes_from_returns(port_rets)
    return {
        "volatility": volatility(port_rets),
        "var_95": value_at_risk(port_rets, 0.95),
        "max_drawdown": max_drawdown(port_closes),
        "sharpe": sharpe_ratio(port_rets),
    }


def _validate_weights(weights: dict[str, float]) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Portfolio weights must sum to ~1.0 (got {total:.4f})."
        )


class PortfolioAgent:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-opus-4-7"):
        self.client = client
        self.model = model

    async def analyze(
        self,
        weights: dict[str, float],
        closes_by_symbol: dict[str, list[float]],
    ) -> dict:
        _validate_weights(weights)

        returns_by_symbol = {s: returns(closes_by_symbol[s]) for s in closes_by_symbol}
        corr = correlation_matrix(returns_by_symbol)
        div_score = diversification_score(corr)
        metrics = portfolio_metrics(weights, closes_by_symbol)

        symbols = list(weights.keys())
        weight_lines = "\n".join(f"  {s}: {weights[s]:.4f}" for s in symbols)

        corr_lines = []
        corr_symbols = list(corr.keys())
        for i, a in enumerate(corr_symbols):
            for j in range(i + 1, len(corr_symbols)):
                b = corr_symbols[j]
                corr_lines.append(f"  {a}-{b}: {corr[a][b]:+.3f}")
        corr_block = "\n".join(corr_lines) if corr_lines else "  (single holding)"

        user_prompt = (
            f"Holdings: {len(symbols)}\n"
            f"Weights:\n{weight_lines}\n\n"
            f"Pairwise return correlations:\n{corr_block}\n\n"
            f"Diversification score (1 - avg off-diagonal corr, clipped): {div_score:.4f}\n\n"
            f"Portfolio risk metrics:\n"
            f"  Annualized volatility: {metrics['volatility']:.4f}\n"
            f"  Max drawdown: {metrics['max_drawdown']:.4f}\n"
            f"  Historical 95% VaR (1-day, positive fraction): {metrics['var_95']:.4f}\n"
            f"  Annualized Sharpe (rf=0): {metrics['sharpe']:.4f}\n\n"
            "Write the portfolio assessment now, under 250 words."
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
            "agent": "portfolio",
            "weights": weights,
            "correlation_matrix": corr,
            "diversification_score": div_score,
            "metrics": metrics,
            "analysis": analysis,
        }
