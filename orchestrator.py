import asyncio
import json
from typing import Any

import anthropic

from agents.technical_analyst import TechnicalAnalystAgent
from agents.pattern_recognizer import PatternRecognizerAgent
from agents.sentiment_analyst import SentimentAnalystAgent
from agents.risk_assessor import RiskAssessorAgent


SYNTHESIS_SYSTEM_PROMPT = """You are the Chief Strategist of a multi-agent stock market analysis desk.

Four specialist agents report to you:
  1. Technical Analyst - trend, momentum, moving averages, volume-based signals.
  2. Pattern Recognizer - chart patterns, support/resistance, formations.
  3. Sentiment Analyst - news and headline sentiment.
  4. Risk Assessor - volatility, drawdown, value-at-risk style metrics.

Your job:
  - Read each specialist's report (which may include an error field if that specialist failed).
  - Weigh conflicting signals and explicitly note where the specialists disagree.
  - Produce a unified, concise recommendation with:
      * Overall directional bias (bullish / neutral / bearish) and conviction level.
      * Key supporting evidence, drawn from the specialists.
      * The dominant risks and what would invalidate the thesis.
      * A short actionable summary (1-3 sentences).

Be decisive but honest about uncertainty. Do not invent data the specialists did not provide.
Keep the full response under ~400 words."""


class StockMarketAnalyzer:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-7",
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        self.model = model
        self.technical = TechnicalAnalystAgent(self.client)
        self.patterns = PatternRecognizerAgent(self.client)
        self.sentiment = SentimentAnalystAgent(self.client)
        self.risk = RiskAssessorAgent(self.client)

    async def analyze(
        self,
        symbol: str,
        closes,
        highs,
        lows,
        volumes,
        headlines,
    ) -> dict:
        tasks = [
            ("technical_analyst", self.technical.analyze(symbol, closes, highs, lows, volumes)),
            ("pattern_recognizer", self.patterns.analyze(symbol, closes, highs, lows)),
            ("sentiment_analyst", self.sentiment.analyze(symbol, headlines)),
            ("risk_assessor", self.risk.analyze(symbol, closes)),
        ]
        names = [n for n, _ in tasks]
        coros = [c for _, c in tasks]

        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        results: list[dict[str, Any]] = []
        for name, result in zip(names, raw_results):
            if isinstance(result, Exception):
                results.append({"agent": name, "error": str(result)})
            else:
                entry = {"agent": name}
                if isinstance(result, dict):
                    entry.update(result)
                else:
                    entry["output"] = result
                results.append(entry)

        synthesis = await self._synthesize(symbol, results)

        return {
            "symbol": symbol,
            "agents": results,
            "synthesis": synthesis,
        }

    async def _synthesize(self, symbol: str, results: list[dict]) -> str:
        sections = []
        for r in results:
            name = r.get("agent", "unknown")
            if "error" in r:
                sections.append(f"### {name}\n[ERROR] {r['error']}")
            else:
                body = {k: v for k, v in r.items() if k != "agent"}
                sections.append(f"### {name}\n{json.dumps(body, indent=2, default=str)}")

        user_content = (
            f"Ticker under review: {symbol}\n\n"
            "Specialist reports:\n\n" + "\n\n".join(sections) +
            "\n\nProduce the unified chief-strategist recommendation."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=[
                {
                    "type": "text",
                    "text": SYNTHESIS_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        return "".join(b.text for b in response.content if b.type == "text").strip()
