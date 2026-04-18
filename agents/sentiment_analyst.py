import json
import re
import anthropic


SYSTEM_PROMPT = """You are a seasoned financial sentiment analyst covering equities and macro news.
You score news headlines for their likely short-term impact on the referenced security.
Scoring rubric (float in [-1, 1]):
  -1.0  extremely bearish / existential threat
  -0.5  clearly negative
   0.0  neutral / mixed / no material signal
  +0.5  clearly positive
  +1.0  extremely bullish / transformational upside
Consider source tone, magnitude, concreteness, and whether the news is already priced in.
You must respond with a single JSON object and nothing else, using exactly these fields:
  "score":    float, aggregate sentiment across all headlines, in [-1, 1]
  "analysis": string, 2-5 sentences summarizing drivers and dispersion across headlines
Do not include markdown fences, commentary, or trailing prose outside the JSON object."""


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _clamp_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score != score:
        return 0.0
    if score < -1.0:
        return -1.0
    if score > 1.0:
        return 1.0
    return score


class SentimentAnalystAgent:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-opus-4-7"):
        self.client = client
        self.model = model

    async def analyze(self, symbol: str, headlines: list[str]) -> dict:
        if not headlines:
            return {
                "agent": "sentiment",
                "headline_count": 0,
                "sentiment_score": 0.0,
                "analysis": "No news available",
            }

        numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
        user_prompt = (
            f"Ticker: {symbol}\n"
            f"Headlines ({len(headlines)}):\n{numbered}\n\n"
            "Score each headline internally, then return the aggregate JSON "
            "object with fields `score` and `analysis` as specified."
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

        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        parsed = _extract_json(raw)
        if parsed is None:
            return {
                "agent": "sentiment",
                "headline_count": len(headlines),
                "sentiment_score": 0.0,
                "analysis": raw,
            }

        score = _clamp_score(parsed.get("score", 0.0))
        analysis = parsed.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            analysis = raw

        return {
            "agent": "sentiment",
            "headline_count": len(headlines),
            "sentiment_score": score,
            "analysis": analysis,
        }
