import html
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


_USER_AGENT = "Mozilla/5.0 (compatible; StockAnalyzer/1.0)"
_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={symbol}&region=US&lang=en-US"
)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    unescaped = html.unescape(text)
    return _WS_RE.sub(" ", unescaped).strip()


def fetch_headlines(symbol: str, limit: int = 10, timeout: float = 5.0) -> list[str]:
    try:
        url = _RSS_URL.format(symbol=symbol)
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        if not raw:
            return []
        root = ET.fromstring(raw)
        titles: list[str] = []
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is None or title_el.text is None:
                continue
            normalized = _normalize(title_el.text)
            if normalized:
                titles.append(normalized)
            if len(titles) >= limit:
                break
        return titles
    except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError, ValueError, OSError, TimeoutError):
        return []
    except Exception:
        return []


def fetch_with_fallback(
    symbol: str,
    fallback_headlines: list[str],
    limit: int = 10,
) -> tuple[list[str], str]:
    headlines = fetch_headlines(symbol, limit=limit)
    if headlines:
        return headlines, "yahoo_rss"
    return list(fallback_headlines)[:limit], "fallback"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python news_fetcher.py <SYMBOL>", file=sys.stderr)
        raise SystemExit(2)
    sym = sys.argv[1]
    results = fetch_headlines(sym)
    print(f"Fetched {len(results)} headlines for {sym}:")
    for i, headline in enumerate(results, 1):
        print(f"  {i}. {headline}")
