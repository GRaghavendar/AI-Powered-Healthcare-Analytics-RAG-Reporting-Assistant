"""Optional DuckDuckGo web search helper for hybrid assistant mode."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


def search_duckduckgo(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """Search DuckDuckGo's HTML endpoint and return lightweight snippets.

    This is optional. The project works without web search, and web search should
    not be used for PHI or sensitive data.
    """

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "Web search dependencies are not installed. Run: pip install requests beautifulsoup4"
        ) from exc

    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[WebSearchResult] = []
    for result in soup.select(".result"):
        link = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if not link:
            continue
        title = " ".join(link.get_text(" ", strip=True).split())
        href = link.get("href", "")
        text = " ".join(snippet.get_text(" ", strip=True).split()) if snippet else ""
        if title and href:
            results.append(WebSearchResult(title=title, url=href, snippet=text))
        if len(results) >= max_results:
            break
    return results
