# web_tools.py
from tavily import TavilyClient
import os

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def simple_search(query: str, max_results: int = 5):
    """
    Simple wrapper around Tavily search.

    Returns a list of text blocks (content) for summarization.
    """
    try:
        resp = tavily.search(query=query, max_results=max_results)
        results = resp.get("results", [])

        processed = []
        for r in results:
            # ✅ Tavily provides "content", NOT "snippet"
            text = r.get("content") or r.get("title") or ""
            if text:
                processed.append(text)

        return processed

    except Exception as e:
        print("simple_search failed:", e)
        return []
