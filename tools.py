"""Tools the LangGraph agent can call: Apple 10-K search and weather."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from pinecone_store import search_10k


@tool
def query_apple_10k(question: str) -> str:
    """Search Apple's FY2025 10-K in Pinecone for financial facts such as revenue.

    Use this for questions about Apple's 10-K, annual report, revenue, net sales,
    fiscal 2025 results, or other facts from that filing.
    """
    return search_10k(question)


@tool
def get_temperature(city: str) -> str:
    """Get the current temperature and weather condition for a city."""
    api_key = os.getenv("weatherAPIKey")
    if not api_key or api_key == "your_weather_api_key_here":
        return "Weather API key is missing. Set weatherAPIKey in your .env file."

    url = "https://api.weatherapi.com/v1/current.json?" + urlencode(
        {"key": api_key, "q": city}
    )
    try:
        with urlopen(Request(url, headers={"User-Agent": "langgraph-branching-agent"}), timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as e:
        return f"Weather API error ({e.code}): {e.read().decode(errors='replace')}"
    except URLError as e:
        return f"Weather API request failed: {e.reason}"

    location = data.get("location", {})
    current = data.get("current", {})
    place = ", ".join(
        part
        for part in (
            location.get("name"),
            location.get("region"),
            location.get("country"),
        )
        if part
    )
    temp_f = current.get("temp_f")
    temp_c = current.get("temp_c")
    condition = (current.get("condition") or {}).get("text", "Unknown")
    return f"Current weather in {place}: {temp_f}°F ({temp_c}°C), {condition}."
