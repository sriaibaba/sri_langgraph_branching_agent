"""LangGraph tool-calling agent: Gemini picks 10-K search or weather."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools import get_temperature, query_apple_10k

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM_PROMPT = """You are a tool-calling assistant.

Use query_apple_10k for Apple financials, 10-K facts, revenue, net sales, or fiscal 2025 results.
Use get_temperature for current weather in a city.

Apple's fiscal 2025 ended September 27, 2025. Answer from retrieved 10-K pages when possible and mention the page number.
If a tool result is missing the fact, say you could not find it in the retrieved pages.
Do not invent numbers.
"""


def _disable_tracing_without_langsmith() -> None:
    key = os.getenv("LANGSMITH_API_KEY", "")
    if not key or "your_" in key.lower():
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def build_agent():
    _disable_tracing_without_langsmith()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing from .env")

    model = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    return create_react_agent(
        model,
        tools=[query_apple_10k, get_temperature],
        prompt=SYSTEM_PROMPT,
    )


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def ask_agent(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]
    return _message_text(messages[-1].content)
