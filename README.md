# LangGraph Branching Agent

Tool-calling agent (same idea as Week 2 weather + Yahoo Finance), extended with **Pinecone** so the model can answer from Apple’s FY2025 10-K.

LangGraph is the loop. The model picks a tool from the question. This is **not** a parallel 3-worker + aggregator graph.

## What it does

```text
You type a question
        |
        v
[LangGraph + Gemini]  picks a tool
        |
        +-- query_apple_10k  --> Pinecone (10-K pages)
        +-- get_temperature  --> WeatherAPI
        |
        v
[LangGraph + Gemini]  writes the answer
```

| Question | Tool |
|---|---|
| What is the total revenue for Apple in 2025? | `query_apple_10k` |
| What is the weather in Milpitas? | `get_temperature` |

Pinecone does not store “revenue = $X”. It stores **one vector per 10-K page**. The model reads the retrieved pages and answers.

## Project layout

```text
langgraph_branching_agent/
  .env                 # keys (do not commit)
  pyproject.toml       # uv dependencies
  data/apple-10k-fy2025.pdf
  ingest_10k.py        # page-chunk the PDF and upsert to Pinecone
  pinecone_store.py    # embeddings + search helpers
  tools.py             # query_apple_10k, get_temperature
  agent.py             # LangGraph create_react_agent
  main.py              # interactive chat
```

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Keys in `.env` (this folder only; they are not shared across projects)

```bash
GOOGLE_API_KEY=...          # Gemini
weatherAPIKey=...           # WeatherAPI.com
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=sriaibaba-apple-10k-2025
```

Index `sriaibaba-apple-10k-2025` is 1024-dimensional (cosine, AWS `us-east-1`). Embeddings use Pinecone `llama-text-embed-v2` so the size matches. Do not write to the older Obama-speech index.

## Setup

```bash
cd langgraph_branching_agent
uv sync
```

`uv` creates `.venv` for this project. You do not need to activate it. Use `uv run` so Python uses that environment.

## Load the 10-K (once)

Official Apple FY2025 10-K (fiscal year ended September 27, 2025) lives at `data/apple-10k-fy2025.pdf`. Chunk **by page** and upsert:

```bash
uv run python ingest_10k.py
```

Expected: 80 non-empty pages in `sriaibaba-apple-10k-2025`.

## Run the agent

```bash
uv run python main.py
```

```text
Hi User Ask Question: What is the total revenue for Apple in 2025?
```

Type `exit` to quit.

You may see a `UserWarning` that `gemini-3.5-flash-lite` ignores `temperature`. That is library noise, not the answer. Ignore it.

## Example

Apple FY2025 total net sales from the 10-K: **$416,161 million** ($416.161 billion).

## Git

Do **not** commit `.env` or `.venv`. They are in `.gitignore`.
