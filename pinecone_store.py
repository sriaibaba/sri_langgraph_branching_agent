"""Shared Pinecone client, 1024-dim embeddings, and 10-K search."""

from __future__ import annotations

import os
from functools import lru_cache

from pinecone import Pinecone

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sriaibaba-apple-10k-2025")
EMBED_MODEL = "llama-text-embed-v2"
EMBED_DIM = 1024


def get_api_key() -> str:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from .env")
    return api_key


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    return Pinecone(api_key=get_api_key())


def get_index():
    return get_pinecone().Index(INDEX_NAME)


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    result = get_pinecone().inference.embed(
        model=EMBED_MODEL,
        inputs=[{"text": text} for text in texts],
        parameters={
            "input_type": input_type,
            "dimension": EMBED_DIM,
            "truncate": "END",
        },
    )
    vectors = []
    for item in result.data:
        values = item.values if hasattr(item, "values") else item["values"]
        if len(values) != EMBED_DIM:
            raise RuntimeError(f"Expected {EMBED_DIM} dims, got {len(values)}")
        vectors.append(list(values))
    return vectors


def search_10k(question: str, top_k: int = 5) -> str:
    vector = embed_texts([question], "query")[0]
    result = get_index().query(vector=vector, top_k=top_k, include_metadata=True)
    matches = result.get("matches") if isinstance(result, dict) else result.matches
    if not matches:
        return "No matching pages found in the Apple 10-K index."

    blocks = []
    for match in matches:
        metadata = match.get("metadata") if isinstance(match, dict) else (match.metadata or {})
        score = match.get("score") if isinstance(match, dict) else match.score
        page = metadata.get("page", "?")
        text = (metadata.get("text") or "").strip()
        blocks.append(f"[page {page} | score {score:.3f}]\n{text}")
    return "\n\n".join(blocks)
