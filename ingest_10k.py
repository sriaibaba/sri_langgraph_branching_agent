"""Load Apple FY2025 10-K, chunk by page, embed at 1024 dims, upsert to Pinecone.

Follows langchain-rag (PyPDFLoader = one Document per page) and Pinecone
skills (same 1024-dim llama-text-embed-v2 for upsert and later query).
Assignment requires page chunks, not RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

from pinecone_store import INDEX_NAME, embed_texts, get_index

PDF_PATH = Path(__file__).parent / "data" / "apple-10k-fy2025.pdf"
BATCH_SIZE = 20
METADATA_TEXT_LIMIT = 35000


def load_pages(pdf_path: Path) -> list[dict]:
    docs = PyPDFLoader(str(pdf_path)).load()
    pages: list[dict] = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        page_num = int(doc.metadata.get("page", 0)) + 1
        pages.append({"page": page_num, "text": text})
    return pages


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    pages = load_pages(PDF_PATH)
    print(f"loaded {len(pages)} non-empty pages from {PDF_PATH.name}")

    index = get_index()
    upserted = 0
    for start in range(0, len(pages), BATCH_SIZE):
        batch = pages[start : start + BATCH_SIZE]
        vectors = embed_texts([item["text"] for item in batch], "passage")
        records = []
        for item, values in zip(batch, vectors):
            text = item["text"]
            if len(text) > METADATA_TEXT_LIMIT:
                text = text[:METADATA_TEXT_LIMIT]
            records.append(
                {
                    "id": f"apple-10k-fy2025-page-{item['page']}",
                    "values": values,
                    "metadata": {
                        "source": "apple-10k-fy2025.pdf",
                        "page": item["page"],
                        "text": text,
                    },
                }
            )
        index.upsert(vectors=records)
        upserted += len(records)
        print(f"upserted pages {batch[0]['page']}-{batch[-1]['page']} ({upserted} total)")

    stats = index.describe_index_stats()
    print(f"index={INDEX_NAME}")
    print(f"vector_count={stats.get('total_vector_count', stats)}")


if __name__ == "__main__":
    main()
