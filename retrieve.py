"""
retrieve.py
-----------
Query-time retrieval against the ChromaDB collection built by ingest.py.

Usage from your Streamlit app / orchestration function:

    from retrieve import retrieve_guidance

    chunks = retrieve_guidance("should I train legs when HRV is low?", k=3)
    for c in chunks:
        print(c["source"], c["page"], c["distance"])
        print(c["text"])

Or run this file directly for a quick interactive test:

    python retrieve.py
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "recovery_guidance"
EMBED_MODEL = "all-MiniLM-L6-v2"

_client = None
_collection = None


def _get_collection():
    """Lazily initialize the Chroma client/collection (loaded once per process)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def retrieve_guidance(query: str, k: int = 3, source_filter: str | None = None) -> list[dict]:
    """
    Retrieve the top-k most relevant guidance chunks for a query.

    Args:
        query: natural-language question or a canned string like
               "low recovery workout modification guidance"
        k: number of chunks to return
        source_filter: optional exact PDF stem (e.g. "05_exercise_safety_red_flags")
                        to restrict retrieval to a single guidance document

    Returns:
        List of dicts: {"text", "source", "page", "chunk_index", "distance"}
        sorted by relevance (lowest distance first).
    """
    collection = _get_collection()

    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for text, meta, dist in zip(docs, metas, dists):
        out.append({
            "text": text,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
        })
    return out


if __name__ == "__main__":
    print("Loaded collection. Type a question (or 'quit').\n")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in {"quit", "exit"}:
            break
        results = retrieve_guidance(q, k=3)
        if not results:
            print("No results. Did you run ingest.py first?\n")
            continue
        for r in results:
            print(f"\n[{r['source']} | page {r['page']} | distance {r['distance']:.3f}]")
            print(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
        print()
