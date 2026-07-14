"""
ingest.py
---------
Reads every PDF in ./guidance_docs, splits each into overlapping text
chunks, embeds them with a local sentence-transformers model, and
stores everything in a persistent ChromaDB collection on disk at
./chroma_db.

Run this once whenever you add/change PDFs in guidance_docs/:
    python ingest.py

Re-running it is safe: the collection is cleared and rebuilt each time
(idempotent), so you never end up with duplicate chunks.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "guidance_docs"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "recovery_guidance"

EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs fully on CPU
CHUNK_SIZE_WORDS = 130              # ~1 short paragraph — these are short reference docs,
                                     # so smaller chunks give more precise retrieval
CHUNK_OVERLAP_WORDS = 25            # keeps context continuous across chunks


# ---------------------------------------------------------------------
# 1. Extract text from a PDF, page by page
# ---------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # collapse whitespace/newlines
        if text.strip():
            pages.append((i, text))
    return pages


# ---------------------------------------------------------------------
# 2. Chunk text into overlapping word windows
# ---------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------
# 3. Build documents/ids/metadatas for every PDF
# ---------------------------------------------------------------------
def build_corpus(pdf_dir: Path):
    documents, ids, metadatas = [], [], []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        pages = extract_pdf_text(pdf_path)
        chunk_counter = 0

        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                chunk_counter += 1
                chunk_id = f"{source_name}::p{page_num}::c{chunk_counter}"
                documents.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "source": source_name,
                    "page": page_num,
                    "chunk_index": chunk_counter,
                })

        print(f"  {pdf_path.name}: {chunk_counter} chunks")

    return documents, ids, metadatas


# ---------------------------------------------------------------------
# 4. Main: (re)build the ChromaDB collection
# ---------------------------------------------------------------------
def main():
    print(f"Reading PDFs from: {PDF_DIR}")
    documents, ids, metadatas = build_corpus(PDF_DIR)
    print(f"Total chunks to embed: {len(documents)}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Drop any existing collection so re-running this script is idempotent
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' to rebuild fresh.")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma's add() has a max batch size depending on version/backend;
    # batching in chunks of 100 keeps this safe regardless.
    BATCH = 100
    for i in range(0, len(documents), BATCH):
        collection.add(
            documents=documents[i:i + BATCH],
            ids=ids[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    print(f"Stored at: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
