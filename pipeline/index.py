"""
Build and query ChromaDB vector index.
Each expert domain gets its own ChromaDB collection.
"""

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from pipeline.extract import iter_case_chunks

CHROMA_DIR = "chroma_db"

_embedding_fn = None
_chroma_client = None

DOMAINS = [
    "drug_offences",
    "sexual_offences",
    "violent_crimes",
    "property_financial",
    "sentencing",
    "criminal_procedure",
    "regulatory",
    "general",
]


def get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = ONNXMiniLM_L6_V2()
    return _embedding_fn


def get_collection(domain: str, client=None, embedding_fn=None):
    if client is None:
        client = get_client()
    if embedding_fn is None:
        embedding_fn = get_embedding_fn()
    return client.get_or_create_collection(
        name=domain,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(csv_path: str = "dataset.csv", batch_size: int = 100):
    """
    Extract text from all PDFs, embed, and store in ChromaDB by domain.
    Safe to re-run — skips already-indexed chunks.
    """
    client = get_client()
    embedding_fn = get_embedding_fn()
    collections = {d: get_collection(d, client, embedding_fn) for d in DOMAINS}

    # Track existing IDs per collection to avoid duplicates
    existing = {d: set(collections[d].get()["ids"]) for d in DOMAINS}

    buffer = {d: {"ids": [], "docs": [], "metas": []} for d in DOMAINS}

    print("Building index...")
    total = 0

    for chunk in iter_case_chunks(csv_path):
        domain = chunk["domain"]
        chunk_id = chunk["chunk_id"]

        if chunk_id in existing[domain]:
            continue

        buf = buffer[domain]
        buf["ids"].append(chunk_id)
        buf["docs"].append(chunk["text"])
        buf["metas"].append({
            "filename": chunk["filename"],
            "citation": chunk["citation"],
            "case_name": chunk["case_name"],
            "area_of_law": chunk["area_of_law"],
            "topic": chunk["topic"],
            "subtopic": chunk["subtopic"],
            "primary_statute": chunk["primary_statute"],
            "domain": domain,
        })

        if len(buf["ids"]) >= batch_size:
            collections[domain].add(
                ids=buf["ids"],
                documents=buf["docs"],
                metadatas=buf["metas"],
            )
            total += len(buf["ids"])
            print(f"  Indexed {len(buf['ids'])} chunks → {domain} (total: {total})")
            buffer[domain] = {"ids": [], "docs": [], "metas": []}

    # Flush remaining
    for domain, buf in buffer.items():
        if buf["ids"]:
            collections[domain].add(
                ids=buf["ids"],
                documents=buf["docs"],
                metadatas=buf["metas"],
            )
            total += len(buf["ids"])
            print(f"  Indexed {len(buf['ids'])} chunks → {domain} (total: {total})")

    print(f"\nDone. Total chunks indexed: {total}")

    for domain in DOMAINS:
        count = collections[domain].count()
        if count > 0:
            print(f"  {domain}: {count} chunks")


def retrieve(query: str, domain: str, n_results: int = 5) -> list[dict]:
    """
    Retrieve top-n relevant chunks from a domain collection.
    Returns list of dicts with text and metadata.
    """
    collection = get_collection(domain)
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["documents"] or not results["documents"][0]:
        return chunks

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "citation": meta.get("citation", ""),
            "case_name": meta.get("case_name", ""),
            "topic": meta.get("topic", ""),
            "subtopic": meta.get("subtopic", ""),
            "primary_statute": meta.get("primary_statute", ""),
            "relevance_score": round(1 - dist, 3),
        })

    return chunks


def retrieve_multi_domain(query: str, domains: list[str], n_per_domain: int = 3) -> dict[str, list[dict]]:
    """Retrieve from multiple domain collections."""
    return {d: retrieve(query, d, n_per_domain) for d in domains}
