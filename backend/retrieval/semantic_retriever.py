import os
import chromadb

from backend.utils.embeddings import get_model
from backend.data.build_vector_index import VECTOR_STORE_PATH, COLLECTION_NAME

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        if not os.path.isdir(VECTOR_STORE_PATH):
            raise RuntimeError(
                f"No vector index found at {VECTOR_STORE_PATH}. "
                "Run `python -m backend.data.build_vector_index` first."
            )
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, source_filter: str = None, top_k: int = 3, threshold: float = 0.35):
    """
    Semantic search over the knowledge base.

    Args:
        query: user question
        source_filter: "guideline" | "peer_reviewed" | None (no filter)
        top_k: max number of chunks to return
        threshold: minimum cosine similarity (0-1) to keep a chunk

    Returns:
        List of {"text", "source_type", "topic", "match_type", "score"}
        or None if nothing clears the threshold.
    """
    collection = _get_collection()
    model = get_model()

    query_embedding = model.encode([query]).tolist()
    where = {"source_type": source_filter} if source_filter else None

    if where:
        results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)
    else:
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    chunk_ids = results.get("ids", [[]])[0]

    if not documents:
        return None

    evidence = []
    for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, chunk_ids):
        similarity = max(0.0, min(1.0, 1 - dist))

        if similarity < threshold:
            continue

        if similarity > 0.75:
            match_type = "exact"
        elif similarity > 0.5:
            match_type = "semantic"
        else:
            match_type = "weak"

        evidence.append({
            "text": doc,
            "source_type": meta.get("source_type", "unknown"),
            "topic": meta.get("topic", ""),
            "document_id": meta.get("doc_id"),
            "chunk_id": chunk_id,
            "match_type": match_type,
            "score": round(similarity, 3),
            "provenance": {
                "origin": "chroma_vector_store",
                "document_id": meta.get("doc_id"),
                "chunk_id": chunk_id,
                "source_type": meta.get("source_type", "unknown"),
                "verified": False
            }
        })

    return evidence if evidence else None