"""
ParmanaAI - Dense Vector Retriever

Wraps ChromaDB vector storage and embedding models, standardizing
output schema and preserving full provenance.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from backend.retrieval.base import BaseRetriever, RetrievalConfig


logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense semantic retriever using ChromaDB and sentence embeddings.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self._collection = None
        self._model = None

    def _init_chroma(self):
        if self._collection is not None:
            return

        try:
            import chromadb
            from backend.utils.embeddings import get_model
            from backend.data.build_vector_index import VECTOR_STORE_PATH, COLLECTION_NAME

            if not os.path.isdir(VECTOR_STORE_PATH):
                logger.warning("Vector store directory not found at %s", VECTOR_STORE_PATH)
                return

            client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
            self._collection = client.get_collection(COLLECTION_NAME)
            self._model = get_model()
        except Exception as exc:
            logger.warning("Dense retriever could not initialize ChromaDB: %s", exc)

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Execute dense semantic retrieval.
        """
        cfg = config or self.config
        top_k = kwargs.get("top_k", cfg.top_k)
        threshold = kwargs.get("threshold", cfg.threshold)
        source_filter = kwargs.get("source_filter", cfg.source_filter)

        self._init_chroma()
        if self._collection is None or self._model is None:
            return []

        try:
            query_embedding = self._model.encode([query]).tolist()
            where = {"source_type": source_filter} if source_filter else None

            if where:
                results = self._collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)
            else:
                results = self._collection.query(query_embeddings=query_embedding, n_results=top_k)

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            chunk_ids = results.get("ids", [[]])[0]

            if not documents:
                return []

            evidence = []
            for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, chunk_ids):
                similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                if similarity < threshold:
                    continue

                if similarity > 0.75:
                    match_type = "exact"
                elif similarity > 0.5:
                    match_type = "semantic"
                else:
                    match_type = "weak"

                doc_id = meta.get("doc_id") or meta.get("document_id")
                evidence.append({
                    "evidence_id": f"dense-{chunk_id}",
                    "text": doc,
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "source_type": meta.get("source_type", "unknown"),
                    "topic": meta.get("topic", ""),
                    "match_type": match_type,
                    "score": round(similarity, 4),
                    "retrieval_metadata": {
                        "strategy": "dense_vector",
                        "cosine_similarity": round(similarity, 4),
                        "distance": round(float(dist), 4),
                    },
                    "provenance": {
                        "origin": "chroma_vector_store",
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "source_type": meta.get("source_type", "unknown"),
                        "verified": False,
                    }
                })

            return evidence
        except Exception as exc:
            logger.exception("Error executing dense retrieval: %s", exc)
            return []
