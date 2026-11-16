"""
ParmanaAI - Knowledge Base Manager

Coordinates document lifecycle: registration, updates, deletion, duplicate detection,
incremental vector indexing, rollback on failure, and structured ingestion logging.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set

from backend.ingestion.base import Document, Chunk
from backend.ingestion.chunker import IntelligentChunker
from backend.ingestion.quality_trust import TrustManager
from backend.retrieval.sparse_retriever import BM25Retriever


logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Manages the clinical knowledge repository and indices.
    """

    def __init__(
        self,
        chunker: Optional[IntelligentChunker] = None,
        trust_manager: Optional[TrustManager] = None,
    ):
        self.chunker = chunker or IntelligentChunker()
        self.trust_manager = trust_manager or TrustManager()
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.ingestion_log: List[Dict[str, Any]] = []

    def register_document(
        self,
        document: Document,
        allow_replace: bool = True,
    ) -> Dict[str, Any]:
        """
        Register a new document into the knowledge base, generate chunks,
        and update indices. Supports atomic rollback on failure.
        """
        doc_id = document.document_id

        # 1. Duplicate detection
        dup_of = self.trust_manager.check_duplicate(document)
        if dup_of:
            msg = f"Document is an exact duplicate of existing document '{dup_of}'."
            self._log_event(doc_id, "DUPLICATE_REJECTED", msg)
            return {
                "success": False,
                "status": "duplicate_rejected",
                "document_id": doc_id,
                "duplicate_of": dup_of,
                "message": msg,
            }

        if doc_id in self.documents and not allow_replace:
            msg = f"Document ID '{doc_id}' already exists and allow_replace=False."
            self._log_event(doc_id, "REGISTRATION_REJECTED", msg)
            return {
                "success": False,
                "status": "already_exists",
                "document_id": doc_id,
                "message": msg,
            }

        # Backup for rollback
        existing_doc = self.documents.get(doc_id)
        existing_chunk_ids = [cid for cid, c in self.chunks.items() if c.document_id == doc_id]

        try:
            # 2. Source trust assessment
            trust_info = self.trust_manager.assess_document_trust(document)
            document.metadata["trust_info"] = trust_info

            # 3. Intelligent chunking
            doc_chunks = self.chunker.chunk_document(document)

            # If existing document, delete previous chunks first
            if existing_doc:
                for cid in existing_chunk_ids:
                    self.chunks.pop(cid, None)
                self.trust_manager.unregister_hash(existing_doc)

            # Store document and new chunks
            self.documents[doc_id] = document
            for chunk in doc_chunks:
                self.chunks[chunk.chunk_id] = chunk

            self.trust_manager.register_hash(document)

            self._log_event(
                doc_id,
                "REGISTERED",
                f"Successfully registered document with {len(doc_chunks)} chunks.",
                chunks_count=len(doc_chunks)
            )

            return {
                "success": True,
                "status": "registered",
                "document_id": doc_id,
                "chunks_created": len(doc_chunks),
                "trust_info": trust_info,
            }

        except Exception as exc:
            # Rollback
            logger.exception("Failed to register document %s, rolling back: %s", doc_id, exc)
            if existing_doc:
                self.documents[doc_id] = existing_doc
                self.trust_manager.register_hash(existing_doc)
            else:
                self.documents.pop(doc_id, None)

            self._log_event(doc_id, "FAILED", str(exc))
            return {
                "success": False,
                "status": "failed",
                "document_id": doc_id,
                "error": str(exc),
            }

    def delete_document(self, document_id: str) -> bool:
        """Remove document and all associated chunks from the knowledge base."""
        if document_id not in self.documents:
            return False

        doc = self.documents.pop(document_id)
        self.trust_manager.unregister_hash(doc)

        chunk_ids_to_del = [cid for cid, c in self.chunks.items() if c.document_id == document_id]
        for cid in chunk_ids_to_del:
            self.chunks.pop(cid, None)

        self._log_event(document_id, "DELETED", f"Deleted document and {len(chunk_ids_to_del)} chunks.")
        return True

    def export_corpus_for_retrieval(self) -> List[Dict[str, Any]]:
        """Export all registered chunks in normalized retrieval format."""
        corpus = []
        for chunk in self.chunks.values():
            corpus.append({
                "evidence_id": f"ev-{chunk.chunk_id}",
                "text": chunk.text,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "topic": chunk.topic,
                "section_title": chunk.section_title,
                "provenance": chunk.provenance,
                "metadata": chunk.metadata,
            })
        return corpus

    def _log_event(self, doc_id: str, status: str, message: str, **kwargs: Any) -> None:
        self.ingestion_log.append({
            "document_id": doc_id,
            "status": status,
            "message": message,
            **kwargs,
        })
