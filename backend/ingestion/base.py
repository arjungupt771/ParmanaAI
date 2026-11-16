"""
ParmanaAI - Knowledge Ingestion Base Schemas and Interfaces

Defines the unified Document, Chunk, and BaseParser abstractions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """
    Standardized internal representation of an ingested clinical document.
    """
    document_id: str
    title: str
    text: str
    source_type: str  # "guideline", "systematic_review", "rct", "observational", "external_unverified"
    topic: str = ""
    author: Optional[str] = None
    publication_date: Optional[str] = None
    source_url: Optional[str] = None
    version: str = "1.0"
    dataset_name: Optional[str] = None
    sections: List[Dict[str, str]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    ingestion_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def content_hash(self) -> str:
        """Deterministic hash of text content for duplicate detection."""
        cleaned = " ".join(self.text.strip().split())
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
            "source_type": self.source_type,
            "topic": self.topic,
            "author": self.author,
            "publication_date": self.publication_date,
            "source_url": self.source_url,
            "version": self.version,
            "dataset_name": self.dataset_name,
            "sections": self.sections,
            "tables": self.tables,
            "references": self.references,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "ingestion_timestamp": self.ingestion_timestamp,
            "content_hash": self.content_hash(),
        }


@dataclass
class Chunk:
    """
    Standardized internal representation of an intelligently chunked passage.
    """
    chunk_id: str
    document_id: str
    text: str
    section_title: str = "General"
    page_number: Optional[int] = None
    source_type: str = "unknown"
    topic: str = ""
    numerical_mentions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "source_type": self.source_type,
            "topic": self.topic,
            "numerical_mentions": self.numerical_mentions,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


class BaseParser(ABC):
    """Abstract document parser interface."""

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if parser handles this file type."""
        pass

    @abstractmethod
    def parse(self, file_path: str, **kwargs: Any) -> Document:
        """Extract content into standardized Document representation."""
        pass
