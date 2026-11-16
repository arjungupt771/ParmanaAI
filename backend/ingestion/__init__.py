"""
ParmanaAI - Knowledge Ingestion Module
"""

from backend.ingestion.base import Document, Chunk, BaseParser
from backend.ingestion.chunker import IntelligentChunker
from backend.ingestion.quality_trust import TrustManager
from backend.ingestion.knowledge_base_manager import KnowledgeBaseManager
from backend.ingestion.dataset_portability import DatasetAdapterRegistry
from backend.ingestion.parsers.txt_markdown_parser import TxtMarkdownParser
from backend.ingestion.parsers.json_dataset_parser import JsonDatasetParser
from backend.ingestion.parsers.csv_parser import CsvDatasetParser
from backend.ingestion.parsers.pdf_parser import PdfParser

__all__ = [
    "Document",
    "Chunk",
    "BaseParser",
    "IntelligentChunker",
    "TrustManager",
    "KnowledgeBaseManager",
    "DatasetAdapterRegistry",
    "TxtMarkdownParser",
    "JsonDatasetParser",
    "CsvDatasetParser",
    "PdfParser",
]
