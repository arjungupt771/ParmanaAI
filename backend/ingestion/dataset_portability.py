"""
ParmanaAI - Dataset Portability and Adapter Framework

Enables onboarding new clinical datasets (e.g. PubMedQA, MedQuAD, institutional guidelines)
without modifying any core retrieval or verification code.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from backend.ingestion.base import Document
from backend.ingestion.parsers.txt_markdown_parser import TxtMarkdownParser
from backend.ingestion.parsers.json_dataset_parser import JsonDatasetParser
from backend.ingestion.parsers.csv_parser import CsvDatasetParser
from backend.ingestion.parsers.pdf_parser import PdfParser
from backend.ingestion.knowledge_base_manager import KnowledgeBaseManager


DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data"
)


class DatasetAdapterRegistry:
    """
    Registry for dataset ingestion adapters.
    """

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.parsers = [
            TxtMarkdownParser(),
            JsonDatasetParser(),
            CsvDatasetParser(),
            PdfParser(),
        ]

    def get_parser_for_file(self, file_path: str):
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def ingest_dataset_file(
        self,
        file_path: str,
        kb_manager: KnowledgeBaseManager,
        dataset_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base using the appropriate parser adapter.
        """
        parser = self.get_parser_for_file(file_path)
        if not parser:
            return {
                "success": False,
                "error": f"No registered adapter can parse file format: {file_path}",
            }

        name = dataset_name or os.path.basename(file_path)

        # Check if parser supports parse_all (JSON, CSV) or parse (TXT, PDF)
        if hasattr(parser, "parse_all"):
            try:
                documents = parser.parse_all(file_path, dataset_name=name, **kwargs)
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        else:
            try:
                documents = [parser.parse(file_path, dataset_name=name, **kwargs)]
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        registered_count = 0
        rejected_count = 0
        details = []

        for doc in documents:
            res = kb_manager.register_document(doc)
            details.append(res)
            if res["success"]:
                registered_count += 1
            else:
                rejected_count += 1

        return {
            "success": registered_count > 0,
            "dataset_name": name,
            "total_documents": len(documents),
            "registered_count": registered_count,
            "rejected_count": rejected_count,
            "details": details,
        }
