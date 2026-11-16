"""
ParmanaAI - CSV Clinical Dataset Parser

Parses tabular medical datasets into structured Document objects.
"""

import csv
import os
from typing import Any, Dict, List

from backend.ingestion.base import BaseParser, Document


class CsvDatasetParser(BaseParser):
    """Parser for .csv files."""

    def can_parse(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in {".csv"}

    def parse(self, file_path: str, **kwargs: Any) -> Document:
        docs = self.parse_all(file_path, **kwargs)
        if not docs:
            raise ValueError(f"No valid rows found in {file_path}")
        return docs[0]

    def parse_all(self, file_path: str, **kwargs: Any) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        documents: List[Document] = []
        dataset_name = kwargs.get("dataset_name") or os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                doc_id = row.get("id") or row.get("document_id") or f"{dataset_name}-row-{idx}"
                title = row.get("title") or row.get("condition") or f"Row {idx}"

                # Text can be in 'text', 'abstract', 'description', 'notes', or formatted row
                text = (
                    row.get("text")
                    or row.get("abstract")
                    or row.get("description")
                    or row.get("summary")
                    or row.get("notes")
                )

                if not text:
                    # Construct textual summary from available key-value pairs
                    parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
                    text = ". ".join(parts)

                if not text or not text.strip():
                    continue

                source_type = row.get("source_type") or kwargs.get("source_type") or "observational"
                topic = row.get("topic") or row.get("condition") or kwargs.get("topic") or ""

                doc = Document(
                    document_id=str(doc_id),
                    title=str(title),
                    text=str(text).strip(),
                    source_type=str(source_type),
                    topic=str(topic),
                    author=row.get("author") or kwargs.get("author"),
                    publication_date=row.get("date") or row.get("year") or kwargs.get("publication_date"),
                    source_url=row.get("url") or kwargs.get("source_url"),
                    version=kwargs.get("version", "1.0"),
                    dataset_name=dataset_name,
                    sections=[{"section_title": "Overview", "content": text}],
                    metadata=dict(row),
                    provenance={
                        "origin": "csv_dataset",
                        "file_path": os.path.abspath(file_path),
                        "row_index": idx,
                        "verified": kwargs.get("verified", False),
                    }
                )
                documents.append(doc)

        return documents
