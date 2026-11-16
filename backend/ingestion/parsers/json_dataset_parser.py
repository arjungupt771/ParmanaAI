"""
ParmanaAI - JSON and JSONL Clinical Dataset Parser

Parses structured medical datasets including PubMedQA, MedQuAD, and guideline corpora.
"""

import json
import os
from typing import Any, Dict, List

from backend.ingestion.base import BaseParser, Document


class JsonDatasetParser(BaseParser):
    """Parser for .json and .jsonl files."""

    def can_parse(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in {".json", ".jsonl"}

    def parse(self, file_path: str, **kwargs: Any) -> Document:
        """
        Parses the first/primary document or returns aggregate document representation.
        Use `parse_all` for multi-record datasets.
        """
        docs = self.parse_all(file_path, **kwargs)
        if not docs:
            raise ValueError(f"No valid records found in {file_path}")
        return docs[0]

    def parse_all(self, file_path: str, **kwargs: Any) -> List[Document]:
        """
        Parse all records from a JSON or JSONL file into a list of Documents.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        records: List[Dict[str, Any]] = []
        ext = os.path.splitext(file_path)[1].lower()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if ext == ".jsonl":
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    # Could be PubMedQA dictionary format or single document
                    if "text" in data or "abstract" in data:
                        records = [data]
                    else:
                        for key, val in data.items():
                            if isinstance(val, dict):
                                val["_record_id"] = str(key)
                                records.append(val)
                            else:
                                records.append({"text": str(val), "_record_id": str(key)})

        documents: List[Document] = []
        dataset_name = kwargs.get("dataset_name") or os.path.basename(file_path)

        for idx, rec in enumerate(records):
            doc_id = str(rec.get("document_id") or rec.get("pmid") or rec.get("_record_id") or f"{dataset_name}-{idx}")
            title = rec.get("title") or rec.get("question") or f"Record {doc_id}"
            
            # Extract main text
            text = ""
            if "text" in rec:
                text = rec["text"]
            elif "abstract" in rec:
                abstract_val = rec["abstract"]
                if isinstance(abstract_val, list):
                    text = " ".join(abstract_val)
                else:
                    text = str(abstract_val)
            elif "content" in rec:
                text = str(rec["content"])
            elif "contexts" in rec and isinstance(rec["contexts"], list):
                text = " ".join(rec["contexts"])
            elif "answer" in rec:
                text = str(rec["answer"])

            if not text or not str(text).strip():
                continue

            text = str(text).strip()
            topic = rec.get("topic") or kwargs.get("topic") or ""
            source_type = rec.get("source_type") or kwargs.get("source_type") or "peer_reviewed"

            provenance = {
                "origin": "json_dataset",
                "file_path": os.path.abspath(file_path),
                "dataset_name": dataset_name,
                "record_index": idx,
                "verified": kwargs.get("verified", source_type in {"guideline", "peer_reviewed"}),
            }

            doc = Document(
                document_id=doc_id,
                title=title,
                text=text,
                source_type=source_type,
                topic=topic,
                author=rec.get("author") or kwargs.get("author"),
                publication_date=rec.get("publication_date") or rec.get("year") or kwargs.get("publication_date"),
                source_url=rec.get("url") or kwargs.get("source_url"),
                version=kwargs.get("version", "1.0"),
                dataset_name=dataset_name,
                sections=[{"section_title": "Abstract", "content": text}],
                metadata={k: v for k, v in rec.items() if k not in {"text", "abstract", "content"}},
                provenance=provenance,
            )
            documents.append(doc)

        return documents
