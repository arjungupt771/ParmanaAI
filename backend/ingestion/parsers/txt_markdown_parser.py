"""
ParmanaAI - TXT and Markdown Clinical Document Parser

Extracts structured sections, headers, references, and metadata from plain
text and Markdown clinical guidelines and medical summaries.
"""

import os
import re
from typing import Any, Dict, List

from backend.ingestion.base import BaseParser, Document


class TxtMarkdownParser(BaseParser):
    """Parser for .txt and .md files."""

    def can_parse(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in {".txt", ".md", ".markdown"}

    def parse(self, file_path: str, **kwargs: Any) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        if not raw_text.strip():
            raise ValueError(f"Document at {file_path} is empty or unreadable.")

        filename = os.path.basename(file_path)
        doc_id = kwargs.get("document_id") or os.path.splitext(filename)[0]

        # Extract title: First markdown heading # or first non-empty line
        title = doc_id
        lines = raw_text.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
            elif stripped and not stripped.startswith("#") and len(stripped) < 120:
                title = stripped
                break

        # Extract sections: Split on markdown headers (#, ##, ###) or ALL-CAPS headers
        sections: List[Dict[str, str]] = []
        current_header = "Introduction"
        current_body: List[str] = []

        header_pattern = re.compile(r"^(?:#{1,4}\s+(.+)|([A-Z0-9\s]{4,}):?)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                header_name = match.group(1) or match.group(2)
                header_name = header_name.strip()
                if current_body:
                    sections.append({
                        "section_title": current_header,
                        "content": "\n".join(current_body).strip()
                    })
                    current_body = []
                current_header = header_name
            else:
                current_body.append(line)

        if current_body:
            sections.append({
                "section_title": current_header,
                "content": "\n".join(current_body).strip()
            })

        # Extract references
        references: List[str] = []
        for section in sections:
            if "reference" in section["section_title"].lower():
                ref_lines = section["content"].splitlines()
                references.extend([l.strip() for l in ref_lines if l.strip()])

        # Infer topic from filename or kwargs
        topic = kwargs.get("topic") or ""
        if not topic:
            clean_name = filename.lower()
            for prefix in ("who_", "icmr_", "cdc_", "aha_"):
                clean_name = clean_name.replace(prefix, "")
            topic = clean_name.replace("_guidelines.txt", "").replace(".txt", "").replace(".md", "")

        source_type = kwargs.get("source_type") or ("guideline" if "guideline" in filename.lower() else "peer_reviewed")

        provenance = {
            "origin": "file_upload",
            "file_path": os.path.abspath(file_path),
            "filename": filename,
            "verified": kwargs.get("verified", source_type in {"guideline", "peer_reviewed"}),
        }

        return Document(
            document_id=doc_id,
            title=title,
            text=raw_text,
            source_type=source_type,
            topic=topic,
            author=kwargs.get("author"),
            publication_date=kwargs.get("publication_date"),
            source_url=kwargs.get("source_url"),
            version=kwargs.get("version", "1.0"),
            dataset_name=kwargs.get("dataset_name"),
            sections=sections,
            references=references,
            metadata=kwargs.get("metadata", {}),
            provenance=provenance,
        )
