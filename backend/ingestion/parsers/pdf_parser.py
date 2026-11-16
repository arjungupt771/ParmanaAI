"""
ParmanaAI - PDF Clinical Document Parser

Extracts text, page numbers, section headers, and metadata from clinical PDFs.
Supports pypdf or PyMuPDF if installed, with structured failure recovery.
"""

import os
import re
from typing import Any, Dict, List

from backend.ingestion.base import BaseParser, Document


class PdfParser(BaseParser):
    """Parser for .pdf files."""

    def can_parse(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() == ".pdf"

    def parse(self, file_path: str, **kwargs: Any) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        filename = os.path.basename(file_path)
        doc_id = kwargs.get("document_id") or os.path.splitext(filename)[0]

        extracted_text_pages: List[Dict[str, Any]] = []
        parsing_error: str = ""

        # Strategy 1: Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_text_pages.append({
                        "page_number": page_idx,
                        "text": page_text.strip()
                    })
        except ImportError:
            # Strategy 2: Try fitz (PyMuPDF)
            try:
                import fitz
                doc = fitz.open(file_path)
                for page_idx, page in enumerate(doc, start=1):
                    page_text = page.get_text() or ""
                    if page_text.strip():
                        extracted_text_pages.append({
                            "page_number": page_idx,
                            "text": page_text.strip()
                        })
                doc.close()
            except ImportError:
                parsing_error = "Neither pypdf nor PyMuPDF is installed. PDF cannot be extracted directly."
        except Exception as exc:
            parsing_error = f"Error extracting PDF: {str(exc)}"

        if not extracted_text_pages:
            # Fallback or error record
            if parsing_error:
                raise RuntimeError(parsing_error)
            raise ValueError(f"No readable text could be extracted from {file_path} (may be scanned image).")

        full_text = "\n\n".join([p["text"] for p in extracted_text_pages])

        # Extract title from first page
        first_page_lines = extracted_text_pages[0]["text"].splitlines()
        title = doc_id
        for line in first_page_lines:
            stripped = line.strip()
            if len(stripped) > 5 and len(stripped) < 150:
                title = stripped
                break

        # Extract sections by page or header pattern
        sections: List[Dict[str, str]] = []
        for p in extracted_text_pages:
            sections.append({
                "section_title": f"Page {p['page_number']}",
                "content": p["text"],
                "page_number": p["page_number"]
            })

        source_type = kwargs.get("source_type") or "guideline"
        topic = kwargs.get("topic") or os.path.splitext(filename)[0].lower()

        return Document(
            document_id=doc_id,
            title=title,
            text=full_text,
            source_type=source_type,
            topic=topic,
            author=kwargs.get("author"),
            publication_date=kwargs.get("publication_date"),
            source_url=kwargs.get("source_url"),
            version=kwargs.get("version", "1.0"),
            dataset_name=kwargs.get("dataset_name"),
            sections=sections,
            metadata={"total_pages": len(extracted_text_pages)},
            provenance={
                "origin": "pdf_document",
                "file_path": os.path.abspath(file_path),
                "filename": filename,
                "pages_extracted": len(extracted_text_pages),
                "verified": kwargs.get("verified", False),
            }
        )
