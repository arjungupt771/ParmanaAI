"""
ParmanaAI - Intelligent Section-Aware Clinical Chunker

Chunks documents while respecting section headers, preserving paragraph
boundaries, preventing separation of numerical claims from their context,
and producing deterministic chunk identifiers.
"""

import re
from typing import Any, Dict, List, Optional

from backend.ingestion.base import Chunk, Document
from backend.agents.numerical_extractor import extract_numerical_mentions


class IntelligentChunker:
    """
    Intelligent medical document chunker.
    """

    def __init__(
        self,
        chunk_size_words: int = 250,
        chunk_overlap_words: int = 40,
        preserve_sections: bool = True,
    ):
        self.chunk_size = chunk_size_words
        self.chunk_overlap = chunk_overlap_words
        self.preserve_sections = preserve_sections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into coherent paragraphs."""
        raw_paras = re.split(r"\n\s*\n", text)
        cleaned = []
        for p in raw_paras:
            s = p.strip()
            if s:
                cleaned.append(s)
        return cleaned

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Produce intelligent chunks from a parsed Document.
        """
        chunks: List[Chunk] = []

        if self.preserve_sections and document.sections:
            # Chunk each section independently so chunk boundaries respect sections
            for section in document.sections:
                sec_title = section.get("section_title", "General")
                sec_content = section.get("content", "")
                page_no = section.get("page_number")
                sec_chunks = self._chunk_text(
                    text=sec_content,
                    doc_id=document.document_id,
                    section_title=sec_title,
                    page_number=page_no,
                    start_idx=len(chunks),
                    document=document,
                )
                chunks.extend(sec_chunks)
        else:
            chunks = self._chunk_text(
                text=document.text,
                doc_id=document.document_id,
                section_title="General",
                page_number=None,
                start_idx=0,
                document=document,
            )

        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        section_title: str,
        page_number: Optional[int],
        start_idx: int,
        document: Document,
    ) -> List[Chunk]:
        """
        Chunk a block of text preserving paragraph boundaries and numerical statements.
        """
        paragraphs = self._split_into_paragraphs(text)
        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        current_paras: List[str] = []
        current_word_count = 0

        for para in paragraphs:
            para_words = para.split()
            para_word_count = len(para_words)

            # If adding this paragraph exceeds target chunk size and we already have content
            if current_word_count + para_word_count > self.chunk_size and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunk_id = f"{doc_id}::chunk-{start_idx + len(chunks)}"
                num_mentions = extract_numerical_mentions(chunk_text)

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=chunk_text,
                    section_title=section_title,
                    page_number=page_number,
                    source_type=document.source_type,
                    topic=document.topic,
                    numerical_mentions=num_mentions,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "dataset_name": document.dataset_name,
                        "version": document.version,
                    },
                    provenance={
                        **document.provenance,
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "section_title": section_title,
                        "source_type": document.source_type,
                    }
                ))

                # Handle overlap: retain trailing paragraphs if they fit overlap budget
                overlap_paras = []
                overlap_count = 0
                for prev_para in reversed(current_paras):
                    prev_len = len(prev_para.split())
                    if overlap_count + prev_len <= self.chunk_overlap:
                        overlap_paras.insert(0, prev_para)
                        overlap_count += prev_len
                    else:
                        break

                current_paras = overlap_paras + [para]
                current_word_count = overlap_count + para_word_count
            else:
                current_paras.append(para)
                current_word_count += para_word_count

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunk_id = f"{doc_id}::chunk-{start_idx + len(chunks)}"
            num_mentions = extract_numerical_mentions(chunk_text)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk_text,
                section_title=section_title,
                page_number=page_number,
                source_type=document.source_type,
                topic=document.topic,
                numerical_mentions=num_mentions,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "dataset_name": document.dataset_name,
                    "version": document.version,
                },
                provenance={
                    **document.provenance,
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "source_type": document.source_type,
                }
            ))

        return chunks
