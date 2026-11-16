"""
ParmanaAI - Phase 7 Knowledge Ingestion & Provenance Tests

Tests:
- TXT, Markdown, JSON, CSV parsing
- Section-aware intelligent chunking
- Source quality assessment and trust management
- Duplicate detection
- Document registration, update, and deletion
- Dataset adapter registry and portability
"""

import json
import os
import tempfile
import pytest

from backend.ingestion.base import Document
from backend.ingestion.parsers.txt_markdown_parser import TxtMarkdownParser
from backend.ingestion.parsers.json_dataset_parser import JsonDatasetParser
from backend.ingestion.parsers.csv_parser import CsvDatasetParser
from backend.ingestion.chunker import IntelligentChunker
from backend.ingestion.quality_trust import TrustManager
from backend.ingestion.knowledge_base_manager import KnowledgeBaseManager
from backend.ingestion.dataset_portability import DatasetAdapterRegistry


def test_txt_markdown_parser_extracts_sections_and_metadata():
    content = """# Clinical Management of Sepsis

## 1. Overview
Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection.

## 2. Fluid Resuscitation
Administer at least 30 mL/kg of IV crystalloid fluid within the first 3 hours.

## References
1. Singer M, et al. Sepsis-3 definitions. JAMA 2016.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    try:
        parser = TxtMarkdownParser()
        assert parser.can_parse(temp_path) is True
        doc = parser.parse(temp_path, topic="sepsis", source_type="guideline")

        assert doc.title == "Clinical Management of Sepsis"
        assert len(doc.sections) >= 3
        assert any(s["section_title"] == "1. Overview" for s in doc.sections)
        assert any(s["section_title"] == "2. Fluid Resuscitation" for s in doc.sections)
        assert len(doc.references) >= 1
    finally:
        os.remove(temp_path)


def test_json_dataset_parser():
    records = [
        {"pmid": "12345", "question": "Does Drug A treat Condition B?", "abstract": "Drug A significantly reduced symptoms.", "topic": "cardiology"},
        {"pmid": "67890", "question": "Does Drug C cause side effects?", "abstract": "No severe side effects observed.", "topic": "neurology"}
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(records, f)
        temp_path = f.name

    try:
        parser = JsonDatasetParser()
        assert parser.can_parse(temp_path) is True
        docs = parser.parse_all(temp_path)
        assert len(docs) == 2
        assert docs[0].document_id == "12345"
        assert "Drug A" in docs[0].text
    finally:
        os.remove(temp_path)


def test_csv_dataset_parser():
    csv_content = "id,condition,summary\n1,Hypertension,Systolic blood pressure >= 140 mmHg.\n2,Diabetes,HbA1c >= 6.5%."
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        parser = CsvDatasetParser()
        assert parser.can_parse(temp_path) is True
        docs = parser.parse_all(temp_path)
        assert len(docs) == 2
        assert docs[0].topic == "Hypertension"
    finally:
        os.remove(temp_path)


def test_intelligent_chunker_preserves_numerical_and_sections():
    doc = Document(
        document_id="doc-test",
        title="Test Document",
        text="Section 1 content with blood pressure 130 mmHg.\n\nSection 2 content with dose 500 mg.",
        source_type="guideline",
        sections=[
            {"section_title": "Section 1", "content": "Section 1 content with blood pressure 130 mmHg."},
            {"section_title": "Section 2", "content": "Section 2 content with dose 500 mg."}
        ]
    )

    chunker = IntelligentChunker(chunk_size_words=100)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].section_title == "Section 1"
    assert chunks[0].chunk_id == "doc-test::chunk-0"
    assert len(chunks[0].numerical_mentions) > 0


def test_trust_manager_and_duplicate_detection():
    trust_mgr = TrustManager(guideline_expiry_years=5)
    doc1 = Document(
        document_id="doc-guideline-1",
        title="Guideline 2018",
        text="Standard clinical guidelines for fever.",
        source_type="guideline",
        publication_date="2018-01-01"
    )

    trust_info = trust_mgr.assess_document_trust(doc1)
    assert trust_info["evidence_level"] == "A"
    assert trust_info["is_potentially_outdated"] is True

    trust_mgr.register_hash(doc1)

    # Identical content in another document
    doc2 = Document(
        document_id="doc-guideline-2",
        title="Copy",
        text="Standard clinical guidelines for fever.",
        source_type="guideline"
    )
    dup = trust_mgr.check_duplicate(doc2)
    assert dup == "doc-guideline-1"


def test_knowledge_base_manager_lifecycle():
    kb = KnowledgeBaseManager()
    doc = Document(
        document_id="doc-kb-1",
        title="Sample Document",
        text="A detailed passage describing patient observation.",
        source_type="peer_reviewed"
    )

    reg_res = kb.register_document(doc)
    assert reg_res["success"] is True
    assert "doc-kb-1" in kb.documents
    assert len(kb.chunks) > 0

    corpus = kb.export_corpus_for_retrieval()
    assert len(corpus) > 0
    assert corpus[0]["document_id"] == "doc-kb-1"

    # Delete
    del_res = kb.delete_document("doc-kb-1")
    assert del_res is True
    assert "doc-kb-1" not in kb.documents
    assert len(kb.chunks) == 0


def test_dataset_adapter_registry_ingestion():
    kb = KnowledgeBaseManager()
    registry = DatasetAdapterRegistry()

    content = "Aspirin reduces myocardial infarction recurrence.\n\nContraindicated in active bleeding."
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    try:
        res = registry.ingest_dataset_file(temp_path, kb, dataset_name="aspirin_study")
        assert res["success"] is True
        assert res["registered_count"] == 1
    finally:
        os.remove(temp_path)
