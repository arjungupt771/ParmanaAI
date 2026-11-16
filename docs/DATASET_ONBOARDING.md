# ParmanaAI Dataset Onboarding Guide

ParmanaAI provides a zero-code dataset onboarding interface, allowing healthcare organizations to ingest new clinical guidelines, PubMed literature dumps, and hospital formularies without altering the underlying retrieval or verification pipelines.

---

## 1. Supported Input Formats

| Format | Parser | Features Preserved |
|---|---|---|
| **PDF** | `PDFParser` | Multi-page text, section headers, multi-column layouts, citation markers |
| **TXT / Markdown** | `TxtMarkdownParser` | Markdown `#` heading hierarchies, paragraph boundaries |
| **JSON / JSONL** | `JSONDatasetParser` | PubMedQA (`CONTEXTS`, `QUESTION`, `final_decision`), MedQuAD QA pairs |
| **CSV** | `CSVDatasetParser` | Tabular clinical metrics, laboratory analyte reference intervals |

---

## 2. Onboarding via REST API

New guidelines can be uploaded directly to `/api/v1/ingest`:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "WHO Guidelines on Hypertension: SBP >= 140 mmHg indicates stage 1 hypertension.",
    "document_title": "WHO Hypertension Guidelines 2024",
    "format": "txt",
    "source_type": "guideline",
    "dataset_name": "who_guidelines"
  }'
```

---

## 3. Onboarding via Python Dataset Registry

To register custom structured datasets (e.g. specialized hospital clinical trial protocols):

```python
from backend.ingestion.dataset_portability import get_global_adapter_registry
from backend.ingestion.base import Document

registry = get_global_adapter_registry()

# Register a custom adapter function
@registry.register_adapter("oncology_protocols")
def parse_oncology_data(item: dict) -> Document:
    return Document(
        document_id=f"onco-{item['id']}",
        title=item["protocol_name"],
        text=item["protocol_body"],
        source_type="rct",
        topic="oncology",
        dataset_name="oncology_protocols",
        metadata={"phase": item.get("phase")},
    )
```

---

## 4. Ingestion Lifecycle & Quality Assurance

Every ingested document passes through the **Intelligent Chunking & Trust Pipeline**:
1. **Deduplication Check**: Computes a SHA-256 hash of the normalized text content. Exact duplicates are rejected.
2. **Section-Aware Chunking**: Chunks text at 250 words with a 40-word overlap, strictly respecting section boundaries and avoiding splitting numerical statements from their clinical units.
3. **Evidence Hierarchy Assignment**: Assigns an initial evidentiary weight based on publication type:
   - Rank A: Clinical Practice Guidelines & Systematic Reviews (Weight: 1.0)
   - Rank B: Randomized Controlled Trials (RCTs) (Weight: 0.9)
   - Rank C: Observational Studies & Case Reports (Weight: 0.7)
   - Rank D: Unverified external uploads (Weight: 0.5)
