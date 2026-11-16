
import hashlib

from backend.agents.generation_agent import GenerationAgent
from backend.agents.verifier import (
    verify_claim,
    compute_hallucination_score,
)
from backend.agents.extractor import extract_claim_records
from backend.agents.refusal_detector import is_refusal
from backend.agents.evidence_normalizer import normalize_evidence
from backend.orchestrator.mode_classifier import classify_query
from backend.retrieval.semantic_retriever import retrieve as semantic_retrieve
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.reranking.evidence_selector import EvidenceSelector


def _chunk_text(text: str, chunk_size: int = 300):
    words = text.split()

    if not words:
        return []

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def _ensure_evidence_id(evidence: dict) -> str:
    """
    Return an existing normalized evidence ID or generate
    a deterministic fallback ID.
    """

    existing_id = evidence.get("evidence_id")

    if existing_id:
        return str(existing_id)

    provenance = evidence.get("provenance") or {}

    if not isinstance(provenance, dict):
        provenance = {}

    identity = "|".join([
        str(evidence.get("source_type", "unknown")),
        str(
            evidence.get("document_id")
            or provenance.get("document_id")
            or ""
        ),
        str(
            evidence.get("chunk_id")
            or provenance.get("chunk_id")
            or ""
        ),
        str(evidence.get("text", "")),
    ])

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return f"ev-{digest[:20]}"


def _serialize_evidence(evidence: dict) -> dict:
    """
    Produce a frontend-safe evidence record.

    Retrieval metadata is preserved. This function does not
    claim that retrieved evidence supports a medical claim.
    """

    provenance = evidence.get("provenance") or {}

    if not isinstance(provenance, dict):
        provenance = {}

    return {
        "evidence_id": evidence.get("evidence_id"),
        "text": evidence.get("text", ""),
        "document_id": (
            evidence.get("document_id")
            or provenance.get("document_id")
        ),
        "chunk_id": (
            evidence.get("chunk_id")
            or provenance.get("chunk_id")
        ),
        "source_type": evidence.get(
            "source_type",
            "unknown"
        ),
        "topic": evidence.get("topic"),
        "match_type": evidence.get(
            "match_type",
            "unknown"
        ),
        "retrieval_score": evidence.get("score"),
        "rerank_score": evidence.get("rerank_score"),
        "origin": provenance.get(
            "origin",
            "unknown"
        ),
        "provenance": provenance,
        "verified": False,
        "verification_status": "assessed_per_claim",
    }


class Orchestrator:

    def __init__(self):
        self.generator = GenerationAgent()
        self.hybrid_retriever = HybridRetriever()
        self.evidence_selector = EvidenceSelector()

    def run(self, query, external_context=None):
        return self.run_pipeline(
            query,
            external_context=external_context
        )

    def run_pipeline(
        self,
        user_query: str,
        external_context: str = None
    ):
        mode = classify_query(user_query)

        if external_context and external_context.strip():
            chunks = _chunk_text(external_context.strip())

            raw_evidence = [
                {
                    "text": chunk,
                    "source_type": "external_context_unverified",
                    "match_type": "unverified",
                    "score": None,
                    "provenance": {
                        "origin": "external_context",
                        "verified": False,
                        "source_type": "unknown",
                    },
                }
                for chunk in chunks
            ]

        else:
            source_filter = (
                "guideline"
                if mode == "FACTUAL"
                else "peer_reviewed"
            )

            try:
                raw_evidence = self.hybrid_retriever.retrieve(
                    user_query,
                    source_filter=source_filter,
                    top_k=8,
                )
            except Exception:
                raw_evidence = None

            if not raw_evidence and mode == "FACTUAL":
                try:
                    raw_evidence = self.hybrid_retriever.retrieve(
                        user_query,
                        source_filter="peer_reviewed",
                        top_k=8,
                    )
                except Exception:
                    raw_evidence = None

            if not raw_evidence:
                raw_evidence = semantic_retrieve(
                    user_query,
                    source_filter=source_filter,
                    top_k=5,
                )

        evidence_origin = (
            "external_context"
            if external_context and external_context.strip()
            else "chroma_vector_store"
        )

        normalized_candidates = normalize_evidence(
            raw_evidence or [],
            origin=evidence_origin,
        )

        for candidate in normalized_candidates:
            candidate["evidence_id"] = _ensure_evidence_id(
                candidate
            )

        # Phase 6: Cross-Encoder Reranking and Context-Aware Selection
        selection_outcome = self.evidence_selector.select(
            query=user_query,
            candidates=normalized_candidates,
        )

        evidence_list = selection_outcome.selected_evidence

        if not evidence_list:
            return {
                "answer": "No verified medical evidence found.",
                "claims": [],
                "evidence": [],
                "hallucination_score": 1.0,
                "relevance_score": 0.0,
            }

        evidence_text = "\n\n".join(
            evidence["text"]
            for evidence in evidence_list
        )

        if mode == "FACTUAL":
            augmented_query = f"""
You are a medical assistant.
Use ONLY the guideline information below.
Do NOT add external knowledge.

GUIDELINES:
{evidence_text}

QUESTION:
{user_query}

Answer in clear bullet points.
"""
        else:
            augmented_query = f"""
You are a medical research assistant.
Summarize evidence cautiously from the abstracts below.
Use phrases like "evidence suggests", "is associated with", "may increase risk".
Do NOT define diseases.

ABSTRACTS:
{evidence_text}

QUESTION:
{user_query}

Answer in 2-3 sentences.
"""

        generation_output = self.generator.generate(
            augmented_query,
            strict=False,
        )

        answer = generation_output.get(
            "answer",
            ""
        ).strip()

        if is_refusal(answer):
            return {
                "answer": "Unable to answer safely.",
                "claims": [],
                "evidence": [],
                "hallucination_score": 0.0,
                "relevance_score": 0.0,
            }

        claim_records = extract_claim_records(
            answer,
            inferential=(mode == "INFERENTIAL"),
        )

        verification = []

        for record in claim_records:
            claim_text = record["claim_text"]

            verification_result = verify_claim(
                claim_text,
                evidence_list,
            )

            # Preserve claim extraction metadata.
            verification_result.update({
                "claim_id": record["claim_id"],
                "extraction_type": record["claim_type"],
                "is_inferential": record["is_inferential"],
                "is_definitive": record["is_definitive"],
                "contains_causal_language": record[
                    "contains_causal_language"
                ],
                "contains_negation": record.get("contains_negation", False),
                "contains_comparative_language": record.get(
                    "contains_comparative_language", False
                ),
                "relationship_type": record.get("relationship_type", "factual"),
                "contains_numerical_information": record[
                    "contains_numerical_information"
                ],
                "numerical_values": record["numerical_values"],
                "numerical_mentions": record["numerical_mentions"],
                "original_sentence": record["original_sentence"],
                "atomic": record["atomic"],
            })

            verification.append(verification_result)

        hallucination_score = compute_hallucination_score(
            verification
        )

        relevance_score = self.generator.relevance_score(
            user_query,
            answer,
        )

        # Response-level retrieval trace.
        evidence_records = [
            _serialize_evidence(evidence)
            for evidence in evidence_list
        ]

        return {
            "answer": answer,
            "claims": verification,
            "evidence": evidence_records,
            "hallucination_score": hallucination_score,
            "relevance_score": relevance_score,
        }


_orchestrator = Orchestrator()


def run_pipeline(
    user_query: str,
    external_context: str = None
):
    return _orchestrator.run_pipeline(
        user_query,
        external_context=external_context,
    )
