from backend.agents.generation_agent import GenerationAgent
from backend.agents.verifier import verify_claim, compute_hallucination_score
from backend.agents.extractor import extract_claim_records
from backend.agents.refusal_detector import is_refusal
from backend.agents.evidence_normalizer import normalize_evidence
from backend.orchestrator.mode_classifier import classify_query
from backend.retrieval.semantic_retriever import retrieve as semantic_retrieve


def _chunk_text(text: str, chunk_size: int = 300):
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


class Orchestrator:
    def __init__(self):
        self.generator = GenerationAgent()

    def run(self, query, external_context=None):
        return self.run_pipeline(query, external_context=external_context)

    def run_pipeline(self, user_query: str, external_context: str = None):
        mode = classify_query(user_query)

        if external_context and external_context.strip():
            chunks = _chunk_text(external_context.strip())
            evidence_list = [
                {
                    "text": chunk,
                    "source_type": "external_context_unverified",
                    "match_type": "unverified",
                    "score": None,
                    "provenance": {
                        "origin": "external_context",
                        "verified": False,
                        "source_type": "unknown"
                        }
                }
                for chunk in chunks
            ]

        else:
            source_filter = "guideline" if mode == "FACTUAL" else "peer_reviewed"
            evidence_list = semantic_retrieve(user_query, source_filter=source_filter, top_k=3)

            if not evidence_list and mode == "FACTUAL":
                print("No guideline evidence found, falling back to abstracts.")
                evidence_list = semantic_retrieve(user_query, source_filter="peer_reviewed", top_k=3)

        evidence_origin = (
            "external_context"
            if external_context and external_context.strip()
            else "chroma_vector_store"
        )
        evidence_list = normalize_evidence(
            evidence_list or [],
            origin=evidence_origin
        )

        if not evidence_list:
            return {
                "answer": "No verified medical evidence found.",
                "claims": [],
                "hallucination_score": 1,
                "relevance_score": 0.0,
            }

        evidence_list = evidence_list[:3]
        evidence_text = "\n\n".join(e["text"] for e in evidence_list)

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

        generation_output = self.generator.generate(augmented_query, strict=False)
        answer = generation_output.get("answer", "").strip()

        if is_refusal(answer):
            return {
                "answer": "Unable to answer safely.",
                "claims": [],
                "hallucination_score": 0,
                "relevance_score": 0.0,
            }

        claim_records = extract_claim_records(
            answer,
            inferential=(mode == "INFERENTIAL")
        )
        verification = []
        for record in claim_records:
            claim_text = record["claim_text"]
            verification_result = verify_claim(
                claim_text,
                evidence_list
            )
            # Preserve extraction metadata alongside verification.
            verification_result.update({
                "claim_id": record["claim_id"],
                "extraction_type": record["claim_type"],
                "is_inferential": record["is_inferential"],
                "is_definitive": record["is_definitive"],
                "contains_causal_language": record[
                    "contains_causal_language"
                ],
                "contains_numerical_information": record[
                    "contains_numerical_information"
                ],
                "numerical_values": record["numerical_values"],
                "numerical_mentions": record["numerical_mentions"],
                "original_sentence": record["original_sentence"],
                "atomic": record["atomic"]
            })
            verification.append(verification_result)

        hallucination_score = compute_hallucination_score(verification)
        relevance_score = self.generator.relevance_score(user_query, answer)

        return {
            "answer": answer,
            "claims": verification,
            "hallucination_score": hallucination_score,
            "relevance_score": relevance_score,
        }


_orchestrator = Orchestrator()

def run_pipeline(user_query: str, external_context: str = None):
    return _orchestrator.run_pipeline(user_query, external_context=external_context)