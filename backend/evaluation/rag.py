import requests
from backend.evaluation.load_pubmedqa import load_pubmedqa
from backend.orchestrator.mode_classifier import classify_query
from backend.retrieval.guidelines_retriever import retrieve_guideline_evidence
from backend.retrieval.pubmed_abstract_retriever import retrieve_pubmed_abstracts
from backend.agents.extractor import extract_claims
from backend.agents.verifier import verify_claim, compute_hallucination_score
from backend.agents.generation_agent import GenerationAgent

API_URL = "http://127.0.0.1:8000/ask"

def evaluate_rag(dataset_path, limit=200):
    """
    Evaluate RAG (Retrieval-Augmented Generation) hallucination on PubMedQA dataset.
    RAG retrieves evidence and generates answers, then computes hallucination scores.
    """
    data = load_pubmedqa(dataset_path, limit=limit)
    generator = GenerationAgent()
    results = []

    processed_questions = 0
    skipped_questions = 0

    for sample in data:
        question = sample.get("question", "")
        if not question:
            continue

        # Classify mode
        mode = classify_query(question)

        # Retrieve evidence
        if mode == "FACTUAL":
            evidence = retrieve_guideline_evidence(question)
        else:
            evidence = retrieve_pubmed_abstracts(question)

        if not evidence:
            evidence = []  # Do not skip, use empty evidence

        # Build RAG prompt (simpler than the full pipeline)
        evidence_text = "\n\n".join(evidence)
        rag_prompt = f"""
Use the following information to answer the question accurately.

EVIDENCE:
{evidence_text}

QUESTION: {question}

Answer concisely.
"""

        # Generate answer
        generation_output = generator.generate(rag_prompt, strict=False)
        answer = generation_output.get("answer", "").strip()

        if not answer:
            skipped_questions += 1
            continue

        processed_questions += 1

        # Extract claims
        claims = extract_claims(answer, inferential=(mode == "INFERENTIAL"))

        # Prepare evidence for verification
        evidence_list = [
            {
                "source_type": "peer_reviewed",
                "text": chunk,
                "match_type": "semantic",
            }
            for chunk in evidence
        ]

        # Verify claims (same as full pipeline for fair comparison)
        verified_claims = []
        for claim in claims:
            result = verify_claim(claim, evidence_list)
            verified_claims.append(result)

        # Compute hallucination score
        hallucination_score = compute_hallucination_score(verified_claims)

        results.append({
            "question": question,
            "label": sample.get("label"),
            "mode": mode,
            "answer": answer,
            "claims": claims,
            "verified_claims": verified_claims,
            "num_claims": len(claims),
            "hallucination_score": hallucination_score,
            "num_evidence_chunks": len(evidence)
        })

    return results, processed_questions, skipped_questions

if __name__ == "__main__":
    results, processed_questions, skipped_questions = evaluate_rag("backend/evaluation/pubmedqa.json", limit=500)

    import json
    with open("backend/evaluation/rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} RAG results to backend/evaluation/rag_results.json")

    # Compute claim-weighted average hallucination score
    total_score = 0.0
    total_claims = 0
    for r in results:
        num_claims = r.get("num_claims", 0)
        total_score += r.get("hallucination_score", 0) * num_claims
        total_claims += num_claims

    avg_score = total_score / max(total_claims, 1)

    print(f"Average RAG hallucination score: {avg_score:.4f}")
    print(f"Total claims evaluated: {total_claims}")

    print(f"Questions processed: {processed_questions}")
    print(f"Questions skipped: {skipped_questions}")

    # Final metric output
    avg_claims_per_sample = total_claims / max(len(results), 1)
    print("\n==== RAG BASELINE METRICS ====")
    print(f"Samples evaluated     : {len(results)}")
    print(f"Total claims          : {total_claims}")
    print(f"Average hallucination : {avg_score:.4f}")
    print(f"Average claims/sample : {avg_claims_per_sample:.2f}")