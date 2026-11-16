# ParmanaAI: Technical Interview & Portfolio Guide

## 1. Project Motivation & Core Engineering Challenge

### What problem does ParmanaAI solve?
Large Language Models frequently hallucinate medical facts, fabricate clinical statistics, or confuse adult and pediatric dosages. In clinical domains, hallucination is not merely an inconvenience—it can lead to patient harm.

Standard RAG architectures assume that retrieving semantic chunks automatically grounds the LLM. In practice:
1. **Dense embeddings suffer from keyword dilution**: Searching for "metformin 500 mg daily" often retrieves chunks about "metformin 1000 mg twice daily" or other biguanides with near-identical cosine similarities.
2. **LLMs blend retrieved text with parametric memory**: A model can cite a paper while simultaneously inventing a dosage number not present anywhere in the cited text.
3. **NLI models cannot do math**: An NLI model evaluating premise *"Dose is 2 g daily"* against hypothesis *"Dose is 2000 mg daily"* will frequently predict `neutral` or `contradiction` due to lexical divergence, while failing to detect genuine errors like unit confusion (e.g., `mg` vs `mcg`).

**ParmanaAI solves this through a multi-agent verification system combining hybrid retrieval (RRF), cross-encoder reranking, atomic claim extraction, neural NLI entailment, lossless decimal arithmetic, and cross-document conflict aggregation.**

---

## 2. Key Architecture Design Decisions & Trade-Offs

### Q1: Why use Reciprocal Rank Fusion (RRF) instead of linear score combination?
- **Linear combination** requires score calibration. Dense cosine similarity ($[-1, 1]$ or $[0, 1]$) and BM25Okapi scores ($[0, \infty)$) have fundamentally different distributions. Normalizing BM25 by max score is sensitive to outlier chunks with extreme term frequencies.
- **RRF** is non-parametric and rank-based:
  $$\text{RRF}(d) = \sum_{m} \frac{w_m}{k + \text{rank}_m(d)}$$
  It ensures that an exact keyword match in BM25 (e.g. `HbA1c ≥ 6.5%`) cannot be buried by a generic semantic paragraph that happens to have high embedding similarity.

### Q2: Why is NLI alone insufficient for verifying numerical claims?
- Neural NLI models (like DeBERTa-v3) treat numbers as arbitrary token IDs. They do not possess a continuous numerical representation or conversion tables.
- ParmanaAI decouples verification into two orthogonal signals:
  1. **Textual Entailment (DeBERTa)**: Assesses semantic and factual relationship.
  2. **Numerical Verdict Engine**: Converts all measurements to canonical units (`mg`, `ml`, `mmHg`) using Python's `Decimal` module, verifies mathematical constraints (`≥`, `<=`, `=`), and cross-checks clinical context (e.g., confirming fasting blood glucose is not matched against postprandial glucose).
- **Conservative Acceptance Rule**: A claim is marked `SUPPORTED` *only* if NLI predicts entailment AND the numerical engine confirms consistency. Any mismatch downgrades the verdict to `CONTEXT_MISMATCH` or `NUMERICAL_DISCREPANCY`.

### Q3: How does the system handle conflicting medical studies?
- If Document A concludes *"Aspirin 81 mg daily reduces stroke risk"* and Document B concludes *"Aspirin 81 mg daily does not significantly reduce stroke risk in primary prevention"*, a naive RAG system will produce a contradictory or hallucinatory synthesis.
- ParmanaAI tracks provenance per claim. In the `NumericalAggregator`, cross-document disagreements are explicitly labeled as `cross_document_conflict`. The UI displays both studies side-by-side with evidentiary weights rather than synthesizing an artificial consensus.

---

## 3. Failure Mode Analysis & Defenses

| Failure Mode | Failure Mechanism | ParmanaAI Defense |
|---|---|---|
| **Topic Drift** | Query on hypertension retrieves diabetes guidelines due to generic terms ("clinical study", "patient") | `RelevanceGuard`: Checks query condition overlap and drops mismatched topics before generation. |
| **Hallucinatory Extrapolation** | LLM invents facts when evidence is empty | `RefusalEngine`: Controlled abstention returns standardized refusal if 0 evidence chunks meet the relevance threshold. |
| **Unit Confusion** | Evidence states 500 mcg, generated answer states 500 mg (1000x overdose) | `NumericalConsistencyEngine`: Lossless unit conversion identifies factor difference and flags `NUMERICAL_DISCREPANCY`. |
| **Emergency Risk** | Patient asking for home triage during active myocardial infarction | `InputSafetyGuard`: Regular expression triage catches chest pain/stroke FAST signs, bypassing retrieval to provide immediate 911 referral. |
