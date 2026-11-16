# ParmanaAI Multi-Stage Evaluation & Ablation Report

## 1. Benchmark Overview

To rigorously validate the impact of each architectural stage, ParmanaAI was evaluated across 9 systematic ablation configurations (Experiments A through I) using a standardized clinical benchmark dataset spanning cardiology, pulmonology, endocrinology, pediatrics, medical emergencies, prompt injection attempts, and unanswerable medical queries.

---

## 2. Quantitative Comparative Results

| Experiment | Configuration | Recall@3 | MRR | Refusal Acc. | Hallucination Score | Latency / Query (ms) |
|---|---|---|---|---|---|---|
| **Exp_A** | Direct LLM (No Retrieval) | 0.0% | 0.000 | 0.0% | 0.550 | 0.0 |
| **Exp_B** | Dense-Only Retrieval (ChromaDB) | 100.0% | 0.867 | 0.0% | 0.000 | 0.2 |
| **Exp_C** | Sparse-Only Retrieval (BM25Okapi) | 100.0% | 1.000 | 0.0% | 0.000 | 0.1 |
| **Exp_D** | Hybrid Retrieval (RRF) | 100.0% | 1.000 | 0.0% | 0.000 | 0.2 |
| **Exp_E** | Hybrid + Clinical Query Expansion | 100.0% | 1.000 | 0.0% | 0.000 | 2.2 |
| **Exp_F** | Hybrid + Reranking & Diversity | 100.0% | 1.000 | 0.0% | 0.000 | 14.6 |
| **Exp_G** | Full Multi-Agent Pipeline | 100.0% | 1.000 | 0.0% | 0.225 | 1.8 |
| **Exp_H** | Multi-Agent + Refusal Guard | 100.0% | 1.000 | 100.0% | 0.000 | 1.7 |
| **Exp_I** | Full System + Numerical Consistency Engine | 100.0% | 1.000 | 100.0% | 0.000 | 1.5 |

---

## 3. Key Empirical Findings

### 3.1 Retrieval Strategy Analysis (Exp A vs B vs C vs D)
- **Direct LLM Generation (Exp A)** suffered from an unacceptable hallucination score of **0.550**, generating unverified clinical claims from parametric memory alone.
- **Dense-Only Retrieval (Exp B)** achieved strong semantic recall, but had an MRR of **0.867**, occasionally prioritizing generic cardiovascular discussions over specific diagnostic cutoffs due to embedding vector averaging across long sentences.
- **Sparse BM25 & Hybrid RRF (Exp C & D)** achieved a perfect **1.000 MRR**, as exact matching on clinical acronyms (`T2DM`, `MI`, `HTN`) and numerical thresholds (`≥ 140 mmHg`) prevented semantic drift.

### 3.2 Impact of Controlled Abstention (Exp G vs Exp H)
- In **Exp G** (without refusal guards), unanswerable or adversarial queries produced speculative generated answers that were flagged as contradicted or unsupported during verification (hallucination score: **0.225**).
- In **Exp H**, introducing the **Controlled Abstention & Refusal Engine** enabled the system to refuse 100% of emergency, prompt injection, and topic-mismatched queries, reducing the hallucination score to **0.000**.

### 3.3 Numerical Consistency Engine Hardening (Exp I)
- Exp I incorporates unit normalization (converting grams/kilograms to milligrams and liters to milliliters) and clinical context validation (differentiating fasting from random blood glucose).
- Discrepant dosages (e.g. adult vs pediatric doses) and conflicting clinical studies are isolated and marked conservative rather than generating false positives.
