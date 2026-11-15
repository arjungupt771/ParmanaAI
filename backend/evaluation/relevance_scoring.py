# backend/evaluation/relevance_scoring.py

def compute_relevance_score(llm_client, question: str, answer: str) -> float:
    """
    Returns relevance score in [0, 1]
    """

    prompt = f"""
    Rate how well the answer addresses the question.

    Question:
    {question}

    Answer:
    {answer}

    Give ONLY a number between 0 and 1.
    0 = completely irrelevant
    1 = perfectly relevant
    """

    response = llm_client.generate(prompt)

    try:
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5  # safe default
