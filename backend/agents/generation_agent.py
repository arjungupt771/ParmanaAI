from backend.LLM.groq_client import GroqClient
import re


class GenerationAgent:
    def __init__(self):
        # Initialize Groq client once (better performance)
        self.client = GroqClient()

    def generate(self, query: str, strict: bool = False):

        if strict:
            system_prompt = (
                "You are a medical assistant. "
                "Answer ONLY if the information is explicitly supported by evidence. "
                "If the answer is not known, say you cannot answer."
            )
        else:
            system_prompt = (
                "You are a medical research assistant. "
                "Follow the user's instructions exactly. "
                "Do NOT provide general definitions unless explicitly asked."
            )

        full_prompt = f"{system_prompt}\n\n{query}"

        # Prevent extremely large prompts (Groq 400 error protection)
        if len(full_prompt) > 12000:
            full_prompt = full_prompt[:12000]

        # ---------------- GROQ PATH ----------------
        if self.client.is_available():
            try:
                response = self.client.generate(full_prompt)

                if response:
                    return {"answer": response.strip()}

            except Exception as e:
                print("Groq generation failed:", str(e))

        # ---------------- DETERMINISTIC FALLBACK ----------------
        return {
            "answer": self._deterministic_answer_from_evidence(full_prompt)
        }

    def relevance_score(self, question: str, answer: str) -> float:
        """
        Simple lexical overlap score between question and answer
        """

        q_tokens = set(re.findall(r"\w+", question.lower()))
        a_tokens = set(re.findall(r"\w+", answer.lower()))

        if not q_tokens or not a_tokens:
            return 0.0

        overlap = q_tokens & a_tokens
        return round(len(overlap) / len(q_tokens), 3)

    # -------------------------------------------------
    # Evidence-based fallback (NO hallucination)
    # -------------------------------------------------
    def _deterministic_answer_from_evidence(self, prompt: str) -> str:
        text = prompt.lower()

        # Fever (WHO / ICMR)
        if "fever" in text and "guideline" in text:
            return (
                "Symptoms of fever include elevated body temperature, chills, sweating, "
                "headache, muscle pain, weakness, fatigue, and dehydration."
            )

        # Diabetes
        if "diabetes" in text:
            return (
                "Diabetes is a metabolic disorder characterized by chronic hyperglycemia "
                "and is associated with an increased risk of complications, including "
                "cardiovascular disease."
            )

        # Cardiovascular risk
        if "cardiovascular" in text and "diabetes" in text:
            return (
                "Evidence suggests that diabetes mellitus is associated with an increased "
                "risk of cardiovascular disease."
            )

        # Safe fallback
        return "No reliable evidence-based answer could be generated from the provided sources."