# import os

# class GroqClient:
#     def __init__(self):
#         try:
#             from groq import Groq
#             self._Groq = Groq
#         except Exception:
#             self._Groq = None

#         self.api_key = os.getenv("GROQ_API_KEY")

#     def is_available(self):
#         return self._Groq is not None and self.api_key is not None

#     def generate(self, prompt: str) -> str:
#         if not self.is_available():
#             raise RuntimeError("Groq unavailable")

#         client = self._Groq(api_key=self.api_key)

#         response = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#         )
#         return response.choices[0].message.content


import os
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        try:
            from groq import Groq
            self._Groq = Groq
        except Exception as e:
            logger.error("Failed to import groq SDK: %s", e)
            self._Groq = None

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.error("GROQ_API_KEY not set in environment")

    def is_available(self):
        return self._Groq is not None and self.api_key is not None

    def generate(self, prompt: str) -> str:
        if not self.is_available():
            raise RuntimeError("Groq unavailable")

        client = self._Groq(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Groq generation call failed: %s", e)
            raise