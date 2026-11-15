# quick_check.py — run once: python quick_check.py
from backend.LLM.groq_client import GroqClient
client = GroqClient()
print("is_available():", client.is_available())
try:
    resp = client.generate("Say hello in one word.")
    print("Success:", resp)
except Exception as e:
    print("Exception:", repr(e))