# quick_check2.py
from dotenv import load_dotenv, find_dotenv
found = find_dotenv()
print("`.env` found at:", found)
load_dotenv(found)

import os
print("GROQ_API_KEY present:", os.getenv("GROQ_API_KEY") is not None)

from backend.LLM.groq_client import GroqClient
client = GroqClient()
print("is_available():", client.is_available())
resp = client.generate("Say hello in one word.")
print("Success:", resp)