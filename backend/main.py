import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "groq":
    assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY NOT FOUND"
else:
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="ParmanaAI Backend",
    description="Multi-Agent Medical Hallucination Reduction System",
    version="1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.on_event("startup")
def startup_event():
    print("ParmanaAI backend started")
    print(f"Using LLM Provider: {provider}")

@app.get("/")
def root():
    return {"status": "ParmanaAI backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}