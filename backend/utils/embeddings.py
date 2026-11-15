"""
Shared embedding utility. Single lazily-loaded SentenceTransformer model,
reused by both retrieval (semantic_retriever.py) and verification
(verifier.py) so the ~90MB model is only ever loaded once per process.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts):
    """Embed a list of strings. Returns an (n, dim) numpy array."""
    model = get_model()
    return np.array(model.encode(list(texts)))


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)