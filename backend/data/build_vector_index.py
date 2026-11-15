
import os
import re
import json
import shutil

import chromadb
from backend.utils.embeddings import get_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/data
GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")
ABSTRACTS_PATH = os.path.join(BASE_DIR, "pubmed_abstracts.json")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

COLLECTION_NAME = "parmana_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _split_paragraphs(text: str):
    """Split guideline text into paragraph-level chunks."""
    text = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if len(p) > 20]


def _topic_from_filename(filename: str) -> str:
    name = filename.lower()
    name = name.replace("who_", "").replace("icmr_", "")
    name = name.replace("_guidelines.txt", "").replace(".txt", "")
    return name


def load_guideline_chunks():
    chunks, metadatas, ids = [], [], []

    if not os.path.isdir(GUIDELINES_DIR):
        print(f"No guidelines directory found at {GUIDELINES_DIR}")
        return chunks, metadatas, ids

    for filename in sorted(os.listdir(GUIDELINES_DIR)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(GUIDELINES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        topic = _topic_from_filename(filename)
        paragraphs = _split_paragraphs(text)

        for i, para in enumerate(paragraphs):
            chunks.append(para)
            metadatas.append({
                "source_type": "guideline",
                "topic": topic,
                "doc_id": filename,
            })
            ids.append(f"guideline::{filename}::{i}")

    return chunks, metadatas, ids


def load_abstract_chunks():
    chunks, metadatas, ids = [], [], []

    if not os.path.exists(ABSTRACTS_PATH):
        print(f"No abstracts file found at {ABSTRACTS_PATH}")
        return chunks, metadatas, ids

    with open(ABSTRACTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        abstract = item.get("abstract", "").strip()
        title = item.get("title", f"abstract_{i}")

        if not abstract:
            continue

        chunks.append(abstract)
        metadatas.append({
            "source_type": "peer_reviewed",
            "topic": title.lower(),
            "doc_id": title,
        })
        ids.append(f"abstract::{i}")

    return chunks, metadatas, ids


def build_index(reset: bool = True):
    if reset and os.path.isdir(VECTOR_STORE_PATH):
        print(f"Removing existing index at {VECTOR_STORE_PATH}")
        shutil.rmtree(VECTOR_STORE_PATH)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it, ~90MB)")
    model = get_model()

    guideline_chunks, guideline_meta, guideline_ids = load_guideline_chunks()
    abstract_chunks, abstract_meta, abstract_ids = load_abstract_chunks()

    all_chunks = guideline_chunks + abstract_chunks
    all_meta = guideline_meta + abstract_meta
    all_ids = guideline_ids + abstract_ids

    if not all_chunks:
        raise RuntimeError(
            "No source documents found to index. "
            "Check backend/data/guidelines/ and pubmed_abstracts.json."
        )

    print(f"Embedding {len(all_chunks)} chunks "
          f"({len(guideline_chunks)} guideline paragraphs, {len(abstract_chunks)} abstracts)...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=all_chunks,
        metadatas=all_meta,
        ids=all_ids,
        embeddings=embeddings,
    )

    print(f"Indexed {len(all_chunks)} chunks into '{COLLECTION_NAME}' at {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    build_index()