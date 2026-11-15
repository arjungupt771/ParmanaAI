import json

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def load_pubmedqa(path, limit=None):
    """Loads the labeled PubMedQA (pqal) dataset.

    Args:
        path: Path to the JSON file.
        limit: Optional maximum number of samples to load. If None, loads all.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []

    items = list(raw_data.items())
    if limit is not None:
        items = items[:limit]

    for _, item in items:
        samples.append({
            "question": item["QUESTION"],
            "context_chunks": chunk_text(" ".join(item["CONTEXTS"])),
            "label": item["final_decision"]
        })

    return samples
