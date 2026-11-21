
"""
ParmanaAI - Natural Language Inference Engine

Uses a pretrained DeBERTa NLI model to evaluate:

Premise: Retrieved evidence
Hypothesis: Extracted claim

Output:
- entailment
- contradiction
- neutral

The model is loaded lazily and reused.
"""

import logging
import os
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "PARMANA_NLI_MODEL",
    "cross-encoder/nli-deberta-v3-small"
)

MAX_LENGTH = int(
    os.getenv("PARMANA_NLI_MAX_LENGTH", "512")
)

_model = None
_tokenizer = None
_device = None


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model():
    """
    Load the NLI model only once per process.
    """

    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info("Loading NLI model: %s", MODEL_NAME)

    _device = get_device()

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    _model.to(_device)
    _model.eval()

    logger.info("NLI model loaded on %s", _device)

    return _model, _tokenizer, _device


def normalize_label(label: str) -> str:
    """
    Normalize model label names.
    """

    label = label.lower().strip()

    if "entail" in label:
        return "entailment"

    if "contrad" in label:
        return "contradiction"

    if "neutral" in label:
        return "neutral"

    raise ValueError(
        f"Unrecognized NLI label: {label}"
    )


def get_label_mapping(model) -> dict[int, str]:
    """
    Read the model's actual label configuration.
    """

    mapping = {}

    for index, label in model.config.id2label.items():
        mapping[int(index)] = normalize_label(label)

    if set(mapping.values()) != {
        "entailment",
        "contradiction",
        "neutral"
    }:
        raise ValueError(
            f"Unexpected NLI label mapping: {mapping}"
        )

    return mapping


def predict_nli(
    premise: str,
    hypothesis: str
) -> dict[str, float]:
    """
    Predict entailment, contradiction and neutral scores.

    Evidence is the premise.
    Claim is the hypothesis.
    """

    if not premise or not premise.strip():
        raise ValueError("Premise cannot be empty")

    if not hypothesis or not hypothesis.strip():
        raise ValueError("Hypothesis cannot be empty")

    model, tokenizer, device = load_model()
    import torch

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation="only_first",
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    label_mapping = get_label_mapping(model)

    scores = {
        label: float(probabilities[index].item())
        for index, label in label_mapping.items()
    }

    return scores


def predict_batch(
    premise: str,
    hypotheses: list[str]
) -> list[dict[str, float]]:
    """
    Evaluate multiple claims against one evidence passage.

    One batch reduces model invocation overhead.
    """

    if not premise.strip():
        raise ValueError("Premise cannot be empty")

    valid_indices = [
        i for i, hypothesis in enumerate(hypotheses)
        if isinstance(hypothesis, str) and hypothesis.strip()
    ]

    if not valid_indices:
        return [
            {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0
            }
            for _ in hypotheses
        ]

    model, tokenizer, device = load_model()
    import torch

    valid_hypotheses = [
        hypotheses[i] for i in valid_indices
    ]

    inputs = tokenizer(
        [premise] * len(valid_hypotheses),
        valid_hypotheses,
        padding=True,
        truncation="only_first",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    label_mapping = get_label_mapping(model)

    results = []

    for row in probabilities:
        results.append({
            label: float(row[index].item())
            for index, label in label_mapping.items()
        })

    output = [
        {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0
        }
        for _ in hypotheses
    ]

    for index, result in zip(valid_indices, results):
        output[index] = result

    return output
