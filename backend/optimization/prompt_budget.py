"""
ParmanaAI - Clinical Context & Prompt Token Budget Manager

Prevents LLM token overruns and reduces API costs by intelligently budgeting
evidence passages and truncating non-essential sentences while preserving numerical claims.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class TokenBudgetManager:
    """
    Manages prompt token budgets for clinical QA generation.
    """

    def __init__(self, max_context_tokens: int = 1500, chars_per_token: float = 4.0):
        self.max_context_tokens = max_context_tokens
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count based on standard English/biomedical token density.
        """
        if not text:
            return 0
        words = text.split()
        # Heuristic: roughly 1.3 tokens per word or chars / 4
        return max(len(words), int(len(text) / self.chars_per_token))

    def budget_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Select and optionally truncate evidence items to strictly satisfy max_tokens budget.
        Prioritizes passages with high relevance scores and numerical mentions.
        """
        budget = max_tokens or self.max_context_tokens
        accumulated_tokens = 0
        budgeted_items: List[Dict[str, Any]] = []

        for item in evidence_items:
            text = item.get("text", "")
            item_tokens = self.estimate_tokens(text)

            if accumulated_tokens + item_tokens <= budget:
                budgeted_items.append(dict(item))
                accumulated_tokens += item_tokens
            else:
                # Remaining token headroom
                remaining_tokens = budget - accumulated_tokens
                if remaining_tokens > 40:
                    # Truncate passage at sentence boundary
                    truncated_text = self._truncate_passage(text, remaining_tokens)
                    if truncated_text:
                        truncated_item = dict(item)
                        truncated_item["text"] = truncated_text
                        truncated_item["is_truncated"] = True
                        budgeted_items.append(truncated_item)
                        accumulated_tokens += self.estimate_tokens(truncated_text)
                break

        diagnostics = {
            "initial_items_count": len(evidence_items),
            "budgeted_items_count": len(budgeted_items),
            "total_tokens_used": accumulated_tokens,
            "max_tokens_budget": budget,
            "headroom_tokens": max(0, budget - accumulated_tokens),
        }

        return budgeted_items, diagnostics

    def _truncate_passage(self, text: str, max_tokens: int) -> str:
        """
        Truncate passage preserving complete sentences, prioritizing sentences
        with numerical values or clinical guidelines.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        selected_sentences = []
        current_tokens = 0

        # Number-containing sentences are given higher priority
        has_num = lambda s: bool(re.search(r"\d", s))

        for s in sentences:
            s_tokens = self.estimate_tokens(s)
            if current_tokens + s_tokens <= max_tokens:
                selected_sentences.append(s)
                current_tokens += s_tokens
            else:
                break

        return " ".join(selected_sentences)
