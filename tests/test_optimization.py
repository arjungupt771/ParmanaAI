"""
ParmanaAI - Phase 16 Performance & Cost Optimization Tests

Tests:
- LRUTTLCache eviction and capacity limits
- TTL expiration behavior
- Cache hit/miss/eviction metrics
- TokenBudgetManager token estimation
- Context truncation satisfying strict token bounds
"""

import time
import pytest

from backend.optimization.cache import LRUTTLCache
from backend.optimization.prompt_budget import TokenBudgetManager


def test_lru_cache_capacity_and_eviction():
    cache = LRUTTLCache(maxsize=3, ttl_seconds=60.0)

    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")

    assert cache.get("k1") == "v1"
    assert cache.get("k2") == "v2"

    # Insert k4, should evict least recently used (k3)
    cache.set("k4", "v4")

    assert cache.get("k3") is None
    assert cache.get("k4") == "v4"
    assert cache.get("k1") == "v1"

    stats = cache.get_stats()
    assert stats["evictions"] == 1
    assert stats["hits"] >= 3
    assert stats["misses"] >= 1
    assert stats["hit_rate"] > 0.6


def test_ttl_cache_expiration():
    # Cache with very short TTL
    cache = LRUTTLCache(maxsize=10, ttl_seconds=0.05)
    cache.set("temp_key", "temp_value")

    assert cache.get("temp_key") == "temp_value"

    time.sleep(0.06)

    # Key should be expired
    assert cache.get("temp_key") is None
    stats = cache.get_stats()
    assert stats["evictions"] == 1


def test_token_budget_manager():
    budget_mgr = TokenBudgetManager(max_context_tokens=100)

    short_text = "Metformin is prescribed for type 2 diabetes."
    estimated = budget_mgr.estimate_tokens(short_text)
    assert 5 <= estimated <= 20

    evidence_items = [
        {
            "evidence_id": "ev-1",
            "text": "First passage on hypertension guidelines with systolic blood pressure threshold 140 mmHg.",
        },
        {
            "evidence_id": "ev-2",
            "text": "Second passage describing clinical trial outcomes in diabetic populations with HbA1c 7.5%.",
        },
        {
            "evidence_id": "ev-3",
            "text": "Third very long passage describing secondary endpoints, cohort analyses, demographic variables, and exclusion criteria across multi-center randomized controlled trials over 10 years.",
        },
    ]

    budgeted, diag = budget_mgr.budget_evidence(evidence_items, max_tokens=35)

    assert len(budgeted) >= 1
    assert diag["total_tokens_used"] <= 35
    assert diag["headroom_tokens"] >= 0
