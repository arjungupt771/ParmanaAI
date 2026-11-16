"""
ParmanaAI - Thread-Safe LRU & TTL Cache for Embeddings and Inference
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class CacheEntry:
    value: Any
    created_at: float


class LRUTTLCache:
    """
    Thread-safe Least-Recently-Used (LRU) cache with Time-To-Live (TTL) expiration.
    """

    def __init__(self, maxsize: int = 2048, ttl_seconds: Optional[float] = 3600.0):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def _is_expired(self, entry: CacheEntry) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - entry.created_at) > self.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if absent or expired.
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                self.misses += 1
                self.evictions += 1
                return None

            # Move to end to indicate recent use
            self._cache.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        """
        Store value under key, evicting oldest entry if capacity is reached.
        """
        with self._lock:
            now = time.time()
            if key in self._cache:
                self._cache[key] = CacheEntry(value=value, created_at=now)
                self._cache.move_to_end(key)
                return

            if len(self._cache) >= self.maxsize:
                # Evict oldest entry
                self._cache.popitem(last=False)
                self.evictions += 1

            self._cache[key] = CacheEntry(value=value, created_at=now)

    def clear(self) -> None:
        """Clear all cache items and counters."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic hit rate and capacity statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = round(self.hits / max(total_requests, 1), 4)
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
            }


global_cache = LRUTTLCache(maxsize=4096, ttl_seconds=3600.0)
