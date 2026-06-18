import asyncio
import time
from typing import Any, Optional
from utils.logger import logger

class AsyncTTLCache:
    def __init__(self, ttl_seconds: int, max_size: int):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            value, expire = self._cache[key]
            if time.time() > expire:
                del self._cache[key]
                return None
            return value
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            if len(self._cache) >= self.max_size:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (value, time.time() + self.ttl)
    
    async def clear(self):
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")