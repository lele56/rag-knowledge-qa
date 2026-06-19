import asyncio
import time
from collections import OrderedDict
from typing import Any, Optional
from utils.logger import logger


class AsyncTTLCache:
    """基于 OrderedDict 的异步 TTL 缓存。

    特性：
      - 自动过期：get 时检查 TTL，过期则删除
      - 容量控制：超出 max_size 时驱逐最早插入的条目（FIFO，O(1)）
      - 线程安全：asyncio.Lock 保护并发访问
    """

    def __init__(self, ttl_seconds: int, max_size: int):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
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
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time() + self.ttl)

    async def clear(self):
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")