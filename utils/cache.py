# utils/cache.py
import asyncio
import json
import time
from collections import OrderedDict
from typing import Any, Optional
from utils.logger import logger


class AsyncTTLCache:
    """基于 OrderedDict 的异步 TTL 缓存（内存后端）。

    特性：
      - 自动过期：get 时检查 TTL，过期则删除
      - 容量控制：超出 max_size 时驱逐最早插入的条目（FIFO，O(1)）
      - 线程安全：asyncio.Lock 保护并发访问
      - 重启丢失：服务重启后缓存清空
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


class AsyncRedisCache:
    """基于 Redis 的异步 TTL 缓存（持久化后端）。

    特性：
      - 持久化：服务重启不丢失
      - 跨进程：多实例共享缓存
      - TTL：Redis 原生 EXPIRE，无需手动惰性删除
      - 降级：Redis 不可用时自动回退到内存缓存

    用法:
        cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)
        await cache.set("key", {"answer": "hello"})
        val = await cache.get("key")
    """

    def __init__(self, redis_url: str, ttl_seconds: int):
        self.ttl = ttl_seconds
        self.redis_url = redis_url
        self._redis = None
        self._fallback: Optional[AsyncTTLCache] = None
        self._init_failed = False

    async def _ensure_redis(self):
        """惰性连接 Redis，失败时使用内存降级。"""
        if self._redis is not None:
            return self._redis
        if self._init_failed:
            return None
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            await self._redis.ping()
            logger.info(f"Redis 缓存已连接: {self.redis_url}")
            return self._redis
        except Exception as e:
            logger.warning(f"Redis 连接失败 ({e})，降级为内存缓存")
            self._init_failed = True
            self._redis = None
            self._fallback = AsyncTTLCache(self.ttl, 1000)
            return None

    async def get(self, key: str) -> Optional[Any]:
        redis = await self._ensure_redis()
        if redis is None:
            return await self._fallback.get(key) if self._fallback else None
        try:
            value = await redis.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis GET 失败: {e}")
            return None

    async def set(self, key: str, value: Any):
        redis = await self._ensure_redis()
        if redis is None:
            if self._fallback:
                await self._fallback.set(key, value)
            return
        try:
            await redis.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Redis SET 失败: {e}")

    async def clear(self):
        redis = await self._ensure_redis()
        if redis is None:
            if self._fallback:
                await self._fallback.clear()
            return
        try:
            await redis.flushdb()
            logger.info("Redis 缓存已清空")
        except Exception as e:
            logger.warning(f"Redis FLUSHDB 失败: {e}")