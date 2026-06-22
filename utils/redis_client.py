# utils/redis_client.py
"""Redis 客户端管理 — 统一连接池，供各模块共享。

用法:
    from utils.redis_client import get_redis

    redis = await get_redis()
    if redis:
        await redis.setex("key", 3600, "value")
        val = await redis.get("key")
"""

import asyncio
from typing import Optional
from config.settings import settings
from utils.logger import logger

_redis = None
_lock = asyncio.Lock()
_init_failed = False


async def get_redis():
    """获取共享 Redis 连接（惰性初始化，自动重连）。

    仅在 CACHE_BACKEND=redis 时可用。
    连接失败时返回 None，各调用方自行降级。
    """
    global _redis, _init_failed

    if settings.cache.backend != "redis":
        return None

    if _redis is not None:
        return _redis

    if _init_failed:
        return None

    async with _lock:
        if _redis is not None:
            return _redis
        if _init_failed:
            return None

        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(
                settings.redis.url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                max_connections=10,
            )
            await _redis.ping()
            logger.info(f"Redis 共享连接已建立: {settings.redis.url}")
            return _redis
        except Exception as e:
            logger.warning(f"Redis 共享连接失败 ({e})，相关功能降级")
            _init_failed = True
            _redis = None
            return None


async def close_redis():
    """关闭 Redis 连接（服务关闭时调用）。"""
    global _redis, _init_failed
    if _redis is not None:
        try:
            await _redis.close()
        except Exception:
            pass
        _redis = None
    _init_failed = False
    logger.info("Redis 连接已关闭")


async def is_redis_available() -> bool:
    """检查 Redis 是否可用。"""
    r = await get_redis()
    return r is not None