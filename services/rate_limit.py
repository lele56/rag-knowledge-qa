# services/rate_limit.py
"""速率限制 — 基于 Redis 的滑动窗口计数器。

用法:
    from services.rate_limit import check_rate_limit

    if await check_rate_limit("user_123"):
        # 允许请求
    else:
        # 拒绝请求
"""

from config.settings import settings
from utils.logger import logger


async def check_rate_limit(session_id: str) -> bool:
    if not settings.rate_limit.enabled:
        return True
    from utils.redis_client import get_redis
    redis = await get_redis()
    if redis is None:
        return True
    key = f"rate:{session_id}"
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)
        return count <= settings.rate_limit.max_per_minute
    except Exception as e:
        logger.warning(f"速率限制检查失败: {e}")
        return True