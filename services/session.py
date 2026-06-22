# services/session.py
"""会话状态持久化 — 基于 Redis 的会话状态存取。

用法:
    from services.session import save_session_state, load_session_state, delete_session_state

    await save_session_state("session_1", {"focus_sources": ["doc1"], "last_question": "..."})
    state = await load_session_state("session_1")
"""

import json
from typing import Optional
from config.settings import settings
from utils.logger import logger


async def save_session_state(session_id: str, state: dict) -> None:
    from utils.redis_client import get_redis
    redis = await get_redis()
    if redis is None:
        return
    ttl = settings.rate_limit.session_ttl * 60
    await redis.setex(f"session:{session_id}", ttl, json.dumps(state, ensure_ascii=False))
    logger.debug(f"会话状态已保存: {session_id}")


async def load_session_state(session_id: str) -> Optional[dict]:
    from utils.redis_client import get_redis
    redis = await get_redis()
    if redis is None:
        return None
    val = await redis.get(f"session:{session_id}")
    if val is None:
        return None
    try:
        logger.debug(f"会话状态已恢复: {session_id}")
        return json.loads(val)
    except Exception:
        return None


async def delete_session_state(session_id: str) -> None:
    from utils.redis_client import get_redis
    redis = await get_redis()
    if redis is None:
        return
    await redis.delete(f"session:{session_id}")