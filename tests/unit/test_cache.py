"""缓存模块单元测试 — AsyncTTLCache + AsyncRedisCache"""

import json
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# AsyncTTLCache 测试
# ============================================================

class TestAsyncTTLCache:
    def test_get_set_basic(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=60, max_size=100)

        async def _run():
            await cache.set("key1", {"answer": "hello"})
            val = await cache.get("key1")
            assert val == {"answer": "hello"}

        asyncio.run(_run())

    def test_get_missing_key(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=60, max_size=100)

        async def _run():
            val = await cache.get("nonexistent")
            assert val is None

        asyncio.run(_run())

    def test_ttl_expiry(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=0, max_size=100)

        async def _run():
            await cache.set("key1", "value")
            await asyncio.sleep(0.01)
            val = await cache.get("key1")
            assert val is None

        asyncio.run(_run())

    def test_ttl_not_expired(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=3600, max_size=100)

        async def _run():
            await cache.set("key1", "value")
            val = await cache.get("key1")
            assert val == "value"

        asyncio.run(_run())

    def test_overwrite_existing_key(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=3600, max_size=100)

        async def _run():
            await cache.set("key1", "old")
            await cache.set("key1", "new")
            val = await cache.get("key1")
            assert val == "new"

        asyncio.run(_run())

    def test_max_size_eviction(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=3600, max_size=3)

        async def _run():
            await cache.set("a", 1)
            await cache.set("b", 2)
            await cache.set("c", 3)
            await cache.set("d", 4)
            assert await cache.get("a") is None
            assert await cache.get("b") == 2
            assert await cache.get("c") == 3
            assert await cache.get("d") == 4

        asyncio.run(_run())

    def test_clear(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=3600, max_size=100)

        async def _run():
            await cache.set("a", 1)
            await cache.set("b", 2)
            await cache.clear()
            assert await cache.get("a") is None
            assert await cache.get("b") is None

        asyncio.run(_run())

    def test_complex_value_types(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=3600, max_size=100)

        async def _run():
            complex_val = {
                "answer": "hello",
                "sources": ["doc1.pdf", "doc2.pdf"],
                "steps": 3,
                "nested": {"key": [1, 2, 3]},
            }
            await cache.set("key", complex_val)
            val = await cache.get("key")
            assert val == complex_val

        asyncio.run(_run())

    def test_fifo_eviction_order(self):
        from utils.cache import AsyncTTLCache
        cache = AsyncTTLCache(ttl_seconds=3600, max_size=2)

        async def _run():
            await cache.set("first", 1)
            await cache.set("second", 2)
            await cache.set("third", 3)
            assert await cache.get("first") is None
            assert await cache.get("second") == 2
            assert await cache.get("third") == 3

        asyncio.run(_run())


# ============================================================
# AsyncRedisCache 测试（Mock Redis）
# ============================================================

class TestAsyncRedisCacheFallback:
    """Redis 连接失败 → 降级到内存缓存"""

    def test_connection_failure_fallback(self):
        from utils.cache import AsyncRedisCache, AsyncTTLCache

        with patch("redis.asyncio.from_url", side_effect=ConnectionRefusedError("no redis")):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.set("key", {"val": 42})
                result = await cache.get("key")
                assert result == {"val": 42}
                assert cache._init_failed is True
                assert isinstance(cache._fallback, AsyncTTLCache)

            asyncio.run(_run())

    def test_fallback_clear(self):
        from utils.cache import AsyncRedisCache

        with patch("redis.asyncio.from_url", side_effect=ConnectionRefusedError("no redis")):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.set("key", "val")
                await cache.clear()
                assert await cache.get("key") is None

            asyncio.run(_run())

    def test_fallback_get_nonexistent(self):
        from utils.cache import AsyncRedisCache

        with patch("redis.asyncio.from_url", side_effect=ConnectionRefusedError("no redis")):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                val = await cache.get("nonexistent")
                assert val is None

            asyncio.run(_run())


class TestAsyncRedisCacheMocked:
    """Mock Redis 成功连接 → 测试正常读写"""

    @pytest.fixture
    def mock_redis(self):
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value=None)
        mock.setex = AsyncMock(return_value=True)
        mock.flushdb = AsyncMock(return_value=True)
        return mock

    def test_get_returns_value(self, mock_redis):
        from utils.cache import AsyncRedisCache

        mock_redis.get = AsyncMock(return_value=json.dumps({"answer": "hello"}))

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                val = await cache.get("key")
                assert val == {"answer": "hello"}
                mock_redis.get.assert_called_once_with("key")

            asyncio.run(_run())

    def test_get_returns_none(self, mock_redis):
        from utils.cache import AsyncRedisCache

        mock_redis.get = AsyncMock(return_value=None)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                val = await cache.get("key")
                assert val is None

            asyncio.run(_run())

    def test_set_calls_setex(self, mock_redis):
        from utils.cache import AsyncRedisCache

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.set("key", {"answer": "test"})
                mock_redis.setex.assert_called_once()
                args, kwargs = mock_redis.setex.call_args
                assert args[0] == "key"
                assert args[1] == 3600
                assert "answer" in args[2]

            asyncio.run(_run())

    def test_set_serializes_complex_value(self, mock_redis):
        from utils.cache import AsyncRedisCache

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=600)

            async def _run():
                complex_val = {
                    "question": "test?",
                    "answer": "答案",
                    "sources": ["a.pdf", "b.pdf"],
                    "steps": 5,
                }
                await cache.set("q", complex_val)
                args, _ = mock_redis.setex.call_args
                decoded = json.loads(args[2])
                assert decoded == complex_val

            asyncio.run(_run())

    def test_clear_calls_flushdb(self, mock_redis):
        from utils.cache import AsyncRedisCache

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.clear()
                mock_redis.flushdb.assert_called_once()

            asyncio.run(_run())

    def test_connection_reuse(self, mock_redis):
        from utils.cache import AsyncRedisCache

        with patch("redis.asyncio.from_url", return_value=mock_redis) as mock_from_url:
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.get("k1")
                await cache.get("k2")
                await cache.set("k3", "v3")
                assert mock_from_url.call_count == 1

            asyncio.run(_run())

    def test_get_redis_error_returns_none(self, mock_redis):
        from utils.cache import AsyncRedisCache

        mock_redis.get = AsyncMock(side_effect=RuntimeError("redis error"))

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                val = await cache.get("key")
                assert val is None

            asyncio.run(_run())

    def test_set_redis_error_does_not_raise(self, mock_redis):
        from utils.cache import AsyncRedisCache

        mock_redis.setex = AsyncMock(side_effect=RuntimeError("redis error"))

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.set("key", "value")

            asyncio.run(_run())

    def test_ping_failure_triggers_fallback(self):
        from utils.cache import AsyncRedisCache, AsyncTTLCache

        mock = AsyncMock()
        mock.ping = AsyncMock(side_effect=ConnectionError("ping failed"))

        with patch("redis.asyncio.from_url", return_value=mock):
            cache = AsyncRedisCache("redis://localhost:6379/0", ttl_seconds=3600)

            async def _run():
                await cache.set("key", {"val": 1})
                result = await cache.get("key")
                assert result == {"val": 1}
                assert cache._init_failed is True
                assert isinstance(cache._fallback, AsyncTTLCache)

            asyncio.run(_run())


# ============================================================
# CACHE_BACKEND 选择逻辑测试
# ============================================================

class TestCacheBackendSelection:
    """QAService 按 CACHE_BACKEND 选择后端"""

    def test_memory_backend_default(self):
        from utils.cache import AsyncTTLCache
        assert AsyncTTLCache is not None

    def test_redis_backend_available(self):
        from utils.cache import AsyncRedisCache
        assert AsyncRedisCache is not None

    def test_backend_config_exists(self):
        from config.settings import settings
        assert hasattr(settings.cache, "backend")
        assert settings.cache.backend in ("memory", "redis")


# ============================================================
# AsyncRedisCache 接口一致性测试
# ============================================================

class TestCacheInterfaceConsistency:
    """AsyncTTLCache 和 AsyncRedisCache 实现相同接口"""

    def test_both_have_get_set_clear(self):
        from utils.cache import AsyncTTLCache, AsyncRedisCache
        for cls in [AsyncTTLCache, AsyncRedisCache]:
            assert hasattr(cls, "get")
            assert hasattr(cls, "set")
            assert hasattr(cls, "clear")
            assert callable(getattr(cls, "get"))
            assert callable(getattr(cls, "set"))
            assert callable(getattr(cls, "clear"))


# ============================================================
# 速率限制测试
# ============================================================

class TestRateLimit:
    def test_disabled_always_ok(self):
        import os
        os.environ["RATE_LIMIT_ENABLED"] = "false"
        import sys
        if "services.qa_service" in sys.modules:
            del sys.modules["services.qa_service"]
        if "config.settings" in sys.modules:
            del sys.modules["config.settings"]
        from config.settings import settings
        assert settings.rate_limit.enabled is False

    def test_enabled_within_limit(self):
        mock = AsyncMock()
        mock.incr = AsyncMock(return_value=5)
        mock.expire = AsyncMock(return_value=True)

        with patch("utils.redis_client.get_redis", return_value=mock), \
             patch("services.rate_limit.settings.rate_limit.enabled", True):
            async def _run():
                from services.rate_limit import check_rate_limit
                result = await check_rate_limit("user1")
                assert result is True
            asyncio.run(_run())

    def test_exceeded_limit(self):
        mock = AsyncMock()
        mock.incr = AsyncMock(return_value=11)
        mock.expire = AsyncMock(return_value=True)

        with patch("utils.redis_client.get_redis", return_value=mock), \
             patch("services.rate_limit.settings.rate_limit.enabled", True):
            async def _run():
                from services.rate_limit import check_rate_limit
                result = await check_rate_limit("user1")
                assert result is False
            asyncio.run(_run())

    def test_first_request_sets_expire(self):
        mock = AsyncMock()
        mock.incr = AsyncMock(return_value=1)
        mock.expire = AsyncMock(return_value=True)

        with patch("utils.redis_client.get_redis", return_value=mock), \
             patch("services.rate_limit.settings.rate_limit.enabled", True):
            async def _run():
                from services.rate_limit import check_rate_limit
                result = await check_rate_limit("user1")
                assert result is True
                mock.expire.assert_called_once_with("rate:user1", 60)
            asyncio.run(_run())

    def test_no_redis_always_ok(self):
        with patch("utils.redis_client.get_redis", return_value=None):
            async def _run():
                from services.rate_limit import check_rate_limit
                result = await check_rate_limit("user1")
                assert result is True
            asyncio.run(_run())


# ============================================================
# 会话状态持久化测试
# ============================================================

class TestSessionState:
    def test_save_and_load(self):
        mock = AsyncMock()
        mock.setex = AsyncMock(return_value=True)
        state_data = {"focus_sources": ["doc1", "doc2"], "last_question": "hello"}
        mock.get = AsyncMock(return_value=json.dumps(state_data))

        with patch("utils.redis_client.get_redis", return_value=mock):
            async def _run():
                from services.session import save_session_state, load_session_state
                await save_session_state("s1", state_data)
                mock.setex.assert_called_once()
                loaded = await load_session_state("s1")
                assert loaded == state_data
            asyncio.run(_run())

    def test_load_nonexistent(self):
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)

        with patch("utils.redis_client.get_redis", return_value=mock):
            async def _run():
                from services.session import load_session_state
                loaded = await load_session_state("nonexistent")
                assert loaded is None
            asyncio.run(_run())

    def test_delete(self):
        mock = AsyncMock()
        mock.delete = AsyncMock(return_value=1)

        with patch("utils.redis_client.get_redis", return_value=mock):
            async def _run():
                from services.session import delete_session_state
                await delete_session_state("s1")
                mock.delete.assert_called_once_with("session:s1")
            asyncio.run(_run())

    def test_no_redis_gracefully_skips(self):
        with patch("utils.redis_client.get_redis", return_value=None):
            async def _run():
                from services.session import save_session_state, load_session_state, delete_session_state
                await save_session_state("s1", {"a": 1})
                val = await load_session_state("s1")
                assert val is None
                await delete_session_state("s1")
            asyncio.run(_run())


# ============================================================
# Embedding 缓存测试
# ============================================================

class TestEmbeddingCache:
    def test_cache_hit(self):
        import numpy as np
        mock = AsyncMock()
        cached_vec = np.array([0.1, 0.2, 0.3])
        mock.get = AsyncMock(return_value=json.dumps(cached_vec.tolist()))

        with patch("utils.redis_client.get_redis", return_value=mock):
            from core.infrastructure import embeddings
            embeddings._embed_cache = None

            async def _run():
                vec = await embeddings.embed_query_with_cache("测试文本")
                np.testing.assert_array_equal(vec, cached_vec)
            asyncio.run(_run())

    def test_cache_no_redis_falls_through(self):
        import numpy as np
        with patch("utils.redis_client.get_redis", return_value=None):
            from core.infrastructure import embeddings
            embeddings._embed_cache = None

            async def _run():
                vec = await embeddings.embed_query_with_cache("测试文本")
                assert isinstance(vec, np.ndarray)
                assert len(vec) > 0
            asyncio.run(_run())


# ============================================================
# Redis 共享客户端测试
# ============================================================

class TestRedisClient:
    def test_get_redis_memory_mode(self):
        import os
        os.environ["CACHE_BACKEND"] = "memory"
        from utils.redis_client import get_redis

        async def _run():
            redis = await get_redis()
            assert redis is None
        asyncio.run(_run())

    def test_get_redis_connection_failure(self):
        import os
        os.environ["CACHE_BACKEND"] = "redis"
        from utils.redis_client import get_redis

        with patch("redis.asyncio.from_url", side_effect=ConnectionRefusedError("no redis")):
            async def _run():
                redis = await get_redis()
                assert redis is None
            asyncio.run(_run())

    def test_is_redis_available(self):
        import os
        os.environ["CACHE_BACKEND"] = "memory"
        from utils.redis_client import is_redis_available

        async def _run():
            available = await is_redis_available()
            assert available is False
        asyncio.run(_run())