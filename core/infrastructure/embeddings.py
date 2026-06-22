# core/infrastructure/embeddings.py
"""编码模型加载与 Embedding 缓存。

提供:
- get_embeddings()          → 同步加载 BGE 模型
- embed_query_with_cache()  → 单条查询编码（Redis 缓存）
- embed_documents_with_cache() → 批量文档编码（Redis 缓存）
"""

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    _USING_NEW_PACKAGE = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _USING_NEW_PACKAGE = False
from langchain_core.embeddings import Embeddings
from config.settings import settings
from utils.logger import logger
from utils.device import get_device

_embeddings = None
_embed_cache = None


def _get_embed_cache():
    global _embed_cache
    if _embed_cache is not None:
        return _embed_cache

    from utils.redis_client import get_redis

    class EmbedCache:
        def __init__(self):
            self._ttl = 86400

        async def _get_redis(self):
            return await get_redis()

        async def get(self, text: str):
            import hashlib
            redis = await self._get_redis()
            if redis is None:
                return None
            key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
            val = await redis.get(key)
            if val is None:
                return None
            import json
            import numpy as np
            logger.debug(f"Embedding 缓存命中: {text[:30]}")
            return np.array(json.loads(val))

        async def set(self, text: str, vec):
            redis = await self._get_redis()
            if redis is None:
                return
            import hashlib
            import json
            key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
            await redis.setex(key, self._ttl, json.dumps(vec.tolist()))

    _embed_cache = EmbedCache()
    return _embed_cache


def get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        device = get_device()
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
            },
        )
        source = "langchain_huggingface" if _USING_NEW_PACKAGE else "langchain_community (legacy)"
        logger.info(f"embeddings model 加载完成 [{source}] ({settings.EMBEDDING_MODEL_PATH}, device={device})")
    return _embeddings


async def embed_query_with_cache(text: str):
    import numpy as np
    cache = _get_embed_cache()
    vec = await cache.get(text)
    if vec is not None:
        return vec
    embeddings = get_embeddings()
    vec = np.array(embeddings.embed_query(text))
    await cache.set(text, vec)
    return vec


async def embed_documents_with_cache(texts: list):
    import numpy as np
    cache = _get_embed_cache()
    cached = []
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        vec = await cache.get(text)
        if vec is not None:
            cached.append((i, vec))
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        embeddings = get_embeddings()
        new_vecs = embeddings.embed_documents(uncached_texts)
        for idx, text, vec in zip(uncached_indices, uncached_texts, new_vecs):
            cached.append((idx, np.array(vec)))
            await cache.set(text, np.array(vec))

    cached.sort(key=lambda x: x[0])
    return [v for _, v in cached]