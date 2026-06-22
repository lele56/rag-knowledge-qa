# core/memory_system/episodic.py
"""
情景记忆 (Episodic Memory)。

存储：把每次问答事件 (question, answer, sources, timestamp) 存进 Qdrant 的
      `episodic_memory` 集合。用 embedding 做向量检索。
检索：给定 query，在 episodic_memory 里向量搜索 top-K，按公式打分。
淘汰：超过 EPISODIC_MAX_STORE 条时，按时间淘汰最旧的。
"""
from typing import List, Tuple, Dict, Any, Optional, Set
import time
import uuid

from .scoring import score_episodic
from .config import cfg
from utils.logger import logger


def _to_float_list(vec) -> List[float]:
    """确保向量是 list[float]（HuggingFaceEmbeddings 可能返回 numpy.ndarray，
    新版本 qdrant-client 的 PointStruct/query_vector 要求 Python list）。"""
    if hasattr(vec, "tolist"):
        return [float(v) for v in vec.tolist()]
    return [float(v) for v in vec]


# ---------------------------------------------------------------------------
# Qdrant: 集合创建 / 写入 / 向量搜索
# ---------------------------------------------------------------------------

def _ensure_collection():
    """确保 `episodic_memory` 集合存在，不存在就创建。"""
    from core.infrastructure.vector_store import _get_client
    from core.infrastructure.embeddings import get_embeddings
    from qdrant_client.http.models import Distance, VectorParams

    client = _get_client()
    col = cfg.EPISODIC_COLLECTION
    try:
        client.get_collection(col)
    except Exception:
        emb = get_embeddings()
        sample_vec = emb.embed_query("test")
        client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=len(sample_vec), distance=Distance.COSINE),
        )
        logger.info(f"[memory] 创建情景记忆 Qdrant 集合: {col}")


def store_episodic(question: str,
                    answer: str,
                    sources: Optional[List[str]] = None,
                    importance: Optional[float] = None) -> str:
    """把一次问答事件写入情景记忆（Qdrant episodic_memory 集合）。"""
    if not cfg.ENABLED:
        return ""

    try:
        _ensure_collection()
        from core.infrastructure.vector_store import _get_client
        from core.infrastructure.embeddings import get_embeddings
        from qdrant_client.http.models import PointStruct

        client = _get_client()
        emb = get_embeddings()
        ts = time.time()
        imp = importance if importance is not None else cfg.IMPORTANCE_INIT
        content = f"Q: {question}\nA: {answer}"

        # 向量化 → 转 list[float]（Qdrant 要求）
        vector = _to_float_list(emb.embed_query(content))
        pid = str(uuid.uuid4())

        client.upsert(
            collection_name=cfg.EPISODIC_COLLECTION,
            points=[PointStruct(
                id=pid,
                vector=vector,
                payload={
                    "question": question,
                    "answer": answer,
                    "sources": ",".join(sources) if sources else "",
                    "importance": float(imp),
                    "timestamp": float(ts),
                    "page_content": content,
                },
            )],
        )
        logger.info(f"[memory] 情景记忆写入: {question[:40]}")

        try:
            _evict_if_too_many()
        except Exception as e:
            logger.warning(f"[memory] 情景记忆淘汰失败: {e}")

        return pid
    except Exception as e:
        logger.warning(f"[memory] 情景记忆写入失败: {e}")
        return ""


def _evict_if_too_many():
    """超过 EPISODIC_MAX_STORE 时，按 timestamp 删最旧的。"""
    from core.infrastructure.vector_store import _get_client

    client = _get_client()
    col = cfg.EPISODIC_COLLECTION
    info = client.get_collection(col)
    count = info.points_count if info else 0
    if count <= cfg.EPISODIC_MAX_STORE:
        return
    to_remove = count - cfg.EPISODIC_MAX_STORE

    # scroll 取出按 payload.timestamp 排序的最旧一批
    records, _ = client.scroll(
        collection_name=col,
        limit=max(to_remove + 20, 100),
        with_payload=True,
        with_vectors=False,
    )
    records_sorted = sorted(records, key=lambda r: r.payload.get("timestamp", 0))
    dead_ids = [r.id for r in records_sorted[:to_remove]]
    if dead_ids:
        client.delete(collection_name=col, points_selector=dead_ids)
        logger.info(f"[memory] 情景记忆淘汰 {len(dead_ids)} 条最旧的")


def _normalize_filter_mem(source_filter) -> Optional[Set[str]]:
    """把 source_filter 归一化成 set[str]（与 base.py 保持一致的语义）。"""
    if source_filter is None:
        return None
    if isinstance(source_filter, (set, frozenset)):
        s = {v for v in source_filter if isinstance(v, str) and v.strip()}
        return s if s else None
    if isinstance(source_filter, (list, tuple)):
        s = {v for v in source_filter if isinstance(v, str) and v.strip()}
        return s if s else None
    if isinstance(source_filter, str):
        s = source_filter.strip()
        return {s} if s else None
    return None


def recall_episodic(query: str, source_filter=None) -> List[Tuple[float, str]]:
    """
    在情景记忆中向量搜索与 query 相关的历史问答。

    Args:
        query: 要搜索的问题
        source_filter: 支持 str / list[str] / set[str] / None。
                      如果提供，只返回 sources 字段包含任一关键词的历史问答
                      （用于避免"问 A 文档却召回 B 文档对话"的内容污染）。
                      多文档聚焦场景：{"doc_a", "doc_b", "doc_c"} → 命中任一即通过

    返回 [(score, text), ...]，按分数降序。
    """
    if not cfg.ENABLED:
        return []

    try:
        from core.infrastructure.vector_store import _get_client
        from core.infrastructure.embeddings import get_embeddings

        _ensure_collection()
        client = _get_client()
        emb = get_embeddings()
        query_vec = _to_float_list(emb.embed_query(query))
    except Exception as e:
        logger.warning(f"[memory] 情景记忆检索准备失败: {e}")
        return []

    try:
        # 新版本 qdrant-client 用 query_points 替代 search
        response = client.query_points(
            collection_name=cfg.EPISODIC_COLLECTION,
            query=query_vec,
            limit=cfg.EPISODIC_TOP_K * 3,
            with_payload=True,
        )
        hits = getattr(response, "points", []) or []
    except Exception as e:
        logger.warning(f"[memory] 情景记忆向量搜索失败: {e}")
        return []

    filter_set = _normalize_filter_mem(source_filter)

    results: List[Tuple[float, str]] = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        q = payload.get("question", "") if isinstance(payload, dict) else ""
        a = payload.get("answer", "") if isinstance(payload, dict) else ""
        sources_str = payload.get("sources", "") if isinstance(payload, dict) else ""
        if not q:
            continue

        # 🔑 多值 OR 过滤：命中任一关键词即通过
        if filter_set:
            matched = False
            sources_low = (sources_str or "").lower()
            q_low = q.lower()
            for sf in filter_set:
                sf_low = sf.lower()
                # 1) sources 字段（更精确）
                if sources_low and sf_low in sources_low:
                    matched = True
                    break
                # 2) 问题文本中的关键词（fallback）
                if sf_low in q_low:
                    matched = True
                    break
            if not matched:
                continue

        vector_sim = float(getattr(hit, "score", 0.0) or 0.0)
        ts = float(payload.get("timestamp", time.time())) if isinstance(payload, dict) else time.time()
        imp = float(payload.get("importance", cfg.IMPORTANCE_INIT)) if isinstance(payload, dict) else cfg.IMPORTANCE_INIT
        score = score_episodic(vector_similarity=vector_sim,
                                timestamp_sec=ts,
                                importance=imp)
        text = f"[之前你问过同一主题] Q: {q} A: {a}"
        results.append((score, text))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:cfg.EPISODIC_TOP_K]