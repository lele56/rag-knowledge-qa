"""
【核心检索器：DirectQdrantRetriever】

关键架构决策：
  之前用 LangChain 的 QdrantVectorStore.as_retriever()，但它**不把 Qdrant payload 字段映射到 Document.metadata**，
  导致 `source` / `doc_id` 等关键信息丢失 — 这是"过滤后全部变 0 条"的根因。

  现在重写为：直接调用 qdrant_client.query_points()，手动把 ScoredPoint 转成 LangChain Document，
  确保 payload.source / payload.doc_id 等字段完整保留在 Document.metadata 中。

  🔑 双层过滤机制（详见 filtering.py）：
    1) Qdrant 侧：doc_id 精确匹配（KEYWORD 索引，最快最可靠）
    2) Python 侧：source/doc_id 子串匹配（兜底，不依赖索引类型）

  🔑 检索后处理（详见 enhanced.py）：
    1) 降噪：质量过滤 + 相似度阈值 + 冗余去除
    2) 上下文感知：给每个 chunk 加上下文窗口和边界标记
"""
from typing import Optional, Any, Tuple, Set, List
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from core.vector_store import _get_client
from core.embeddings import get_embeddings
from config.settings import settings
from utils.logger import logger

from core.retrievers.filtering import (
    normalize_filter,
    build_qdrant_filter,
    doc_matches_source_filter,
)
from core.retrievers.enhanced import (
    scored_point_to_doc,
    denoise_docs,
    enrich_with_context,
)


# ============================================================
# 自定义向量检索器：直接用 qdrant_client，不经过 LangChain 包装
# ============================================================

class DirectQdrantRetriever(BaseRetriever):
    """
    直接用 qdrant_client.query_points() 做向量检索，
    确保 Qdrant payload 中的 source/doc_id 完整映射到 Document.metadata。

    对比 LangChain 的 QdrantVectorStore.as_retriever()：
      - 优势 1：metadata 字段完整（source / doc_id / file_type 等不丢失）
      - 优势 2：Filter 对象直接传给 qdrant_client，不存在 "dict vs Filter 对象" 的格式问题
      - 优势 3：检索逻辑透明，可调试
      - 代价：自己实现简单的 MMR 重排序

    🔑 双层过滤机制：
      1) Qdrant 侧：doc_id 精确匹配（KEYWORD 索引，最快最可靠）
      2) Python 侧：source/doc_id 子串匹配（兜底，不依赖索引类型）
    """

    k: int = 8
    fetch_k: int = 20
    lambda_mult: float = 0.7
    qdrant_filter: Optional[Any] = None
    source_filter_set: Optional[Set[str]] = None

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        client = _get_client()
        emb = get_embeddings()
        query_vec = emb.embed_query(query)
        col = settings.QDRANT_COLLECTION_NAME

        # 1) Qdrant 侧：先用 query_filter 做 payload 过滤（doc_id 精确匹配）
        try:
            result = client.query_points(
                collection_name=col,
                query=query_vec,
                limit=self.fetch_k,
                query_filter=self.qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(result, 'points', []) or []
        except Exception as e:
            logger.warning(f"Qdrant 向量检索失败: {e}，回退到无过滤")
            try:
                result = client.query_points(
                    collection_name=col,
                    query=query_vec,
                    limit=self.fetch_k,
                    with_payload=True,
                    with_vectors=False,
                )
                points = getattr(result, 'points', []) or []
            except Exception as e2:
                logger.error(f"Qdrant 向量检索彻底失败: {e2}")
                return []

        if not points:
            return []

        # 2) ScoredPoint → LangChain Document
        docs: List[Document] = []
        for p in points:
            d = scored_point_to_doc(p)
            if d:
                docs.append(d)

        if not docs:
            return []

        # 3) Python 侧兜底过滤：检查 source/doc_id 是否匹配任一关键词
        if self.source_filter_set:
            sf_set = self.source_filter_set
            before = len(docs)
            docs = [d for d in docs if doc_matches_source_filter(d, sf_set)]
            if before > 0 and len(docs) == 0:
                logger.info(f"   Python 过滤: {before} → 0 条，放松过滤条件重试")
                doc_ids = {f for f in sf_set if f.startswith("doc_")}
                if doc_ids:
                    docs = [d for d in docs if doc_matches_source_filter(d, doc_ids)]
            elif len(docs) < before:
                logger.info(f"   Python 过滤: {before} → {len(docs)} 条")

        # 4) 降噪：质量过滤 + 相似度阈值 + 冗余去除（聚焦模式下降低阈值，保留更多上下文）
        docs = denoise_docs(docs, min_score=0.3, focused_mode=bool(self.source_filter_set))

        if not docs:
            return []

        # 5) 上下文感知：给每个 chunk 加上下文窗口和边界标记
        docs = enrich_with_context(docs, window=1)

        # 6) 多文档覆盖保障：聚焦 ≥2 篇时，每篇文档至少保留 min_per_doc 条
        docs = self._ensure_per_doc_coverage(docs)

        # 7) 按相似度得分降序取 top-k
        docs.sort(key=lambda d: d.metadata.get("_score", 0.0), reverse=True)
        return docs[: self.k]

    def _ensure_per_doc_coverage(self, docs: List[Document]) -> List[Document]:
        """多文档聚焦时，确保每篇文档至少贡献 min_per_doc 个 chunk。

        原则：先按来源分组，每篇取 top-min_per_doc，剩余按分数补满。
        仅当 source_filter 包含 ≥2 个非 doc_ 关键词时生效。
        """
        if not self.source_filter_set:
            return docs
        human_keywords = {f for f in self.source_filter_set if not f.startswith("doc_")}
        if len(human_keywords) < 2:
            return docs

        min_per_doc = max(2, self.k // max(len(human_keywords), 1))

        # 按 source 分组（保持原始顺序）
        by_source: dict = defaultdict(list)
        for d in docs:
            src = str(d.metadata.get("source", "unknown"))
            by_source[src].append(d)

        # 每篇取 top-min_per_doc → seen 追踪已选
        result: list[Document] = []
        seen: set = set()
        for src, doc_list in by_source.items():
            taken = min(min_per_doc, len(doc_list))
            for d in doc_list[:taken]:
                key = (d.page_content[:80], src)
                if key not in seen:
                    seen.add(key)
                    result.append(d)

        # 剩余 slots：从所有未选中的 docs 中按分数补满到 self.k
        if len(result) < self.k:
            extra = [d for d in docs if d not in result]
            extra.sort(key=lambda d: d.metadata.get("_score", 0.0), reverse=True)
            for d in extra:
                if len(result) >= self.k:
                    break
                key = (d.page_content[:80], str(d.metadata.get("source", "")))
                if key not in seen:
                    seen.add(key)
                    result.append(d)

        logger.info(
            f"  📊 多文档覆盖: {len(docs)} → {len(result)} 条 "
            f"(min_per_doc={min_per_doc}, sources={len(by_source)})"
        )
        return result


# ============================================================
# 对外接口：构造 base retriever
# ============================================================

def get_base_retriever(source_filter=None) -> Tuple[BaseRetriever, Optional[Set[str]]]:
    """
    获取 base retriever（向量检索 + 过滤 + 降噪 + 上下文感知），返回 (retriever, python_filter_set)。

    Args:
        source_filter: 可选的过滤条件，支持 str / list[str] / set[str] / None。
                      - "聚焦到这一批文档" → 显著缩小检索范围，提升精度和速度
                      - None → 跨所有文档检索

    Returns:
        (retriever, python_filter_set) —— 后者用于更高层的兜底过滤。
    """
    filters = normalize_filter(source_filter)
    qfilter = build_qdrant_filter(filters)

    # 聚焦场景下优化：缩小 k 和 fetch_k，加快检索
    is_focused = bool(filters)
    if is_focused:
        # 统计文档数量：非 doc_ 前缀的关键词数 ≈ 文档数
        doc_count = len({f for f in filters if not f.startswith("doc_")})
        doc_count = max(doc_count, 1)

        k = max(settings.RETRIEVAL_K, doc_count * 3)   # 每篇文档至少 3 个候选位
        fetch_k = max(k * 5, doc_count * 15)             # 每篇文档至少 15 个候选
        logger.info(
            f"base retriever: DirectQdrant, doc_count={doc_count}, k={k}, fetch_k={fetch_k}, filter={{{', '.join(sorted(filters))[:60]}}}"
        )
    else:
        doc_count = 0
        k = max(settings.RETRIEVAL_K, 8)
        fetch_k = max(k * 4, 24)
        logger.info(f"base retriever: DirectQdrant, k={k}, fetch_k={fetch_k}")

    retriever = DirectQdrantRetriever(
        k=k,
        fetch_k=fetch_k,
        lambda_mult=0.7,
        qdrant_filter=qfilter,
        source_filter_set=filters,
    )

    # 只对非聚焦或无过滤场景保留之前那行简短日志
    # （聚焦场景的日志已在上面输出，避免重复）
    return retriever, filters