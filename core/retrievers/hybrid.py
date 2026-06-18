# core/retrievers/hybrid.py
"""混合检索：向量搜索 (MMR) + BM25 关键词 → 合并去重 → CrossEncoder 重排序

两阶段检索（无 source_filter 时）：
  Stage 1: 广搜 → 按 doc_id 分组 → 选出最相关的 N 篇文档
  Stage 2: 对每篇文档做聚焦搜索 → 合并 → 降噪 → 重排
"""
from dataclasses import dataclass
from typing import List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from utils.logger import logger
from config.settings import settings
from core.vector_store import _get_client
from core.embeddings import get_embeddings
from core.retrievers.enhanced import scored_point_to_doc, denoise_docs, enrich_with_context

_STATE = "_hr_int_state_v1"


def _normalize_pf(pf) -> Optional[Set[str]]:
    """把 python_filter 的各种输入归一化成 set[str]（与 base.py 保持一致的语义）。"""
    if pf is None:
        return None
    if isinstance(pf, str):
        return {pf.strip()} if pf.strip() else None
    if isinstance(pf, (set, frozenset, list, tuple)):
        s = {v for v in pf if isinstance(v, str) and v.strip()}
        return s if s else None
    return None


def _get_all_doc_ids() -> set[str]:
    """从 doc_id registry 获取所有已注册的 doc_id"""
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        return reg.get_all_doc_ids()
    except Exception:
        return set()


@dataclass
class _HybridState:
    """HybridRetriever 内部状态容器，绕过 Pydantic 属性拦截。"""
    vector_retriever: Any = None
    bm25_retriever: Any = None
    reranker: Any = None
    rerank_top_k: int = 3
    no_rerank: bool = False
    python_filter: Optional[Set[str]] = None


class HybridRetriever(BaseRetriever):
    """混合检索器。支持 Python 侧来源兜底过滤（Qdrant payload 索引缺失时生效）。"""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: Optional[BaseRetriever],
        reranker: Any = None,
        rerank_top_k: int = 3,
        no_rerank: bool = False,
        python_filter=None,
    ):
        try:
            super().__init__()
        except Exception:
            try:
                super().__init__(callback_manager=None)
            except Exception:
                pass

        self.__dict__[_STATE] = _HybridState(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            rerank_top_k=rerank_top_k,
            no_rerank=no_rerank,
            python_filter=_normalize_pf(python_filter),
        )

    @property
    def _s(self) -> _HybridState:
        """内部状态访问器。"""
        return self.__dict__[_STATE]

    def _merge_and_dedup(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """合并向量+BM25结果并去重。

        去重策略（优先级从高到低）：
          1. chunk_index + source：同一文档内的相同 chunk（Qdrant 中重复存储的）
          2. 内容前 100 字符：真正重复的内容
        """
        seen_chunk_ids = set()  # (source, chunk_index)
        seen_content_prefix = set()
        merged = []
        i_v, i_b = 0, 0
        while i_v < len(vector_docs) or i_b < len(bm25_docs):
            if i_v < len(vector_docs):
                d = vector_docs[i_v]
                meta = d.metadata if isinstance(d.metadata, dict) else {}
                src = str(meta.get("source", ""))
                idx = str(meta.get("chunk_index", ""))
                chunk_key = f"{src}::{idx}"
                content_key = d.page_content[:100]
                is_dup = (chunk_key in seen_chunk_ids) or (content_key in seen_content_prefix)
                if not is_dup:
                    if chunk_key:
                        seen_chunk_ids.add(chunk_key)
                    seen_content_prefix.add(content_key)
                    merged.append(Document(
                        page_content=d.page_content,
                        metadata={**d.metadata, "_src": "vector", "_rank_v": i_v},
                    ))
                i_v += 1
            if i_b < len(bm25_docs):
                d = bm25_docs[i_b]
                meta = d.metadata if isinstance(d.metadata, dict) else {}
                src = str(meta.get("source", ""))
                idx = str(meta.get("chunk_index", ""))
                chunk_key = f"{src}::{idx}"
                content_key = d.page_content[:100]
                is_dup = (chunk_key in seen_chunk_ids) or (content_key in seen_content_prefix)
                if not is_dup:
                    if chunk_key:
                        seen_chunk_ids.add(chunk_key)
                    seen_content_prefix.add(content_key)
                    merged.append(Document(
                        page_content=d.page_content,
                        metadata={**d.metadata, "_src": "bm25", "_rank_b": i_b},
                    ))
                i_b += 1
        return merged

    def _apply_python_filter(self, docs: List[Document]) -> List[Document]:
        """在 Python 侧做来源兜底过滤（Qdrant payload 索引缺失时生效）。
        支持多值 OR 过滤：命中任意关键词即保留。
        """
        pf = self._s.python_filter
        if not pf:
            return docs
        if isinstance(pf, str):
            filter_set = {pf}
        else:
            filter_set = set(pf)

        filtered = []
        for d in docs:
            meta = d.metadata if isinstance(d.metadata, dict) else {}
            doc_id = str(meta.get("doc_id", "") or "")
            source = str(meta.get("source", "") or "").lower()
            for f in filter_set:
                if f.startswith("doc_"):
                    if doc_id == f:
                        filtered.append(d)
                        break
                else:
                    if f.lower() in source:
                        filtered.append(d)
                        break
        if len(filtered) != len(docs):
            preview = ", ".join(sorted(filter_set))[:60]
            logger.info(f"  → Python 侧兜底过滤: {len(docs)} → {len(filtered)} 条 (filter={{{preview}}})")
        return filtered

    def _heuristic_sort(self, docs: List[Document], top_k: int) -> List[Document]:
        """CrossEncoder 不可用时的兜底排序：按"向量相似度 + BM25 排名"的启发式打分。

        这是个轻量级 fallback——当 HuggingFace 429 限流导致 CrossEncoder 模型下载失败时，
        不会出现"重排序失败 → 静默返回空 → 检索不到内容"的连锁问题。
        """
        if not docs:
            return docs

        n = len(docs)
        ranked = []
        for d in docs:
            meta = d.metadata if isinstance(d.metadata, dict) else {}
            score = float(meta.get("_score", 0.5) or 0.5)
            r_v = float(meta.get("_rank_v", n))
            r_b = float(meta.get("_rank_b", n))
            # RRF 风格的融合分数：向量分占主导，排名做微调
            fused = score * 0.7 + (1.0 / (1.0 + r_v)) * 0.2 + (1.0 / (1.0 + r_b)) * 0.1
            ranked.append((fused, d))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]

    def _single_stage_retrieve(self, query: str) -> List[Document]:
        """单阶段检索（fallback：registry 无数据时使用）"""
        vector_docs = self._s.vector_retriever.invoke(query)
        bm25_retriever = self._s.bm25_retriever
        bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
        return self._merge_rerank(query, vector_docs, bm25_docs)

    def _two_stage_retrieve(self, query: str) -> List[Document]:
        """两阶段检索：先定位相关文档，再在文档内深度搜索。

        Stage 1: MultiQuery 广搜 → 按文档相关性打分 → 选出 top-3 相关文档
        Stage 2: 对每篇相关文档做 Qdrant 聚焦搜索 → 合并 → 按分数排序
        """
        doc_ids = _get_all_doc_ids()
        if not doc_ids:
            logger.warning("两阶段: registry 中无文档，回退到普通检索")
            return self._single_stage_retrieve(query)

        # ---- Stage 1: 用 MultiQuery 广搜，算文档相关性 ----
        try:
            vector_docs = self._s.vector_retriever.invoke(query)
        except Exception as e:
            logger.warning(f"两阶段 Stage1 失败: {e}，回退普通检索")
            return self._single_stage_retrieve(query)

        doc_best_score: dict[str, float] = {}
        for d in vector_docs:
            did = str(d.metadata.get("doc_id", ""))
            if not did:
                continue
            score = float(d.metadata.get("_score", 0) or 0)
            if score > doc_best_score.get(did, 0.0):
                doc_best_score[did] = score

        if not doc_best_score:
            logger.warning("两阶段: Stage1 未命中任何文档")
            return self._single_stage_retrieve(query)

        # 取 top-5 最相关文档（给 reranker 更多候选）
        sorted_docs = sorted(doc_best_score.items(), key=lambda x: x[1], reverse=True)
        N = min(5, len(sorted_docs))
        top_docs = [d for d, _ in sorted_docs[:N]]
        logger.info(
            f"Stage1 (MultiQuery): 全库 {len(doc_ids)} 篇 → 相关 {N} 篇: "
            f"{', '.join(f'{d[:30]}({doc_best_score[d]:.3f})' for d in top_docs)}"
        )

        # ---- Stage 2: Qdrant 聚焦搜索 ----
        client = _get_client()
        emb = get_embeddings()
        query_vec = emb.embed_query(query)
        col = settings.QDRANT_COLLECTION_NAME
        top_k = self._s.rerank_top_k
        k_per_doc = max(10, top_k * 2)

        all_docs: list[Document] = []

        def _search_one_doc(did: str) -> list[Document]:
            docs: list[Document] = []
            qfilter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=did))])
            try:
                result = client.query_points(
                    collection_name=col,
                    query=query_vec,
                    limit=k_per_doc,
                    query_filter=qfilter,
                    with_payload=True,
                    with_vectors=False,
                )
                points = getattr(result, 'points', []) or []
                for p in points:
                    d = scored_point_to_doc(p)
                    if d:
                        docs.append(d)
            except Exception as e:
                logger.warning(f"  Stage2 [{did[:30]}]: 查询失败: {e}")
            return docs

        with ThreadPoolExecutor(max_workers=min(len(top_docs), 2)) as executor:
            futures = {executor.submit(_search_one_doc, did): did for did in top_docs}
            for future in as_completed(futures):
                all_docs.extend(future.result())

        if not all_docs:
            return []

        # 去重、按 Qdrant 分数排序
        seen: set[str] = set()
        merged: list[Document] = []
        for d in sorted(all_docs, key=lambda d: d.metadata.get("_score", 0.0), reverse=True):
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(d)

        logger.info(f"  Stage2: {N} 篇相关文档 → {len(merged)} 条 chunk")

        merged = enrich_with_context(merged, window=1)
        return self._rerank(query, merged)

    def _merge_rerank(self, query: str, vector_docs: list, bm25_docs: list) -> list[Document]:
        """合并、去重、降噪、重排（单阶段检索用）"""
        merged = self._merge_and_dedup(list(vector_docs), list(bm25_docs))
        logger.info(f"  → 合并去重后 {len(merged)} 条")

        is_focused = bool(self._s.python_filter)
        merged = denoise_docs(merged, query=query, min_score=0.28, focused_mode=is_focused)
        merged = enrich_with_context(merged, window=1)

        top_k = self._s.rerank_top_k
        if self._s.no_rerank:
            return self._heuristic_sort(merged, top_k)

        reranker = self._s.reranker
        if reranker is None:
            return self._heuristic_sort(merged, top_k)

        try:
            reranker.top_n = top_k
            reranked = reranker.compress_documents(merged, query)
            logger.info(f"  → CrossEncoder 重排序后 {len(reranked)} 条")
            return reranked
        except Exception as e:
            logger.warning(f"CrossEncoder 运行失败: {e}，退回启发式打分")
            return self._heuristic_sort(merged, top_k)

    def _rerank(self, query: str, merged: list) -> list[Document]:
        """重排序（两阶段已合并，只需降噪+重排）"""
        top_k = self._s.rerank_top_k

        if self._s.no_rerank:
            logger.info(f"  → 无重排序，直接取 top-{top_k}")
            return self._heuristic_sort(merged, top_k)

        reranker = self._s.reranker
        if reranker is None:
            logger.info(f"  → CrossEncoder 不可用，启发式打分取 top-{top_k}")
            return self._heuristic_sort(merged, top_k)

        try:
            reranker.top_n = top_k
            reranked = reranker.compress_documents(merged, query)
            logger.info(f"  → CrossEncoder 重排序后 {len(reranked)} 条")
            return reranked
        except Exception as e:
            logger.warning(f"CrossEncoder 运行失败: {e}，退回启发式打分")
            return self._heuristic_sort(merged, top_k)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        is_focused = bool(self._s.python_filter)

        if not is_focused:
            return self._two_stage_retrieve(query)

        vector_docs = self._s.vector_retriever.invoke(query)
        bm25_retriever = self._s.bm25_retriever
        bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
        logger.info(f"混合检索 → 向量 {len(vector_docs)} 条, BM25 {len(bm25_docs)} 条")

        vector_docs = self._apply_python_filter(vector_docs)
        bm25_docs = self._apply_python_filter(bm25_docs)
        return self._merge_rerank(query, vector_docs, bm25_docs)