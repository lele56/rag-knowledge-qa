from typing import Optional, Set
from core.retrievers.base import get_base_retriever
from core.retrievers.multi_query import get_multi_query_retriever
from core.retrievers.hyde import HyDERetriever
from core.retrievers.hybrid import HybridRetriever
from core.retrievers.bm25 import get_bm25_retriever
from core.retrievers.filtering import normalize_filter
from core.reranker import get_reranker
from core.llm import get_llm
from config.settings import settings
from utils.logger import logger


def _make_hybrid(
    vector_retriever,
    source_filter=None,
    python_filter=None,
    override_top_k=None,
) -> HybridRetriever:
    """
    构建混合检索器。如果提供 source_filter，BM25 也会在检索后做来源过滤。

    注意：BM25 是基于本地内存的倒排索引，不支持 Qdrant 那样的 payload filter，
    所以我们通过 `get_bm25_retriever(k=..., source_filter=...)` 支持检索后过滤。
    """
    try:
        bm25 = get_bm25_retriever(k=settings.RETRIEVAL_K, source_filter=source_filter)
    except Exception as e:
        logger.warning(f"BM25 初始化失败: {e}，退化为纯向量检索")
        bm25 = None
    try:
        reranker = get_reranker()
    except Exception as e:
        logger.warning(f"重排序模型加载失败: {e}，不做 CrossEncoder 重排序")
        reranker = None
    final_top_k = override_top_k if override_top_k else settings.RERANK_TOP_K
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        reranker=reranker,
        rerank_top_k=final_top_k,
        python_filter=python_filter,  # 兜底过滤
    )


def get_retriever(source_filter=None, override_top_k=None, override_strategy=None):
    """
    获取最终给 RAG 链用的检索器。

    统一 filter 类型：支持 str / list[str] / set[str] / None。
    语义："一组关键词，命中任意一个即通过"。

    性能优化：
      - 有 source_filter（聚焦模式）时 → MultiQuery 仅生成 2 个子查询（聚焦下改写收益低，省时间）
      - 无 source_filter（全库检索）时 → 使用 settings 配置的完整策略

    Args:
        source_filter: 可选的过滤条件（str / list[str] / set[str] / None）。
                      - 聚焦到特定文档（单篇或多篇）→ 向量检索和 BM25 都会做 payload 过滤
                      - 多文档支持："上传了3篇论文，只在这3篇里搜"
                      - None → 跨所有文档检索
        override_top_k: 覆盖默认的 rerank_top_k，多文档场景下自动放大。
        override_strategy: 临时覆写检索策略（用于评估），None 则使用 settings。
    """
    strategy = (override_strategy or settings.RETRIEVAL_STRATEGY).lower()

    base, python_filter = get_base_retriever(source_filter=source_filter)

    normalized_filter = normalize_filter(source_filter)
    filter_preview = ""
    if normalized_filter:
        preview = ", ".join(sorted(normalized_filter))[:60]
        filter_preview = f", filter={{{preview}}}"

    if strategy == "multi_query":
        if normalized_filter:
            mq_count = 2
        else:
            mq_count = settings.MULTI_QUERY_COUNT
        logger.info(f"Retrieval: MultiQuery(x{mq_count}) + Hybrid (MMR + BM25 + CrossEncoder{filter_preview})")
        mq = get_multi_query_retriever(base, mq_count)
        return _make_hybrid(mq, source_filter=source_filter, python_filter=python_filter, override_top_k=override_top_k)
    elif strategy == "hyde":
        logger.info(f"Retrieval: HyDE + Hybrid (MMR + BM25 + CrossEncoder{filter_preview})")
        hyde = HyDERetriever(llm=get_llm(), base_retriever=base, include_original=True)
        return _make_hybrid(hyde, source_filter=source_filter, python_filter=python_filter, override_top_k=override_top_k)
    else:
        logger.info(f"Retrieval: Hybrid (MMR + BM25 + CrossEncoder{filter_preview})")
        return _make_hybrid(base, source_filter=source_filter, python_filter=python_filter, override_top_k=override_top_k)