# core/retrievers/bm25.py
"""BM25 关键词检索器：基于 langchain_community 的 BM25Retriever。

与向量检索互补，不会漏掉含精确关键词的内容。
"""
import re
from typing import List, Any, Optional, Set
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LangChainBM25
from core.retrievers.filtering import normalize_filter, doc_matches_source_filter
from utils.logger import logger

try:
    import rank_bm25
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

# 全局 BM25 实例缓存
_bm25_instance = None  # 类型: Optional["BM25Retriever"]


def _tokenize(text: str) -> List[str]:
    """简单分词（中英文混合）：中文按单字切、英文按词切。"""
    tokens = []
    if not text:
        return tokens
    for match in re.finditer(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower()):
        tokens.append(match.group())
    return tokens or list(text.lower())


_STATE = "_bm25_int_state_v1"


class BM25Retriever(LangChainBM25):
    """BM25 关键词检索器（继承 langchain_community 实现）。

    附加：支持 source 过滤，与混合检索兼容。
    """

    source_filter_set: Optional[Set[str]] = None

    @classmethod
    def from_documents(
        cls,
        docs: List[Document],
        k: int = 8,
        source_filter=None,
        **kwargs,
    ) -> "BM25Retriever":
        """从文档列表构建 BM25 索引。"""
        if not _BM25_OK:
            raise RuntimeError("请先安装: pip install rank_bm25")

        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(d.page_content) for d in docs]
        bm25 = BM25Okapi(tokenized)
        retriever = cls(
            vectorizer=bm25,
            docs=docs,
            k=k,
            **kwargs
        )
        retriever.source_filter_set = normalize_filter(source_filter)
        logger.info(f"BM25 索引已构建 ({len(docs)} 个 chunks)")
        return retriever

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        q_tokens = _tokenize(query)
        doc_scores = self.vectorizer.get_scores(q_tokens)

        sorted_pairs = sorted(
            enumerate(zip(self.docs, doc_scores)),
            key=lambda x: x[1][1],
            reverse=True
        )

        filters = self.source_filter_set
        kept: List[Document] = []
        for idx, (doc, score) in sorted_pairs:
            if filters and not doc_matches_source_filter(doc, filters):
                continue
            doc.metadata["_score"] = score
            kept.append(doc)
            if len(kept) >= self.k:
                break

        return kept


def get_bm25_retriever(k: int = 8, source_filter=None) -> Optional["BM25Retriever"]:
    from core.infrastructure.vector_store import get_all_documents
    all_docs = get_all_documents()
    if not all_docs:
        return None
    return BM25Retriever.from_documents(docs=all_docs, k=k, source_filter=source_filter)


def reset_bm25_retriever():
    """重置全局 BM25 缓存（文档更新后调用）"""
    global _bm25_instance
    _bm25_instance = None
    logger.info("BM25 索引缓存已重置")