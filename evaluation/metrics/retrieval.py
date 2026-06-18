# evaluation/metrics/retrieval.py
"""检索质量指标计算

支持两种匹配方式：
  1. keyword_match: 检查 retrieved_doc 的 source/content 是否包含期望关键词
  2. doc_id_match: 检查 retrieved_doc 的 doc_id 是否在期望列表中
"""
import math
from typing import List, Optional

from .types import RetrievalResult


class RetrievalMetrics:
    """检索质量指标计算"""

    @staticmethod
    def _doc_text(doc) -> str:
        m = doc.metadata if hasattr(doc, "metadata") else {}
        src = str(m.get("source", "") or "")
        section = str(m.get("section", "") or "")
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        return f"{src} {section} {content}".lower()

    @staticmethod
    def _is_relevant(doc, expected_keywords: List[str]) -> bool:
        """检查文档是否与期望关键词匹配"""
        if not expected_keywords:
            return True
        txt = RetrievalMetrics._doc_text(doc)
        return any(kw.lower() in txt for kw in expected_keywords)

    @staticmethod
    def _relevance_scores(docs, expected_keywords: List[str]) -> List[int]:
        """返回每个文档的相关性分数（1=相关, 0=不相关）"""
        return [1 if RetrievalMetrics._is_relevant(d, expected_keywords) else 0 for d in docs]

    @classmethod
    def recall_at_k(cls, docs, expected_keywords: List[str], k: int) -> float:
        """Recall@K: 前 K 个结果中是否有命中"""
        if not docs or k <= 0 or not expected_keywords:
            return 0.0
        top_k = docs[:k]
        hit = any(cls._is_relevant(d, expected_keywords) for d in top_k)
        return 1.0 if hit else 0.0

    @classmethod
    def precision_at_k(cls, docs, expected_keywords: List[str], k: int) -> float:
        """Precision@K: 前 K 个结果中相关文档占比"""
        if not docs or k <= 0:
            return 0.0
        top_k = docs[:k]
        if not expected_keywords:
            return 1.0
        relevant = sum(1 for d in top_k if cls._is_relevant(d, expected_keywords))
        return relevant / min(k, len(top_k))

    @classmethod
    def mrr(cls, docs, expected_keywords: List[str]) -> float:
        """MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名"""
        if not docs or not expected_keywords:
            return 0.0
        for rank, d in enumerate(docs, start=1):
            if cls._is_relevant(d, expected_keywords):
                return 1.0 / rank
        return 0.0

    @classmethod
    def ndcg_at_k(cls, docs, expected_keywords: List[str], k: int) -> float:
        """NDCG@K: 归一化折损累积增益"""
        if not docs or k <= 0:
            return 0.0
        scores = cls._relevance_scores(docs, expected_keywords)
        top_scores = scores[:k]

        dcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(top_scores)
        )
        ideal = sorted(scores, reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal)
        )
        return dcg / idcg if idcg > 0 else 0.0

    @classmethod
    def first_hit_rank(cls, docs, expected_keywords: List[str]) -> Optional[int]:
        """返回第一个命中文档的排名（1-based），未命中返回 None"""
        if not docs or not expected_keywords:
            return None
        for rank, d in enumerate(docs, start=1):
            if cls._is_relevant(d, expected_keywords):
                return rank
        return None

    @classmethod
    def evaluate(cls, docs, expected_keywords: List[str], latency_ms: float = 0.0, question: str = "") -> RetrievalResult:
        """一次性计算所有检索指标"""
        return RetrievalResult(
            question=question,
            recall_at_1=cls.recall_at_k(docs, expected_keywords, 1),
            recall_at_3=cls.recall_at_k(docs, expected_keywords, 3),
            recall_at_5=cls.recall_at_k(docs, expected_keywords, 5),
            precision_at_1=cls.precision_at_k(docs, expected_keywords, 1),
            precision_at_3=cls.precision_at_k(docs, expected_keywords, 3),
            precision_at_5=cls.precision_at_k(docs, expected_keywords, 5),
            mrr=cls.mrr(docs, expected_keywords),
            ndcg_at_5=cls.ndcg_at_k(docs, expected_keywords, 5),
            hit_rate=1.0 if cls.first_hit_rank(docs, expected_keywords) else 0.0,
            first_hit_rank=cls.first_hit_rank(docs, expected_keywords),
            latency_ms=latency_ms,
            retrieved_count=len(docs),
        )