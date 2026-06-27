# evaluation/metrics/retrieval.py
"""检索质量指标计算

支持三种匹配方式（优先级从高到低）：
  1. gold_chunk: 检查 chunk 的 chunk_id 是否在 gold_chunks 中（最精确）
  2. gold_doc: 检查 chunk 的 doc_id 是否在 gold_docs 中（文档级）
  3. keyword_match: 检查 retrieved_doc 的 source/content 是否包含期望关键词（兜底）
  4. llm_judge: 使用 LLM 判断文档是否与问题相关（更准确，但更慢）
"""
import math
from typing import List, Optional, Any

from .types import RetrievalResult
from config.prompts import LLM_RELEVANCE_PROMPT


def _make_chunk_id(doc) -> str:
    """从文档元数据构建 chunk_id：{doc_id}_chunk_{chunk_index}

    优先使用 chunk_index（全局编号，与测试集生成器一致），
    回退到 doc_chunk_index（文档内编号）。
    """
    m = doc.metadata if hasattr(doc, "metadata") else {}
    doc_id = str(m.get("doc_id", "") or "")
    idx = m.get("chunk_index")
    if idx is None:
        idx = m.get("doc_chunk_index")
    if doc_id and idx is not None:
        return f"{doc_id}_chunk_{int(idx)}"
    return ""


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
    def _keyword_hit_count(doc, expected_keywords: List[str]) -> int:
        """统计文档中命中多少个期望关键词"""
        if not expected_keywords:
            return 0
        txt = RetrievalMetrics._doc_text(doc)
        return sum(1 for kw in expected_keywords if kw.lower() in txt)

    @staticmethod
    def _is_relevant_by_gold(doc, gold_chunks: List[str], gold_docs: List[str]) -> bool:
        """用 gold_chunks / gold_docs 判断 chunk 是否相关。

        优先级：gold_chunks（精确匹配）> gold_docs（文档级匹配）
        """
        m = doc.metadata if hasattr(doc, "metadata") else {}

        if gold_chunks:
            chunk_id = _make_chunk_id(doc)
            if chunk_id and chunk_id in gold_chunks:
                return True

        if gold_docs:
            doc_id = str(m.get("doc_id", "") or "")
            if doc_id and doc_id in gold_docs:
                return True

        return False

    @staticmethod
    def _is_relevant(
        doc,
        expected_keywords: List[str],
        gold_chunks: Optional[List[str]] = None,
        gold_docs: Optional[List[str]] = None,
        min_hits: int = 2,
    ) -> bool:
        """检查文档是否与期望答案相关。

        优先级：gold_chunks > gold_docs > keyword_match
        要求至少命中 min_hits 个关键词才算相关，避免偶然命中单个词汇。
        """
        gold_chunks = gold_chunks or []
        gold_docs = gold_docs or []

        if gold_chunks or gold_docs:
            if RetrievalMetrics._is_relevant_by_gold(doc, gold_chunks, gold_docs):
                return True

        if not expected_keywords:
            return False
        hits = RetrievalMetrics._keyword_hit_count(doc, expected_keywords)
        threshold = min(min_hits, len(expected_keywords))
        return hits >= threshold

    @staticmethod
    def _is_relevant_llm(llm, question: str, doc) -> float:
        """使用 LLM 判断文档是否与问题相关。

        Returns:
            0.0 ~ 1.0 的相关性分数（归一化后的 0/1 判断）
        """
        if llm is None:
            return 0.0
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        prompt = LLM_RELEVANCE_PROMPT.format(
            question=question,
            doc_text=content[:2000],
        )
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            text = (response.content if hasattr(response, "content") else str(response)).strip()
            for ch in text:
                if ch.isdigit():
                    return float(ch)
            return 0.0
        except Exception as e:
            logger.debug(f"LLM 评分解析失败: {e}")
            return 0.0

    @classmethod
    def _llm_relevance_scores(cls, llm, question: str, docs, expected_keywords: List[str]) -> List[int]:
        """LLM + 关键词混合判断：LLM 判断为主，关键词为快速预筛。

        策略：先用关键词快速过滤明显不相关的文档，再用 LLM 精判剩余文档。
        """
        if not docs:
            return []

        scores = []
        for doc in docs:
            # 关键词预筛：命中 0 个关键词的直接判为不相关（跳过 LLM 调用）
            kw_hits = cls._keyword_hit_count(doc, expected_keywords)
            if kw_hits == 0:
                scores.append(0)
            elif kw_hits >= min(2, len(expected_keywords)):
                # 命中 2+ 关键词的，关键词置信度已足够，直接判相关
                scores.append(1)
            else:
                # 命中 1 个关键词的，用 LLM 精判
                llm_score = cls._is_relevant_llm(llm, question, doc)
                scores.append(int(llm_score >= 0.5))
        return scores

    @staticmethod
    def _relevance_scores(docs, expected_keywords: List[str],
                          gold_chunks=None, gold_docs=None) -> List[int]:
        """返回每个文档的相关性分数（1=相关, 0=不相关）"""
        gc, gd = (gold_chunks or []), (gold_docs or [])
        return [1 if RetrievalMetrics._is_relevant(d, expected_keywords, gc, gd) else 0 for d in docs]

    @classmethod
    def _norm_gold(cls, gold_chunks=None, gold_docs=None):
        return (gold_chunks or []), (gold_docs or [])

    @classmethod
    def recall_at_k(cls, docs, expected_keywords: List[str], k: int,
                    gold_chunks=None, gold_docs=None) -> float:
        """Recall@K: 前 K 个结果中是否有命中"""
        gc, gd = cls._norm_gold(gold_chunks, gold_docs)
        if not docs or k <= 0:
            return 0.0
        if not (expected_keywords or gc or gd):
            return 0.0
        top_k = docs[:k]
        hit = any(cls._is_relevant(d, expected_keywords, gc, gd) for d in top_k)
        return 1.0 if hit else 0.0

    @classmethod
    def precision_at_k(cls, docs, expected_keywords: List[str], k: int,
                       gold_chunks=None, gold_docs=None) -> float:
        """Precision@K: 前 K 个结果中相关文档占比"""
        if not docs or k <= 0:
            return 0.0
        gc, gd = cls._norm_gold(gold_chunks, gold_docs)
        top_k = docs[:k]
        if not (expected_keywords or gc or gd):
            return 1.0
        relevant = sum(1 for d in top_k if cls._is_relevant(d, expected_keywords, gc, gd))
        return relevant / min(k, len(top_k))

    @classmethod
    def mrr(cls, docs, expected_keywords: List[str],
            gold_chunks=None, gold_docs=None) -> float:
        """MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名"""
        gc, gd = cls._norm_gold(gold_chunks, gold_docs)
        if not docs:
            return 0.0
        if not (expected_keywords or gc or gd):
            return 0.0
        for rank, d in enumerate(docs, start=1):
            if cls._is_relevant(d, expected_keywords, gc, gd):
                return 1.0 / rank
        return 0.0

    @classmethod
    def ndcg_at_k(cls, docs, expected_keywords: List[str], k: int,
                  gold_chunks=None, gold_docs=None) -> float:
        """NDCG@K: 归一化折损累积增益"""
        if not docs or k <= 0:
            return 0.0
        gc, gd = cls._norm_gold(gold_chunks, gold_docs)
        scores = cls._relevance_scores(docs, expected_keywords, gc, gd)
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
    def first_hit_rank(cls, docs, expected_keywords: List[str],
                       gold_chunks=None, gold_docs=None) -> Optional[int]:
        """返回第一个命中文档的排名（1-based），未命中返回 None"""
        gc, gd = cls._norm_gold(gold_chunks, gold_docs)
        if not docs:
            return None
        if not (expected_keywords or gc or gd):
            return None
        for rank, d in enumerate(docs, start=1):
            if cls._is_relevant(d, expected_keywords, gc, gd):
                return rank
        return None

    @classmethod
    def evaluate(cls, docs, expected_keywords: List[str],
                 gold_chunks=None, gold_docs=None,
                 latency_ms: float = 0.0, question: str = "") -> RetrievalResult:
        """一次性计算所有检索指标（gold_chunk > gold_doc > keyword 三级判断）"""
        gc, gd = cls._norm_gold(gold_chunks, gold_docs)
        return RetrievalResult(
            question=question,
            recall_at_1=cls.recall_at_k(docs, expected_keywords, 1, gc, gd),
            recall_at_3=cls.recall_at_k(docs, expected_keywords, 3, gc, gd),
            recall_at_5=cls.recall_at_k(docs, expected_keywords, 5, gc, gd),
            precision_at_1=cls.precision_at_k(docs, expected_keywords, 1, gc, gd),
            precision_at_3=cls.precision_at_k(docs, expected_keywords, 3, gc, gd),
            precision_at_5=cls.precision_at_k(docs, expected_keywords, 5, gc, gd),
            mrr=cls.mrr(docs, expected_keywords, gc, gd),
            ndcg_at_5=cls.ndcg_at_k(docs, expected_keywords, 5, gc, gd),
            hit_rate=1.0 if cls.first_hit_rank(docs, expected_keywords, gc, gd) else 0.0,
            first_hit_rank=cls.first_hit_rank(docs, expected_keywords, gc, gd),
            latency_ms=latency_ms,
            retrieved_count=len(docs),
        )

    @classmethod
    def evaluate_with_llm(cls, llm, docs, expected_keywords: List[str], question: str = "", latency_ms: float = 0.0) -> RetrievalResult:
        """使用 LLM 混合判断计算所有检索指标。

        策略：关键词预筛 + LLM 精判。
        - 命中 0 个关键词 → 直接判不相关
        - 命中 2+ 个关键词 → 直接判相关
        - 命中 1 个关键词 → LLM 精判
        """
        if not docs or not expected_keywords:
            return RetrievalResult(question=question, retrieved_count=len(docs))

        top_k = 5
        top_docs = docs[:top_k]
        scores = cls._llm_relevance_scores(llm, question, top_docs, expected_keywords)

        def _recall(k):
            return 1.0 if any(scores[:k]) else 0.0

        def _precision(k):
            if k <= 0:
                return 0.0
            return sum(scores[:k]) / min(k, len(scores[:k]))

        def _mrr():
            for rank, s in enumerate(scores, start=1):
                if s == 1:
                    return 1.0 / rank
            return 0.0

        def _ndcg(k):
            top = scores[:k]
            dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(top))
            ideal = sorted(scores, reverse=True)[:k]
            idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
            return dcg / idcg if idcg > 0 else 0.0

        def _first_hit():
            for rank, s in enumerate(scores, start=1):
                if s == 1:
                    return rank
            return None

        return RetrievalResult(
            question=question,
            recall_at_1=_recall(1),
            recall_at_3=_recall(3),
            recall_at_5=_recall(5),
            precision_at_1=_precision(1),
            precision_at_3=_precision(3),
            precision_at_5=_precision(5),
            mrr=_mrr(),
            ndcg_at_5=_ndcg(5),
            hit_rate=1.0 if _first_hit() else 0.0,
            first_hit_rank=_first_hit(),
            latency_ms=latency_ms,
            retrieved_count=len(docs),
        )