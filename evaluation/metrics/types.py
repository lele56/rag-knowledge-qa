# evaluation/metrics/types.py
"""评估指标的数据结构定义"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """单次检索评估结果"""
    question: str
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    hit_rate: float = 0.0
    first_hit_rank: Optional[int] = None
    latency_ms: float = 0.0
    retrieved_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_at_5": self.precision_at_5,
            "mrr": self.mrr,
            "ndcg_at_5": self.ndcg_at_5,
            "hit_rate": self.hit_rate,
            "first_hit_rank": self.first_hit_rank,
            "latency_ms": self.latency_ms,
            "retrieved_count": self.retrieved_count,
        }


@dataclass
class GenerationResult:
    """单次生成评估结果"""
    question: str
    answer: str = ""
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    keyword_recall: float = 0.0
    keyword_f1: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer[:500],
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "keyword_recall": self.keyword_recall,
            "keyword_f1": self.keyword_f1,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
        }


@dataclass
class EvalSummary:
    """评估汇总"""
    total_questions: int = 0
    retrieval: Dict[str, float] = field(default_factory=dict)
    generation: Dict[str, float] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    retrieval_details: List[RetrievalResult] = field(default_factory=list)
    generation_details: List[GenerationResult] = field(default_factory=list)