# evaluation/metrics/__init__.py
"""评估指标：检索质量 + 生成质量 + 性能指标。

所有指标均为纯 Python 实现，不依赖外部评估库。
生成质量通过 LLM-as-judge 实现，无需 RAGAS。
"""
from .types import RetrievalResult, GenerationResult, EvalSummary
from .retrieval import RetrievalMetrics
from .generation import GenerationMetrics
from .performance import PerformanceMetrics

__all__ = [
    "RetrievalResult",
    "GenerationResult",
    "EvalSummary",
    "RetrievalMetrics",
    "GenerationMetrics",
    "PerformanceMetrics",
]