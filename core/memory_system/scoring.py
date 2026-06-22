# core/memory_system/scoring.py
"""三层记忆的评分公式（完全按你给的公式实现）。"""
import time
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# 1. 时间相关的辅助函数
# ---------------------------------------------------------------------------

def _hours_ago(timestamp_sec: float) -> float:
    """返回给定时间戳（秒）距现在多少小时。"""
    return max(0.0, (time.time() - timestamp_sec) / 3600.0)


def time_decay(timestamp_sec: float, half_life_hours: float = 24.0) -> float:
    """指数衰减。half_life_hours 表示衰减到 0.5 所需的小时数。"""
    h = _hours_ago(timestamp_sec)
    return max(0.01, 0.5 ** (h / half_life_hours))


def recency_factor(timestamp_sec: float, days_cap: int = 30) -> float:
    """时间近因性：越近越接近 1.0，越久越接近 0.0。"""
    age_days = max(0.0, (time.time() - timestamp_sec) / 86400.0)
    return 1.0 / (1.0 + age_days / days_cap)


# ---------------------------------------------------------------------------
# 2. 重要性权重：(0.8 + importance * 0.4)
# ---------------------------------------------------------------------------

def importance_weight(importance: float) -> float:
    """重要性权重：importance ∈ [0, 1] → 权重 ∈ [0.8, 1.2]。"""
    importance = max(0.0, min(1.0, importance))
    return 0.8 + importance * 0.4


# ---------------------------------------------------------------------------
# 3. 三层评分公式（完全按你给的）
# ---------------------------------------------------------------------------

def score_working(similarity: float,
                  timestamp_sec: float,
                  importance: float) -> float:
    """
    工作记忆：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)
    """
    return (similarity * time_decay(timestamp_sec, half_life_hours=6.0)) * \
           importance_weight(importance)


def score_episodic(vector_similarity: float,
                   timestamp_sec: float,
                   importance: float) -> float:
    """
    情景记忆：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)
    """
    base = vector_similarity * 0.8 + recency_factor(timestamp_sec) * 0.2
    return base * importance_weight(importance)


def score_semantic(vector_similarity: float,
                   graph_similarity: float,
                   importance: float) -> float:
    """
    语义记忆：(向量相似度 × 0.7 + 图相似度 × 0.3) × (0.8 + 重要性 × 0.4)
    """
    base = vector_similarity * 0.7 + graph_similarity * 0.3
    return base * importance_weight(importance)