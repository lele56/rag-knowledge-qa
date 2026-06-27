# core/memory/long_term/config.py
"""记忆系统的配置（从 settings.py 读取，集中一处）。"""
from config.settings import settings


class MemoryConfig:
    """三层记忆系统的配置，全部在这里。"""
    # 总开关
    ENABLED: bool = settings.LS_ENABLED

    # 工作记忆 (Working)
    WORKING_TOP_K: int = settings.LS_WORKING_TOP_K

    # 情景记忆 (Episodic) — Qdrant
    EPISODIC_COLLECTION: str = settings.LS_EPISODIC_COLLECTION
    EPISODIC_TOP_K: int = settings.LS_EPISODIC_TOP_K
    EPISODIC_MAX_STORE: int = settings.LS_EPISODIC_MAX_STORE

    # 语义记忆 (Semantic) — Neo4j
    SEMANTIC_TOP_K: int = settings.LS_SEMANTIC_TOP_K
    SEMANTIC_NEIGHBOR: int = settings.LS_SEMANTIC_NEIGHBOR

    # 重要性 (0~1)
    IMPORTANCE_INIT: float = settings.LS_IMPORTANCE_INIT
    IMPORTANCE_GROWTH: float = settings.LS_IMPORTANCE_GROWTH
    FORGET_THRESHOLD: float = settings.LS_FORGET_THRESHOLD

    # 最终合并输出
    RECALL_TOTAL: int = settings.LS_RECALL_TOTAL


cfg = MemoryConfig()