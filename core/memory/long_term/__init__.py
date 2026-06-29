# core/memory/long_term/__init__.py
# 长期记忆系统导出

from typing import List, Optional, Set, Dict

from .config import MemoryConfig, cfg
from .scoring import score_working, score_episodic, score_semantic
from .working import recall_working
from .semantic import recall_semantic, store_semantic
from .episodic import recall_episodic, store_episodic
from utils.logger import logger


class MemorySystem:
    """长期记忆系统统一入口：封装 episodic + semantic 的 store/recall。

    设计原则：
      - 延迟初始化：不检查依赖，直到首次 store/recall 调用
      - 健康检查：首次操作前验证 Neo4j + Qdrant 可用性，缓存结果
      - 优雅降级：单组件失败不影响其他组件，仅记录警告
    """

    def __init__(self):
        self._health: Optional[bool] = None          # None=未检查, True=健康, False=不健康
        self._health_detail: Dict[str, bool] = {}    # 各组件健康状态

    def health_check(self) -> bool:
        """验证 Neo4j + Qdrant 是否可用，结果缓存 300 秒。"""
        import time as _time
        now = _time.time()
        if self._health is not None and hasattr(self, "_health_ts"):
            if now - self._health_ts < 300:
                return self._health

        self._health_detail = {}

        # 检查 Neo4j
        try:
            from core.infrastructure.graph_store import get_graph
            graph = get_graph()
            graph.query("RETURN 1 AS ok LIMIT 1")
            self._health_detail["neo4j"] = True
        except Exception as e:
            self._health_detail["neo4j"] = False
            logger.warning(f"长期记忆 → Neo4j 不可用: {e}")

        # 检查 Qdrant
        try:
            from core.infrastructure.vector_store import _get_client
            client = _get_client()
            client.get_collections()
            self._health_detail["qdrant"] = True
        except Exception as e:
            self._health_detail["qdrant"] = False
            logger.warning(f"长期记忆 → Qdrant 不可用: {e}")

        self._health = all(self._health_detail.values()) if self._health_detail else False
        self._health_ts = now
        return self._health

    @property
    def healthy(self) -> bool:
        return self._health if self._health is not None else self.health_check()

    def store(self, question: str, answer: str, sources: Optional[List[str]] = None) -> None:
        """同时写入情景记忆和语义记忆。"""
        if not cfg.ENABLED:
            return
        if not self.healthy:
            logger.warning("长期记忆系统不健康，跳过 store")
            return
        if self._health_detail.get("qdrant", True):
            try:
                store_episodic(question, answer, sources=sources)
            except Exception as e:
                logger.warning(f"情景记忆写入失败: {e}")
        if self._health_detail.get("neo4j", True):
            try:
                store_semantic(question, answer)
            except Exception as e:
                logger.warning(f"语义记忆写入失败: {e}")

    def recall(self, query: str, source_filter: Optional[Set[str]] = None) -> str:
        """从长期记忆中检索相关内容，返回格式化文本。"""
        if not cfg.ENABLED:
            return ""
        if not self.healthy:
            return ""

        parts: List[str] = []

        try:
            working = recall_working(query)
            if working:
                parts.append("【工作记忆】\n" + "\n".join(t for _, t in working[:2]))
        except Exception as e:
            logger.warning(f"工作记忆检索失败: {e}")

        if self._health_detail.get("qdrant", True):
            try:
                episodic = recall_episodic(query, source_filter=source_filter)
                if episodic:
                    parts.append("【情景记忆】\n" + "\n".join(t for _, t in episodic[:3]))
            except Exception as e:
                logger.warning(f"情景记忆检索失败: {e}")

        if self._health_detail.get("neo4j", True):
            try:
                semantic = recall_semantic(query)
                if semantic:
                    parts.append("【语义记忆】\n" + "\n".join(t for _, t in semantic[:3]))
            except Exception as e:
                logger.warning(f"语义记忆检索失败: {e}")

        return "\n\n".join(parts) if parts else ""


_memory_system: Optional[MemorySystem] = None


def get_memory_system() -> MemorySystem:
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem()
        logger.info("长期记忆系统已初始化")
    return _memory_system


__all__ = [
    "MemoryConfig",
    "MemorySystem",
    "get_memory_system",
    "score_working",
    "score_episodic",
    "score_semantic",
    "recall_working",
    "recall_semantic",
    "recall_episodic",
    "store_episodic",
    "store_semantic",
]