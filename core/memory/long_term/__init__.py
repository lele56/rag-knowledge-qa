# core/memory/long_term/__init__.py
# 长期记忆系统导出

from typing import List, Optional, Set

from .config import MemoryConfig, cfg
from .scoring import score_working, score_episodic, score_semantic
from .working import recall_working
from .semantic import recall_semantic, store_semantic
from .episodic import recall_episodic, store_episodic
from utils.logger import logger


class MemorySystem:
    """长期记忆系统统一入口：封装 episodic + semantic 的 store/recall。"""

    def store(self, question: str, answer: str, sources: Optional[List[str]] = None) -> None:
        """同时写入情景记忆和语义记忆。"""
        if not cfg.ENABLED:
            return
        try:
            store_episodic(question, answer, sources=sources)
        except Exception as e:
            logger.warning(f"情景记忆写入失败: {e}")
        try:
            store_semantic(question, answer)
        except Exception as e:
            logger.warning(f"语义记忆写入失败: {e}")

    def recall(self, query: str, source_filter: Optional[Set[str]] = None) -> str:
        """从长期记忆中检索相关内容，返回格式化文本。"""
        if not cfg.ENABLED:
            return ""

        parts: List[str] = []

        try:
            working = recall_working(query)
            if working:
                parts.append("【工作记忆】\n" + "\n".join(t for _, t in working[:2]))
        except Exception as e:
            logger.warning(f"工作记忆检索失败: {e}")

        try:
            episodic = recall_episodic(query, source_filter=source_filter)
            if episodic:
                parts.append("【情景记忆】\n" + "\n".join(t for _, t in episodic[:3]))
        except Exception as e:
            logger.warning(f"情景记忆检索失败: {e}")

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
    "store_semantic",
    "recall_episodic",
    "store_episodic",
]