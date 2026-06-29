# core/memory/manager.py
"""
【MemoryManager：统一记忆入口】

设计原则：
  - 短期记忆 = LangChain ConversationBufferWindowMemory
  - 长期记忆 = MemorySystem（episodic: Qdrant + semantic: Neo4j）
"""
from typing import Optional, List, Set, Any
from dataclasses import dataclass, field

from utils.logger import logger
from config.settings import settings


@dataclass
class MemoryItem:
    content: str
    sources: List[str] = field(default_factory=list)
    score: float = 0.0

    def format(self) -> str:
        srcs = ", ".join(self.sources) if self.sources else "（来源不明）"
        if len(self.content) > 300:
            return f"{self.content[:297]}... [来源: {srcs}]"
        return f"{self.content} [来源: {srcs}]"


class MemoryManager:
    """记忆管理器：短期走 LangChain，长期走 MemorySystem。"""

    def __init__(self):
        self._lts = None
        logger.info("🧠 MemoryManager 已初始化 (short-term → LangChain, long-term → MemorySystem)")

    def _ensure_lts(self) -> Optional[Any]:
        if self._lts is None:
            try:
                from core.memory.long_term import get_memory_system
                self._lts = get_memory_system()
                if self._lts.healthy:
                    logger.info("🧠 长期记忆系统已连接 (episodic + semantic)")
                else:
                    logger.warning("🧠 长期记忆系统已连接，但部分后端不可用（仅工作记忆生效）")
            except Exception as e:
                logger.warning(f"长期记忆系统不可用: {e}")
                self._lts = False
        return self._lts if self._lts is not False else None

    def remember(self, question: str, answer: str, save_long: bool = False) -> None:
        if not question or not answer:
            return
        try:
            from core.memory.short_term import get_memory
            get_memory().save_context({"input": question}, {"answer": answer})
        except Exception as e:
            logger.warning(f"写入短期记忆失败: {e}")

        if save_long:
            self.save_long_term(question, answer)
        elif settings.DEBUG:
            logger.debug("🧠 短期记忆 +1")

    def clear_short_term(self) -> None:
        from core.memory.short_term import clear_memory
        clear_memory()

    def save_long_term(self, question: str, answer: str, sources: Optional[List[str]] = None) -> bool:
        lts = self._ensure_lts()
        if not lts:
            return False
        try:
            lts.store(question, answer, sources=sources)
            return True
        except Exception as e:
            logger.warning(f"长期记忆写入失败: {e}")
            return False

    def retrieve_long_term(
        self,
        query: str,
        source_filter: Optional[Set[str]] = None,
        top_k: int = 3,
    ) -> List[MemoryItem]:
        lts = self._ensure_lts()
        if not lts:
            return []
        try:
            text = lts.recall(query, source_filter=source_filter)
            if not text or not text.strip() or "暂无相关" in text or "检索失败" in text:
                return []
            results: List[MemoryItem] = []
            for i, para in enumerate([p for p in text.strip().split("\n") if p.strip()][:top_k]):
                results.append(MemoryItem(content=para, score=0.5 - i * 0.05))
            return results
        except Exception as e:
            logger.warning(f"长期记忆检索失败: {e}")
            return []


_mm_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _mm_instance
    if _mm_instance is None:
        _mm_instance = MemoryManager()
    return _mm_instance