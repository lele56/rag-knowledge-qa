"""
上下文策略 — 不同场景使用不同的上下文构建方式

- CompactStrategy: 紧凑模式，适合简单问答
- FullStrategy:     完整模式，包含所有信息
- EvidenceOnly:     仅证据模式，不含对话历史
- MultiDocStrategy: 多文档对比模式
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document
from .types import ContextPacket, ContextConfig
from .builder import ContextBuilder


class ContextStrategy:
    """上下文策略基类"""

    def __init__(self, builder: ContextBuilder):
        self.builder = builder

    def build(
        self,
        user_query: str,
        retrieved_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        long_term_memories: Optional[List[str]] = None,
    ) -> str:
        raise NotImplementedError


class CompactStrategy(ContextStrategy):
    """紧凑模式 — 仅包含检索证据 + 系统指令，适合简单问答"""

    def __init__(self, builder: ContextBuilder):
        super().__init__(builder)
        self.builder.config.max_tokens = 3000
        self.builder.config.max_retrieval_chunks = 5
        self.builder.config.max_history_turns = 2
        self.builder.config.max_memory_items = 2

    def build(
        self,
        user_query: str,
        retrieved_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        long_term_memories: Optional[List[str]] = None,
    ) -> str:
        return self.builder.build(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            chat_history=chat_history,
            long_term_memories=long_term_memories,
            system_instructions="简洁回答，基于文档证据。",
        )


class FullStrategy(ContextStrategy):
    """完整模式 — 包含所有上下文信息，适合复杂分析"""

    def __init__(self, builder: ContextBuilder):
        super().__init__(builder)
        self.builder.config.max_tokens = 8000
        self.builder.config.max_retrieval_chunks = 12
        self.builder.config.max_history_turns = 10
        self.builder.config.max_memory_items = 8

    def build(
        self,
        user_query: str,
        retrieved_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        long_term_memories: Optional[List[str]] = None,
    ) -> str:
        return self.builder.build(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            chat_history=chat_history,
            long_term_memories=long_term_memories,
            system_instructions="你是一个知识库问答助手，请详细分析并回答用户问题。",
        )


class EvidenceOnly(ContextStrategy):
    """仅证据模式 — 不包含对话历史，只基于检索结果"""

    def __init__(self, builder: ContextBuilder):
        super().__init__(builder)
        self.builder.config.max_tokens = 4000
        self.builder.config.max_history_turns = 0

    def build(
        self,
        user_query: str,
        retrieved_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        long_term_memories: Optional[List[str]] = None,
    ) -> str:
        return self.builder.build(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            chat_history=None,  # 不包含对话历史
            long_term_memories=long_term_memories,
            system_instructions="基于文档证据回答，不要编造。",
        )


class MultiDocStrategy(ContextStrategy):
    """多文档对比模式 — 适合对比多篇文档的场景"""

    def __init__(self, builder: ContextBuilder):
        super().__init__(builder)
        self.builder.config.max_tokens = 6000
        self.builder.config.max_retrieval_chunks = 10
        self.builder.config.enable_mmr = True
        self.builder.config.mmr_lambda = 0.5  # 偏向多样性（多文档对比）

    def build(
        self,
        user_query: str,
        retrieved_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        long_term_memories: Optional[List[str]] = None,
    ) -> str:
        return self.builder.build(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            chat_history=chat_history,
            long_term_memories=long_term_memories,
            system_instructions="你是一个知识库问答助手，请对比分析不同文档的观点。",
        )


# 策略工厂
def get_strategy(name: str) -> ContextStrategy:
    """获取上下文策略

    Args:
        name: 策略名称 (compact, full, evidence, multidoc)
    """
    builder = ContextBuilder()
    strategies = {
        "compact": CompactStrategy,
        "full": FullStrategy,
        "evidence": EvidenceOnly,
        "multidoc": MultiDocStrategy,
    }
    cls = strategies.get(name, FullStrategy)
    return cls(builder)