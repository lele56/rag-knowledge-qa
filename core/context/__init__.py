# core/context/__init__.py
# GSSC 上下文构建模块导出

from .types import ContextPacket, ContextConfig
from .strategies import (
    ContextStrategy,
    CompactStrategy,
    FullStrategy,
    EvidenceOnly,
    MultiDocStrategy,
    get_strategy,
)
from .builder import ContextBuilder, get_context_builder, build_rag_context

__all__ = [
    "ContextPacket",
    "ContextConfig",
    "ContextStrategy",
    "CompactStrategy",
    "FullStrategy",
    "EvidenceOnly",
    "MultiDocStrategy",
    "get_strategy",
    "ContextBuilder",
    "get_context_builder",
    "build_rag_context",
]