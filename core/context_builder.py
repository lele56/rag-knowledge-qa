# core/context_builder.py
"""向后兼容：从 core/context/ 子模块重新导出

实际代码已拆分到：
  - core/context/types.py    — ContextPacket, ContextConfig
  - core/context/builder.py  — ContextBuilder, build_rag_context
"""
from core.context.types import ContextPacket, ContextConfig
from core.context.builder import ContextBuilder, get_context_builder, build_rag_context

__all__ = [
    "ContextPacket",
    "ContextConfig",
    "ContextBuilder",
    "get_context_builder",
    "build_rag_context",
]