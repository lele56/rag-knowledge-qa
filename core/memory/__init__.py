# core/memory/__init__.py
"""记忆模块 — 短期记忆 + 记忆管理器

用法:
    from core.memory import get_memory, clear_memory, get_chat_history_as_text
    from core.memory import get_memory_manager, MemoryManager, MemoryItem
"""

from core.memory.short_term import (
    get_memory,
    clear_memory,
    get_chat_history_as_text,
    ConversationMemory,
)
from core.memory.manager import (
    get_memory_manager,
    MemoryManager,
    MemoryItem,
)

__all__ = [
    "get_memory",
    "clear_memory",
    "get_chat_history_as_text",
    "ConversationMemory",
    "get_memory_manager",
    "MemoryManager",
    "MemoryItem",
]