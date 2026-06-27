# core/memory/short_term.py
"""短期记忆 — 基于 LangChain InMemoryChatMessageHistory（非 deprecated）。

窗口模式：最近 N 轮对话，内存中滚动。
长期记忆：MemorySystem（episodic: Qdrant + semantic: Neo4j）
"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from config.settings import settings
from utils.logger import logger


class ConversationMemory:
    def __init__(self, window_size: int = 6):
        self._store = InMemoryChatMessageHistory()
        self._window_size = window_size

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self._store.add_message(HumanMessage(content=inputs.get("input", "")))
        self._store.add_message(AIMessage(content=outputs.get("answer", "")))
        self._trim()

    def load_memory_variables(self, _=None) -> dict:
        return {"chat_history": list(self._store.messages)}

    def clear(self) -> None:
        self._store.clear()

    def _trim(self) -> None:
        max_msgs = self._window_size * 2
        msgs = self._store.messages
        if len(msgs) > max_msgs:
            self._store.messages = msgs[-max_msgs:]


_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is not None:
        return _memory
    _memory = ConversationMemory(window_size=settings.MEMORY_WINDOW_SIZE)
    logger.info(f"短期记忆: window (最近 {settings.MEMORY_WINDOW_SIZE} 轮)")
    return _memory


def clear_memory() -> None:
    global _memory
    _memory = None
    logger.info("短期记忆已清空")


def get_chat_history_as_text() -> str:
    try:
        mem = get_memory()
    except Exception as e:
        return f"_读取历史记录失败: {e}_"

    try:
        variables = mem.load_memory_variables()
    except Exception as e:
        return f"_读取失败: {e}_"

    messages = variables.get("chat_history", [])
    if not messages:
        return "_暂无对话记录_"

    lines = []
    for idx, msg in enumerate(messages, start=1):
        try:
            content = getattr(msg, "content", str(msg))
            msg_type = type(msg).__name__
            if "Human" in msg_type or "user" in msg_type.lower():
                role = "🧑 提问"
            elif "AI" in msg_type or "assistant" in msg_type.lower():
                role = "🤖 回答"
            else:
                role = f"💬 {msg_type}"
            lines.append(f"**{idx}. {role}**\n\n{content}\n")
        except Exception as e:
            logger.debug(f"消息格式化失败: {e}")
            lines.append(f"**{idx}.** {msg}\n")

    if not lines:
        return "_暂无对话记录_"
    return "\n---\n\n".join(lines)