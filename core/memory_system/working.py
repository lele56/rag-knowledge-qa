# core/memory_system/working.py
"""
工作记忆 (Working Memory)。

策略：直接复用你现有的 `core/memory.py`（ConversationBufferMemory 等）。
工作记忆是"最近几轮对话"，不需要持久化，所以：
  - 直接从 memory.load_memory_variables({}) 里把 chat_history 拿出来
  - 用简单关键词/TF-IDF 和当前 query 算相似度（不用调 embedding，轻量快速）
  - 再套 scoring.score_working() 得到最终分数
"""
from typing import List, Dict, Any, Tuple

from .scoring import score_working
from .config import cfg


# ---------------------------------------------------------------------------
# 极简相似度：关键词重叠 + 长度归一化
# （不调用 embedding 模型，确保快；工作记忆只看最近几轮，够用了）
# ---------------------------------------------------------------------------

def _lightweight_similarity(query: str, text: str) -> float:
    """Jaccard 相似度：query 和 text 的词元交集 / query 词元数"""
    if not query or not text:
        return 0.0
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(1.0, len(q_tokens))


def _extract_recent_dialogs() -> List[Dict[str, Any]]:
    """从记忆系统中提取最近对话，格式化为 [{question, answer, ts}]。"""
    try:
        from core.memory import get_memory
        mem = get_memory()
        variables = mem.load_memory_variables({})
        messages = variables.get("chat_history", [])
    except Exception:
        return []

    results = []
    now_ts = float(__import__("time").time())
    # 成对组合：(用户消息, 助手消息)
    pending_q = None
    for msg in messages:
        try:
            content = getattr(msg, "content", None) or str(msg)
            msg_type = type(msg).__name__
        except Exception:
            continue

        if "Human" in msg_type or "user" in msg_type.lower():
            pending_q = content
        elif "AI" in msg_type or "assistant" in msg_type.lower() and pending_q is not None:
            results.append({
                "question": pending_q,
                "answer": content,
                "ts": now_ts - len(results) * 60.0,  # 近似时间：越新越靠近现在
            })
            pending_q = None
    # 只保留最近 5 轮
    return results[-5:]


def recall_working(query: str) -> List[Tuple[float, str]]:
    """
    从工作记忆中检索与 query 相关的对话。
    返回 [(score, text), ...]，按分数降序。
    """
    if not cfg.ENABLED:
        return []

    dialogs = _extract_recent_dialogs()
    if not dialogs:
        return []

    scored: List[Tuple[float, str]] = []
    for d in dialogs:
        text = f"Q: {d['question']} A: {d['answer']}"
        sim = _lightweight_similarity(query, d["question"])
        score = score_working(similarity=sim,
                               timestamp_sec=d["ts"],
                               importance=0.4)  # 工作记忆 importance 给一个固定中间值
        if score > 0:
            scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:cfg.WORKING_TOP_K]  # 工作记忆取的条数