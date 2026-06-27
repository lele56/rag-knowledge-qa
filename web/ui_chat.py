# web/ui_chat.py
"""Tab 1: 智能问答 — CSS 样式 + 聊天事件处理"""

import re
from pathlib import Path
from typing import List, Union, Dict, Optional
from datetime import datetime

from services.qa_service import get_qa_service
from core.memory import get_chat_history_as_text, clear_memory, get_memory
from utils.logger import logger

# ============================================================
# CSS 样式
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
.header-bar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px 28px; border-radius: 12px; margin-bottom: 16px;
    color: white; display: flex; justify-content: space-between; align-items: center;
}
.header-bar h1 { color: white !important; margin: 0 !important; font-size: 1.6em !important; }
.header-bar .subtitle { opacity: 0.85; font-size: 0.9em; margin-top: 4px; }
.status-badge {
    padding: 6px 16px; border-radius: 20px; font-size: 0.85em; font-weight: 600;
}
.status-ok { background: #22c55e22; color: #16a34a; border: 1px solid #22c55e44; }
.status-warn { background: #f59e0b22; color: #d97706; border: 1px solid #f59e0b44; }
.status-err { background: #ef444422; color: #dc2626; border: 1px solid #ef444444; }
.upload-card {
    border: 2px dashed #d1d5db; border-radius: 12px; padding: 24px;
    text-align: center; transition: all 0.2s;
}
.upload-card:hover { border-color: #667eea; background: #f8f7ff; }
.file-table { max-height: 320px; overflow-y: auto; }
.stat-card {
    padding: 16px 20px; border-radius: 10px; background: #f8fafc;
    border: 1px solid #e2e8f0; text-align: center;
}
.stat-card .value { font-size: 1.8em; font-weight: 700; color: #667eea; }
.stat-card .label { font-size: 0.85em; color: #64748b; margin-top: 2px; }
.footer { text-align: center; color: #94a3b8; font-size: 0.8em; padding: 16px 0; }
::selection { background: #667eea44; }
"""

# ============================================================
# 全局状态
# ============================================================
qa = None
doc_svc = None


# ============================================================
# 辅助函数
# ============================================================
def _file_to_path(f: Union[str, dict, None]) -> Optional[Path]:
    if f is None:
        return None
    if isinstance(f, dict):
        p = f.get("path") or f.get("name")
        return Path(p) if p else None
    return Path(str(f))


def _render_uploaded_list(records: List[Dict]) -> str:
    if not records:
        return "*暂无上传记录*"
    sorted_records = sorted(records, key=lambda r: r["time"], reverse=True)
    lines = []
    for idx, r in enumerate(sorted_records, start=1):
        lines.append(f"- **{idx}.** `{r['name']}` — {r['status']} （{r.get('chunks', '-')} chunks, {r['time']}）")
    return "\n".join(lines)


# ============================================================
# Tab 1: 智能问答
# ============================================================
async def chat_respond(message, history):
    global qa
    if not message or not message.strip():
        yield "", history
        return

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    try:
        from core.agent.rag_agent import get_rag_agent
        agent = get_rag_agent()
        collected_sources = set()

        async for step in agent.astream(message):
            if step.type.value == "thought":
                history[-1]["content"] = f"🤔 **思考中**: {step.content[:120]}..."
                yield "", history
            elif step.type.value == "action":
                history[-1]["content"] = f"🔍 **检索**: `{step.tool_input}`"
                yield "", history
            elif step.type.value == "observation":
                if step.tool_result:
                    for m in re.finditer(r"来源[：:]\s*([^\n]+)", step.tool_result):
                        collected_sources.add(m.group(1).strip())
                    for m in re.finditer(r"source[：:]\s*([^\n]+)", step.tool_result, re.IGNORECASE):
                        collected_sources.add(m.group(1).strip())
                history[-1]["content"] = "📋 分析结果中..."
                yield "", history
            elif step.type.value == "finish":
                full_response = step.content or ""
                if collected_sources:
                    full_response += "\n\n---\n📚 **来源参考**\n" + "\n".join(f"- {s}" for s in sorted(collected_sources))
                history[-1]["content"] = full_response
                try:
                    get_memory().save_context({"input": message}, {"answer": full_response})
                except Exception as e:
                    logger.debug(f"保存对话上下文失败: {e}")
                yield "", history
                return

        history[-1]["content"] = "⚠️ 无法在限定步数内完成，请尝试更具体的问题。"
        yield "", history
    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        if qa is None:
            qa = get_qa_service()
        result = await qa.ask(message)
        answer = result.get("answer", "系统错误，请重试。")
        if result.get("sources"):
            answer += "\n\n---\n📚 **来源参考**\n" + "\n".join(f"- {s}" for s in set(result["sources"]))
        history[-1]["content"] = answer
        yield "", history


def chat_retry(message, history):
    if history and len(history) >= 2:
        history = history[:-2]
    return message, history or []


def chat_undo(history):
    if history and len(history) >= 2:
        message = history[-2]["content"]
        history = history[:-2]
        return message, history
    return "", history or []


def refresh_history():
    try:
        return get_chat_history_as_text()
    except Exception as e:
        logger.debug(f"刷新对话历史失败: {e}")
        return "_暂无对话记录_"