# web/ui_handlers.py
"""Gradio UI 的 CSS 样式 + 各 Tab 的事件处理函数

从 gradio_app.py 中抽离，避免单个文件代码堆积。
"""
import re
from pathlib import Path
from typing import List, Union, Dict, Optional
from datetime import datetime

from services.qa_service import get_qa_service
from services.document_service import get_document_service
from core.vector_store import get_qdrant_status
from core.graph_store import get_graph_status
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
                except Exception:
                    pass
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
    """重新生成：移除最后一条助手回复，重新触发"""
    if history and len(history) >= 2:
        history = history[:-2]  # 移除 user + assistant
    return message, history or []


def chat_undo(history):
    """撤销最后一条回复"""
    if history and len(history) >= 2:
        message = history[-2]["content"]
        history = history[:-2]
        return message, history
    return "", history or []


# ============================================================
# Tab 2: 上传文件
# ============================================================
async def upload_files(files, records):
    global doc_svc
    if records is None:
        records = []
    if not files:
        return "⚠️ 请先选择文件", _render_uploaded_list(records), None, records

    paths, file_names = [], []
    for f in files:
        p = _file_to_path(f)
        if p and p.exists():
            paths.append(p)
            file_names.append(p.name)

    if not paths:
        return "❌ 没有可处理的文件", _render_uploaded_list(records), None, records

    try:
        if doc_svc is None:
            doc_svc = get_document_service()
        time_str = datetime.now().strftime("%H:%M:%S")

        result = await doc_svc.add_documents_async(paths)
        chunk_count = result.get("chunk_count", 0) if isinstance(result, dict) else result
        duplicates = result.get("duplicates", []) if isinstance(result, dict) else []
        new_files = result.get("new_files", file_names) if isinstance(result, dict) else file_names
        focus_sources = result.get("focus_sources", []) if isinstance(result, dict) else []

        # 将文档加入 agent 的会话文档列表（含重复文件——它们已有 chunks）
        if focus_sources:
            try:
                from core.agent.rag_agent import get_rag_agent
                get_rag_agent().set_focus(set(focus_sources))
            except Exception:
                pass

        new_records = list(records)

        # 新文件 → "⏳ 后台写入中"
        for name in new_files:
            new_records.append({"name": name, "status": "⏳ 后台写入中", "chunks": chunk_count, "time": time_str})

        # 重复文件 → "♻️ 已存在，跳过写入"
        for name in duplicates:
            new_records.append({"name": name, "status": "♻️ 已存在（跳过写入，已加入会话列表）", "chunks": 0, "time": time_str})

        # 组装状态信息
        parts = []
        if new_files:
            parts.append(f"✅ **已接收 {len(new_files)} 个新文件**  |  {chunk_count} chunks")
        if duplicates:
            parts.append(f"♻️ **跳过 {len(duplicates)} 个重复文件**（已加入会话列表）")
        status = "  |  ".join(parts) if parts else "⚠️ 未处理任何文件"

        return status, _render_uploaded_list(new_records), None, new_records
    except Exception as e:
        logger.error(f"上传失败: {e}")
        time_str = datetime.now().strftime("%H:%M:%S")
        new_records = list(records)
        for name in file_names:
            new_records.append({"name": name, "status": "❌ 失败", "chunks": 0, "time": time_str})
        return f"❌ **上传失败**: `{e}`", _render_uploaded_list(new_records), None, new_records


def refresh_upload_status(records):
    """刷新上传记录状态：检查后台写入是否完成/失败，标记真实状态。"""
    global doc_svc
    if not records:
        return "*暂无上传记录*", records

    if doc_svc is None:
        doc_svc = get_document_service()

    # 获取后台写入失败记录
    bg_errors = doc_svc.get_bg_errors()
    failed_files = set()
    failed_info = {}
    for err in bg_errors:
        for f in err["files"]:
            failed_files.add(f)
            failed_info[f] = err["error"][:100]

    # 获取知识库中已注册的文件
    kb_files = set()
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        kb_files = set(reg.get_all_sources())
    except Exception:
        pass

    # 更新每条记录的状态
    updated = []
    for r in records:
        name = r.get("name", "")
        if name in failed_files:
            r = dict(r)
            r["status"] = f"❌ 写入失败: {failed_info[name]}"
        elif name in kb_files:
            r = dict(r)
            r["status"] = "✅ 已入库"
        elif r.get("status", "").startswith("⏳"):
            r = dict(r)
            r["status"] = "⏳ 后台写入中"
        updated.append(r)

    # 统计
    ok_count = sum(1 for r in updated if r.get("status", "").startswith("✅"))
    fail_count = sum(1 for r in updated if r.get("status", "").startswith("❌"))
    pending_count = sum(1 for r in updated if r.get("status", "").startswith("⏳"))

    parts = [f"📊 已入库 {ok_count} 篇"]
    if fail_count:
        parts.append(f"❌ 失败 {fail_count} 篇")
    if pending_count:
        parts.append(f"⏳ 处理中 {pending_count} 篇")
    status = "  |  ".join(parts)

    return _render_uploaded_list(updated), updated


# ============================================================
# Tab 3: 文档管理
# ============================================================
def get_doc_stats() -> str:
    stats = get_qdrant_status()
    graph = get_graph_status()
    return (
        f"| 项目 | 数值 |\n|---|---|\n"
        f"| Qdrant 向量库 | {stats['points']} chunks |\n"
        f"| Neo4j 图库 | {graph['nodes']} 节点 |\n"
    )


def list_all_docs() -> str:
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        sources = sorted(get_doc_id_registry().get_all_sources())
        if not sources:
            return "_知识库中暂无文档_"
        return "\n".join(f"- {s}" for s in sources)
    except Exception as e:
        return f"_读取失败: {e}_"


def clear_memory_action() -> str:
    try:
        clear_memory()
        try:
            from core.agent.rag_agent import get_rag_agent
            get_rag_agent().clear_session_docs()
        except Exception:
            pass
        return "✅ 对话记忆已清空，会话文档列表已重置"
    except Exception as e:
        return f"❌ 清空失败: {e}"


# ============================================================
# Tab 4: 系统状态 + 调试
# ============================================================
def refresh_status():
    q = get_qdrant_status()
    n = get_graph_status()
    now = datetime.now().strftime("%H:%M:%S")

    # 检查后台写入失败
    try:
        bg_errors = get_document_service().get_bg_errors()
    except Exception:
        bg_errors = []

    bg_warn = ""
    if bg_errors:
        failed_files = []
        for err in bg_errors:
            failed_files.append(", ".join(err["files"]))
        bg_warn = f"  ·  ⚠️ 后台写入失败: {', '.join(failed_files[:2])}"

    ok_count = sum([q["ok"], n["ok"]])
    if ok_count == 2:
        badge = '<span class="status-badge status-ok">✅ 全部正常</span>'
    elif ok_count == 1:
        badge = '<span class="status-badge status-warn">⚠️ 部分异常</span>'
    else:
        badge = '<span class="status-badge status-err">❌ 服务异常</span>'
    return f"{badge}  {q['msg']}  ·  {n['msg']}  ·  _{now}_{bg_warn}"


def debug_retrieve(keyword: str):
    if not keyword or not keyword.strip():
        return "_请输入测试关键词_"
    try:
        from core.retriever_factory import get_retriever
        docs = get_retriever().invoke(keyword.strip())
        if not docs:
            return "⚠️ **检索到 0 条** — 可能还没上传文档或 chunks 尚未入库"
        lines = [f"✅ **检索到 {len(docs)} 条** (关键词: `{keyword}`)\n"]
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "未知")
            sec = d.metadata.get("section", "")
            score = d.metadata.get("_score", "")
            content = d.page_content[:300].replace("\n", " ")
            meta_line = f"来源: {src}" + (f" · 章节: {sec}" if sec else "") + (f" · 得分: {score:.3f}" if score else "")
            lines.append(f"### [{i}] {meta_line}\n> {content}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ **检索报错**: `{e}`"


def refresh_history():
    try:
        return get_chat_history_as_text()
    except Exception:
        return "_暂无对话记录_"