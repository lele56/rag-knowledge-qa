# web/ui_admin.py
"""Tab 3 + Tab 4: 文档管理 + 系统状态/调试"""

from datetime import datetime

from services.document_service import get_document_service
from core.infrastructure.vector_store import get_qdrant_status
from core.infrastructure.graph_store import get_graph_status
from core.memory import clear_memory, get_chat_history_as_text
from utils.logger import logger


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
        except Exception as e:
            logger.debug(f"清空会话文档失败: {e}")
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

    try:
        bg_errors = get_document_service().get_bg_errors()
    except Exception as e:
        logger.debug(f"获取后台错误失败: {e}")
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
        from core.retrievers.factory import get_retriever
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