"""
RAG 专用工具 — 使用 LangChain @tool 装饰器自动生成 schema

工具列表:
- rag_search:   知识库检索
- doc_focus:    聚焦/切换文档
- list_docs:    列出已上传文档
- memory_recall: 回忆长期记忆
- memory_save:  保存到长期记忆
"""

from typing import Optional, Callable
from langchain_core.tools import tool

# 全局回调引用（由 Agent 在初始化时注入）
_retriever_fn: Optional[Callable] = None
_source_filter: Optional[Callable] = None
_focus_callback: Optional[Callable] = None
_list_callback: Optional[Callable] = None
_memory_manager = None
_SENTINEL = object()


def set_tool_deps(
    retriever_fn=_SENTINEL,
    source_filter=_SENTINEL,
    focus_callback=_SENTINEL,
    list_callback=_SENTINEL,
    memory_manager=_SENTINEL,
):
    """注入工具依赖（由 Agent 初始化时调用，传 None 可清除对应依赖）"""
    global _retriever_fn, _source_filter, _focus_callback, _list_callback, _memory_manager
    if retriever_fn is not _SENTINEL:
        _retriever_fn = retriever_fn
    if source_filter is not _SENTINEL:
        _source_filter = source_filter
    if focus_callback is not _SENTINEL:
        _focus_callback = focus_callback
    if list_callback is not _SENTINEL:
        _list_callback = list_callback
    if memory_manager is not _SENTINEL:
        _memory_manager = memory_manager


@tool
def rag_search(query: str, top_k: int = 8) -> str:
    """在知识库中搜索相关文档片段。适用场景：用户询问文档内容时。

    Args:
        query: 搜索查询词
        top_k: 返回结果数，默认8，多文档建议10+
    """
    if _retriever_fn is None:
        return "错误: 检索器未初始化，请先上传文档"

    try:
        kwargs = {"top_k": top_k}
        if _source_filter:
            try:
                focus = _source_filter()
                if focus:
                    from chains.source_detection import resolve_source_filter
                    enriched = resolve_source_filter("__USE_FOCUS__", focus)
                    kwargs["source_filter"] = enriched or focus
            except Exception:
                pass

        docs = _retriever_fn(query, **kwargs)
        if not docs:
            return "未找到相关文档。请尝试换一个关键词，或确认文档已上传。"

        parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            src = meta.get("source", "未知")
            section = meta.get("section", "")
            header = f"[{i}] 来源: {src}"
            if section:
                header += f" | 章节: {section}"
            parts.append(f"{header}\n{content[:500]}")

        return "\n\n".join(parts)
    except Exception as e:
        return f"检索失败: {e}"


@tool
def doc_focus(document_name: str) -> str:
    """聚焦到指定文档。适用场景：用户说'这篇文章'、'那个文档'需要先聚焦。

    Args:
        document_name: 文档名称或关键词
    """
    if not document_name:
        return "错误: 请提供文档名称"
    if _focus_callback is None:
        return "聚焦功能未配置"

    try:
        result = _focus_callback(document_name)
        return f"已聚焦到文档: {result or document_name}"
    except Exception as e:
        return f"聚焦失败: {e}"


@tool
def list_docs() -> str:
    """列出知识库中所有已上传的文档。适用场景：用户问'有哪些文档'、'知识库有什么'时。"""
    if _list_callback is None:
        return "功能未配置"

    try:
        docs = _list_callback()
        if not docs:
            return "知识库中暂无文档"
        return "知识库中的文档:\n" + "\n".join(f"- {d}" for d in docs)
    except Exception as e:
        return f"获取文档列表失败: {e}"


@tool
def memory_recall(query: str, limit: int = 3) -> str:
    """回忆长期记忆中的相关信息。适用场景：需要参考之前对话中的重要信息时。

    Args:
        query: 搜索关键词
        limit: 返回条数，默认3
    """
    if _memory_manager is None:
        return "记忆系统未初始化"

    try:
        results = _memory_manager.retrieve_long_term(query=query, top_k=limit)
        if not results:
            return "未找到相关记忆"
        formatted = []
        for r in results:
            if hasattr(r, 'content'):
                formatted.append(f"- {r.content}")
            elif isinstance(r, str):
                formatted.append(f"- {r}")
            else:
                formatted.append(f"- {r}")
        return "\n".join(formatted)
    except Exception as e:
        return f"记忆回忆失败: {e}"


@tool
def memory_save(content: str) -> str:
    """保存重要信息到长期记忆。适用场景：用户要求记住某些信息时。

    Args:
        content: 要保存的内容
    """
    if not content:
        return "错误: 请提供要保存的内容"
    if _memory_manager is None:
        return "记忆系统未初始化"

    try:
        if hasattr(_memory_manager, 'add_to_working'):
            _memory_manager.add_to_working(content, "(系统保存)")
        return f"已保存: {content[:100]}"
    except Exception as e:
        return f"保存失败: {e}"