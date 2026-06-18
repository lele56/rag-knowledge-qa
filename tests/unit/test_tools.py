"""工具系统单元测试 — 适配 LangChain @tool 装饰器"""

import pytest


# ============================================================
# 工具依赖重置 fixture
# ============================================================

@pytest.fixture(autouse=True)
def _reset_tool_deps():
    """每个测试前后重置工具依赖，避免全局状态污染"""
    from core.tools.rag_tools import set_tool_deps
    set_tool_deps(
        retriever_fn=None,
        source_filter=None,
        focus_callback=None,
        list_callback=None,
        memory_manager=None,
    )
    yield
    set_tool_deps(
        retriever_fn=None,
        source_filter=None,
        focus_callback=None,
        list_callback=None,
        memory_manager=None,
    )


# ============================================================
# RAG 工具测试
# ============================================================

class TestRAGSearchTool:
    def test_basic_properties(self):
        from core.tools.rag_tools import rag_search
        assert rag_search.name == "rag_search"
        assert "搜索" in rag_search.description

    def test_run_without_retriever(self):
        from core.tools.rag_tools import rag_search
        result = rag_search.invoke({"query": "test"})
        assert "未初始化" in result

    def test_run_with_retriever(self):
        from core.tools.rag_tools import rag_search, set_tool_deps
        from langchain_core.documents import Document

        def mock_retrieve(query, **kwargs):
            return [Document(page_content="测试内容", metadata={"source": "test.pdf"})]

        set_tool_deps(retriever_fn=mock_retrieve)
        result = rag_search.invoke({"query": "test"})
        assert "测试内容" in result

    def test_run_empty_result(self):
        from core.tools.rag_tools import rag_search, set_tool_deps

        def mock_retrieve(query, **kwargs):
            return []

        set_tool_deps(retriever_fn=mock_retrieve)
        result = rag_search.invoke({"query": "test"})
        assert "未找到" in result


class TestDocFocusTool:
    def test_basic_properties(self):
        from core.tools.rag_tools import doc_focus
        assert doc_focus.name == "doc_focus"

    def test_run(self):
        from core.tools.rag_tools import doc_focus, set_tool_deps
        set_tool_deps(focus_callback=lambda name: name)
        result = doc_focus.invoke({"document_name": "test_doc.pdf"})
        assert "聚焦" in result
        assert "test_doc.pdf" in result

    def test_run_empty(self):
        from core.tools.rag_tools import doc_focus
        result = doc_focus.invoke({"document_name": ""})
        assert "错误" in result or "文档名称" in result

    def test_run_no_callback(self):
        from core.tools.rag_tools import doc_focus
        result = doc_focus.invoke({"document_name": "doc.pdf"})
        assert "未配置" in result


class TestListDocsTool:
    def test_basic_properties(self):
        from core.tools.rag_tools import list_docs
        assert list_docs.name == "list_docs"

    def test_run_with_docs(self):
        from core.tools.rag_tools import list_docs, set_tool_deps
        set_tool_deps(list_callback=lambda: ["doc1.pdf", "doc2.pdf"])
        result = list_docs.invoke({})
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    def test_run_empty(self):
        from core.tools.rag_tools import list_docs, set_tool_deps
        set_tool_deps(list_callback=lambda: [])
        result = list_docs.invoke({})
        assert "暂无" in result

    def test_run_no_callback(self):
        from core.tools.rag_tools import list_docs
        result = list_docs.invoke({})
        assert "未配置" in result


# ============================================================
# 记忆工具测试
# ============================================================

class TestMemoryRecallTool:
    def test_basic_properties(self):
        from core.tools.rag_tools import memory_recall
        assert memory_recall.name == "memory_recall"

    def test_run(self):
        from core.tools.rag_tools import memory_recall, set_tool_deps

        class FakeMemory:
            def retrieve_long_term(self, query, top_k):
                return ["记忆1", "记忆2"]

        set_tool_deps(memory_manager=FakeMemory())
        result = memory_recall.invoke({"query": "test"})
        assert "记忆1" in result

    def test_run_no_manager(self):
        from core.tools.rag_tools import memory_recall
        result = memory_recall.invoke({"query": "test"})
        assert "未初始化" in result


class TestMemorySaveTool:
    def test_basic_properties(self):
        from core.tools.rag_tools import memory_save
        assert memory_save.name == "memory_save"

    def test_run(self):
        from core.tools.rag_tools import memory_save, set_tool_deps

        class FakeMemory:
            def add_to_working(self, content, source):
                pass

        set_tool_deps(memory_manager=FakeMemory())
        result = memory_save.invoke({"content": "重要信息"})
        assert "已保存" in result

    def test_run_no_manager(self):
        from core.tools.rag_tools import memory_save
        result = memory_save.invoke({"content": "test"})
        assert "未初始化" in result


# ============================================================
# 工具注册表完整流程测试
# ============================================================

class TestToolRegistryIntegration:
    def test_register_multiple_tools(self):
        from core.tools.registry import ToolRegistry
        from core.tools.rag_tools import (
            rag_search, doc_focus, list_docs,
            memory_recall, memory_save, set_tool_deps,
        )
        from langchain_core.documents import Document

        set_tool_deps(
            retriever_fn=lambda q, **kw: [Document(page_content="test", metadata={})],
            focus_callback=lambda name: name,
            list_callback=lambda: ["doc.pdf"],
        )

        registry = ToolRegistry()
        tools = [rag_search, doc_focus, list_docs, memory_recall, memory_save]
        for t in tools:
            registry.register(t)

        for t in tools:
            assert registry.get(t.name) is t

        desc = registry.get_descriptions()
        for t in tools:
            assert t.name in desc
            assert t.description in desc

    def test_registry_execute(self):
        from core.tools.registry import ToolRegistry
        from core.tools.rag_tools import list_docs, set_tool_deps

        set_tool_deps(list_callback=lambda: ["a.pdf", "b.pdf"])

        registry = ToolRegistry()
        registry.register(list_docs)

        result = registry.execute("list_docs", {})
        assert "a.pdf" in result

    def test_registry_execute_unknown(self):
        from core.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = registry.execute("unknown_tool", {})
        assert "未找到" in result or "错误" in result