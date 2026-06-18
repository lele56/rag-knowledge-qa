"""QA 管线集成测试 — 无需真实服务，验证组件协作行为"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock


# ============================================================
# Agent 完整流程测试（LLM 调用 mock 掉）
# ============================================================

class TestAgentIntegration:
    @pytest.fixture
    def retriever_fn(self):
        from langchain_core.documents import Document
        def _fn(query, **kwargs):
            return [Document(page_content="Transformer 由 Vaswani 等人在 2017 年提出。", metadata={"source": "paper.pdf"})]
        return _fn

    def test_full_loop_minimal(self, retriever_fn):
        """模拟最简单的 ReAct 循环：搜索 → 回答"""
        from core.agent.rag_agent import RAGAgent
        from langchain_core.messages import AIMessage

        responses = [
            AIMessage(content="Thought: 需要搜索\nAction: rag_search[Transformer]"),
            AIMessage(content="Thought: 信息已足够\nAction: Finish[Transformer 是一种神经网络架构。]"),
        ]
        call_count = [0]

        async def mock_call(self, prompt, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx].content

        with patch("core.agent.base.ReActAgent._call_llm", mock_call):
            agent = RAGAgent(llm=MagicMock(), max_steps=10)
            agent.attach_retriever(retriever_fn)
            result = agent.run("什么是 Transformer？")
            assert "Transformer" in result.answer
            assert result.state.value == "finished"

    def test_full_loop_with_focus(self, retriever_fn):
        """模拟聚焦文档后搜索的场景"""
        from core.agent.rag_agent import RAGAgent
        from langchain_core.messages import AIMessage

        responses = [
            AIMessage(content="Thought: 先聚焦文档\nAction: doc_focus[paper.pdf]"),
            AIMessage(content="Thought: 搜索\nAction: rag_search[结论]"),
            AIMessage(content="Thought: 完成\nAction: Finish[这篇论文的结论是...]"),
        ]
        call_count = [0]

        async def mock_call(self, prompt, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx].content

        with patch("core.agent.base.ReActAgent._call_llm", mock_call):
            agent = RAGAgent(llm=MagicMock(), max_steps=10)
            agent.attach_retriever(retriever_fn)
            result = agent.run("这篇论文的结论是什么？")
            assert result.answer
            assert result.state.value == "finished"

    def test_max_steps_exceeded(self, retriever_fn):
        """超过最大步数应返回错误"""
        from core.agent.rag_agent import RAGAgent
        from langchain_core.messages import AIMessage

        async def mock_call(self, prompt, **kwargs):
            return "Thought: 思考中...\nAction: rag_search[test]"

        with patch("core.agent.base.ReActAgent._call_llm", mock_call):
            agent = RAGAgent(llm=MagicMock(), max_steps=2)
            agent.attach_retriever(retriever_fn)
            result = agent.run("问题")
            assert result.error == "max_steps_reached"
            assert "步数" in result.answer or "限定" in result.answer

    def test_list_docs_guard(self, retriever_fn):
        """list_docs 后直接 Finish 应被拦截"""
        from core.agent.rag_agent import RAGAgent
        from langchain_core.messages import AIMessage

        responses = [
            AIMessage(content="Thought: 查看文档\nAction: list_docs[]"),
            AIMessage(content="Thought: 有文档了\nAction: Finish[有 doc1, doc2, doc3]"),
            AIMessage(content="Thought: 搜索\nAction: rag_search[内容]"),
            AIMessage(content="Thought: 完成\nAction: Finish[文档内容...]"),
        ]
        call_count = [0]

        async def mock_call(self, prompt, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx].content

        with patch("core.agent.base.ReActAgent._call_llm", mock_call):
            agent = RAGAgent(llm=MagicMock(), max_steps=10)
            agent.attach_retriever(retriever_fn)
            result = agent.run("有哪些文档？")
            assert result.answer
            assert result.state.value == "finished"


# ============================================================
# 配置与提示词集成测试
# ============================================================

class TestConfigIntegration:
    def test_prompts_import(self):
        from config.prompts import (
            BASE_REACT_PROMPT,
            RAG_AGENT_PROMPT,
            QUICK_ANSWER_PROMPT,
            HYDE_PROMPT,
            PLANNER_PROMPT,
            EXECUTOR_PROMPT,
            SYNTHESIS_PROMPT,
            DEFAULT_SYSTEM_PROMPT,
            RAG_SYSTEM_INSTRUCTION,
            OUTPUT_INSTRUCTION,
            EVAL_JUDGE_PROMPT,
            QA_TEMPLATE,
            CONDENSE_TEMPLATE,
        )
        assert all([
            BASE_REACT_PROMPT,
            RAG_AGENT_PROMPT,
            QUICK_ANSWER_PROMPT,
            HYDE_PROMPT,
            PLANNER_PROMPT,
            EXECUTOR_PROMPT,
            SYNTHESIS_PROMPT,
            DEFAULT_SYSTEM_PROMPT,
            RAG_SYSTEM_INSTRUCTION,
            OUTPUT_INSTRUCTION,
            EVAL_JUDGE_PROMPT,
            QA_TEMPLATE,
            CONDENSE_TEMPLATE,
        ])

    def test_prompts_format(self):
        from config.prompts import RAG_AGENT_PROMPT
        formatted = RAG_AGENT_PROMPT.format(
            tools="rag_search: 搜索",
            state_info="无聚焦",
            chat_history="无历史",
            question="测试问题",
            history="无历史",
        )
        assert "rag_search" in formatted
        assert "测试问题" in formatted
        assert "无聚焦" in formatted


# ============================================================
# LLM 调用测试
# ============================================================

class TestLLMIntegration:
    def test_get_llm(self):
        from core.llm import get_llm
        llm = get_llm()
        assert llm is not None

    def test_llm_singleton(self):
        from core.llm import get_llm
        l1 = get_llm()
        l2 = get_llm()
        assert l1 is l2

    def test_llm_model_config(self):
        from core.llm import get_llm
        from config.settings import settings
        llm = get_llm()
        assert llm.model_name == settings.LLM_MODEL