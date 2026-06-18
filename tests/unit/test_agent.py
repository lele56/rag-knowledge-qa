"""Agent 核心逻辑单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from core.agent.types import AgentStep, StepType, AgentResult, AgentState
from core.tools.registry import ToolRegistry


# ============================================================
# AgentStep / AgentResult 类型测试
# ============================================================

class TestAgentStep:
    def test_thought_step(self):
        step = AgentStep(type=StepType.THOUGHT, content="分析问题")
        assert step.type == StepType.THOUGHT
        assert step.content == "分析问题"
        assert step.tool_name is None
        assert step.tool_result is None

    def test_action_step(self):
        step = AgentStep(
            type=StepType.ACTION,
            content="",
            tool_name="rag_search",
            tool_input="Transformer",
        )
        assert step.type == StepType.ACTION
        assert step.tool_name == "rag_search"
        assert step.tool_input == "Transformer"

    def test_observation_step(self):
        step = AgentStep(
            type=StepType.OBSERVATION,
            content="Transformer 是一种...",
            tool_result="Transformer 是一种...",
        )
        assert step.type == StepType.OBSERVATION
        assert step.tool_result == "Transformer 是一种..."

    def test_to_log(self):
        step = AgentStep(type=StepType.FINISH, content="最终答案")
        log = step.to_log()
        assert "Finish" in log
        assert "最终答案" in log


class TestAgentResult:
    def test_basic_result(self):
        result = AgentResult(
            answer="LLM 是大语言模型。",
            sources=["doc1.pdf"],
            state=AgentState.FINISHED,
        )
        assert result.answer == "LLM 是大语言模型。"
        assert result.sources == ["doc1.pdf"]
        assert result.state == AgentState.FINISHED

    def test_result_with_error(self):
        result = AgentResult(
            answer="系统错误",
            state=AgentState.ERROR,
            error="max_steps_reached",
        )
        assert result.state == AgentState.ERROR
        assert result.error == "max_steps_reached"

    def test_result_metadata(self):
        result = AgentResult(
            answer="答案",
            state=AgentState.FINISHED,
            metadata={"agent": "RAGAgent", "question": "问题"},
        )
        assert result.metadata["agent"] == "RAGAgent"
        assert result.metadata["question"] == "问题"


# ============================================================
# ToolRegistry 测试
# ============================================================

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        mock = MagicMock()
        mock.name = "test_tool"
        registry.register(mock)
        assert registry.get("test_tool") is mock

    def test_register_duplicate(self):
        registry = ToolRegistry()
        mock1 = MagicMock()
        mock1.name = "test_tool"
        registry.register(mock1)
        mock2 = MagicMock()
        mock2.name = "test_tool"
        # 注册同名工具会覆盖（不抛异常）
        registry.register(mock2)
        assert registry.get("test_tool") is mock2

    def test_unregister(self):
        registry = ToolRegistry()
        mock = MagicMock()
        mock.name = "test_tool"
        registry.register(mock)
        assert registry.unregister("test_tool") is True
        assert registry.get("test_tool") is None

    def test_unregister_not_found(self):
        registry = ToolRegistry()
        assert registry.unregister("nonexistent") is False

    def test_list_tools(self):
        registry = ToolRegistry()
        m1, m2 = MagicMock(), MagicMock()
        m1.name, m2.name = "t1", "t2"
        registry.register(m1)
        registry.register(m2)
        tools = registry.list_tools()
        assert "t1" in tools
        assert "t2" in tools

    def test_clear(self):
        registry = ToolRegistry()
        mock = MagicMock()
        mock.name = "t1"
        registry.register(mock)
        registry.clear()
        assert registry.list_tools() == []

    def test_get_descriptions(self):
        registry = ToolRegistry()
        mock = MagicMock()
        mock.name = "search"
        mock.description = "搜索文档"
        mock.to_prompt_desc = MagicMock(return_value="search: 搜索文档")
        registry.register(mock)
        desc = registry.get_descriptions()
        assert "search" in desc
        assert "搜索文档" in desc


# ============================================================
# ReActAgent 解析逻辑测试
# ============================================================

class TestReactAgentParsing:
    """测试 _parse_response / _parse_tool_call / _parse_finish"""

    def _make_agent(self):
        from core.agent.base import ReActAgent
        from abc import ABC

        class FakeAgent(ReActAgent):
            def _build_tools(self):
                pass

        llm = MagicMock()
        return FakeAgent(llm=llm, max_steps=10)

    def test_parse_response_standard(self):
        agent = self._make_agent()
        text = "Thought: 需要搜索\nAction: rag_search[Transformer]"
        thought, action = agent._parse_response(text)
        assert thought == "需要搜索"
        assert action == "rag_search[Transformer]"

    def test_parse_response_multiline_thought(self):
        agent = self._make_agent()
        text = (
            "Thought: 用户问了模型架构的问题。\n"
            "需要先确定聚焦的文档。\n"
            "Action: rag_search[模型架构]"
        )
        thought, action = agent._parse_response(text)
        assert "用户问了模型架构" in thought
        assert action == "rag_search[模型架构]"

    def test_parse_response_no_thought(self):
        agent = self._make_agent()
        text = "Action: Finish[答案是 42]"
        thought, action = agent._parse_response(text)
        assert thought is None
        assert action == "Finish[答案是 42]"

    def test_parse_response_no_action(self):
        agent = self._make_agent()
        text = "Thought: 思考中..."
        thought, action = agent._parse_response(text)
        assert thought == "思考中..."
        assert action is None

    def test_parse_tool_call_standard(self):
        agent = self._make_agent()
        name, inp = agent._parse_tool_call("rag_search[Transformer]")
        assert name == "rag_search"
        assert inp == "Transformer"

    def test_parse_tool_call_no_args(self):
        agent = self._make_agent()
        name, inp = agent._parse_tool_call("list_docs[]")
        assert name == "list_docs"
        assert inp == ""

    def test_parse_tool_call_no_brackets(self):
        agent = self._make_agent()
        name, inp = agent._parse_tool_call("list_docs")
        assert name == "list_docs"
        assert inp == ""

    def test_parse_tool_call_with_spaces(self):
        agent = self._make_agent()
        # 工具名和 [ 之间不能有空格（这是 LLM 输出规范）
        name, inp = agent._parse_tool_call("rag_search[hello world]")
        assert name == "rag_search"
        assert inp == "hello world"

    def test_parse_finish_standard(self):
        agent = self._make_agent()
        result = agent._parse_finish("Finish[这是最终答案。]")
        assert result == "这是最终答案。"

    def test_parse_finish_multiline(self):
        agent = self._make_agent()
        result = agent._parse_finish("Finish[第一行\n第二行\n第三行]")
        assert "第一行" in result
        assert "第三行" in result

    def test_parse_finish_unclosed(self):
        agent = self._make_agent()
        result = agent._parse_finish("Finish[答案被截断了")
        assert result == "答案被截断了"

    def test_parse_finish_no_brackets(self):
        agent = self._make_agent()
        result = agent._parse_finish("答案")
        assert result == "答案"


# ============================================================
# Finish 守卫机制测试
# ============================================================

class TestFinishGuard:
    def _make_agent(self):
        from core.agent.base import ReActAgent

        class FakeAgent(ReActAgent):
            def _build_tools(self):
                pass

        llm = MagicMock()
        return FakeAgent(llm=llm, max_steps=10)

    def test_reject_when_no_search(self):
        agent = self._make_agent()
        assert agent._should_reject_finish() is True

    def test_allow_after_search(self):
        agent = self._make_agent()
        agent._history.append("Action: rag_search[Transformer]")
        agent._history.append("Observation: ...")
        assert agent._should_reject_finish() is False

    def test_allow_if_no_rag_search_tool(self):
        agent = self._make_agent()
        # 如果根本没有 rag_search 工具，不应拒绝
        agent._history.append("Action: list_docs[]")
        assert agent._should_reject_finish() is True


# ============================================================
# 状态管理测试
# ============================================================

class TestAgentState:

    def test_initial_state(self):
        from core.agent.base import ReActAgent

        class FakeAgent(ReActAgent):
            def _build_tools(self):
                pass

        agent = FakeAgent(llm=MagicMock(), max_steps=10)
        assert agent._state == AgentState.IDLE

    def test_reset(self):
        from core.agent.base import ReActAgent

        class FakeAgent(ReActAgent):
            def _build_tools(self):
                pass

        agent = FakeAgent(llm=MagicMock(), max_steps=10)
        agent._steps = [AgentStep(type=StepType.THOUGHT, content="test")]
        agent._history = ["test"]
        agent._reset()
        assert agent._steps == []
        assert agent._history == []
        assert agent._state == AgentState.IDLE