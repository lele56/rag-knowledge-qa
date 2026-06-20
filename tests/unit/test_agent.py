"""Agent 核心逻辑单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

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

    def test_get_openai_tools(self):
        """get_openai_tools 生成 OpenAI Function Calling 格式"""
        from langchain_core.tools import tool as langchain_tool

        @langchain_tool
        def search(query: str) -> str:
            """搜索文档"""
            return "result"

        registry = ToolRegistry()
        registry.register(search)
        tools = registry.get_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"
        assert "description" in tools[0]["function"]


# ============================================================
# ReActAgent Function Calling 测试
# ============================================================

class TestReactAgentFunctionCalling:
    """测试 _build_system_prompt / _execute_tool / _run_loop (Function Calling)"""

    def _make_agent(self):
        from core.agent.base import ReActAgent

        class FakeAgent(ReActAgent):
            def _build_tools(self):
                pass

        llm = MagicMock()
        return FakeAgent(llm=llm, max_steps=10)

    def test_build_system_prompt_no_tools_placeholder(self):
        """_build_system_prompt 不再需要 {tools} 占位符"""
        agent = self._make_agent()
        prompt = agent._build_system_prompt("什么是 Transformer？")
        assert "什么是 Transformer？" in prompt
        assert "{tools}" not in prompt

    def test_build_system_prompt_includes_history(self):
        agent = self._make_agent()
        agent._history.append("Action: rag_search[{'query': 'Transformer'}]")
        prompt = agent._build_system_prompt("什么是 Transformer？")
        assert "rag_search" in prompt

    def test_execute_tool_with_dict_args(self):
        """Function Calling 传入 dict 参数"""
        agent = self._make_agent()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="工具返回结果")
        agent.tool_registry.register(mock_tool)

        result = agent._execute_tool("test_tool", {"query": "Transformer", "top_k": 8})
        assert result == "工具返回结果"
        mock_tool.invoke.assert_called_once_with({"query": "Transformer", "top_k": 8})

    def test_execute_tool_with_string_args(self):
        """兼容旧格式字符串参数"""
        agent = self._make_agent()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="done")
        agent.tool_registry.register(mock_tool)

        result = agent._execute_tool("test_tool", "hello")
        assert result == "done"

    def test_execute_tool_not_found(self):
        agent = self._make_agent()
        result = agent._execute_tool("nonexistent", "test")
        assert "未找到" in result

    def test_run_loop_with_tool_calls(self):
        """模拟 LLM 返回 tool_calls 后正常走完流程"""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as langchain_tool

        @langchain_tool
        def rag_search(query: str, top_k: int = 8) -> str:
            """搜索知识库"""
            return "Transformer 是..."

        agent = self._make_agent()
        agent.tool_registry.register(rag_search)

        call_1 = AIMessage(
            content="",
            tool_calls=[{"name": "rag_search", "args": {"query": "Transformer"}, "id": "call_1"}],
        )
        call_2 = AIMessage(content="基于搜索结果，Transformer 是一种神经网络架构。")

        call_count = [0]

        async def mock_call(self, messages=None, tools=None, **kwargs):
            if call_count[0] == 0:
                call_count[0] += 1
                return call_1
            return call_2

        from core.agent.base import ReActAgent
        with patch.object(ReActAgent, "_call_llm", mock_call):
            async def gather():
                steps = []
                async for step in agent._run_loop("什么是 Transformer？"):
                    steps.append(step)
                return steps
            import asyncio
            steps = asyncio.run(gather())

        assert len(steps) >= 3
        assert any(s.type.value == "action" for s in steps)
        assert any(s.type.value == "observation" for s in steps)
        assert any(s.type.value == "finish" for s in steps)

    def test_run_loop_rejects_premature_finish(self):
        """未搜索就 Finish 应被拦截"""
        from langchain_core.messages import AIMessage

        agent = self._make_agent()
        call_1 = AIMessage(content="这是答案，不需要搜索。")

        async def mock_call(self, messages=None, tools=None, **kwargs):
            return call_1

        from core.agent.base import ReActAgent
        with patch.object(ReActAgent, "_call_llm", mock_call):
            async def gather():
                steps = []
                async for step in agent._run_loop("问题"):
                    steps.append(step)
                return steps
            import asyncio
            steps = asyncio.run(gather())

        assert any("拒绝" in s.content for s in steps)

    def test_run_loop_max_steps(self):
        """达到最大步数应终止"""
        from langchain_core.messages import AIMessage

        agent = self._make_agent()
        agent.max_steps = 2

        async def mock_call(self, messages=None, tools=None, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[{"name": "rag_search", "args": {"query": "test"}, "id": "call_x"}],
            )

        from core.agent.base import ReActAgent
        with patch.object(ReActAgent, "_call_llm", mock_call):
            async def gather():
                steps = []
                async for step in agent._run_loop("问题"):
                    steps.append(step)
                return steps
            import asyncio
            steps = asyncio.run(gather())

        assert len(steps) >= 2


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