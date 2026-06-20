"""
ReAct Agent 基类 — 思考 + 行动（Native Function Calling 版）

借鉴 HelloAgents 的 ReActAgent 设计，提供：
- 思考 → 行动 → 观察 循环
- 可插拔工具注册表
- 可配置最大步数
- 完整的执行追踪（AgentStep）
- 同步/异步双模式
- 原生 Function Calling（工具描述不占 prompt token）

子类只需实现 _build_tools() 和 _get_system_prompt()
"""
import re
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from core.tools import ToolRegistry
from core.agent.types import AgentState, StepType, AgentStep, AgentResult
from utils.logger import logger


class ReActAgent(ABC):
    """ReAct Agent 基类 — 推理与行动结合的智能体

    用法:
        class MyAgent(ReActAgent):
            def _build_tools(self):
                self.tool_registry.register(MyTool())

            def _get_system_prompt(self):
                return "你是一个助手..."

        agent = MyAgent(llm=llm, max_steps=5)
        result = agent.run("你的问题")
        # 或流式
        async for step in agent.astream("你的问题"):
            print(step.content)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_steps: int = 10,
        tool_registry: Optional[ToolRegistry] = None,
        prompt_template: Optional[str] = None,
        name: str = "ReActAgent",
    ):
        from config.prompts import BASE_REACT_PROMPT

        self.llm = llm
        self.max_steps = max_steps
        self.tool_registry = tool_registry or ToolRegistry()
        self.prompt_template = prompt_template or BASE_REACT_PROMPT
        self.name = name

        self._steps: List[AgentStep] = []
        self._history: List[str] = []
        self._state: AgentState = AgentState.IDLE

        self._build_tools()

    # ================================================================
    # 子类钩子
    # ================================================================

    @abstractmethod
    def _build_tools(self):
        """子类在此注册自己的工具"""
        pass

    def _get_system_prompt(self) -> str:
        from config.prompts import DEFAULT_SYSTEM_PROMPT
        return DEFAULT_SYSTEM_PROMPT

    # ================================================================
    # 公共接口
    # ================================================================

    def run(self, question: str, **kwargs) -> AgentResult:
        return asyncio.run(self.arun(question, **kwargs))

    async def arun(self, question: str, **kwargs) -> AgentResult:
        """异步运行，返回最终 AgentResult"""
        self._reset()
        logger.info(f"🤖 [{self.name}] 开始处理: {question[:60]}...")
        try:
            async for step in self._run_loop(question, **kwargs):
                if step.type == StepType.FINISH:
                    return self._build_result(question, step.content)
            return self._build_result(question, "无法在限定步数内完成，请尝试更具体的问题。", error="max_steps_reached")
        except Exception as e:
            logger.error(f"❌ [{self.name}] 执行异常: {e}")
            self._state = AgentState.ERROR
            return self._build_result(question, "系统错误，请重试。", error=str(e))

    async def astream(self, question: str, **kwargs) -> AsyncIterator[AgentStep]:
        """流式运行，逐步产出每个 AgentStep"""
        self._reset()
        logger.info(f"🤖 [{self.name}] 流式开始: {question[:60]}...")
        try:
            async for step in self._run_loop(question, **kwargs):
                yield step
        except Exception as e:
            logger.error(f"❌ [{self.name}] 流式异常: {e}")
            self._state = AgentState.ERROR

    # ================================================================
    # 核心：统一的 ReAct 循环（arun 和 astream 共用）
    # ================================================================

    async def _run_loop(self, question: str, **kwargs) -> AsyncIterator[AgentStep]:
        """ReAct 主循环（Native Function Calling）：LLM 通过 tool_calls 调用工具，直到给出纯文本回答或超步数"""
        tools = self.tool_registry.get_openai_tools()

        messages: list = [
            SystemMessage(content=self._build_system_prompt(question)),
        ]

        chat_history = self._get_chat_history()
        if chat_history and chat_history != "（暂无对话历史）":
            messages.append(HumanMessage(content=f"对话历史：\n{chat_history}"))

        messages.append(HumanMessage(content=question))

        for step_no in range(1, self.max_steps + 1):
            self._state = AgentState.THINKING

            response = await self._call_llm(messages=messages, tools=tools, **kwargs)
            if not response:
                logger.error(f"[{self.name}] LLM 返回空响应")
                break

            content = response.content if hasattr(response, 'content') else ""
            tool_calls = getattr(response, 'tool_calls', None) or []

            if tool_calls:
                messages.append(response)

                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})

                    self._history.append(f"Action: {tool_name}[{tool_args}]")
                    yield self._add_step(StepType.ACTION, "", tool_name, str(tool_args))

                    observation = self._execute_tool(tool_name, tool_args)
                    self._history.append(f"Observation: {observation}")
                    yield self._add_step(StepType.OBSERVATION, observation, tool_result=observation)

                    messages.append(ToolMessage(
                        content=str(observation),
                        tool_call_id=tc.get("id", ""),
                    ))

                if step_no >= 4 and self._has_retrieved():
                    messages.append(HumanMessage(
                        content="⚠️ 【强制收尾】已达步数限制！你已检索到足够内容，"
                                "下一轮必须直接给出答案，禁止任何其他操作！"
                    ))
            else:
                if self._should_reject_finish():
                    self._history.append(
                        "⚠️ 系统拒绝: 你还没有搜索文档内容！请立即执行以下步骤:\n"
                        "1. 从 list_docs 结果中选一个最可能的文档\n"
                        "2. 调用 rag_search 搜索内容\n"
                        "3. 搜索到内容后再回答"
                    )
                    messages.append(HumanMessage(
                        content="⚠️ 你还没有搜索文档内容！请先调用 rag_search 搜索知识库，不要直接回答。"
                    ))
                    yield self._add_step(StepType.THOUGHT, "（系统拒绝：未搜索即回答，要求重新搜索）")
                    continue

                final_answer = content.strip()
                if not final_answer:
                    logger.warning(f"[{self.name}] 空内容，流程终止")
                    break

                self._state = AgentState.FINISHED
                yield self._add_step(StepType.FINISH, final_answer)
                logger.info(f"🎉 [{self.name}] 完成: {final_answer[:80]}...")
                return

        logger.warning(f"⏰ [{self.name}] 达到最大步数 {self.max_steps}")

    # ================================================================
    # 内部辅助
    # ================================================================

    def _reset(self):
        self._steps.clear()
        self._history.clear()
        self._state = AgentState.IDLE

    def _add_step(
        self,
        step_type: StepType,
        content: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> AgentStep:
        step = AgentStep(
            type=step_type,
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=tool_result,
        )
        self._steps.append(step)
        logger.info(f"  [{self.name}] {step.to_log()}")
        return step

    def _build_system_prompt(self, question: str) -> str:
        history_str = "\n".join(self._history) if self._history else "（无历史记录）"
        state_info = self._get_state_info()
        chat_history = self._get_chat_history()
        return self.prompt_template.format(
            question=question,
            history=history_str,
            state_info=state_info,
            chat_history=chat_history,
        )

    def _get_state_info(self) -> str:
        return "当前没有聚焦的文档。如果用户说'这篇文章'，请先使用 list_docs 查看可用文档。"

    def _get_chat_history(self) -> str:
        return "（暂无对话历史）"

    def _should_reject_finish(self) -> bool:
        """如果从未调用过 rag_search，拒绝 Finish（用户可能只是在问有哪些文档）"""
        for h in self._history:
            if "rag_search[" in h:
                return False
        return True

    def _has_retrieved(self) -> bool:
        for h in self._history:
            if "rag_search[" in h:
                return True
        return False

    async def _call_llm(self, prompt: str = None, messages: list = None, tools: list = None, **kwargs):
        """调用 LLM。支持字符串 prompt（兼容旧接口）和 messages + tools（Function Calling）。"""
        try:
            if messages is not None:
                from core.llm import call_llm_messages_with_retry
                return await call_llm_messages_with_retry(messages, tools=tools)
            from core.llm import call_llm_with_retry
            return await call_llm_with_retry(prompt)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None

    def _execute_tool(self, tool_name: str, tool_input) -> str:
        """执行工具。支持字符串参数（旧格式）和字典参数（Function Calling JSON）。"""
        try:
            tool = self.tool_registry.get(tool_name)
            if tool is None:
                return f"错误: 未找到工具 '{tool_name}'"
            if isinstance(tool_input, str):
                return str(tool.invoke({"query": tool_input, "input": tool_input, "document_name": tool_input}))
            else:
                return str(tool.invoke(tool_input))
        except Exception as e:
            return f"工具执行失败: {e}"

    def _build_result(
        self, question: str, answer: str, error: Optional[str] = None,
    ) -> AgentResult:
        return AgentResult(
            answer=answer,
            sources=self._collect_sources(),
            steps=list(self._steps),
            state=self._state,
            error=error,
            metadata={"question": question, "agent": self.name},
        )

    def _collect_sources(self) -> List[str]:
        sources = set()
        for step in self._steps:
            if step.type == StepType.OBSERVATION and step.tool_result:
                for m in re.finditer(r"来源[：:]\s*([^\n]+)", step.tool_result):
                    sources.add(m.group(1).strip())
                for m in re.finditer(r"source[：:]\s*([^\n]+)", step.tool_result, re.IGNORECASE):
                    sources.add(m.group(1).strip())
        return sorted(sources) if sources else []

    # ================================================================
    # 工具管理
    # ================================================================

    def add_tool(self, tool):
        self.tool_registry.register(tool)

    def remove_tool(self, name: str) -> bool:
        return self.tool_registry.unregister(name)

    def list_tools(self) -> List[str]:
        return self.tool_registry.list_tools()