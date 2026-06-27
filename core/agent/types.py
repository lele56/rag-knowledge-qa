# core/agent/types.py
"""
Agent 基础类型定义

借鉴 HelloAgents 的 Agent 设计，为 ReAct 模式提供类型支持。
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


class AgentState(Enum):
    """Agent 运行状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    FINISHED = "finished"
    ERROR = "error"


class StepType(Enum):
    """步骤类型"""
    THOUGHT = "thought"       # 推理
    ACTION = "action"         # 工具调用
    OBSERVATION = "observation"  # 观察结果
    FINISH = "finish"         # 最终答案


@dataclass
class AgentStep:
    """Agent 执行步骤"""
    type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> str:
        """生成日志字符串"""
        if self.type == StepType.THOUGHT:
            return f"🤔 Thought: {self.content}"
        elif self.type == StepType.ACTION:
            return f"🎬 Action: {self.tool_name}[{self.tool_input}]"
        elif self.type == StepType.OBSERVATION:
            return f"👀 Observation: {self.tool_result[:200]}"
        elif self.type == StepType.FINISH:
            return f"🎉 Finish: {self.content[:200]}"
        return self.content


@dataclass
class AgentResult:
    """Agent 执行结果"""
    answer: str
    sources: List[str] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    state: AgentState = AgentState.FINISHED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.metadata.get("question", ""),
            "answer": self.answer,
            "sources": self.sources,
            "state": self.state.value,
            "steps_count": len(self.steps),
            "error": self.error,
        }