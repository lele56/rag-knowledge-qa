# core/tools/base.py
"""工具系统 — 使用 LangChain BaseTool + @tool 装饰器"""

from langchain_core.tools import BaseTool, tool, StructuredTool
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    """工具参数描述（兼容旧代码）"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class Tool:
    """工具基类（兼容旧代码，推荐直接使用 @tool 装饰器）"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def get_parameters(self) -> List[ToolParameter]:
        return []

    def run(self, parameters: Dict[str, Any]) -> str:
        raise NotImplementedError

    def to_prompt_desc(self) -> str:
        return f"- {self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"Tool(name={self.name})"