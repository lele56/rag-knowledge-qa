# core/tools/registry.py
"""工具注册表 — 基于 LangChain BaseTool"""

from typing import Optional, Dict, Any, List
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function


class ToolRegistry:
    """工具注册表 — 统一管理工具注册、查找、执行"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def execute(self, name: str, parameters: Dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"错误: 未找到工具 '{name}'"
        try:
            return tool.invoke(parameters)
        except Exception as e:
            return f"错误: 执行工具 '{name}' 失败: {e}"

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_descriptions(self) -> str:
        if not self._tools:
            return "暂无可用工具"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """生成 OpenAI 兼容的 Function Calling schema，传给 LLM API 的 tools 参数。

        工具描述不再占用 prompt token，由 LLM 在 API 层面解析。
        """
        tools = []
        for tool in self._tools.values():
            func_def = convert_to_openai_function(tool)
            tools.append({"type": "function", "function": func_def})
        return tools

    def clear(self) -> None:
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# 全局单例
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry