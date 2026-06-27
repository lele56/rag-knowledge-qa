# core/tools/pipeline.py
"""
工具管道 — 链式调用 + 条件分支

借鉴 HelloAgents 的 ToolChain 设计，支持：
- 顺序执行: tool1 → tool2 → tool3
- 条件分支: 根据结果决定下一步
- 并行执行: 同时调用多个工具
- 结果聚合: 合并多个工具的输出
"""

from typing import Dict, Any, List, Optional, Callable
from .base import Tool
from .registry import ToolRegistry
from utils.logger import logger


class ToolPipeline:
    """工具管道 — 按顺序执行多个工具

    用法:
        pipeline = ToolPipeline(registry)
        pipeline.add("rag_search", {"query": "Transformer"})
        pipeline.add("memory_recall", {"query": "Transformer"})
        results = pipeline.run()
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._steps: List[dict] = []

    def add(self, tool_name: str, parameters: Dict[str, Any]) -> "ToolPipeline":
        """添加一个步骤"""
        self._steps.append({
            "tool": tool_name,
            "params": dict(parameters),
        })
        return self  # 链式调用

    def run(self) -> Dict[str, str]:
        """执行所有步骤，返回 {tool_name: result}"""
        results = {}
        for step in self._steps:
            name = step["tool"]
            params = step["params"]
            try:
                result = self._registry.execute(name, params)
                results[name] = result
                logger.debug(f"Pipeline: {name} → {result[:80]}...")
            except Exception as e:
                results[name] = f"错误: {e}"
                logger.error(f"Pipeline: {name} 失败: {e}")
        return results

    def clear(self) -> None:
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._steps)


class ConditionalPipeline:
    """条件管道 — 根据前一步结果决定下一步

    用法:
        pipeline = ConditionalPipeline(registry)
        pipeline.when(
            condition=lambda r: "未找到" in r.get("rag_search", ""),
            then="rag_search", then_params={"query": "备选查询"},
            otherwise="memory_recall", otherwise_params={"query": "原查询"}
        )
        result = pipeline.run("rag_search", {"query": "初始查询"})
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._branches: Dict[str, dict] = {}  # tool_name → {condition, then, then_params, else, else_params}

    def when(
        self,
        tool_name: str,
        condition: Callable[[Dict[str, str]], bool],
        then: str,
        then_params: Dict[str, Any],
        otherwise: Optional[str] = None,
        otherwise_params: Optional[Dict[str, Any]] = None,
    ) -> "ConditionalPipeline":
        """注册条件分支"""
        self._branches[tool_name] = {
            "condition": condition,
            "then": then,
            "then_params": then_params,
            "otherwise": otherwise,
            "otherwise_params": otherwise_params or {},
        }
        return self

    def run(self, first_tool: str, first_params: Dict[str, Any]) -> Dict[str, str]:
        """执行条件管道"""
        results = {}

        # 第一步
        result = self._registry.execute(first_tool, first_params)
        results[first_tool] = result

        # 检查是否有分支
        branch = self._branches.get(first_tool)
        if branch:
            if branch["condition"](results):
                # 走 then 分支
                next_result = self._registry.execute(branch["then"], branch["then_params"])
                results[branch["then"]] = next_result
            elif branch["otherwise"]:
                next_result = self._registry.execute(branch["otherwise"], branch["otherwise_params"])
                results[branch["otherwise"]] = next_result

        return results


class ParallelPipeline:
    """并行管道 — 同时调用多个工具

    用法:
        pipeline = ParallelPipeline(registry)
        pipeline.parallel("rag_search", {"query": "Transformer"})
        pipeline.parallel("memory_recall", {"query": "Transformer"})
        results = pipeline.run()  # 并行执行
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._calls: List[tuple] = []

    def parallel(self, tool_name: str, parameters: Dict[str, Any]) -> "ParallelPipeline":
        self._calls.append((tool_name, parameters))
        return self

    def run(self) -> Dict[str, str]:
        """并行执行所有工具调用"""
        import concurrent.futures

        results = {}

        def _call(name, params):
            try:
                return name, self._registry.execute(name, params)
            except Exception as e:
                return name, f"错误: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._calls)) as executor:
            futures = {executor.submit(_call, name, params): name for name, params in self._calls}
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                results[name] = result

        return results

    def clear(self) -> None:
        self._calls.clear()