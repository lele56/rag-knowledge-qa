# core/agent/rag_agent.py
"""
RAG ReAct Agent — 知识库问答专用 Agent

继承 ReActAgent，预置 RAG 工具：
- rag_search:    知识库检索
- doc_focus:     聚焦文档
- list_docs:     列出文档
- memory_recall: 回忆长期记忆
- memory_save:   保存到长期记忆

核心流程:
1. 接收用户问题
2. ReAct 循环: Thought → 调用工具 → Observation → 继续... → Finish
3. 返回最终答案 + 来源

这是对旧 RAGWithMemory 的全面重构，模块化、可插拔。
"""

from typing import Optional, List, Set, Dict, Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from core.agent.base import ReActAgent
from core.agent.types import AgentResult
from core.tools import ToolRegistry
from core.tools.rag_tools import (
    rag_search,
    list_docs,
    memory_recall,
    memory_save,
    graph_query,
    set_tool_deps,
)
from core.context.builder import ContextBuilder, ContextConfig
from config.prompts import RAG_AGENT_PROMPT, QUICK_ANSWER_PROMPT, RAG_SYSTEM_INSTRUCTION
from utils.logger import logger


class RAGAgent(ReActAgent):
    """RAG ReAct Agent — 知识库问答专用

    用法:
        agent = RAGAgent(llm=llm)
        agent.attach_retriever(retriever_fn)    # 注入检索器
        agent.attach_memory(memory_manager)      # 注入记忆管理器
        agent.attach_focus(focus_callback)       # 注入聚焦回调
        agent.attach_doc_list(list_callback)     # 注入文档列表

        result = agent.run("这篇文章的主要内容是什么？")
        print(result.answer)
        print(result.sources)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_steps: int = 10,
        name: str = "RAGAgent",
    ):
        tool_registry = ToolRegistry()
        tool_registry.register(rag_search)
        tool_registry.register(list_docs)
        tool_registry.register(memory_recall)
        tool_registry.register(memory_save)
        tool_registry.register(graph_query)

        super().__init__(
            llm=llm,
            max_steps=max_steps,
            tool_registry=tool_registry,
            prompt_template=RAG_AGENT_PROMPT,
            name=name,
        )

        # 会话级文档列表：上传时追加，agent 以此为检索范围
        self._session_docs: Set[str] = set()

        # 可注入的依赖
        self._retriever_fn: Optional[Callable] = None
        self._memory_manager = None
        self._focus_set: Optional[Set[str]] = None

        # 上下文构建器（给 Finish 答案之前的上下文增强）
        self._context_builder = ContextBuilder(
            config=ContextConfig(max_tokens=6000, reserve_ratio=0.15)
        )

    # ================================================================
    # 钩子实现
    # ================================================================

    def _build_tools(self) -> None:
        """工具已在 __init__ 中构建"""
        pass

    def _get_state_info(self) -> str:
        """覆盖基类：返回当前会话文档列表（只显示人类可读文件名，隐藏 doc_xxx 内部 ID）"""
        if self._session_docs:
            human_names = sorted(
                f for f in self._session_docs if not f.startswith("doc_")
            )
            if not human_names:
                human_names = sorted(self._session_docs)
            return (
                f"当前会话文档（{len(human_names)} 篇）: {', '.join(human_names)}。\n"
                f"用户提到'这篇文章'、'那个文档'时指的都是这些文档。"
            )
        return "当前会话未上传任何文档。可用 list_docs 查看知识库全部文档。"

    def _get_chat_history(self) -> str:
        """覆盖基类：从短期记忆获取对话历史"""
        return get_chat_history_for_prompt()

    # ================================================================
    # 依赖注入
    # ================================================================

    def attach_retriever(self, retriever_fn: Callable[[str, int], List[Document]]) -> "RAGAgent":
        """注入检索器"""
        self._retriever_fn = retriever_fn
        set_tool_deps(retriever_fn=retriever_fn)
        return self

    def attach_memory(self, memory_manager) -> "RAGAgent":
        """注入记忆管理器"""
        self._memory_manager = memory_manager
        set_tool_deps(memory_manager=memory_manager)
        return self

    def attach_graph(self, graph_chain) -> "RAGAgent":
        """注入知识图谱链（供 Agent 的 graph_query 工具调用）"""
        set_tool_deps(graph_chain=graph_chain)
        return self

    def attach_focus(self, focus_callback: Callable[[str], str]) -> "RAGAgent":
        """注入文档聚焦回调"""
        set_tool_deps(focus_callback=focus_callback)
        return self

    def attach_doc_list(self, list_callback: Callable[[], List[str]]) -> "RAGAgent":
        """注入文档列表回调"""
        set_tool_deps(list_callback=list_callback)
        return self

    def set_focus(self, focus_set: Optional[Set[str]]) -> None:
        """添加文档到会话级文档列表（累积，不覆盖）。

        每次上传文档时调用，将新文档的 doc_id + 关键词追加到 _session_docs。
        Agent 的检索范围自动更新为全部 session_docs。
        """
        if not focus_set:
            return
        self._session_docs.update(focus_set)
        set_tool_deps(source_filter=lambda: self._session_docs)
        human = sorted(f for f in self._session_docs if not f.startswith("doc_"))
        logger.info(f"📋 会话文档列表: {human or sorted(self._session_docs)} (共 {len(self._session_docs)} 项)")

    def clear_session_docs(self) -> None:
        """清空会话文档列表（新会话开始时调用）"""
        self._session_docs.clear()
        set_tool_deps(source_filter=None)
        logger.info("📋 会话文档列表已清空")

    # ================================================================
    # 便捷方法
    # ================================================================

    def ask(self, question: str) -> AgentResult:
        """同步问答（兼容旧接口）"""
        return self.run(question)

    async def aask(self, question: str) -> AgentResult:
        """异步问答"""
        return await self.arun(question)

    def quick_answer(self, question: str, retrieved_docs: List[Document]) -> str:
        """快速回答（跳过 ReAct 循环，直接用检索结果 + LLM 生成）"""
        if not retrieved_docs:
            return "抱歉，知识库中没有找到相关内容。"

        context = self._context_builder.build(
            user_query=question,
            retrieved_docs=retrieved_docs,
            system_instructions=RAG_SYSTEM_INSTRUCTION,
        )

        prompt = QUICK_ANSWER_PROMPT.format(context=context, question=question)
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"快速回答失败: {e}")
            return "系统错误，请重试。"


# ================================================================
# 对话历史辅助
# ================================================================

def get_chat_history_for_prompt(max_turns: int = 6) -> str:
    """从短期记忆获取最近对话，格式化为 prompt 文本"""
    try:
        from core.memory import get_memory
        mem = get_memory()
        msgs = mem.load_memory_variables({}).get("chat_history", [])
        if not msgs:
            return "（暂无对话历史）"
        lines = []
        for m in msgs[-max_turns * 2:]:
            role = "用户" if (hasattr(m, 'type') and m.type == "human") else "助手"
            content = m.content if hasattr(m, 'content') else str(m)
            lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines)
    except Exception:
        return "（暂无对话历史）"


# ================================================================
# 工厂函数
# ================================================================

_rag_agent: Optional[RAGAgent] = None


def create_rag_agent(
    llm: Optional[BaseChatModel] = None,
    **kwargs,
) -> RAGAgent:
    """创建 RAG Agent（工厂函数）

    自动注入项目中的检索器、记忆管理器等依赖。
    """
    global _rag_agent

    if llm is None:
        from core.infrastructure.llm import get_llm
        llm = get_llm()

    agent = RAGAgent(llm=llm, **kwargs)

    # 自动注入检索器
    try:
        from core.retrievers.factory import get_retriever
        retriever = get_retriever()

        def _retrieve(query: str, top_k: int = 8, **kwargs) -> List[Document]:
            source_filter = kwargs.get("source_filter", None)
            if source_filter:
                from core.retrievers.factory import get_retriever as _get_retriever
                normalized = _normalize_source_filter(source_filter)
                doc_count = len(normalized) if normalized else 0
                per_doc_min = max(1, min(3, top_k // max(1, doc_count)))
                effective_k = max(top_k, doc_count * per_doc_min) if doc_count > 1 else top_k
                
                logger.info(
                    f"Retrieval: top_k={top_k}, docs_in_filter={doc_count}, "
                    f"per_doc_min={per_doc_min}, effective_k={effective_k}"
                )
                
                r = _get_retriever(
                    source_filter=source_filter,
                    override_top_k=effective_k,
                )
                results = r.invoke(query)

                # 多文档覆盖补检：主检索后检查哪些文档缺失，逐文档单独补搜
                human_keywords = {f for f in normalized if not f.startswith("doc_")}
                if len(human_keywords) >= 2:
                    # 收集主检索已覆盖的 source
                    covered = set()
                    for d in results:
                        src = str(d.metadata.get("source", "") or "").lower()
                        if src:
                            covered.add(src)

                    # 找出完全缺失的文档（关键词不出现在任何 covered source 中）
                    missing = set()
                    for kw in human_keywords:
                        kw_low = kw.lower()
                        if not any(kw_low in c for c in covered):
                            missing.add(kw)

                    if missing:
                        logger.info(f"  🔄 主检索缺失 {len(missing)} 篇文档，逐篇补检: {missing}")
                        for kw in missing:
                            try:
                                r2 = _get_retriever(
                                    source_filter={kw},
                                    override_top_k=per_doc_min,
                                )
                                extra = r2.invoke(query)
                                if extra:
                                    results.extend(extra)
                                    logger.info(f"    ✅ {kw}: +{len(extra)} 条")
                                else:
                                    logger.info(f"    ⚠️ {kw}: 0 条（可能确实不相关）")
                            except Exception as e:
                                logger.warning(f"    ❌ {kw}: 补检失败: {e}")

                return results
            return retriever.invoke(query)

        def _normalize_source_filter(f) -> Set[str]:
            if f is None:
                return set()
            if isinstance(f, (set, frozenset)):
                return {v for v in f if isinstance(v, str) and v.strip()}
            if isinstance(f, (list, tuple)):
                return {v for v in f if isinstance(v, str) and v.strip()}
            if isinstance(f, str):
                s = f.strip()
                return {s} if s else set()
            return set()

        agent.attach_retriever(_retrieve)
        logger.info("RAGAgent: 已注入检索器")
    except Exception as e:
        logger.warning(f"RAGAgent: 注入检索器失败: {e}")

    # 自动注入记忆管理器
    try:
        from core.memory import get_memory_manager
        memory_manager = get_memory_manager()
        agent.attach_memory(memory_manager)
        logger.info("RAGAgent: 已注入记忆管理器")
    except Exception as e:
        logger.warning(f"RAGAgent: 注入记忆管理器失败: {e}")

    # 自动注入文档列表
    try:
        from core.doc.doc_id_registry import get_doc_id_registry

        def _list_docs() -> List[str]:
            reg = get_doc_id_registry()
            return reg.get_all_sources()

        agent.attach_doc_list(_list_docs)
        logger.info("RAGAgent: 已注入文档列表")
    except Exception as e:
        logger.warning(f"RAGAgent: 注入文档列表失败: {e}")

    _rag_agent = agent
    return agent


def get_rag_agent() -> RAGAgent:
    """获取全局 RAG Agent 单例"""
    global _rag_agent
    if _rag_agent is None:
        create_rag_agent()
    return _rag_agent