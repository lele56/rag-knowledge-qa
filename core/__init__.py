# core/__init__.py
# 核心模块统一导出
#
# 单例入口（全局缓存，避免重复创建）：
#   from core.llm import get_llm
#   from core.embeddings import get_embeddings
#   from core.vector_store import get_vector_store
#   from core.reranker import get_reranker
#   from core.memory import get_memory, clear_memory
#   from core.memory_manager import get_memory_manager
#   from core.retriever_factory import get_retriever
#   from core.context_builder import ContextBuilder, get_context_builder, build_rag_context
#   from core.graph_store import get_graph, get_graph_status
#
# 子模块：
#   core.agent          — Agent 实现
#   core.context        — GSSC 上下文构建
#   core.doc            — 文档加载、切块、ID 注册
#   core.memory_system  — 长期记忆系统
#   core.retrievers     — 检索器（base, hybrid, bm25, hyde, multi_query）
#   core.tools          — Agent 工具（rag_tools, pipeline, registry）