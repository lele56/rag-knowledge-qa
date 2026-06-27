# core/__init__.py
"""核心模块统一导出。

基础设施:
    from core.infrastructure import get_llm, get_embeddings, get_reranker
    from core.infrastructure import get_vector_store, get_graph, get_graph_status

记忆:
    from core.memory import get_memory, clear_memory, get_memory_manager

检索器工厂:
    from core.retrievers.factory import get_retriever

上下文构建:
    from core.context.builder import ContextBuilder, ContextConfig, build_rag_context

子模块:
    core.agent          — Agent 实现
    core.context        — GSSC 上下文构建
    core.doc            — 文档加载、切块、ID 注册
    core.infrastructure — LLM/Embedding/重排序/向量存储/图谱存储
    core.memory         — 短期记忆 + 记忆管理器
    core.memory.long_term  — 长期记忆系统（episodic + semantic + working）
    core.retrievers     — 检索器（base, hybrid, bm25, hyde, multi_query, factory）
    core.tools          — Agent 工具（rag_tools, pipeline, registry）
"""