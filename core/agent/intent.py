# -*- coding: utf-8 -*-
"""简单意图识别 — 基于启发式规则 + LLM 兜底，区分三种问题类型

- fact_lookup:    简单事实查询（用 rag_search）
- comparison:     对比/分析/多跳推理（用 enhanced_search）
- graph_relation: 关系/连接类问题（用 graph_query）

用法:
    from core.agent.intent import classify, Intent

    intent = classify("Linformer 和 MoR 分别优化了什么？")
    if intent == Intent.COMPARISON:
        # 走 enhanced_search
"""

from __future__ import annotations

import enum
import re
from typing import Optional

from utils.logger import logger


class Intent(str, enum.Enum):
    FACT_LOOKUP = "fact_lookup"
    COMPARISON = "comparison"
    GRAPH_RELATION = "graph_relation"


# 对比/分析类触发词
_COMPARISON_PATTERNS = [
    r"(区别|异同|差异|对比|比较|分别|各自|哪个更|哪[一几]个更|有什么不同|有何不同)",
    r"(联系|关系|cross.doc|cross_doc|多跳|多文档|跨文档)",
    r"(与|和|及|vs\.?|VS\.?|versus).{2,30}(区别|对比|比较|关系|差异|联系|有什么|有何)",
    r"(什么关系|如何互补|能否结合|结合起来|之间.{1,5}关系)",
]

# 图/关系类触发词
_GRAPH_PATTERNS = [
    r"(知识图谱|图结构|节点|关系图|知识关联|图谱)",
    r"(实体.{1,5}关系|关联实体|路径查询|最短路径)",
    r"(包含哪些|由哪些|组成|构成|属于|从属|依赖关系)",
]


def pre_classify(question: str) -> Optional[Intent]:
    """基于启发式规则快速判断意图。
    
    Returns:
        Intent 或 None（规则无法判断时返回 None，走 LLM 兜底）
    """
    q = question.strip()
    if not q:
        return None

    for pattern in _GRAPH_PATTERNS:
        if re.search(pattern, q):
            return Intent.GRAPH_RELATION

    for pattern in _COMPARISON_PATTERNS:
        if re.search(pattern, q):
            return Intent.COMPARISON

    return None


async def classify(question: str) -> Intent:
    """完整的意图分类：规则优先，LLM 兜底。"""
    intent = pre_classify(question)
    if intent is not None:
        logger.debug(f"意图识别(规则): {intent}")
        return intent

    try:
        from core.infrastructure.llm import get_llm
        llm = get_llm()
        prompt = _INTENT_PROMPT.format(question=question)
        resp = await llm.ainvoke(prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
    except Exception as e:
        logger.warning(f"意图识别 LLM 调用失败: {e}，默认 fact_lookup")
        return Intent.FACT_LOOKUP

    intent_map = {
        "fact_lookup": Intent.FACT_LOOKUP,
        "comparison": Intent.COMPARISON,
        "graph_relation": Intent.GRAPH_RELATION,
    }
    result = intent_map.get(text, Intent.FACT_LOOKUP)
    logger.debug(f"意图识别(LLM): {result}")
    return result


_INTENT_PROMPT = """判断以下问题的类型，只回复一个词：
- fact_lookup: 简单事实查询，单一文档或单一知识点
- comparison: 需要对比、分析、跨文档推理或多跳
- graph_relation: 询问实体间的关系、构成、依赖

问题: {question}

类型: """