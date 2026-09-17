# -*- coding: utf-8 -*-
# core/retrievers/strategies.py
"""增强检索策略 — 合并多查询(Multi-Query)和多跳(Multi-Hop)为统一模块

核心思路:
  1. LLM 将复杂问题拆解为 2-3 个定向子查询（多跳的拆解能力）
  2. 所有子查询并行执行（多查询的速度优势）
  3. 合并去重返回

对比:
  多查询: 同义改写 → 召回高但噪音大
  多跳:   串行依赖 → 慢且在 10 篇知识库里没必要串行
  增强:   拆解 + 并行 → 取两家长处，适合小规模知识库

消融实验:
  MULTI_HOP_ENABLED=false → baseline（单次检索）
  MULTI_HOP_ENABLED=true  → 增强检索（拆解 + 并行多轮）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from utils.logger import logger


_DECOMPOSE_SYSTEM = """你是问题拆解专家。将复杂问题拆解为 2-3 个定向子查询，每个子查询直指知识库中某个文档的关键信息。

规则:
1. 每行一个子查询，用数字编号
2. 子查询之间尽量指向不同文档/不同维度
3. 如果问题很简单只需一次检索，返回空

示例:
问题: "GA-LightGBM 和 GA-VMD-TCN 分别用什么优化算法？有什么区别？"
1. GA-LightGBM 使用的优化算法和超参数调优方法
2. GA-VMD-TCN 使用的遗传算法优化 VMD 参数的方法
3. 两种方法的主要区别"""


@dataclass
class EnhancedResult:
    query: str
    sub_queries: List[str] = field(default_factory=list)
    docs: List[Document] = field(default_factory=list)
    total_docs: int = 0
    is_multi: bool = False
    error: str = ""


class EnhancedRetriever:
    """增强检索器 — 拆解 + 并行多轮检索。

    Usage:
        er = EnhancedRetriever(retriever_fn=my_retriever, llm=my_llm)
        result = er.search("复杂问题")
        # result.docs: 合并去重后的所有文档

        # 消融实验: 关闭增强 = 单次检索
        er.set_enabled(False)
        result = er.search("复杂问题")
    """

    def __init__(
        self,
        retriever_fn: Optional[Callable] = None,
        llm: Optional[BaseChatModel] = None,
        enabled: Optional[bool] = None,
        sub_query_count: Optional[int] = None,
    ):
        self._retriever_fn = retriever_fn
        self._llm = llm
        self._enabled = enabled if enabled is not None else settings.multi_hop.enabled
        self._sub_count = sub_query_count or settings.multi_hop.hops_top_k

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def set_retriever(self, retriever_fn: Callable):
        self._retriever_fn = retriever_fn

    def set_llm(self, llm: BaseChatModel):
        self._llm = llm

    def search(self, query: str, top_k: int = 4) -> EnhancedResult:
        if not self._retriever_fn:
            return EnhancedResult(query=query, error="检索器未注入")

        if not self._enabled or not self._llm:
            return self._single_search(query, top_k)

        sub_queries = self._decompose(query)
        if not sub_queries:
            return self._single_search(query, top_k)

        logger.info(f"增强检索: {query[:50]} → {len(sub_queries)} 子查询: {sub_queries}")

        all_docs = self._parallel_search(sub_queries, top_k)
        merged = self._merge(all_docs)

        return EnhancedResult(
            query=query,
            sub_queries=sub_queries,
            docs=merged,
            total_docs=len(merged),
            is_multi=True,
        )

    def _single_search(self, query: str, top_k: int) -> EnhancedResult:
        docs = self._retriever_fn(query, top_k=top_k)
        return EnhancedResult(query=query, docs=docs, total_docs=len(docs))

    def _decompose(self, query: str) -> List[str]:
        prompt = f"问题: {query}\n\n子查询（如不需要拆解则留空）:"
        try:
            response = self._llm.invoke([
                SystemMessage(content=_DECOMPOSE_SYSTEM),
                HumanMessage(content=prompt),
            ])
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()

            if not text or "不需要" in text or "一步" in text:
                return []

            queries = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^\d+[\.\)、]\s*(.+)", line)
                if m:
                    q = m.group(1).strip()
                    if q and len(q) > 3:
                        queries.append(q)
            return queries[:3]
        except Exception as e:
            logger.warning(f"增强检索: 问题拆解失败: {e}")
            return []

    def _parallel_search(self, sub_queries: List[str], top_k: int) -> List[List[Document]]:
        results = []
        for sub_q in sub_queries:
            try:
                docs = self._retriever_fn(sub_q, top_k=top_k)
                results.append(docs)
            except Exception as e:
                logger.warning(f"增强检索: 子查询 '{sub_q[:40]}' 失败: {e}")
                results.append([])
        return results

    @staticmethod
    def _merge(doc_lists: List[List[Document]]) -> List[Document]:
        seen = set()
        merged = []
        for docs in doc_lists:
            for doc in docs:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                key = content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged


_enhanced_retriever: Optional[EnhancedRetriever] = None


def get_enhanced_retriever(
    retriever_fn: Optional[Callable] = None,
    llm: Optional[BaseChatModel] = None,
) -> EnhancedRetriever:
    global _enhanced_retriever
    if _enhanced_retriever is None:
        if llm is None:
            from core.infrastructure.llm import get_llm
            llm = get_llm()
        _enhanced_retriever = EnhancedRetriever(retriever_fn=retriever_fn, llm=llm)
    elif retriever_fn is not None:
        _enhanced_retriever.set_retriever(retriever_fn)
    return _enhanced_retriever