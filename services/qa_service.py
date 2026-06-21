"""QA 服务 — 使用 RAGAgent 的 ReAct 模式问答"""

import asyncio
from typing import Dict, Any, List
from config.settings import settings
from config.prompts import GRAPH_INTENT_PROMPT
from utils.cache import AsyncTTLCache
from utils.logger import logger
from core.agent.rag_agent import get_rag_agent, RAGAgent
from core.llm import get_llm
from chains.graph_chain import get_graph_chain


async def _should_use_graph(question: str) -> bool:
    """用 LLM 判断用户意图是否需要图谱查询"""
    try:
        llm = get_llm()
        prompt = GRAPH_INTENT_PROMPT.format(question=question)
        resp = await llm.ainvoke(prompt)
        content = resp.content.strip().upper() if hasattr(resp, 'content') else str(resp).strip().upper()
        return "YES" in content
    except Exception as e:
        logger.warning(f"图谱意图判断失败: {e}")
        return False


def _should_auto_store_semantic(question: str, answer: str) -> bool:
    """判断是否应自动存入语义记忆（Neo4j）"""
    if not settings.LS_ENABLED:
        return False
    if len(answer) < 200:
        return False
    from core.memory_system.semantic import _extract_concepts
    concepts = _extract_concepts(f"{question} {answer}")
    return len(concepts) >= 2


class QAService:
    """QA 服务层 — 封装 RAG Agent + 图谱增强

    用法:
        svc = QAService()
        result = await svc.ask("问题")
        print(result["answer"])
    """

    def __init__(self):
        self.agent: RAGAgent = get_rag_agent()
        self.graph = get_graph_chain()
        self.agent.attach_graph(self.graph)
        self.cache = (
            AsyncTTLCache(settings.CACHE_TTL_SECONDS, settings.CACHE_MAX_SIZE)
            if settings.CACHE_ENABLED else None
        )

    async def ask(self, question: str) -> Dict[str, Any]:
        if self.cache:
            cached = await self.cache.get(question)
            if cached:
                logger.info(f"Cache hit: {question[:50]}")
                return cached

        try:
            result = await self.agent.aask(question)
            answer = result.answer
            sources = result.sources or []

            # 图谱增强：LLM 判断意图，而非关键词匹配
            if await _should_use_graph(question):
                try:
                    graph_ans = await asyncio.to_thread(self.graph.run, question)
                    if graph_ans and "未找到" not in str(graph_ans):
                        answer += f"\n\n【知识图谱】{graph_ans}"
                except Exception as e:
                    logger.warning(f"Graph failed: {e}")

            # 自动存入语义记忆（满足条件时）
            if _should_auto_store_semantic(question, answer):
                try:
                    from core.memory_system.semantic import store_semantic
                    store_semantic(question, answer)
                    logger.info(f"语义记忆自动存入: {question[:40]}")
                except Exception as e:
                    logger.warning(f"语义记忆自动存入失败: {e}")

            resp = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "agent_steps": len(result.steps),
                "agent_state": result.state.value,
            }

            if self.cache:
                await self.cache.set(question, resp)
            return resp

        except Exception as e:
            logger.error(f"QA failed: {e}")
            return {
                "question": question,
                "answer": "系统错误，请重试。",
                "sources": [],
                "agent_steps": 0,
                "agent_state": "error",
            }


_service: QAService = None


def get_qa_service() -> QAService:
    global _service
    if _service is None:
        _service = QAService()
    return _service