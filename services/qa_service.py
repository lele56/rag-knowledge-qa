# services/qa_service.py
"""QA 服务 — 使用 RAGAgent 的 ReAct 模式问答"""

import asyncio
from typing import Dict, Any
from config.settings import settings
from config.prompts import GRAPH_INTENT_PROMPT
from utils.cache import AsyncTTLCache, AsyncRedisCache
from utils.logger import logger
from core.agent.rag_agent import get_rag_agent, RAGAgent
from core.infrastructure.llm import get_llm
from chains.graph_chain import get_graph_chain
from services.rate_limit import check_rate_limit


async def _should_use_graph(question: str) -> bool:
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
    if not settings.LS_ENABLED:
        return False
    if len(answer) < 200:
        return False
    from core.memory.long_term.semantic import _extract_concepts
    concepts = _extract_concepts(f"{question} {answer}")
    return len(concepts) >= 2


class QAService:
    def __init__(self):
        self.agent: RAGAgent = get_rag_agent()
        self.graph = get_graph_chain()
        self.agent.attach_graph(self.graph)

        if settings.CACHE_ENABLED:
            if settings.cache.backend == "redis":
                self.cache = AsyncRedisCache(settings.redis.url, settings.cache.ttl_seconds)
            else:
                self.cache = AsyncTTLCache(settings.cache.ttl_seconds, settings.cache.max_size)
        else:
            self.cache = None

    async def ask(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        if self.cache:
            cached = await self.cache.get(question)
            if cached:
                logger.info(f"Cache hit: {question[:50]}")
                return cached

        if not await check_rate_limit(session_id):
            return {
                "question": question,
                "answer": "请求过于频繁，请稍后再试。",
                "sources": [],
                "agent_steps": 0,
                "agent_state": "rate_limited",
            }

        try:
            result = await self.agent.aask(question)
            answer = result.answer
            sources = result.sources or []

            if await _should_use_graph(question):
                try:
                    graph_ans = await asyncio.to_thread(self.graph.run, question)
                    if graph_ans and "未找到" not in str(graph_ans):
                        answer += f"\n\n【知识图谱】{graph_ans}"
                except Exception as e:
                    logger.warning(f"Graph failed: {e}")

            if _should_auto_store_semantic(question, answer):
                try:
                    from core.memory.long_term.semantic import store_semantic
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