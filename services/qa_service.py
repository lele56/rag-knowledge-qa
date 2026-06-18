"""QA 服务 — 使用 RAGAgent 的 ReAct 模式问答"""

import asyncio
from typing import Dict, Any
from config.settings import settings
from utils.cache import AsyncTTLCache
from utils.logger import logger
from core.agent.rag_agent import get_rag_agent, RAGAgent
from chains.graph_chain import get_graph_chain


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
        self.cache = (
            AsyncTTLCache(settings.CACHE_TTL_SECONDS, settings.CACHE_MAX_SIZE)
            if settings.CACHE_ENABLED else None
        )

    async def ask(self, question: str) -> Dict[str, Any]:
        # 缓存
        if self.cache:
            cached = await self.cache.get(question)
            if cached:
                logger.info(f"Cache hit: {question[:50]}")
                return cached

        try:
            # ReAct Agent 执行
            result = await self.agent.aask(question)
            answer = result.answer
            sources = result.sources

            # 知识图谱增强
            if any(kw in question for kw in ["关系", "依赖", "组成", "属于", "连接", "影响"]):
                try:
                    graph_ans = await asyncio.to_thread(self.graph.run, question)
                    if graph_ans and "未找到" not in graph_ans:
                        answer += f"\n\n【知识图谱】{graph_ans}"
                except Exception as e:
                    logger.warning(f"Graph failed: {e}")

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