import asyncio
import time
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from config.settings import settings
from utils.logger import logger

_llm: Optional[BaseChatModel] = None


def _create_llm() -> BaseChatModel:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0,
        max_tokens=4096,
        request_timeout=120,
        max_retries=2,
    )


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = _create_llm()
    return _llm


async def call_llm_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    """带重试的 LLM 调用（字符串 prompt）"""
    last_error = None
    for attempt in range(max_retries):
        try:
            llm = get_llm()
            if hasattr(llm, 'ainvoke'):
                resp = await llm.ainvoke(prompt)
                return resp.content if hasattr(resp, 'content') else str(resp)
            else:
                resp = llm.invoke(prompt)
                return resp.content if hasattr(resp, 'content') else str(resp)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"LLM 调用失败 (第 {attempt + 1} 次)，{delay}s 后重试: {e}")
                await asyncio.sleep(delay)
    raise last_error


async def call_llm_messages_with_retry(
    messages: list, tools: list = None, max_retries: int = 3, base_delay: float = 2.0,
):
    """带重试的 LLM 调用（messages + 可选 tools），返回完整的 AIMessage（含 tool_calls）。"""
    last_error = None
    for attempt in range(max_retries):
        try:
            llm = get_llm()
            kwargs = {}
            if tools:
                kwargs["tools"] = tools
            if hasattr(llm, 'ainvoke'):
                resp = await llm.ainvoke(messages, **kwargs)
            else:
                resp = llm.invoke(messages, **kwargs)
            return resp
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"LLM 调用失败 (第 {attempt + 1} 次)，{delay}s 后重试: {e}")
                await asyncio.sleep(delay)
    raise last_error