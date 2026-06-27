# core/infrastructure/reranker.py
"""CrossEncoder 重排器：基于 langchain_classic 官方实现。"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config.settings import settings
from utils.logger import logger
from utils.device import get_device

_reranker = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        device = get_device()
        ce_device = 0 if device == "cuda" else "cpu"
        model_kwargs = {"device": ce_device}
        if token := os.environ.get("HF_TOKEN"):
            model_kwargs["token"] = token
        encoder = HuggingFaceCrossEncoder(
            model_name=settings.RERANKER_MODEL_PATH,
            model_kwargs=model_kwargs,
        )
        _reranker = CrossEncoderReranker(
            model=encoder,
            top_n=3,
        )
        logger.info(f"CrossEncoder 重排器加载完成: {settings.RERANKER_MODEL_PATH} (device={device})")
    return _reranker


def rerank_documents(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    """重排文档，返回得分最高的 top_k 条。"""
    if not docs:
        return []
    reranker = get_reranker()
    reranker.top_n = top_k
    return reranker.compress_documents(docs, query)