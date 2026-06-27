# core/retrievers/enhanced.py
"""检索增强：Qdrant 格式转换 + 相似度降噪 + 上下文标记"""
import re
from typing import Optional, Any, List
import numpy as np
from langchain_core.documents import Document
from core.infrastructure.embeddings import get_embeddings
from utils.logger import logger


# ============================================================
# ScoredPoint → LangChain Document
# ============================================================

def scored_point_to_doc(point: Any) -> Optional[Document]:
    payload = getattr(point, 'payload', None) or {}
    if not isinstance(payload, dict):
        return None
    content = payload.get("page_content", "")
    if not content:
        return None
    meta = payload.get("metadata", None) if isinstance(payload.get("metadata"), dict) else {}
    if not meta:
        meta = {k: v for k, v in payload.items() if k != "page_content"}
    if "source" not in meta and "source" in payload:
        meta["source"] = payload["source"]
    if "doc_id" not in meta and "doc_id" in payload:
        meta["doc_id"] = payload["doc_id"]
    score = float(getattr(point, 'score', 0.0) or 0.0)
    meta["_score"] = score
    return Document(page_content=content, metadata=meta)


# ============================================================
# 降噪：向量相似度过滤 + 内容指纹去重
# ============================================================

def _embedding_similarity_filter(
    docs: List[Document],
    query: str,
    threshold: float = 0.3,
    k: int = 100,
) -> List[Document]:
    """用 embedding 余弦相似度过滤文档，替代已废弃的 langchain_classic EmbeddingsFilter。"""
    if not docs:
        return []
    embeddings = get_embeddings()
    query_vec = np.array(embeddings.embed_query(query))
    doc_vecs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    # 余弦相似度
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    scores = np.dot(doc_norms, query_norm)
    # 筛选 + 排序
    indices = np.argsort(scores)[::-1]
    result = []
    for i in indices:
        if scores[i] >= threshold:
            result.append(docs[i])
        if len(result) >= k:
            break
    return result


def denoise_docs(
    docs: List[Document],
    query: str = "",
    min_score: float = 0.3,
    focused_mode: bool = False,
) -> List[Document]:
    """向量相似度过滤 + 指纹去重。"""
    if not docs:
        return docs
    before = len(docs)

    if focused_mode:
        min_score = max(min_score - 0.2, 0.08)
        logger.info(f"   降噪: 聚焦模式，相似度阈值={min_score:.2f}")

    if query:
        try:
            filtered = _embedding_similarity_filter(docs, query, threshold=min_score)
        except Exception as e:
            logger.warning(f"相似度过滤失败: {e}，退回原始列表")
            filtered = list(docs)
    else:
        filtered = [d for d in docs if float(d.metadata.get("_score", 1.0) or 1.0) >= min_score]

    if not filtered:
        filtered = docs[:3]

    deduped = _dedup_docs(filtered)
    after = len(deduped)
    if before != after:
        logger.info(f"   降噪: {before} → {after} 条")
    return deduped


def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    kept: List[Document] = []
    for d in docs:
        content = d.page_content or ""
        body = content
        if body.startswith("[章节"):
            nl_idx = body.find("\n")
            if nl_idx > 0:
                body = body[nl_idx + 1:]
        body_clean = re.sub(r"\s+", "", body)
        if len(body_clean) > 180:
            fingerprint = body_clean[30:180]
        elif len(body_clean) > 80:
            fingerprint = body_clean[20:170]
        else:
            fingerprint = body_clean
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        src = str(meta.get("source", "") or "")
        idx = str(meta.get("chunk_index", ""))
        if src and idx:
            fingerprint = f"{src}::{idx}::{fingerprint}"
        if fingerprint and fingerprint in seen:
            continue
        if fingerprint:
            seen.add(fingerprint)
        kept.append(d)
    return kept


# ============================================================
# 上下文标记
# ============================================================

def enrich_with_context(
    docs: List[Document],
    window: int = 1,
) -> List[Document]:
    if not docs or window <= 0:
        return docs
    for d in docs:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        idx = meta.get("doc_chunk_index")
        total = meta.get("doc_chunk_total")
        src = str(meta.get("source", "未知文档") or "未知文档")
        section = str(meta.get("section", "") or "")
        if idx is not None and total is not None:
            ctx_start = max(0, idx - window)
            ctx_end = min(total - 1, idx + window)
            meta["context_window"] = f"[{ctx_start}..{ctx_end}] / {total}"
            meta["doc_boundary"] = "start" if idx == 0 else ("end" if idx == total - 1 else "middle")
        header_parts = [f"文档: {src}"]
        if idx is not None and total is not None:
            header_parts.append(f"片段: {int(idx) + 1}/{total}")
        if section:
            header_parts.append(f"章节: {section}")
        header = f"[{' | '.join(header_parts)}]"
        content = d.page_content or ""
        if not content.strip().startswith("[文档:"):
            d.page_content = f"{header}\n{content}"
    return docs