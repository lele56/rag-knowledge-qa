# core/doc/document_chunk.py
"""文档分块：RecursiveCharacterTextSplitter / SemanticChunker 双模式 + 质量过滤"""
import re
from pathlib import Path
from typing import List, Literal
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from config.settings import settings
from utils.logger import logger


# ---------- token 计数 ----------

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    except Exception:
        cn = len(re.findall(r"[\u4e00-\u9fff]", text))
        en = len(text) - cn
        return int(cn / 1.5 + en / 4)


# ---------- 分块 ----------

def split_text_to_chunks(
    text: str,
    base_metadata: dict,
    strategy: Literal["recursive", "semantic"] = "recursive",
) -> List[Document]:
    """用 RecursiveCharacterTextSplitter 或 SemanticChunker 分块。

    recursive: 分层分隔符 \\n\\n → \\n → .。!！?？;； → 空格 → 字
    semantic:  基于 embedding 相似度的语义分块，适合学术论文
    """
    if not text.strip():
        return []

    chunk_size = settings.chunking.token_max
    chunk_overlap = settings.chunking.overlap_token

    if strategy == "semantic":
        return _semantic_split(text, base_metadata, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", "! ", "！", "? ", "？", ";", "；", " ", ""],
        length_function=count_tokens,
        keep_separator=True,
    )
    return _build_chunks(splitter.split_text(text), base_metadata)


def _semantic_split(
    text: str, base_metadata: dict, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from core.infrastructure.embeddings import get_embeddings
        splitter = SemanticChunker(
            embeddings=get_embeddings(),
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=chunk_size // 4,
        )
        docs = splitter.create_documents([text], [base_metadata])
        return _build_chunks([d.page_content for d in docs], base_metadata)
    except ImportError:
        logger.warning("langchain_experimental 未安装，回退到 recursive")
        return split_text_to_chunks(text, base_metadata, strategy="recursive")
    except Exception as e:
        logger.warning(f"SemanticChunker 失败: {e}，回退到 recursive")
        return split_text_to_chunks(text, base_metadata, strategy="recursive")


def _build_chunks(text_chunks: List[str], base_metadata: dict) -> List[Document]:
    chunks = []
    for tc in text_chunks:
        tc = tc.strip()
        if not tc:
            continue
        chunks.append(_build_chunk_doc(tc, count_tokens(tc), base_metadata))
    return chunks


# ---------- chunk 文档构建 ----------

def _build_chunk_doc(text: str, tokens: int, base_metadata: dict) -> Document:
    section_path = base_metadata.get("section", "")
    content = f"[章节: {section_path}]\n{text}" if section_path else text

    meta = dict(base_metadata)
    meta["chunk_tokens"] = tokens

    non_ws = len(re.sub(r"\s+", "", text))
    total = len(text) if text else 1
    char_density = non_ws / total
    meaningful = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z]", text))
    meaningful_ratio = meaningful / total if total > 0 else 0
    length_factor = min(tokens / 200.0, 1.0)
    meta["quality_score"] = round(
        0.3 * length_factor + 0.35 * char_density + 0.35 * min(meaningful_ratio * 2, 1.0), 3
    )
    meta["is_short"] = tokens < 30
    meta["is_heading_like"] = bool(
        re.search(r"^[\d一二三四五六七八九十第章节\s]{0,5}(章|节|篇|部分|chapter|section)\b",
                  text, re.IGNORECASE)) and len(text) < 80
    meta["is_toc_like"] = bool(re.search(
        r"(目录|contents|table\s*of\s*contents|参考文献|bibliography|appendix)",
        text[:100], re.IGNORECASE))

    return Document(page_content=content, metadata=meta)


# ---------- markitdown 解析 ----------

def try_markitdown(path: Path) -> List[Document]:
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(path))
        text = result.text_content
        if not text or len(text.strip()) < 20:
            return None
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = md_splitter.split_text(text)
        for s in sections:
            title_parts = [v for k, v in s.metadata.items() if k.startswith("Header")]
            if title_parts:
                s.metadata["section"] = " / ".join(title_parts)
        return sections
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"markitdown 解析 {path.name} 失败: {e}")
        return None


# ---------- 质量过滤 ----------

def filter_low_quality_chunks(chunks: List[Document]) -> List[Document]:
    cfg = settings.chunking
    kept = []
    dropped = 0
    for c in chunks:
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        content = c.page_content or ""
        body_len = len(re.sub(r"\s+", "", content))

        if meta.get("is_toc_like"):
            dropped += 1
            continue
        if meta.get("is_heading_like") and meta.get("chunk_tokens", 0) < cfg.quality_min_heading_tokens:
            dropped += 1
            continue
        q = float(meta.get("quality_score", 1.0) or 1.0)
        t = int(meta.get("chunk_tokens", 0) or 0)
        if q < cfg.quality_min_score and t < cfg.quality_min_tokens:
            dropped += 1
            continue
        if body_len < cfg.quality_min_body_len:
            dropped += 1
            continue
        kept.append(c)
    if dropped:
        logger.info(f"  → 分块时过滤: 丢弃 {dropped} 个低质量 chunk，保留 {len(kept)} 个")
    return kept