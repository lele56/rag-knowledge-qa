# core/doc/document_chunk.py
"""文档分块：RecursiveCharacterTextSplitter + 质量过滤

分块策略:
  - recursive: 分层分隔符 \n\n → \n → .。!！?？;； → 空格 → 字
  - semantic:  已废弃（2026-08-18），embedding 成本高，效果提升不明显
"""
import re
from pathlib import Path
from typing import List, Literal, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from config.settings import settings
from utils.logger import logger
from utils.token_utils import count_tokens


# ============================================================
# 常量
# ============================================================

# 质量评分权重
QUALITY_WEIGHT_LENGTH = 0.3
QUALITY_WEIGHT_DENSITY = 0.35
QUALITY_WEIGHT_MEANINGFUL = 0.35

# 阈值
SHORT_CHUNK_TOKEN_THRESHOLD = 30
HEADING_MAX_LENGTH = 80
TOC_CHECK_LENGTH = 100
MIN_MARKITDOWN_TEXT_LENGTH = 20


# ---------- 分块 ----------

def split_text_to_chunks(
    text: str,
    base_metadata: dict,
    strategy: Literal["recursive", "semantic"] = "recursive",
) -> List[Document]:
    """用 RecursiveCharacterTextSplitter 分块。

    recursive: 分层分隔符 \n\n → \n → .。!！?？;； → 空格 → 字
    semantic:  已废弃，自动回退到 recursive
    """
    if not text.strip():
        return []

    chunk_size = settings.chunking.token_max
    chunk_overlap = settings.chunking.overlap_token

    if strategy == "semantic":
        logger.warning("semantic 分块策略已废弃，使用 recursive 代替")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", "! ", "！", "? ", "？", ";", "；", " ", ""],
        length_function=count_tokens,
        keep_separator=True,
    )
    return _build_chunks(splitter.split_text(text), base_metadata)


def _build_chunks(text_chunks: List[str], base_metadata: dict) -> List[Document]:
    """将文本块列表转换为 Document 对象列表。
    
    Args:
        text_chunks: 原始文本块列表
        base_metadata: 基础元数据（source、doc_id 等）
        
    Returns:
        Document 对象列表（已过滤空块）
    """
    chunks = []
    for tc in text_chunks:
        tc = tc.strip()
        if not tc:
            continue
        chunks.append(_build_chunk_doc(tc, count_tokens(tc), base_metadata))
    return chunks


# ---------- chunk 文档构建 ----------

def _build_chunk_doc(text: str, tokens: int, base_metadata: dict) -> Document:
    """构建单个 chunk 的 Document 对象，包含质量评分。
    
    Args:
        text: chunk 文本内容
        tokens: token 数量
        base_metadata: 基础元数据
        
    Returns:
        带质量评分的 Document 对象
    """
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
        QUALITY_WEIGHT_LENGTH * length_factor
        + QUALITY_WEIGHT_DENSITY * char_density
        + QUALITY_WEIGHT_MEANINGFUL * min(meaningful_ratio * 2, 1.0),
        3,
    )
    meta["is_short"] = tokens < SHORT_CHUNK_TOKEN_THRESHOLD
    meta["is_heading_like"] = bool(
        re.search(r"^[\d一二三四五六七八九十第章节\s]{0,5}(章|节|篇|部分|chapter|section)\b",
                  text, re.IGNORECASE)) and len(text) < HEADING_MAX_LENGTH
    meta["is_toc_like"] = bool(re.search(
        r"(目录|contents|table\s*of\s*contents|参考文献|bibliography|appendix)",
        text[:TOC_CHECK_LENGTH], re.IGNORECASE))

    return Document(page_content=content, metadata=meta)


# ---------- markitdown 解析 ----------

def try_markitdown(path: Path) -> Optional[List[Document]]:
    """尝试用 markitdown 解析文档为带章节结构的文本。
    
    Args:
        path: 文档文件路径
        
    Returns:
        解析后的章节列表，如果解析失败或内容过短返回 None
    """
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(path))
        text = result.text_content
        if not text or len(text.strip()) < MIN_MARKITDOWN_TEXT_LENGTH:
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
    """过滤低质量 chunk，保留有价值的文本块。
    
    过滤规则（满足任一即丢弃）:
      1. 目录/参考文献/附录等无实质内容
      2. 短标题（如"第一章"）且 token 数不足
      3. 质量评分过低 或 token 数过少
      4. 正文去空格后长度不足
    
    Args:
        chunks: 原始 chunk 列表
        
    Returns:
        过滤后的 chunk 列表
    """
    cfg = settings.chunking
    kept = []
    dropped = 0
    drop_reasons = {"toc": 0, "heading": 0, "low_quality": 0, "too_short": 0}
    
    for c in chunks:
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        content = c.page_content or ""
        body_len = len(re.sub(r"\s+", "", content))

        # 规则 1: 过滤目录/参考文献/附录
        if meta.get("is_toc_like"):
            drop_reasons["toc"] += 1
            dropped += 1
            continue
        
        # 规则 2: 过滤短标题（如"第一章"、"Section 2"）
        if meta.get("is_heading_like") and meta.get("chunk_tokens", 0) < cfg.quality_min_heading_tokens:
            drop_reasons["heading"] += 1
            dropped += 1
            continue
        
        # 规则 3: 过滤低质量或过短 chunk（OR 逻辑，更严格）
        q = float(meta.get("quality_score", 1.0) or 1.0)
        t = int(meta.get("chunk_tokens", 0) or 0)
        if q < cfg.quality_min_score or t < cfg.quality_min_tokens:
            drop_reasons["low_quality"] += 1
            dropped += 1
            continue
        
        # 规则 4: 过滤正文过短的 chunk
        if body_len < cfg.quality_min_body_len:
            drop_reasons["too_short"] += 1
            dropped += 1
            continue
        
        kept.append(c)
    
    if dropped:
        reason_str = ", ".join(f"{k}: {v}" for k, v in drop_reasons.items() if v > 0)
        logger.info(f"  → 分块时过滤: 丢弃 {dropped} 个低质量 chunk ({reason_str})，保留 {len(kept)} 个")
    return kept