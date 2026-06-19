# core/doc/document_loader.py
import hashlib
import logging
import warnings
from pathlib import Path
from typing import List, Set, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, PDFPlumberLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from utils.logger import logger
from core.doc.document_chunk import split_text_to_chunks, try_markitdown, filter_low_quality_chunks
from config.settings import settings

logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")

# ---------- doc_id ----------

def make_doc_id(path: Path) -> str:
    stem = path.stem.lower().strip()
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"doc_{h}"


# ---------- PDF 解析器 ----------

def _try_unstructured_pdf(path: Path) -> Optional[List[Document]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No languages specified")
            loader = UnstructuredPDFLoader(str(path), strategy="fast")
            docs = loader.load()
        if not docs or all(not d.page_content.strip() for d in docs):
            return None
        for i, d in enumerate(docs):
            d.metadata.setdefault("section", f"第 {i + 1} 页")
        return docs
    except Exception as e:
        logger.warning(f"UnstructuredPDFLoader 解析 {path.name} 失败: {e}")
        return None


def _try_pdfplumber(path: Path) -> Optional[List[Document]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No languages specified")
            loader = PDFPlumberLoader(str(path))
            docs = loader.load()
        if not docs:
            return None
        for i, d in enumerate(docs):
            d.metadata.setdefault("section", f"第 {i + 1} 页")
        return docs
    except Exception as e:
        logger.warning(f"PDFPlumberLoader 解析 {path.name} 失败: {e}")
        return None


def _fallback_load(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


# ---------- 主入口 ----------

def load_and_split_documents(file_paths: List[Path]) -> List[Document]:
    import time as _t

    all_chunks: List[Document] = []
    t_total = _t.time()

    for path in file_paths:
        ext = path.suffix.lower()
        try:
            size_kb = path.stat().st_size / 1024 if path.exists() else 0
            logger.info(f"📄 开始解析: {path.name} ({size_kb:.0f} KB)")

            # PDF 四级回退：Unstructured → PDFPlumber → markitdown → PyPDFLoader
            t0 = _t.time()
            if ext == ".pdf":
                sections = _try_unstructured_pdf(path)
                source_type = "unstructured"
                if sections is None:
                    sections = _try_pdfplumber(path)
                    source_type = "pdfplumber"
                if sections is None:
                    sections = try_markitdown(path)
                    source_type = "markitdown"
            else:
                sections = try_markitdown(path)
                source_type = "markitdown"
            if sections is None:
                sections = _fallback_load(path)
                source_type = "fallback"
            dt_md = _t.time() - t0

            doc_id = make_doc_id(path)
            for s in sections:
                s.metadata["source"] = str(path.name)
                s.metadata["doc_id"] = doc_id
                s.metadata["file_type"] = ext

            logger.info(
                f"  → 文本提取: {dt_md:.1f}s, {len(sections)} 段落 "
                f"(方式={source_type})"
            )

            t0 = _t.time()
            chunk_strategy = settings.chunking.strategy
            doc_chunks = []
            for section in sections:
                chunks = split_text_to_chunks(
                    section.page_content, section.metadata, strategy=chunk_strategy
                )
                doc_chunks.extend(chunks)
            dt_split = _t.time() - t0

            n_chunks = len(doc_chunks)
            for idx, c in enumerate(doc_chunks):
                c.metadata["doc_chunk_index"] = idx
                c.metadata["doc_chunk_total"] = n_chunks
                c.metadata["has_prev"] = idx > 0
                c.metadata["has_next"] = idx < n_chunks - 1
            logger.info(
                f"  → 分块: {dt_split:.1f}s, {len(doc_chunks)} chunks"
            )
            all_chunks.extend(doc_chunks)

        except Exception as e:
            logger.error(f"处理失败 {path.name}: {e}")

    all_chunks = filter_low_quality_chunks(all_chunks)

    for i, c in enumerate(all_chunks):
        c.metadata["chunk_index"] = i
        c.metadata["chunk_total"] = len(all_chunks)

    total_tokens = sum(c.metadata.get("chunk_tokens", 0) for c in all_chunks)
    low_q = sum(1 for c in all_chunks if c.metadata.get("quality_score", 1) < 0.4)
    dt_total = _t.time() - t_total
    logger.info(
        f"✅ 切分完成 -> {len(all_chunks)} chunks, ~{total_tokens} tokens, "
        f"低质量 {low_q} 个, 总耗时 {dt_total:.1f}s"
    )

    seen_doc_ids: Set[str] = set()
    for c in all_chunks:
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        did = str(meta.get("doc_id", "") or "")
        src = str(meta.get("source", "") or "")
        if did and src and did not in seen_doc_ids:
            seen_doc_ids.add(did)
            try:
                from core.doc.doc_id_registry import get_doc_id_registry
                reg = get_doc_id_registry()
                reg.register(did, src)
            except Exception as _e:
                logger.debug(f"注册 doc_id 失败 (非致命): {_e}")
    if seen_doc_ids:
        logger.info(f"📝 已注册 {len(seen_doc_ids)} 个 doc_id")

    return all_chunks