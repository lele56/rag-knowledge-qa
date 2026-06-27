# services/document_service.py
from pathlib import Path
import asyncio
from core.doc.document_loader import load_and_split_documents, make_doc_id
from core.infrastructure.vector_store import get_vector_store, add_documents_in_batches, _get_client, delete_by_doc_id, get_doc_chunk_sample
from core.retrievers.bm25 import reset_bm25_retriever
from config.settings import settings
from utils.logger import logger


class DocumentService:
    def __init__(self):
        self.store = get_vector_store()
        self._bg_tasks: dict = {}
        self._bg_errors: list = []  # 后台写入失败记录

    def _check_duplicate(self, path: Path) -> bool:
        """检查是否已存在可用 chunk（registry + Qdrant 双重验证）。

        三层防线：
          1. registry 未注册 → 不是重复
          2. registry 已注册但 Qdrant 中 0 chunks → 清理注册表，不是重复
          3. Qdrant 有 chunks 但内容损坏（无空格等）→ 清理旧数据，不是重复

        只有 registry 已注册 且 Qdrant 有 chunk 且内容正常 → 才是真重复。
        返回 True 表示重复（应跳过写入）。
        """
        try:
            from core.doc.doc_id_registry import get_doc_id_registry
            from config.settings import settings
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            reg = get_doc_id_registry()
            doc_id = make_doc_id(path)

            # 第 1 层：registry 未注册 → 不是重复
            existing = reg.get_source_for_doc_id(doc_id)
            if not existing:
                return False

            # 第 2 层：registry 已注册，检查 Qdrant 实际点数
            client = _get_client()
            try:
                count_result = client.count(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                    exact=True,
                )
                chunk_count = count_result.count
            except Exception as e:
                logger.warning(f"Qdrant 查询失败（按新文件处理）: {e}")
                return False

            if chunk_count == 0:
                logger.warning(
                    f"🔍 重复检测: {path.name} → 已注册但 Qdrant 无 chunk（注册表残留），"
                    f"清理注册表后重新上传"
                )
                reg.unregister(doc_id)
                return False

            # 第 3 层：Qdrant 有 chunk，检查内容质量
            if self._is_content_corrupted(doc_id, path.name):
                logger.warning(
                    f"🔍 重复检测: {path.name} → {chunk_count} chunks 但内容损坏，"
                    f"清理旧数据后重新上传"
                )
                delete_by_doc_id(doc_id)
                reg.unregister(doc_id)
                return False

            logger.info(f"🔍 重复检测: {path.name} → 已注册({existing})，{chunk_count} chunks 正常，跳过写入")
            return True

        except Exception as e:
            logger.warning(f"重复检测异常（跳过，按新文件处理）: {e}")
            return False

    def _is_content_corrupted(self, doc_id: str, filename: str) -> bool:
        """检查 chunk 内容是否损坏（如 PDF 解析无空格导致单词粘连）。

        检测策略：扫描每个 chunk 中最长连续 ASCII 字母串。
        正常英文文本中单词边界由空格分隔，最长连续字母串通常 < 20。
        损坏文本（单词粘连）会出现 50+ 连续字母的异常长串。
        """
        import re

        samples = get_doc_chunk_sample(doc_id, sample_size=8)
        if not samples:
            return False

        bad_count = 0
        for s in samples:
            content = s.get("content", "")
            # 找出所有连续 ASCII 字母串
            runs = re.findall(r'[A-Za-z]{20,}', content)
            if not runs:
                continue
            max_run = max(len(r) for r in runs)
            if max_run >= 50:
                bad_count += 1
                logger.info(
                    f"  ⚠️ 损坏 chunk: 最长连续字母串={max_run} 字符 "
                    f"[{s.get('source', '?')}] 示例={runs[0][:60]}"
                )

        # 超过一半的英文 chunk 异常 → 判定为损坏文件
        is_corrupted = bad_count > 0 and bad_count >= len(samples) / 2
        if is_corrupted:
            logger.warning(f"  ❌ 内容损坏判定: {bad_count}/{len(samples)} 个样本单词粘连 ({filename})")
        return is_corrupted

    def add_documents(self, paths: list[Path]) -> int:
        """同步版：切分 + 嵌入 + 写入，全程阻塞（兼容旧调用）"""
        if not paths:
            logger.info("没有提供任何文件路径")
            return 0

        new_paths: list[Path] = []
        duplicates: list[str] = []
        for p in paths:
            if self._check_duplicate(p):
                duplicates.append(Path(p).name)
            else:
                new_paths.append(p)

        if duplicates:
            logger.info(f"🔍 重复文件跳过: {duplicates}")

        if not new_paths:
            logger.info("所有文件均为重复，无需写入")
            return 0

        chunks = load_and_split_documents(new_paths)
        if not chunks:
            logger.info("没有生成任何 chunks")
            return 0

        logger.info(f"切分完成 -> {len(chunks)} chunks，开始分批写入向量库...")
        added = add_documents_in_batches(self.store, chunks)
        logger.info(f"✅ 完成: {added}/{len(chunks)} chunks 已写入")

        try:
            reset_bm25_retriever()
        except Exception as e:
            logger.debug(f"BM25 重置失败: {e}")

        return added

    def _build_focus_sources(self, paths: list[Path]) -> set:
        focus_sources = set()
        for p in paths:
            p_path = Path(p)
            focus_sources.add(make_doc_id(p_path))
            focus_sources.add(p_path.stem)
        return focus_sources

    async def add_documents_async(self, paths: list[Path]) -> dict:
        """异步版：切分 → 重复检测 → 后台嵌入+写入。

        重复文件跳过写入。
        返回 {"chunk_count": int, "duplicates": list[str], "new_files": list[str], "focus_sources": list[str]}
        （focus_sources 由调用方注入到 agent 的会话文档列表）
        """
        if not paths:
            logger.info("没有提供任何文件路径")
            return {"chunk_count": 0, "duplicates": [], "new_files": [], "focus_sources": []}

        # 逐文件检测重复
        new_paths: list[Path] = []
        duplicates: list[str] = []
        for p in paths:
            if self._check_duplicate(p):
                duplicates.append(Path(p).name)
            else:
                new_paths.append(p)

        if duplicates:
            logger.info(f"🔍 重复文件跳过写入: {duplicates}")

        # 构建 focus_sources（含重复文件——它们已有 chunks 在 Qdrant 中）
        all_focus = self._build_focus_sources(paths)

        if not new_paths:
            logger.info("所有文件均为重复，无需写入")
            return {"chunk_count": 0, "duplicates": duplicates, "new_files": [], "focus_sources": list(all_focus)}

        chunks = load_and_split_documents(new_paths)
        if not chunks:
            logger.info("没有生成任何 chunks")
            return {"chunk_count": 0, "duplicates": duplicates, "new_files": [], "focus_sources": list(all_focus)}

        chunk_count = len(chunks)
        new_file_names = [Path(p).name for p in new_paths]

        logger.info(f"切分完成 -> {chunk_count} chunks（{len(new_paths)} 个新文件），后台异步写入向量库...")

        async def _bg_work():
            try:
                added = await asyncio.to_thread(add_documents_in_batches, self.store, chunks)
                logger.info(f"✅ 后台完成: {added}/{chunk_count} chunks 已写入")
                await asyncio.to_thread(reset_bm25_retriever)
            except Exception as e:
                err_msg = str(e)[:200]
                logger.error(f"后台写入失败: {e}")
                self._bg_errors.append({
                    "files": new_file_names,
                    "chunks": chunk_count,
                    "error": err_msg,
                    "time": __import__("datetime").datetime.now().strftime("%H:%M:%S"),
                })

        asyncio.create_task(_bg_work())
        return {"chunk_count": chunk_count, "duplicates": duplicates, "new_files": new_file_names, "focus_sources": list(all_focus)}

    def get_bg_errors(self) -> list:
        """返回后台写入失败记录列表，每次调用后清空。"""
        errors = list(self._bg_errors)
        self._bg_errors.clear()
        return errors

_service = None
def get_document_service() -> DocumentService:
    global _service
    if _service is None:
        _service = DocumentService()
    return _service