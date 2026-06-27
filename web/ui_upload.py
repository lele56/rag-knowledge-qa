# web/ui_upload.py
"""Tab 2: 上传文件 — 文档上传 + 状态刷新"""

from typing import List, Dict
from datetime import datetime

from services.document_service import get_document_service
from utils.logger import logger
from web.ui_chat import _file_to_path, _render_uploaded_list, doc_svc


async def upload_files(files, records):
    global doc_svc
    if records is None:
        records = []
    if not files:
        return "⚠️ 请先选择文件", _render_uploaded_list(records), None, records

    paths, file_names = [], []
    for f in files:
        p = _file_to_path(f)
        if p and p.exists():
            paths.append(p)
            file_names.append(p.name)

    if not paths:
        return "❌ 没有可处理的文件", _render_uploaded_list(records), None, records

    try:
        if doc_svc is None:
            doc_svc = get_document_service()
        time_str = datetime.now().strftime("%H:%M:%S")

        result = await doc_svc.add_documents_async(paths)
        chunk_count = result.get("chunk_count", 0) if isinstance(result, dict) else result
        duplicates = result.get("duplicates", []) if isinstance(result, dict) else []
        new_files = result.get("new_files", file_names) if isinstance(result, dict) else file_names
        focus_sources = result.get("focus_sources", []) if isinstance(result, dict) else []

        if focus_sources:
            try:
                from core.agent.rag_agent import get_rag_agent
                get_rag_agent().set_focus(set(focus_sources))
            except Exception as e:
                logger.debug(f"设置聚焦源失败: {e}")

        new_records = list(records)

        for name in new_files:
            new_records.append({"name": name, "status": "⏳ 后台写入中", "chunks": chunk_count, "time": time_str})

        for name in duplicates:
            new_records.append({"name": name, "status": "♻️ 已存在（跳过写入，已加入会话列表）", "chunks": 0, "time": time_str})

        parts = []
        if new_files:
            parts.append(f"✅ **已接收 {len(new_files)} 个新文件**  |  {chunk_count} chunks")
        if duplicates:
            parts.append(f"♻️ **跳过 {len(duplicates)} 个重复文件**（已加入会话列表）")
        status = "  |  ".join(parts) if parts else "⚠️ 未处理任何文件"

        return status, _render_uploaded_list(new_records), None, new_records
    except Exception as e:
        logger.error(f"上传失败: {e}")
        time_str = datetime.now().strftime("%H:%M:%S")
        new_records = list(records)
        for name in file_names:
            new_records.append({"name": name, "status": "❌ 失败", "chunks": 0, "time": time_str})
        return f"❌ **上传失败**: `{e}`", _render_uploaded_list(new_records), None, new_records


def refresh_upload_status(records):
    global doc_svc
    if not records:
        return "*暂无上传记录*", records

    if doc_svc is None:
        doc_svc = get_document_service()

    bg_errors = doc_svc.get_bg_errors()
    failed_files = set()
    failed_info = {}
    for err in bg_errors:
        for f in err["files"]:
            failed_files.add(f)
            failed_info[f] = err["error"][:100]

    kb_files = set()
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        kb_files = set(reg.get_all_sources())
    except Exception as e:
        logger.debug(f"获取文档注册表失败: {e}")
        kb_files = set()
    for r in records:
        name = r.get("name", "")
        if name in failed_files:
            r = dict(r)
            r["status"] = f"❌ 写入失败: {failed_info[name]}"
        elif name in kb_files:
            r = dict(r)
            r["status"] = "✅ 已入库"
        elif r.get("status", "").startswith("⏳"):
            r = dict(r)
            r["status"] = "⏳ 后台写入中"
        updated.append(r)

    ok_count = sum(1 for r in updated if r.get("status", "").startswith("✅"))
    fail_count = sum(1 for r in updated if r.get("status", "").startswith("❌"))
    pending_count = sum(1 for r in updated if r.get("status", "").startswith("⏳"))

    parts = [f"📊 已入库 {ok_count} 篇"]
    if fail_count:
        parts.append(f"❌ 失败 {fail_count} 篇")
    if pending_count:
        parts.append(f"⏳ 处理中 {pending_count} 篇")
    status = "  |  ".join(parts)

    return _render_uploaded_list(updated), updated