#!/usr/bin/env python3
"""
重建 Qdrant 集合 —— 清空并重新导入所有论文文档

用途：当分块策略改变后，清空旧数据重新分块入库
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from utils.logger import logger


def clear_collection():
    """清空整个 Qdrant 集合"""
    from core.vector_store import _get_client, get_vector_store

    client = _get_client()
    col_name = settings.QDRANT_COLLECTION_NAME

    try:
        try:
            info = client.get_collection(col_name)
            count = info.points_count
            logger.info(f"当前集合 {col_name} 共有 {count} 个点")
            client.delete_collection(col_name)
            logger.info(f"已删除集合 {col_name}")
        except Exception:
            logger.info(f"集合 {col_name} 不存在，跳过删除")

        # 触发 vector_store 重建，自动创建集合
        from core.vector_store import _store as _store_global
        import core.vector_store as vs
        vs._store = None
        vs._payload_indexes_ensured = False
        store = get_vector_store()
        logger.info(f"集合 {col_name} 已自动重建")

        # 清空 doc_id 注册表
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        reg.clear()
        logger.info("已清空 doc_id 注册表")

        # 重置 BM25 缓存
        from core.retrievers.bm25 import reset_bm25_retriever
        reset_bm25_retriever()
        logger.info("已重置 BM25 索引缓存")

        return True
    except Exception as e:
        logger.error(f"清空集合失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="清空 Qdrant 并重新导入文档")
    parser.add_argument(
        "dir",
        nargs="?",
        default=r"C:\Users\25677\Desktop\论文",
        help="论文目录路径",
    )
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        logger.error(f"目录不存在: {dir_path}")
        sys.exit(1)

    # 收集文件
    files = []
    for ext in [".pdf", ".PDF", ".md", ".MD", ".txt", ".TXT"]:
        for f in dir_path.rglob(f"*{ext}"):
            if not f.name.startswith("."):
                files.append(f)

    # 去重
    files = sorted(set(files))

    if not files:
        logger.warning(f"目录 {dir_path} 中未找到支持的文档 (.pdf/.md/.txt)")
        return

    logger.info("=" * 60)
    logger.info(f"⚠️  即将清空 Qdrant 集合，用新分块策略重新导入")
    logger.info(f"   分块策略: {settings.chunking.strategy}")
    logger.info(f"   token_max: {settings.chunking.token_max}")
    logger.info(f"   overlap: {settings.chunking.overlap_token}")
    logger.info(f"   目标目录: {dir_path}")
    logger.info(f"   文档数量: {len(files)}")
    for f in files:
        logger.info(f"     - {f.name}")
    logger.info("=" * 60)

    # 1) 清空
    if not clear_collection():
        logger.error("清空失败，终止")
        sys.exit(1)

    # 2) 导入
    from services.document_service import get_document_service
    svc = get_document_service()
    total_chunks = svc.add_documents(files)

    logger.info(f"🎉 重建完成！总共 {total_chunks} 个 chunks")
    sys.exit(0)


if __name__ == "__main__":
    main()