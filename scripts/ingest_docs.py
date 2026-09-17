# -*- coding: utf-8 -*-
"""批量文档入库脚本

用法:
    python scripts/ingest_docs.py                          # 默认 data/ 目录递归扫描
    python scripts/ingest_docs.py --dir data/test_docs     # 指定目录
    python scripts/ingest_docs.py --dir data/test_docs --rebuild  # 清空后重建
"""

import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*FontBBox.*")
logging.getLogger("pypdf").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings
from services.document_service import get_document_service
from utils.logger import logger

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".markdown"}


def clear_collection():
    from core.infrastructure.vector_store import _get_client

    client = _get_client()
    col_name = settings.QDRANT_COLLECTION_NAME
    logger.warning(f"清空集合 {col_name} ...")
    client.delete_collection(col_name)
    logger.info(f"已删除集合 {col_name}，下次写入会自动重建")


def main(data_dir: str, rebuild: bool = False):
    target = Path(data_dir).resolve()
    if not target.exists():
        logger.error(f"目录不存在: {target}")
        return

    if rebuild:
        clear_collection()

    all_files = sorted(target.rglob("*"))
    files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        logger.warning("没有找到支持的文档 (pdf/md/txt)")
        return

    logger.info(f"找到 {len(files)} 个文件待入库:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  - {f.relative_to(target)} ({size_kb:.0f} KB)")

    svc = get_document_service()
    count = svc.add_documents(files)
    logger.info(f"入库完成: {count} chunks")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量文档入库")
    parser.add_argument("--dir", type=str, default=None, help="文档目录 (默认: data/)")
    parser.add_argument("--rebuild", action="store_true", help="清空 Qdrant 集合后重新入库")
    args = parser.parse_args()

    target_dir = args.dir or str(Path(__file__).resolve().parent.parent / "data")
    main(target_dir, rebuild=args.rebuild)