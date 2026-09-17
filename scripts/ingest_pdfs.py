# -*- coding: utf-8 -*-
"""批量 PDF 入库脚本

用法:
    python scripts/ingest_pdfs.py                    # 默认 data/test_docs/
    python scripts/ingest_pdfs.py --dir data/mydocs  # 指定目录
"""

from pathlib import Path
import sys
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import logger
from services.document_service import DocumentService


async def main(pdf_dir: Path):
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"目录 {pdf_dir} 中没有 PDF")
        return

    logger.info(f"找到 {len(pdfs)} 篇 PDF: {[p.name for p in pdfs]}")

    svc = DocumentService()
    ok_paths = [p for p in pdfs if p.exists()]
    if not ok_paths:
        logger.error("没有有效文件")
        return

    try:
        added = svc.add_documents(ok_paths)
        logger.info(f"入库完成: {added} chunks")
    except Exception as e:
        logger.error(f"入库失败: {e}")

    for p in pdfs:
        logger.info(f"  📄 {p.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量 PDF 入库")
    parser.add_argument("--dir", type=str, default=str(Path("data/test_docs")), help="PDF 目录")
    args = parser.parse_args()

    asyncio.run(main(Path(args.dir)))