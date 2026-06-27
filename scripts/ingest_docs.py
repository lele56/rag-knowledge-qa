import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*FontBBox.*")
logging.getLogger("pypdf").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.document_service import get_document_service
from utils.logger import logger

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".markdown"}

def main():
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        logger.error("data directory not found")
        return

    all_files = list(data_dir.rglob("*"))
    files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        logger.warning("No supported files found (pdf/md/txt)")
        return

    logger.info(f"找到 {len(files)} 个文件待入库:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  - {f.relative_to(data_dir)} ({size_kb:.0f} KB)")

    svc = get_document_service()
    count = svc.add_documents(files)
    logger.info(f"Ingested {count} chunks")

if __name__ == "__main__":
    main()