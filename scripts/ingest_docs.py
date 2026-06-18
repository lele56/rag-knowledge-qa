import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from services.document_service import get_document_service
from utils.logger import logger

def main():
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        logger.error("data directory not found")
        return
    files = list(data_dir.glob("*"))
    if not files:
        logger.warning("No files found")
        return
    svc = get_document_service()
    count = svc.add_documents(files)
    logger.info(f"Ingested {count} chunks")

if __name__ == "__main__":
    main()