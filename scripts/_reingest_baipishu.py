"""临时脚本 - 替换文档：删除 AI 白皮书 → 入库新文档（recursive 分块）"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")
from qdrant_client import QdrantClient, models
from config.settings import settings
from core.infrastructure.vector_store import get_vector_store, add_documents_in_batches
from core.doc.doc_id_registry import get_doc_id_registry
from utils.logger import logger

# ── 配置 ──
OLD_DOC_ID = "doc_86063b2c"  # AI 白皮书
NEW_FILE = "data/test_docs/A Survey of Large Language Models.pdf"
NEW_DOC_ID = "doc_49f4c7eb"  # 上次入库生成的 doc_id

# ── 1. 删除旧 AI 白皮书数据 ──
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
client.delete(
    collection_name=settings.QDRANT_COLLECTION_NAME,
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=OLD_DOC_ID))])
    )
)
print(f"已删除旧数据: {OLD_DOC_ID}")
get_doc_id_registry().unregister(OLD_DOC_ID)
print(f"已从注册表移除: {OLD_DOC_ID}")

# ── 2. 删除上次 semantic 分块产生的碎片数据 ──
client.delete(
    collection_name=settings.QDRANT_COLLECTION_NAME,
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=NEW_DOC_ID))])
    )
)
print(f"已清理碎片数据: {NEW_DOC_ID}")
get_doc_id_registry().unregister(NEW_DOC_ID)
print(f"已从注册表移除: {NEW_DOC_ID}")

# ── 3. 强制用 recursive 策略重新解析 ──
path = Path(NEW_FILE)
if not path.exists():
    print(f"文件不存在: {path}")
    sys.exit(1)

settings.chunking.strategy = "recursive"
print(f"分块策略: {settings.chunking.strategy}")

from core.doc.document_loader import load_and_split_documents
chunks = load_and_split_documents([path])
print(f"解析完成: {len(chunks)} chunks")

# ── 4. 入库 ──
store = get_vector_store()
add_documents_in_batches(store, chunks)
print(f"入库完成: {len(chunks)} chunks")

# ── 5. 预览 ──
print("\n=== 内容预览 ===")
for i, c in enumerate(chunks[:5]):
    text = c.page_content.replace('\n', ' ')[:200]
    print(f"  [{i}] {text}")