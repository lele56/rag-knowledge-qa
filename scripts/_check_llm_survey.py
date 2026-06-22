"""检查 LLM Survey 的 chunks 是否在 Qdrant 中"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")
from qdrant_client import QdrantClient, models
from config.settings import settings

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# 按 doc_id 过滤查 LLM Survey
result, _ = client.scroll(
    collection_name=settings.QDRANT_COLLECTION_NAME,
    scroll_filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="doc_49f4c7eb"))]
    ),
    limit=5,
    with_payload=True,
)

print(f"LLM Survey chunks 预览: {len(result)}")
for i, p in enumerate(result):
    payload = p.payload or {}
    text = (payload.get("page_content", "") or "")[:200].replace("\n", " ")
    print(f"\n  [{i}] doc_id={payload.get('doc_id','')} source={payload.get('source','')}")
    print(f"       text: {text}")
    # 检查 payload 中有哪些 key
    if i == 0:
        print(f"       payload keys: {list(payload.keys())}")