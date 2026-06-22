# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

os.environ["PYTHONIOENCODING"] = "utf-8"

from config.settings import settings
from core.infrastructure.vector_store import _get_client
from core.doc.doc_id_registry import get_doc_id_registry

print("=" * 60)
print("Qdrant Knowledge Base Status")
print("=" * 60)

client = _get_client()
col_name = settings.QDRANT_COLLECTION_NAME

try:
    info = client.get_collection(col_name)
    total = info.points_count
    print(f"\nCollection: {col_name}")
    print(f"Total chunks: {total}")
except Exception as e:
    print(f"\nFailed to get collection info: {e}")
    sys.exit(1)

reg = get_doc_id_registry()
doc_ids = reg.get_all_doc_ids()
print(f"\nRegistered documents: {len(doc_ids)}")
print("-" * 40)

from qdrant_client.http.models import Filter, FieldCondition, MatchValue
for doc_id in sorted(doc_ids):
    src_name = reg.get_source_for_doc_id(doc_id) or "?"
    try:
        count_result = client.count(
            collection_name=col_name,
            count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            exact=True,
        )
        chunk_count = count_result.count
    except Exception:
        chunk_count = "?"
    print(f"  {doc_id}  ->  {src_name}  ({chunk_count} chunks)")

print("\n" + "=" * 60)
print("Done")