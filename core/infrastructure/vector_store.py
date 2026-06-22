# core/infrastructure/vector_store.py
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from langchain_qdrant import QdrantVectorStore
from core.infrastructure.embeddings import get_embeddings
from config.settings import settings
from utils.logger import logger

_client = None
_store = None
# 记录 payload 索引是否已尝试创建（避免每次 get_vector_store 都调 Qdrant）
_payload_indexes_ensured = False

# upsert 批次大小：Qdrant 单次可承载上万 points，1000 完全没问题
_UPSERT_BATCH = 1000


def _to_float_list(vec) -> List[float]:
    """numpy.ndarray → list[float]。新版本 qdrant-client 的 PointStruct 要求 Python list。"""
    if hasattr(vec, "tolist"):
        return [float(v) for v in vec.tolist()]
    return [float(v) for v in vec]


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=120,  # HTTP 请求超时
            grpc_port=6334,
            prefer_grpc=False,
        )
    return _client


def _ensure_payload_indexes(client: QdrantClient, collection_name: str):
    """确保 source/doc_id 等关键过滤字段有 payload 索引（云版 Qdrant 要求 filter 字段必须建索引）。

    字段说明：
      - source (TEXT):     文件名，子串匹配，向后兼容
      - doc_id (KEYWORD):  结构化 doc_id（如 doc_2a7f9e），精确匹配，速度最快最准
      - file_type (KEYWORD): .pdf/.md/.txt 等，备用过滤
    """
    global _payload_indexes_ensured
    if _payload_indexes_ensured:
        return
    fields_to_index = [
        ("source", PayloadSchemaType.TEXT),
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("file_type", PayloadSchemaType.KEYWORD),
    ]
    for field_name, schema_type in fields_to_index:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info(f"✅ Qdrant payload 索引已创建: {field_name} ({schema_type})")
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg or "exists" in msg:
                pass
            else:
                logger.warning(f"创建 payload 索引 {field_name} 失败（非致命）: {e}")
    _payload_indexes_ensured = True


def get_vector_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        embeddings = get_embeddings()
        client = _get_client()
        col = settings.QDRANT_COLLECTION_NAME

        # 确保集合存在：先查，不存在才创建；网络超时不误判为"不存在"
        exists = False
        try:
            client.get_collection(col)
            exists = True
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "doesn't exist" in msg or "404" in msg:
                exists = False
            else:
                # 网络超时/其他错误 → 假设集合存在，让后续操作自行判断
                logger.warning(f"检查集合 {col} 失败（假设已存在）: {e}")
                exists = True

        if not exists:
            sample = embeddings.embed_query("test")
            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=len(sample), distance=Distance.COSINE),
            )
            logger.info(f"Created collection {col}")

        _ensure_payload_indexes(client, col)
        _store = QdrantVectorStore(
            client=client,
            collection_name=col,
            embedding=embeddings,
        )
    return _store


def add_documents_in_batches(store, docs) -> int:
    """【显存优化版】分批计算 embedding → 分批 upsert 到 Qdrant。

    避免一次性把所有文本送 GPU 导致 OOM：
      - embedding 分批（EMBED_BATCH 条/批），每批后清理 GPU 缓存
      - upsert 也分批（_UPSERT_BATCH 条/批），失败自动重试
    """
    import uuid
    import time as _time

    if not docs:
        return 0

    total = len(docs)
    t0 = _time.time()
    texts = [doc.page_content for doc in docs]

    EMBED_BATCH = 32  # GPU 嵌入批次大小（语义分块时需降低以避免 OOM）

    logger.info(f"🧮 计算 {total} 个 chunks 的 embedding（{EMBED_BATCH}条/批）...")

    try:
        from core.infrastructure.embeddings import get_embeddings
        embeddings = get_embeddings()
        vectors = []

        for i in range(0, total, EMBED_BATCH):
            batch_texts = texts[i:i + EMBED_BATCH]
            batch_vecs = embeddings.embed_documents(batch_texts)
            vectors.extend(batch_vecs)

            # 清理 GPU 碎片，避免显存累积
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            done = min(i + EMBED_BATCH, total)
            elapsed = _time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(f"  📊 embedding {done}/{total} ({rate:.0f} chunks/s)")

    except Exception as e:
        logger.error(f"embedding 计算失败: {e}，回退到慢速方法")
        store.add_documents(docs)
        return total

    t1 = _time.time()
    logger.info(
        f"✅ embedding 完成，耗时 {t1 - t0:.1f}s，"
        f"平均 {total/(t1-t0):.1f} chunks/s"
    )

    # 2) 构造 points（注意：vector 必须是 list[float]，不能是 numpy.ndarray）
    points = []
    for doc, vec in zip(docs, vectors):
        pid = str(uuid.uuid4())
        payload = dict(doc.metadata) if doc.metadata else {}
        payload["page_content"] = doc.page_content
        points.append({"id": pid, "vector": _to_float_list(vec), "payload": payload})

    # 3) 分批 upsert，失败自动重试（最多 3 次，指数退避）
    #    wait=False: 不等待磁盘 flush，Qdrant 立即返回，写入在后台完成
    client = _get_client()
    added = 0
    batch_total = (total + _UPSERT_BATCH - 1) // _UPSERT_BATCH
    for i in range(0, total, _UPSERT_BATCH):
        batch = points[i:i + _UPSERT_BATCH]
        b_idx = i // _UPSERT_BATCH + 1

        for retry in range(3):
            tb = _time.time()
            try:
                client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=batch,
                    wait=False,
                )
                added += len(batch)
                dt = _time.time() - tb
                logger.info(f"📥 upsert {b_idx}/{batch_total}: {added}/{total} (本批 {dt:.1f}s)")
                break
            except Exception as e:
                wait_s = 2 ** retry
                logger.warning(f"⚠️ upsert 第 {b_idx} 批失败 (重试 {retry+1}/3, {wait_s}s 后重试): {e}")
                _time.sleep(wait_s)
        else:
            logger.error(f"❌ upsert 第 {b_idx} 批彻底失败，已跳过 {len(batch)} 个 chunks")

    t2 = _time.time()
    logger.info(
        f"🎯 全部完成！写入 {added}/{total} chunks，"
        f"总耗时 {t2 - t0:.1f}s，平均 {added/(t2-t0):.1f} chunks/s"
    )

    # 打印当前 collection 总点数，方便排查"上传了但检索不到"的问题
    try:
        info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
        logger.info(f"📊 当前 collection 总点数: {info.points_count}")
    except Exception:
        pass

    return added

def get_all_documents() -> list:
    """把 Qdrant 里所有 chunks 拉出来（供 BM25 建关键词索引用）。
    返回 [{"content": str, "metadata": dict}] 列表。
    """
    from langchain_core.documents import Document
    client = _get_client()
    offset = None
    all_docs = []
    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                offset=offset,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning(f"从 Qdrant scroll 读取失败: {e}")
            break
        for r in records:
            payload = r.payload or {}
            content = payload.get("page_content", "")
            if not content:
                continue
            meta = {k: v for k, v in payload.items() if k != "page_content"}
            all_docs.append(Document(page_content=content, metadata=meta))
        offset = next_offset
        if not offset:
            break
    logger.info(f"从 Qdrant 读取 {len(all_docs)} 个 chunks 供 BM25 索引")
    return all_docs


def delete_by_doc_id(doc_id: str) -> int:
    """按 doc_id 删除 Qdrant 中该文档的所有 chunks。

    Returns:
        删除的 chunk 数量。失败返回 -1。
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    client = _get_client()
    try:
        # 先统计
        count_result = client.count(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            exact=True,
        )
        total = count_result.count
        if total == 0:
            return 0

        # 删除
        result = client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )
        deleted = result.status == "completed"
        if deleted:
            logger.info(f"🗑️  Qdrant 删除: {doc_id} → {total} chunks")
            return total
        else:
            logger.warning(f"Qdrant 删除 {doc_id} 状态异常: {result.status}")
            return 0
    except Exception as e:
        logger.error(f"Qdrant 删除 {doc_id} 失败: {e}")
        return -1


def get_doc_chunk_sample(doc_id: str, sample_size: int = 5) -> list:
    """获取指定 doc_id 的 chunk 内容样本（不加载向量）。

    Returns:
        [{"content": str, "source": str}, ...] 列表。
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    client = _get_client()
    try:
        hits, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            limit=sample_size,
            with_payload=True,
            with_vectors=False,
        )
        samples = []
        for pt in hits:
            payload = pt.payload or {}
            samples.append({
                "content": payload.get("page_content", ""),
                "source": payload.get("source", "?"),
            })
        return samples
    except Exception as e:
        logger.warning(f"获取 chunk 样本失败 ({doc_id}): {e}")
        return []


def get_qdrant_status() -> dict:
    """检查 Qdrant 状态。返回 {"ok": bool, "msg": str, "points": int}。
    如果连接失败，会重置缓存客户端并重试一次。
    """
    global _client

    def _try_check() -> dict:
        client = _get_client()
        try:
            info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
            points_count = info.points_count if info else 0
        except Exception:
            # collection 不存在 → Qdrant 本身可用，只是没有数据
            points_count = 0
        return {
            "ok": True,
            "points": points_count,
            "msg": f"✅ Qdrant 正常（{points_count} 条知识）",
        }

    try:
        return _try_check()
    except Exception:
        _client = None
        try:
            return _try_check()
        except Exception as e:
            return {"ok": False, "points": 0, "msg": f"❌ Qdrant 不可用: {type(e).__name__}"}