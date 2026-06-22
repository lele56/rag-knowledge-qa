# core/retrievers/filtering.py
"""
【检索过滤模块：将各种输入归一化，并构造 Qdrant Filter】

职责：
  1. 把 str / list / set / None 归一化成 set[str]
  2. 区分 doc_id（精确匹配）和 keywords（source 子串匹配）
  3. 构造 qdrant_client.Filter 对象（只做 doc_id 精确匹配，keywords 留到 Python 侧）
  4. Python 侧兜底过滤：source/doc_id 子串匹配
"""
from typing import Optional, Any, Tuple, Set, List
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny


# ============================================================
# 输入归一化
# ============================================================

def normalize_filter(source_filter) -> Optional[Set[str]]:
    """把各种输入形态（str / list[str] / set[str] / None）归一成 set[str]。

    空输入 / 空集合 / 全空白字符串 → 返回 None（无过滤）。
    """
    if source_filter is None:
        return None
    if isinstance(source_filter, (set, frozenset)):
        s = {v for v in source_filter if isinstance(v, str) and v.strip()}
        return s if s else None
    if isinstance(source_filter, (list, tuple)):
        s = {v for v in source_filter if isinstance(v, str) and v.strip()}
        return s if s else None
    if isinstance(source_filter, str):
        s = source_filter.strip()
        return {s} if s else None
    return None


def split_doc_ids_and_keywords(filters: Set[str]) -> Tuple[Set[str], Set[str]]:
    """把 filter 关键词分成两组：doc_ids（精确匹配用）+ keywords（source 子串匹配用）。"""
    doc_ids = set()
    keywords = set()
    for f in filters:
        if f.startswith("doc_") and len(f) < 30:
            doc_ids.add(f)
        else:
            keywords.add(f)
    return doc_ids, keywords


# ============================================================
# Qdrant 过滤构造
# ============================================================

def build_qdrant_filter(source_filter) -> Optional[Filter]:
    """构造 qdrant_client.Filter 对象。

    规则：
      - 通过 registry 查找 keywords → doc_id，Qdrant 侧精确过滤（最快最准）
      - 无对应 doc_id → 退回 Python 侧兜底
      - 无过滤条件 → None
    """
    filters = normalize_filter(source_filter)
    if not filters:
        return None

    keywords = {f for f in filters if not f.startswith("doc_")}
    if not keywords:
        return None

    # 通过 registry 查找 keywords → doc_id
    doc_ids = set()
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        doc_ids = reg.lookup_doc_ids(keywords)
    except Exception:
        pass

    if not doc_ids:
        # 没有 doc_id匹配 → 退回 Python 侧兜底
        return None

    if len(doc_ids) == 1:
        return Filter(must=[FieldCondition(
            key="doc_id",
            match=MatchValue(value=next(iter(doc_ids))),
        )])
    return Filter(must=[FieldCondition(
        key="doc_id",
        match=MatchAny(any=list(doc_ids)),
    )])


# ============================================================
# Python 侧兜底过滤
# ============================================================

def doc_matches_source_filter(doc: Document, filter_set: Set[str]) -> bool:
    """判断一个 Document 是否匹配目标来源关键词集合。

    匹配逻辑（任一命中即通过）：
      - doc_id 精确匹配（最快最准）
      - source 子串匹配（向后兼容没有 doc_id 的旧数据）
      - 关键词是文件名 stem 时，匹配完整文件名
    """
    if not filter_set:
        return True
    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    doc_id = str(meta.get("doc_id", "") or "")
    source = str(meta.get("source", "") or "")
    doc_id_low = doc_id.lower()
    source_low = source.lower()

    for f in filter_set:
        f_low = f.lower()
        if doc_id_low and f_low.startswith("doc_") and f_low == doc_id_low:
            return True
        if source_low and f_low in source_low:
            return True
        if source_low and any(kw in source_low for kw in filter_set if not kw.startswith("doc_")):
            return True

    return False