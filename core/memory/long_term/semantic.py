# core/memory/long_term/semantic.py
"""
语义记忆 (Semantic Memory)。

存储：从问答文本中抽取简单的"概念关键词"，在 Neo4j 中创建/更新
      (:Concept {name, summary, importance, last_accessed}) 节点，
      并用 [:RELATES_TO] 边表示概念之间的关联。
检索：1) 用轻量关键词匹配 + embedding 向量匹配找 top-K Concept
      2) 沿 [:RELATES_TO] 边拓展邻居（图相似度部分）
      3) 按 score_semantic() 打分排序
"""
from typing import List, Tuple, Dict, Any, Optional, Set
import time
import re

from .scoring import score_semantic
from .config import cfg
from utils.logger import logger


# ---------------------------------------------------------------------------
# 极简概念抽取：基于常见关键词列表 + 分词 + 停用词过滤
# 不引入 spaCy，避免新依赖。够用，快。
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "也", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "我", "他", "她", "它",
    "这", "那", "之", "而", "吗", "呢", "吧", "啊", "什么", "怎么", "如何",
    "什么", "哪个", "哪些", "怎么", "为什么", "如何",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "it", "this", "that", "be", "for", "with", "as",
}

# 你领域常见的"种子概念"（大小写不敏感的简单匹配）
_DOMAIN_KEYWORDS = [
    "向量检索", "bm25", "混合检索", "知识图谱", "语义记忆", "情景记忆",
    "工作记忆", "向量数据库", "qdrant", "neo4j", "embedding", "嵌入",
    "llm", "大模型", "检索增强", "rag", "重排序", "reranker", "prompt",
    "langchain", "文档", "知识库", "记忆", "上下文", "问答",
]


def _extract_concepts(text: str, max_concepts: int = 5) -> List[str]:
    """
    从文本中抽取候选概念名。
    规则：1) 命中 _DOMAIN_KEYWORDS 的；2) 连续的中文词(3~12字) 且非停用词。
    返回去重后的 top-N。
    """
    if not text:
        return []
    lower = text.lower()
    concepts: List[str] = []

    # 1) 命中种子词的
    for kw in _DOMAIN_KEYWORDS:
        if kw.lower() in lower:
            concepts.append(kw)

    # 2) 简单中文词组（3~12 个中文字符）
    for m in re.findall(r"[\u4e00-\u9fa5]{3,12}", text):
        if m not in _STOPWORDS and m not in concepts:
            concepts.append(m)

    # 3) 简单英文词 (3+)
    for m in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,20}", text):
        low = m.lower()
        if low not in _STOPWORDS and low not in [c.lower() for c in concepts]:
            concepts.append(m)

    return concepts[:max_concepts]


# ---------------------------------------------------------------------------
# Neo4j: 概念节点的读写与关系
# ---------------------------------------------------------------------------

def _cypher(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """跑一条 Cypher 查询，捕获异常。"""
    try:
        from core.infrastructure.graph_store import get_graph
        graph = get_graph()
        return graph.query(query, params=params or {})
    except Exception as e:
        logger.warning(f"[memory] Neo4j 查询失败: {e}")
        return []


def store_semantic(question: str, answer: str,
                    importance: Optional[float] = None) -> List[str]:
    """
    从问答文本中抽取概念，写入/更新 Neo4j :Concept 节点。
    返回被 touch 的概念名列表。
    """
    if not cfg.ENABLED:
        return []

    text = f"{question} {answer}"
    concepts = _extract_concepts(text)
    if not concepts:
        return []

    now_ts = float(time.time())
    imp = importance if importance is not None else cfg.IMPORTANCE_INIT
    summary = f"围绕「{question[:60]}...」的讨论"

    touched: List[str] = []
    for c in concepts:
        # MERGE: 不存在就创建，存在就更新重要性/最后访问时间/摘要
        params = {
            "name": c,
            "summary": summary,
            "importance": float(imp),
            "now": now_ts,
            "growth": float(cfg.IMPORTANCE_GROWTH),
        }
        _cypher("""
            MERGE (n:Concept {name: $name})
            ON CREATE SET n.summary = $summary,
                          n.importance = $importance,
                          n.last_accessed = $now,
                          n.created_at = $now
            ON MATCH  SET n.importance = CASE WHEN n.importance + $growth > 1.0 THEN 1.0 ELSE n.importance + $growth END,
                          n.last_accessed = $now,
                          n.summary = coalesce($summary, n.summary)
        """, params)
        touched.append(c)

    # 给同时出现的概念加 [:RELATES_TO] 边（强度基于共同出现次数）
    for i, a in enumerate(touched):
        for b in touched[i + 1:]:
            _cypher("""
                MATCH (x:Concept {name: $a}), (y:Concept {name: $b})
                MERGE (x)-[r:RELATES_TO]->(y)
                ON CREATE SET r.strength = 0.5, r.count = 1
                ON MATCH  SET r.count = coalesce(r.count, 0) + 1,
                              r.strength = CASE WHEN coalesce(r.strength, 0.5) + 0.1 > 1.0 THEN 1.0 ELSE coalesce(r.strength, 0.5) + 0.1 END
            """, {"a": a, "b": b})

    logger.info(f"[memory] 语义记忆 touch 概念: {touched}")
    return touched


def recall_semantic(query: str) -> List[Tuple[float, str]]:
    """
    从 Neo4j 检索与 query 相关的 Concept 节点。
    步骤：1) 从 query 抽概念，作为"直接命中"
          2) 直接命中节点沿 [:RELATES_TO] 拓展邻居
          3) 打分 & 排序
    返回 [(score, text), ...]
    """
    if not cfg.ENABLED:
        return []

    query_concepts = _extract_concepts(query)

    # 1) 直接命中：按名字匹配
    direct_hits: List[Dict[str, Any]] = []
    for name in query_concepts:
        rows = _cypher(
            "MATCH (n:Concept {name: $name}) "
            "RETURN n.name AS name, n.summary AS summary, "
            "       n.importance AS importance, n.last_accessed AS ts",
            {"name": name},
        )
        for r in rows:
            direct_hits.append(r)

    # 2) 邻居拓展
    neighbors_by_name: Dict[str, float] = {}  # name -> 图相似度
    for hit in direct_hits:
        name = hit.get("name")
        if not name:
            continue
        rows = _cypher(
            "MATCH (a:Concept {name: $name})-[r:RELATES_TO]-(b:Concept) "
            "RETURN b.name AS name, b.summary AS summary, "
            "       b.importance AS importance, b.last_accessed AS ts, "
            "       coalesce(r.strength, 0.5) AS strength "
            "ORDER BY strength DESC LIMIT $limit",
            {"name": name, "limit": cfg.SEMANTIC_NEIGHBOR * 3},
        )
        for r in rows:
            n = r.get("name")
            if n and n not in [h.get("name") for h in direct_hits]:
                neighbors_by_name[n] = max(
                    neighbors_by_name.get(n, 0.0),
                    float(r.get("strength", 0.5)),
                )

    # 把邻居节点补 full info
    neighbor_hits: List[Dict[str, Any]] = []
    for n in list(neighbors_by_name.keys())[:cfg.SEMANTIC_NEIGHBOR * 2]:
        rows = _cypher(
            "MATCH (n:Concept {name: $name}) "
            "RETURN n.name AS name, n.summary AS summary, "
            "       n.importance AS importance, n.last_accessed AS ts",
            {"name": n},
        )
        for r in rows:
            neighbor_hits.append({**r, "_graph_sim": neighbors_by_name[n]})

    # 3) 合并打分
    all_nodes: List[Tuple[float, str]] = []

    # 直接命中：向量相似度视为 0.8（名字匹配算强相关）
    for h in direct_hits:
        name = h.get("name", "")
        summary = h.get("summary", "")
        imp = float(h.get("importance", cfg.IMPORTANCE_INIT))
        score = score_semantic(vector_similarity=0.8,
                                graph_similarity=0.0,
                                importance=imp)
        all_nodes.append((score, f"[已知概念·{name}] {summary}"))

    # 邻居：向量相似度较低，但图相似度 > 0
    for h in neighbor_hits:
        name = h.get("name", "")
        summary = h.get("summary", "")
        imp = float(h.get("importance", cfg.IMPORTANCE_INIT))
        graph_sim = float(h.get("_graph_sim", 0.3))
        score = score_semantic(vector_similarity=0.2,
                                graph_similarity=graph_sim,
                                importance=imp)
        all_nodes.append((score, f"[相关概念·{name}] {summary}"))

    all_nodes.sort(key=lambda x: x[0], reverse=True)
    return all_nodes[:cfg.SEMANTIC_TOP_K + cfg.SEMANTIC_NEIGHBOR]


# ---------------------------------------------------------------------------
# 后台维护：importance 衰减 + 遗忘
# ---------------------------------------------------------------------------

def consolidate_semantic() -> int:
    """
    对超过 7 天未被访问的 Concept 做 importance 衰减，
    低于 FORGET_THRESHOLD 的 Concept 被删除（软遗忘）。
    不建议每次问答都跑，可以定期（比如每 20 次问答跑一次）。
    """
    if not cfg.ENABLED:
        return 0
    now_ts = float(time.time())
    # 7 天 = 604800 秒
    _cypher(
        "MATCH (n:Concept) "
        "WHERE $now - coalesce(n.last_accessed, $now) > 604800 "
        "SET n.importance = n.importance * 0.9",
        {"now": now_ts},
    )