# chains/source_detection.py
"""
【来源检测模块：从用户问题中识别目标文档 + 会话级多文档聚焦管理】

核心职责：
  1. 从用户问题中检测"目标文档来源"（文件名 / 论文标题关键词 / 指代）
  2. 管理会话级多文档聚焦状态（set[str]）：用户说"这篇文章"时用它
  3. 提供一组辅助函数供 RAG 链内部调用

这个模块是 RAG 链的"聚焦大脑"——它决定检索时只看哪些文档，
从而避免"问 A 文档却召回 B 文档内容"的污染问题。
"""
from typing import Optional, List, Set, Any
from collections import Counter
import re

from utils.logger import logger


# ============================================================
# 检测关键词表
# ============================================================

# 指代类关键词：用户没说文件名，但表示"当前正在说的那篇"
_REFERENCE_KWS = (
    "这篇文章", "这个文档", "这篇论文", "这篇", "该文档",
    "该文章", "该论文", "这个文件", "这个pdf", "这个PDF",
    "本文", "此文", "此文档", "这份文件", "这份文档", "这份论文",
    "上传的这篇", "刚上传的", "刚才上传的", "你刚才读到的", "最近上传的",
)

# 明确的论文/学术文件名特征词（用于启发式检测）
_TITLE_KWS = (
    # 中文论文常见词
    "量化", "模型", "研究", "基于", "优化", "策略", "论文", "综述",
    "算法", "系统", "设计", "实现", "分析", "应用",
    # ML/DL 模型名
    "LightGBM", "SVM", "LSTM", "XGBoost", "Transformer", "LLaMA",
    "BERT", "GPT", "PCA", "CNN", "RNN", "GAN",
    # 典型 arxiv/学术文件格式关键词
    "arxiv", "arXiv", "v1", "v2", "MOR",
)


# ============================================================
# 启发式检测：这串文本是不是"文件名/论文标题"
# ============================================================

def looks_like_filename(candidate: str) -> bool:
    """启发式判断一个字符串是否像文件名/论文标题。"""
    if not candidate or len(candidate) < 5:
        return False
    c = candidate
    if "_" in c or "-" in c or c.lower().endswith(".pdf"):
        return True
    if re.search(r"\d{3,}", c) or re.search(r"[vV]\d", c):
        return True
    return any(kw in c for kw in _TITLE_KWS)


# ============================================================
# 核心：从用户问题中检测目标文档来源
# ============================================================

def detect_source_hint(query: str) -> Optional[str]:
    """
    从用户问题中检测目标文档来源（文件名/论文标题关键词）。

    支持的格式示例（全部都能识别）：
      - 完整 PDF 名: "MOR2507.10524v1.pdf 摘要"
      - 版本号格式: "2507.10524v1 讲的什么"
      - 中文长标题: "基于贝叶斯优化的GA-LightGBM模型股指期货量化策略研究"
      - 纯文件名: "Happy-LLM-0727.pdf"
      - 指代: "这篇文章讲什么" / "上传的这篇文档摘要" / "刚上传的讲什么"

    返回：
      - 用于 Qdrant payload.source 子串匹配的关键词字符串
      - "__USE_FOCUS__"（特殊标记，代表"使用会话级聚焦文档"）
      - None（未检测到明确来源）

    关键设计：指代检测优先执行，避免"这篇论文讲什么"被误判为文件名。
    """
    q = query.strip()
    if not q:
        return None

    # ====== 策略 0：先检查是否有疑问词 ======
    is_pure_question = bool(re.search(r"(如何|什么|怎么|为什么|哪|哪里|谁|？|\?)", q))

    # ====== 策略 1：指代检测（优先级最高！） ======
    if any(kw in q for kw in _REFERENCE_KWS):
        return "__USE_FOCUS__"

    # ====== 策略 2：带 .pdf/.PDF 后缀 → 明确文件名 ======
    pdf_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_\-\.]+)\.(?:pdf|PDF)", q)
    if pdf_match:
        return pdf_match.group(1)

    # ====== 策略 3：典型学术文件格式 ======
    arxiv_match = re.search(r"([A-Za-z]{0,5}\d{3,4}\.\d{4,6}[vV]?\d*)", q)
    if arxiv_match:
        return arxiv_match.group(1)

    # ====== 策略 4：含 "_" 或 "-" 的长连续字符（像文件名） ======
    filename_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_\-]{8,})", q)
    if filename_match:
        candidate = filename_match.group(1)
        if ("_" in candidate or "-" in candidate) and looks_like_filename(candidate):
            return candidate

    # ====== 策略 5："基于 xxx 的 yyy" 中文论文标题模式（非疑问句才启用） ======
    if not is_pure_question:
        based_match = re.search(
            r"基于[\u4e00-\u9fa5A-Za-z0-9_\-]{3,}的[\u4e00-\u9fa5A-Za-z0-9_\-]{3,}", q)
        if based_match:
            candidate = based_match.group(0)
            if not re.search(r"(如何|什么|怎么|为什么|哪|哪里|谁)", candidate):
                return candidate

    return None


# ============================================================
# 归一化与过滤解析
# ============================================================

def normalize_focus_set(focus) -> Optional[Set[str]]:
    """把各种输入形态（str/List[str]/Set[str]/None）归一化成 set[str]；空输入返回 None。"""
    if not focus:
        return None
    if isinstance(focus, (set, frozenset)):
        s = {v for v in focus if isinstance(v, str) and v.strip()}
        return s if s else None
    if isinstance(focus, (list, tuple)):
        s = {v for v in focus if isinstance(v, str) and v.strip()}
        return s if s else None
    if isinstance(focus, str):
        s = focus.strip()
        return {s} if s else None
    return None


def resolve_source_filter(hint: Optional[str], focus) -> Optional[Set[str]]:
    """把 detect_source_hint 的结果 + 会话聚焦状态 解析为最终的过滤关键词集合。

    规则（set[str] 语义：任一命中即通过）：
      - hint 为 None → 用当前聚焦（可能是 set[str] 或 None）
      - hint 为 "__USE_FOCUS__" → 用当前聚焦
      - hint 为明确的文件名/标题 → 返回单元素集合

    🔧 P0: 通过 DocIdRegistry 自动补充 doc_id（如 doc_a1b2c3d4），
        这样 Qdrant 侧可以用 MatchValue("doc_id", "doc_a1b2c3d4") 精确过滤，
        避免"在整个 collection 里搜向量，但目标文档的 chunk 进不了 top-N"的问题。
    """
    focus_set = normalize_focus_set(focus)
    if hint is None:
        base = focus_set
    elif hint == "__USE_FOCUS__":
        base = focus_set
    else:
        base = {hint}

    if not base:
        return None

    # 🔧 用 registry 把关键词 → doc_id
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        enriched = reg.enrich_filter(base)
        if enriched:
            return enriched
    except Exception as e:
        # registry 出错不应该阻断检索流程——退化为仅关键词匹配
        logger.debug(f"source_detection: doc_id registry 查询失败: {e}")

    return base


# ============================================================
# 便捷：直接获取 doc_id 过滤（给 HybridRetriever 等场景用）
# ============================================================

def resolve_doc_ids_only(source_filter) -> Optional[Set[str]]:
    """只返回 doc_id 集合（不含文件名关键词），用于 Qdrant payload 过滤。"""
    s = normalize_focus_set(source_filter)
    if not s:
        return None
    try:
        from core.doc.doc_id_registry import get_doc_id_registry
        reg = get_doc_id_registry()
        doc_ids = reg.lookup_doc_ids(s)
        return doc_ids if doc_ids else None
    except Exception:
        return None


# ============================================================
# 兜底检查：document 是否匹配目标来源
# ============================================================

def doc_matches_source(doc, source_filter) -> bool:
    """判断一个检索到的 document 是否匹配目标来源关键词集合（用于兜底判断）。

    支持三层匹配（任一命中即通过）：
      1. doc_id 精确匹配（最快最准）
      2. source 子串匹配（向后兼容没有 doc_id 的旧 chunks）
      3. 内容子串匹配（兜底，防止 payload 字段完全缺失）
    """
    filter_set = normalize_focus_set(source_filter)
    if not filter_set:
        return True
    source = ""
    doc_id = ""
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        source = str(doc.metadata.get("source", "") or "")
        doc_id = str(doc.metadata.get("doc_id", "") or "")
    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    source_low = source.lower()
    doc_id_low = doc_id.lower()
    content_low = content.lower()
    for hint_str in filter_set:
        hint_low = hint_str.lower()
        if doc_id_low and doc_id_low == hint_low:
            return True
        if source_low and hint_low in source_low:
            return True
        if hint_low in content_low:
            return True
    return False


# ============================================================
# 聚焦选择：从一组来源中挑选 top 3 高频作为会话聚焦
# ============================================================

def pick_focus_source(sources: List[str]) -> Optional[Set[str]]:
    """从一组 sources 中挑选前几名作为"会话聚焦集合"。

    策略：按频率排序，取 top-3 高频来源（去重后）。
    多文档场景下，用户可能同时关注多个文件，这比"只取最高频一个"更合理。
    """
    if not sources:
        return None
    valid = [s for s in sources if s and s.lower() not in ("unknown", "未知")]
    if not valid:
        return None
    counter = Counter(valid)
    sorted_items = counter.most_common(3)
    result = {s for s, _ in sorted_items}
    return result if result else None