# core/doc/doc_id_registry.py
"""
【文档 ID 注册表：文件名 → doc_id 的双向映射】

核心职责：
  1. 维持一个 source_keyword → doc_id 的映射表（用于检索时的 payload filter）
  2. 维持 doc_id → known_sources 的反向映射（用于调试/展示）
  3. 持久化到 JSON 文件（重启后恢复）

为什么需要这个？
  - 用户提问里的是文件名/论文标题关键词（如 "Happy-LLM-0727"、"基于贝叶斯优化的GA-LightGBM..."）
  - Qdrant 的 MatchValue 需要精确的 doc_id（如 "doc_a1b2c3d4"）
  - 关键词 → doc_id 的映射在检索时能把用户提到的"文件名"转换为结构化的 doc_id
  - 有了 doc_id，Qdrant 就能在向量搜索前先把范围限制在目标文档内，避免无关文档的 chunks 抢占 top-N

使用方式：
    registry = get_doc_id_registry()
    registry.register("doc_a1b2c3d4", "Happy-LLM-0727.pdf")  # 文档上传时调用
    doc_ids = registry.lookup_doc_ids({"Happy-LLM-0727"})     # 检索时调用
    enriched = registry.enrich_filter({"Happy-LLM-0727"})     # 便捷：直接 enrich filter set
"""
from typing import Dict, Optional, Set, List
from pathlib import Path
import json, hashlib, time

from config.settings import settings
from utils.logger import logger


# ============================================================
# doc_id 生成（与 document_loader.py 的 make_doc_id 保持一致）
# ============================================================

def make_doc_id_from_stem(stem: str) -> str:
    """基于文件名 stem 生成稳定的 doc_id。与 document_loader.make_doc_id 语义一致。"""
    s = str(stem or "").strip().lower()
    if not s:
        s = f"unknown_{int(time.time())}"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"doc_{h}"


# ============================================================
# 注册表
# ============================================================

class DocIdRegistry:
    """文件名 → doc_id 的双向映射，持久化到 JSON 文件。"""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else Path(settings.LOCAL_DATA_DIR) / "doc_id_registry.json"
        # key -> doc_id
        self._keyword_to_doc_id: Dict[str, str] = {}
        # doc_id -> set of source names
        self._doc_id_to_sources: Dict[str, Set[str]] = {}
        self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._keyword_to_doc_id = {k: v for k, v in data.get("kw2did", {}).items()}
            self._doc_id_to_sources = {k: set(v) for k, v in data.get("did2srcs", {}).items()}
            n = len(self._doc_id_to_sources)
            if n > 0:
                logger.info(f"✅ doc_id registry 已加载: {n} 个文档 (from {self._path.name})")
        except Exception as e:
            logger.warning(f"doc_id registry 加载失败: {e}，从空开始")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "kw2did": self._keyword_to_doc_id,
                "did2srcs": {k: list(v) for k, v in self._doc_id_to_sources.items()},
                "updated_at": int(time.time()),
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"doc_id registry 保存失败: {e}")

    # ---------- 注册（文档上传时调用） ----------

    def register(self, doc_id: str, source: str, extra_keywords: Optional[List[str]] = None) -> None:
        """注册一个文档：把其文件名、stem、doc_id 都登记进来。"""
        if not doc_id or not source:
            return

        # 记录 source（完整文件名）
        self._doc_id_to_sources.setdefault(doc_id, set()).add(source)

        # 建立关键词 → doc_id 的映射（多个变体，提升匹配率）
        p = Path(source)
        stem = p.stem.lower()
        name = p.name.lower()
        name_no_ext = stem
        # 基础变体
        keywords = [stem, name, name_no_ext, source, source.lower()]
        # stem 的短片段（去常见分隔符后的核心词）
        cleaned = stem.replace("_", " ").replace("-", " ").replace(".", " ").strip()
        if cleaned and cleaned != stem:
            keywords.append(cleaned)
            keywords.append(cleaned.replace(" ", ""))
        # 额外关键词（调用方传入）
        if extra_keywords:
            for kw in extra_keywords:
                if kw:
                    keywords.append(kw)
                    keywords.append(kw.lower())

        # 去空白
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                self._keyword_to_doc_id[kw.strip()] = doc_id
                # 同时注册无空格版本（便于子串匹配）
                no_space = kw.strip().replace(" ", "")
                if no_space != kw.strip():
                    self._keyword_to_doc_id[no_space] = doc_id

        logger.info(f"📝 注册 doc_id: {doc_id} ← {source} (共 {len(keywords)} 个变体)")
        self._save()

    # ---------- 查找（检索时调用） ----------

    def lookup_doc_ids(self, source_filter: Optional[Set[str]]) -> Set[str]:
        """给定一组来源关键词，返回匹配的 doc_id 集合。"""
        if not source_filter:
            return set()
        found: Set[str] = set()
        for keyword in source_filter:
            if not keyword or not isinstance(keyword, str):
                continue
            kw = keyword.strip()
            kw_low = kw.lower()
            # 精确匹配
            if kw in self._keyword_to_doc_id:
                found.add(self._keyword_to_doc_id[kw])
                continue
            if kw_low in self._keyword_to_doc_id:
                found.add(self._keyword_to_doc_id[kw_low])
                continue
            # doc_id 本身就是 filter 关键词（调用方直接传 doc_xxx）
            if kw.startswith("doc_") and kw in self._doc_id_to_sources:
                found.add(kw)
                continue
            # 子串匹配（宽一点，确保 "基于贝叶斯优化..." 能命中 "基于贝叶斯优化的GA-LightGBM..."）
            for reg_kw, did in self._keyword_to_doc_id.items():
                if kw_low in reg_kw or reg_kw in kw_low:
                    found.add(did)
                    break
        return found

    def enrich_filter(self, source_filter: Optional[Set[str]]) -> Optional[Set[str]]:
        """返回干净的 filter（仅含人类可读的文件名/关键词，不含内部 doc_id）。"""
        if not source_filter:
            return None
        cleaned = set()
        for k in source_filter:
            if isinstance(k, str) and k.strip():
                cleaned.add(k.strip())
        if not cleaned:
            return None
        human_names = {k for k in cleaned if not k.startswith("doc_")}
        if not human_names:
            human_names = cleaned
        doc_ids = self.lookup_doc_ids(human_names)
        if doc_ids:
            logger.info(f"🎯 filter: {human_names} → {len(doc_ids)} doc_id(s) 命中: {doc_ids}")
        return human_names

    # ---------- 调试/展示 ----------

    def get_all_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_sources.keys())

    def get_all_sources(self) -> List[str]:
        """返回所有已注册文档的原始文件名列表"""
        sources = []
        for srcs in self._doc_id_to_sources.values():
            for s in srcs:
                sources.append(s)
        return sorted(set(sources))

    def get_source_for_doc_id(self, doc_id: str) -> Optional[str]:
        srcs = self._doc_id_to_sources.get(doc_id)
        if srcs:
            return next(iter(srcs))
        return None

    def unregister(self, doc_id: str) -> None:
        """注销一个文档：从 kw2did 和 did2srcs 中移除，并持久化。"""
        if not doc_id or doc_id not in self._doc_id_to_sources:
            return
        # 移除 kw → doc_id 映射
        to_remove = [kw for kw, did in self._keyword_to_doc_id.items() if did == doc_id]
        for kw in to_remove:
            del self._keyword_to_doc_id[kw]
        # 移除 doc_id → sources 映射
        srcs = self._doc_id_to_sources.pop(doc_id, set())
        logger.info(f"🗑️  注销 doc_id: {doc_id} ← {srcs}")
        self._save()

    def clear(self) -> None:
        """清空所有注册表记录（用于重建 Qdrant 集合时调用）"""
        self._keyword_to_doc_id.clear()
        self._doc_id_to_sources.clear()
        self._save()
        logger.info("🗑️  doc_id 注册表已清空")

    def dump(self) -> str:
        lines = [f"=== DocId Registry ({len(self._doc_id_to_sources)} docs) ==="]
        for did, srcs in self._doc_id_to_sources.items():
            line = f"  {did} ← {', '.join(sorted(srcs))}"
            if len(line) > 100:
                line = line[:97] + "..."
            lines.append(line)
        return "\n".join(lines)


# ============================================================
# 单例工厂
# ============================================================

_registry_instance = None


def get_doc_id_registry() -> DocIdRegistry:
    """返回 DocIdRegistry 单例（文档上传时、检索时共享同一个实例）。"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DocIdRegistry()
    return _registry_instance


# 兼容旧代码
get_doc_id_registry_for_upload = get_doc_id_registry