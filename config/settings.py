# config/settings.py
"""配置管理 — 基于 pydantic-settings 的分层配置。

特性:
  - 自动从 .env / 环境变量加载
  - 启动时校验必填项和类型
  - 分层结构: Qdrant / Neo4j / LLM / Retrieval / Memory / Chunking
  - 兼容旧代码的 settings.OPENAI_API_KEY 等大写下划线访问
"""

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Tuple
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ================================================================
# 分组配置
# ================================================================

class _BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")


class QdrantSettings(_BaseConfig):
    url: str = Field(alias="QDRANT_URL", default="")
    api_key: str = Field(alias="QDRANT_API_KEY", default="")
    collection_name: str = Field(default="knowledge_base")

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.api_key)


class Neo4jSettings(_BaseConfig):
    uri: str = Field(alias="NEO4J_URI", default="")
    username: str = Field(alias="NEO4J_USERNAME", default="")
    password: str = Field(alias="NEO4J_PASSWORD", default="")
    database: str = Field(alias="NEO4J_DATABASE", default="neo4j")

    @property
    def is_configured(self) -> bool:
        return bool(self.uri and self.username and self.password)


class LLMSettings(_BaseConfig):
    api_key: str = Field(alias="OPENAI_API_KEY", default="")
    base_url: str = Field(alias="OPENAI_BASE_URL", default="https://api.openai.com/v1")
    model: str = Field(alias="LLM_MODEL", default="gpt-4o")
    request_timeout: int = Field(default=120, ge=10)
    max_retries: int = Field(default=2, ge=0, le=5)


class EmbeddingSettings(_BaseConfig):
    model_path: str = Field(alias="EMBEDDING_MODEL_PATH", default="./models/bge-small-zh")
    use_cuda: bool = Field(alias="USE_CUDA", default=True)


class RerankerSettings(_BaseConfig):
    model_path: str = Field(alias="RERANKER_MODEL_PATH", default="BAAI/bge-reranker-v2-m3")


class RetrievalSettings(_BaseConfig):
    strategy: str = Field(alias="RETRIEVAL_STRATEGY", default="simple")
    k: int = Field(alias="RETRIEVAL_K", default=8, ge=1, le=50)
    rerank_top_k: int = Field(alias="RERANK_TOP_K", default=3, ge=1, le=20)
    multi_query_count: int = Field(alias="MULTI_QUERY_COUNT", default=3, ge=1, le=10)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"simple", "multi_query", "hyde"}
        if v not in allowed:
            raise ValueError(f"strategy 必须是 {allowed} 之一，当前: {v}")
        return v


class MemorySettings(_BaseConfig):
    window_size: int = Field(alias="MEMORY_WINDOW_SIZE", default=10, ge=1, le=50)


class LongTermMemorySettings(_BaseConfig):
    enabled: bool = Field(alias="LS_ENABLED", default=True)
    episodic_top_k: int = Field(alias="LS_EPISODIC_TOP_K", default=3, ge=1, le=20)
    episodic_collection: str = Field(alias="LS_EPISODIC_COLLECTION", default="episodic_memory")
    episodic_max_store: int = Field(alias="LS_EPISODIC_MAX_STORE", default=200, ge=10, le=10000)
    semantic_top_k: int = Field(alias="LS_SEMANTIC_TOP_K", default=3, ge=1, le=20)
    semantic_neighbor: int = Field(alias="LS_SEMANTIC_NEIGHBOR", default=1, ge=0, le=5)
    importance_init: float = Field(alias="LS_IMPORTANCE_INIT", default=0.5, ge=0.0, le=1.0)
    importance_growth: float = Field(alias="LS_IMPORTANCE_GROWTH", default=0.1, ge=0.0, le=1.0)
    forget_threshold: float = Field(alias="LS_FORGET_THRESHOLD", default=0.05, ge=0.0, le=0.5)
    recall_total: int = Field(alias="LS_RECALL_TOTAL", default=5, ge=1, le=20)
    working_top_k: int = Field(alias="LS_WORKING_TOP_K", default=2, ge=1, le=10)


class ChunkingSettings(_BaseConfig):
    size: int = Field(alias="CHUNK_SIZE", default=1000, ge=100, le=10000)
    overlap: int = Field(alias="CHUNK_OVERLAP", default=200, ge=0, le=1000)
    token_target: int = Field(alias="CHUNK_TOKEN_TARGET", default=500, ge=50, le=2000)
    token_max: int = Field(alias="CHUNK_TOKEN_MAX", default=800, ge=50, le=4000)
    overlap_token: int = Field(alias="CHUNK_OVERLAP_TOKEN", default=80, ge=0, le=500)
    strategy: str = Field(alias="CHUNK_STRATEGY", default="recursive")
    quality_min_score: float = Field(default=0.2, ge=0.0, le=1.0)
    quality_min_tokens: int = Field(default=50, ge=0, le=500)
    quality_min_heading_tokens: int = Field(default=30, ge=0, le=200)
    quality_min_body_len: int = Field(default=15, ge=0, le=200)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"recursive", "semantic"}
        if v not in allowed:
            raise ValueError(f"chunk strategy 必须是 {allowed} 之一，当前: {v}")
        return v


class CacheSettings(_BaseConfig):
    enabled: bool = Field(alias="CACHE_ENABLED", default=True)
    backend: str = Field(alias="CACHE_BACKEND", default="memory")
    ttl_seconds: int = Field(alias="CACHE_TTL_SECONDS", default=3600, ge=0)
    max_size: int = Field(alias="CACHE_MAX_SIZE", default=1000, ge=1, le=100000)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"memory", "redis"}
        if v not in allowed:
            raise ValueError(f"cache backend 必须是 {allowed} 之一，当前: {v}")
        return v


class RedisSettings(_BaseConfig):
    url: str = Field(alias="REDIS_URL", default="redis://localhost:6379/0")


class RateLimitSettings(_BaseConfig):
    enabled: bool = Field(alias="RATE_LIMIT_ENABLED", default=False)
    max_per_minute: int = Field(alias="RATE_LIMIT_MAX_PER_MINUTE", default=10, ge=1, le=100)
    session_ttl: int = Field(alias="SESSION_TTL_MINUTES", default=30, ge=1, le=1440)


class ServerSettings(_BaseConfig):
    host: str = Field(alias="SERVER_HOST", default="0.0.0.0")
    port: int = Field(alias="SERVER_PORT", default=7860, ge=1, le=65535)


# ================================================================
# 辅助函数
# ================================================================

_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _read_dotenv(env_path: str | None = None) -> Dict[str, str]:
    if env_path is None:
        env_path = os.environ.get("DOTENV_PATH", str(Path(__file__).parent.parent / ".env"))
    result: Dict[str, str] = {}
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = _ENV_LINE_RE.match(line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2)
                val = val.strip().strip("\"'")
                result[key] = val
    except FileNotFoundError:
        pass
    return result


# ================================================================
# 顶层 Settings
# ================================================================

class Settings(BaseSettings):
    """全局配置聚合器。

    用法:
        from config.settings import settings
        print(settings.llm.model)
        print(settings.OPENAI_API_KEY)  # 兼容旧代码
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False,
    )

    # ---- 子配置 ----
    qdrant: QdrantSettings = QdrantSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    memory: MemorySettings = MemorySettings()
    long_term_memory: LongTermMemorySettings = LongTermMemorySettings()
    chunking: ChunkingSettings = ChunkingSettings()
    cache: CacheSettings = CacheSettings()
    redis: RedisSettings = RedisSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    server: ServerSettings = ServerSettings()

    # ---- 顶层 ----
    debug: bool = Field(default=False)
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    local_data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    working_memory_max_items: int = Field(default=10, ge=1, le=100)

    # ---- 数据重组：将扁平 env 变量注入子模型 ----
    @model_validator(mode="before")
    @classmethod
    def _nest_env_vars(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """将扁平的 .env 变量按前缀重组为嵌套结构，供子模型校验。

        由于 pydantic v2 在 model_validator(mode="before") 运行前
        已消费掉与子模型字段名匹配的扁平键（如 qdrant_url），
        因此直接从 .env 文件读取原始数据以确保完整。
        """
        prefix_map: Dict[str, str] = {
            "QDRANT_": "qdrant",
            "NEO4J_": "neo4j",
            "OPENAI_": "llm",
            "LLM_": "llm",
            "EMBEDDING_": "embedding",
            "RERANKER_": "reranker",
            "RETRIEVAL_": "retrieval",
            "RERANK_": "retrieval",
            "MULTI_QUERY_": "retrieval",
            "MEMORY_": "memory",
            "LS_": "long_term_memory",
            "CHUNK_": "chunking",
            "CACHE_": "cache",
            "REDIS_": "redis",
            "RATE_LIMIT_": "rate_limit",
            "SESSION_": "rate_limit",
            "SERVER_": "server",
        }

        raw_env = _read_dotenv()

        nested: Dict[str, Dict[str, Any]] = {}
        for key, val in raw_env.items():
            key_upper = key.upper()
            for prefix, sub_field in prefix_map.items():
                if key_upper.startswith(prefix):
                    nested.setdefault(sub_field, {})[key_upper] = val
                    break

        if nested:
            for sub_field, sub_data in nested.items():
                if sub_field not in values:
                    values[sub_field] = sub_data
        return values

    # ---- 启动校验 ----
    @model_validator(mode="after")
    def check_required_services(self) -> "Settings":
        """启动时校验关键服务是否配置。"""
        if not self.llm.api_key:
            raise ValueError("OPENAI_API_KEY 未设置！请在 .env 中配置。")
        if not self.qdrant.is_configured and not self.neo4j.is_configured:
            raise ValueError("QDRANT_URL 和 NEO4J_URI 至少配置一个！")
        return self

    # ---- 兼容性：自动代理旧代码以大写下划线方式访问 ----
    _COMPAT_MAP: ClassVar[Dict[str, Tuple[str, str]]] = {
        "QDRANT_URL":              ("qdrant", "url"),
        "QDRANT_API_KEY":          ("qdrant", "api_key"),
        "QDRANT_COLLECTION_NAME":  ("qdrant", "collection_name"),
        "NEO4J_URI":               ("neo4j", "uri"),
        "NEO4J_USERNAME":          ("neo4j", "username"),
        "NEO4J_PASSWORD":          ("neo4j", "password"),
        "NEO4J_DATABASE":          ("neo4j", "database"),
        "OPENAI_API_KEY":          ("llm", "api_key"),
        "OPENAI_BASE_URL":         ("llm", "base_url"),
        "LLM_MODEL":               ("llm", "model"),
        "EMBEDDING_MODEL_PATH":    ("embedding", "model_path"),
        "RERANKER_MODEL_PATH":     ("reranker", "model_path"),
        "RETRIEVAL_STRATEGY":      ("retrieval", "strategy"),
        "RETRIEVAL_K":             ("retrieval", "k"),
        "RERANK_TOP_K":            ("retrieval", "rerank_top_k"),
        "MULTI_QUERY_COUNT":       ("retrieval", "multi_query_count"),
        "MEMORY_WINDOW_SIZE":      ("memory", "window_size"),
        "LS_ENABLED":              ("long_term_memory", "enabled"),
        "LS_EPISODIC_TOP_K":       ("long_term_memory", "episodic_top_k"),
        "LS_EPISODIC_COLLECTION":  ("long_term_memory", "episodic_collection"),
        "LS_EPISODIC_MAX_STORE":   ("long_term_memory", "episodic_max_store"),
        "LS_SEMANTIC_TOP_K":       ("long_term_memory", "semantic_top_k"),
        "LS_SEMANTIC_NEIGHBOR":    ("long_term_memory", "semantic_neighbor"),
        "LS_IMPORTANCE_INIT":      ("long_term_memory", "importance_init"),
        "LS_IMPORTANCE_GROWTH":    ("long_term_memory", "importance_growth"),
        "LS_FORGET_THRESHOLD":     ("long_term_memory", "forget_threshold"),
        "LS_RECALL_TOTAL":         ("long_term_memory", "recall_total"),
        "LS_WORKING_TOP_K":        ("long_term_memory", "working_top_k"),
        "CHUNK_SIZE":              ("chunking", "size"),
        "CHUNK_OVERLAP":           ("chunking", "overlap"),
        "CHUNK_TOKEN_TARGET":      ("chunking", "token_target"),
        "CHUNK_TOKEN_MAX":         ("chunking", "token_max"),
        "CHUNK_OVERLAP_TOKEN":     ("chunking", "overlap_token"),
        "CACHE_ENABLED":           ("cache", "enabled"),
        "CACHE_BACKEND":           ("cache", "backend"),
        "CACHE_TTL_SECONDS":       ("cache", "ttl_seconds"),
        "CACHE_MAX_SIZE":          ("cache", "max_size"),
        "REDIS_URL":               ("redis", "url"),
        "RATE_LIMIT_ENABLED":      ("rate_limit", "enabled"),
        "RATE_LIMIT_MAX_PER_MINUTE": ("rate_limit", "max_per_minute"),
        "SESSION_TTL_MINUTES":     ("rate_limit", "session_ttl"),
        "USE_CUDA":                ("embedding", "use_cuda"),
        "BASE_DIR":                ("_self", "base_dir"),
        "LOCAL_DATA_DIR":          ("_self", "local_data_dir"),
        "DEBUG":                   ("_self", "debug"),
        "WORKING_MEMORY_MAX_ITEMS": ("_self", "working_memory_max_items"),
    }

    def __getattr__(self, name: str):
        """动态代理旧式大写属性访问 → settings.LLM_MODEL → settings.llm.model"""
        if name in self._COMPAT_MAP:
            section, attr = self._COMPAT_MAP[name]
            if section == "_self":
                return object.__getattribute__(self, attr)
            return getattr(getattr(self, section), attr)
        raise AttributeError(f"'Settings' 没有属性 '{name}'")


# ---- 单例 ----
settings = Settings()