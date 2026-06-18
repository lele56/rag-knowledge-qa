# core/context/types.py
"""上下文构建的数据类型定义"""
import re
from typing import Set, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


def _tokenize_chinese(text: str) -> Set[str]:
    """中文分词：用字符 bigram 做简单分词，避免依赖 jieba 等外部库"""
    text = text.lower().strip()
    tokens = set()
    for w in re.findall(r'[a-zA-Z0-9]+', text):
        tokens.add(w)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    for i in range(len(chinese_chars)):
        tokens.add(chinese_chars[i])
        if i < len(chinese_chars) - 1:
            tokens.add(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


def _count_tokens(text: str) -> int:
    """使用 tiktoken 精确计数，回退到启发式估算"""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    except Exception:
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other = len(text) - chinese
        return int(chinese / 2 + other / 4)


@dataclass
class ContextPacket:
    """上下文信息包"""
    content: str
    source: str = "unknown"
    importance: float = 0.5
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return _count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_tokens: int = 6000
    reserve_ratio: float = 0.15
    min_relevance: float = 0.2
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    max_history_turns: int = 6
    max_memory_items: int = 5
    max_retrieval_chunks: int = 8

    @property
    def available_tokens(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))