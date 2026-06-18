"""上下文构建器单元测试"""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch


# ============================================================
# ContextConfig 测试
# ============================================================

class TestContextConfig:
    def test_defaults(self):
        from core.context_builder import ContextConfig
        cfg = ContextConfig()
        assert cfg.max_tokens == 6000
        assert cfg.enable_mmr is True
        assert cfg.mmr_lambda == 0.7

    def test_custom(self):
        from core.context_builder import ContextConfig
        cfg = ContextConfig(max_tokens=2000, enable_mmr=False)
        assert cfg.max_tokens == 2000
        assert cfg.enable_mmr is False

    def test_available_tokens(self):
        from core.context_builder import ContextConfig
        cfg = ContextConfig(max_tokens=6000, reserve_ratio=0.15)
        assert cfg.available_tokens == 5100  # 6000 * 0.85


# ============================================================
# ContextBuilder 测试
# ============================================================

class TestContextBuilder:
    @pytest.fixture
    def builder(self):
        from core.context_builder import ContextBuilder, ContextConfig
        return ContextBuilder(ContextConfig())

    def test_empty_docs(self, builder):
        result = builder.build(user_query="测试问题", retrieved_docs=[])
        assert "测试问题" in result

    def test_single_doc(self, builder):
        from langchain_core.documents import Document
        docs = [Document(page_content="Transformer 是一种神经网络架构，用于处理序列数据。")]
        result = builder.build(user_query="Transformer 是什么", retrieved_docs=docs)
        assert "Transformer" in result

    def test_multiple_docs(self, builder):
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Transformer 使用注意力机制处理序列数据。"),
            Document(page_content="BERT 是基于 Transformer 的预训练模型。"),
        ]
        result = builder.build(user_query="Transformer BERT", retrieved_docs=docs)
        assert "Transformer" in result
        assert "BERT" in result

    def test_build_with_sources(self, builder):
        from langchain_core.documents import Document
        docs = [
            Document(
                page_content="Transformer 架构内容",
                metadata={"source": "paper.pdf", "page": 1},
            )
        ]
        result = builder.build(user_query="paper 内容", retrieved_docs=docs)
        assert "paper.pdf" in result

    def test_build_with_similarity(self, builder):
        from langchain_core.documents import Document
        docs = [
            Document(
                page_content="相关内容",
                metadata={"source": "doc.pdf", "similarity_score": 0.95},
            )
        ]
        result = builder.build(user_query="内容 相关", retrieved_docs=docs)
        assert "doc.pdf" in result

    def test_build_output_instruction_present(self, builder):
        from langchain_core.documents import Document
        docs = [Document(page_content="Transformer 架构")]
        result = builder.build(user_query="Transformer 架构", retrieved_docs=docs)
        assert "证据" in result

    def test_build_with_system_instructions(self, builder):
        from langchain_core.documents import Document
        docs = [Document(page_content="Transformer 架构")]
        result = builder.build(
            user_query="Transformer",
            retrieved_docs=docs,
            system_instructions="请简洁回答",
        )
        assert "简洁回答" in result


# ============================================================
# 上下文策略测试
# ============================================================

class TestContextStrategies:
    def test_compact_strategy(self):
        from core.context.strategies import CompactStrategy, ContextBuilder
        from langchain_core.documents import Document
        builder = ContextBuilder()
        strategy = CompactStrategy(builder)
        docs = [Document(page_content="Transformer 是一种深度学习架构")]
        result = strategy.build("Transformer 是什么", docs)
        assert "Transformer" in result

    def test_full_strategy(self):
        from core.context.strategies import FullStrategy, ContextBuilder
        from langchain_core.documents import Document
        builder = ContextBuilder()
        strategy = FullStrategy(builder)
        docs = [Document(page_content="Transformer 是一种深度学习架构")]
        result = strategy.build("Transformer 是什么", docs)
        assert "Transformer" in result

    def test_multi_doc_strategy(self):
        from core.context.strategies import MultiDocStrategy, ContextBuilder
        from langchain_core.documents import Document
        builder = ContextBuilder()
        strategy = MultiDocStrategy(builder)
        docs = [
            Document(page_content="Transformer 架构", metadata={"source": "doc1.pdf"}),
            Document(page_content="BERT 模型", metadata={"source": "doc2.pdf"}),
        ]
        result = strategy.build("Transformer BERT", docs)
        assert "Transformer" in result
        assert "BERT" in result

    def test_evidence_only_strategy(self):
        from core.context.strategies import EvidenceOnly, ContextBuilder
        from langchain_core.documents import Document
        builder = ContextBuilder()
        strategy = EvidenceOnly(builder)
        docs = [Document(page_content="Transformer 架构")]
        result = strategy.build("Transformer 架构", docs)
        assert "Transformer" in result

    def test_get_strategy_factory(self):
        from core.context.strategies import get_strategy
        for name in ["compact", "full", "evidence", "multidoc"]:
            s = get_strategy(name)
            assert s is not None