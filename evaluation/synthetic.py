# evaluation/synthetic.py
"""合成测试集生成器：基于 RAGAS TestsetGenerator 自动生成问答对。

三阶段流水线：知识图谱提取 → 问题生成（easy/medium/hard） → 参考答案标注。
"""
from __future__ import annotations
import json
from typing import List, Any, Optional
from pathlib import Path

from utils.logger import logger


class SyntheticTestSet:
    """合成测试集：基于 RAGAS TestsetGenerator 从文档库自动生成问答对。"""

    def __init__(self, llm: Any, embedding: Any = None, retriever_fn: Any = None):
        """
        Args:
            llm: LangChain BaseChatModel（用于生成问题和答案）
            embedding: LangChain Embeddings（用于 RAGAS 知识图谱提取）
            retriever_fn: 检索函数 (query, top_k) -> List[Document]（可选，用于标注 gold chunks）
        """
        self._llm = llm
        self._embedding = embedding
        self._retriever_fn = retriever_fn

    def _build_generator(self):
        """构建 RAGAS TestsetGenerator。"""
        from ragas.testset import TestsetGenerator

        emb = self._embedding
        if emb is None:
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings()
            logger.warning("未提供 embedding，使用默认 OpenAIEmbeddings")

        return TestsetGenerator(llm=self._llm, embedding_model=emb)

    def generate_from_docs(
        self,
        docs: List[Any],
        questions_per_doc: int = 3,
    ) -> List[dict]:
        """使用 RAGAS TestsetGenerator 从文档列表生成测试集。

        Args:
            docs: LangChain Document 列表
            questions_per_doc: 每篇文档生成的问题数

        Returns:
            [{"question": "...", "gold_docs": ["doc_id"], "gold_chunks": [...],
              "expected_keywords": [...], "ground_truth": "...", "difficulty": "..."}, ...]
        """
        generator = self._build_generator()

        # RAGAS 按文档分布生成
        testset = generator.generate_with_langchain_docs(
            docs,
            test_size=len(docs) * questions_per_doc,
            with_debugging_logs=False,
        )

        cases = []
        df = testset.to_pandas()

        for i, row in df.iterrows():
            question = row.get("question", "")
            ground_truth = row.get("ground_truth", "")
            source = row.get("source", "")

            # 检索 gold chunks
            gold_chunks = []
            if self._retriever_fn and question:
                try:
                    ret_docs = self._retriever_fn(question, top_k=5)
                    gold_chunks = [
                        d.metadata.get("chunk_id", "")
                        for d in ret_docs
                        if hasattr(d, "metadata")
                    ]
                except Exception as e:
                    logger.warning(f"检索 gold chunks 失败: {e}")

            case = {
                "question": question,
                "gold_docs": [source] if source else [],
                "gold_chunks": gold_chunks,
                "expected_keywords": [],
                "ground_truth": ground_truth,
                "difficulty": "medium",
            }
            cases.append(case)

        logger.info(f"RAGAS 合成测试集生成完成: {len(cases)} 题")
        return cases

    def generate_and_save(
        self,
        docs: List[Any],
        output_path: str,
        questions_per_doc: int = 3,
    ) -> str:
        """生成并保存测试集到 JSON 文件。"""
        cases = self.generate_from_docs(docs, questions_per_doc)
        output = Path(output_path)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存到 {output}")
        return str(output)

    @staticmethod
    def load(path: str) -> List[dict]:
        """加载测试集 JSON 文件。"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)