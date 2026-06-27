# evaluation/ragas_metrics.py
"""RAGAS 评估指标：基于 RAGAS 官方库的 Context Precision / Recall / Answer Correctness。

与现有 GenerationMetrics 互补：
  - GenerationMetrics: Faithfulness / Answer Relevancy / Context Relevancy
  - RagasMetrics:     Context Precision / Context Recall / Answer Correctness
"""
from __future__ import annotations
from typing import List, Optional, Any
from dataclasses import dataclass, field

from utils.logger import logger


@dataclass
class RagasResult:
    """单题 RAGAS 评估结果。"""
    question: str
    answer: str = ""
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    details: dict = field(default_factory=dict)


class RagasMetrics:
    """RAGAS 评估指标包装器，内部使用 RAGAS 官方库。"""

    @staticmethod
    def _to_ragas_dataset(
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str] = None,
    ):
        """转换为 RAGAS Dataset 格式。"""
        from datasets import Dataset

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        return Dataset.from_dict(data)

    @classmethod
    def evaluate_batch(
        cls,
        llm: Any,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str] = None,
        metrics: List[str] = None,
    ) -> dict:
        """批量 RAGAS 评估。

        Args:
            llm: LangChain BaseChatModel
            questions: 问题列表
            answers: 生成答案列表
            contexts_list: 每个问题对应的检索上下文列表
            ground_truths: 参考答案列表（可选，用于 context_recall 和 answer_correctness）
            metrics: 指定指标列表，默认全部

        Returns:
            {"context_precision": 0.85, "context_recall": 0.72, "answer_correctness": 0.68}
        """
        from ragas import evaluate
        from ragas.metrics.collections import (
            context_precision,
            context_recall,
            answer_correctness,
        )

        dataset = cls._to_ragas_dataset(
            questions, answers, contexts_list, ground_truths
        )

        selected = []
        metric_names = metrics or ["context_precision"]
        metric_map = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

        for name in metric_names:
            if name in metric_map:
                selected.append(metric_map[name])

        if not selected:
            from ragas.metrics.collections import context_precision as cp
            selected = [cp]

        result = evaluate(
            dataset,
            metrics=selected,
            llm=llm,
        )
        return dict(result)

    @classmethod
    def evaluate(
        cls,
        llm: Any,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> dict:
        """单项 RAGAS 评估。

        Returns:
            {"context_precision": float, "context_recall": float, "answer_correctness": float}
        """
        metrics = ["context_precision"]
        if ground_truth:
            metrics.extend(["context_recall", "answer_correctness"])

        try:
            results = cls.evaluate_batch(
                llm=llm,
                questions=[question],
                answers=[answer],
                contexts_list=[contexts],
                ground_truths=[ground_truth] if ground_truth else None,
                metrics=metrics,
            )
            return {
                "context_precision": results.get("context_precision", 0.0),
                "context_recall": results.get("context_recall", 0.0),
                "answer_correctness": results.get("answer_correctness", 0.0),
            }
        except Exception as e:
            logger.warning(f"RAGAS 评估失败: {e}")
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
            }