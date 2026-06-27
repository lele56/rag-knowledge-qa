# evaluation/runner.py
"""评估执行器：编排检索 + 生成评估流程。"""
from __future__ import annotations
import time
from typing import Optional, List, Callable, Dict, Any, Tuple
from pathlib import Path

from evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    PerformanceMetrics,
    RetrievalResult,
    GenerationResult,
    EvalSummary,
)
from evaluation.testset import TestSet, TestCase
from utils.logger import logger

# LangChain Document 类型
try:
    from langchain_core.documents import Document
except ImportError:
    Document = Any


class EvalRunner:
    """评估执行器。

    支持三种评估模式：
      - retrieval: 仅评估检索质量（Recall, Precision, MRR, NDCG）
      - generation: 仅评估生成质量（Faithfulness, Relevance）
      - full: 端到端评估（检索 + 生成）
    """

    def __init__(
        self,
        retriever_fn: Optional[Callable] = None,
        qa_fn: Optional[Callable] = None,
        llm: Any = None,
        use_llm_relevance: bool = False,
    ):
        """
        Args:
            retriever_fn: 检索函数 (query: str, top_k: int) -> List[Document]
            qa_fn: 问答函数 (question: str) -> str
            llm: LLM 实例（用于 LLM-as-judge 打分）
            use_llm_relevance: 检索评估是否使用 LLM 判断相关性（默认只用关键词）
        """
        self._retriever_fn = retriever_fn
        self._qa_fn = qa_fn
        self._llm = llm
        self._use_llm_relevance = use_llm_relevance

    # ---------- 检索评估 ----------

    def _run_retrieval(self, case: TestCase, top_k: int = 5) -> RetrievalResult:
        """执行单次检索评估"""
        if not self._retriever_fn:
            raise RuntimeError("未注入检索函数，请调用 attach_retriever()")

        t0 = time.time()
        try:
            docs = self._retriever_fn(case.question, top_k=top_k * 4)
        except Exception as e:
            logger.warning(f"检索失败 [{case.question[:30]}]: {e}")
            return RetrievalResult(question=case.question, retrieved_count=0)
        latency_ms = (time.time() - t0) * 1000

        if self._use_llm_relevance and self._llm:
            result = RetrievalMetrics.evaluate_with_llm(
                self._llm, docs, case.expected_keywords,
                question=case.question, latency_ms=latency_ms,
            )
        else:
            result = RetrievalMetrics.evaluate(
                docs, case.expected_keywords,
                gold_chunks=case.gold_chunks,
                gold_docs=case.gold_docs,
                latency_ms=latency_ms, question=case.question
            )
        return result

    def evaluate_retrieval(self, testset: TestSet, top_k: int = 5) -> EvalSummary:
        """批量检索评估"""
        results: List[RetrievalResult] = []
        latencies: List[float] = []

        for i, case in enumerate(testset, 1):
            logger.info(f"  [{i}/{len(testset)}] 检索评估: {case.question[:50]}...")
            r = self._run_retrieval(case, top_k=top_k)
            results.append(r)
            latencies.append(r.latency_ms)

        summary = self._summarize(results, latencies, "retrieval")
        return summary

    # ---------- 生成评估 ----------

    def _run_generation(self, case: TestCase) -> GenerationResult:
        """执行单次生成评估"""
        if not self._qa_fn:
            raise RuntimeError("未注入问答函数，请调用 attach_qa()")
        if not self._llm:
            raise RuntimeError("未注入 LLM，请调用 attach_llm()")

        # 1) 生成答案
        t0 = time.time()
        try:
            answer = self._qa_fn(case.question)
        except Exception as e:
            logger.warning(f"生成失败 [{case.question[:30]}]: {e}")
            return GenerationResult(question=case.question, answer=f"ERROR: {e}")
        latency_ms = (time.time() - t0) * 1000

        # 2) 检索上下文（用于评估忠实度）
        context = ""
        if self._retriever_fn:
            try:
                docs = self._retriever_fn(case.question, top_k=5)
                context = "\n\n".join(
                    (d.page_content if hasattr(d, "page_content") else str(d))[:500]
                    for d in docs
                )
            except Exception as e:
                logger.debug(f"上下文提取失败: {e}")

        # 3) LLM 裁判打分
        try:
            result = GenerationMetrics.evaluate(
                self._llm, case.question, answer, context,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.warning(f"LLM 打分失败: {e}")
            result = GenerationResult(
                question=case.question, answer=answer,
                faithfulness=0.5, answer_relevance=0.5, context_relevance=0.5,
                latency_ms=latency_ms,
            )
        return result

    def evaluate_generation(self, testset: TestSet) -> EvalSummary:
        """批量生成评估"""
        results: List[GenerationResult] = []
        latencies: List[float] = []

        for i, case in enumerate(testset, 1):
            logger.info(f"  [{i}/{len(testset)}] 生成评估: {case.question[:50]}...")
            r = self._run_generation(case)
            results.append(r)
            latencies.append(r.latency_ms)

        summary = self._summarize(results, latencies, "generation")
        return summary

    # ---------- 端到端评估 ----------

    def evaluate_full(self, testset: TestSet, top_k: int = 5) -> EvalSummary:
        """端到端评估：检索 + 生成"""
        ret_results: List[RetrievalResult] = []
        gen_results: List[GenerationResult] = []
        ret_latencies: List[float] = []
        gen_latencies: List[float] = []

        for i, case in enumerate(testset, 1):
            logger.info(f"  [{i}/{len(testset)}] 端到端: {case.question[:50]}...")

            # 检索评估
            if self._retriever_fn:
                rr = self._run_retrieval(case, top_k=top_k)
                ret_results.append(rr)
                ret_latencies.append(rr.latency_ms)

            # 生成评估
            if self._qa_fn and self._llm:
                gr = self._run_generation(case)
                gen_results.append(gr)
                gen_latencies.append(gr.latency_ms)

        return EvalSummary(
            total_questions=len(testset),
            retrieval=self._agg_retrieval(ret_results) if ret_results else {},
            generation=self._agg_generation(gen_results) if gen_results else {},
            performance=PerformanceMetrics.summarize(ret_latencies + gen_latencies),
            retrieval_details=ret_results,
            generation_details=gen_results,
        )

    # ---------- 汇总 ----------

    @staticmethod
    def _agg_retrieval(results: List[RetrievalResult]) -> Dict[str, float]:
        if not results:
            return {}
        n = len(results)
        return {
            "recall@1": sum(r.recall_at_1 for r in results) / n,
            "recall@3": sum(r.recall_at_3 for r in results) / n,
            "recall@5": sum(r.recall_at_5 for r in results) / n,
            "precision@1": sum(r.precision_at_1 for r in results) / n,
            "precision@3": sum(r.precision_at_3 for r in results) / n,
            "precision@5": sum(r.precision_at_5 for r in results) / n,
            "mrr": sum(r.mrr for r in results) / n,
            "ndcg@5": sum(r.ndcg_at_5 for r in results) / n,
            "hit_rate": sum(r.hit_rate for r in results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in results) / n,
        }

    @staticmethod
    def _agg_generation(results: List[GenerationResult]) -> Dict[str, float]:
        if not results:
            return {}
        n = len(results)
        return {
            "faithfulness": sum(r.faithfulness for r in results) / n,
            "answer_relevance": sum(r.answer_relevance for r in results) / n,
            "context_relevance": sum(r.context_relevance for r in results) / n,
            "keyword_recall": sum(r.keyword_recall for r in results) / n,
            "keyword_f1": sum(r.keyword_f1 for r in results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in results) / n,
            "avg_tokens": sum(r.tokens_used for r in results) / n,
        }

    def _summarize(self, results: list, latencies: list[float],
                   mode: str = "retrieval") -> EvalSummary:
        agg = self._agg_retrieval if mode == "retrieval" else self._agg_generation
        detail_key = "retrieval_details" if mode == "retrieval" else "generation_details"
        return EvalSummary(
            total_questions=len(results),
            performance=PerformanceMetrics.summarize(latencies),
            **{mode: agg(results), detail_key: results},
        )

    # ---------- 依赖注入 ----------

    # ---------- RAGAS 评估 ----------

    def evaluate_ragas(self, testset: TestSet, top_k: int = 5) -> dict:
        """RAGAS 兼容评估：Context Precision / Recall / Answer Correctness。

        需要注入 retriever_fn、qa_fn、llm。
        """
        from evaluation.ragas_metrics import RagasMetrics

        if not self._llm:
            raise RuntimeError("RAGAS 评估需要注入 LLM，请调用 attach_llm()")

        precisions: List[float] = []
        recalls: List[float] = []
        correctnesses: List[float] = []

        for i, case in enumerate(testset, 1):
            logger.info(f"  [{i}/{len(testset)}] RAGAS: {case.question[:50]}...")

            # 检索上下文
            contexts: List[str] = []
            if self._retriever_fn:
                try:
                    docs = self._retriever_fn(case.question, top_k=top_k)
                    contexts = [
                        d.page_content if hasattr(d, "page_content") else str(d)
                        for d in docs
                    ]
                except Exception as e:
                    logger.warning(f"    检索失败: {e}")

            # 生成答案
            answer = ""
            if self._qa_fn:
                try:
                    answer = self._qa_fn(case.question)
                except Exception as e:
                    logger.warning(f"    生成失败: {e}")

            # 参考答案
            ground_truth = getattr(case, "ground_truth", "") or ""

            # RAGAS 指标
            metrics = RagasMetrics.evaluate(
                self._llm, case.question, answer, contexts, ground_truth
            )
            precisions.append(metrics["context_precision"])
            if ground_truth:
                recalls.append(metrics["context_recall"])
                correctnesses.append(metrics["answer_correctness"])

        n = len(testset)
        return {
            "context_precision": sum(precisions) / n if precisions else 0,
            "context_recall": sum(recalls) / len(recalls) if recalls else 0,
            "answer_correctness": sum(correctnesses) / len(correctnesses) if correctnesses else 0,
        }

    # ---------- 依赖注入 ----------

    def attach_retriever(self, fn: Callable) -> "EvalRunner":
        self._retriever_fn = fn
        return self

    def attach_qa(self, fn: Callable) -> "EvalRunner":
        self._qa_fn = fn
        return self

    def attach_llm(self, llm: Any) -> "EvalRunner":
        self._llm = llm
        return self

    def use_llm_relevance(self, enabled: bool = True) -> "EvalRunner":
        """启用/禁用 LLM 相关性判断（检索评估）"""
        self._use_llm_relevance = enabled
        return self