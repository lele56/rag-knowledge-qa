# -*- coding: utf-8 -*-
# scripts/eval_ablation.py
"""消融实验：对比不同检索方案的指标差异

用法:
    # 实验①: 纯向量 vs 向量+BM25+重排序
    python scripts/eval_ablation.py --mode hybrid

    # 实验②: 单次检索 vs 增强检索(拆解+并行)
    python scripts/eval_ablation.py --mode enhanced

    # 通用参数
    python scripts/eval_ablation.py --mode hybrid --limit 3   # 只跑前 3 题
    python scripts/eval_ablation.py --mode enhanced --top-k 3  # 指定 top_k
    python scripts/eval_ablation.py --mode hybrid --output result.json  # 导出 JSON
"""

from __future__ import annotations

import argparse
import time
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from evaluation.testset import TestSet
from evaluation.runner import EvalRunner
from evaluation.metrics import RetrievalMetrics
from utils.logger import logger


# ── 安全打印 ──
def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode("ascii"), **kwargs)


def _print_header(title: str, width: int = 70):
    _safe_print(f"\n{'=' * width}")
    _safe_print(f"  {title}")
    _safe_print(f"{'=' * width}")


def _print_row(cols: list, widths: list, sep: str = "  "):
    line = sep.join(f"{str(c):<{w}}" for c, w in zip(cols, widths))
    _safe_print(line)


# ── 评估工具 ──
def _eval_all(runner: EvalRunner, testset: TestSet, top_k: int, label: str) -> list:
    results = []
    for i, case in enumerate(testset, 1):
        _safe_print(f"  [{i}/{len(testset)}] {label}: {case.question[:50]}...")
        t0 = time.time()
        try:
            docs = runner._retriever_fn(case.question, top_k=top_k)
        except Exception as e:
            logger.warning(f"检索失败 [{case.question[:30]}]: {e}")
            docs = []
        latency_ms = (time.time() - t0) * 1000
        r = RetrievalMetrics.evaluate(
            docs,
            case.expected_keywords,
            gold_docs=case.gold_docs,
            latency_ms=latency_ms,
            question=case.question,
        )
        results.append(r)
    return results


def _print_details(testset, baseline_results, exp_results, blabel, elabel):
    _print_header("每题 Recall 对比")
    _print_row(
        ["#", "题目", f"R@5({blabel})", f"R@5({elabel})", "提升", f"MRR({blabel})", f"MRR({elabel})"],
        [3, 42, 10, 10, 6, 10, 10],
    )
    _safe_print("-" * 95)

    for i, (case, br, er) in enumerate(zip(testset, baseline_results, exp_results), 1):
        q = case.question[:40]
        delta = er.recall_at_5 - br.recall_at_5
        arrow = "+" if delta > 0 else ("=" if delta == 0 else "-")
        _print_row(
            [
                i, q,
                f"{br.recall_at_5:.0%}",
                f"{er.recall_at_5:.0%}",
                f"{arrow}{abs(delta):.0%}",
                f"{br.mrr:.3f}",
                f"{er.mrr:.3f}",
            ],
            [3, 42, 10, 10, 6, 10, 10],
        )


def _print_summary(baseline_results, exp_results, blabel, elabel, conclusion_fn=None):
    n = len(baseline_results)

    def _avg(attr, results=baseline_results):
        return sum(getattr(r, attr, 0) for r in results) / n

    _print_header("消融实验汇总")

    metrics = [
        ("Recall@1", "recall_at_1"),
        ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5"),
        ("Hit Rate", "hit_rate"),
    ]

    _print_row(["指标", blabel, elabel, "Δ", "提升%"], [16, 10, 10, 10, 10])
    _safe_print("-" * 60)

    results = {}
    for name, attr in metrics:
        b = _avg(attr, baseline_results)
        e = _avg(attr, exp_results)
        delta = e - b
        pct = f"{(delta / b) * 100:+.1f}%" if b > 0 else "N/A"
        sign = "+" if delta >= 0 else ""
        _print_row([name, f"{b:.2%}", f"{e:.2%}", f"{sign}{delta:.2%}", pct], [16, 10, 10, 10, 10])
        results[name] = {"baseline": round(b, 4), "enhanced": round(e, 4), "delta": round(delta, 4)}

    b_lat = sum(r.latency_ms for r in baseline_results) / n
    e_lat = sum(r.latency_ms for r in exp_results) / n
    _safe_print("-" * 60)
    _print_row(["Avg Latency(ms)", f"{b_lat:.0f}", f"{e_lat:.0f}", f"{e_lat - b_lat:+.0f}", ""], [16, 10, 10, 10, 10])
    results["avg_latency_ms"] = {"baseline": round(b_lat, 0), "enhanced": round(e_lat, 0)}

    if conclusion_fn:
        conclusion_fn(baseline_results, exp_results)

    return results


# ============================================================
# 实验①: 纯向量 vs 向量+BM25+重排序
# ============================================================
def run_hybrid_ablation(testset: TestSet, top_k: int = 5) -> dict:
    """对比纯向量检索与混合检索（BM25 + RRF 融合 + CrossEncoder 重排序）。"""

    from core.retrievers.hybrid import HybridRetriever
    from core.retrievers.factory import _make_hybrid
    from core.retrievers.base import get_base_retriever

    # ── Baseline: 纯向量检索（无 BM25，无重排序）──
    _safe_print("\n[1/2] Baseline: 纯向量检索")
    base_retriever, _ = get_base_retriever(source_filter=None)
    pure_vector = HybridRetriever(
        vector_retriever=base_retriever,
        bm25_retriever=None,
        reranker=None,
        no_rerank=True,
    )

    def pure_fn(query: str, top_k: int = top_k):
        return pure_vector.invoke(query, top_k=top_k)

    pure_runner = EvalRunner(retriever_fn=pure_fn)

    # ── 实验组: 向量+BM25+重排序 ──
    _safe_print("\n[2/2] 实验组: 向量+BM25+重排序")
    hybrid = _make_hybrid(base_retriever, source_filter=None)

    def hybrid_fn(query: str, top_k: int = top_k):
        return hybrid.invoke(query, top_k=top_k)

    hybrid_runner = EvalRunner(retriever_fn=hybrid_fn)

    # ── 评估 ──
    pure_results = _eval_all(pure_runner, testset, top_k, "纯向量")
    hybrid_results = _eval_all(hybrid_runner, testset, top_k, "混合")

    _print_details(testset, pure_results, hybrid_results, "纯向量", "混合")

    def conclusion(baseline, exp):
        _safe_print(f"\n{'─' * 60}")
        recall_delta = _avg_simple(exp, "recall_at_5") - _avg_simple(baseline, "recall_at_5")
        if recall_delta > 0.05:
            _safe_print(f"  结论: 向量+BM25+重排序显著提升召回率 (+{recall_delta:.1%})")
        elif recall_delta > 0:
            _safe_print(f"  结论: 混合检索小幅改善召回 (+{recall_delta:.1%})")
        else:
            _safe_print("  结论: 混合检索未显示提升，检查 BM25 索引或 reranker 加载状态")
        _safe_print(f"  {'─' * 60}\n")

    return _print_summary(pure_results, hybrid_results, "纯向量", "混合", conclusion)


# ============================================================
# 实验②: 单次检索 vs 增强检索（拆解+并行）
# ============================================================
def run_enhanced_ablation(testset: TestSet, top_k: int = 5) -> dict:
    """对比单次检索与增强检索（问题拆解 + 并行子查询 + 合并去重）。"""

    from core.retrievers.factory import get_retriever

    # ── Baseline: 标准检索 ──
    _safe_print("\n[1/2] Baseline: 单次检索")
    baseline_retriever = get_retriever(source_filter=None, override_strategy="simple")

    def baseline_fn(query: str, top_k: int = top_k):
        return baseline_retriever.invoke(query, top_k=top_k)

    baseline_runner = EvalRunner(retriever_fn=baseline_fn)

    # ── 实验组: 增强检索 ──
    _safe_print("\n[2/2] 实验组: 增强检索 (拆解 + 并行)")
    from core.infrastructure.llm import get_llm
    from core.retrievers.strategies import EnhancedRetriever
    llm = get_llm()

    base_for_enhanced = get_retriever(source_filter=None, override_strategy="simple")
    enhanced = EnhancedRetriever(
        retriever_fn=lambda q, k=top_k: base_for_enhanced.invoke(q, top_k=k),
        llm=llm,
        enabled=True,
    )

    def enhanced_fn(query: str, top_k: int = top_k):
        result = enhanced.search(query, top_k=top_k)
        return result.docs

    enhanced_runner = EvalRunner(retriever_fn=enhanced_fn)

    # ── 评估 ──
    baseline_results = _eval_all(baseline_runner, testset, top_k, "单次")
    enhanced_results = _eval_all(enhanced_runner, testset, top_k, "增强")

    _print_details(testset, baseline_results, enhanced_results, "单次", "增强")

    def conclusion(baseline, exp):
        _safe_print(f"\n{'─' * 60}")
        recall_delta = _avg_simple(exp, "recall_at_5") - _avg_simple(baseline, "recall_at_5")
        if recall_delta > 0.05:
            _safe_print(f"  结论: 增强检索显著提升多跳问题召回 (+{recall_delta:.1%})")
        elif recall_delta > 0:
            _safe_print(f"  结论: 增强检索小幅改善多跳召回 (+{recall_delta:.1%})")
        else:
            _safe_print("  结论: 增强检索未显示提升，优化拆解 prompt 或增加子查询数")
        _safe_print(f"  {'─' * 60}\n")

    return _print_summary(baseline_results, enhanced_results, "单次", "增强", conclusion)


def _avg_simple(results, attr):
    n = len(results)
    if n == 0:
        return 0
    return sum(getattr(r, attr, 0) for r in results) / n


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="消融实验：对比不同检索方案")
    parser.add_argument(
        "--mode", type=str, default="enhanced", choices=["hybrid", "enhanced"],
        help="hybrid=纯向量vs混合检索 | enhanced=单次vs增强检索",
    )
    parser.add_argument("--testset", type=Path,
                        default=Path("tests/data/multi_hop_eval.json"),
                        help="测试集路径")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top_k")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 题 (0=全部)")
    parser.add_argument("--output", type=Path, default=None, help="结果 JSON 路径")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path("results") / f"ablation_{args.mode}.json"

    testset = TestSet.from_json(args.testset)
    if args.limit > 0:
        testset.cases = testset.cases[:args.limit]

    mode_label = {"hybrid": "纯向量 vs 混合检索", "enhanced": "单次检索 vs 增强检索"}
    _safe_print(f"消融实验: {mode_label[args.mode]}  ({len(testset)} 题, top_k={args.top_k})")

    for i, case in enumerate(testset, 1):
        if not case.gold_docs or len(case.gold_docs) < 2:
            logger.warning(f"  第 {i} 题 gold_docs 少于 2 个: {case.question[:40]}")
        if not case.expected_keywords:
            logger.warning(f"  第 {i} 题无 expected_keywords: {case.question[:40]}")

    if args.mode == "hybrid":
        summary = run_hybrid_ablation(testset, top_k=args.top_k)
    else:
        summary = run_enhanced_ablation(testset, top_k=args.top_k)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _safe_print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()