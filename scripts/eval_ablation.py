# -*- coding: utf-8 -*-
# scripts/eval_ablation.py
"""增强检索消融实验

对比 baseline（单次检索）与 enhanced（问题拆解 + 并行多轮检索）的差异。

用法:
    python scripts/eval_ablation.py                     # 使用默认 testset
    python scripts/eval_ablation.py --testset path.json  # 自定义测试集
    python scripts/eval_ablation.py --limit 5            # 只跑前 5 题
    python scripts/eval_ablation.py --top-k 3            # 指定 top_k

输出:
    - 每题 Recall@K / MRR / Hit Rate 对比
    - 汇总指标对比表
    - 延迟对比
"""

from __future__ import annotations

import argparse
import time
import sys
import json
from pathlib import Path
from typing import Optional

from config.settings import settings
from evaluation.testset import TestSet
from evaluation.runner import EvalRunner
from evaluation.metrics import RetrievalMetrics
from utils.logger import logger


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


def run_ablation(testset: TestSet, top_k: int = 5) -> dict:
    """执行增强检索消融实验。"""

    from core.retrievers.factory import get_retriever

    # ── Baseline: 标准检索器 ──
    _safe_print("\n[1/2] Baseline: 单次检索 (增强关闭)")
    baseline_retriever = get_retriever(source_filter=None, override_strategy="simple")

    def baseline_fn(query: str, top_k: int = top_k):
        return baseline_retriever.invoke(query, top_k=top_k)

    baseline_runner = EvalRunner(retriever_fn=baseline_fn)

    # ── Enhanced: 带拆解 + 并行的增强检索 ──
    _safe_print("\n[2/2] Enhanced: 拆解 + 并行多轮")
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

    # ── 逐题评估 ──
    baseline_results = _eval_all(baseline_runner, testset, top_k, "baseline")
    enhanced_results = _eval_all(enhanced_runner, testset, top_k, "enhanced")

    # ── 打印对比 ──
    _print_details(testset, baseline_results, enhanced_results)
    summary = _print_summary(baseline_results, enhanced_results)

    return summary


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


def _print_details(testset, baseline_results, enhanced_results):
    _print_header("每题 Recall 对比")
    _print_row(
        ["#", "题目", "R@5(基线)", "R@5(增强)", "提升", "MRR(基线)", "MRR(增强)"],
        [3, 42, 10, 10, 6, 10, 10],
    )
    _safe_print("-" * 95)

    for i, (case, br, er) in enumerate(zip(testset, baseline_results, enhanced_results), 1):
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


def _print_summary(baseline_results, enhanced_results):
    n = len(baseline_results)

    def _avg(attr):
        return sum(getattr(r, attr, 0) for r in baseline_results) / n

    def _avg_e(attr):
        return sum(getattr(r, attr, 0) for r in enhanced_results) / n

    _print_header("消融实验汇总")

    metrics = [
        ("Recall@1", "recall_at_1"),
        ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5"),
        ("Hit Rate", "hit_rate"),
    ]

    _print_row(["指标", "基线", "增强", "Δ", "提升%"], [16, 10, 10, 10, 10])
    _safe_print("-" * 60)

    results = {}
    for name, attr in metrics:
        b = _avg(attr)
        e = _avg_e(attr)
        delta = e - b
        pct = f"{(delta / b) * 100:+.1f}%" if b > 0 else "N/A"
        sign = "+" if delta >= 0 else ""
        _print_row([name, f"{b:.2%}", f"{e:.2%}", f"{sign}{delta:.2%}", pct], [16, 10, 10, 10, 10])
        results[name] = {"baseline": round(b, 4), "enhanced": round(e, 4), "delta": round(delta, 4)}

    # 延迟
    b_lat = sum(r.latency_ms for r in baseline_results) / n
    e_lat = sum(r.latency_ms for r in enhanced_results) / n
    _safe_print("-" * 60)
    _print_row(["Avg Latency(ms)", f"{b_lat:.0f}", f"{e_lat:.0f}", f"{e_lat - b_lat:+.0f}", ""], [16, 10, 10, 10, 10])
    results["avg_latency_ms"] = {"baseline": round(b_lat, 0), "enhanced": round(e_lat, 0)}

    # 结论
    _safe_print(f"\n{'─' * 60}")
    recall_delta = _avg_e("recall_at_5") - _avg("recall_at_5")
    if recall_delta > 0.05:
        _safe_print("  结论: 增强检索显著提升跨文档多跳问题的召回率 (+{:.1%})".format(recall_delta))
    elif recall_delta > 0:
        _safe_print("  结论: 增强检索小幅改善跨文档多跳召回 (+{:.1%})".format(recall_delta))
    else:
        _safe_print("  结论: 增强检索未显示提升，考虑优化拆解 prompt 或增加子查询数")
    _safe_print(f"  {'─' * 60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="增强检索消融实验")
    parser.add_argument("--testset", type=Path,
                        default=Path("tests/data/multi_hop_eval.json"),
                        help="多跳测试集路径")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索 top_k")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅评估前 N 题 (0=全部)")
    parser.add_argument("--output", type=Path, default=None,
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    testset = TestSet.from_json(args.testset)
    if args.limit > 0:
        testset.cases = testset.cases[:args.limit]

    _safe_print(f"加载 {len(testset)} 道多跳测试题  (top_k={args.top_k})")

    # 验证测试集
    for i, case in enumerate(testset, 1):
        if not case.gold_docs or len(case.gold_docs) < 2:
            logger.warning(f"  第 {i} 题 gold_docs 少于 2 个: {case.question[:40]}")
        if not case.expected_keywords:
            logger.warning(f"  第 {i} 题无 expected_keywords: {case.question[:40]}")

    summary = run_ablation(testset, top_k=args.top_k)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _safe_print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()