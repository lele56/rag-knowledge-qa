# evaluation/reporter.py
"""评估报告生成：控制台表格、JSON 导出、Markdown 报告。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from evaluation.metrics import EvalSummary, RetrievalResult, GenerationResult


class EvalReporter:
    """评估报告生成器"""

    @staticmethod
    def _fmt_pct(v: float) -> str:
        return f"{v * 100:.1f}%" if v <= 1 else f"{v:.1f}%"

    @staticmethod
    def _color(v: float, threshold_good: float = 0.8, threshold_ok: float = 0.5) -> str:
        if v >= threshold_good:
            return f"🟢 {EvalReporter._fmt_pct(v)}"
        elif v >= threshold_ok:
            return f"🟡 {EvalReporter._fmt_pct(v)}"
        return f"🔴 {EvalReporter._fmt_pct(v)}"

    # ---------- 控制台输出 ----------

    @classmethod
    def console(cls, summary: EvalSummary, title: str = "评估报告") -> None:
        """打印控制台格式的评估报告"""
        print()
        print("=" * 70)
        print(f"  {title}")
        print(f"  题目数: {summary.total_questions}  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if summary.retrieval:
            print()
            print("  📊 检索质量")
            print("  " + "-" * 50)
            r = summary.retrieval
            metrics = [
                ("Recall@1", r.get("recall@1", 0)),
                ("Recall@3", r.get("recall@3", 0)),
                ("Recall@5", r.get("recall@5", 0)),
                ("Precision@1", r.get("precision@1", 0)),
                ("Precision@3", r.get("precision@3", 0)),
                ("Precision@5", r.get("precision@5", 0)),
                ("MRR", r.get("mrr", 0)),
                ("NDCG@5", r.get("ndcg@5", 0)),
                ("Hit Rate", r.get("hit_rate", 0)),
            ]
            for name, val in metrics:
                print(f"  {name:<15} {cls._color(val)}")

        if summary.generation:
            print()
            print("  📝 生成质量")
            print("  " + "-" * 50)
            g = summary.generation
            gen_metrics = [
                ("忠实度", g.get("faithfulness", 0)),
                ("答案相关性", g.get("answer_relevance", 0)),
                ("上下文相关性", g.get("context_relevance", 0)),
            ]
            for name, val in gen_metrics:
                print(f"  {name:<15} {cls._color(val)}")

        if summary.performance:
            print()
            print("  ⚡ 性能")
            print("  " + "-" * 50)
            p = summary.performance
            perf_metrics = [
                ("平均延迟", f"{p.get('mean_ms', 0):.0f}ms"),
                ("中位数延迟", f"{p.get('median_ms', 0):.0f}ms"),
                ("P95 延迟", f"{p.get('p95_ms', 0):.0f}ms"),
                ("P99 延迟", f"{p.get('p99_ms', 0):.0f}ms"),
                ("最慢", f"{p.get('max_ms', 0):.0f}ms"),
                ("总耗时", f"{p.get('total_ms', 0) / 1000:.1f}s"),
            ]
            for name, val in perf_metrics:
                print(f"  {name:<15} {val}")

        # 每题详情
        if summary.retrieval_details:
            print()
            print("  📋 检索详情")
            print("  " + "-" * 50)
            for i, r in enumerate(summary.retrieval_details, 1):
                hit = "✅" if r.hit_rate > 0 else "❌"
                rank = f"第{r.first_hit_rank}位" if r.first_hit_rank else "未命中"
                print(
                    f"  {hit} [{i}] {r.question[:50]:<50} "
                    f"R@1={r.recall_at_1:.0f} R@5={r.recall_at_5:.0f} "
                    f"MRR={r.mrr:.2f} {rank} ({r.latency_ms:.0f}ms)"
                )

        if summary.generation_details:
            print()
            print("  📋 生成详情")
            print("  " + "-" * 50)
            for i, g in enumerate(summary.generation_details, 1):
                f_color = cls._color(g.faithfulness, 0.7, 0.4)
                print(
                    f"  [{i}] {g.question[:50]:<50} "
                    f"忠实={f_color} 相关={cls._color(g.answer_relevance, 0.7, 0.4)} "
                    f"({g.latency_ms:.0f}ms)"
                )

        print()
        print("=" * 70)

    # ---------- JSON 导出 ----------

    @classmethod
    def to_json(cls, summary: EvalSummary, path: Path) -> None:
        """导出详细结果到 JSON"""
        data = {
            "meta": {
                "total_questions": summary.total_questions,
                "timestamp": datetime.now().isoformat(),
            },
            "summary": {
                "retrieval": summary.retrieval,
                "generation": summary.generation,
                "performance": summary.performance,
            },
            "details": {
                "retrieval": [
                    {
                        "question": r.question,
                        "recall@1": r.recall_at_1,
                        "recall@3": r.recall_at_3,
                        "recall@5": r.recall_at_5,
                        "precision@1": r.precision_at_1,
                        "precision@3": r.precision_at_3,
                        "precision@5": r.precision_at_5,
                        "mrr": r.mrr,
                        "ndcg@5": r.ndcg_at_5,
                        "hit_rate": r.hit_rate,
                        "first_hit_rank": r.first_hit_rank,
                        "latency_ms": r.latency_ms,
                    }
                    for r in summary.retrieval_details
                ],
                "generation": [
                    {
                        "question": g.question,
                        "answer": g.answer[:500],
                        "faithfulness": g.faithfulness,
                        "answer_relevance": g.answer_relevance,
                        "context_relevance": g.context_relevance,
                        "latency_ms": g.latency_ms,
                        "tokens_used": g.tokens_used,
                    }
                    for g in summary.generation_details
                ],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  📄 详细结果已保存: {path}")

    # ---------- Markdown 报告 ----------

    @classmethod
    def to_markdown(cls, summary: EvalSummary, path: Optional[Path] = None) -> str:
        """生成 Markdown 格式报告，可选保存到文件"""
        lines = []
        lines.append(f"# 评估报告")
        lines.append(f"")
        lines.append(f"- **题目数**: {summary.total_questions}")
        lines.append(f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")

        if summary.retrieval:
            lines.append(f"## 检索质量")
            lines.append(f"")
            lines.append(f"| 指标 | 分数 |")
            lines.append(f"|------|------|")
            r = summary.retrieval
            for name in ["recall@1", "recall@3", "recall@5", "precision@1", "precision@3", "precision@5", "mrr", "ndcg@5", "hit_rate"]:
                val = r.get(name, 0)
                lines.append(f"| {name} | {cls._fmt_pct(val)} |")
            lines.append(f"")

        if summary.generation:
            lines.append(f"## 生成质量")
            lines.append(f"")
            lines.append(f"| 指标 | 分数 |")
            lines.append(f"|------|------|")
            g = summary.generation
            for name, key in [("忠实度", "faithfulness"), ("答案相关性", "answer_relevance"), ("上下文相关性", "context_relevance")]:
                val = g.get(key, 0)
                lines.append(f"| {name} | {cls._fmt_pct(val)} |")
            lines.append(f"")

        if summary.performance:
            lines.append(f"## 性能")
            lines.append(f"")
            p = summary.performance
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|------|")
            for name in ["mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]:
                val = p.get(name, 0)
                lines.append(f"| {name} | {val:.0f}ms |")
            lines.append(f"")

        md = "\n".join(lines)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"  📄 Markdown 报告已保存: {path}")
        return md