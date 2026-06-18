import json, sys, time
from pathlib import Path
from evaluation.runner import EvalRunner
from evaluation.testset import TestSet
from config.settings import settings

# 支持命令行覆写策略: python scripts/_eval_now.py multi_query
strategy = sys.argv[1] if len(sys.argv) > 1 else None

testset = TestSet.from_json(Path("data/test_question_example.json"))
print(f"加载 {len(testset)} 道测试题  (策略: {strategy or settings.RETRIEVAL_STRATEGY})\n")

from core.retriever_factory import get_retriever
from core.vector_store import get_vector_store
from core.embeddings import get_embeddings

store = get_vector_store()
embeddings = get_embeddings()
retriever = get_retriever(source_filter=None, override_strategy=strategy)

def retriever_fn(query, top_k=5):
    return retriever.invoke(query, top_k=top_k)

runner = EvalRunner(retriever_fn=retriever_fn)
summary = runner.evaluate_retrieval(testset, top_k=5)

s = summary.retrieval
p = summary.performance

# 每题命中情况
print("\n" + "=" * 85)
print("  每题命中情况  (recall@1 / recall@3 / recall@5)")
print("=" * 85)
hit_count = 0
total_recall1 = 0.0
for i, r in enumerate(summary.retrieval_details, 1):
    icon = "✓" if r.recall_at_1 > 0 else "✗"
    q = r.question[:45]
    print(f"  {icon} [{i:2d}] {q:<45}  {r.recall_at_1:.0f} / {r.recall_at_3:.0f} / {r.recall_at_5:.0f}  (first_rank={r.first_hit_rank or '-'})")
    if r.recall_at_1 > 0:
        hit_count += 1
    total_recall1 += r.recall_at_1
print(f"\n  → Recall@1 命中: {hit_count}/{len(testset)} ({total_recall1/len(testset):.1%})")

# 汇总
print("\n" + "=" * 70)
print("  检索评估汇总")
print("=" * 70)
print(f"  Recall@1:     {s['recall@1']:.2%}")
print(f"  Recall@3:     {s['recall@3']:.2%}")
print(f"  Recall@5:     {s['recall@5']:.2%}")
print(f"  Precision@1:  {s['precision@1']:.2%}")
print(f"  Precision@3:  {s['precision@3']:.2%}")
print(f"  Precision@5:  {s['precision@5']:.2%}")
print(f"  MRR:          {s['mrr']:.4f}")
print(f"  NDCG@5:       {s['ndcg@5']:.4f}")
print(f"  Hit Rate:     {s['hit_rate']:.2%}")
print(f"  Avg Latency:  {p['mean_ms']:.0f}ms")
print(f"  P50/P95:      {p['p50_ms']:.0f}ms / {p['p95_ms']:.0f}ms")

now = time.strftime("%Y-%m-%dT%H:%M:%S")
output = {
    "meta": {
        "total_questions": len(testset),
        "timestamp": now,
        "chunk_strategy": getattr(settings, "CHUNK_STRATEGY", "unknown"),
        "retrieval_strategy": strategy or settings.RETRIEVAL_STRATEGY,
    },
    "summary": {
        "retrieval": s,
        "generation": {},
        "performance": p,
    },
    "details": {"retrieval": [r.to_dict() for r in summary.retrieval_details]},
}
with open("data/test_question_example_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存 -> data/test_question_example_results.json")