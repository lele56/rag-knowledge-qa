# -*- coding: utf-8 -*-
# scripts/bench_cache.py
"""缓存命中率 Benchmark

模拟真实用户场景：100 次提问中约 30% 为重复/变体，
统计缓存拦截率与 LLM 调用节省比例。

用法:
    python scripts/bench_cache.py
    python scripts/bench_cache.py --queries 200 --dup-ratio 0.35
"""

import sys
import asyncio
import argparse
import json
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cache import AsyncTTLCache
from services.qa_service import get_cost_tracker

_BASE_QUESTIONS = [
    "什么是Transformer的自注意力机制？",
    "BERT和GPT的主要区别是什么？",
    "如何评估RAG系统的检索效果？",
    "向量数据库Qdrant的索引原理是什么？",
    "大模型微调中LoRA的原理是什么？",
    "什么是检索增强生成RAG？",
    "LangChain中的ReAct Agent如何工作？",
    "知识图谱在问答系统中的作用是什么？",
    "文档分块的最佳实践是什么？",
    "CrossEncoder重排序相比双塔模型的优势？",
    "如何处理PDF中的表格数据？",
    "Neo4j在图检索中的应用场景？",
    "BM25算法和向量检索的互补性？",
    "多模态模型如何处理文本和图像？",
    "Prompt工程的核心技巧有哪些？",
]

_DUP_VARIANTS = [
    "Transformer自注意力机制的原理是什么？",
    "BERT和GPT有什么不同？",
    "怎样评估RAG的检索质量？",
    "Qdrant的索引是怎么实现的？",
    "LoRA微调的原理？",
    "什么是RAG？",
    "ReAct Agent怎么用？",
    "知识图谱在问答里有什么用？",
    "文档怎么分块比较好？",
    "CrossEncoder和双塔模型的区别？",
    "PDF表格怎么解析？",
    "Neo4j怎么用于图检索？",
    "BM25和向量检索如何结合？",
    "多模态模型怎么处理图片和文字？",
    "Prompt工程有什么技巧？",
]


def generate_queries(n: int, dup_ratio: float = 0.3) -> list:
    """生成 N 个模拟提问，dup_ratio 比例来自重复/变体池。"""
    n_dup = int(n * dup_ratio)
    n_unique = n - n_dup

    unique = _BASE_QUESTIONS[:]
    while len(unique) < n_unique:
        unique.append(random.choice(_BASE_QUESTIONS))

    dup = random.choices(_DUP_VARIANTS, k=n_dup)

    all_q = unique[:n_unique] + dup
    random.shuffle(all_q)
    return all_q


async def run_benchmark(queries: list, ttl: int = 3600, max_size: int = 1000):
    cache = AsyncTTLCache(ttl=ttl, max_size=max_size)
    tracker = get_cost_tracker()

    hits = 0
    calls = 0
    latencies = []
    call_latencies = []

    for i, q in enumerate(queries, 1):
        tracker.on_request()
        t0 = time.perf_counter()

        cached = await cache.get(q)
        if cached:
            hits += 1
            tracker.on_cache_hit()
            latencies.append((time.perf_counter() - t0) * 1000)
            if i <= 5:
                print(f"  [{i}/{len(queries)}]  命中: {q[:40]}... ({latencies[-1]:.1f}ms)")
            continue

        calls += 1
        latency = (time.perf_counter() - t0) * 1000
        call_latencies.append(latency)
        latencies.append(latency)

        simulated_result = f"模拟回答: {q}"
        await cache.set(q, simulated_result)

        if i <= 5:
            print(f"  [{i}/{len(queries)}] MISS: {q[:40]}... ({latency:.1f}ms)")

    print(f"\n{'=' * 60}")
    print(f"  📊 缓存命中率 Benchmark 结果")
    print(f"  {'-' * 50}")
    print(f"  总请求数:       {len(queries)}")
    print(f"  缓存命中:       {hits}  ({hits / len(queries) * 100:.1f}%)")
    print(f"  LLM 实际调用:   {calls}  (节省 {hits} 次, {hits / len(queries) * 100:.1f}%)")
    print(f"  缓存命中位延迟: {sum(latencies[:hits]) / max(hits, 1):.1f}ms (avg)")
    print(f"  LLM 调用位延迟: {sum(call_latencies) / max(calls, 1):.1f}ms (avg, 模拟)")
    print(f"  {'─' * 50}")
    print(f"  结论: 缓存命中率 {tracker.hit_rate:.0f}%，LLM 调用降低 {tracker.cost_saved_ratio:.0f}%")
    print(f"{'=' * 60}\n")

    return {
        "total": len(queries),
        "cache_hits": hits,
        "llm_calls": calls,
        "hit_rate": round(hits / len(queries) * 100, 1),
        "llm_saving_pct": round(hits / len(queries) * 100, 1),
        "avg_cache_latency_ms": round(sum(latencies[:hits]) / max(hits, 1), 1) if hits else 0,
        "avg_llm_latency_ms": round(sum(call_latencies) / max(calls, 1), 1) if calls else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="缓存命中率 Benchmark")
    parser.add_argument("--queries", type=int, default=100, help="模拟提问总数")
    parser.add_argument("--dup-ratio", type=float, default=0.3, help="重复/变体比例 (0-1)")
    args = parser.parse_args()

    queries = generate_queries(args.queries, args.dup_ratio)
    print(f"生成 {len(queries)} 个模拟提问 (重复比例 ≈{args.dup_ratio:.0%})\n")

    result = asyncio.run(run_benchmark(queries))

    out_path = Path("results") / "cache_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()