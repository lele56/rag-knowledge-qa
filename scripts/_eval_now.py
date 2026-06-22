import json, sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.runner import EvalRunner
from evaluation.testset import TestSet
from config.settings import settings

def _safe_print(s: str):
    """GBK 兼容的 print"""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("gbk", errors="replace").decode("gbk"))

parser = argparse.ArgumentParser(description="离线评估脚本")
parser.add_argument("--mode", choices=["retrieval", "gen", "full"], default="retrieval",
                    help="评估模式: retrieval(检索), gen(生成), full(端到端)")
parser.add_argument("--strategy", default=None,
                    help="检索策略覆写: simple / multi_query / hyde")
parser.add_argument("--top-k", type=int, default=5,
                    help="检索 top_k 数量")
parser.add_argument("--limit", type=int, default=0,
                    help="仅评估前 N 题 (0=全部)")
args = parser.parse_args()

testset = TestSet.from_json(Path("data/test_question_example.json"))
if args.limit > 0:
    testset.cases = testset.cases[:args.limit]
_safe_print(f"加载 {len(testset)} 道测试题  (策略: {args.strategy or settings.RETRIEVAL_STRATEGY}, 模式: {args.mode})\n")

from core.retrievers.factory import get_retriever
from core.infrastructure.vector_store import get_vector_store
from core.infrastructure.embeddings import get_embeddings

store = get_vector_store()
embeddings = get_embeddings()
retriever = get_retriever(source_filter=None, override_strategy=args.strategy)

def retriever_fn(query, top_k=5):
    return retriever.invoke(query, top_k=top_k)

runner = EvalRunner(retriever_fn=retriever_fn)

# ---------- 生成评估需要 QA 函数 ----------
if args.mode in ("gen", "full"):
    from core.infrastructure.llm import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    llm = get_llm()
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库问答助手。请根据提供的上下文回答问题。如果上下文不足以回答，请如实说明。"),
        ("user", "## 上下文\n{context}\n\n## 问题\n{question}\n\n请用中文回答，简明扼要，不超过500字。"),
    ])
    qa_chain = qa_prompt | llm

    def qa_fn(question: str) -> str:
        docs = retriever.invoke(question, top_k=args.top_k)
        context = "\n\n".join(d.page_content[:800] for d in docs)
        resp = qa_chain.invoke({"context": context, "question": question})
        return resp.content if hasattr(resp, "content") else str(resp)

    runner.attach_qa(qa_fn).attach_llm(llm)

# ---------- 执行评估 ----------
if args.mode == "retrieval":
    summary = runner.evaluate_retrieval(testset, top_k=args.top_k)
elif args.mode == "gen":
    summary = runner.evaluate_generation(testset)
else:
    summary = runner.evaluate_full(testset, top_k=args.top_k)

s = summary.retrieval
g = summary.generation
p = summary.performance

# ---------- 每题详情 ----------
_safe_print("\n" + "=" * 85)
_safe_print("  每题得分详情")
_safe_print("=" * 85)

if args.mode in ("retrieval", "full") and summary.retrieval_details:
    for i, r in enumerate(summary.retrieval_details, 1):
        icon = "V" if r.recall_at_1 > 0 else "X"
        q = r.question[:45]
        _safe_print(f"  {icon} [{i:2d}] {q:<45}  R@1={r.recall_at_1:.0f} R@3={r.recall_at_3:.0f} R@5={r.recall_at_5:.0f}")

if args.mode in ("gen", "full") and summary.generation_details:
    for i, gr in enumerate(summary.generation_details, 1):
        q = gr.question[:45]
        _safe_print(f"     [{i:2d}] {q:<45}  faith={gr.faithfulness:.2f} rel={gr.answer_relevance:.2f} ctx={gr.context_relevance:.2f}")

# ---------- 汇总 ----------
_safe_print("\n" + "=" * 70)
_safe_print(f"  评估汇总  ({args.mode})")
_safe_print("=" * 70)

if s:
    _safe_print(f"\n  [检索指标]")
    _safe_print(f"  Recall@1:     {s['recall@1']:.2%}")
    _safe_print(f"  Recall@3:     {s['recall@3']:.2%}")
    _safe_print(f"  Recall@5:     {s['recall@5']:.2%}")
    _safe_print(f"  MRR:          {s['mrr']:.4f}")
    _safe_print(f"  NDCG@5:       {s['ndcg@5']:.4f}")
    _safe_print(f"  Hit Rate:     {s['hit_rate']:.2%}")

if g:
    _safe_print(f"\n  [生成指标]")
    _safe_print(f"  Faithfulness:      {g.get('faithfulness', 0):.2%}")
    _safe_print(f"  Answer Relevance:  {g.get('answer_relevance', 0):.2%}")
    _safe_print(f"  Context Relevance: {g.get('context_relevance', 0):.2%}")

if p:
    _safe_print(f"\n  [性能指标]")
    _safe_print(f"  Avg Latency:  {p['mean_ms']:.0f}ms")
    _safe_print(f"  P50/P95:      {p['p50_ms']:.0f}ms / {p['p95_ms']:.0f}ms")

now = time.strftime("%Y-%m-%dT%H:%M:%S")
output = {
    "meta": {
        "total_questions": len(testset),
        "timestamp": now,
        "mode": args.mode,
        "chunk_strategy": getattr(settings, "CHUNK_STRATEGY", "unknown"),
        "retrieval_strategy": args.strategy or settings.RETRIEVAL_STRATEGY,
    },
    "summary": {
        "retrieval": s,
        "generation": g,
        "performance": p,
    },
    "details": {
        "retrieval": [r.to_dict() for r in summary.retrieval_details],
        "generation": [gr.to_dict() for gr in summary.generation_details],
    },
}
with open("data/test_question_example_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
_safe_print(f"\n结果已保存 -> data/test_question_example_results.json")