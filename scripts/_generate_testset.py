# -*- coding: utf-8 -*-
"""测试集生成：自定义 LLM 一条龙。

为什么不用 RAGAS：
  - RAGAS synthesizer 依赖 KnowledgeGraph 中的 summary/entities/themes 属性
  - 这些属性必须由 LLM transforms（SummaryExtractor/NERExtractor/ThemesExtractor）生成
  - 跳过 transforms → synthesizer 无法工作

方案：直接调 LLM 生成问题+答案+gold_chunks，每篇文档 1 次调用。
"""
import json
import random
import sys
import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from config.settings import settings
from core.infrastructure.vector_store import _get_client
from utils.logger import logger

random.seed(42)

OUTPUT = PROJECT_ROOT / "data" / "test_question_ragas.json"
PER_DOC = {"easy": 2, "medium": 2, "hard": 1}
MAX_CTX_CHARS = 15000  # 上下文总预算（qwen3.6-plus 128K 窗口，多采样才能覆盖文档各处）
MIN_CHUNK_LEN = 80  # 太短的 chunk 跳过（目录行通常 < 80 字符）
MIN_PER_CHUNK = 300  # 单个 chunk 最少字符数（避免截太碎失去上下文意义）

# chunk 元组：(doc_id, chunk_index, page_content, is_toc_like, chunk_total)
ChunkInfo = tuple[str, int, str, bool, int]
TestCase = dict[str, Any]

GEN_PROMPT = """你是一个专业的测试题生成器。根据以下文档片段生成 {total} 道中文问答题。

每个片段前有 [chunk_X] 标记，X 是片段编号。问题中不要出现"chunk_X"字样，用自然语言描述即可。

重要约束：
1. 只从文档正文内容出题。如果某个片段是参考文献列表、目录、索引、致谢或附录，请忽略它，不要基于它出题。
2. 答案（ground_truth）必须能在给定片段中找到原文依据，不得编造或使用外部知识。
3. 每道题尽量基于不同的片段出题，覆盖更多给出的 [chunk_X]，不要所有题都围绕同一个片段。easy 用 1 个片段，medium 用 1-2 个，hard 用 2-3 个。如果片段不足，优先保证 easy 和 medium 分布在不同片段上。

难度要求：
- easy（简单，{easy}题）：答案可直接从某个片段摘录，不需要推理
- medium（中等，{medium}题）：需要组合多个片段的信息来回答
- hard（困难，{hard}题）：需要跨片段推理、对比，或涉及深层原理

输出格式（严格 JSON 数组，不要输出其他内容）：
[
  {{"question": "...", "difficulty": "easy", "ground_truth": "...", "source_chunk_ids": [277, 280]}},
  ...
]

要求：source_chunk_ids 必须是片段前 [chunk_X] 中出现的编号，只能是整数。

文档标题：{title}
文档内容：
{context}
"""


def load_all_docs_from_qdrant() -> dict[str, list[ChunkInfo]]:
    """从 Qdrant 加载所有文档，按 source 分组。

    返回 {source: [(doc_id, chunk_index, page_content, is_toc_like, chunk_total)]}。
    """
    client = _get_client()
    col = settings.QDRANT_COLLECTION_NAME
    by_source: dict[str, list[ChunkInfo]] = defaultdict(list)
    offset = None
    while True:
        result = client.scroll(
            collection_name=col, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        points, next_offset = result
        if not points:
            break
        for p in points:
            payload = p.payload
            if not isinstance(payload, dict):
                continue
            content = payload.get("page_content", "")
            if not content:
                continue
            src = payload.get("source", "unknown")
            doc_id = payload.get("doc_id", "")
            chunk_idx = payload.get("chunk_index")
            if chunk_idx is None:
                chunk_idx = payload.get("doc_chunk_index")
            if not doc_id or chunk_idx is None:
                continue
            is_toc = bool(payload.get("is_toc_like", False))
            chunk_total = int(payload.get("chunk_total", 0) or payload.get("doc_chunk_total", 0) or 0)
            by_source[src].append((doc_id, int(chunk_idx), content, is_toc, chunk_total))
        offset = next_offset
        if next_offset is None:
            break
    logger.info(f"从 Qdrant 加载了 {sum(len(v) for v in by_source.values())} 个 chunk，{len(by_source)} 篇文档")
    return by_source


def is_short_chunk(content: str) -> bool:
    """判断是否为太短的 chunk（目录项、空白页等）。"""
    return len(content) < MIN_CHUNK_LEN


def sample_context(
    chunks: list[ChunkInfo],
    max_chars: int = MAX_CTX_CHARS,
) -> tuple[str, dict[int, str]]:
    """均匀分层采样：文档越大段越多，每段截断到预算内。

    返回 (带编号的文本, chunk_id → real_id 映射)。
    """
    good = [(did, cid, txt, is_toc, total) for did, cid, txt, is_toc, total in chunks
            if not is_short_chunk(txt) and not is_toc]
    if len(good) < 1:
        return "", {}

    good.sort(key=lambda x: x[1])
    n_good = len(good)

    # 分段数：幂函数平滑增长，文档越大段越多，硬性最低 3 段（保证出题多样性）
    n_seg_by_size = max(3, int(n_good ** 0.4) + 2)  # 366→12, 41→6, 10→4
    avg_len = sum(len(t) for _, _, t, _, _ in good) / n_good
    n_seg_by_budget = max(3, int(max_chars / max(avg_len, MIN_PER_CHUNK)))
    n_seg = max(3, min(n_seg_by_size, n_seg_by_budget))
    n_seg = min(n_seg, n_good)
    per_chunk = max(MIN_PER_CHUNK, max_chars // n_seg)

    selected_indices = []
    step = n_good / n_seg
    for i in range(n_seg):
        start = int(i * step)
        end = max(start + 1, int((i + 1) * step))
        if start >= n_good:
            break
        idx = random.randint(start, min(end, n_good) - 1)
        selected_indices.append(idx)

    selected = []
    id_map: dict[int, str] = {}
    for i in selected_indices:
        did, cid, txt, _, _ = good[i]
        snippet = txt[:per_chunk]
        selected.append(f"[chunk_{cid}]\n{snippet}")
        id_map[cid] = f"{did}_chunk_{cid}"

    return "\n\n---\n\n".join(selected), id_map


def call_llm(
    llm: ChatOpenAI,
    prompt: str,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """调用 LLM 生成 JSON 数组，自动重试。重试 3 次仍失败则抛出异常。"""
    for attempt in range(max_retries):
        try:
            resp = llm.invoke(prompt)
            text = (resp.content if hasattr(resp, 'content') else str(resp)).strip()
            # 清理 markdown 代码块包裹
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败 (第 {attempt+1}/{max_retries} 次)")
        except Exception as e:
            logger.warning(f"LLM 调用失败 (第 {attempt+1}/{max_retries} 次): {e}")
    raise RuntimeError(f"LLM 调用 {max_retries} 次均失败")


def _parse_generated(
    generated: list[dict[str, Any]],
    src: str,
    id_map: dict[int, str],
    sampled_ids: set[int],
) -> list[TestCase]:
    """解析 LLM 返回的 JSON，映射 chunk_id 并校验。

    返回解析后的测试用例列表。
    """
    cases: list[TestCase] = []
    for item in generated:
        q = (item.get("question", "") or "").strip()
        ans = (item.get("ground_truth", "") or "").strip()
        diff = item.get("difficulty", "easy")
        raw_ids = item.get("source_chunk_ids", [])
        valid_ids = [int(x) for x in raw_ids if int(x) in sampled_ids]
        if not valid_ids:
            logger.warning(
                f"{src}: LLM 返回的 chunk_id {raw_ids} 均不在采样范围 {sampled_ids}，"
                f"兜底用第一个 chunk"
            )
            valid_ids = [list(sampled_ids)[0]]
        gold_chunks = [id_map[cid] for cid in valid_ids]
        gold_docs = list(set(c.split("_chunk_")[0] for c in gold_chunks))
        if q and ans:
            cases.append({
                "question": q,
                "ground_truth": ans,
                "source": src,
                "difficulty": diff,
                "gold_chunks": gold_chunks,
                "gold_docs": gold_docs,
            })
    return cases


def _generate_for_source(
    src: str,
    chunks: list[ChunkInfo],
    llm: ChatOpenAI,
) -> list[TestCase]:
    """对单篇文档生成测试题。返回 [] 表示失败。"""
    ctx, id_map = sample_context(chunks)
    sampled_ids = set(id_map.keys())
    prompt = GEN_PROMPT.format(
        title=src, context=ctx,
        total=PER_DOC["easy"] + PER_DOC["medium"] + PER_DOC["hard"],
        easy=PER_DOC["easy"], medium=PER_DOC["medium"], hard=PER_DOC["hard"],
    )
    print(f"生成: {src} ({len(chunks)} chunks, 采样 {len(sampled_ids)} 个, ctx ~{len(ctx)} chars)...")

    if not sampled_ids:
        logger.warning(f"{src} 过滤后无有效 chunk，跳过")
        return []

    try:
        generated_cases = call_llm(llm, prompt)
        cases = _parse_generated(generated_cases, src, id_map, sampled_ids)
        print(f"  → {len(cases)} 题")
        return cases
    except Exception as e:
        logger.error(f"生成失败 {src}: {e}")
        print(f"  ⚠ 失败: {e}")
        return []


def _to_output(cases: list[TestCase]) -> list[dict[str, Any]]:
    """转换为最终输出格式。"""
    return [
        {
            "question": c["question"],
            "expected_keywords": [],
            "expected_answer": c["ground_truth"],
            "gold_docs": c["gold_docs"],
            "gold_chunks": c["gold_chunks"],
            "category": "auto",
            "difficulty": c["difficulty"],
            "tags": [],
        }
        for c in cases
    ]


def main() -> None:
    by_source = load_all_docs_from_qdrant()
    if not by_source:
        logger.error("Qdrant 中没有文档")
        return

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.7,
        max_tokens=8192,
        request_timeout=120,
    )
    print(f"LLM: {settings.LLM_MODEL} | API: {settings.OPENAI_BASE_URL}")

    all_sources = sorted(by_source.keys())
    print(f"共 {len(all_sources)} 篇文档\n")

    all_cases: list[TestCase] = []
    failed: list[str] = []

    for src in all_sources:
        cases = _generate_for_source(src, by_source[src], llm)
        if cases:
            all_cases.extend(cases)
        else:
            failed.append(src)

    print(f"\nLLM 共生成 {len(all_cases)} 题")

    # 重试失败的文档
    for src in failed:
        print(f"重试: {src}...")
        cases = _generate_for_source(src, by_source[src], llm)
        if cases:
            all_cases.extend(cases)
            print(f"  → 成功")
        else:
            print(f"  ⚠ 重试仍失败")

    # 输出
    output_cases = _to_output(all_cases)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_cases, f, ensure_ascii=False, indent=2)

    stats = defaultdict(int)
    for c in output_cases:
        stats[c["difficulty"]] += 1
    print(f"\n已保存 {len(output_cases)} 题到 {OUTPUT}")
    print(f"难度分布: {dict(stats)}")


if __name__ == "__main__":
    main()