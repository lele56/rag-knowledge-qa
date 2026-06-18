"""GSSC 上下文构建器 — Gather → Select → Structure → Compress

借鉴 HelloAgents 的 GSSC 模式，将 ad-hoc 上下文拼接替换为结构化、可预算的构建流程。
Structure 层使用 LangChain ChatPromptTemplate，Compress 层使用
RecursiveCharacterTextSplitter 做智能截断。
"""
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger
from config.prompts import OUTPUT_INSTRUCTION

from .types import ContextPacket, ContextConfig, _tokenize_chinese, _count_tokens

# GSSC 结构化模板
_GSSC_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "{system_instructions}"),
    ("human", "{current_task}"),
    ("system", "{evidence_section}"),
    ("system", "{history_section}"),
    ("system", "{output_instructions}"),
])


class ContextBuilder:
    """GSSC 上下文构建器

    用法:
        builder = ContextBuilder(config=ContextConfig(max_tokens=6000))
        context = builder.build(
            user_query="用户问题",
            retrieved_docs=[...],
            chat_history=[...],
            long_term_memories=[...],
            system_instructions="系统指令"
        )
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def build(
        self,
        user_query: str,
        retrieved_docs: Optional[List[Document]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        long_term_memories: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None,
    ) -> str:
        packets = self._gather(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            chat_history=chat_history,
            long_term_memories=long_term_memories,
            system_instructions=system_instructions,
            additional_packets=additional_packets,
        )
        selected = self._select(packets, user_query)
        structured = self._structure(selected, user_query)
        final = self._compress(structured)

        logger.info(
            f"ContextBuilder: {len(packets)} 收集 → {len(selected)} 筛选 "
            f"→ {_count_tokens(structured)} 结构化 "
            f"→ {_count_tokens(final)} 压缩完成"
        )
        return final

    def _gather(
        self,
        user_query: str,
        retrieved_docs: List[Document],
        chat_history: List[Dict[str, str]],
        long_term_memories: List[str],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket],
    ) -> List[ContextPacket]:
        packets: List[ContextPacket] = []

        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions, source="system", importance=1.0,
            ))

        if retrieved_docs:
            chunks_text = self._format_retrieved_docs(
                retrieved_docs[: self.config.max_retrieval_chunks]
            )
            packets.append(ContextPacket(
                content=chunks_text, source="retrieval", importance=0.9,
            ))

        if long_term_memories:
            memories_text = self._format_memories(
                long_term_memories[: self.config.max_memory_items]
            )
            packets.append(ContextPacket(
                content=memories_text, source="memory", importance=0.7,
            ))

        if chat_history:
            recent = chat_history[-self.config.max_history_turns * 2:]
            history_text = "\n".join(
                f"[{m.get('role', '未知')}] {m.get('content', '')}"
                for m in recent
            )
            packets.append(ContextPacket(
                content=history_text, source="history", importance=0.5,
            ))

        packets.extend(additional_packets or [])
        return packets

    def _select(
        self, packets: List[ContextPacket], user_query: str,
    ) -> List[ContextPacket]:
        if not packets:
            return []

        query_tokens = _tokenize_chinese(user_query)
        for p in packets:
            content_tokens = _tokenize_chinese(p.content)
            if query_tokens:
                overlap = len(query_tokens & content_tokens)
                p.relevance_score = min(1.0, overlap / len(query_tokens))
            else:
                p.relevance_score = 0.0

        system_packets = [p for p in packets if p.source == "system"]
        candidates = [p for p in packets if p.source != "system"]

        scored = sorted(
            candidates,
            key=lambda p: p.relevance_score * p.importance,
            reverse=True,
        )

        filtered = [p for p in scored if p.relevance_score >= self.config.min_relevance]

        if self.config.enable_mmr and len(filtered) > 1:
            filtered = self._mmr_select(filtered, query_tokens)

        budget = self.config.available_tokens
        selected: List[ContextPacket] = []
        used = 0

        for p in system_packets:
            est = p.token_estimate
            if used + est <= budget:
                selected.append(p)
                used += est

        for p in filtered:
            est = p.token_estimate
            if used + est > budget:
                break
            selected.append(p)
            used += est

        return selected

    def _mmr_select(
        self, candidates: List[ContextPacket], query_tokens: set,
    ) -> List[ContextPacket]:
        if len(candidates) <= 1:
            return candidates

        selected: List[ContextPacket] = [candidates[0]]
        remaining = candidates[1:]

        while remaining and len(selected) < len(candidates):
            best_score = -float("inf")
            best_idx = 0

            for i, p in enumerate(remaining):
                rel = p.relevance_score
                max_sim = 0.0
                p_tokens = _tokenize_chinese(p.content)
                for s in selected:
                    s_tokens = _tokenize_chinese(s.content)
                    if p_tokens and s_tokens:
                        sim = len(p_tokens & s_tokens) / len(p_tokens | s_tokens)
                        max_sim = max(max_sim, sim)

                mmr = self.config.mmr_lambda * rel - (1 - self.config.mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _structure(
        self, packets: List[ContextPacket], user_query: str,
    ) -> str:
        by_source: Dict[str, List[ContextPacket]] = {}
        for p in packets:
            by_source.setdefault(p.source, []).append(p)

        sys_text = "\n".join(p.content for p in by_source.get("system", []))
        task_text = f"用户问题：{user_query}"

        evidence_parts = []
        if "retrieval" in by_source:
            evidence_parts.extend(p.content for p in by_source["retrieval"])
        if "memory" in by_source:
            evidence_parts.append("【长期记忆】")
            evidence_parts.extend(p.content for p in by_source["memory"])

        history_text = "\n".join(p.content for p in by_source.get("history", []))

        messages = _GSSC_TEMPLATE.invoke({
            "system_instructions": sys_text or "（无特殊指令）",
            "current_task": task_text,
            "evidence_section": ("[参考证据]\n" + "\n\n".join(evidence_parts)) if evidence_parts else "",
            "history_section": ("[对话历史]\n" + history_text) if history_text else "",
            "output_instructions": "[输出要求]\n" + OUTPUT_INSTRUCTION,
        })
        return messages.to_string()

    def _compress(self, context: str) -> str:
        budget = self.config.available_tokens
        current = _count_tokens(context)

        if current <= budget:
            return context

        logger.warning(f"ContextBuilder: 上下文超预算 ({current} > {budget})，执行智能截断")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "。", "! ", "！", "? ", "？", ";", "；", " ", ""],
            length_function=_count_tokens,
            keep_separator=True,
        )
        chunks = splitter.split_text(context)
        return chunks[0] if chunks else (_compress_legacy(context, budget))

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            content = d.page_content or ""
            meta = d.metadata if isinstance(d.metadata, dict) else {}
            src = meta.get("source", "未知")
            section = meta.get("section", "")
            header = f"[片段{i}] 来源: {src}"
            if section:
                header += f" | 章节: {section}"
            parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)

    def _format_memories(self, memories: List[str]) -> str:
        return "\n".join(f"- {m}" for m in memories)


def _compress_legacy(context: str, budget: int) -> str:
    lines = context.split("\n")
    compressed: List[str] = []
    used = 0
    for line in lines:
        est = _count_tokens(line)
        if used + est > budget:
            break
        compressed.append(line)
        used += est
    return "\n".join(compressed)


_default_builder: Optional[ContextBuilder] = None


def get_context_builder(config: Optional[ContextConfig] = None) -> ContextBuilder:
    global _default_builder
    if _default_builder is None or config is not None:
        _default_builder = ContextBuilder(config=config)
    return _default_builder


def build_rag_context(
    user_query: str,
    retrieved_docs: List[Document],
    chat_history: Optional[List[Dict[str, str]]] = None,
    long_term_memories: Optional[List[str]] = None,
    system_instructions: Optional[str] = None,
) -> str:
    builder = get_context_builder()
    return builder.build(
        user_query=user_query,
        retrieved_docs=retrieved_docs,
        chat_history=chat_history,
        long_term_memories=long_term_memories,
        system_instructions=system_instructions,
    )