# evaluation/metrics/generation.py
"""生成质量指标（LLM 裁判打分 + 关键词匹配）

三个 LLM 裁判维度：
  - Faithfulness（忠实度）：答案是否完全基于检索到的上下文
  - Answer Relevance（答案相关性）：答案是否直接回答了问题
  - Context Relevance（上下文相关性）：检索到的上下文是否与问题相关

一个轻量维度（无需 LLM）：
  - Keyword Recall / F1：答案中是否覆盖了预期关键词
"""
from .types import GenerationResult
from config.prompts import FAITHFULNESS_PROMPT, ANSWER_RELEVANCE_PROMPT, CONTEXT_RELEVANCE_PROMPT


class GenerationMetrics:
    """生成质量指标：通过 LLM 裁判打分"""

    FAITHFULNESS_PROMPT = FAITHFULNESS_PROMPT
    ANSWER_RELEVANCE_PROMPT = ANSWER_RELEVANCE_PROMPT
    CONTEXT_RELEVANCE_PROMPT = CONTEXT_RELEVANCE_PROMPT

    @staticmethod
    def _llm_score(llm, prompt: str) -> float:
        """用 LLM 打分，返回 0-1 之间的归一化分数"""
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            text = (response.content if hasattr(response, "content") else str(response)).strip()
            for ch in text:
                if ch.isdigit():
                    return int(ch) / 5.0
            return 0.5
        except Exception as e:
            logger.debug(f"生成质量评分解析失败: {e}")
            return 0.5

    @classmethod
    def faithfulness(cls, llm, context: str, answer: str) -> float:
        """评估答案忠实度"""
        prompt = cls.FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:2000])
        return cls._llm_score(llm, prompt)

    @classmethod
    def answer_relevance(cls, llm, question: str, answer: str) -> float:
        """评估答案相关性"""
        prompt = cls.ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer[:2000])
        return cls._llm_score(llm, prompt)

    @classmethod
    def context_relevance(cls, llm, question: str, context: str) -> float:
        """评估上下文相关性"""
        prompt = cls.CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context[:3000])
        return cls._llm_score(llm, prompt)

    @classmethod
    def evaluate(
        cls, llm, question: str, answer: str, context: str, latency_ms: float = 0.0, tokens: int = 0
    ) -> GenerationResult:
        """一次性计算所有生成指标"""
        return GenerationResult(
            question=question,
            answer=answer,
            faithfulness=cls.faithfulness(llm, context, answer),
            answer_relevance=cls.answer_relevance(llm, question, answer),
            context_relevance=cls.context_relevance(llm, question, context),
            latency_ms=latency_ms,
            tokens_used=tokens,
        )

    # ---------- 轻量级关键词匹配（无需 LLM） ----------

    @staticmethod
    def keyword_recall(answer: str, keywords: list[str]) -> float:
        """关键词召回率：预期关键词中有多少出现在答案中。

        Args:
            answer: 生成的答案文本
            keywords: 预期关键词列表

        Returns:
            0.0 ~ 1.0 的召回率
        """
        if not keywords:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return hits / len(keywords)

    @staticmethod
    def keyword_f1(answer: str, keywords: list[str]) -> float:
        """关键词 F1：综合召回和精确（避免答案堆砌关键词刷分）。

        精确率用答案中实际命中的关键词数 / 答案总词数（粗略估计）。
        """
        if not keywords:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        recall = hits / len(keywords)
        precision = hits / max(len(answer_lower.split()), 1)
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    @classmethod
    def evaluate_keywords(
        cls, question: str, answer: str, keywords: list[str],
        latency_ms: float = 0.0, tokens: int = 0,
    ) -> GenerationResult:
        """轻量级关键词评估（无需 LLM，仅基于关键词匹配）"""
        return GenerationResult(
            question=question,
            answer=answer,
            keyword_recall=cls.keyword_recall(answer, keywords),
            keyword_f1=cls.keyword_f1(answer, keywords),
            latency_ms=latency_ms,
            tokens_used=tokens,
        )