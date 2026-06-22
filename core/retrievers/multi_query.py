# core/retrievers/multi_query.py
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from core.infrastructure.llm import get_llm
from config.settings import settings
from config.prompts import MULTI_QUERY_PROMPT_FAST


def get_multi_query_retriever(base_retriever, query_count: int = 3):
    kwargs = dict(
        retriever=base_retriever,
        llm=get_llm(),
        include_original=True,
    )
    if query_count <= 2:
        kwargs["prompt"] = PromptTemplate(
            input_variables=["question"],
            template=MULTI_QUERY_PROMPT_FAST,
        )
    return MultiQueryRetriever.from_llm(**kwargs)