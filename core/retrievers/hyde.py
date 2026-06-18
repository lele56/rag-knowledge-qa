from dataclasses import dataclass
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompts import HYDE_PROMPT
from utils.logger import logger

_STATE = "_hyde_int_state_v1"


@dataclass
class _HyDEState:
    """HyDERetriever 内部状态容器，绕过 Pydantic 属性拦截。"""
    llm: Any = None
    base_retriever: Any = None
    include_original: bool = True
    chain: Any = None


class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embedding (HyDE) 检索器。

    直接操作 __dict__ 存内部依赖，完全绕开 Pydantic v1/v2 属性拦截。
    """

    def __init__(self, llm: BaseChatModel, base_retriever, include_original: bool = True):
        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        chain = prompt | llm | StrOutputParser()

        try:
            super().__init__()
        except Exception:
            try:
                super().__init__(callback_manager=None)
            except Exception:
                pass

        self.__dict__[_STATE] = _HyDEState(
            llm=llm,
            base_retriever=base_retriever,
            include_original=include_original,
            chain=chain,
        )

    @property
    def _s(self) -> _HyDEState:
        return self.__dict__[_STATE]

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        logger.info(f"HyDE generating for: {query[:50]}...")
        hypo = self._s.chain.invoke({"question": query})
        docs = self._s.base_retriever.invoke(hypo)
        if self._s.include_original:
            orig = self._s.base_retriever.invoke(query)
            seen = {d.page_content for d in docs}
            for d in orig:
                if d.page_content not in seen:
                    docs.append(d)
                    seen.add(d.page_content)
        return docs