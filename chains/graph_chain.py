# chains/graph_chain.py
from langchain_neo4j import GraphCypherQAChain
from core.infrastructure.llm import get_llm
from core.infrastructure.graph_store import get_graph
from utils.logger import logger

_chain = None

def get_graph_chain():
    global _chain
    if _chain is None:
        _chain = GraphCypherQAChain.from_llm(
            llm=get_llm(),
            graph=get_graph(),
            verbose=True,
            allow_dangerous_requests=True
        )
        logger.info("Graph chain initialized")
    return _chain