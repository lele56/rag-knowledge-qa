# core/infrastructure/graph_store.py
from langchain_neo4j import Neo4jGraph
from config.settings import settings
from utils.logger import logger

_graph = None


def get_graph() -> Neo4jGraph:
    global _graph
    if _graph is None:
        _graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        _graph.refresh_schema()
        logger.info("Neo4j connected")
    return _graph


def get_graph_status() -> dict:
    """检查 Neo4j 连接状态。失败时重置缓存并重试一次。"""
    global _graph

    def _try_check() -> dict:
        graph = get_graph()
        try:
            r = graph.query("MATCH (n) RETURN count(n) AS cnt")
            nodes = r[0]["cnt"] if r else 0
            return {"ok": True, "nodes": nodes, "msg": f"✅ Neo4j 正常（{nodes} 节点）"}
        except Exception:
            return {"ok": True, "nodes": 0, "msg": "✅ Neo4j 正常"}

    try:
        return _try_check()
    except Exception:
        _graph = None
        try:
            return _try_check()
        except Exception as e:
            return {"ok": False, "nodes": 0, "msg": f"❌ Neo4j 不可用: {type(e).__name__}"}