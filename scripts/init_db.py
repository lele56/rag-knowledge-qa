import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.graph_store import get_graph
from utils.logger import logger

def main():
    graph = get_graph()
    try:
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        logger.info("Neo4j constraints ready")
    except Exception as e:
        logger.warning(f"Constraint creation skipped: {e}")

if __name__ == "__main__":
    main()