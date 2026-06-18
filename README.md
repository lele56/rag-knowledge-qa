# LangChain RAG + Graph QA 系统

企业级个人知识库问答助手，支持：
- 混合检索 (Qdrant)
- 重排序 (Reranker)
- HyDE / MultiQuery 检索策略
- Neo4j 图增强 (GraphRAG)
- 多种记忆类型 (buffer/window/summary/vectorstore/entity)
- Gradio Web 界面

## 快速开始
1. `pip install -r requirements.txt`
2. 复制 `.env.example` 为 `.env`，填入你的云服务凭证
3. 将文档放入 `data/` 目录
4. `python scripts/ingest_docs.py`
5. `python main.py`