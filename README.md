# RAG 知识库问答助手

> 基于 LangChain 的企业级知识库问答系统，支持混合检索、重排序、长期记忆与图增强。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-red.svg)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-blue.svg)](https://neo4j.com/)

## 特性

| 模块 | 能力 |
|------|------|
| **检索** | 两阶段检索（全库定位 → 文档聚焦）+ 混合检索（向量 MMR + BM25）+ CrossEncoder 重排序 |
| **策略** | Simple / MultiQuery 多查询变体 / HyDE 假设性文档嵌入，可切换 |
| **记忆** | 短期记忆（滑动窗口）+ 长期记忆（情景记忆 + 语义记忆 + 工作记忆） |
| **图增强** | Neo4j 知识图谱辅助推理（GraphRAG） |
| **Agent** | ReAct 模式，支持工具调用（文档聚焦、记忆召回、记忆保存） |
| **评估** | 内置检索评估框架：Recall@K、MRR、NDCG、Hit Rate、延迟统计 |
| **界面** | Gradio Web UI，支持流式对话、文档管理、记忆管理 |

## 架构

```
用户 → Gradio Web UI
         ↓
    QA Service (缓存 + 流式)
         ↓
    ReAct Agent ← 工具调用（文档聚焦 / 记忆召回）
         ↓
    RAG Chain
    ├── 检索器（两阶段混合检索）
    │   ├── Stage 1: 全库广搜 → 定位 Top-N 文档
    │   ├── Stage 2: 文档内聚焦搜索 → 合并去重
    │   └── CrossEncoder 重排序
    ├── 上下文构建（GSSC 策略）
    │   ├── Compact / Full / EvidenceOnly / MultiDoc
    │   └── 来源检测与过滤
    ├── Memory System
    │   ├── Short-term: 滑动窗口（最近 N 轮）
    │   └── Long-term: Episodic (Qdrant) + Semantic (Neo4j) + Working
    └── LLM 生成
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/lele56/rag-knowledge-qa.git
cd rag-knowledge-qa

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入你的凭证：
#   - Qdrant Cloud URL + API Key
#   - Neo4j AuraDB URI + 密码
#   - LLM API Key + Base URL
#   - 本地 Embedding 模型路径
```

### 3. 下载嵌入模型

```bash
# 下载 BGE-M3 嵌入模型（或其他 HuggingFace 嵌入模型）
mkdir -p models
cd models
git lfs install
git clone https://huggingface.co/BAAI/bge-m3
```

或者使用 HuggingFace 在线加载（在 `.env` 中设置 `EMBEDDING_MODEL_PATH=BAAI/bge-m3`）。

### 4. 初始化数据库（可选）

```bash
# Neo4j 图数据库初始化（需要 Neo4j 已配置）
python scripts/init_db.py
```

### 5. 导入文档

```bash
# 将你的 PDF/Markdown/TXT 文档放入 data/ 目录
# 然后运行：
python scripts/ingest_docs.py
```

### 6. 启动

```bash
python main.py          # 或 python web/gradio_app.py
# 打开 http://localhost:7860
```

## 检索策略

| 策略 | 说明 | 速度 | 适用场景 |
|------|------|------|----------|
| `simple` | 纯向量 + 两阶段聚焦 + 重排序 | ⚡ 快 | 日常使用，推荐 |
| `multi_query` | LLM 生成多个查询变体再检索 | 🐢 慢 | 查询歧义较大时 |
| `hyde` | 生成假设性文档再检索 | 🐢 最慢 | 精度要求极高时 |

在 `.env` 中切换：

```bash
RETRIEVAL_STRATEGY=simple  # simple | multi_query | hyde
```

## 评估

```bash
# 生成测试集（基于文档自动生成问答对）
python scripts/_generate_testset.py --testset data/test_question_ragas.json

# 检索评估
python scripts/_eval_now.py --mode retrieval

# 生成评估（需 LLM）
python scripts/_eval_now.py --mode gen

# 端到端评估
python scripts/_eval_now.py --mode full

# 指定策略 + 限制题目数 + 自定义路径
python scripts/_eval_now.py --mode retrieval --strategy multi_query --limit 10 \
    --testset data/test_question_ragas.json \
    --output data/test_question_ragas_results.json
```

示例输出（50 题，simple 策略）：

```
======================================================================
  检索评估汇总
======================================================================
  Recall@1:     80.00%
  Recall@3:     84.00%
  Recall@5:     84.00%
  Precision@5:  77.33%
  MRR:          0.8167
  NDCG@5:       0.8194
  Hit Rate:     84.00%
  Avg Latency:  9474ms
  P50/P95:      6111ms / 42478ms
```

## 项目结构

```
├── chains/              # LangChain 链（GraphRAG、来源检测）
├── config/              # 配置（pydantic-settings + 提示词模板）
├── core/                # 核心模块
│   ├── agent/           # ReAct Agent + 工具调用编排
│   ├── context/         # 上下文构建（GSSC 策略：Compact/Full/EvidenceOnly/MultiDoc）
│   ├── doc/             # 文档加载、切分、注册
│   ├── infrastructure/  # LLM / Embedding / 重排序 / 向量存储 / 图谱存储
│   ├── memory/          # 短期记忆（滑动窗口）+ 长期记忆（情景/语义/工作）+ 记忆管理器
│   │   └── long_term/   # 长期记忆子系统（episodic/semantic/working + 评分）
│   ├── retrievers/      # 检索器（混合/BM25/增强/HyDE/MultiQuery/过滤）+ 工厂
│   └── tools/           # Agent 工具（文档聚焦、记忆召回、保存）+ 注册器 + 管道
├── data/                # 文档数据 + 测试集 + 评估结果
├── docker-compose.yml   # Docker 编排（Qdrant + Neo4j + Redis）
├── evaluation/          # 评估框架（检索 + 生成 + 性能 + RAGAS + 合成测试集）
│   └── metrics/         # 指标计算（Recall/MRR/NDCG + Faithfulness/Relevance）
├── scripts/             # 脚本（导入文档、评估、生成测试集、重建索引）
├── services/            # 业务服务层（QA / 文档 / 速率限制 / 会话）
├── tests/               # 测试（单元 + 集成 + conftest fixtures）
│   ├── unit/            # 单元测试（agent/retrievers/context/tools/cache）
│   └── integration/     # 集成测试（QA pipeline）
├── utils/               # 工具（缓存、日志、Redis 客户端、设备检测、Token 计数）
├── web/                 # Gradio Web UI（聊天/上传/管理/事件处理）
├── .env.example         # 环境变量模板
├── Makefile             # 开发常用命令
└── main.py              # 入口（→ web/gradio_app.py）
```

## 技术栈

| 技术 | 用途 |
|------|------|
| LangChain 0.3+ | RAG 编排框架 |
| Qdrant Cloud | 向量数据库 |
| Neo4j AuraDB | 图数据库（GraphRAG） |
| Redis | 缓存 / 会话 / 速率限制 |
| Docker | 本地基础设施编排 |
| BGE-M3 / BGE-Reranker-v2 | 嵌入 + 重排序 |
| Gradio 4 | Web UI |
| OpenAI 兼容 API | LLM 推理 |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码检查
ruff check .

# 格式化
ruff format .
```

## License

MIT