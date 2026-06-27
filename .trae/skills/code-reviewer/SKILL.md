---
name: "code-reviewer"
description: "审查本项目的 Python 代码，重点关注安全性、性能、可读性和项目规范。当用户要求代码审查、code review、检查代码质量，或在合并前审查变更时调用。"
---

# 代码审查 Skill

针对本项目（企业级 RAG 知识库问答助手）的代码审查，重点关注以下四个维度。

---

## 项目技术栈

| 维度 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 异步 | asyncio、LangChain async API |
| 配置 | pydantic-settings, .env |
| 数据库 | Qdrant（向量）、Neo4j（图）、Redis（缓存） |
| 检索 | 向量（MMR） + BM25 + CrossEncoder 重排 |
| Agent | ReAct (LangChain) |
| Web | Gradio |
| 测试 | pytest + pytest-asyncio + pytest-cov |
| 代码检查 | ruff + mypy |

---

## 一、安全性（优先级最高）

### 1.1 密钥与敏感信息
- **绝不允许** 硬编码 API Key、密码、Token 等密钥
- 环境变量必须通过 `config/settings.py` 的 `pydantic-settings` 管理，使用 `Field(alias="...")` 映射
- `.env` 文件已在 `.gitignore` 中，确保不提交
- 检查 `os.environ.get()` 和 `os.getenv()` 的调用是否应改用 `settings.XXX`

### 1.2 注入攻击
- **SQL/NoSQL 注入**：检查 Neo4j Cypher 查询是否使用参数化查询，而非字符串拼接
- **Prompt 注入**：用户输入拼接到 LLM prompt 时，是否有适当的转义或限制
- 检查 `Qdrant` 的 `Filter` 构造是否使用 `FieldCondition` 和 `MatchValue`，而非直接拼接查询字符串

### 1.3 输入验证
- 对外接口（Gradio handlers、API endpoints）是否对用户输入做了长度/类型校验
- 文件上传是否有大小限制和类型白名单
- `settings.py` 中的 `Field` 是否设置了合理的 `ge`/`le`/`min_length`/`max_length` 约束

### 1.4 依赖安全
- 检查 `requirements.txt` 或 `pyproject.toml` 中依赖是否有已知 CVE
- 是否使用了过时或有安全漏洞的包版本
- 检查是否有 `eval()`、`exec()`、`pickle.loads()` 等危险函数调用

### 1.5 数据安全
- 日志中是否打印了用户敏感信息（问题内容、文档内容）— 如果打印，建议脱敏或降级
- 缓存键是否包含敏感信息，Redis 是否配置了密码

---

## 二、性能

### 2.1 异步与并发
- I/O 密集型操作（LLM 调用、数据库查询、HTTP 请求）是否使用了 `async/await`
- 同步阻塞操作是否在 `asyncio.to_thread()` 中执行，避免阻塞事件循环
- `ThreadPoolExecutor` 使用是否合理，线程数是否可配置
- 检查是否有 `time.sleep()` 阻塞调用，应使用 `asyncio.sleep()`

### 2.2 数据库查询
- Qdrant 查询是否使用了合理的 `limit`，避免 `scroll` 全量加载
- Neo4j 查询是否有索引支持，避免全图扫描
- Redis 是否有合理的 TTL 和内存上限配置

### 2.3 缓存策略
- 检查是否可以利用 `utils/cache.py` 的 `AsyncTTLCache` 或 `AsyncRedisCache`
- LLM 响应是否可缓存（相同 prompt 的重复调用）
- Embedding 计算是否有缓存

### 2.4 Embedding 与模型推理
- Embedding 是否批量调用 `embed_documents()` 而非逐个 `embed_query()`
- 本地模型加载是否做了单例缓存（`get_embeddings()` 等工厂函数）
- GPU 推理是否合理，batch_size 是否可调

### 2.5 内存
- 大文件加载是否使用流式处理（`PyPDF` 分页读取）
- 是否存在循环引用导致内存泄漏
- `OrderedDict` 缓存是否有容量上限

---

## 三、可读性与可维护性

### 3.1 项目规范（必须遵守）
- 文件头必须有 `# -*- coding: utf-8 -*-` 和模块路径注释
- 模块级 docstring 使用 `"""..."""` 三引号，描述模块职责
- 日志使用 `from utils.logger import logger`，不使用 `print()` 或 `logging.getLogger()` 直接裸调
- 配置统一从 `from config.settings import settings` 获取，不在代码中直接读环境变量
- 类型标注：所有函数签名必须有完整的类型提示（`typing` 模块）

### 3.2 命名规范
- 类名：`PascalCase`（如 `HybridRetriever`、`QAService`）
- 函数/方法：`snake_case`（如 `_get_all_doc_ids`、`setup_logger`）
- 私有函数：前缀 `_`（如 `_normalize_pf`、`_should_use_graph`）
- 常量：`UPPER_SNAKE_CASE`（如 `MAX_CTX_CHARS`）
- 模块文件：`snake_case.py`（如 `vector_store.py`、`qa_service.py`）

### 3.3 函数设计
- 单一职责：一个函数只做一件事，超过 50 行需拆分
- 避免深层嵌套：if/for 嵌套不超过 3 层
- 避免魔法数字：提取为命名常量放在文件顶部
- 布尔参数：超过 2 个布尔参数建议用枚举或配置对象替代

### 3.4 注释与文档
- 复杂逻辑必须有注释解释"为什么这样做"，而非"做了什么"
- 公开 API 函数使用 docstring，包含参数说明和返回值
- TODO/FIXME/HACK 标记需要附上原因和日期
- 不写无意义的注释（如 `# 增加 1` 在 `i += 1` 旁边）

### 3.5 导入规范
- 标准库 → 第三方库 → 项目内模块，三组之间空行分隔
- 禁止 `from module import *`（通配符导入）
- 禁止循环导入

---

## 四、错误处理与健壮性

### 4.1 异常处理
- 禁止裸 `except:` 或 `except Exception: pass`（静默吞异常）
- 捕获异常后必须记录日志（`logger.error`/`logger.warning`）
- 外部服务调用（LLM、Qdrant、Neo4j、Redis）必须有 try/except 和降级策略
- 使用 `finally` 或上下文管理器释放资源（连接、文件句柄）

### 4.2 重试机制
- LLM 调用是否有重试配置（`settings.LLM.max_retries`）
- 重试是否使用指数退避（exponential backoff），而非固定间隔
- 检查 `settings.LLM.request_timeout` 是否在合理范围（默认 120s）

### 4.3 降级与容错
- 关键服务不可用时是否有 fallback（如 Redis 降级到内存缓存、Neo4j 降级到纯向量检索）
- 检查 `_should_use_graph` 这类可选增强失败时是否影响主流程

---

## 五、项目特定检查清单

### 检索模块 (`core/retrievers/`)
- 新增检索器是否继承 `BaseRetriever` 并实现 `_get_relevant_documents()`
- 是否在 `factory.py` 中注册
- 检索结果是否经过 `denoise_docs()` 降噪

### 配置 (`config/settings.py`)
- 新增配置项是否使用 `Field(alias="...")` 映射环境变量
- 是否有合理的默认值和校验（`ge`/`le`/`min_length`）
- 敏感字段是否有 `model_config = SettingsConfigDict(extra="ignore")`

### 服务层 (`services/`)
- 是否有速率限制检查（`check_rate_limit`）
- 缓存键设计是否合理（避免碰撞）
- 会话管理是否正确

### 测试 (`tests/`)
- 新增功能是否有对应的单元测试
- 测试是否使用 `pytest.mark.asyncio` 标记异步测试
- Mock 是否合理，不 Mock 自己项目内部的函数

### 脚本 (`scripts/`)
- 临时脚本是否以 `_` 前缀命名（`_xxx.py`）
- 是否使用 `if __name__ == "__main__":` 保护入口
- 不依赖项目运行时的脚本是否标注清楚

---

## 审查输出格式

审查完成后，按以下格式输出：

```
## 代码审查报告

### 严重问题 —— 必须修复
- [ ] [安全性] 问题描述 + 位置 + 修复建议
- [ ] [性能] ...

### 建议优化 —— 推荐修复
- [ ] [可读性] ...
- [ ] [性能] ...

### 正面反馈 —— 做得好的地方
- ✅ ...
```

对每个问题标注严重级别：🔴 严重 / 🟡 建议 / 🟢 小优化。