"""测试配置 & fixtures"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置测试环境变量（必须在任何模块导入之前）
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-unit-testing")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """创建 session 级别的事件循环，供所有 async 测试共享"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()