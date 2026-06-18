# ============================================================
# 知识库问答助手 - Makefile (开发常用命令)
# 兼容: Git Bash / WSL / Linux / macOS
# ============================================================

.PHONY: help install dev test lint typecheck clean run

help:  ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## 安装基础依赖
	pip install -e .

dev:  ## 安装全部开发依赖
	pip install -e ".[dev,api]"

test:  ## 运行测试
	pytest tests/ -v --tb=short

test-cov:  ## 运行测试并生成覆盖率报告
	pytest tests/ -v --cov --cov-report=html --cov-report=term-missing

lint:  ## 代码检查 (ruff)
	ruff check core/ services/ chains/ config/ web/ utils/

typecheck:  ## 类型检查 (mypy)
	mypy core/ services/ --ignore-missing-imports

format:  ## 格式化代码
	ruff check --fix core/ services/ chains/ config/ web/ utils/
	ruff format core/ services/ chains/ config/ web/ utils/

clean:  ## 清理缓存文件
	@echo "清理 __pycache__ ..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "清理 .pyc 文件 ..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "清理缓存目录 ..."
	@rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage 2>/dev/null || true
	@echo "清理完成!"

run:  ## 启动 Gradio Web 界面
	python main.py