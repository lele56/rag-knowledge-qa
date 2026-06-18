# evaluation/testset.py
"""测试集管理：加载、验证、生成测试用例。"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """单个测试用例"""
    question: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_answer: str = ""
    category: str = "retrieval"
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_keywords": self.expected_keywords,
            "expected_answer": self.expected_answer,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(
            question=d.get("question", ""),
            expected_keywords=d.get("expected_keywords", []) or d.get("expected_sources", []),
            expected_answer=d.get("expected_answer", "") or d.get("expected_answer_keywords", ""),
            category=d.get("category", "retrieval"),
            difficulty=d.get("difficulty", "medium"),
            tags=d.get("tags", []),
        )


class TestSet:
    """测试用例集合"""

    def __init__(self, cases: Optional[List[TestCase]] = None):
        self.cases: List[TestCase] = cases or []

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def add(self, case: TestCase) -> None:
        self.cases.append(case)

    def filter_by_category(self, category: str) -> "TestSet":
        return TestSet([c for c in self.cases if c.category == category])

    def filter_by_difficulty(self, difficulty: str) -> "TestSet":
        return TestSet([c for c in self.cases if c.difficulty == difficulty])

    def sample(self, n: int) -> "TestSet":
        return TestSet(random.sample(self.cases, min(n, len(self.cases))))

    # ---------- 序列化 ----------

    @classmethod
    def from_json(cls, path: Path) -> "TestSet":
        """从 JSON 文件加载测试集"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cases = [TestCase.from_dict(item) for item in data]
        elif isinstance(data, dict) and "cases" in data:
            cases = [TestCase.from_dict(item) for item in data["cases"]]
        else:
            raise ValueError(f"无法解析测试集格式: {path}")
        return cls(cases)

    def to_json(self, path: Path) -> None:
        """保存测试集到 JSON 文件"""
        data = [c.to_dict() for c in self.cases]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- 自动生成 ----------

    @classmethod
    def generate_from_documents(
        cls,
        doc_names: List[str],
        questions_per_doc: int = 3,
    ) -> "TestSet":
        """根据文档名自动生成测试用例模板（需人工补充 expected_keywords）"""
        cases = []
        templates = [
            "{doc} 的主要内容是什么？",
            "{doc} 的核心观点有哪些？",
            "{doc} 中提到了哪些关键概念？",
            "{doc} 的结论是什么？",
            "{doc} 使用了什么方法？",
        ]
        for doc in doc_names:
            doc_stem = Path(doc).stem if "." in doc else doc
            for i in range(min(questions_per_doc, len(templates))):
                q = templates[i].format(doc=doc_stem)
                cases.append(TestCase(
                    question=q,
                    expected_keywords=[doc_stem],
                    category="retrieval",
                    difficulty="easy",
                    tags=[doc_stem, "auto-generated"],
                ))
        return cls(cases)

    def validate(self) -> List[str]:
        """验证测试集，返回错误列表"""
        errors = []
        for i, c in enumerate(self.cases):
            if not c.question.strip():
                errors.append(f"第 {i+1} 题: question 为空")
            if not c.expected_keywords:
                errors.append(f"第 {i+1} 题: expected_keywords 为空")
        return errors

    def stats(self) -> Dict:
        """测试集统计信息"""
        by_cat = {}
        by_diff = {}
        for c in self.cases:
            by_cat[c.category] = by_cat.get(c.category, 0) + 1
            by_diff[c.difficulty] = by_diff.get(c.difficulty, 0) + 1
        return {
            "total": len(self.cases),
            "by_category": by_cat,
            "by_difficulty": by_diff,
        }