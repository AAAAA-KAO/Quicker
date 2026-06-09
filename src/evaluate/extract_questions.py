#!/usr/bin/env python3
"""extract_questions.py — 从 knowledge base 中提取 question_id / question / pico 字段。

输入:  results/mimic_cpg_knowledge_base.json
输出:  data/evaluate/retrieve/original_question.json

用法:
    python src/evaluate/extract_questions.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量 (以项目根目录为基准)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "results" / "mimic_cpg_knowledge_base.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluate" / "retrieve"
OUTPUT_PATH = OUTPUT_DIR / "original_question.json"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录已就绪: {OUTPUT_DIR}")


def load_knowledge_base(path: Path) -> list[dict]:
    """加载 knowledge base JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"knowledge base 文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"knowledge base 顶层应为 list，实际为 {type(data).__name__}")

    print(f"[OK] 已加载 {len(data)} 条记录: {path}")
    return data


def extract_questions(data: list[dict]) -> list[dict]:
    """从每条记录中提取 question_id / question / pico，组成新字典列表。"""
    extracted: list[dict] = []
    skipped: int = 0

    for entry in data:
        qid = entry.get("question_id")
        question = entry.get("question")
        pico = entry.get("pico")

        if not qid or not question:
            print(f"[WARN] 跳过缺少 question_id 或 question 的记录: {qid=}, question={str(question)[:80]}")
            skipped += 1
            continue

        extracted.append({
            "question_id": qid,
            "question": question,
            "pico": pico or {},
        })

    print(f"[OK] 成功提取 {len(extracted)} 条，跳过 {skipped} 条")
    return extracted


def save_output(data: list[dict], path: Path) -> None:
    """保存提取结果到 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 结果已写入: {path} ({len(data)} 条记录)")


def main() -> None:
    ensure_output_dir()
    kb = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    extracted = extract_questions(kb)
    save_output(extracted, OUTPUT_PATH)


if __name__ == "__main__":
    main()
