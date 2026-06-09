#!/usr/bin/env python3
"""extract_pico_questions.py — 从 Q2CRBench PICO_Information.json 提取问题与 PICO 元素。

输入:  Q2CRBench-3/2021 ACR RA/PICO_Information.json
输出:  data/evaluate/retrieve/out_questions.json

用法:
    python src/evaluate/extract_pico_questions.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量 (以项目根目录为基准)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PICO_INFORMATION_PATH = PROJECT_ROOT / "Q2CRBench-3" / "2021 ACR RA" / "PICO_Information.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluate" / "retrieve"
OUTPUT_PATH = OUTPUT_DIR / "out_questions.json"

# question_id 前缀，与数据来源对应
QUESTION_ID_PREFIX = "ra"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录已就绪: {OUTPUT_DIR}")


def load_pico_information(path: Path) -> list[dict]:
    """加载 PICO_Information.json 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"PICO_Information 文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"PICO_Information 顶层应为 list，实际为 {type(data).__name__}")

    print(f"[OK] 已加载 {len(data)} 条记录: {path}")
    return data


def extract_questions(data: list[dict]) -> list[dict]:
    """将 Q2CRBench 格式转换为统一的问题-PICO 格式。"""
    extracted: list[dict] = []
    skipped: int = 0

    for entry in data:
        index = entry.get("Index")
        question = entry.get("Question")
        p_value = entry.get("P")
        i_value = entry.get("I")
        c_value = entry.get("C", [])
        o_value = entry.get("O")

        if not index or not question:
            print(f"[WARN] 跳过缺少 Index 或 Question 的记录: {index=}")
            skipped += 1
            continue

        question_id = f"{QUESTION_ID_PREFIX}-{index.lower()}"

        pico: dict = {
            "P": p_value or "",
            "I": i_value or "",
            "C": c_value if isinstance(c_value, list) else [],
            "O": o_value if isinstance(o_value, dict) else {},
        }

        extracted.append({
            "question_id": question_id,
            "question": question,
            "pico": pico,
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
    raw = load_pico_information(PICO_INFORMATION_PATH)
    extracted = extract_questions(raw)
    save_output(extracted, OUTPUT_PATH)


if __name__ == "__main__":
    main()
