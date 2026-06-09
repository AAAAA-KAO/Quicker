#!/usr/bin/env python3
"""rephrase_questions.py — 基于 LLM 对问题换一种说法，并重新提取 PICO 组件。

流程:
    1. 加载 original_question.json
    2. 调用 LLM 改写问题（换一种说法）
    3. 调用 QuestionDecomposition 对改写后的问题重新提取 PICO
    4. 用新提取的 PICO 替换原来的 PICO
    5. 保存到新的 JSON 文件

输入:  data/evaluate/retrieve/original_questions.json
输出:  data/evaluate/retrieve/refine_questions.json           (改写后的问题 + 原始 PICO)
       data/evaluate/retrieve/refine_questions_new_pico.json  (改写后的问题 + 新提取的 PICO)

用法:
    python src/evaluate/rephrase_questions.py [--num-variants K]

    默认 K=1（每个问题生成 1 个改写版本）。
    设置 K>1 时，每个问题生成 K 个不同的改写版本，均保留原始 question_id。

命令行参数:
    --num-variants, -k:  每个问题生成的改写版本数（默认 1）。

配置 (可选环境变量，均有默认值):
    DEEPSEEK_API_KEY   — API key
    DEEPSEEK_BASE_URL  — 默认 https://api.deepseek.com
    DEEPSEEK_MODEL     — 默认 deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# 路径常量 (以项目根目录为基准)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 将 src 目录加入 sys.path，以便导入 question_decomposition 模块
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from question_decomposition import QuestionDecomposition

INPUT_PATH = PROJECT_ROOT / "data" / "evaluate" / "retrieve" / "original_questions.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluate" / "retrieve"
OUTPUT_PATH = OUTPUT_DIR / "refine_questions.json"
OUTPUT_PATH_NEW_PICO = OUTPUT_DIR / "refine_questions_new_pico.json"

# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 LLM 改写临床问题并重新提取 PICO，支持批量生成多个变体。"
    )
    parser.add_argument(
        "--num-variants", "-k",
        type=int,
        default=1,
        help="每个问题生成的改写版本数（默认 1）。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical evidence retrieval expert. Your task is to rewrite a clinical question using different wording while preserving all PICO elements.

Core Rules:
1. **PICO elements MUST NOT change**:
   - P (Patient/Population)
   - I (Intervention)
   - C (Comparison)
   - O (Outcome)
   The user will provide the original question and its PICO decomposition. Rewrite based on this information.

2. **Rewriting Requirements**:
   - Use different natural language phrasing without changing the medical meaning of any PICO element
   - Maintain a professional, clinical tone
   - Adjust sentence structure (e.g., restructure the question, vary word order and syntax, use synonyms for non-PICO terms)
   - Keep approximately the same length as the original question
   - Do NOT add new clinical elements or remove existing ones
   - **CRITICAL: Keep the output in the SAME LANGUAGE as the original question (English). Do NOT translate to another language.**

3. **Output Format**:
   - Output ONLY the rewritten question text
   - No explanation, prefix, suffix, or markdown formatting
   - Just the rephrased question on one or two lines"""


def build_llm() -> ChatOpenAI:
    """构建 LLM 实例（用于改写问题，温度稍高以增加表达多样性）。"""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.7,
    )


def build_decomposer_llm() -> ChatOpenAI:
    """构建 LLM 实例（用于 PICO 分解，温度=0 以保证确定性输出）。"""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
    )


def load_original_questions(path: Path) -> list[dict]:
    """加载 original_question.json。"""
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}  (请先运行 extract_questions.py)")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"顶层应为 list，实际为 {type(data).__name__}")

    print(f"[OK] 已加载 {len(data)} 条原始问题: {path}")
    return data


def build_human_payload(
    original_question: str,
    pico: dict,
    variant: int | None = None,
    num_variants: int = 1,
) -> str:
    """构建发送给 LLM 的人类消息内容。

    Args:
        original_question: 原始临床问题。
        pico: 原始 PICO 分解。
        variant: 当前变体编号（1-based），None 表示不区分变体。
        num_variants: 变体总数。
    """
    pico_str = json.dumps(pico, ensure_ascii=False, indent=2)
    base = (
        f"Original Question:\n{original_question}\n\n"
        f"PICO Decomposition:\n{pico_str}\n\n"
        f"Please rewrite the above question using different wording while keeping all PICO elements unchanged. "
        f"IMPORTANT: Output the rewritten question in English (the same language as the original)."
    )
    if variant is not None and num_variants > 1:
        base += (
            f"\n\nThis is variant {variant} of {num_variants}. "
            f"Please produce a phrasing that is clearly different from what you would write for other variants "
            f"(e.g., vary sentence structure, word order, or use different synonyms)."
        )
    return base


def rephrase_single(
    llm: ChatOpenAI,
    question_id: str,
    original_question: str,
    pico: dict,
    retry: int = 2,
    variant: int | None = None,
    num_variants: int = 1,
) -> str:
    """调用 LLM 重写单个问题，失败时自动重试。"""
    human_payload = build_human_payload(
        original_question, pico,
        variant=variant, num_variants=num_variants,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_payload),
    ]

    last_error: Exception | None = None
    for attempt in range(1, retry + 2):  # 1 initial + retry
        try:
            response = llm.invoke(messages)
            result = response.content.strip()

            # 清理可能的 markdown 残留
            if result.startswith("```"):
                lines = result.splitlines()
                result = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            if not result:
                raise ValueError("LLM 返回了空内容")

            return result

        except Exception as exc:
            last_error = exc
            if attempt <= retry:
                wait = 2 ** attempt
                print(f"  [RETRY] question_id={question_id} 第 {attempt} 次失败，{wait}s 后重试: {exc}")
                time.sleep(wait)
            else:
                print(f"  [FAIL] question_id={question_id} 重试 {retry} 次后仍失败: {exc}")

    raise last_error  # type: ignore[misc]


def rephrase_all(
    llm: ChatOpenAI,
    original_data: list[dict],
    decomposer: QuestionDecomposition | None = None,
    num_variants: int = 1,
) -> tuple[list[dict], list[dict]]:
    """对每条记录调用 LLM 生成 k 个改写变体，并（可选）重新提取 PICO。

    Args:
        llm: 用于改写问题的 LLM 实例。
        original_data: 原始问题列表。
        decomposer: QuestionDecomposition 实例，传入则对改写后的问题重新提取 PICO。
        num_variants: 每个问题生成的改写版本数（默认 1）。

    Returns:
        (refined_data, refined_data_new_pico):
          - refined_data: 改写后的问题 + 原始 PICO
          - refined_data_new_pico: 改写后的问题 + 新提取的 PICO
    """
    refined_data: list[dict] = []
    refined_data_new_pico: list[dict] = []
    total = len(original_data)

    for idx, entry in enumerate(original_data, start=1):
        qid = entry["question_id"]
        original_q = entry["question"]
        original_pico = entry.get("pico", {})

        print(f"\n[{idx}/{total}] 正在处理 question_id={qid} ...")
        print(f"  原始问题: {original_q[:120]}{'...' if len(original_q) > 120 else ''}")

        for v in range(1, num_variants + 1):
            variant_qid = qid
            variant_label = f" [变体 {v}/{num_variants}]" if num_variants > 1 else ""

            # Step 1: 改写问题
            try:
                refined_q = rephrase_single(
                    llm, variant_qid, original_q, original_pico,
                    variant=v if num_variants > 1 else None,
                    num_variants=num_variants,
                )
            except Exception:
                refined_q = original_q
                print(f"  [FALLBACK] 改写失败{variant_label}，使用原始问题作为兜底")

            print(f"  改写结果{variant_label}: {refined_q[:120]}{'...' if len(refined_q) > 120 else ''}")

            # 保存原始 PICO 版本
            refined_data.append({
                "question_id": variant_qid,
                "question": refined_q,
                "pico": original_pico,
            })

            # Step 2: 重新提取 PICO（如果提供了 decomposer）
            new_pico = original_pico  # 默认使用原始 PICO 作为兜底
            if decomposer is not None:
                try:
                    new_pico = decomposer.decompose(refined_q)
                    print(f"  新 PICO{variant_label} 提取成功: P={new_pico.get('P', '')[:60]}...")
                except Exception as exc:
                    new_pico = original_pico
                    print(f"  [FALLBACK] PICO 重新提取失败{variant_label}，使用原始 PICO 作为兜底: {exc}")

            refined_data_new_pico.append({
                "question_id": variant_qid,
                "question": refined_q,
                "pico": new_pico,
            })

            # 变体之间稍作延迟，避免触发 rate limit
            if v < num_variants:
                time.sleep(0.2)

        # 问题之间稍作延迟
        if idx < total:
            time.sleep(0.3)

    return refined_data, refined_data_new_pico


def save_output(data: list[dict], path: Path) -> None:
    """保存结果到 JSON 文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 结果已写入: {path} ({len(data)} 条记录)")


def main() -> None:
    args = parse_args()
    num_variants: int = args.num_variants

    print("=" * 60)
    print("LLM 问题改写 + PICO 重新提取")
    print(f"模型:        {MODEL}")
    print(f"每问题变体数: {num_variants}")
    print(f"输入:        {INPUT_PATH}")
    print(f"输出1:       {OUTPUT_PATH}           (改写问题 + 原始 PICO)")
    print(f"输出2:       {OUTPUT_PATH_NEW_PICO}  (改写问题 + 新提取 PICO)")
    if num_variants > 1:
        print(f"  (所有变体均保留原始 question_id)")
    print("=" * 60)

    original_data = load_original_questions(INPUT_PATH)

    # 构建 LLM 实例
    llm = build_llm()
    decomposer_llm = build_decomposer_llm()
    decomposer = QuestionDecomposition(decomposer_llm)

    # 改写问题 + 重新提取 PICO（每个问题生成 num_variants 个变体）
    refined_data, refined_data_new_pico = rephrase_all(
        llm, original_data,
        decomposer=decomposer,
        num_variants=num_variants,
    )

    # 保存两个版本
    save_output(refined_data, OUTPUT_PATH)
    save_output(refined_data_new_pico, OUTPUT_PATH_NEW_PICO)

    # 统计 PICO 变化
    pico_changed = sum(
        1 for a, b in zip(refined_data, refined_data_new_pico)
        if a["pico"] != b["pico"]
    )
    print(f"\n[统计] 条目总数: {len(refined_data)} (原始问题 {len(original_data)} × {num_variants} 变体)")
    print(f"[统计] PICO 发生变化的条目: {pico_changed}/{len(refined_data)}")
    print("\n完成!")


if __name__ == "__main__":
    main()
