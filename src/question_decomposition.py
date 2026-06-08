"""Decompose a clinical question into PICO components via LLM.

功能：
    接收一个临床问题字符串，通过 LangChain 调用 DeepSeek LLM 将问题分解为
    PICO 组件（Population / Intervention / Comparison / Outcome），PICO
    模板遵循 data/mimic-cpg/template.json 中 "pico" 字段的结构。

    P：人群，字符串。
    I：干预，字符串。
    C：对照，字符串列表。
    O：结局，字典 —— key 为 C 中的元素，value 为对应的结局列表。

输入：
    --question：待分解的临床问题字符串（必填）。

输出：
    终端打印提取的 PICO 组件，不写入文件。

在项目根目录运行：
    conda run -n quicker python src/question_decomposition.py \\
        --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?"

命令行参数：
    --question, -q: 待分解的临床问题（必填）。
    --env-file: 环境变量文件路径，默认 .env。
    --log-file: 日志文件路径；不传则写入 logs/question_decomposition.log。
    --model: LLM 模型名，默认从 DEEPSEEK_MODEL 环境变量读取，回退为 deepseek-v4-flash。
    --api-key: DeepSeek API key；不传则从环境变量 DEEPSEEK_API_KEY 读取。
    --base-url: DeepSeek base URL；不传则从环境变量 DEEPSEEK_BASE_URL 读取，回退为 https://api.deepseek.com。
    --temperature: LLM 温度，默认 0。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.logging import log_step, setup_logging


SYSTEM_PROMPT = (
    "You are a clinical evidence-based medicine assistant. "
    "Your task is to decompose a clinical question into PICO components. "
    "Return only valid JSON. Do not include markdown fences or extra text."
)

PICO_SCHEMA: dict[str, Any] = {
    "P": "string — population / patient group",
    "I": "string — intervention / exposure / index test",
    "C": ["list of comparator or control strings"],
    "O": {
        "comparator/control string from C": [
            "list of clinically relevant outcome strings for this comparator"
        ]
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose a clinical question into PICO components via LLM."
    )
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="Clinical question to decompose.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: .env).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: logs/question_decomposition.log).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (defaults to DEEPSEEK_MODEL env var or deepseek-v4-flash).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key (defaults to DEEPSEEK_API_KEY env var).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="DeepSeek base URL (defaults to DEEPSEEK_BASE_URL env var or https://api.deepseek.com).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0).",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class QuestionDecomposition:
    """Decompose clinical questions into PICO components using an LLM."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def decompose(self, question: str) -> dict[str, Any]:
        system = SystemMessage(content=SYSTEM_PROMPT)

        human_content = json.dumps(
            {
                "task": "Decompose the clinical question into PICO components.",
                "schema": {"pico": PICO_SCHEMA},
                "constraints": [
                    "P and I must be strings (single string each, empty if not applicable).",
                    "C must be a list of comparator/control strings (empty list if none).",
                    "O must be an object. Its keys must be drawn from the C list.",
                    "Use English medical terminology.",
                ],
                "question": question,
            },
            ensure_ascii=False,
        )
        human = HumanMessage(content=human_content)

        response = self._llm.invoke([system, human])
        raw = str(response.content).strip()
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        if raw.startswith("```"):
            raw = raw.lstrip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`").strip()

        import re
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM response does not contain valid JSON: {raw[:300]}")

        payload = json.loads(match.group(0))
        pico = payload.get("pico", {}) if isinstance(payload, dict) else {}

        if not isinstance(pico, dict):
            raise ValueError(f"pico field is not a dict: {pico}")

        c_value = pico.get("C", [])
        if isinstance(c_value, str):
            c_value = [c_value] if c_value.strip() else []
        if not isinstance(c_value, list):
            c_value = []
        c_value = [str(item).strip() for item in c_value if str(item).strip()]

        o_value = pico.get("O", {})
        if not isinstance(o_value, dict):
            o_value = {}

        normalized_o: dict[str, list[str]] = {}
        for key, value in o_value.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if isinstance(value, list):
                normalized_o[key_text] = [str(item).strip() for item in value if str(item).strip()]
            elif value:
                normalized_o[key_text] = [str(value).strip()]
            else:
                normalized_o[key_text] = []

        return {
            "P": str(pico.get("P", "") or "").strip(),
            "I": str(pico.get("I", "") or "").strip(),
            "C": c_value,
            "O": normalized_o,
        }


def build_llm(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def print_pico(pico: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("PICO Decomposition Result")
    print("=" * 60)
    print(f"\n  P (Population):     {pico['P'] or '(empty)'}")
    print(f"  I (Intervention):   {pico['I'] or '(empty)'}")

    c_list = pico.get("C", [])
    if c_list:
        print(f"\n  C (Comparators):")
        for item in c_list:
            print(f"    - {item}")

    o_dict = pico.get("O", {})
    if o_dict:
        print(f"\n  O (Outcomes):")
        for comparator, outcomes in o_dict.items():
            outcomes_str = ", ".join(outcomes) if outcomes else "(empty)"
            print(f"    [{comparator}]: {outcomes_str}")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    logger = setup_logging("question_decomposition", log_file=args.log_file)

    log_step(logger, "加载环境变量")
    load_env_file(args.env_file)

    model = args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Set it in .env, pass --api-key, or set the environment variable."
        )

    log_step(logger, f"初始化 LLM: model={model} base_url={base_url}")
    llm = build_llm(model=model, api_key=api_key, base_url=base_url, temperature=args.temperature)

    decomposer = QuestionDecomposition(llm)

    log_step(logger, f"分解临床问题: {args.question}")
    try:
        pico = decomposer.decompose(args.question)
    except Exception as exc:
        logger.exception("PICO decomposition failed: %s", exc)
        raise

    print_pico(pico)
    logger.debug("PICO result: %s", json.dumps(pico, ensure_ascii=False, indent=2))
    log_step(logger, "完成")


if __name__ == "__main__":
    main()
