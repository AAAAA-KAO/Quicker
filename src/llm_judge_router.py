"""Route clinical QA retrieval results with an LLM-as-a-judge.

功能：
    接收临床问题、PICO 结构和 hybrid_retrieval.py 返回/导出的检索问答对，
    使用 LangChain 调用 LLM 从 3 个维度判断“检索到的问答对能否回答临床问题”：
        1. 检索匹配强度：候选问答的问题、疾病、主题、检索分数是否与临床问题匹配。
        2. 候选答案一致性：候选答案之间是否支持同一临床结论，是否存在冲突或只间接相关。
        3. PICO 覆盖度：候选问答是否覆盖输入 PICO 中的人群、干预、对照和结局。

输入：
    --question 输入临床问题字符串。
    --pico-json 或 --pico-file 输入 PICO。结构与 src/hybrid_retrieval.py 一致：
        P: 人群，字符串。
        I: 干预，字符串。
        C: 对照，列表。
        O: 结局，字典；key 为 C 中的元素，value 为结局列表。
    --retrieval-json 或 --retrieval-file 输入检索问答对，支持以下结构：
        1. hybrid_retrieval.hybrid_retrieve() 风格结果：
           {"dense": [SearchResult...], "sparse": [...], "hybrid": [...]}
           默认读取 "hybrid"，可通过 --retrieval-method 指定 dense/sparse/hybrid。
        2. SearchResult 字典列表：
           [{"index": 0, "score": 0.82, "record": {"question": ..., "answer": ...}}]
        3. 知识库 record 字典列表：
           [{"question_id": ..., "question": ..., "answer": ..., "disease": ..., "pico": ...}]
           record 字段来自 results/mimic_cpg_knowledge_base.json，常见字段包括
           question_id、question、answer、disease、topic、pico、source、synonyms、search_text。

输出：
    不保存文件。终端打印结构化 JSON：
        {
          "判断": "yes/no",
          "理由": "最终判断理由",
          "维度理由": {
            "检索匹配强度": "...",
            "候选答案一致性": "...",
            "PICO覆盖度": "..."
          },
          "维度评级": {
            "检索匹配强度": "强/中/弱",
            "候选答案一致性": "一致/部分一致/冲突/不足",
            "PICO覆盖度": "完整/部分/不足"
          },
          "依据候选排名": [1, 2]
        }

在项目根目录运行：
    conda run -n quicker python src/llm_judge_router.py \\
        --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?" \\
        --pico-json '{"P":"pediatric patients with suspected appendicitis","I":"clinical scores alone","C":["imaging or laboratory-assisted diagnosis"],"O":{"imaging or laboratory-assisted diagnosis":["diagnostic accuracy","missed appendicitis"]}}' \\
        --retrieval-file results/retrieval_results.json

命令行参数：
    --question, -q: 待判断的临床问题，必填。
    --pico-json: PICO JSON 字符串；可直接传 PICO，也可传包含 "pico" 字段的字典。
    --pico-file: PICO JSON 文件路径；可为 PICO 字典、含 "pico" 字段的字典，
        或 template.json 这类列表文件。
    --retrieval-json: 检索结果 JSON 字符串。
    --retrieval-file: 检索结果 JSON 文件路径。
    --retrieval-method: 当输入为 {"dense","sparse","hybrid"} 结构时选择使用哪一路，
        默认 hybrid。
    --max-candidates: 最多交给 LLM judge 的候选问答数，默认 5。
    --max-answer-chars: 每条候选答案最多保留字符数，默认 4000，用于控制上下文长度。
    --env-file: 环境变量文件路径，默认 .env。
    --log-file: 日志文件路径；不传则写入 logs/llm_judge_router.log。
    --model: LLM 模型名，默认从 DEEPSEEK_MODEL 环境变量读取，回退为 deepseek-v4-flash。
    --api-key: DeepSeek API key；不传则从 DEEPSEEK_API_KEY 环境变量读取。
    --base-url: DeepSeek base URL；不传则从 DEEPSEEK_BASE_URL 读取，
        回退为 https://api.deepseek.com。
    --temperature: LLM 温度，默认 0。
    --max-concurrency: LangChain batch 最大并发数，默认 4。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from hybrid_retrieval import normalize_pico
from utils.logging import log_step, setup_logging


DIMENSION_KEYS = ("检索匹配强度", "候选答案一致性", "PICO覆盖度")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class CandidateQAPair:
    """A normalized candidate QA pair passed to the LLM judge."""

    rank: int
    index: int | None
    score: float | None
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use an LLM judge to decide whether retrieved QA pairs can answer a clinical question."
    )
    parser.add_argument("--question", "-q", required=True, help="Clinical question to judge.")

    pico_group = parser.add_mutually_exclusive_group()
    pico_group.add_argument("--pico-json", default="", help="PICO JSON string.")
    pico_group.add_argument("--pico-file", type=Path, default=None, help="Path to a PICO JSON file.")

    retrieval_group = parser.add_mutually_exclusive_group(required=True)
    retrieval_group.add_argument("--retrieval-json", default="", help="Retrieval results JSON string.")
    retrieval_group.add_argument("--retrieval-file", type=Path, default=None, help="Path to retrieval results JSON.")

    parser.add_argument(
        "--retrieval-method",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Which result list to judge when retrieval input contains dense/sparse/hybrid keys.",
    )
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
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


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pico_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.pico_json:
        return normalize_pico(json.loads(args.pico_json))
    if args.pico_file:
        return normalize_pico(load_json_file(args.pico_file))
    return normalize_pico({})


def load_retrieval_payload(args: argparse.Namespace) -> Any:
    if args.retrieval_json:
        return json.loads(args.retrieval_json)
    if args.retrieval_file:
        return load_json_file(args.retrieval_file)
    raise ValueError("Either --retrieval-json or --retrieval-file is required.")


def coerce_to_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        attrs = {name: getattr(value, name) for name in ("index", "score", "record") if hasattr(value, name)}
        if attrs:
            return attrs
        raise ValueError(f"Candidate item must be a dict or SearchResult-like object, got {type(value)!r}.")
    return dict(value)


def select_retrieval_items(payload: Any, retrieval_method: str) -> list[Any]:
    if isinstance(payload, dict):
        if retrieval_method in payload:
            selected = payload[retrieval_method]
        elif "results" in payload:
            selected = payload["results"]
        elif "retrieval_results" in payload:
            selected = payload["retrieval_results"]
        elif "record" in payload or "question" in payload or "answer" in payload:
            selected = [payload]
        else:
            available = ", ".join(sorted(str(key) for key in payload.keys()))
            raise ValueError(
                f"Cannot find retrieval list. Expected '{retrieval_method}', 'results', "
                f"'retrieval_results', or a single record. Available keys: {available}"
            )
    elif isinstance(payload, list):
        selected = payload
    else:
        raise ValueError(f"Retrieval payload must be a dict or list, got {type(payload)!r}.")

    if not isinstance(selected, list):
        raise ValueError(f"Selected retrieval payload must be a list, got {type(selected)!r}.")
    return selected


def normalize_retrieval_results(
    payload: Any,
    retrieval_method: str = "hybrid",
    max_candidates: int = 5,
) -> list[CandidateQAPair]:
    items = select_retrieval_items(payload, retrieval_method)
    candidates: list[CandidateQAPair] = []

    for rank, raw_item in enumerate(items[:max_candidates], start=1):
        item = coerce_to_plain_dict(raw_item)
        if isinstance(item.get("record"), dict):
            record = dict(item["record"])
            index = parse_optional_int(item.get("index"))
            score = parse_optional_float(item.get("score"))
        else:
            record = dict(item)
            index = parse_optional_int(record.pop("index", None))
            score = parse_optional_float(record.pop("score", None))

        candidates.append(CandidateQAPair(rank=rank, index=index, score=score, record=record))

    return candidates


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate_text(value: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return value
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars].rstrip() + "...[truncated]"
    if isinstance(value, list):
        return [truncate_text(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_text(item, max_chars) for key, item in value.items()}
    return value


def compact_candidate(candidate: CandidateQAPair, max_answer_chars: int) -> dict[str, Any]:
    record = candidate.record
    return {
        "rank": candidate.rank,
        "index": candidate.index,
        "retrieval_score": candidate.score,
        "question_id": record.get("question_id", ""),
        "question": record.get("question", ""),
        "answer": truncate_text(record.get("answer", ""), max_answer_chars),
        "disease": record.get("disease", ""),
        "topic": record.get("topic", ""),
        "pico": record.get("pico", {}),
        "source": record.get("source", {}),
    }


def build_judge_messages(
    question: str,
    pico: dict[str, Any],
    candidates: Sequence[CandidateQAPair],
    max_answer_chars: int,
) -> list[Any]:
    system_prompt = (
        "You are an evidence-based clinical QA routing judge. "
        "Decide whether the retrieved guideline QA pairs are sufficient to directly answer "
        "the user's clinical question. Be conservative: answer yes only when the retrieved "
        "QA pairs clearly match the question, provide a coherent candidate answer, and cover "
        "the important PICO elements. Do not invent facts outside the candidate QA pairs. "
        "Return only valid JSON. Do not include markdown fences or extra text."
    )
    human_payload = {
        "task": "Judge whether the retrieved candidate QA pairs can directly answer the clinical question.",
        "clinical_question": question,
        "input_pico": normalize_pico(pico),
        "candidate_qa_pairs": [
            compact_candidate(candidate, max_answer_chars=max_answer_chars) for candidate in candidates
        ],
        "evaluation_dimensions": {
            "检索匹配强度": [
                "Compare clinical question intent with candidate question/disease/topic.",
                "Use retrieval_score and rank as secondary signals; do not rely on score alone.",
            ],
            "候选答案一致性": [
                "Check whether candidate answers support the same conclusion.",
                "Mark weak if answers conflict, are vague, or only indirectly answer the question.",
            ],
            "PICO覆盖度": [
                "Check population P, intervention/exposure I, comparators C, and outcomes O.",
                "Partial coverage may still be no if missing elements change the clinical meaning.",
            ],
        },
        "output_schema": {
            "判断": "yes or no",
            "理由": "final concise reason for the routing decision",
            "维度理由": {
                "检索匹配强度": "reason",
                "候选答案一致性": "reason",
                "PICO覆盖度": "reason",
            },
            "维度评级": {
                "检索匹配强度": "强/中/弱",
                "候选答案一致性": "一致/部分一致/冲突/不足",
                "PICO覆盖度": "完整/部分/不足",
            },
            "依据候选排名": ["integer ranks of candidates that support the decision"],
            "基于候选的简短答案": "brief answer only if 判断 is yes; otherwise empty string",
        },
        "constraints": [
            "The top-level JSON keys must be exactly the Chinese keys shown in output_schema.",
            "The 维度理由 object must contain exactly the three requested dimension keys.",
            "判断 must be lowercase yes or no.",
            "If no candidate can answer directly, 判断 must be no.",
        ],
    }
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(human_payload, ensure_ascii=False)),
    ]


def build_llm(model: str, api_key: str, base_url: str, temperature: float) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def parse_llm_json(content: str) -> dict[str, Any]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM response does not contain a JSON object: {content[:300]}")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"LLM JSON response must be an object: {payload!r}")
    return payload


def normalize_dimension_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        value = {}

    aliases = {
        "retrieval_match_strength": "检索匹配强度",
        "candidate_answer_consistency": "候选答案一致性",
        "answer_consistency": "候选答案一致性",
        "pico_coverage": "PICO覆盖度",
    }
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = aliases.get(str(key), str(key))
        if normalized_key in DIMENSION_KEYS:
            normalized[normalized_key] = stringify_reason(item)

    for key in DIMENSION_KEYS:
        normalized.setdefault(key, "")
    return normalized


def stringify_reason(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value).strip()


def normalize_ranks(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ranks: list[int] = []
    for item in value:
        rank = parse_optional_int(item)
        if rank is not None:
            ranks.append(rank)
    return ranks


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload.get("判断", payload.get("decision", payload.get("can_answer", "no")))
    if isinstance(decision, bool):
        decision_text = "yes" if decision else "no"
    else:
        decision_text = str(decision).strip().lower()
    if decision_text not in {"yes", "no"}:
        decision_text = "no"

    reason = stringify_reason(payload.get("理由", payload.get("reason", "")))
    dimension_reasons = normalize_dimension_map(
        payload.get("维度理由", payload.get("dimension_reasons", {}))
    )
    dimension_ratings = normalize_dimension_map(
        payload.get("维度评级", payload.get("dimension_judgments", payload.get("dimension_ratings", {})))
    )

    return {
        "判断": decision_text,
        "理由": reason,
        "维度理由": dimension_reasons,
        "维度评级": dimension_ratings,
        "依据候选排名": normalize_ranks(payload.get("依据候选排名", payload.get("best_candidate_ranks", []))),
        "基于候选的简短答案": stringify_reason(
            payload.get("基于候选的简短答案", payload.get("answer_basis", ""))
        ),
    }


def judge_route_batch(
    llm: Any,
    cases: Sequence[dict[str, Any]],
    max_concurrency: int,
) -> list[dict[str, Any]]:
    messages_batch = [
        build_judge_messages(
            question=case["question"],
            pico=case["pico"],
            candidates=case["candidates"],
            max_answer_chars=case["max_answer_chars"],
        )
        for case in cases
    ]
    responses = llm.batch(messages_batch, config={"max_concurrency": max_concurrency})
    return [normalize_judge_payload(parse_llm_json(str(response.content))) for response in responses]


def judge_route(
    question: str,
    pico: dict[str, Any],
    retrieved_qa_pairs: Any,
    llm: Any,
    retrieval_method: str = "hybrid",
    max_candidates: int = 5,
    max_answer_chars: int = 4000,
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Judge one clinical question using LangChain batch invocation."""

    candidates = normalize_retrieval_results(
        retrieved_qa_pairs,
        retrieval_method=retrieval_method,
        max_candidates=max_candidates,
    )
    cases = [
        {
            "question": question,
            "pico": normalize_pico(pico),
            "candidates": candidates,
            "max_answer_chars": max_answer_chars,
        }
    ]
    return judge_route_batch(llm=llm, cases=cases, max_concurrency=max_concurrency)[0]


def safe_args_for_log(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("api_key"):
        payload["api_key"] = "***"
    if payload.get("retrieval_json"):
        payload["retrieval_json"] = f"<{len(payload['retrieval_json'])} chars>"
    return payload


def main() -> None:
    args = parse_args()
    logger = setup_logging("llm_judge_router", log_file=args.log_file)

    log_step(logger, "加载环境变量")
    load_env_file(args.env_file)
    logger.debug("Arguments: %s", safe_args_for_log(args))

    log_step(logger, "加载并规范化输入")
    pico = load_pico_from_args(args)
    retrieval_payload = load_retrieval_payload(args)
    candidates = normalize_retrieval_results(
        retrieval_payload,
        retrieval_method=args.retrieval_method,
        max_candidates=args.max_candidates,
    )
    logger.debug("question=%s", args.question)
    logger.debug("pico=%s", json.dumps(pico, ensure_ascii=False, sort_keys=True))
    logger.debug("candidate_count=%d", len(candidates))
    logger.debug(
        "candidate_summaries=%s",
        [
            {
                "rank": candidate.rank,
                "index": candidate.index,
                "score": candidate.score,
                "question_id": candidate.record.get("question_id", ""),
                "question": candidate.record.get("question", ""),
            }
            for candidate in candidates
        ],
    )

    model = args.model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Set it in .env, pass --api-key, or set the environment variable."
        )

    log_step(logger, f"初始化 LLM: model={model} base_url={base_url}")
    llm = build_llm(model=model, api_key=api_key, base_url=base_url, temperature=args.temperature)

    log_step(logger, "批量调用 LLM judge 执行智能路由")
    result = judge_route_batch(
        llm=llm,
        cases=[
            {
                "question": args.question,
                "pico": pico,
                "candidates": candidates,
                "max_answer_chars": args.max_answer_chars,
            }
        ],
        max_concurrency=args.max_concurrency,
    )[0]
    logger.debug("judge_result=%s", json.dumps(result, ensure_ascii=False, indent=2))

    log_step(logger, "打印结构化路由结果")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    log_step(logger, "完成智能路由")


if __name__ == "__main__":
    main()
