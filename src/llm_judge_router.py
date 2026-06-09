"""Route clinical QA retrieval results with an LLM-as-a-judge.

功能：
    接收临床问题、PICO 结构和 hybrid_retrieval.py 返回/导出的检索问答对，
    使用 LangChain 调用 LLM 从 3 个维度判断"检索到的问答对能否回答临床问题"：
        1. 检索匹配强度：候选问答的问题、疾病、主题、检索分数是否与临床问题匹配。
        2. 候选答案一致性：候选答案之间是否支持同一临床结论，是否存在冲突或只间接相关。
        3. PICO 覆盖度：候选问答是否覆盖输入 PICO 中的人群、干预、对照和结局。

两种运行模式：
    1. 单问题模式（--question）：
        对单个临床问题进行 LLM judge 路由判断。
    2. 批量模式（--input-file）：
        从一个 JSON 文件（如 results/retrieve/out_questions.json）中读取多个问题，
        批量执行 LLM judge 判断，结果保存为单个 JSON 列表文件到
        --output（默认 results/route/results.json）。

输入文件格式（批量模式）：
    JSON 列表，每个元素为字典，需包含以下字段：
        question: 临床问题字符串
        pico: PICO 字典
        hybrid_results: 检索结果列表（LLM judge 仅基于 hybrid_results 进行判断）
        question_id (可选): 用于命名输出文件的标识符
    示例文件：results/retrieve/out_questions.json

输出（批量模式）：
    结果保存为单个 JSON 文件（--output），是一个列表，每个字典对应输入文件中的一个字典，
    包含 question_id、question、pico、judge_result、candidates_used 字段。

在项目根目录运行（单问题模式）：
    conda run -n quicker python src/llm_judge_router.py \\
        --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?" \\
        --pico-json '{"P":"pediatric patients with suspected appendicitis","I":"clinical scores alone","C":["imaging or laboratory-assisted diagnosis"],"O":{"imaging or laboratory-assisted diagnosis":["diagnostic accuracy","missed appendicitis"]}}'

批量模式运行：
    conda run -n quicker python src/llm_judge_router.py \\
        --input-file results/retrieve/out_questions.json \\
        --output results/route/out_judge_results.json

命令行参数：
    --question, -q: 待判断的临床问题（单问题模式必填）。
    --input-file, -i: 批量模式输入 JSON 文件路径。
    --output: 批量模式输出 JSON 文件路径，默认 results/route/results.json。
    --pico-json: PICO JSON 字符串；可直接传 PICO，也可传包含 "pico" 字段的字典。
    --pico-file: PICO JSON 文件路径；可为 PICO 字典、含 "pico" 字段的字典，
        或 template.json 这类列表文件。
    --retrieval-json: 检索结果 JSON 字符串。
    --retrieval-file: 检索结果 JSON 文件路径。
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from hybrid_retrieval import normalize_pico
from utils.logging import log_step, setup_logging


DIMENSION_KEYS = ("retrieval_match_strength", "candidate_answer_consistency", "pico_coverage")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_OUTPUT_FILE = Path("results/route/results.json")
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

    # 运行模式：单问题 vs 批量
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--question", "-q",
        default=None,
        help="Clinical question to judge (single-question mode).",
    )
    mode_group.add_argument(
        "--input-file", "-i",
        type=Path,
        default=None,
        help="Path to a JSON file containing a list of questions with retrieval results (batch mode). "
             "Example: results/retrieve/out_questions.json",
    )

    pico_group = parser.add_mutually_exclusive_group()
    pico_group.add_argument("--pico-json", default="", help="PICO JSON string.")
    pico_group.add_argument("--pico-file", type=Path, default=None, help="Path to a PICO JSON file.")

    retrieval_group = parser.add_mutually_exclusive_group()
    retrieval_group.add_argument("--retrieval-json", default="", help="Retrieval results JSON string.")
    retrieval_group.add_argument(
        "--retrieval-file",
        type=Path,
        default=Path("results/retrieve/results.json"),
        help="Path to retrieval results JSON file (default: results/retrieve/results.json).",
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
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path (default: results/route/results.json). "
             "In batch mode this is written as a list; in single mode results are appended.",
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
        payload = load_json_file(args.retrieval_file)
        if isinstance(payload, list):
            return _lookup_retrieval_by_question(payload, args.question)
        return payload
    raise ValueError("Either --retrieval-json or --retrieval-file is required.")


def _lookup_retrieval_by_question(
    entries: list[dict[str, Any]], question: str
) -> list[Any]:
    """Find the entry matching the question in a retrieval results list.

    Each entry is expected to have a ``question`` field and a ``hybrid_results``
    field (as written by ``hybrid_retrieval.py``). Returns the ``hybrid_results``
    list directly so the caller can pass it to ``normalize_retrieval_results``.
    """
    normalized_question = question.strip()
    for entry in entries:
        if entry.get("question", "").strip() == normalized_question:
            return entry.get("hybrid_results", [])

    available = [e.get("question", "")[:80] for e in entries]
    raise ValueError(
        f"Question not found in retrieval results file.\n"
        f"  Input question: {question!r}\n"
        f"  File contains {len(entries)} entries: {available}"
    )


def coerce_to_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        attrs = {name: getattr(value, name) for name in ("index", "score", "record") if hasattr(value, name)}
        if attrs:
            return attrs
        raise ValueError(f"Candidate item must be a dict or SearchResult-like object, got {type(value)!r}.")
    return dict(value)


def select_retrieval_items(payload: Any) -> list[Any]:
    """Extract retrieval result items from a payload, always preferring hybrid_results.

    When ``payload`` is a dict, looks for the ``hybrid`` key first, then falls back
    to ``results``, ``retrieval_results``, or treats the dict itself as a single
    record. When ``payload`` is a list, returns it directly.
    """
    if isinstance(payload, dict):
        if "hybrid" in payload:
            selected = payload["hybrid"]
        elif "results" in payload:
            selected = payload["results"]
        elif "retrieval_results" in payload:
            selected = payload["retrieval_results"]
        elif "record" in payload or "question" in payload or "answer" in payload:
            selected = [payload]
        else:
            available = ", ".join(sorted(str(key) for key in payload.keys()))
            raise ValueError(
                f"Cannot find retrieval list. Expected 'hybrid', 'results', "
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
    max_candidates: int = 5,
) -> list[CandidateQAPair]:
    items = select_retrieval_items(payload)
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
        "Return only valid JSON with English keys and English natural-language values. "
        "Do not include markdown fences or extra text."
    )
    human_payload = {
        "task": "Judge whether the retrieved candidate QA pairs can directly answer the clinical question.",
        "clinical_question": question,
        "input_pico": normalize_pico(pico),
        "candidate_qa_pairs": [
            compact_candidate(candidate, max_answer_chars=max_answer_chars) for candidate in candidates
        ],
        "evaluation_dimensions": {
            "retrieval_match_strength": [
                "Compare clinical question intent with candidate question/disease/topic.",
                "Use retrieval_score and rank as secondary signals; do not rely on score alone.",
            ],
            "candidate_answer_consistency": [
                "Check whether candidate answers support the same conclusion.",
                "Mark weak if answers conflict, are vague, or only indirectly answer the question.",
            ],
            "pico_coverage": [
                "Check population P, intervention/exposure I, comparators C, and outcomes O.",
                "Partial coverage may still be no if missing elements change the clinical meaning.",
            ],
        },
        "output_schema": {
            "decision": "yes or no",
            "reason": "final concise reason for the routing decision in English",
            "dimension_reasons": {
                "retrieval_match_strength": "reason in English",
                "candidate_answer_consistency": "reason in English",
                "pico_coverage": "reason in English",
            },
            "dimension_ratings": {
                "retrieval_match_strength": "strong/moderate/weak",
                "candidate_answer_consistency": "consistent/partly_consistent/conflicting/insufficient",
                "pico_coverage": "complete/partial/insufficient",
            },
            "supporting_candidate_ranks": ["integer ranks of candidates that support the decision"],
            "candidate_based_brief_answer": "brief answer in English only if decision is yes; otherwise empty string",
        },
        "constraints": [
            "The top-level JSON keys must be exactly the English keys shown in output_schema.",
            "The dimension_reasons object must contain exactly the three requested dimension keys.",
            "The dimension_ratings object must contain exactly the three requested dimension keys.",
            "All natural-language strings in the returned JSON must be written in English.",
            "decision must be lowercase yes or no.",
            "If no candidate can answer directly, decision must be no.",
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
        "检索匹配强度": "retrieval_match_strength",
        "retrieval_match": "retrieval_match_strength",
        "match_strength": "retrieval_match_strength",
        "候选答案一致性": "candidate_answer_consistency",
        "答案一致性": "candidate_answer_consistency",
        "answer_consistency": "candidate_answer_consistency",
        "PICO覆盖度": "pico_coverage",
        "pico coverage": "pico_coverage",
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
    decision = payload.get("decision", payload.get("判断", payload.get("can_answer", "no")))
    if isinstance(decision, bool):
        decision_text = "yes" if decision else "no"
    else:
        decision_text = str(decision).strip().lower()
    if decision_text not in {"yes", "no"}:
        decision_text = "no"

    reason = stringify_reason(payload.get("reason", payload.get("理由", "")))
    dimension_reasons = normalize_dimension_map(
        payload.get("dimension_reasons", payload.get("维度理由", {}))
    )
    dimension_ratings = normalize_dimension_map(
        payload.get("dimension_ratings", payload.get("dimension_judgments", payload.get("维度评级", {})))
    )

    return {
        "decision": decision_text,
        "reason": reason,
        "dimension_reasons": dimension_reasons,
        "dimension_ratings": dimension_ratings,
        "supporting_candidate_ranks": normalize_ranks(
            payload.get(
                "supporting_candidate_ranks",
                payload.get("best_candidate_ranks", payload.get("依据候选排名", [])),
            )
        ),
        "candidate_based_brief_answer": stringify_reason(
            payload.get(
                "candidate_based_brief_answer",
                payload.get("answer_basis", payload.get("基于候选的简短答案", "")),
            )
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
    max_candidates: int = 5,
    max_answer_chars: int = 4000,
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Judge one clinical question using LangChain batch invocation."""

    candidates = normalize_retrieval_results(
        retrieved_qa_pairs,
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


def append_results_to_json(file_path: Path, entry: dict[str, Any]) -> None:
    """Append one run result dict to a JSON file storing a list of results."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            data = []
    else:
        data = []
    data.append(entry)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def safe_args_for_log(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("api_key"):
        payload["api_key"] = "***"
    if payload.get("retrieval_json"):
        payload["retrieval_json"] = f"<{len(payload['retrieval_json'])} chars>"
    return payload


def _build_case_from_entry(
    entry: dict[str, Any],
    max_candidates: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    """Build a single case dict for judge_route_batch from an input file entry.

    Each entry is expected to have ``question``, ``pico``, and a
    ``hybrid_results`` retrieval result list. The LLM judge only inspects
    ``hybrid_results`` — ``dense_results`` and ``sparse_results`` are ignored.
    """
    retrieval_payload = {"hybrid": entry.get("hybrid_results", [])}
    candidates = normalize_retrieval_results(
        retrieval_payload,
        max_candidates=max_candidates,
    )
    return {
        "question": entry["question"],
        "pico": normalize_pico(entry.get("pico", {})),
        "candidates": candidates,
        "max_answer_chars": max_answer_chars,
    }


def run_batch(
    input_file: Path,
    output_file: Path,
    llm: Any,
    max_candidates: int = 5,
    max_answer_chars: int = 4000,
    max_concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Run LLM judge routing on all entries from a JSON input file.

    The LLM judge only inspects ``hybrid_results`` from each entry;
    ``dense_results`` and ``sparse_results`` are ignored.

    Parameters
    ----------
    input_file:
        Path to a JSON file containing a list of question entries.
        Each entry must have ``question``, ``pico``, and ``hybrid_results``
        (a list of retrieval result items).
    output_file:
        Path to the output JSON file. A list of result dicts is written,
        one dict per input entry.
    llm:
        Configured LangChain ChatOpenAI instance.
    max_candidates:
        Max candidate QA pairs per question.
    max_answer_chars:
        Max chars per candidate answer.
    max_concurrency:
        LangChain batch max concurrency.

    Returns
    -------
    A list of result dicts, each containing the original entry metadata plus
    the judge result.
    """
    entries = load_json_file(input_file)
    if not isinstance(entries, list):
        raise ValueError(f"Input file must contain a JSON list, got {type(entries).__name__}")

    # Build cases for all entries
    cases: list[dict[str, Any]] = []
    for entry in entries:
        case = _build_case_from_entry(
            entry,
            max_candidates=max_candidates,
            max_answer_chars=max_answer_chars,
        )
        cases.append(case)

    # Batch judge all cases
    judge_results = judge_route_batch(llm=llm, cases=cases, max_concurrency=max_concurrency)

    # Assemble results (one dict per input entry)
    results: list[dict[str, Any]] = []
    for idx, (entry, judge_result) in enumerate(zip(entries, judge_results)):
        question_id = entry.get("question_id") or f"q{idx:04d}"
        result_entry: dict[str, Any] = {
            "question_id": question_id,
            "question": entry["question"],
            "pico": entry.get("pico", {}),
            "judge_result": judge_result,
            "candidates_used": [
                compact_candidate(c, max_answer_chars)
                for c in normalize_retrieval_results(
                    {"hybrid": entry.get("hybrid_results", [])},
                    max_candidates=max_candidates,
                )
            ],
        }
        results.append(result_entry)

    # Write single output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return results


def main() -> None:
    args = parse_args()
    logger = setup_logging("llm_judge_router", log_file=args.log_file)

    # Validate mode: either --question or --input-file is required
    if not args.question and not args.input_file:
        raise SystemExit(
            "Either --question (single mode) or --input-file (batch mode) is required."
        )

    log_step(logger, "加载环境变量")
    load_env_file(args.env_file)
    logger.debug("Arguments: %s", safe_args_for_log(args))

    # --- LLM 初始化（两种模式共用）---
    model = args.model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Set it in .env, pass --api-key, or set the environment variable."
        )

    log_step(logger, f"初始化 LLM: model={model} base_url={base_url}")
    llm = build_llm(model=model, api_key=api_key, base_url=base_url, temperature=args.temperature)

    # ==================================================================
    # 批量模式：--input-file
    # ==================================================================
    if args.input_file:
        log_step(logger, f"批量模式：从 {args.input_file} 加载输入")
        input_file = args.input_file
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        entries = load_json_file(input_file)
        if not isinstance(entries, list):
            raise ValueError(f"Input file must contain a JSON list, got {type(entries).__name__}")
        logger.info("Loaded %d questions from %s", len(entries), input_file)

        log_step(logger, f"批量 LLM judge 路由 → 输出文件 {args.output}")
        results = run_batch(
            input_file=input_file,
            output_file=args.output,
            llm=llm,
            max_candidates=args.max_candidates,
            max_answer_chars=args.max_answer_chars,
            max_concurrency=args.max_concurrency,
        )

        # 打印汇总
        yes_count = sum(1 for r in results if r["judge_result"]["decision"] == "yes")
        no_count = len(results) - yes_count
        print(json.dumps({
            "mode": "batch",
            "total": len(results),
            "yes": yes_count,
            "no": no_count,
            "output_file": str(args.output),
        }, ensure_ascii=False, indent=2))
        log_step(logger, f"批量模式完成: {len(results)} 个问题, yes={yes_count}, no={no_count}")
        return

    # ==================================================================
    # 单问题模式：--question
    # ==================================================================
    log_step(logger, "加载并规范化输入")
    pico = load_pico_from_args(args)
    retrieval_payload = load_retrieval_payload(args)
    candidates = normalize_retrieval_results(
        retrieval_payload,
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

    log_step(logger, f"保存路由结果到 {args.output}")
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": args.question,
        "pico": pico,
        "candidates": [compact_candidate(c, args.max_answer_chars) for c in candidates],
        "judge_result": result,
    }
    append_results_to_json(args.output, entry)
    log_step(logger, "完成智能路由")


if __name__ == "__main__":
    main()
