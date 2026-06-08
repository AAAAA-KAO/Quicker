"""Run the full EvidenceQA clinical question pipeline.

功能：
    将 question_decomposition.py、hybrid_retrieval.py 和 llm_judge_router.py
    的核心功能集成到一个端到端脚本中：
        1. 使用 LLM batch 调用将临床问题分解为 PICO。
        2. 基于临床问题 + PICO 执行 Qdrant dense、BM25 sparse 和 hybrid 检索。
        3. 使用 LLM-as-a-judge batch 调用，从检索匹配强度、候选答案一致性、
           PICO 覆盖度 3 个维度判断检索问答对能否直接回答临床问题。

输入：
    --question 输入临床问题字符串，必填。
    可选 --pico-json 或 --pico-file 直接提供 PICO；如果提供，则跳过 PICO 分解。
    PICO 结构与 src/hybrid_retrieval.py 保持一致：
        P: 人群，字符串。
        I: 干预，字符串。
        C: 对照，列表。
        O: 结局，字典；key 为 C 中的元素，value 为结局列表。

输出：
    不保存结果文件。终端打印结构化 JSON，包含：
        question: 原始临床问题。
        pico: LLM 分解或用户输入的 PICO。
        retrieval: dense/sparse/hybrid 三路检索结果摘要。
        route: LLM judge 路由结果，包括 yes/no 判断、最终理由、三维度理由与评级。
    运行细节写入日志文件，默认 logs/evidenceqa_pipeline.log。

在项目根目录运行：
    conda run -n quicker python src/evidenceqa_pipeline.py \\
        --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?" \\
        --top-k 3 \\
        --max-concurrency 2 \\
        --qdrant-host localhost \\
        --qdrant-port 6333

命令行参数：
    --question, -q: 待处理的临床问题，必填。
    --pico-json: 可选 PICO JSON 字符串；可直接传 PICO，也可传包含 "pico" 字段的字典。
    --pico-file: 可选 PICO JSON 文件路径；可为 PICO 字典、含 "pico" 字段的字典，
        或 template.json 这类列表文件。
    --env-file: 环境变量文件路径，默认 .env。
    --log-file: 日志文件路径；不传则写入 logs/evidenceqa_pipeline.log。
    --model: LLM 模型名，默认从 DEEPSEEK_MODEL 读取，回退为 deepseek-v4-flash。
    --api-key: DeepSeek API key；不传则从 DEEPSEEK_API_KEY 读取。
    --base-url: DeepSeek base URL；不传则从 DEEPSEEK_BASE_URL 读取，
        回退为 https://api.deepseek.com。
    --temperature: LLM 温度，默认 0。
    --max-concurrency: LangChain LLM batch 最大并发数，默认 4。
    --bm25-index-file: BM25 pickle 索引路径，默认 results/clinical_qa_bm25.pkl。
    --qdrant-path: Qdrant 本地存储路径，默认
        data/qdrant_storage/collections/clinical_qa_dense。
    --qdrant-url: Qdrant 服务 URL；设置后优先于本地路径。
    --qdrant-host: Qdrant 服务 host；未设置 --qdrant-url 时生效。
    --qdrant-port: Qdrant 服务端口，默认 6333。
    --collection-name: Qdrant collection 名称；默认由 --qdrant-path 推断。
    --embedding-model-name: 查询向量编码模型，默认 BAAI/bge-m3，应与建库模型一致。
    --device: sentence-transformers 运行设备，默认 cpu。
    --normalize-embeddings / --no-normalize-embeddings: 是否归一化查询向量，默认开启。
    --top-k: 每种检索方式返回候选数，默认 5。
    --dense-weight: hybrid 检索中的稠密分数权重，默认 0.6。
    --sparse-weight: hybrid 检索中的 BM25 分数权重，默认 0.4。
    --retrieval-method: 交给 LLM judge 的候选来源，默认 hybrid。
    --max-answer-chars: 每条候选答案最多保留字符数，默认 4000。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from hybrid_retrieval import (
    DEFAULT_BM25_INDEX_FILE,
    DEFAULT_MODEL_NAME as DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_QDRANT_COLLECTION_PATH,
    EMPTY_PICO,
    RetrievalConfig,
    SearchResult,
    normalize_pico,
)
from hybrid_retrieval import ClinicalQAHybridRetriever
from llm_judge_router import (
    judge_route_batch,
    normalize_retrieval_results,
)
from utils.logging import log_step, setup_logging


DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

DECOMPOSITION_SYSTEM_PROMPT = (
    "You are a clinical evidence-based medicine assistant. "
    "Your task is to decompose a clinical question into PICO components. "
    "Return only valid JSON. Do not include markdown fences or extra text."
)

PICO_SCHEMA: dict[str, Any] = {
    "P": "string - population / patient group",
    "I": "string - intervention / exposure / index test",
    "C": ["list of comparator or control strings"],
    "O": {
        "comparator/control string from C": [
            "list of clinically relevant outcome strings for this comparator"
        ]
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PICO decomposition, hybrid retrieval, and LLM judge routing in one pipeline."
    )
    parser.add_argument("--question", "-q", required=True, help="Clinical question to process.")

    pico_group = parser.add_mutually_exclusive_group()
    pico_group.add_argument("--pico-json", default="", help="Optional PICO JSON string.")
    pico_group.add_argument("--pico-file", type=Path, default=None, help="Optional PICO JSON file.")

    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=4)

    parser.add_argument("--bm25-index-file", type=Path, default=DEFAULT_BM25_INDEX_FILE)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_COLLECTION_PATH)
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-host", default="")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--embedding-model-name", default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--sparse-weight", type=float, default=0.4)

    parser.add_argument(
        "--retrieval-method",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Which retrieval result list is passed to the LLM judge.",
    )
    parser.add_argument("--max-answer-chars", type=int, default=4000)
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


def load_pico_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.pico_json:
        return normalize_pico(json.loads(args.pico_json))
    if args.pico_file:
        return normalize_pico(load_json_file(args.pico_file))
    return None


def build_llm(model: str, api_key: str, base_url: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def build_decomposition_messages(question: str) -> list[Any]:
    human_payload = {
        "task": "Decompose the clinical question into PICO components.",
        "schema": {"pico": PICO_SCHEMA},
        "constraints": [
            "P and I must be strings (single string each, empty if not applicable).",
            "C must be a list of comparator/control strings (empty list if none).",
            "O must be an object. Its keys must be drawn from the C list.",
            "Use English medical terminology.",
        ],
        "question": question,
    }
    return [
        SystemMessage(content=DECOMPOSITION_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(human_payload, ensure_ascii=False)),
    ]


def parse_llm_json(content: str) -> dict[str, Any]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM response does not contain valid JSON: {content[:300]}")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"LLM JSON response must be an object: {payload!r}")
    return payload


def decompose_questions_batch(
    llm: Any,
    questions: Sequence[str],
    max_concurrency: int,
) -> list[dict[str, Any]]:
    messages_batch = [build_decomposition_messages(question) for question in questions]
    responses = llm.batch(messages_batch, config={"max_concurrency": max_concurrency})
    picos: list[dict[str, Any]] = []
    for response in responses:
        payload = parse_llm_json(str(response.content))
        picos.append(normalize_pico(payload))
    return picos


def build_retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        bm25_index_file=args.bm25_index_file,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
        model_name=args.embedding_model_name,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )


def serialize_search_result(result: SearchResult) -> dict[str, Any]:
    if is_dataclass(result):
        payload = asdict(result)
    else:
        payload = {
            "index": getattr(result, "index", None),
            "score": getattr(result, "score", None),
            "record": getattr(result, "record", {}),
        }
    return payload


def compact_answer(answer: Any, max_chars: int = 1200) -> Any:
    if isinstance(answer, str):
        return answer if len(answer) <= max_chars else answer[:max_chars].rstrip() + "...[truncated]"
    if isinstance(answer, list):
        return [compact_answer(item, max_chars=max_chars) for item in answer]
    if isinstance(answer, dict):
        return {key: compact_answer(value, max_chars=max_chars) for key, value in answer.items()}
    return answer


def summarize_retrieval_results(
    results: dict[str, Sequence[SearchResult]],
    max_answer_chars: int,
) -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for method, method_results in results.items():
        method_summary: list[dict[str, Any]] = []
        for rank, result in enumerate(method_results, start=1):
            record = result.record
            method_summary.append(
                {
                    "rank": rank,
                    "index": result.index,
                    "score": result.score,
                    "question_id": record.get("question_id", ""),
                    "question": record.get("question", ""),
                    "answer": compact_answer(record.get("answer", ""), max_answer_chars),
                    "disease": record.get("disease", ""),
                    "topic": record.get("topic", ""),
                }
            )
        summary[method] = method_summary
    return summary


def serialize_retrieval_results(results: dict[str, Sequence[SearchResult]]) -> dict[str, list[dict[str, Any]]]:
    return {
        method: [serialize_search_result(result) for result in method_results]
        for method, method_results in results.items()
    }


def safe_args_for_log(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("api_key"):
        payload["api_key"] = "***"
    if payload.get("pico_json"):
        payload["pico_json"] = f"<{len(payload['pico_json'])} chars>"
    return payload


def print_pipeline_result(
    question: str,
    pico: dict[str, Any],
    retrieval_results: dict[str, Sequence[SearchResult]],
    route_result: dict[str, Any],
    max_answer_chars: int,
) -> None:
    payload = {
        "question": question,
        "pico": pico,
        "retrieval": summarize_retrieval_results(
            retrieval_results,
            max_answer_chars=max_answer_chars,
        ),
        "route": route_result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    logger = setup_logging("evidenceqa_pipeline", log_file=args.log_file)

    log_step(logger, "加载环境变量")
    load_env_file(args.env_file)
    logger.debug("Arguments: %s", safe_args_for_log(args))

    model = args.model or os.getenv("DEEPSEEK_MODEL", DEFAULT_LLM_MODEL)
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Set it in .env, pass --api-key, or set the environment variable."
        )

    log_step(logger, f"初始化 LLM: model={model} base_url={base_url}")
    llm = build_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
    )

    user_pico = load_pico_from_args(args)
    if user_pico is None:
        log_step(logger, "批量调用 LLM 分解临床问题为 PICO")
        picos = decompose_questions_batch(
            llm=llm,
            questions=[args.question],
            max_concurrency=args.max_concurrency,
        )
        pico = picos[0] if picos else dict(EMPTY_PICO)
    else:
        log_step(logger, "使用命令行输入的 PICO，跳过 PICO 分解")
        pico = user_pico
    pico = normalize_pico(pico)
    logger.debug("question=%s", args.question)
    logger.debug("pico=%s", json.dumps(pico, ensure_ascii=False, sort_keys=True))

    log_step(logger, "初始化混合检索器")
    retrieval_config = build_retrieval_config(args)
    logger.debug(
        "retrieval_config=%s",
        {
            "bm25_index_file": str(retrieval_config.bm25_index_file),
            "qdrant_path": str(retrieval_config.qdrant_path),
            "qdrant_url": retrieval_config.qdrant_url,
            "qdrant_host": retrieval_config.qdrant_host,
            "qdrant_port": retrieval_config.qdrant_port,
            "collection_name": retrieval_config.collection_name,
            "model_name": retrieval_config.model_name,
            "device": retrieval_config.device,
            "normalize_embeddings": retrieval_config.normalize_embeddings,
            "top_k": retrieval_config.top_k,
            "dense_weight": retrieval_config.dense_weight,
            "sparse_weight": retrieval_config.sparse_weight,
        },
    )
    retriever = ClinicalQAHybridRetriever(retrieval_config)

    log_step(logger, "执行 dense/sparse/hybrid 混合检索")
    retrieval_results = retriever.retrieve(args.question, pico)
    logger.debug(
        "retrieval_counts=%s",
        {method: len(method_results) for method, method_results in retrieval_results.items()},
    )
    logger.debug(
        "hybrid_top_questions=%s",
        [
            {
                "rank": rank,
                "index": result.index,
                "score": result.score,
                "question_id": result.record.get("question_id", ""),
                "question": result.record.get("question", ""),
            }
            for rank, result in enumerate(retrieval_results.get("hybrid", []), start=1)
        ],
    )

    log_step(logger, "规范化检索候选并批量调用 LLM judge")
    serialized_results = serialize_retrieval_results(retrieval_results)
    candidates = normalize_retrieval_results(
        serialized_results,
        retrieval_method=args.retrieval_method,
        max_candidates=args.top_k,
    )
    route_result = judge_route_batch(
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
    logger.debug("route_result=%s", json.dumps(route_result, ensure_ascii=False, indent=2))

    log_step(logger, "打印端到端结果")
    print_pipeline_result(
        question=args.question,
        pico=pico,
        retrieval_results=retrieval_results,
        route_result=route_result,
        max_answer_chars=args.max_answer_chars,
    )
    log_step(logger, "完成端到端 EvidenceQA 流程")


if __name__ == "__main__":
    main()
