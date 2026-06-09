"""Run hybrid dense+sparse retrieval for clinical QA.

功能：
    接收一个临床问题字符串及其 PICO 组件（或包含多个问题的 JSON 文件），
    将问题与 PICO 拼接成完整检索文本，并联执行两路检索：
        1. Qdrant 稠密向量检索，默认读取
           data/qdrant_storage/collections/clinical_qa_dense，返回前 5 个结果。
        2. BM25 稀疏检索，默认读取 results/clinical_qa_bm25.pkl，返回前 5 个结果。
    然后对两路候选分数分别做最大最小归一化，并按
    dense_weight * dense_score + sparse_weight * sparse_score 融合排序，
    默认权重为 0.6 和 0.4，返回前 5 个混合检索结果。

    支持两种模式：
        单问题模式：通过 --question + --pico-json/--pico-file 指定单个临床问题。
        批量模式：通过 --questions-file 指定包含多个问题的 JSON 文件，
            逐个检索并增量保存结果；即使 question_id 重复也不会跳过。

输入：
    --question 或 --questions-file（二者必选其一）：
        --question -q: 输入临床问题字符串。
        --questions-file: 包含多个问题的 JSON 文件路径，每条记录包含
            question_id、question 和 pico 字段。
            示例文件：data/evaluate/retrieve/original_questions.json
    --pico-json 或 --pico-file 输入 PICO（仅单问题模式）。PICO 结构遵循
    data/mimic-cpg/template.json 中 "pico" 字段：
        P: 人群，字符串。
        I: 干预，字符串。
        C: 对照，列表。
        O: 结局，字典；key 为 C 中的元素。

输出：
    终端打印每个问题的 Qdrant dense、BM25 sparse、hybrid 三种方法的前 5 个结果，
    每条结果包含 question、answer、disease 字段。运行过程通过项目日志工具
    记录到日志文件；结果增量保存到 --output-file（默认 results/retrieve/results.json）。

在项目根目录运行（单问题模式）：
    conda run -n quicker python src/hybrid_retrieval.py \\
        --qdrant-host localhost \\
        --qdrant-port 6333 \\
        --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?" \\
        --pico-json '{"P":"pediatric patients with suspected appendicitis","I":"clinical scores alone","C":["imaging or laboratory-assisted diagnosis"],"O":{"imaging or laboratory-assisted diagnosis":["diagnostic accuracy","missed appendicitis"]}}'

批量模式：
    conda run -n quicker python src/hybrid_retrieval.py \\
        --qdrant-host localhost \\
        --qdrant-port 6333 \\
        --questions-file data/evaluate/retrieve/original_questions.json

命令行参数：
    --question, -q: 待检索的临床问题（与 --questions-file 互斥）。
    --questions-file: 包含多个问题对象的 JSON 文件路径（与 --question 互斥）。
    --pico-json: PICO JSON 字符串；可直接传 PICO，也可传包含 "pico" 字段的字典（仅单问题模式）。
    --pico-file: PICO JSON 文件路径；可为 PICO 字典、含 "pico" 字段的字典，
        或 template.json 这类列表文件（仅单问题模式）。
    --log-file: 日志文件路径；不传则写入 logs/hybrid_retrieval.log。
    --bm25-index-file: BM25 pickle 索引路径，默认 results/clinical_qa_bm25.pkl。
    --qdrant-path: Qdrant 本地存储路径；默认使用用户给定的 collection 路径
        data/qdrant_storage/collections/clinical_qa_dense。脚本会自动推断存储根目录
        data/qdrant_storage 和 collection 名称 clinical_qa_dense。
    --qdrant-url: Qdrant 服务 URL；设置后优先于本地路径。
    --qdrant-host: Qdrant 服务 host；未设置 --qdrant-url 时生效。
    --qdrant-port: Qdrant 服务端口，默认 6333。
    --collection-name: Qdrant collection 名称；默认从 --qdrant-path 推断，
        推断失败时使用 clinical_qa_dense。
    --model-name: 查询向量编码模型，默认 BAAI/bge-m3，应与建库模型一致。
    --device: sentence-transformers 运行设备，默认 cpu。
    --normalize-embeddings / --no-normalize-embeddings: 是否归一化查询向量，
        默认开启，适配 Cosine 检索。
    --top-k: 每种检索方式返回并打印的条数，默认 10。
    --dense-weight: 混合检索中的稠密分数权重，默认 0.6。
    --sparse-weight: 混合检索中的 BM25 分数权重，默认 0.4。
    --output-file: 检索结果输出 JSON 文件路径，默认 results/retrieve/results.json。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from utils.logging import log_step, setup_logging


DEFAULT_QDRANT_COLLECTION_PATH = Path("data/qdrant_storage/collections/clinical_qa_dense")
DEFAULT_BM25_INDEX_FILE = Path("results/clinical_qa_bm25.pkl")
DEFAULT_OUTPUT_FILE = Path("results/retrieve/results.json")
DEFAULT_COLLECTION_NAME = "clinical_qa_dense"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
EMPTY_PICO: dict[str, Any] = {"P": "", "I": "", "C": [], "O": {}}


@dataclass(frozen=True)
class SearchResult:
    """One retrieval hit from dense, sparse, or hybrid search."""

    index: int
    score: float
    record: dict[str, Any]


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for loading indexes and running hybrid retrieval."""

    bm25_index_file: Path = DEFAULT_BM25_INDEX_FILE
    qdrant_path: Path = DEFAULT_QDRANT_COLLECTION_PATH
    qdrant_url: str = ""
    qdrant_host: str = ""
    qdrant_port: int = 6333
    collection_name: str | None = None
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "cpu"
    normalize_embeddings: bool = True
    top_k: int = 10
    dense_weight: float = 0.6
    sparse_weight: float = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense Qdrant, sparse BM25, and weighted hybrid retrieval."
    )

    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument("--question", "-q", default=None, help="Clinical question to retrieve.")
    question_group.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Path to a JSON file containing an array of question objects. "
        "Each object must have 'question_id', 'question', and optionally 'pico' fields. "
        "Example: data/evaluate/retrieve/original_questions.json",
    )

    pico_group = parser.add_mutually_exclusive_group()
    pico_group.add_argument("--pico-json", default="", help="PICO JSON string (single question mode only).")
    pico_group.add_argument("--pico-file", type=Path, default=None, help="Path to a PICO JSON file (single question mode only).")

    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--bm25-index-file", type=Path, default=DEFAULT_BM25_INDEX_FILE)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_COLLECTION_PATH)
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-host", default="")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--sparse-weight", type=float, default=0.4)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def normalize_pico(pico: Any) -> dict[str, Any]:
    """Normalize PICO to the template shape used by data/mimic-cpg/template.json."""

    if isinstance(pico, list):
        if not pico:
            return dict(EMPTY_PICO)
        first_item = pico[0]
        if isinstance(first_item, dict) and "pico" in first_item:
            pico = first_item["pico"]
        else:
            pico = first_item

    if isinstance(pico, dict) and "pico" in pico and isinstance(pico["pico"], dict):
        pico = pico["pico"]

    if not isinstance(pico, dict):
        raise ValueError("PICO must be a dict, a dict containing a 'pico' field, or a template-style list.")

    comparisons = pico.get("C", [])
    if isinstance(comparisons, str):
        comparisons = [comparisons] if comparisons.strip() else []
    if not isinstance(comparisons, list):
        comparisons = []
    normalized_c = [str(item).strip() for item in comparisons if str(item).strip()]

    outcomes = pico.get("O", {})
    normalized_o: dict[str, Any] = {}
    if isinstance(outcomes, dict):
        for key, value in outcomes.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if isinstance(value, list):
                normalized_o[key_text] = [str(item).strip() for item in value if str(item).strip()]
            elif value is None:
                normalized_o[key_text] = []
            else:
                value_text = str(value).strip()
                normalized_o[key_text] = [value_text] if value_text else []

    return {
        "P": str(pico.get("P", "") or "").strip(),
        "I": str(pico.get("I", "") or "").strip(),
        "C": normalized_c,
        "O": normalized_o,
    }


def load_pico_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.pico_json:
        return normalize_pico(json.loads(args.pico_json))
    if args.pico_file:
        with args.pico_file.open("r", encoding="utf-8") as handle:
            return normalize_pico(json.load(handle))
    return dict(EMPTY_PICO)


def build_query_text(question: str, pico: dict[str, Any]) -> str:
    """Concatenate the clinical question and PICO into one Python string."""

    normalized_pico = normalize_pico(pico)
    return "\n".join(
        [
            f"question: {question.strip()}",
            f"pico: {json.dumps(normalized_pico, ensure_ascii=False, sort_keys=True)}",
        ]
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())


def load_bm25_index(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"BM25 index at {path} must be a dict payload.")
    if "bm25" not in payload or "records" not in payload:
        raise ValueError(f"BM25 index at {path} must contain 'bm25' and 'records'.")

    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError(f"BM25 records in {path} must be a list.")
    return payload["bm25"], records


def load_embedding_model(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency sentence-transformers. Run with the project environment, "
            "for example: conda run -n quicker python src/hybrid_retrieval.py ..."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def encode_query(model: Any, query_text: str, normalize_embeddings: bool) -> list[float]:
    embedding = model.encode(
        [query_text],
        batch_size=1,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return embedding.astype("float32").tolist()


def infer_qdrant_path_and_collection(
    qdrant_path: Path,
    collection_name: str | None,
) -> tuple[Path, str]:
    if qdrant_path.parent.name == "collections":
        storage_path = qdrant_path.parent.parent
        inferred_collection_name = qdrant_path.name
    else:
        storage_path = qdrant_path
        inferred_collection_name = DEFAULT_COLLECTION_NAME
    return storage_path, collection_name or inferred_collection_name


def make_qdrant_client(config: RetrievalConfig) -> tuple[Any, str]:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "Missing dependency qdrant-client. Run with the project environment, "
            "for example: conda run -n quicker python src/hybrid_retrieval.py ..."
        ) from exc

    storage_path, collection_name = infer_qdrant_path_and_collection(
        config.qdrant_path,
        config.collection_name,
    )
    if config.qdrant_url:
        client = QdrantClient(url=config.qdrant_url, check_compatibility=False)
    elif config.qdrant_host:
        client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            check_compatibility=False,
        )
    else:
        client = QdrantClient(path=str(storage_path))

    return client, collection_name


def ensure_qdrant_collection(client: Any, collection_name: str, config: RetrievalConfig) -> None:
    try:
        client.get_collection(collection_name)
    except Exception as exc:
        storage_path, inferred_collection = infer_qdrant_path_and_collection(
            config.qdrant_path,
            config.collection_name,
        )
        raise RuntimeError(
            "Cannot open Qdrant collection "
            f"{collection_name!r}. If you are using local storage, check --qdrant-path "
            f"(current inferred storage root: {storage_path}, collection: {inferred_collection}). "
            "If the collection is managed by a running Qdrant service, pass --qdrant-host/--qdrant-port "
            "or --qdrant-url instead of reading the storage directory directly."
        ) from exc


def dense_search(
    client: Any,
    collection_name: str,
    model: Any,
    query_text: str,
    records: Sequence[dict[str, Any]],
    top_k: int,
    normalize_embeddings: bool,
) -> list[SearchResult]:
    query_embedding = encode_query(model, query_text, normalize_embeddings)
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    hits = response.points if hasattr(response, "points") else response

    results: list[SearchResult] = []
    for hit in hits:
        try:
            point_id = int(hit.id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Qdrant point ids must be integer ids aligned with BM25 records for score fusion."
            ) from exc

        payload = hit.payload or {}
        if 0 <= point_id < len(records):
            record = dict(records[point_id])
            record.update(payload)
        else:
            record = dict(payload)
        results.append(SearchResult(index=point_id, score=float(hit.score), record=record))
    return results


def bm25_search(
    bm25: Any,
    query_text: str,
    records: Sequence[dict[str, Any]],
    top_k: int,
) -> list[SearchResult]:
    scores = bm25.get_scores(tokenize(query_text))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
    return [
        SearchResult(index=idx, score=float(scores[idx]), record=dict(records[idx]))
        for idx in ranked_indices
    ]


def minmax_normalize(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}

    values = list(score_map.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {idx: 1.0 for idx in score_map}
    return {idx: (score - min_score) / (max_score - min_score) for idx, score in score_map.items()}


def combine_hybrid_results(
    dense_results: Sequence[SearchResult],
    sparse_results: Sequence[SearchResult],
    records: Sequence[dict[str, Any]],
    top_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[SearchResult]:
    dense_scores = minmax_normalize({result.index: result.score for result in dense_results})
    sparse_scores = minmax_normalize({result.index: result.score for result in sparse_results})
    dense_records = {result.index: result.record for result in dense_results}
    sparse_records = {result.index: result.record for result in sparse_results}

    candidate_indices = set(dense_scores) | set(sparse_scores)
    hybrid_results: list[SearchResult] = []
    for idx in candidate_indices:
        score = dense_weight * dense_scores.get(idx, 0.0) + sparse_weight * sparse_scores.get(idx, 0.0)
        if 0 <= idx < len(records):
            record = dict(records[idx])
        else:
            record = {}
        record.update(sparse_records.get(idx, {}))
        record.update(dense_records.get(idx, {}))
        hybrid_results.append(SearchResult(index=idx, score=score, record=record))

    return sorted(hybrid_results, key=lambda result: result.score, reverse=True)[:top_k]


class ClinicalQAHybridRetriever:
    """Reusable retriever that keeps BM25, embedding model, and Qdrant client loaded."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.bm25, self.records = load_bm25_index(self.config.bm25_index_file)
        self.qdrant_client, self.collection_name = make_qdrant_client(self.config)
        ensure_qdrant_collection(self.qdrant_client, self.collection_name, self.config)
        self.model = load_embedding_model(self.config.model_name, self.config.device)

    def retrieve(self, question: str, pico: dict[str, Any]) -> dict[str, list[SearchResult]]:
        query_text = build_query_text(question, pico)

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                dense_search,
                self.qdrant_client,
                self.collection_name,
                self.model,
                query_text,
                self.records,
                self.config.top_k,
                self.config.normalize_embeddings,
            )
            sparse_future = executor.submit(
                bm25_search,
                self.bm25,
                query_text,
                self.records,
                self.config.top_k,
            )
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()

        hybrid_results = combine_hybrid_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            records=self.records,
            top_k=self.config.top_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )

        return {
            "dense": dense_results[: self.config.top_k],
            "sparse": sparse_results[: self.config.top_k],
            "hybrid": hybrid_results,
        }


def hybrid_retrieve(
    question: str,
    pico: dict[str, Any],
    config: RetrievalConfig | None = None,
) -> dict[str, list[SearchResult]]:
    """Run dense, sparse, and weighted hybrid retrieval for one clinical question."""

    return ClinicalQAHybridRetriever(config).retrieve(question, pico)


def format_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return " | ".join(str(item) for item in answer)
    return str(answer or "")


def print_results(title: str, results: Sequence[SearchResult]) -> None:
    print(f"\n=== {title} ===")
    for rank, result in enumerate(results, start=1):
        record = result.record
        print(f"\n[{rank}] score={result.score:.4f}")
        print(f"question: {record.get('question', '')}")
        print(f"answer: {format_answer(record.get('answer', ''))}")
        print(f"disease: {record.get('disease', '')}")


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


def build_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        bm25_index_file=args.bm25_index_file,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )


def load_questions_from_file(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array of question objects from a file.

    Each object must have at least ``question``, and optionally ``question_id``
    and ``pico``.  Returns the list of objects unchanged — callers normalise
    PICO fields themselves.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Questions file must contain a JSON array, got {type(data).__name__}: {path}")

    validated: list[dict[str, Any]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Each item in questions file must be an object, item {idx} is {type(entry).__name__}")
        if "question" not in entry or not str(entry.get("question", "")).strip():
            raise ValueError(f"Item {idx} in questions file is missing a non-empty 'question' field")
        validated.append(entry)

    return validated


def run_single_question(
    question: str,
    pico: dict[str, Any],
    config: RetrievalConfig,
    logger: Any,
    *,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Execute retrieval for one question and return the result entry dict.

    This is the shared core used by both single-question and batch modes.
    """
    query_text = build_query_text(question, pico)
    logger.debug("question=%s", question)
    logger.debug("pico=%s", json.dumps(pico, ensure_ascii=False, sort_keys=True))
    logger.debug("query_text=%s", query_text)

    print("\n" + "=" * 72)
    if question_id:
        print(f"Question ID: {question_id}")
    print("Query text:")
    print(query_text)

    results = hybrid_retrieve(question, pico, config)
    logger.debug(
        "result_counts=%s",
        {method: len(method_results) for method, method_results in results.items()},
    )

    print_results("Qdrant Dense Top Results", results["dense"])
    print_results("BM25 Sparse Top Results", results["sparse"])
    print_results("Hybrid Top Results", results["hybrid"])

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "pico": pico,
        "query_text": query_text,
        "dense_results": [asdict(r) for r in results["dense"]],
        "sparse_results": [asdict(r) for r in results["sparse"]],
        "hybrid_results": [asdict(r) for r in results["hybrid"]],
    }
    if question_id:
        entry["question_id"] = question_id

    return entry


def main() -> None:
    args = parse_args()
    logger = setup_logging("hybrid_retrieval", log_file=args.log_file)

    config = build_config(args)
    logger.debug(
        "retrieval_config=%s",
        {
            "bm25_index_file": str(config.bm25_index_file),
            "qdrant_path": str(config.qdrant_path),
            "qdrant_url": config.qdrant_url,
            "qdrant_host": config.qdrant_host,
            "qdrant_port": config.qdrant_port,
            "collection_name": config.collection_name,
            "model_name": config.model_name,
            "device": config.device,
            "normalize_embeddings": config.normalize_embeddings,
            "top_k": config.top_k,
            "dense_weight": config.dense_weight,
            "sparse_weight": config.sparse_weight,
        },
    )

    # ------------------------------------------------------------------
    # Single-question mode
    # ------------------------------------------------------------------
    if args.question:
        log_step(logger, "加载并规范化 PICO 输入")
        pico = load_pico_from_args(args)

        log_step(logger, "执行混合检索（单问题模式）")
        entry = run_single_question(args.question, pico, config, logger)

        log_step(logger, f"保存检索结果到 {args.output_file}")
        append_results_to_json(args.output_file, entry)
        log_step(logger, "完成混合检索")
        return

    # ------------------------------------------------------------------
    # Batch mode (--questions-file)
    # ------------------------------------------------------------------
    log_step(logger, f"从文件加载问题列表: {args.questions_file}")
    questions = load_questions_from_file(args.questions_file)
    logger.info("共加载 %d 个问题", len(questions))

    total = len(questions)
    processed = 0

    for idx, item in enumerate(questions, start=1):
        qid = item.get("question_id", f"batch-{idx}")
        question_text = str(item.get("question", "")).strip()
        raw_pico = item.get("pico", {})
        pico = normalize_pico(raw_pico)

        log_step(logger, f"[{idx}/{total}] 检索问题: {qid}")
        logger.info("question=%s", question_text[:120])

        try:
            entry = run_single_question(
                question_text,
                pico,
                config,
                logger,
                question_id=qid,
            )
        except Exception as exc:
            logger.error("[%d/%d] 检索失败: %s — %s", idx, total, qid, exc)
            continue

        log_step(logger, f"[{idx}/{total}] 保存结果: {qid}")
        append_results_to_json(args.output_file, entry)
        processed += 1

    logger.info(
        "批量检索完成: 总计 %d, 已处理 %d",
        total,
        processed,
    )
    log_step(logger, "完成批量混合检索")


if __name__ == "__main__":
    main()
