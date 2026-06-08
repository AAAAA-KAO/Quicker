"""Build dense and sparse retrieval indexes for the clinical QA knowledge base.

功能：
    读取临床指南问答知识库 JSON，使用 sentence-transformers 加载
    HuggingFace 模型 BAAI/bge-m3 为 search_text 批量生成 1024 维稠密向量，
    写入本地 Qdrant collection；同时使用 rank_bm25 构建 search_text 的
    BM25 稀疏索引并通过 pickle 持久化。索引完成后，脚本会用一条测试查询
    分别执行稠密检索、BM25 检索和简单加权混合检索，并打印前 5 个结果。

输入：
    --input-file 指向 mimic_cpg_knowledge_base.json，默认
    results/mimic_cpg_knowledge_base.json。文件应为问答字典列表，每个字典
    至少包含 search_text、question_id、question、answer、disease、topic、
    pico 和 source 字段。

输出：
    1. Qdrant collection，默认名称 clinical_qa_dense。
    2. 本地 BM25 pickle 文件，默认 results/clinical_qa_bm25.pkl。
    3. 终端打印 dense、BM25、hybrid 三种检索方式的前 k 个结果。

在项目根目录运行：
    conda run -n quicker python src/build_retrieval_index.py

命令行参数：
    --input-file: 知识库 JSON 路径，默认 results/mimic_cpg_knowledge_base.json。
    --bm25-output-file: BM25 pickle 保存路径，默认 results/clinical_qa_bm25.pkl。
    --log-file: 日志文件路径；不传则写入 logs/build_retrieval_index.log。
    --model-name: sentence-transformers 模型名，默认 BAAI/bge-m3。
    --device: 模型运行设备，默认 cpu。
    --encode-batch-size: search_text 批量编码大小，默认 16。
    --query-batch-size: 查询编码批量大小，默认 1。
    --normalize-embeddings: 是否归一化向量，默认开启，适配余弦检索。
    --qdrant-url: Qdrant URL；设置后优先使用该 URL，默认空。
    --qdrant-host: Qdrant host，默认 localhost。
    --qdrant-port: Qdrant port，默认 6333。
    --collection-name: Qdrant collection 名称，默认 clinical_qa_dense。
    --vector-size: 向量维度，默认 1024。
    --distance: Qdrant 距离函数，默认 Cosine。
    --qdrant-upsert-batch-size: Qdrant 批量写入大小，默认 64。
    --recreate-collection: 重建 collection，默认开启。
    --no-recreate-collection: 不删除已有 collection，直接 upsert。
    --test-query: 索引完成后的测试查询。
    --top-k: 每种检索方法返回条数，默认 5。
    --hybrid-candidate-k: 混合检索候选池大小，默认 20。
    --dense-weight: 混合检索中稠密分数权重，默认 0.6。
    --sparse-weight: 混合检索中 BM25 分数权重，默认 0.4。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.logging import log_step, setup_logging


DEFAULT_QUERY = "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?"


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qdrant dense vectors and BM25 sparse index for clinical QA retrieval."
    )
    parser.add_argument("--input-file", type=Path, default=Path("results/mimic_cpg_knowledge_base.json"))
    parser.add_argument("--bm25-output-file", type=Path, default=Path("results/clinical_qa_bm25.pkl"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection-name", default="clinical_qa_dense")
    parser.add_argument("--vector-size", type=int, default=1024)
    parser.add_argument("--distance", choices=["Cosine", "Dot", "Euclid", "Manhattan"], default="Cosine")
    parser.add_argument("--qdrant-upsert-batch-size", type=int, default=64)
    parser.set_defaults(recreate_collection=True)
    parser.add_argument("--recreate-collection", dest="recreate_collection", action="store_true")
    parser.add_argument("--no-recreate-collection", dest="recreate_collection", action="store_false")
    parser.add_argument("--test-query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hybrid-candidate-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--sparse-weight", type=float, default=0.4)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of QA dictionaries in {path}.")
    validate_records(records)
    return records


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    required_fields = {
        "search_text",
        "question_id",
        "question",
        "answer",
        "disease",
        "topic",
        "pico",
        "source",
    }
    for idx, record in enumerate(records):
        missing = required_fields - set(record)
        if missing:
            raise ValueError(f"Record {idx} is missing fields: {sorted(missing)}")
        if not str(record["search_text"]).strip():
            raise ValueError(f"Record {idx} has empty search_text.")


def load_embedding_model(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency sentence-transformers. Install it in the quicker environment first."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    model: Any,
    texts: Sequence[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> list[list[float]]:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32").tolist()


def make_qdrant_client(args: argparse.Namespace) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("Missing dependency qdrant-client. Install it in the quicker environment first.") from exc

    if args.qdrant_url:
        return QdrantClient(url=args.qdrant_url)
    return QdrantClient(host=args.qdrant_host, port=args.qdrant_port)


def recreate_collection_if_needed(client: Any, args: argparse.Namespace) -> None:
    from qdrant_client import models

    distance = getattr(models.Distance, args.distance.upper())
    vectors_config = models.VectorParams(size=args.vector_size, distance=distance)

    collection_exists = False
    try:
        collection_exists = bool(client.collection_exists(args.collection_name))
    except Exception:
        try:
            client.get_collection(args.collection_name)
            collection_exists = True
        except Exception:
            collection_exists = False

    if args.recreate_collection:
        if collection_exists:
            client.delete_collection(args.collection_name)
            collection_exists = False
    elif collection_exists:
        return

    if not collection_exists:
        client.create_collection(
            collection_name=args.collection_name,
            vectors_config=vectors_config,
        )


def make_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": record["question_id"],
        "question": record["question"],
        "disease": record["disease"],
        "topic": record["topic"],
        "answer": record["answer"],
    }


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def upload_to_qdrant(
    client: Any,
    collection_name: str,
    records: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    batch_size: int,
) -> None:
    from qdrant_client import models

    points = [
        models.PointStruct(
            id=idx,
            vector=list(embedding),
            payload=make_payload(record),
        )
        for idx, (record, embedding) in enumerate(zip(records, embeddings))
    ]
    for batch in batched(points, batch_size):
        client.upsert(collection_name=collection_name, points=list(batch))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())


def build_bm25(records: Sequence[dict[str, Any]]) -> tuple[Any, list[list[str]]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("Missing dependency rank_bm25. Install it in the quicker environment first.") from exc

    tokenized_corpus = [tokenize(record["search_text"]) for record in records]
    return BM25Okapi(tokenized_corpus), tokenized_corpus


def save_bm25_index(
    output_file: Path,
    bm25: Any,
    tokenized_corpus: Sequence[Sequence[str]],
    records: Sequence[dict[str, Any]],
    model_name: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "tokenized_corpus": list(tokenized_corpus),
        "records": list(records),
        "tokenizer": "regex:[a-z0-9]+(?:[-'][a-z0-9]+)? lowercased",
        "embedding_model": model_name,
    }
    with output_file.open("wb") as handle:
        pickle.dump(payload, handle)


def dense_search(
    client: Any,
    collection_name: str,
    query_embedding: Sequence[float],
    records: Sequence[dict[str, Any]],
    limit: int,
) -> list[SearchResult]:
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=list(query_embedding),
            limit=limit,
            with_payload=True,
        )
    except AttributeError:
        response = client.query_points(
            collection_name=collection_name,
            query=list(query_embedding),
            limit=limit,
            with_payload=True,
        )
        hits = response.points if hasattr(response, "points") else response

    results: list[SearchResult] = []
    for hit in hits:
        point_id = int(hit.id)
        payload = hit.payload or {}
        record = dict(records[point_id])
        record.update(payload)
        results.append(SearchResult(index=point_id, score=float(hit.score), record=record))
    return results


def bm25_search(
    bm25: Any,
    query: str,
    records: Sequence[dict[str, Any]],
    limit: int,
) -> list[SearchResult]:
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:limit]
    return [
        SearchResult(index=idx, score=float(scores[idx]), record=records[idx])
        for idx in ranked_indices
    ]


def normalize_score_map(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {idx: 1.0 if score > 0 else 0.0 for idx, score in score_map.items()}
    return {idx: (score - min_score) / (max_score - min_score) for idx, score in score_map.items()}


def hybrid_search(
    dense_results: Sequence[SearchResult],
    sparse_results: Sequence[SearchResult],
    records: Sequence[dict[str, Any]],
    dense_weight: float,
    sparse_weight: float,
    limit: int,
) -> list[SearchResult]:
    dense_scores = normalize_score_map({result.index: result.score for result in dense_results})
    sparse_scores = normalize_score_map({result.index: result.score for result in sparse_results})
    candidate_indices = set(dense_scores) | set(sparse_scores)

    hybrid_results = [
        SearchResult(
            index=idx,
            score=dense_weight * dense_scores.get(idx, 0.0) + sparse_weight * sparse_scores.get(idx, 0.0),
            record=records[idx],
        )
        for idx in candidate_indices
    ]
    return sorted(hybrid_results, key=lambda result: result.score, reverse=True)[:limit]


def print_results(title: str, results: Sequence[SearchResult]) -> None:
    print(f"\n=== {title} ===")
    for rank, result in enumerate(results, start=1):
        record = result.record
        answer = record.get("answer", "")
        if isinstance(answer, list):
            answer_text = " | ".join(str(item) for item in answer)
        else:
            answer_text = str(answer)
        print(f"[{rank}] score={result.score:.4f}")
        print(f"question: {record.get('question', '')}")
        print(f"answer: {answer_text}")
        print(f"disease: {record.get('disease', '')}")


def main() -> None:
    args = parse_args()
    logger = setup_logging("build_retrieval_index", log_file=args.log_file)

    log_step(logger, f"加载知识库：{args.input_file}")
    records = load_records(args.input_file)
    search_texts = [record["search_text"] for record in records]
    logger.debug("Loaded records=%d", len(records))

    log_step(logger, f"加载 embedding 模型：{args.model_name}")
    model = load_embedding_model(args.model_name, args.device)

    log_step(logger, "批量生成 search_text 稠密向量")
    embeddings = encode_texts(
        model=model,
        texts=search_texts,
        batch_size=args.encode_batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )
    if embeddings and len(embeddings[0]) != args.vector_size:
        raise ValueError(
            f"Embedding dim is {len(embeddings[0])}, but --vector-size is {args.vector_size}."
        )
    logger.debug("Embedding count=%d dim=%s", len(embeddings), len(embeddings[0]) if embeddings else None)

    log_step(logger, f"连接 Qdrant 并创建 collection：{args.collection_name}")
    qdrant_client = make_qdrant_client(args)
    recreate_collection_if_needed(qdrant_client, args)

    log_step(logger, "上传稠密向量到 Qdrant")
    upload_to_qdrant(
        client=qdrant_client,
        collection_name=args.collection_name,
        records=records,
        embeddings=embeddings,
        batch_size=args.qdrant_upsert_batch_size,
    )

    log_step(logger, "构建并保存 BM25 索引")
    bm25, tokenized_corpus = build_bm25(records)
    save_bm25_index(args.bm25_output_file, bm25, tokenized_corpus, records, args.model_name)
    logger.debug("BM25 index saved to %s", args.bm25_output_file)

    log_step(logger, "执行测试查询")
    query_embedding = encode_texts(
        model=model,
        texts=[args.test_query],
        batch_size=args.query_batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )[0]
    dense_candidate_limit = max(args.top_k, args.hybrid_candidate_k)
    sparse_candidate_limit = max(args.top_k, args.hybrid_candidate_k)

    dense_candidates = dense_search(
        client=qdrant_client,
        collection_name=args.collection_name,
        query_embedding=query_embedding,
        records=records,
        limit=dense_candidate_limit,
    )
    sparse_candidates = bm25_search(
        bm25=bm25,
        query=args.test_query,
        records=records,
        limit=sparse_candidate_limit,
    )
    hybrid_results = hybrid_search(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        records=records,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        limit=args.top_k,
    )

    print(f"\nTest query: {args.test_query}")
    print_results("Dense Qdrant Top Results", dense_candidates[: args.top_k])
    print_results("Sparse BM25 Top Results", sparse_candidates[: args.top_k])
    print_results("Hybrid Top Results", hybrid_results)
    log_step(logger, "完成：检索索引构建与测试查询")


if __name__ == "__main__":
    main()
