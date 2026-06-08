"""
Phase2：文献检索脚本。

运行示例：
    conda run -n quicker python Phase2-literature_search.py \
        --YOUR_CONFIG_PATH config/config.json

脚本功能：
    读取 Phase1 输出的 PICO 信息，调用 PubMed 检索流程生成检索策略与检索结果，
    并对检索结果进行启发式过滤（去重、移除无摘要记录、移除配置指定的出版类型）。
    命令行参数会覆盖配置文件中的同名配置。

输入：
    1. --YOUR_CONFIG_PATH 指向的 JSON 配置文件。
    2. {YOUR_QUESTION_DECOMPOSITION_PATH}/PICO_Information.json。

输出：
    1. PubMed 检索策略与原始结果，保存到 save_base/model/use_agent_* 目录。
    2. {YOUR_DATASET_PATH}/quicker_data(PICO_IDX{pico_idx})_ls.json，供 Phase3 使用。

命令行参数：
    --YOUR_CONFIG_PATH：必填，项目配置文件路径。
    --YOUR_QUESTION_DECOMPOSITION_PATH：可选，Phase1 输出目录。
    --YOUR_DATASET_PATH：可选，数据集根目录。
    --save_base：可选，检索结果保存根目录；未传入时由配置中的 Literature_Search
        路径和 search_backend 生成。
    --disease：可选，疾病/主题名称。
    --pico_idx：可选，PICO 编号；未传入或为 auto 时根据配置生成。
    --additional_parameters：可选，JSON 字符串或 JSON 文件路径，传给 PubMed API。
    --filters：可选，JSON 字符串或 JSON 文件路径，传给检索器的过滤条件。
    --use_agent / --no-use_agent：可选，是否使用 agentic 检索策略生成。
    --invalid_publication_types：可选，JSON 字符串或 JSON 文件路径，指定需剔除的出版类型列表。
    --quicker_data_output_path：可选，Phase2 汇总输出 JSON 路径。
    --LOG_DIR：可选，日志目录；未传入时读取 config.logging.log_dir。
    --DOTENV_PATH：可选，.env 文件路径；未传入时不主动加载 .env。
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from utils.cli_config import (
    add_common_config_args,
    choose,
    choose_json,
    config_path,
    get_nested,
    load_config,
    phase_config,
    prepare_environment,
    resolve_dataset_path,
    resolve_pico_idx,
)


def load_pico(question_decomposition_path: str, pico_idx: str) -> dict:
    pico_path = Path(question_decomposition_path) / "PICO_Information.json"
    if not pico_path.exists():
        raise FileNotFoundError(f"PICO information file not found: {pico_path}")

    question_decomposition_data = pd.read_json(pico_path, dtype={"Index": str})
    matched = question_decomposition_data[
        question_decomposition_data["Index"].astype(str) == str(pico_idx)
    ]
    if matched.empty:
        raise ValueError(f"PICO index {pico_idx} not found in {pico_path}")
    return matched.to_dict(orient="records")[0]


def resolve_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    phase_settings = phase_config(config, "phase2_literature_search")
    search_backend = choose(
        args.search_backend,
        phase_settings.get("search_backend"),
        "search_backend/pipeline.phase2_literature_search.search_backend",
    )
    if search_backend != "pubmed":
        raise NotImplementedError(f"Search backend {search_backend} is not implemented")

    args.search_backend = search_backend
    args.YOUR_DATASET_PATH = resolve_dataset_path(args, config)
    args.YOUR_QUESTION_DECOMPOSITION_PATH = choose(
        args.YOUR_QUESTION_DECOMPOSITION_PATH,
        config_path(config, "question_decomposition"),
        "YOUR_QUESTION_DECOMPOSITION_PATH/pipeline.paths.question_decomposition",
    )
    args.YOUR_LITERATURE_SEARCH_PATH = choose(
        args.YOUR_LITERATURE_SEARCH_PATH,
        config_path(config, "literature_search"),
        "YOUR_LITERATURE_SEARCH_PATH/pipeline.paths.literature_search",
    )
    args.save_base = choose(
        args.save_base,
        str(Path(args.YOUR_LITERATURE_SEARCH_PATH) / search_backend / "Results"),
        "save_base",
    )
    args.disease = choose(
        args.disease,
        get_nested(config, ("pipeline", "disease")),
        "disease/pipeline.disease",
    )
    args.pico_idx = resolve_pico_idx(args.pico_idx, config)
    args.additional_parameters = choose_json(
        args.additional_parameters,
        phase_settings.get("additional_parameters"),
        "additional_parameters",
        expected_type=dict,
    )
    args.filters = choose_json(
        args.filters,
        phase_settings.get("filters"),
        "filters",
        expected_type=dict,
    )
    args.use_agent = choose(
        args.use_agent,
        phase_settings.get("use_agent"),
        "use_agent/pipeline.phase2_literature_search.use_agent",
    )
    args.invalid_publication_types = choose_json(
        args.invalid_publication_types,
        phase_settings.get("invalid_publication_types"),
        "invalid_publication_types",
        expected_type=list,
    )
    args.quicker_data_output_path = choose(
        args.quicker_data_output_path,
        str(Path(args.YOUR_DATASET_PATH) / f"quicker_data(PICO_IDX{args.pico_idx})_ls.json"),
        "quicker_data_output_path",
    )
    return args


def filter_search_results(search_results: list[dict], invalid_publication_types: list[str]) -> list[dict]:
    records_with_abstract = [
        record for record in search_results if record.get("Abstract") is not None
    ]

    seen_paper_indices = set()
    deduplicated_records = []
    for record in records_with_abstract:
        paper_index = record.get("Paper_Index")
        if paper_index in seen_paper_indices:
            continue
        seen_paper_indices.add(paper_index)
        deduplicated_records.append(record)

    invalid_types = set(invalid_publication_types)
    return [
        record
        for record in deduplicated_records
        if not any(
            publication_type in invalid_types
            for publication_type in record.get("Publication Types", [])
        )
    ]


def run(args: argparse.Namespace) -> None:
    from utils.Evidence_Retrieval.pubmedretrieval import PubMedRetrieval

    config = load_config(args.YOUR_CONFIG_PATH)
    args = resolve_args(args, config)
    log_dir = prepare_environment(args, config)

    from utils.logger import get_detail_logger, get_workflow_logger, setup_loggers

    setup_loggers(
        log_file=os.path.join(
            log_dir,
            Path(args.YOUR_DATASET_PATH).name,
            "Literature_Search",
            f"{args.pico_idx}.log",
        )
    )
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    model_config = get_nested(config, ("model", "literature_search_model"), {})
    original_qd_dict = load_pico(args.YOUR_QUESTION_DECOMPOSITION_PATH, args.pico_idx)

    clinical_question = original_qd_dict["Question"]
    population = original_qd_dict["P"]
    intervention = original_qd_dict["I"]
    comparison = original_qd_dict["C"]
    outcome = original_qd_dict.get("O", {})
    model_name = model_config["model_name"]
    save_path = os.path.join(
        args.save_base,
        model_name,
        "use_agent_" + str(args.use_agent),
    )

    wf_logger.info(
        "Initializing PubMedRetrieval for %s with PICO %s",
        args.disease,
        args.pico_idx,
    )
    retriever = PubMedRetrieval(
        disease=args.disease,
        clinical_question=clinical_question,
        population=population,
        intervention=intervention,
        comparison=comparison,
        api_key=model_config["API_KEY"],
        base_url=model_config["BASE_URL"],
        model_setting={
            "search_term_formation": model_name,
            "search_strategy_formation": model_name,
        },
        use_agent=args.use_agent,
        save_path=save_path,
        pico_idx=args.pico_idx,
        filters=args.filters,
        additional_parameters=args.additional_parameters,
    )

    wf_logger.info("Executing PubMedRetrieval")
    retriever.run()
    dt_logger.info("Search terms: %s", retriever.search_terms)

    save_results_path = Path(save_path) / f"PICO{args.pico_idx}.json"
    with save_results_path.open("r", encoding="utf-8") as file:
        raw_search_results = json.load(file)

    wf_logger.info("Filter duplicate, no-abstract, or invalid publication-type records")
    dt_logger.info("Total records: %s", len(raw_search_results))
    search_results = filter_search_results(
        raw_search_results,
        args.invalid_publication_types,
    )
    dt_logger.info("Records after filtering: %s", len(search_results))

    quicker_data = {
        "disease": args.disease,
        "clinical_question": clinical_question,
        "pico_idx": args.pico_idx,
        "population": population,
        "intervention": intervention,
        "comparison": comparison,
        "outcome": outcome,
        "search_results": search_results,
    }

    output_path = Path(args.quicker_data_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(quicker_data, file, indent=4, ensure_ascii=False)
    print(f"Quicker literature-search data saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2 literature search.")
    add_common_config_args(parser)
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", default=None)
    parser.add_argument("--YOUR_DATASET_PATH", default=None)
    parser.add_argument("--YOUR_LITERATURE_SEARCH_PATH", default=None)
    parser.add_argument("--save_base", default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--pico_idx", default=None)
    parser.add_argument("--search_backend", default=None)
    parser.add_argument("--additional_parameters", default=None)
    parser.add_argument("--filters", default=None)
    parser.add_argument(
        "--use_agent",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--invalid_publication_types", default=None)
    parser.add_argument("--quicker_data_output_path", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
