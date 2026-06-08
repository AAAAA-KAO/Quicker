"""
Phase5：推荐意见形成脚本。

运行示例：
    conda run -n quicker python Phase5-recommendation_formulation.py \
        --YOUR_CONFIG_PATH config/config.json

脚本功能：
    读取 Phase4 证据评价结果，调用推荐形成模型生成最终推荐意见，并保存包含
    evidence_assessment_results、supplementary_information 和 final_result 的 JSON。
    命令行参数会覆盖配置文件中的同名配置。

输入：
    1. --YOUR_CONFIG_PATH 指向的 JSON 配置文件。
    2. {YOUR_QUESTION_DECOMPOSITION_PATH}/PICO_Information.json。
    3. {YOUR_RECOMMENDATION_FORMATION_PATH}/outcomeinfo 或该目录下匹配 PICO 的
       outcomeinfo_PICO*.json；也可通过参数先从 Phase4 复制。

输出：
    {YOUR_RECOMMENDATION_FORMATION_PATH}/quicker_data(PICO_IDX{pico_idx})_{timestamp}.json

命令行参数：
    --YOUR_CONFIG_PATH：必填，项目配置文件路径。
    --YOUR_QUESTION_DECOMPOSITION_PATH：可选，Phase1 输出目录。
    --YOUR_LITERATURE_SEARCH_PATH：可选，Phase2 工作目录。
    --YOUR_STUDY_SELECTION_PATH：可选，Phase3 输出目录。
    --YOUR_EVIDENCE_ASSESSMENT_PATH：可选，Phase4 输出目录。
    --YOUR_PAPER_LIBRARY_PATH：可选，PDF 本地库目录。
    --YOUR_RECOMMENDATION_FORMATION_PATH：可选，Phase5 工作与输出目录。
    --disease：可选，疾病/主题名称。
    --pico_idx：可选，PICO 编号；未传入或为 auto 时根据配置生成。
    --overall_certainty：可选，总体证据确定性。
    --supplementary_information：可选，推荐形成时补充给模型的信息。
    --transfer_evidence_assessment_files / --no-transfer_evidence_assessment_files：可选，
        是否先从 Phase4 复制 paperinfo/outcomeinfo。
    --print_state：可选，运行前打印临床问题与 comparator 列表。
    --LOG_DIR：可选，日志目录；未传入时读取 config.logging.log_dir。
    --DOTENV_PATH：可选，.env 文件路径；未传入时不主动加载 .env。
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from utils.cli_config import (
    add_common_config_args,
    choose,
    config_path,
    get_nested,
    load_config,
    phase_config,
    prepare_environment,
    resolve_pico_idx,
)


def transfer_outcome_and_paperinfo(
    source_dir_path: str,
    target_dir_path: str,
    only_index: Optional[str] = None,
) -> None:
    """
    Transfer the outcomeinfo and paperinfo folders from source_dir_path to
    target_dir_path. If only_index is provided, only files containing
    '_PICO{only_index}' are copied.
    """
    source_dir = Path(source_dir_path)
    target_dir = Path(target_dir_path)

    for folder_name in ("outcomeinfo", "paperinfo"):
        source_folder = source_dir / folder_name
        target_folder = target_dir / folder_name
        if not source_folder.exists():
            continue

        if only_index:
            target_folder.mkdir(parents=True, exist_ok=True)
            for item in source_folder.iterdir():
                if f"_PICO{only_index}" not in item.name:
                    continue
                target_item = target_folder / item.name
                if item.is_dir():
                    shutil.copytree(item, target_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target_item)
        else:
            shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)


def iter_outcomeinfo_files(
    recommendation_formation_path: str,
    pico_idx: str,
) -> Iterable[Path]:
    """Yield matching outcomeinfo files in the notebook's expected locations."""
    base_path = Path(recommendation_formation_path)
    pattern = f"outcomeinfo_PICO{pico_idx}*.json"
    seen = set()

    for folder in (base_path, base_path / "outcomeinfo"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def load_evidence_assessment_results(
    recommendation_formation_path: str,
    pico_idx: str,
    overall_certainty: str,
) -> Dict[str, dict]:
    evidence_assessment_results = {}
    outcomeinfo_files = list(
        iter_outcomeinfo_files(recommendation_formation_path, pico_idx)
    )

    if not outcomeinfo_files:
        raise FileNotFoundError(
            "No outcomeinfo file found. Expected files like "
            f"outcomeinfo_PICO{pico_idx}.json under "
            f"{recommendation_formation_path} or its outcomeinfo subfolder."
        )

    for file_path in outcomeinfo_files:
        with file_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

        if not isinstance(result, list) or not result:
            raise ValueError(f"{file_path} should contain a non-empty JSON list.")

        comparator = result[0].get("comparator")
        if not comparator:
            raise ValueError(f"{file_path} does not contain result[0]['comparator'].")

        if comparator in evidence_assessment_results:
            logging.warning("Overwriting duplicated comparator: %s", comparator)

        evidence_assessment_results[comparator] = {
            "outcome_list": result,
            "overall_certainty": overall_certainty,
        }

    return evidence_assessment_results


def load_clinical_question(question_decomposition_path: str, pico_idx: str) -> str:
    pico_file_path = Path(question_decomposition_path) / "PICO_Information.json"
    if not pico_file_path.exists():
        raise FileNotFoundError(f"PICO information file not found: {pico_file_path}")

    with pico_file_path.open("r", encoding="utf-8") as f:
        pico_list = json.load(f)

    for pico in pico_list:
        if str(pico.get("Index")) == pico_idx:
            return pico["Question"]

    raise ValueError(f"PICO index {pico_idx} not found in {pico_file_path}")


def import_recommendation_dependencies():
    try:
        from langchain_openai import ChatOpenAI
        from utils.Recommendation_formation.recommendation import Recommendation
    except ModuleNotFoundError as exc:
        missing_name = exc.name or str(exc)
        raise SystemExit(
            "Missing Python dependency: "
            f"{missing_name}\n"
            "Install the project dependencies first, preferably in Python 3.11:\n"
            "  python3.11 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI, Recommendation


def get_recommendation_model(config_path: str):
    ChatOpenAI, _ = import_recommendation_dependencies()

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    model_config = config["model"]["recommendation_formation_model"]
    provider = model_config["provider"]
    if provider != "OpenAI":
        raise NotImplementedError(f"Provider {provider} is not implemented")

    model_kwargs = {
        "openai_api_key": model_config["API_KEY"],
        "base_url": model_config["BASE_URL"],
        "model": model_config["model_name"],
    }
    if model_config.get("temperature") is not None:
        model_kwargs["temperature"] = model_config["temperature"]
    return ChatOpenAI(**model_kwargs)


def save_result(
    output_dir: str,
    pico_idx: str,
    data: dict,
) -> Path:
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = save_dir / f"quicker_data(PICO_IDX{pico_idx})_{timestamp}.json"
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    return file_path


def resolve_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    phase_settings = phase_config(config, "phase5_recommendation_formulation")
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
    args.YOUR_STUDY_SELECTION_PATH = choose(
        args.YOUR_STUDY_SELECTION_PATH,
        config_path(config, "study_selection"),
        "YOUR_STUDY_SELECTION_PATH/pipeline.paths.study_selection",
    )
    args.YOUR_EVIDENCE_ASSESSMENT_PATH = choose(
        args.YOUR_EVIDENCE_ASSESSMENT_PATH,
        config_path(config, "evidence_assessment"),
        "YOUR_EVIDENCE_ASSESSMENT_PATH/pipeline.paths.evidence_assessment",
    )
    args.YOUR_PAPER_LIBRARY_PATH = choose(
        args.YOUR_PAPER_LIBRARY_PATH,
        config_path(config, "paper_library"),
        "YOUR_PAPER_LIBRARY_PATH/pipeline.paths.paper_library",
    )
    args.YOUR_RECOMMENDATION_FORMATION_PATH = choose(
        args.YOUR_RECOMMENDATION_FORMATION_PATH,
        config_path(config, "recommendation_formation"),
        "YOUR_RECOMMENDATION_FORMATION_PATH/pipeline.paths.recommendation_formation",
    )
    args.disease = choose(
        args.disease,
        get_nested(config, ("pipeline", "disease")),
        "disease/pipeline.disease",
    )
    args.pico_idx = resolve_pico_idx(args.pico_idx, config)
    args.overall_certainty = choose(
        args.overall_certainty,
        phase_settings.get("overall_certainty"),
        "overall_certainty/pipeline.phase5_recommendation_formulation.overall_certainty",
    )
    args.supplementary_information = choose(
        args.supplementary_information,
        phase_settings.get("supplementary_information"),
        "supplementary_information/pipeline.phase5_recommendation_formulation.supplementary_information",
        required=False,
    )
    args.transfer_evidence_assessment_files = choose(
        args.transfer_evidence_assessment_files,
        phase_settings.get("transfer_evidence_assessment_files"),
        "transfer_evidence_assessment_files/pipeline.phase5_recommendation_formulation.transfer_evidence_assessment_files",
    )
    if args.print_state is None:
        args.print_state = False
    if args.supplementary_information is None:
        args.supplementary_information = ""
    return args


def run(args: argparse.Namespace) -> None:
    config = load_config(args.YOUR_CONFIG_PATH)
    args = resolve_args(args, config)
    prepare_environment(args, config)

    if args.transfer_evidence_assessment_files:
        transfer_outcome_and_paperinfo(
            source_dir_path=args.YOUR_EVIDENCE_ASSESSMENT_PATH,
            target_dir_path=args.YOUR_RECOMMENDATION_FORMATION_PATH,
            only_index=args.pico_idx,
        )

    evidence_assessment_results = load_evidence_assessment_results(
        recommendation_formation_path=args.YOUR_RECOMMENDATION_FORMATION_PATH,
        pico_idx=args.pico_idx,
        overall_certainty=args.overall_certainty,
    )
    clinical_question = load_clinical_question(
        question_decomposition_path=args.YOUR_QUESTION_DECOMPOSITION_PATH,
        pico_idx=args.pico_idx,
    )

    _, Recommendation = import_recommendation_dependencies()
    model = get_recommendation_model(args.YOUR_CONFIG_PATH)
    recommendation = Recommendation(
        disease=args.disease,
        evidence_assessment_result=evidence_assessment_results,
        model=model,
        supplementary_information=args.supplementary_information,
    )

    if args.print_state:
        print(f"clinical_question: {clinical_question}")
        print(f"comparators: {list(evidence_assessment_results.keys())}")

    final_result = recommendation.get_recommendation()
    output_data = {
        "disease": args.disease,
        "clinical_question": clinical_question,
        "pico_idx": args.pico_idx,
        "evidence_assessment_results": evidence_assessment_results,
        "supplementary_information": args.supplementary_information,
        "final_result": final_result,
    }
    output_path = save_result(
        output_dir=args.YOUR_RECOMMENDATION_FORMATION_PATH,
        pico_idx=args.pico_idx,
        data=output_data,
    )
    print(f"Saved recommendation formulation result to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recommendation formulation.")
    add_common_config_args(parser)
    parser.add_argument(
        "--YOUR_QUESTION_DECOMPOSITION_PATH",
        type=str,
        default=None,
        help="Question decomposition folder.",
    )
    parser.add_argument(
        "--YOUR_LITERATURE_SEARCH_PATH",
        type=str,
        default=None,
        help="Literature search folder.",
    )
    parser.add_argument(
        "--YOUR_STUDY_SELECTION_PATH",
        type=str,
        default=None,
        help="Study selection folder.",
    )
    parser.add_argument(
        "--YOUR_EVIDENCE_ASSESSMENT_PATH",
        type=str,
        default=None,
        help="Evidence assessment folder.",
    )
    parser.add_argument(
        "--YOUR_PAPER_LIBRARY_PATH",
        type=str,
        default=None,
        help="Paper library folder.",
    )
    parser.add_argument(
        "--YOUR_RECOMMENDATION_FORMATION_PATH",
        type=str,
        default=None,
        help="Recommendation formulation folder.",
    )
    parser.add_argument(
        "--disease",
        type=str,
        default=None,
        help="Disease name or clinical topic.",
    )
    parser.add_argument(
        "--pico_idx",
        type=str,
        default=None,
        help="PICO index from PICO_Information.json.",
    )
    parser.add_argument(
        "--overall_certainty",
        type=str,
        default=None,
        help='Overall certainty, e.g. "VERY LOW", "LOW", "MODERATE", or "HIGH".',
    )
    parser.add_argument(
        "--supplementary_information",
        type=str,
        default=None,
        help="Supplementary information for recommendation formulation.",
    )
    parser.add_argument(
        "--transfer_evidence_assessment_files",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Copy matching outcomeinfo/paperinfo files before running.",
    )
    parser.add_argument(
        "--print_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print Quicker state before recommendation formulation.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
