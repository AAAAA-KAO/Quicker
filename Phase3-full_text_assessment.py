"""
Phase3 第二阶段：全文评估脚本。

运行示例：
    conda run -n quicker python Phase3-full_text_assessment.py \
        --YOUR_CONFIG_PATH config/config.json

脚本功能：
    读取 Phase3 第一阶段输出的题录纳入论文列表，检查 Paper_Library 中是否已有
    所需 PDF。若 PDF 缺失，脚本会写出缺失 PDF 清单并提示用户自行下载；默认
    不继续执行全文评估，也不会自动下载 PDF。PDF 齐全后，脚本运行全文评估，
    生成 Study_Selection 阶段的 paperinfo 和 outcomeinfo。

输入：
    1. --YOUR_CONFIG_PATH 指向的 JSON 配置文件。
    2. record_included_input_path 指向的题录纳入论文列表 JSON。
    3. {YOUR_DATASET_PATH}/quicker_data(PICO_IDX{pico_idx})_ls.json。
    4. Paper_Library/PICO{pico_idx}/{paper_uid}/ 下的本地 PDF。

输出：
    1. {YOUR_STUDY_SELECTION_PATH}/paperinfo/paperinfo_PICO*.json。
    2. {YOUR_STUDY_SELECTION_PATH}/outcomeinfo/outcomeinfo_PICO*.json。
    3. Study_Selection/Results/full_text_assessment 下的 QuickerData 运行结果。
    4. 若缺失 PDF，则输出 missing_pdf_json_path 清单。

命令行参数：
    --YOUR_CONFIG_PATH：必填，项目配置文件路径。
    --YOUR_DATASET_PATH：可选，数据集根目录。
    --YOUR_QUESTION_DECOMPOSITION_PATH：可选，Phase1 输出目录。
    --YOUR_LITERATURE_SEARCH_PATH：可选，Phase2 工作目录。
    --YOUR_STUDY_SELECTION_PATH：可选，Phase3 工作目录。
    --YOUR_PAPER_LIBRARY_PATH：可选，PDF 本地库目录。
    --YOUR_REPORTS_PATH：可选，报告输出目录。
    --disease：可选，疾病/主题名称。
    --pico_idx：可选，PICO 编号；未传入或为 auto 时根据配置生成。
    --full_text_assessment_method：可选，全文评估方法。
    --reupdate_component_list：可选，JSON 字符串或 JSON 文件路径，指定需要重新抽取的 PICO 组件。
    --study：可选，JSON 字符串或 JSON 文件路径，指定研究类型列表。
    --inclusion_criteria：可选，纳入标准。
    --exclusion_criteria：可选，排除标准。
    --quickerdata_ls_path：可选，Phase2 汇总输出 JSON 路径。
    --record_included_input_path：可选，Phase3 第一阶段输出的题录纳入论文列表。
    --missing_pdf_json_path：可选，缺失 PDF 清单 JSON 输出路径。
    --missing_pdf_markdown_path：可选，缺失 PDF 清单 Markdown 输出路径。
    --stop_when_missing_pdf / --no-stop_when_missing_pdf：可选，缺失 PDF 时是否停止。
    --reuse_existing_outputs / --no-reuse_existing_outputs：可选，已有全文评估输出时
        是否直接复用，避免重新抽取全文。
    --print_state：可选，打印 QuickerData 阶段状态。
    --LOG_DIR：可选，日志目录；未传入时读取 config.logging.log_dir。
    --DOTENV_PATH：可选，.env 文件路径；未传入时不主动加载 .env。
"""

import argparse
import os
from pathlib import Path

from utils.cli_config import (
    add_common_config_args,
    choose,
    choose_json,
    config_path,
    get_nested,
    load_config,
    load_json_file,
    load_valid_json_file,
    phase_config,
    prepare_environment,
    resolve_dataset_path,
    resolve_pico_idx,
    resolve_reports_path,
)
from utils.pdf_manifest import (
    build_pdf_manifest,
    resolve_manifest_path,
    write_pdf_manifest,
)


def load_pico(question_decomposition_path: str, pico_idx: str) -> dict:
    pico_path = Path(question_decomposition_path) / "PICO_Information.json"
    if not pico_path.exists():
        raise FileNotFoundError(f"PICO information file not found: {pico_path}")
    pico_list = load_json_file(pico_path)
    for pico in pico_list:
        if str(pico.get("Index")) == str(pico_idx):
            return pico
    raise ValueError(f"PICO index {pico_idx} not found in {pico_path}")


def load_record_included_list(path: str) -> list[dict]:
    data = load_json_file(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("record_included_studies"), list):
        return data["record_included_studies"]
    raise ValueError(
        "record_included_input_path must contain a JSON list or a JSON object "
        "with record_included_studies."
    )


def full_text_results_dir(
    study_selection_path: str,
    full_text_assessment_method: str,
    pico_idx: str,
) -> Path:
    return (
        Path(study_selection_path)
        / "Results"
        / "full_text_assessment"
        / full_text_assessment_method
        / pico_idx
    )


def valid_full_text_result(data: dict, pico_idx: str) -> bool:
    required_keys = {
        "record_included_studies",
        "full_text_included_studies",
        "total_outcome_list",
    }
    return (
        str(data.get("pico_idx")) == str(pico_idx)
        and required_keys.issubset(data.keys())
        and isinstance(data.get("record_included_studies"), list)
        and isinstance(data.get("full_text_included_studies"), list)
        and isinstance(data.get("total_outcome_list"), list)
    )


def paired_full_text_outputs(study_selection_path: str, pico_idx: str) -> list[tuple[Path, Path]]:
    base_path = Path(study_selection_path)
    paper_folder = base_path / "paperinfo"
    outcome_folder = base_path / "outcomeinfo"
    if not paper_folder.exists() or not outcome_folder.exists():
        return []

    paper_prefix = f"paperinfo_PICO{pico_idx}"
    outcome_prefix = f"outcomeinfo_PICO{pico_idx}"
    pairs = []
    for paper_path in sorted(paper_folder.glob(f"{paper_prefix}*.json")):
        if "_full_text_assessed_but_not_included" in paper_path.name:
            continue
        postfix = paper_path.stem[len(paper_prefix):]
        outcome_path = outcome_folder / f"{outcome_prefix}{postfix}.json"
        if (
            load_valid_json_file(paper_path, list) is not None
            and load_valid_json_file(outcome_path, list) is not None
        ):
            pairs.append((paper_path, outcome_path))
    return pairs


def find_reusable_full_text_result(
    study_selection_path: str,
    full_text_assessment_method: str,
    pico_idx: str,
) -> tuple[Path | None, list[tuple[Path, Path]]]:
    result_dir = full_text_results_dir(
        study_selection_path,
        full_text_assessment_method,
        pico_idx,
    )
    result_files = sorted(result_dir.glob(f"quicker_data(PICO_IDX{pico_idx})_*.json"))
    valid_results = [
        path
        for path in result_files
        if (
            (data := load_valid_json_file(path, dict)) is not None
            and valid_full_text_result(data, pico_idx)
        )
    ]
    if not valid_results:
        return None, []
    return valid_results[-1], paired_full_text_outputs(study_selection_path, pico_idx)


def resolve_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    phase_settings = phase_config(config, "phase3_study_selection")
    pdf_settings = get_nested(config, ("pipeline", "pdf_handling"), {}) or {}

    args.YOUR_DATASET_PATH = resolve_dataset_path(args, config)
    args.YOUR_REPORTS_PATH = resolve_reports_path(args, config)
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
    args.YOUR_PAPER_LIBRARY_PATH = choose(
        args.YOUR_PAPER_LIBRARY_PATH,
        config_path(config, "paper_library"),
        "YOUR_PAPER_LIBRARY_PATH/pipeline.paths.paper_library",
    )
    args.disease = choose(
        args.disease,
        get_nested(config, ("pipeline", "disease")),
        "disease/pipeline.disease",
    )
    args.pico_idx = resolve_pico_idx(args.pico_idx, config)
    args.full_text_assessment_method = choose(
        args.full_text_assessment_method,
        get_nested(config, ("study_selection", "full_text_assessment_method")),
        "full_text_assessment_method/study_selection.full_text_assessment_method",
    )
    args.reupdate_component_list = choose_json(
        args.reupdate_component_list,
        get_nested(config, ("study_selection", "reupdate_component_list")),
        "reupdate_component_list/study_selection.reupdate_component_list",
        expected_type=list,
    )
    args.study = choose_json(
        args.study,
        phase_settings.get("study"),
        "study/pipeline.phase3_study_selection.study",
        expected_type=list,
    )
    args.inclusion_criteria = choose(
        args.inclusion_criteria,
        phase_settings.get("inclusion_criteria"),
        "inclusion_criteria/pipeline.phase3_study_selection.inclusion_criteria",
        required=False,
    )
    args.exclusion_criteria = choose(
        args.exclusion_criteria,
        phase_settings.get("exclusion_criteria"),
        "exclusion_criteria/pipeline.phase3_study_selection.exclusion_criteria",
        required=False,
    )
    args.stop_when_missing_pdf = choose(
        args.stop_when_missing_pdf,
        pdf_settings.get("stop_when_missing_pdf"),
        "stop_when_missing_pdf/pipeline.pdf_handling.stop_when_missing_pdf",
    )
    args.reuse_existing_outputs = choose(
        args.reuse_existing_outputs,
        phase_settings.get("reuse_existing_outputs"),
        "reuse_existing_outputs/pipeline.phase3_study_selection.reuse_existing_outputs",
        required=False,
    )
    if args.reuse_existing_outputs is None:
        args.reuse_existing_outputs = True
    args.quickerdata_ls_path = choose(
        args.quickerdata_ls_path,
        str(Path(args.YOUR_DATASET_PATH) / f"quicker_data(PICO_IDX{args.pico_idx})_ls.json"),
        "quickerdata_ls_path",
    )
    args.record_included_input_path = choose(
        args.record_included_input_path,
        str(
            Path(args.YOUR_STUDY_SELECTION_PATH)
            / "record_included_studies"
            / f"record_included_PICO{args.pico_idx}.json"
        ),
        "record_included_input_path",
    )

    stage_name = "phase3_full_text_assessment"
    json_pattern = pdf_settings.get("missing_pdf_json")
    markdown_pattern = pdf_settings.get("missing_pdf_markdown")
    args.missing_pdf_json_path = choose(
        args.missing_pdf_json_path,
        resolve_manifest_path(args.YOUR_REPORTS_PATH, json_pattern, stage_name, args.pico_idx)
        if json_pattern
        else None,
        "missing_pdf_json_path",
    )
    args.missing_pdf_markdown_path = choose(
        args.missing_pdf_markdown_path,
        resolve_manifest_path(args.YOUR_REPORTS_PATH, markdown_pattern, stage_name, args.pico_idx)
        if markdown_pattern
        else None,
        "missing_pdf_markdown_path",
        required=False,
    )
    return args


def build_quicker(args: argparse.Namespace):
    from utils.General.quicker import Quicker, QuickerData

    quicker_data = QuickerData(disease=args.disease, pico_idx=args.pico_idx)
    return Quicker(
        config_path=args.YOUR_CONFIG_PATH,
        question_deconstruction_database_path=args.YOUR_QUESTION_DECOMPOSITION_PATH,
        literature_search_database_path=args.YOUR_LITERATURE_SEARCH_PATH,
        study_selection_database_path=args.YOUR_STUDY_SELECTION_PATH,
        evidence_assessment_database_path=None,
        quicker_data=quicker_data,
        paper_library_base=args.YOUR_PAPER_LIBRARY_PATH,
    )


def configure_quicker_for_cli(quicker, args: argparse.Namespace) -> None:
    quicker.config["study_selection"]["full_text_assessment_method"] = (
        args.full_text_assessment_method
    )
    quicker.config["study_selection"]["reupdate_component_list"] = (
        args.reupdate_component_list
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.YOUR_CONFIG_PATH)
    args = resolve_args(args, config)
    log_dir = prepare_environment(args, config)
    os.environ["QUICKER_DISABLE_PDF_DOWNLOAD"] = "1"

    from utils.logger import get_detail_logger, get_workflow_logger, setup_loggers

    setup_loggers(
        log_file=os.path.join(
            log_dir,
            Path(args.YOUR_DATASET_PATH).name,
            "Study_Selection",
            f"{args.pico_idx}_full_text_assessment.log",
        )
    )
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    if args.reuse_existing_outputs:
        reusable_result, output_pairs = find_reusable_full_text_result(
            study_selection_path=args.YOUR_STUDY_SELECTION_PATH,
            full_text_assessment_method=args.full_text_assessment_method,
            pico_idx=args.pico_idx,
        )
        if reusable_result is not None:
            wf_logger.info("Reuse existing full text assessment result: %s", reusable_result)
            print(f"Existing full text assessment result reused: {reusable_result}")
            for paper_path, outcome_path in output_pairs:
                print(f"Existing paper info: {paper_path}")
                print(f"Existing outcome info: {outcome_path}")
            return

    record_included_list = load_record_included_list(args.record_included_input_path)
    manifest = build_pdf_manifest(
        papers=record_included_list,
        paper_library_path=args.YOUR_PAPER_LIBRARY_PATH,
        pico_idx=args.pico_idx,
        stage="phase3_full_text_assessment",
    )
    json_output, markdown_output = write_pdf_manifest(
        manifest,
        json_path=args.missing_pdf_json_path,
        markdown_path=args.missing_pdf_markdown_path,
    )
    if manifest["missing_pdf_count"] and args.stop_when_missing_pdf:
        print(f"Missing PDF manifest saved to: {json_output}")
        if markdown_output:
            print(f"Missing PDF markdown guide saved to: {markdown_output}")
        raise SystemExit(
            "PDF 文件尚未齐全。请按清单下载并放置 PDF 后，再运行全文评估脚本。"
        )

    quickerdata_ls = load_json_file(args.quickerdata_ls_path)
    pico = load_pico(args.YOUR_QUESTION_DECOMPOSITION_PATH, args.pico_idx)
    quicker = build_quicker(args)
    configure_quicker_for_cli(quicker, args)
    from utils.General.quicker import QuickerStage

    data_dict = {
        "clinical_question": quickerdata_ls["clinical_question"],
        "population": quickerdata_ls["population"],
        "intervention": quickerdata_ls["intervention"],
        "comparison": quickerdata_ls["comparison"],
        "outcome": quickerdata_ls.get("outcome") or pico.get("O", {}),
        "study": args.study,
        "search_results": quickerdata_ls["search_results"],
        "search_config": {},
        "annotation": {},
    }
    quicker._add_data_to_quickerdata_for_test(
        stage=QuickerStage.LITERATURE_SEARCH,
        default_value=data_dict,
    )
    quicker.set_inclusion_exclusion_criteria(
        inclusion_criteria=args.inclusion_criteria or "",
        exclusion_criteria=args.exclusion_criteria or "",
    )

    if args.print_state:
        print(quicker.quicker_data.check_stage_state())
        print(quicker.quicker_data.not_none_data)

    wf_logger.info("Run full text assessment for PICO %s", args.pico_idx)
    record_included_list, full_text_included_list, total_outcome_list = (
        quicker.select_studies_by_full_text_assessment(record_included_list)
    )
    quicker.quicker_data.update_data(
        {
            "record_included_studies": record_included_list,
            "full_text_included_studies": full_text_included_list,
            "total_outcome_list": total_outcome_list,
        }
    )

    results_save_path = (
        Path(args.YOUR_STUDY_SELECTION_PATH)
        / "Results"
        / "full_text_assessment"
        / args.full_text_assessment_method
        / args.pico_idx
    )
    results_save_path.mkdir(parents=True, exist_ok=True)
    quicker.quicker_data.to_json(str(results_save_path))
    dt_logger.info("Full text assessment output folder: %s", results_save_path)

    print(f"Full text assessment results saved under: {results_save_path}")
    print(f"Paper info folder: {Path(args.YOUR_STUDY_SELECTION_PATH) / 'paperinfo'}")
    print(f"Outcome info folder: {Path(args.YOUR_STUDY_SELECTION_PATH) / 'outcomeinfo'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase3 full text assessment.")
    add_common_config_args(parser)
    parser.add_argument("--YOUR_DATASET_PATH", default=None)
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", default=None)
    parser.add_argument("--YOUR_LITERATURE_SEARCH_PATH", default=None)
    parser.add_argument("--YOUR_STUDY_SELECTION_PATH", default=None)
    parser.add_argument("--YOUR_PAPER_LIBRARY_PATH", default=None)
    parser.add_argument("--YOUR_REPORTS_PATH", default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--pico_idx", default=None)
    parser.add_argument("--full_text_assessment_method", default=None)
    parser.add_argument("--reupdate_component_list", default=None)
    parser.add_argument("--study", default=None)
    parser.add_argument("--inclusion_criteria", default=None)
    parser.add_argument("--exclusion_criteria", default=None)
    parser.add_argument("--quickerdata_ls_path", default=None)
    parser.add_argument("--record_included_input_path", default=None)
    parser.add_argument("--missing_pdf_json_path", default=None)
    parser.add_argument("--missing_pdf_markdown_path", default=None)
    parser.add_argument(
        "--stop_when_missing_pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--reuse_existing_outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--print_state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
