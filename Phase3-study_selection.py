"""
Phase3 第一阶段：题录筛选与 PDF 人工下载清单生成脚本。

运行示例：
    conda run -n quicker python Phase3-study_selection.py \
        --YOUR_CONFIG_PATH config/config.json

脚本功能：
    读取 Phase2 输出的检索结果，运行题录筛选（record screening），生成进入
    全文评估阶段的论文列表。脚本不会自动下载 PDF；它会检查 Paper_Library 中
    是否已有 PDF，并把缺失 PDF 信息写入 JSON 清单，提示用户自行下载。

输入：
    1. --YOUR_CONFIG_PATH 指向的 JSON 配置文件。
    2. {YOUR_DATASET_PATH}/quicker_data(PICO_IDX{pico_idx})_ls.json。
    3. {YOUR_QUESTION_DECOMPOSITION_PATH}/PICO_Information.json（用于补充 outcome）。

输出：
    1. record_included_output_path：题录筛选后进入全文评估的论文列表 JSON。
    2. missing_pdf_json_path：缺失 PDF 清单 JSON，默认保存在
        Study_Selection/record_included_studies。
    3. 题录筛选 CSV 结果，保存到 Study_Selection/Results/screening_records。

命令行参数：
    --YOUR_CONFIG_PATH：必填，项目配置文件路径。
    --YOUR_DATASET_PATH：可选，数据集根目录。
    --YOUR_QUESTION_DECOMPOSITION_PATH：可选，Phase1 输出目录。
    --YOUR_LITERATURE_SEARCH_PATH：可选，Phase2 工作目录。
    --YOUR_STUDY_SELECTION_PATH：可选，Phase3 工作目录。
    --YOUR_PAPER_LIBRARY_PATH：可选，PDF 本地库目录。
    --disease：可选，疾病/主题名称。
    --pico_idx：可选，PICO 编号；未传入或为 auto 时根据配置生成。
    --record_screening_method：可选，题录筛选方法。
    --exp_num：可选，题录筛选重复实验次数。
    --threshold：可选，达到多少次 Included 判定后进入全文评估。
    --study：可选，JSON 字符串或 JSON 文件路径，指定研究类型列表。
    --inclusion_criteria：可选，纳入标准。
    --exclusion_criteria：可选，排除标准。
    --quickerdata_ls_path：可选，Phase2 汇总输出 JSON 路径。
    --record_included_output_path：可选，题录纳入论文列表输出路径。
    --missing_pdf_json_path：可选，缺失 PDF 清单 JSON 输出路径。
    --reuse_existing_outputs / --no-reuse_existing_outputs：可选，是否复用已有题录筛选输出。
    --LOG_DIR：可选，日志目录；未传入时读取 config.logging.log_dir。
    --DOTENV_PATH：可选，.env 文件路径；未传入时不主动加载 .env。
"""

import argparse
import json
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
    phase_config,
    prepare_environment,
    resolve_dataset_path,
    resolve_pico_idx,
    write_json_file,
)
from utils.pdf_manifest import (
    build_pdf_manifest,
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


def resolve_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    phase_settings = phase_config(config, "phase3_study_selection")

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
    args.record_screening_method = choose(
        args.record_screening_method,
        get_nested(config, ("study_selection", "record_screening_method")),
        "record_screening_method/study_selection.record_screening_method",
    )
    args.exp_num = choose(
        args.exp_num,
        get_nested(config, ("study_selection", "exp_num")),
        "exp_num/study_selection.exp_num",
    )
    args.threshold = choose(
        args.threshold,
        get_nested(config, ("study_selection", "threshold")),
        "threshold/study_selection.threshold",
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
    args.reuse_existing_outputs = choose(
        args.reuse_existing_outputs,
        phase_settings.get("reuse_existing_outputs"),
        "reuse_existing_outputs/pipeline.phase3_study_selection.reuse_existing_outputs",
        required=False,
    )
    args.quickerdata_ls_path = choose(
        args.quickerdata_ls_path,
        str(Path(args.YOUR_DATASET_PATH) / f"quicker_data(PICO_IDX{args.pico_idx})_ls.json"),
        "quickerdata_ls_path",
    )
    args.record_included_output_path = choose(
        args.record_included_output_path,
        str(
            Path(args.YOUR_STUDY_SELECTION_PATH)
            / "record_included_studies"
            / f"record_included_PICO{args.pico_idx}.json"
        ),
        "record_included_output_path",
    )

    args.missing_pdf_json_path = choose(
        args.missing_pdf_json_path,
        str(
            Path(args.YOUR_STUDY_SELECTION_PATH)
            / "record_included_studies"
            / f"missing_pdfs_phase3_record_screening_PICO{args.pico_idx}.json"
        ),
        "missing_pdf_json_path",
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
    quicker.config["study_selection"]["record_screening_method"] = (
        args.record_screening_method
    )
    quicker.config["study_selection"]["exp_num"] = int(args.exp_num)
    quicker.config["study_selection"]["threshold"] = int(args.threshold)


def run(args: argparse.Namespace) -> None:
    config = load_config(args.YOUR_CONFIG_PATH)
    args = resolve_args(args, config)
    log_dir = prepare_environment(args, config)

    from utils.logger import get_detail_logger, get_workflow_logger, setup_loggers

    setup_loggers(
        log_file=os.path.join(
            log_dir,
            Path(args.YOUR_DATASET_PATH).name,
            "Study_Selection",
            f"{args.pico_idx}_record_screening.log",
        )
    )
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

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

    record_output_path = Path(args.record_included_output_path)
    if args.reuse_existing_outputs and record_output_path.exists():
        wf_logger.info("Reuse existing record-included output: %s", record_output_path)
        record_included_list = load_json_file(record_output_path)
    else:
        wf_logger.info("Run record screening for PICO %s", args.pico_idx)
        processed_search_results = quicker.preprocess_search_results()
        record_included_list = quicker.select_studies_by_record_screening(
            processed_search_results=processed_search_results
        )
        write_json_file(record_output_path, record_included_list)

    manifest = build_pdf_manifest(
        papers=record_included_list,
        paper_library_path=args.YOUR_PAPER_LIBRARY_PATH,
        pico_idx=args.pico_idx,
        stage="phase3_record_screening",
    )
    json_output, _ = write_pdf_manifest(
        manifest,
        json_path=args.missing_pdf_json_path,
        markdown_path=None,
    )

    dt_logger.info("Record-included papers: %s", record_included_list)
    print(f"Record-included studies saved to: {record_output_path}")
    print(f"PDF manifest saved to: {json_output}")
    print(f"Missing PDF count: {manifest['missing_pdf_count']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase3 record screening and PDF manifest generation."
    )
    add_common_config_args(parser)
    parser.add_argument("--YOUR_DATASET_PATH", default=None)
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", default=None)
    parser.add_argument("--YOUR_LITERATURE_SEARCH_PATH", default=None)
    parser.add_argument("--YOUR_STUDY_SELECTION_PATH", default=None)
    parser.add_argument("--YOUR_PAPER_LIBRARY_PATH", default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--pico_idx", default=None)
    parser.add_argument("--record_screening_method", default=None)
    parser.add_argument("--exp_num", type=int, default=None)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--study", default=None)
    parser.add_argument("--inclusion_criteria", default=None)
    parser.add_argument("--exclusion_criteria", default=None)
    parser.add_argument("--quickerdata_ls_path", default=None)
    parser.add_argument("--record_included_output_path", default=None)
    parser.add_argument("--missing_pdf_json_path", default=None)
    parser.add_argument(
        "--reuse_existing_outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
