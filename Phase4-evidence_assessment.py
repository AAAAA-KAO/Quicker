import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv

load_dotenv(".env")
if not os.getenv("LOG_DIR"):
    os.environ["LOG_DIR"] = "log"

from utils.logger import get_detail_logger, get_workflow_logger, setup_loggers


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_arg(value: Optional[str]) -> Any:
    if value is None:
        return None

    value = value.strip()
    if not value:
        return {}

    candidate_path = Path(value)
    if candidate_path.exists():
        return load_json(candidate_path)
    return json.loads(value)


def load_mapping_arg(value: Optional[str]) -> Optional[Dict[str, str]]:
    loaded = load_json_arg(value)

    if loaded is None:
        return None
    if not isinstance(loaded, dict):
        raise ValueError("Comparator postfix mapping must be a JSON object.")

    return {str(key): str(val) for key, val in loaded.items()}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_json(path)
    config["_config_path"] = str(path)
    return config


def transfer_outcome_and_paperinfo(
    source_dir_path: str,
    target_dir_path: str,
    only_index: Optional[str] = None,
) -> list[Path]:
    """
    Copy Phase3 outcomeinfo and paperinfo files into the evidence assessment
    workspace. If only_index is provided, only files containing '_PICO{index}'
    are copied.
    """
    copied = []
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
                copied.append(target_item)
        else:
            shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
            copied.append(target_folder)

    return copied


def comparator_postfix(comparator: str) -> str:
    return f"_c{hashlib.md5(comparator.encode('utf-8')).hexdigest()[:6]}"


def input_files_exist(
    evidence_assessment_path: str,
    pico_idx: str,
    comparator_postfix_value: Optional[str],
) -> bool:
    suffix = comparator_postfix_value or ""
    base_path = Path(evidence_assessment_path)
    outcome_path = base_path / "outcomeinfo" / f"outcomeinfo_PICO{pico_idx}{suffix}.json"
    paper_path = base_path / "paperinfo" / f"paperinfo_PICO{pico_idx}{suffix}.json"
    return outcome_path.exists() and paper_path.exists()


def extract_postfix(path: Path, prefix: str) -> str:
    if not path.stem.startswith(prefix):
        return ""
    return path.stem[len(prefix):]


def discover_input_postfixes(evidence_assessment_path: str, pico_idx: str) -> list[str]:
    base_path = Path(evidence_assessment_path)
    paper_folder = base_path / "paperinfo"
    outcome_folder = base_path / "outcomeinfo"
    if not paper_folder.exists() or not outcome_folder.exists():
        return []

    paper_prefix = f"paperinfo_PICO{pico_idx}"
    outcome_prefix = f"outcomeinfo_PICO{pico_idx}"
    postfixes = []

    for paper_path in sorted(paper_folder.glob(f"{paper_prefix}*.json")):
        if "_full_text_assessed_but_not_included" in paper_path.name:
            continue
        postfix = extract_postfix(paper_path, paper_prefix)
        outcome_path = outcome_folder / f"{outcome_prefix}{postfix}.json"
        if outcome_path.exists():
            postfixes.append(postfix)

    return postfixes


def resolve_input_postfix(
    evidence_assessment_path: str,
    pico_idx: str,
    comparator: str,
    comparisons: list[str],
    explicit_postfix: Optional[str],
    input_postfix_map: Optional[Dict[str, str]],
    derive_postfix: bool,
) -> Optional[str]:
    candidates = []

    if explicit_postfix is not None:
        candidates.append(explicit_postfix)
    if input_postfix_map and comparator in input_postfix_map:
        candidates.append(input_postfix_map[comparator])
    if derive_postfix:
        candidates.append(comparator_postfix(comparator))
    if len(comparisons) == 1:
        candidates.append("")
        candidates.extend(discover_input_postfixes(evidence_assessment_path, pico_idx))

    seen = set()
    for candidate in candidates:
        candidate = candidate or ""
        if candidate in seen:
            continue
        seen.add(candidate)
        if input_files_exist(evidence_assessment_path, pico_idx, candidate):
            return candidate

    return None


def iter_outcomeinfo_files(base_path: str, pico_idx: str) -> Iterable[Path]:
    folder = Path(base_path) / "outcomeinfo"
    if not folder.exists():
        return []
    return sorted(folder.glob(f"outcomeinfo_PICO{pico_idx}*.json"))


def outcomeinfo_score(path: Path) -> int:
    try:
        outcomes = load_json(path)
    except Exception:
        return 0
    if not isinstance(outcomes, list):
        return 0

    score = 0
    for outcome in outcomes:
        grade = outcome.get("assessment_results", {}).get("GRADE", {})
        if not isinstance(grade, dict):
            continue
        score += len(grade)
        if any(
            key in grade
            for key in (
                "Risk of bias",
                "No of participants",
                "Effect",
                "Certainty",
                "result_interpretation",
            )
        ):
            score += 100
    return score


def assessed_outputs_exist(evidence_assessment_path: str, pico_idx: str) -> list[Path]:
    return [
        path
        for path in iter_outcomeinfo_files(evidence_assessment_path, pico_idx)
        if outcomeinfo_score(path) >= 100
    ]


def load_pico(question_decomposition_path: str, pico_idx: str) -> dict:
    pico_path = Path(question_decomposition_path) / "PICO_Information.json"
    if not pico_path.exists():
        raise FileNotFoundError(f"PICO information file not found: {pico_path}")

    pico_list = load_json(pico_path)
    if not isinstance(pico_list, list):
        raise ValueError(f"{pico_path} must contain a JSON list.")

    for pico in pico_list:
        if str(pico.get("Index")) == pico_idx:
            return pico

    raise ValueError(f"PICO index {pico_idx} not found in {pico_path}")


def import_quicker_dependencies():
    try:
        from utils.General.quicker import Quicker, QuickerData, QuickerStage
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

    return Quicker, QuickerData, QuickerStage


def build_annotation(args: argparse.Namespace, config: dict) -> dict:
    phase_config = config.get("pipeline", {}).get("phase4_evidence_assessment", {})
    if args.annotation_json is not None:
        loaded = load_json_arg(args.annotation_json)
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("Annotation must be a JSON object.")
        return loaded or {}
    return phase_config.get("annotation", {})


def build_input_postfix_map(args: argparse.Namespace, config: dict) -> Dict[str, str]:
    phase_config = config.get("pipeline", {}).get("phase4_evidence_assessment", {})
    loaded = load_mapping_arg(args.input_comparator_postfix_map_json)
    if loaded is not None:
        return loaded
    return phase_config.get("input_comparator_postfix_map", {})


def build_output_postfix_map(
    args: argparse.Namespace,
    config: dict,
) -> Optional[Dict[str, str]]:
    phase_config = config.get("pipeline", {}).get("phase4_evidence_assessment", {})
    loaded = load_mapping_arg(args.output_comparator_postfix_map_json)
    if loaded is not None:
        return loaded
    return phase_config.get("output_comparator_postfix_map")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.YOUR_CONFIG_PATH)
    log_dir = os.getenv("LOG_DIR", "log")
    setup_loggers(
        log_file=os.path.join(
            log_dir,
            args.YOUR_DATASET_PATH.split("/")[-1],
            "Evidence_Assessment",
            f"{args.pico_idx}.log",
        )
    )
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    copied_inputs = []
    if args.transfer_study_selection_files:
        copied_inputs = transfer_outcome_and_paperinfo(
            source_dir_path=args.YOUR_STUDY_SELECTION_PATH,
            target_dir_path=args.YOUR_EVIDENCE_ASSESSMENT_PATH,
            only_index=args.pico_idx,
        )
        dt_logger.info("Copied Phase3 inputs: %s", [str(path) for path in copied_inputs])

    if args.reuse_existing_outputs:
        existing_outputs = assessed_outputs_exist(
            args.YOUR_EVIDENCE_ASSESSMENT_PATH,
            args.pico_idx,
        )
        if existing_outputs:
            wf_logger.info(
                "Evidence assessment skipped because assessed outputs already exist."
            )
            for path in existing_outputs:
                print(f"Existing assessed output: {path}")
            return

    Quicker, QuickerData, QuickerStage = import_quicker_dependencies()
    pico = load_pico(args.YOUR_QUESTION_DECOMPOSITION_PATH, args.pico_idx)
    comparisons = [str(item) for item in pico.get("C", [])]
    if args.comparator:
        comparisons = [args.comparator]
    if not comparisons:
        raise ValueError(f"No comparator found for PICO {args.pico_idx}.")

    quicker_data = QuickerData(disease=args.disease, pico_idx=args.pico_idx)
    quicker = Quicker(
        config_path=args.YOUR_CONFIG_PATH,
        question_deconstruction_database_path=args.YOUR_QUESTION_DECOMPOSITION_PATH,
        literature_search_database_path=args.YOUR_LITERATURE_SEARCH_PATH,
        study_selection_database_path=args.YOUR_STUDY_SELECTION_PATH,
        evidence_assessment_database_path=args.YOUR_EVIDENCE_ASSESSMENT_PATH,
        quicker_data=quicker_data,
        paper_library_base=args.YOUR_PAPER_LIBRARY_PATH,
    )

    output_postfix_map = build_output_postfix_map(args, config)
    if output_postfix_map is not None:
        quicker.comparator_postfix_map = output_postfix_map

    data_dict = {
        "pico_idx": args.pico_idx,
        "clinical_question": pico["Question"],
        "population": pico["P"],
        "intervention": pico["I"],
        "comparison": comparisons,
        "valid_comparison_list": comparisons,
        "outcome": pico.get("O", {}),
        "annotation": build_annotation(args, config),
    }
    quicker._add_data_to_quickerdata_for_test(
        stage=QuickerStage.STUDY_SELECTION,
        default_value=data_dict,
    )

    input_postfix_map = build_input_postfix_map(args, config)
    assessed_comparators = []

    wf_logger.info(
        "Run evidence assessment for PICO %s with %s comparator(s).",
        args.pico_idx,
        len(comparisons),
    )
    for comparator in comparisons:
        input_postfix = resolve_input_postfix(
            evidence_assessment_path=args.YOUR_EVIDENCE_ASSESSMENT_PATH,
            pico_idx=args.pico_idx,
            comparator=comparator,
            comparisons=comparisons,
            explicit_postfix=args.input_comparator_postfix,
            input_postfix_map=input_postfix_map,
            derive_postfix=args.derive_comparator_postfix,
        )

        if input_postfix is None:
            message = (
                "No matching paperinfo/outcomeinfo input files found for "
                f"comparator: {comparator}"
            )
            if args.skip_comparators_without_inputs:
                wf_logger.info("%s; skipped.", message)
                continue
            raise FileNotFoundError(message)

        wf_logger.info(
            "Assess comparator %s using input postfix %s.",
            comparator,
            input_postfix or "<none>",
        )
        quicker.load_outcome_list(comparator_postfix=input_postfix)
        quicker.load_paper_list(comparator_postfix=input_postfix)

        if args.print_state:
            print(quicker.quicker_data.check_stage_state())
            print(quicker.quicker_data.not_none_data)

        quicker.assess_evidence(comparator=comparator)
        assessed_comparators.append(comparator)

    if not assessed_comparators:
        raise RuntimeError("No comparator was assessed.")

    outputs = list(iter_outcomeinfo_files(args.YOUR_EVIDENCE_ASSESSMENT_PATH, args.pico_idx))
    wf_logger.info("Evidence assessment completed for PICO %s.", args.pico_idx)
    print(f"Assessed comparators: {assessed_comparators}")
    for path in outputs:
        print(f"Output outcomeinfo: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase4 evidence assessment.")
    parser.add_argument(
        "--YOUR_CONFIG_PATH",
        type=str,
        default="config/config.json",
        help="Path to config.json.",
    )
    parser.add_argument(
        "--YOUR_DATASET_PATH",
        type=str,
        default="data/2021ACR RA",
        help="Dataset root path.",
    )
    parser.add_argument(
        "--YOUR_QUESTION_DECOMPOSITION_PATH",
        type=str,
        default="data/2021ACR RA/Question_Decomposition",
        help="Question decomposition folder.",
    )
    parser.add_argument(
        "--YOUR_LITERATURE_SEARCH_PATH",
        type=str,
        default="data/2021ACR RA/Literature_Search",
        help="Literature search folder.",
    )
    parser.add_argument(
        "--YOUR_STUDY_SELECTION_PATH",
        type=str,
        default="data/2021ACR RA/Study_Selection",
        help="Study selection folder.",
    )
    parser.add_argument(
        "--YOUR_EVIDENCE_ASSESSMENT_PATH",
        type=str,
        default="data/2021ACR RA/Evidence_Assessment",
        help="Evidence assessment folder.",
    )
    parser.add_argument(
        "--YOUR_PAPER_LIBRARY_PATH",
        type=str,
        default="data/2021ACR RA/Paper_Library",
        help="Paper library folder.",
    )
    parser.add_argument(
        "--disease",
        type=str,
        default="Rheumatoid Arthritis (RA)",
        help="Disease name or clinical topic.",
    )
    parser.add_argument(
        "--pico_idx",
        type=str,
        default="dff23ac6",
        help="PICO index from PICO_Information.json.",
    )
    parser.add_argument(
        "--comparator",
        type=str,
        default=None,
        help="Run only this comparator. Defaults to all comparators in PICO.",
    )
    parser.add_argument(
        "--input_comparator_postfix",
        type=str,
        default=None,
        help="Explicit input postfix, e.g. _c649f30.",
    )
    parser.add_argument(
        "--input_comparator_postfix_map_json",
        type=str,
        default=None,
        help="JSON object or JSON file path mapping comparator text to input postfix.",
    )
    parser.add_argument(
        "--output_comparator_postfix_map_json",
        type=str,
        default=None,
        help=(
            "JSON object or JSON file path mapping comparator text to output postfix. "
            "Use '{}' to save without comparator postfix."
        ),
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        default=None,
        help="JSON object or JSON file path for evidence-assessment annotations.",
    )
    parser.add_argument(
        "--transfer_study_selection_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy Phase3 paperinfo/outcomeinfo files before running.",
    )
    parser.add_argument(
        "--derive_comparator_postfix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive default comparator postfix from comparator MD5 hash.",
    )
    parser.add_argument(
        "--skip_comparators_without_inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip comparators when matching input files are missing.",
    )
    parser.add_argument(
        "--reuse_existing_outputs",
        action="store_true",
        help="Skip assessment if assessed evidence outputs already exist.",
    )
    parser.add_argument(
        "--print_state",
        action="store_true",
        help="Print Quicker state before each comparator assessment.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
