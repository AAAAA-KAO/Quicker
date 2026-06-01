import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional


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
    provider = model_config.get("provider", "OpenAI")
    if provider != "OpenAI":
        raise NotImplementedError(f"Provider {provider} is not implemented")

    return ChatOpenAI(
        openai_api_key=model_config["API_KEY"],
        base_url=model_config["BASE_URL"],
        model=model_config["model_name"],
        temperature=model_config.get("temperature", 1.0),
    )


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


def run(args: argparse.Namespace) -> None:
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
    parser.add_argument(
        "--YOUR_CONFIG_PATH",
        type=str,
        default="config/config.json",
        help="Path to config.json.",
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
        "--YOUR_RECOMMENDATION_FORMATION_PATH",
        type=str,
        default="data/2021ACR RA/Recommendation_Formation",
        help="Recommendation formulation folder.",
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
        "--overall_certainty",
        type=str,
        default="LOW",
        help='Overall certainty, e.g. "VERY LOW", "LOW", "MODERATE", or "HIGH".',
    )
    parser.add_argument(
        "--supplementary_information",
        type=str,
        default="",
        help="Supplementary information for recommendation formulation.",
    )
    parser.add_argument(
        "--transfer_evidence_assessment_files",
        action="store_true",
        help="Copy matching outcomeinfo/paperinfo files before running.",
    )
    parser.add_argument(
        "--print_state",
        action="store_true",
        help="Print Quicker state before recommendation formulation.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
