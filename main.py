import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class QuestionDecompositionOutput(BaseModel):
    P: list[str] = Field(description="The population of the question")
    I: list[str] = Field(description="The intervention of the question")
    C: list[str] = Field(description="The comparison of the question")
    O: dict[str, list[str]] = Field(description="The outcome of the question")


class MissingPDFsError(RuntimeError):
    def __init__(
        self,
        message: str,
        missing_pdfs: list[dict],
        json_path: Path,
        markdown_path: Path,
    ):
        super().__init__(message)
        self.missing_pdfs = missing_pdfs
        self.json_path = json_path
        self.markdown_path = markdown_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def pipeline_paths(config: dict, project_root: Path) -> dict[str, Path]:
    return {
        key: resolve_path(project_root, value)
        for key, value in config["pipeline"]["paths"].items()
    }


def setup_pipeline_logging(config: dict, project_root: Path) -> logging.Logger:
    log_dir = resolve_path(project_root, config["logging"]["log_dir"])
    os.environ["LOG_DIR"] = str(log_dir)

    from utils.logger import get_workflow_logger, setup_loggers

    dataset_name = config["pipeline"]["dataset_name"]
    log_file = log_dir / dataset_name / "main" / config["logging"]["log_file_name"]
    setup_loggers(log_file=str(log_file))
    return get_workflow_logger(__name__)


def build_chat_model(config: dict, model_key: str) -> ChatOpenAI:
    model_config = config["model"][model_key]
    provider = model_config.get("provider", "OpenAI")
    if provider != "OpenAI":
        raise NotImplementedError(f"Provider {provider} is not implemented.")

    model_kwargs = {
        "openai_api_key": model_config["API_KEY"],
        "base_url": model_config["BASE_URL"],
        "model": model_config["model_name"],
        "temperature": model_config.get("temperature", 1.0),
    }
    if model_config.get("timeout") is not None:
        model_kwargs["timeout"] = model_config["timeout"]
    return ChatOpenAI(**model_kwargs)


def get_pico_idx(config: dict) -> str:
    configured = config["pipeline"].get("pico_idx")
    if configured and configured != "auto":
        return configured

    clinical_question = config["pipeline"]["clinical_question"]
    dataset_name = config["pipeline"]["dataset_name"]
    return hashlib.sha256(
        (clinical_question + dataset_name).encode("utf-8")
    ).hexdigest()[:8]


def normalize_text_component(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def normalize_list_component(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def get_paper_value(paper: Any, key: str, default: Any = None) -> Any:
    if isinstance(paper, dict):
        return paper.get(key, default)
    return getattr(paper, key, default)


def paper_library_pico_path(paths: dict[str, Path], pico_idx: str) -> Path:
    return paths["paper_library"] / f"PICO{pico_idx}"


def expected_pdf_path(paths: dict[str, Path], pico_idx: str, paper_uid: str) -> Path:
    return paper_library_pico_path(paths, pico_idx) / paper_uid / f"{paper_uid}.pdf"


def folder_has_pdf(folder: Path) -> bool:
    return folder.exists() and any(folder.glob("*.pdf"))


def declared_pdf_folder(config: dict, save_folder_path: str | None) -> Path | None:
    if not save_folder_path:
        return None
    return resolve_path(config["_project_root"], save_folder_path)


def find_missing_pdf_items(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    papers: list[Any],
) -> list[dict]:
    missing = []
    seen = set()
    for paper in papers:
        paper_uid = get_paper_value(paper, "paper_uid")
        if not paper_uid or paper_uid in seen:
            continue
        seen.add(paper_uid)

        expected_path = expected_pdf_path(paths, pico_idx, str(paper_uid))
        declared_folder = declared_pdf_folder(
            config,
            get_paper_value(paper, "save_folder_path"),
        )

        if folder_has_pdf(expected_path.parent):
            continue
        if declared_folder is not None and folder_has_pdf(declared_folder):
            continue

        missing.append(
            {
                "paper_uid": str(paper_uid),
                "title": get_paper_value(paper, "title", ""),
                "pmid": get_paper_value(paper, "pmid", ""),
                "doi": get_paper_value(paper, "doi", ""),
                "expected_pdf_path": str(expected_path),
                "expected_folder": str(expected_path.parent),
            }
        )
    return missing


def write_missing_pdf_request(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    stage_name: str,
    missing_pdfs: list[dict],
) -> tuple[Path, Path]:
    pdf_config = config["pipeline"]["pdf_handling"]
    json_path = paths["reports"] / pdf_config["missing_pdf_json"].format(
        pico_idx=pico_idx,
        stage=stage_name,
    )
    markdown_path = paths["reports"] / pdf_config["missing_pdf_markdown"].format(
        pico_idx=pico_idx,
        stage=stage_name,
    )
    save_json(
        json_path,
        {
            "stage": stage_name,
            "pico_idx": pico_idx,
            "missing_pdf_count": len(missing_pdfs),
            "missing_pdfs": missing_pdfs,
        },
    )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Missing PDF Request: {stage_name}",
        "",
        f"- PICO index: `{pico_idx}`",
        f"- Missing PDF count: `{len(missing_pdfs)}`",
        "",
        "Please download each PDF and place it at the exact path shown below.",
        "",
    ]
    for index, item in enumerate(missing_pdfs, start=1):
        lines.extend(
            [
                f"## {index}. {item['paper_uid']}",
                "",
                f"- Title: {item['title']}",
                f"- PMID: {item['pmid']}",
                f"- DOI: {item['doi']}",
                f"- Folder: `{item['expected_folder']}`",
                f"- Preferred file path: `{item['expected_pdf_path']}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def ensure_pdf_files_available(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    stage_name: str,
    papers: list[Any],
) -> None:
    pdf_config = config["pipeline"].get("pdf_handling", {})
    if not pdf_config.get("stop_when_missing_pdf", True):
        return

    missing_pdfs = find_missing_pdf_items(config, paths, pico_idx, papers)
    if not missing_pdfs:
        return

    json_path, markdown_path = write_missing_pdf_request(
        config=config,
        paths=paths,
        pico_idx=pico_idx,
        stage_name=stage_name,
        missing_pdfs=missing_pdfs,
    )
    raise MissingPDFsError(
        message=(
            f"{len(missing_pdfs)} PDFs are missing for {stage_name}. "
            f"Download request written to {markdown_path}."
        ),
        missing_pdfs=missing_pdfs,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def pico_file_path(paths: dict[str, Path]) -> Path:
    return paths["question_decomposition"] / "PICO_Information.json"


def load_pico_records(paths: dict[str, Path]) -> list[dict]:
    path = pico_file_path(paths)
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} should contain a JSON list.")
    return data


def find_pico(paths: dict[str, Path], pico_idx: str) -> dict | None:
    for pico in load_pico_records(paths):
        if str(pico.get("Index")) == pico_idx:
            return pico
    return None


def upsert_pico(paths: dict[str, Path], pico: dict) -> Path:
    records = load_pico_records(paths)
    replaced = False
    for index, existing in enumerate(records):
        if str(existing.get("Index")) == str(pico["Index"]):
            records[index] = pico
            replaced = True
            break
    if not replaced:
        records.append(pico)
    path = pico_file_path(paths)
    save_json(path, records)
    return path


def run_phase1_question_decomposition(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase1_question_decomposition"]
    existing_pico = find_pico(paths, pico_idx)
    if phase_config["reuse_existing_pico"] and existing_pico:
        logger.info("Phase1 skipped because PICO %s already exists.", pico_idx)
        return {
            "status": "skipped",
            "reason": "existing PICO reused",
            "outputs": [str(pico_file_path(paths))],
        }

    from utils.PICO.prompt import get_zero_shot_pipeline_prompt

    clinical_question = config["pipeline"]["clinical_question"]
    dataset_name = config["pipeline"]["dataset_name"]
    logger.info("Phase1 started for PICO %s.", pico_idx)

    model = build_chat_model(config, "question_decomposition_model")
    structured_model = model.with_structured_output(QuestionDecompositionOutput)
    prompt = get_zero_shot_pipeline_prompt(dataset_name)
    chain = prompt | RunnableParallel(
        generation_chain=structured_model,
        prompt_value=RunnablePassthrough(),
    )

    result = chain.invoke({"Question": clinical_question})
    generation = result["generation_chain"]
    if isinstance(generation, dict):
        population = generation["P"]
        intervention = generation["I"]
        comparison = generation["C"]
        outcome = generation["O"]
    else:
        population = generation.P
        intervention = generation.I
        comparison = generation.C
        outcome = generation.O

    pico = {
        "Index": pico_idx,
        "Question": clinical_question,
        "P": normalize_text_component(population),
        "I": normalize_text_component(intervention),
        "C": normalize_list_component(comparison),
        "O": outcome,
    }
    output_path = upsert_pico(paths, pico)
    logger.info("Phase1 completed and saved %s.", output_path)
    return {"status": "completed", "outputs": [str(output_path)]}


def quicker_data_literature_search_path(paths: dict[str, Path], pico_idx: str) -> Path:
    return paths["dataset"] / f"quicker_data(PICO_IDX{pico_idx})_ls.json"


def filter_search_results(
    search_results: list[dict],
    invalid_publication_types: list[str],
) -> list[dict]:
    with_abstract = [
        record for record in search_results if record.get("Abstract") is not None
    ]

    deduplicated = {}
    for record in with_abstract:
        paper_index = record.get("Paper_Index")
        if paper_index is None:
            continue
        deduplicated.setdefault(str(paper_index), record)

    invalid_types = set(invalid_publication_types)
    return [
        record
        for record in deduplicated.values()
        if not any(
            publication_type in invalid_types
            for publication_type in record.get("Publication Types", [])
        )
    ]


def run_phase2_literature_search(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase2_literature_search"]
    output_path = quicker_data_literature_search_path(paths, pico_idx)
    if phase_config["reuse_existing_quicker_data"] and output_path.exists():
        logger.info("Phase2 skipped because %s already exists.", output_path)
        return {"status": "skipped", "reason": "existing search data reused", "outputs": [str(output_path)]}

    from utils.Evidence_Retrieval.pubmedretrieval import PubMedRetrieval

    pico = find_pico(paths, pico_idx)
    if not pico:
        raise FileNotFoundError(f"PICO {pico_idx} was not found in {pico_file_path(paths)}.")

    model_config = config["model"]["literature_search_model"]
    save_path = (
        paths["literature_search"]
        / phase_config["search_backend"]
        / "Results"
        / model_config["model_name"]
        / f"use_agent_{phase_config['use_agent']}"
    )
    model_setting = {
        "search_term_formation": model_config["model_name"],
        "search_strategy_formation": model_config["model_name"],
    }

    logger.info("Phase2 started for PICO %s.", pico_idx)
    retriever = PubMedRetrieval(
        disease=config["pipeline"]["disease"],
        clinical_question=pico["Question"],
        population=pico["P"],
        intervention=pico["I"],
        comparison=pico["C"],
        outcome=pico.get("O"),
        api_key=model_config["API_KEY"],
        base_url=model_config["BASE_URL"],
        model_setting=model_setting,
        use_agent=phase_config["use_agent"],
        round_limit=phase_config["round_limit"],
        save_path=str(save_path),
        pico_idx=pico_idx,
        search_terms=phase_config.get("search_terms"),
        filters=phase_config["filters"],
        additional_parameters=phase_config["additional_parameters"],
    )
    retriever.run()

    raw_results_path = save_path / f"PICO{pico_idx}.json"
    if not raw_results_path.exists():
        raise FileNotFoundError(f"PubMed retrieval did not create {raw_results_path}.")

    raw_results = load_json(raw_results_path)
    filtered_results = filter_search_results(
        raw_results,
        invalid_publication_types=phase_config["invalid_publication_types"],
    )

    quicker_data = {
        "disease": config["pipeline"]["disease"],
        "clinical_question": pico["Question"],
        "pico_idx": pico_idx,
        "population": pico["P"],
        "intervention": pico["I"],
        "comparison": pico["C"],
        "outcome": pico.get("O", {}),
        "search_results": filtered_results,
    }
    save_json(output_path, quicker_data)
    logger.info("Phase2 completed and saved %s.", output_path)
    return {
        "status": "completed",
        "outputs": [str(raw_results_path), str(output_path)],
        "record_count": len(filtered_results),
    }


def has_study_selection_outputs(paths: dict[str, Path], pico_idx: str) -> list[Path]:
    candidates = []
    for folder_name, prefix in (
        ("outcomeinfo", "outcomeinfo"),
        ("paperinfo", "paperinfo"),
    ):
        folder = paths["study_selection"] / folder_name
        if folder.exists():
            candidates.extend(sorted(folder.glob(f"{prefix}_PICO{pico_idx}*.json")))
    return candidates


def run_phase3_study_selection(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase3_study_selection"]
    existing_outputs = has_study_selection_outputs(paths, pico_idx)
    if phase_config["reuse_existing_outputs"] and existing_outputs:
        logger.info("Phase3 skipped because study-selection outputs already exist.")
        return {
            "status": "skipped",
            "reason": "existing study-selection outputs reused",
            "outputs": [str(path) for path in existing_outputs],
        }

    from utils.General.quicker import Quicker, QuickerData, QuickerStage

    quickerdata_path = quicker_data_literature_search_path(paths, pico_idx)
    if not quickerdata_path.exists():
        raise FileNotFoundError(f"Phase2 output not found: {quickerdata_path}")
    quickerdata_ls = load_json(quickerdata_path)

    logger.info("Phase3 started for PICO %s.", pico_idx)
    quicker_data = QuickerData(disease=config["pipeline"]["disease"], pico_idx=pico_idx)
    quicker = Quicker(
        config_path=str(config["_config_path"]),
        question_deconstruction_database_path=str(paths["question_decomposition"]),
        literature_search_database_path=str(paths["literature_search"]),
        study_selection_database_path=str(paths["study_selection"]),
        evidence_assessment_database_path=str(paths["evidence_assessment"]),
        quicker_data=quicker_data,
        paper_library_base=str(paths["paper_library"]),
    )
    data_dict = {
        "clinical_question": quickerdata_ls["clinical_question"],
        "population": quickerdata_ls["population"],
        "intervention": quickerdata_ls["intervention"],
        "comparison": quickerdata_ls["comparison"],
        "outcome": quickerdata_ls.get("outcome", {}),
        "study": phase_config["study"],
        "search_results": quickerdata_ls["search_results"],
    }
    quicker._add_data_to_quickerdata_for_test(
        stage=QuickerStage.LITERATURE_SEARCH,
        default_value=data_dict,
    )
    quicker.set_inclusion_exclusion_criteria(
        inclusion_criteria=phase_config["inclusion_criteria"],
        exclusion_criteria=phase_config["exclusion_criteria"],
    )

    if getattr(quicker.quicker_data, "record_included_studies"):
        logger.info("Phase3 record screening skipped because records are loaded.")
        record_included_list = getattr(quicker.quicker_data, "record_included_studies")
    else:
        logger.info("Phase3 record screening started.")
        processed_search_results = quicker.preprocess_search_results()
        record_included_list = quicker.select_studies_by_record_screening(
            processed_search_results=processed_search_results
        )

    if config["study_selection"]["full_text_assessment_method"] is None:
        logger.info("Phase3 full-text assessment skipped by config.")
        quicker.quicker_data.update_data(
            {
                "record_included_studies": record_included_list,
                "full_text_included_studies": [],
                "total_outcome_list": [],
            }
        )
    else:
        ensure_pdf_files_available(
            config=config,
            paths=paths,
            pico_idx=pico_idx,
            stage_name="phase3_study_selection",
            papers=record_included_list,
        )
        (
            record_included_list,
            full_text_included_list,
            total_outcome_list,
        ) = quicker.select_studies_by_full_text_assessment(record_included_list)
        quicker.quicker_data.update_data(
            {
                "record_included_studies": record_included_list,
                "full_text_included_studies": full_text_included_list,
                "total_outcome_list": total_outcome_list,
            }
        )

    method = config["study_selection"]["record_screening_method"]
    method_label = method if isinstance(method, str) else "precomputed"
    results_save_path = (
        paths["study_selection"]
        / "Results"
        / "screening_records"
        / method_label
        / pico_idx
    )
    quicker.quicker_data.to_json(str(results_save_path))
    outputs = has_study_selection_outputs(paths, pico_idx)
    outputs.append(results_save_path)
    logger.info("Phase3 completed for PICO %s.", pico_idx)
    return {"status": "completed", "outputs": [str(path) for path in outputs]}


def transfer_outcome_and_paperinfo(
    source_dir_path: Path,
    target_dir_path: Path,
    only_index: str | None,
) -> list[Path]:
    copied = []
    for folder_name in ("outcomeinfo", "paperinfo"):
        source_folder = source_dir_path / folder_name
        target_folder = target_dir_path / folder_name
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


def matching_outcomeinfo_files(
    base_path: Path,
    pico_idx: str,
    require_assessed: bool,
) -> list[Path]:
    candidates = []
    for folder in (base_path, base_path / "outcomeinfo"):
        if not folder.exists():
            continue
        candidates.extend(sorted(folder.glob(f"outcomeinfo_PICO{pico_idx}*.json")))

    if require_assessed:
        candidates = [path for path in candidates if outcomeinfo_score(path) >= 100]
    return candidates


def run_phase4_evidence_assessment(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase4_evidence_assessment"]
    copied = []
    if phase_config["transfer_study_selection_files"]:
        copied = transfer_outcome_and_paperinfo(
            source_dir_path=paths["study_selection"],
            target_dir_path=paths["evidence_assessment"],
            only_index=pico_idx,
        )

    existing_outputs = matching_outcomeinfo_files(
        paths["evidence_assessment"],
        pico_idx,
        require_assessed=phase_config["require_assessed_outputs_for_skip"],
    )
    if phase_config["reuse_existing_outputs"] and existing_outputs:
        logger.info("Phase4 skipped because assessed evidence outputs already exist.")
        return {
            "status": "skipped",
            "reason": "existing evidence-assessment outputs reused",
            "outputs": [str(path) for path in existing_outputs],
            "copied_inputs": [str(path) for path in copied],
        }

    from utils.General.quicker import Quicker, QuickerData, QuickerStage

    pico = find_pico(paths, pico_idx)
    if not pico:
        raise FileNotFoundError(f"PICO {pico_idx} was not found in {pico_file_path(paths)}.")

    logger.info("Phase4 evidence assessment started for PICO %s.", pico_idx)
    quicker_data = QuickerData(disease=config["pipeline"]["disease"], pico_idx=pico_idx)
    quicker = Quicker(
        config_path=str(config["_config_path"]),
        question_deconstruction_database_path=str(paths["question_decomposition"]),
        literature_search_database_path=str(paths["literature_search"]),
        study_selection_database_path=str(paths["study_selection"]),
        evidence_assessment_database_path=str(paths["evidence_assessment"]),
        quicker_data=quicker_data,
        paper_library_base=str(paths["paper_library"]),
    )
    output_postfix_map = phase_config.get("output_comparator_postfix_map")
    if output_postfix_map is not None:
        quicker.comparator_postfix_map = output_postfix_map

    comparison_list = pico["C"]
    data_dict = {
        "pico_idx": pico_idx,
        "clinical_question": pico["Question"],
        "population": pico["P"],
        "intervention": pico["I"],
        "comparison": comparison_list,
        "valid_comparison_list": comparison_list,
        "outcome": pico.get("O", {}),
        "annotation": phase_config["annotation"],
    }
    quicker._add_data_to_quickerdata_for_test(
        stage=QuickerStage.STUDY_SELECTION,
        default_value=data_dict,
    )

    assessed = []
    input_postfix_map = phase_config["input_comparator_postfix_map"]
    for comparator in comparison_list:
        comparator_postfix = input_postfix_map.get(comparator)
        quicker.load_outcome_list(comparator_postfix=comparator_postfix)
        quicker.load_paper_list(comparator_postfix=comparator_postfix)
        ensure_pdf_files_available(
            config=config,
            paths=paths,
            pico_idx=pico_idx,
            stage_name="phase4_evidence_assessment",
            papers=quicker.quicker_data.paper_list,
        )
        assessed.append(quicker.assess_evidence(comparator=comparator))

    quicker.quicker_data.update_data({"evidence_assessment_results": assessed})
    outputs = matching_outcomeinfo_files(
        paths["evidence_assessment"],
        pico_idx,
        require_assessed=False,
    )
    logger.info("Phase4 evidence assessment completed for PICO %s.", pico_idx)
    return {
        "status": "completed",
        "outputs": [str(path) for path in outputs],
        "copied_inputs": [str(path) for path in copied],
    }


def run_phase4_meta_analysis(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase4_meta_analysis"]
    if not phase_config["enabled"]:
        logger.info("Phase4 meta-analysis skipped by config.")
        return {"status": "skipped", "reason": "disabled by config", "outputs": []}

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except ModuleNotFoundError as exc:
        if phase_config["skip_when_dependency_missing"]:
            logger.warning("Phase4 meta-analysis skipped because rpy2 is missing.")
            return {
                "status": "skipped",
                "reason": f"missing dependency: {exc.name}",
                "outputs": [],
            }
        raise

    output_dir = paths["reports"] / "meta_analysis" / pico_idx
    output_dir.mkdir(parents=True, exist_ok=True)
    pandas2ri.activate()
    ro.r("library(meta)")
    outputs = []

    for index, dataset in enumerate(phase_config["binary_outcomes"]):
        df = pd.DataFrame(dataset["data"])
        ro.globalenv["meta_data"] = pandas2ri.py2rpy(df)
        ro.r(dataset["r_expression"])
        summary = "\n".join(ro.r("capture.output(summary(res))"))
        output_path = output_dir / f"binary_meta_{index + 1}.txt"
        output_path.write_text(summary, encoding="utf-8")
        outputs.append(output_path)

    for index, dataset in enumerate(phase_config["continuous_outcomes"]):
        df = pd.DataFrame(dataset["data"])
        ro.globalenv["meta_data"] = pandas2ri.py2rpy(df)
        ro.r(dataset["r_expression"])
        summary = "\n".join(ro.r("capture.output(summary(res))"))
        output_path = output_dir / f"continuous_meta_{index + 1}.txt"
        output_path.write_text(summary, encoding="utf-8")
        outputs.append(output_path)

    logger.info("Phase4 meta-analysis completed for PICO %s.", pico_idx)
    return {"status": "completed", "outputs": [str(path) for path in outputs]}


def iter_outcomeinfo_files(base_path: Path, pico_idx: str) -> Iterable[Path]:
    seen = set()
    for folder in (base_path, base_path / "outcomeinfo"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob(f"outcomeinfo_PICO{pico_idx}*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def load_evidence_assessment_results(
    recommendation_path: Path,
    pico_idx: str,
    overall_certainty: str,
    require_assessed_evidence: bool,
) -> dict[str, dict]:
    selected: dict[str, tuple[int, Path, dict]] = {}
    for path in iter_outcomeinfo_files(recommendation_path, pico_idx):
        score = outcomeinfo_score(path)
        if require_assessed_evidence and score < 100:
            continue

        outcomes = load_json(path)
        if not isinstance(outcomes, list) or not outcomes:
            continue
        comparator = outcomes[0].get("comparator")
        if not comparator:
            continue

        payload = {
            "outcome_list": outcomes,
            "overall_certainty": overall_certainty,
        }
        if comparator not in selected or score > selected[comparator][0]:
            selected[comparator] = (score, path, payload)

    if not selected:
        raise FileNotFoundError(
            f"No usable outcomeinfo file found for PICO {pico_idx} under {recommendation_path}."
        )

    return {
        comparator: payload
        for comparator, (_, _, payload) in selected.items()
    }


def load_clinical_question(paths: dict[str, Path], pico_idx: str) -> str:
    pico = find_pico(paths, pico_idx)
    if not pico:
        raise FileNotFoundError(f"PICO {pico_idx} was not found in {pico_file_path(paths)}.")
    return pico["Question"]


def run_phase5_recommendation_formulation(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase5_recommendation_formulation"]
    copied = []
    if phase_config["transfer_evidence_assessment_files"]:
        copied = transfer_outcome_and_paperinfo(
            source_dir_path=paths["evidence_assessment"],
            target_dir_path=paths["recommendation_formation"],
            only_index=pico_idx,
        )

    existing_results = sorted(
        paths["recommendation_formation"].glob(f"quicker_data(PICO_IDX{pico_idx})_*.json")
    )
    if phase_config["reuse_existing_result"] and existing_results:
        latest = existing_results[-1]
        logger.info("Phase5 skipped because %s already exists.", latest)
        return {
            "status": "skipped",
            "reason": "existing recommendation result reused",
            "outputs": [str(latest)],
            "copied_inputs": [str(path) for path in copied],
        }

    from utils.Recommendation_formation.recommendation import Recommendation

    logger.info("Phase5 started for PICO %s.", pico_idx)
    evidence_assessment_results = load_evidence_assessment_results(
        recommendation_path=paths["recommendation_formation"],
        pico_idx=pico_idx,
        overall_certainty=phase_config["overall_certainty"],
        require_assessed_evidence=phase_config["require_assessed_evidence"],
    )
    clinical_question = load_clinical_question(paths, pico_idx)
    recommendation = Recommendation(
        disease=config["pipeline"]["disease"],
        evidence_assessment_result=evidence_assessment_results,
        model=build_chat_model(config, "recommendation_formation_model"),
        supplementary_information=phase_config["supplementary_information"],
    )
    final_result = recommendation.get_recommendation()

    output_data = {
        "disease": config["pipeline"]["disease"],
        "clinical_question": clinical_question,
        "pico_idx": pico_idx,
        "evidence_assessment_results": evidence_assessment_results,
        "supplementary_information": phase_config["supplementary_information"],
        "final_result": final_result,
    }
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = (
        paths["recommendation_formation"]
        / f"quicker_data(PICO_IDX{pico_idx})_{timestamp}.json"
    )
    save_json(output_path, output_data)
    logger.info("Phase5 completed and saved %s.", output_path)
    return {
        "status": "completed",
        "outputs": [str(output_path)],
        "copied_inputs": [str(path) for path in copied],
    }


def save_run_manifest(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    results: dict[str, dict],
) -> Path:
    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": config["pipeline"]["dataset_name"],
        "disease": config["pipeline"]["disease"],
        "clinical_question": config["pipeline"]["clinical_question"],
        "pico_idx": pico_idx,
        "results": results,
    }
    output_path = paths["reports"] / f"run_manifest_{pico_idx}.json"
    save_json(output_path, manifest)
    return output_path


def run_pipeline(config: dict, project_root: Path, logger: logging.Logger) -> dict:
    paths = pipeline_paths(config, project_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    pico_idx = get_pico_idx(config)
    stages = config["pipeline"]["stages"]
    results = {}

    if stages["phase1_question_decomposition"]:
        results["phase1_question_decomposition"] = run_phase1_question_decomposition(
            config, paths, pico_idx, logger
        )
    if stages["phase2_literature_search"]:
        results["phase2_literature_search"] = run_phase2_literature_search(
            config, paths, pico_idx, logger
        )
    if stages["phase3_study_selection"]:
        results["phase3_study_selection"] = run_phase3_study_selection(
            config, paths, pico_idx, logger
        )
    if stages["phase4_evidence_assessment"]:
        results["phase4_evidence_assessment"] = run_phase4_evidence_assessment(
            config, paths, pico_idx, logger
        )
    if stages["phase4_meta_analysis"]:
        results["phase4_meta_analysis"] = run_phase4_meta_analysis(
            config, paths, pico_idx, logger
        )
    if stages["phase5_recommendation_formulation"]:
        results["phase5_recommendation_formulation"] = (
            run_phase5_recommendation_formulation(config, paths, pico_idx, logger)
        )

    manifest_path = save_run_manifest(config, paths, pico_idx, results)
    logger.info("Run manifest saved to %s.", manifest_path)
    results["run_manifest"] = {"status": "completed", "outputs": [str(manifest_path)]}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Quicker Phase1-Phase5 pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to the pipeline configuration JSON.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(".env")
    project_root = Path(__file__).resolve().parent
    args = parse_args()
    config_path = resolve_path(project_root, args.config)
    config = load_json(config_path)
    config["_config_path"] = config_path
    config["_project_root"] = project_root

    logger = setup_pipeline_logging(config, project_root)
    logger.info("Starting Quicker pipeline with config %s.", config_path)
    try:
        results = run_pipeline(config, project_root, logger)
    except MissingPDFsError as exc:
        logger.warning(str(exc))
        print(str(exc))
        print(f"Missing PDF JSON: {exc.json_path}")
        print(f"Missing PDF Markdown: {exc.markdown_path}")
        raise SystemExit(2) from exc
    logger.info("Pipeline finished: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
