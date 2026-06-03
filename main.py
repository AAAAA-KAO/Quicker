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
    C: list[str] = Field(
        description=(
            "The comparison arms of the question. Return each comparator as a "
            "separate list item."
        )
    )
    O: dict[str, list[str]] = Field(
        description=(
            "The outcomes of the question. Every key must exactly match one "
            "item from C, and every item from C must have one key in O."
        )
    )


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


def apply_runtime_environment(config: dict, project_root: Path, pico_idx: str) -> None:
    runtime_config = config["pipeline"].get("runtime_environment", {})
    for key, value in runtime_config.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value).format(pico_idx=pico_idx)

    pdf_source_dir = config["pipeline"].get("pdf_handling", {}).get(
        "local_pdf_source_dir"
    )
    if pdf_source_dir:
        formatted_source = str(pdf_source_dir).format(pico_idx=pico_idx)
        os.environ["PAPER_SOURCE_DIR"] = str(resolve_path(project_root, formatted_source))


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


def normalize_outcome_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(normalize_outcome_values(item))
        return list(dict.fromkeys(values))
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(normalize_outcome_values(item))
        return list(dict.fromkeys(values))

    text = str(value).strip()
    return [text] if text else []


def normalize_outcome_mapping(outcome: Any, comparisons: list[str]) -> dict[str, list[str]]:
    normalized_comparisons = [str(comparison) for comparison in comparisons]
    if not normalized_comparisons:
        return {}

    if not isinstance(outcome, dict):
        common_outcomes = normalize_outcome_values(outcome)
        return {
            comparison: common_outcomes
            for comparison in normalized_comparisons
        }

    normalized_outcomes = {
        str(key): normalize_outcome_values(value)
        for key, value in outcome.items()
    }
    generic_keys = {
        "O",
        "Outcome",
        "Outcomes",
        "outcome",
        "outcomes",
        "All",
        "all",
        "Overall",
        "overall",
    }
    common_outcomes = []
    for key in generic_keys:
        common_outcomes.extend(normalized_outcomes.get(key, []))

    exact_comparators = set(normalized_comparisons)
    matched_any_comparator = any(
        comparison in normalized_outcomes for comparison in exact_comparators
    )
    if not common_outcomes and not matched_any_comparator:
        for key, values in normalized_outcomes.items():
            if key not in exact_comparators:
                common_outcomes.extend(values)
        common_outcomes = list(dict.fromkeys(common_outcomes))

    mapped_outcomes = {}
    for comparison in normalized_comparisons:
        values = normalized_outcomes.get(comparison)
        if values is None:
            comparison_lower = comparison.lower()
            fuzzy_values = []
            for key, key_values in normalized_outcomes.items():
                key_lower = key.lower()
                if comparison_lower in key_lower or key_lower in comparison_lower:
                    fuzzy_values.extend(key_values)
            values = list(dict.fromkeys(fuzzy_values)) or common_outcomes
        mapped_outcomes[comparison] = list(dict.fromkeys(values))

    return mapped_outcomes


def decode_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            decoded, _ = decoder.raw_decode(stripped[index:])
            return decoded
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object or array found in Phase1 model output.")


def merge_single_key_dicts(items: list[Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            key_text = str(key)
            if key_text not in merged:
                merged[key_text] = value
                continue
            existing_values = normalize_list_component(merged[key_text])
            existing_values.extend(normalize_list_component(value))
            merged[key_text] = existing_values
    return merged


def phase1_generation_to_components(generation: Any) -> tuple[Any, Any, Any, Any]:
    if isinstance(generation, QuestionDecompositionOutput):
        return generation.P, generation.I, generation.C, generation.O

    data = generation
    if hasattr(generation, "content"):
        data = generation.content
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        data = merge_single_key_dicts(data)
    elif isinstance(data, str):
        data = decode_json_from_text(data)
        if isinstance(data, list):
            data = merge_single_key_dicts(data)

    if not isinstance(data, dict):
        raise ValueError(
            f"Phase1 model output should be a dict or list of dicts, got {type(data).__name__}."
        )

    return data.get("P", []), data.get("I", []), data.get("C", []), data.get("O", {})


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


def load_json_list(path: Path) -> list:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} should contain a JSON list.")
    return data


def json_list_has_items(path: Path) -> bool:
    try:
        return len(load_json_list(path)) > 0
    except Exception:
        return False


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
    prompt = get_zero_shot_pipeline_prompt(dataset_name)
    chain = prompt | RunnableParallel(
        generation_chain=model,
        prompt_value=RunnablePassthrough(),
    )

    result = chain.invoke({"Question": clinical_question})
    generation = result["generation_chain"]
    logger.info(
        "Phase1 raw LLM output: %s",
        getattr(generation, "content", generation),
    )
    population, intervention, comparison, outcome = phase1_generation_to_components(
        generation
    )

    comparison_list = normalize_list_component(comparison)
    outcome_map = normalize_outcome_mapping(outcome, comparison_list)

    pico = {
        "Index": pico_idx,
        "Question": clinical_question,
        "P": normalize_text_component(population),
        "I": normalize_text_component(intervention),
        "C": comparison_list,
        "O": outcome_map,
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
    outcome_folder = paths["study_selection"] / "outcomeinfo"
    paper_folder = paths["study_selection"] / "paperinfo"

    outcome_files = (
        sorted(outcome_folder.glob(f"outcomeinfo_PICO{pico_idx}*.json"))
        if outcome_folder.exists()
        else []
    )
    paper_files = (
        [
            path
            for path in sorted(paper_folder.glob(f"paperinfo_PICO{pico_idx}*.json"))
            if "_full_text_assessed_but_not_included" not in path.name
            and json_list_has_items(path)
        ]
        if paper_folder.exists()
        else []
    )
    return outcome_files + paper_files if paper_files else []


def comparator_postfix(comparator: str) -> str:
    return f"_c{hashlib.md5(comparator.encode('utf-8')).hexdigest()[:6]}"


def infer_comparator_from_postfix(comparisons: list[str], postfix: str) -> str | None:
    if not postfix and len(comparisons) == 1:
        return comparisons[0]
    for comparator in comparisons:
        if comparator_postfix(comparator) == postfix:
            return comparator
    return None


def extract_paperinfo_postfix(path: Path, pico_idx: str) -> str:
    prefix = f"paperinfo_PICO{pico_idx}"
    if not path.stem.startswith(prefix):
        return ""
    return path.stem[len(prefix):]


def clean_extracted_outcome(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    cleaned = []
    for item in values:
        text = str(item).strip()
        if not text or text in {"[]", "<option>[]</option>"}:
            continue
        if "Insufficient evidence to draw a conclusion" in text:
            continue
        cleaned.append(text)
    return cleaned


def study_design_name(value: Any) -> str:
    if not value:
        return "RANDOMIZED_CONTROLLED_TRIAL"
    try:
        from utils.Evidence_Assessment.paper import StudyDesign

        return StudyDesign(value).name
    except Exception:
        return str(value)


def ensure_phase3_fallback_outcomes(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> list[Path]:
    phase_config = config["pipeline"]["phase3_study_selection"]
    fallback_config = phase_config.get("fallback_outcome_generation", {})
    if not fallback_config.get("enabled", False):
        return []

    pico = find_pico(paths, pico_idx)
    if not pico:
        return []

    paper_folder = paths["study_selection"] / "paperinfo"
    outcome_folder = paths["study_selection"] / "outcomeinfo"
    if not paper_folder.exists():
        return []

    outcome_folder.mkdir(parents=True, exist_ok=True)
    generated = []
    comparison_list = [str(item) for item in pico.get("C", [])]
    outcome_map = pico.get("O", {}) if isinstance(pico.get("O", {}), dict) else {}

    for paper_path in sorted(paper_folder.glob(f"paperinfo_PICO{pico_idx}*.json")):
        if "_full_text_assessed_but_not_included" in paper_path.name:
            continue
        if not json_list_has_items(paper_path):
            continue

        postfix = extract_paperinfo_postfix(paper_path, pico_idx)
        outcome_path = outcome_folder / f"outcomeinfo_PICO{pico_idx}{postfix}.json"
        if (
            fallback_config.get("only_when_outcomeinfo_empty", True)
            and outcome_path.exists()
            and json_list_has_items(outcome_path)
        ):
            continue

        comparator = infer_comparator_from_postfix(comparison_list, postfix)
        if comparator is None:
            logger.warning(
                "Cannot infer comparator for %s; fallback outcome generation skipped.",
                paper_path,
            )
            continue

        papers = load_json_list(paper_path)
        configured_outcomes = (
            normalize_list_component(outcome_map[comparator])
            if comparator in outcome_map
            else []
        )
        extracted_outcomes = []
        if fallback_config.get("include_extracted_outcomes", False):
            for paper in papers:
                characteristics = paper.get("characteristics") or {}
                outcome_info = characteristics.get("outcome") or {}
                extracted_outcomes.extend(
                    clean_extracted_outcome(outcome_info.get("outcome"))
                )

        outcome_names = []
        if fallback_config.get("use_configured_outcomes", True):
            outcome_names.extend(configured_outcomes)
        outcome_names.extend(extracted_outcomes)
        outcome_names = list(dict.fromkeys(outcome_names))
        if not outcome_names:
            continue

        related_paper_list = [
            str(paper["paper_uid"]) for paper in papers if paper.get("paper_uid")
        ]
        design = study_design_name(papers[0].get("study_design"))
        fallback_outcomes = []
        for outcome_name in outcome_names:
            fallback_outcomes.append(
                {
                    "outcome_uid": hashlib.sha256(
                        (comparator + outcome_name + design).encode("utf-8")
                    ).hexdigest()[:8],
                    "clinical_question": pico["Question"],
                    "population": pico["P"],
                    "intervention": pico["I"],
                    "comparator": comparator,
                    "outcome": outcome_name,
                    "importance": fallback_config.get("importance", "CRITICAL"),
                    "related_paper_list": related_paper_list,
                    "assessment_results": {"GRADE": {"Study design": design}},
                }
            )

        save_json(outcome_path, fallback_outcomes)
        generated.append(outcome_path)
        logger.info("Generated fallback outcomeinfo %s.", outcome_path)

    return generated


def run_phase3_study_selection(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    logger: logging.Logger,
) -> dict:
    phase_config = config["pipeline"]["phase3_study_selection"]
    existing_outputs = has_study_selection_outputs(paths, pico_idx)
    if phase_config["reuse_existing_outputs"] and existing_outputs:
        generated = ensure_phase3_fallback_outcomes(config, paths, pico_idx, logger)
        logger.info("Phase3 skipped because study-selection outputs already exist.")
        return {
            "status": "skipped",
            "reason": "existing study-selection outputs reused",
            "outputs": [str(path) for path in existing_outputs + generated],
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
    generated = ensure_phase3_fallback_outcomes(config, paths, pico_idx, logger)
    outputs = has_study_selection_outputs(paths, pico_idx)
    outputs.extend(generated)
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


def phase4_input_files_exist(
    paths: dict[str, Path],
    pico_idx: str,
    comparator_postfix_value: str | None,
) -> bool:
    suffix = comparator_postfix_value or ""
    outcome_path = (
        paths["evidence_assessment"]
        / "outcomeinfo"
        / f"outcomeinfo_PICO{pico_idx}{suffix}.json"
    )
    paper_path = (
        paths["evidence_assessment"]
        / "paperinfo"
        / f"paperinfo_PICO{pico_idx}{suffix}.json"
    )
    return (
        outcome_path.exists()
        and paper_path.exists()
        and json_list_has_items(outcome_path)
        and json_list_has_items(paper_path)
    )


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

    try:
        from utils.General.quicker import Quicker, QuickerData, QuickerStage
    except ModuleNotFoundError as exc:
        if phase_config.get("skip_when_dependency_missing", False):
            logger.warning("Phase4 skipped because dependency is missing: %s.", exc.name)
            outputs = matching_outcomeinfo_files(
                paths["evidence_assessment"],
                pico_idx,
                require_assessed=False,
            )
            return {
                "status": "skipped",
                "reason": f"missing dependency: {exc.name}",
                "outputs": [str(path) for path in outputs],
                "copied_inputs": [str(path) for path in copied],
            }
        raise

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
        comparator_postfix_value = input_postfix_map.get(comparator)
        if comparator_postfix_value is None and phase_config.get(
            "derive_comparator_postfix", True
        ):
            comparator_postfix_value = comparator_postfix(comparator)
        if not phase4_input_files_exist(
            paths=paths,
            pico_idx=pico_idx,
            comparator_postfix_value=comparator_postfix_value,
        ):
            if phase_config.get("skip_comparators_without_inputs", True):
                logger.info("Phase4 skipped comparator without inputs: %s.", comparator)
                continue
        quicker.load_outcome_list(comparator_postfix=comparator_postfix_value)
        quicker.load_paper_list(comparator_postfix=comparator_postfix_value)
        ensure_pdf_files_available(
            config=config,
            paths=paths,
            pico_idx=pico_idx,
            stage_name="phase4_evidence_assessment",
            papers=quicker.quicker_data.paper_list,
        )
        assessed.append(quicker.assess_evidence(comparator=comparator))

    if not assessed:
        logger.info("Phase4 skipped because no comparator inputs were available.")
        return {
            "status": "skipped",
            "reason": "no comparator inputs available",
            "outputs": [],
            "copied_inputs": [str(path) for path in copied],
        }

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


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:5]}***{value[-4:]}"


def generate_experiment_report(
    config: dict,
    paths: dict[str, Path],
    pico_idx: str,
    results: dict[str, dict],
    manifest_path: Path,
) -> Path:
    model_config = config["model"]
    phase5_outputs = results.get("phase5_recommendation_formulation", {}).get(
        "outputs", []
    )
    report_path = paths["reports"] / f"experiment_report_{pico_idx}.md"
    lines = [
        "# Quicker Phase1-Phase5 集成运行实验报告",
        "",
        "## 1. 实验概览",
        "",
        f"- 数据集：`{config['pipeline']['dataset_name']}`",
        f"- 疾病/主题：`{config['pipeline']['disease']}`",
        f"- 临床问题：`{config['pipeline']['clinical_question']}`",
        f"- PICO 索引：`{pico_idx}`",
        "- 集成入口脚本：`main.py`",
        f"- 运行配置文件：`{config['_config_path']}`",
        f"- 运行清单：`{manifest_path}`",
        f"- 最新推荐结果输出：`{phase5_outputs[-1] if phase5_outputs else '未生成'}`",
        "",
        "本次集成将 Phase1 到 Phase5 串联到同一个 Python 入口中运行。单独的 `Phase3-study_selection(full-text_assessment only).py` 未纳入集成流程，因为常规 Phase3 已包含全文评估功能。",
        "",
        "本次重新运行前已清除该临床问题在工作数据中的旧 PICO 记录，并检查确认不存在同一 PICO 索引的中间数据、报告、PDF 目录或其他可复用输出文件。",
        "",
        "## 2. 模型与运行配置",
        "",
        f"- LLM 模型：`{model_config['recommendation_formation_model']['model_name']}`",
        f"- API base URL：`{model_config['recommendation_formation_model']['BASE_URL']}`",
        f"- API key：`{mask_secret(model_config['recommendation_formation_model']['API_KEY'])}`",
        f"- Embedding 模型：`{model_config['embeddings']['model_name']}`",
        f"- 文献初筛方法：`{config['study_selection']['record_screening_method']}`",
        f"- 全文评估方法：`{config['study_selection']['full_text_assessment_method']}`",
        f"- 文献初筛重复次数：`{config['study_selection']['exp_num']}`",
        f"- 进入全文评估阈值：`{config['study_selection']['threshold']}`",
        f"- Phase4 meta-analysis：`{config['pipeline']['phase4_meta_analysis']['enabled']}`",
        "",
        "## 3. 集成工作流",
        "",
        "### Phase1：问题分解",
        "",
        "目标：将临床问题转换为结构化 PICO 信息。",
        "",
        "步骤：读取临床问题和模型配置；在 `pico_idx=auto` 时计算稳定索引；抽取 Population、Intervention、Comparators 和按 comparator 映射的 Outcomes；保存或更新 PICO 记录。",
        "",
        "输入：`config/config.json` 中的临床问题、数据集名称和模型配置。",
        "",
        f"输出：`{pico_file_path(paths)}`",
        "",
        "### Phase2：文献检索",
        "",
        "目标：生成 PubMed 检索策略，检索文献记录，并过滤不可用记录。",
        "",
        "步骤：读取 PICO；生成检索词和检索式；应用 PubMed 参数和过滤器；检索记录；移除无摘要、重复和无效文献类型；保存下游 Quicker 数据包。",
        "",
        f"输入：`{pico_file_path(paths)}`",
        "",
        f"输出：`{quicker_data_literature_search_path(paths, pico_idx)}`",
        "",
        "### Phase3：研究筛选",
        "",
        "目标：对检索记录做题录/摘要筛选，并对潜在纳入研究执行全文评估。",
        "",
        "步骤：加载 Phase2 数据；构建 PICO 筛选上下文；执行多轮题录筛选并按阈值投票；检查 PDF；用 RAG 抽取全文 PICO/研究特征；保存 paperinfo 和 outcomeinfo。若已有 paperinfo 但 outcomeinfo 为空，可按配置生成 fallback outcomeinfo 以保证下游可继续运行。",
        "",
        f"输入：`{quicker_data_literature_search_path(paths, pico_idx)}`、PDF 文件 `{paper_library_pico_path(paths, pico_idx)}`",
        "",
        f"输出：`{paths['study_selection'] / 'paperinfo'}`、`{paths['study_selection'] / 'outcomeinfo'}`、`{paths['study_selection'] / 'Results'}`",
        "",
        "### Phase4：证据评价",
        "",
        "目标：针对已筛选出的 outcomes 评价证据确定性和 GRADE 相关信息。",
        "",
        "步骤：转移 Phase3 的 paperinfo/outcomeinfo；按 comparator 加载输入；跳过没有输入文件的 comparator；检查 PDF；创建或复用论文向量库；执行配置中的 GRADE 评价；保存证据评价输出。",
        "",
        f"输入：`{paths['evidence_assessment'] / 'paperinfo'}`、`{paths['evidence_assessment'] / 'outcomeinfo'}`",
        "",
        f"输出：`{paths['evidence_assessment'] / 'outcomeinfo'}`",
        "",
        "### Phase4 可选项：Meta-Analysis",
        "",
        "目标：在配置启用时运行二分类或连续型 outcome 的统计学 meta-analysis 示例。",
        "",
        f"输出目录：`{paths['reports'] / 'meta_analysis' / pico_idx}`",
        "",
        "### Phase5：推荐形成",
        "",
        "目标：将证据评价结果综合为最终临床推荐文本。",
        "",
        "步骤：转移 Evidence Assessment 输出；加载可用 outcome evidence profile；解释每个 outcome；按 comparator 汇总；综合推荐依据；生成最终推荐；保存完整结果包。",
        "",
        f"输入：`{paths['recommendation_formation'] / 'outcomeinfo'}`",
        "",
        f"输出：`{paths['recommendation_formation']}`",
        "",
        "## 4. 本次运行结果",
        "",
    ]

    for stage_name, stage_result in results.items():
        lines.append(f"- `{stage_name}`：`{stage_result.get('status')}`")
        if stage_result.get("reason"):
            lines.append(f"  - 原因：{stage_result['reason']}")
        if stage_result.get("outputs"):
            lines.append("  - 输出：")
            lines.extend(f"    - `{output}`" for output in stage_result["outputs"])

    lines.extend(
        [
            "",
            "## 5. 修改的代码或配置",
            "",
            "- `main.py`：整合 Phase1-Phase5 入口；增加配置驱动的运行环境、PDF 缺失检查、Phase3 fallback outcomeinfo、Phase4 comparator 输入跳过、运行清单和实验报告生成；本次新增 Phase1 原始 JSON 解析兼容逻辑，可处理 `qwen3.5-plus` 返回的单键字典列表，并新增 outcome 标准化，确保 `O` 的 key 与 `C` 列表元素一一对应。",
            "- `utils/Evidence_Assessment/paper.py`：PDF 本地源目录改为由配置经环境变量注入；`PyPaperBot` 下载依赖改为可选导入，避免本地 PDF 流程被可选依赖阻断。",
            f"- `config/config.json`：配置 `{model_config['recommendation_formation_model']['model_name']}`、DashScope OpenAI-compatible base URL、API key、text-embedding-v4、临床问题、路径、阶段开关和各阶段运行参数。",
            "- `config/config-template.json`：补充新增配置字段模板。",
            "",
            "## 6. PDF 与中间文件说明",
            "",
            f"- PDF 主目录：`{paper_library_pico_path(paths, pico_idx)}`",
            f"- 缺失 PDF 请求文件模板：`{config['pipeline']['pdf_handling']['missing_pdf_markdown']}`",
            "- 若缺少 PDF，流程会停止并在 `reports/` 下写出缺失清单；补齐 PDF 后可重新运行 `python main.py --config config/config.json`。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pipeline(config: dict, project_root: Path, logger: logging.Logger) -> dict:
    paths = pipeline_paths(config, project_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    pico_idx = get_pico_idx(config)
    apply_runtime_environment(config, project_root, pico_idx)
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
    report_path = generate_experiment_report(
        config=config,
        paths=paths,
        pico_idx=pico_idx,
        results=results,
        manifest_path=manifest_path,
    )
    logger.info("Experiment report saved to %s.", report_path)
    results["experiment_report"] = {
        "status": "completed",
        "outputs": [str(report_path)],
    }
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
