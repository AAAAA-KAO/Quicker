from __future__ import annotations

import base64
import copy
import csv
import glob
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "web" / "frontend"
RUNTIME_DIR = ROOT_DIR / "web" / "runtime"
TASKS_DIR = RUNTIME_DIR / "tasks"
CONFIGS_DIR = RUNTIME_DIR / "configs"
UPLOADS_DIR = RUNTIME_DIR / "uploads"

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DISEASES = {
    "Rheumatoid Arthritis (RA)": {
        "label": "Rheumatoid Arthritis (RA)",
        "kb_disease": "Rheumatoid Arthritis (RA)",
        "kb_disease_aliases": [
            "Rheumatoid Arthritis",
            "RA",
            "2021 ACR RA",
            "2021ACR RA",
        ],
        "dataset_name": "2021ACR RA",
        "dataset_path": "data/2021ACR RA",
        "config_template": "config/config.json",
        "disease": "Rheumatoid Arthritis (RA)",
    },
    "Appendicitis": {
        "label": "Appendicitis",
        "kb_disease": "acute appendicitis",
        "kb_disease_aliases": [
            "appendicitis",
            "acute appendicitis",
        ],
        "dataset_name": "Appendicitis",
        "dataset_path": "data/Appendicitis",
        "config_template": "config/config.json",
        "disease": "Appendicitis",
    },
}

PHASE_SEQUENCE = [
    ("phase1", "Phase1-question_decomposition.py"),
    ("phase2", "Phase2-literature_search.py"),
    ("phase3_record", "Phase3-study_selection.py"),
    ("phase3_full_text", "Phase3-full_text_assessment.py"),
    ("phase4", "Phase4-evidence_assessment.py"),
    ("phase5", "Phase5-recommendation_formulation.py"),
]

TASKS: dict[str, dict[str, Any]] = {}
TASK_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_runtime_dirs() -> None:
    for path in (TASKS_DIR, CONFIGS_DIR, UPLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def derive_pico_idx(question: str, dataset_name: str) -> str:
    return hashlib.sha256((question + dataset_name).encode("utf-8")).hexdigest()[:8]


def api_error(message: str, status: int = 400, details: Any = None) -> tuple[int, dict[str, Any]]:
    return status, {"ok": False, "error": message, "details": details}


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def append_event(task: dict[str, Any], message: str, level: str = "info") -> None:
    task.setdefault("events", []).append(
        {
            "time": utc_now(),
            "level": level,
            "message": message,
        }
    )
    task["updated_at"] = utc_now()
    persist_task(task)


def compact_final_artifact(final: Any) -> Any:
    if not isinstance(final, dict):
        return final
    path = final.get("path")
    compact = {
        "path": path,
        "recommendation": final.get("recommendation", ""),
        "final_result": {
            "omitted": "完整推荐结果较大，已保存在 path 指向的 JSON 文件中。",
            "path": path,
        },
    }
    return compact


def compact_record_artifact(record: Any, limit: int = 20) -> Any:
    if not isinstance(record, dict):
        return record
    data = record.get("data")
    if not isinstance(data, list):
        return record
    compact = dict(record)
    compact["total_count"] = len(data)
    compact["data"] = data[:limit]
    if len(data) > limit:
        compact["omitted_count"] = len(data) - limit
    return compact


def compact_pdf_manifest_artifact(artifact: Any, existing_limit: int = 5) -> Any:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("data"), dict):
        return artifact
    compact = dict(artifact)
    manifest = dict(artifact["data"])
    existing = manifest.get("existing_pdfs")
    if isinstance(existing, list) and len(existing) > existing_limit:
        manifest["existing_pdfs"] = existing[:existing_limit]
        manifest["existing_pdfs_omitted_count"] = len(existing) - existing_limit
    compact["data"] = manifest
    return compact


def task_snapshot(task: dict[str, Any], compact: bool = False) -> dict[str, Any]:
    payload = {key: value for key, value in task.items() if key != "_thread"}
    if compact and isinstance(payload.get("artifacts"), dict):
        artifacts = dict(payload["artifacts"])
        if "final_recommendation" in artifacts:
            artifacts["final_recommendation"] = compact_final_artifact(artifacts["final_recommendation"])
        if "record_included" in artifacts:
            artifacts["record_included"] = compact_record_artifact(artifacts["record_included"])
        if "pdf_manifest" in artifacts:
            artifacts["pdf_manifest"] = compact_pdf_manifest_artifact(artifacts["pdf_manifest"])
        payload = dict(payload)
        payload["artifacts"] = artifacts
    return copy.deepcopy(payload)


def persist_task(task: dict[str, Any]) -> None:
    task_id = task["task_id"]
    write_json(TASKS_DIR / f"{task_id}.json", task_snapshot(task))


def set_task_state(
    task: dict[str, Any],
    status: str,
    current_stage: str | None = None,
    awaiting: str | None = None,
) -> None:
    task["status"] = status
    if current_stage is not None:
        task["current_stage"] = current_stage
    task["awaiting"] = awaiting
    persist_task(task)


def load_runtime_tasks() -> None:
    ensure_runtime_dirs()
    for path in TASKS_DIR.glob("*.json"):
        try:
            task = read_json(path, {})
            if task.get("task_id"):
                TASKS[task["task_id"]] = task
        except Exception:
            continue


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return task_snapshot(task, compact=True)


def disease_config(disease: str) -> dict[str, Any]:
    if disease not in DISEASES:
        raise ValueError(f"Unsupported disease: {disease}")
    return DISEASES[disease]


def normalize_disease_for_match(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def disease_aliases(disease: str) -> set[str]:
    meta = disease_config(disease)
    raw_aliases = {
        meta.get("label", ""),
        meta.get("kb_disease", ""),
        meta.get("dataset_name", ""),
        meta.get("disease", ""),
        *meta.get("kb_disease_aliases", []),
    }
    return {
        alias
        for alias in (normalize_disease_for_match(value) for value in raw_aliases)
        if alias
    }


def record_disease_values(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("disease", "Disease", "dataset", "Dataset", "dataset_name", "Dataset_Name"):
        value = normalize_disease_for_match(record.get(key, ""))
        if value:
            values.add(value)
    source = record.get("source")
    if isinstance(source, dict):
        for key in ("disease", "dataset", "source_file"):
            value = normalize_disease_for_match(source.get(key, ""))
            if value:
                values.add(value.removesuffix(".json"))
    return values


def record_matches_disease(record: dict[str, Any], disease: str) -> bool:
    aliases = disease_aliases(disease)
    if not aliases:
        return True
    return bool(record_disease_values(record) & aliases)


def filter_retrieval_results_by_disease(
    retrieval_results: dict[str, list[Any]],
    disease: str,
) -> dict[str, list[Any]]:
    filtered: dict[str, list[Any]] = {}
    for method, items in retrieval_results.items():
        filtered[method] = [
            item for item in items
            if record_matches_disease(getattr(item, "record", {}), disease)
        ]
    return filtered


def retrieval_counts(retrieval_results: dict[str, list[Any]]) -> dict[str, int]:
    return {method: len(items) for method, items in retrieval_results.items()}


def create_task(disease: str, question: str) -> dict[str, Any]:
    meta = disease_config(disease)
    task_id = uuid.uuid4().hex[:12]
    pico_idx = derive_pico_idx(question, meta["dataset_name"])
    task = {
        "task_id": task_id,
        "disease": disease,
        "question": question,
        "pico_idx": pico_idx,
        "status": "created",
        "current_stage": "created",
        "awaiting": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "events": [],
        "artifacts": {},
        "uploaded_files": [],
        "config_path": str((CONFIGS_DIR / f"{task_id}.json").resolve()),
    }
    with TASK_LOCK:
        TASKS[task_id] = task
        persist_task(task)
    return task


def build_task_config(task: dict[str, Any]) -> dict[str, Any]:
    meta = disease_config(task["disease"])
    template_path = ROOT_DIR / meta["config_template"]
    config = read_json(template_path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config template: {template_path}")

    dataset_path = meta["dataset_path"]
    pipeline = config.setdefault("pipeline", {})
    pipeline["dataset_name"] = meta["dataset_name"]
    pipeline["dataset_path"] = dataset_path
    pipeline["disease"] = meta["disease"]
    pipeline["clinical_question"] = task["question"]
    pipeline["pico_idx"] = "auto"
    paths = pipeline.setdefault("paths", {})
    paths["dataset"] = dataset_path
    paths["question_decomposition"] = f"{dataset_path}/Question_Decomposition"
    paths["literature_search"] = f"{dataset_path}/Literature_Search"
    paths["study_selection"] = f"{dataset_path}/Study_Selection"
    paths["evidence_assessment"] = f"{dataset_path}/Evidence_Assessment"
    paths["paper_library"] = f"{dataset_path}/Paper_Library"
    paths["recommendation_formation"] = f"{dataset_path}/Recommendation_Formation"
    paths.setdefault("reports", "reports")
    return config


def prepare_task_config(task: dict[str, Any]) -> Path:
    config_path = Path(task["config_path"])
    if not config_path.exists():
        config = build_task_config(task)
        write_json(config_path, config)
        append_event(task, f"已生成任务配置: {safe_rel(config_path)}")
    return config_path


def dataset_root_from_config(config_path: Path) -> Path:
    config = read_json(config_path)
    dataset_path = config.get("pipeline", {}).get("paths", {}).get("dataset")
    if not dataset_path:
        dataset_path = config.get("pipeline", {}).get("dataset_path")
    return ROOT_DIR / dataset_path


def phase_command(script_name: str, config_path: Path) -> list[str]:
    raw = os.getenv("QUICKER_PHASE_PYTHON")
    if raw:
        base = raw.split()
    else:
        base = ["conda", "run", "-n", "quicker", "python"]
    return [*base, script_name, "--YOUR_CONFIG_PATH", str(config_path)]


def run_phase(task: dict[str, Any], stage: str, script_name: str, config_path: Path) -> int:
    append_event(task, f"开始执行 {stage}: {script_name}")
    set_task_state(task, "running", current_stage=stage)
    cmd = phase_command(script_name, config_path)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if not text:
            continue
        output_lines.append(text)
        append_event(task, text, level="process")
    return_code = process.wait()
    task.setdefault("phase_outputs", {})[stage] = output_lines[-80:]
    persist_task(task)
    if return_code != 0:
        raise RuntimeError(f"{stage} failed with exit code {return_code}")
    append_event(task, f"{stage} 执行完成")
    return return_code


def latest_recommendation_file(config_path: Path, pico_idx: str) -> Path | None:
    config = read_json(config_path)
    rec_path = config.get("pipeline", {}).get("paths", {}).get("recommendation_formation")
    if not rec_path:
        return None
    folder = ROOT_DIR / rec_path
    if not folder.exists():
        return None
    candidates = sorted(folder.glob(f"quicker_data(PICO_IDX{pico_idx})_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def read_final_recommendation(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    data = read_json(path)
    final_result = data.get("final_result")
    if not final_result:
        return None
    recommendation = final_result.get("recommendation") if isinstance(final_result, dict) else final_result
    return {
        "path": str(path.resolve()),
        "recommendation": recommendation,
        "final_result": final_result,
        "raw": data,
    }


def find_existing_final(task: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    final = read_final_recommendation(latest_recommendation_file(config_path, task["pico_idx"]))
    if final:
        task["artifacts"]["final_recommendation"] = final
        append_event(task, "发现本地已有最终推荐，跳过推理阶段")
        set_task_state(task, "completed", current_stage="completed")
    return final


def pico_file(config_path: Path) -> Path:
    config = read_json(config_path)
    path = config.get("pipeline", {}).get("paths", {}).get("question_decomposition")
    return ROOT_DIR / path / "PICO_Information.json"


def read_pico_for_task(config_path: Path, pico_idx: str) -> dict[str, Any] | None:
    path = pico_file(config_path)
    data = read_json(path, [])
    if not isinstance(data, list):
        return None
    for item in data:
        if str(item.get("Index")) == str(pico_idx):
            return item
    return None


def write_pico_for_task(config_path: Path, pico_idx: str, pico: dict[str, Any], question: str) -> None:
    path = pico_file(config_path)
    existing = read_json(path, [])
    if not isinstance(existing, list):
        existing = []
    next_item = {
        "Index": pico_idx,
        "Question": question,
        "P": pico.get("P", []),
        "I": pico.get("I", []),
        "C": pico.get("C", []),
        "O": pico.get("O", {}),
    }
    existing = [item for item in existing if str(item.get("Index")) != str(pico_idx)]
    existing.append(next_item)
    write_json(path, existing)


def phase3_record_path(config_path: Path, pico_idx: str) -> Path:
    config = read_json(config_path)
    base = config.get("pipeline", {}).get("paths", {}).get("study_selection")
    return ROOT_DIR / base / "record_included_studies" / f"record_included_PICO{pico_idx}.json"


def phase3_manifest_path(config_path: Path, pico_idx: str) -> Path:
    config = read_json(config_path)
    base = config.get("pipeline", {}).get("paths", {}).get("study_selection")
    return ROOT_DIR / base / "record_included_studies" / f"missing_pdfs_phase3_record_screening_PICO{pico_idx}.json"


def collect_phase3_artifacts(task: dict[str, Any], config_path: Path) -> None:
    record_path = phase3_record_path(config_path, task["pico_idx"])
    manifest_path = phase3_manifest_path(config_path, task["pico_idx"])
    task["artifacts"]["record_included"] = {
        "path": str(record_path.resolve()),
        "data": read_json(record_path, []),
    }
    task["artifacts"]["pdf_manifest"] = {
        "path": str(manifest_path.resolve()),
        "data": refresh_pdf_manifest(read_json(manifest_path, {})),
    }
    persist_task(task)


def find_literature_search_file(config_path: Path, pico_idx: str) -> Path | None:
    """Find the literature search JSON output for a given PICO index."""
    config = read_json(config_path)
    base = config.get("pipeline", {}).get("paths", {}).get("literature_search")
    if not base:
        return None
    pattern = str(ROOT_DIR / base / "**" / f"PICO{pico_idx}.json")
    matches = glob.glob(pattern, recursive=True)
    return Path(matches[0]) if matches else None


def extract_literature_search_sample(path: Path, max_samples: int = 3) -> dict[str, Any]:
    """Extract total count and sample papers from a literature search JSON file."""
    data = read_json(path, [])
    if not isinstance(data, list):
        return {"total_count": 0, "sample": []}
    sample: list[dict[str, str]] = []
    for paper in data[:max_samples]:
        sample.append({
            "title": str(paper.get("Title", "") or ""),
            "pmid": str(paper.get("Paper_Index", "") or ""),
            "year": str(paper.get("Published", "") or "").split("-")[0],
        })
    return {"total_count": len(data), "sample": sample}


def find_screening_csv(config_path: Path, pico_idx: str) -> Path | None:
    """Find the screening CSV output for a given PICO index."""
    config = read_json(config_path)
    base = config.get("pipeline", {}).get("paths", {}).get("study_selection")
    if not base:
        return None
    pattern = str(ROOT_DIR / base / "Results" / "screening_records" / "*" / pico_idx / "*.csv")
    matches = glob.glob(pattern, recursive=True)
    return Path(matches[0]) if matches else None


def extract_screening_summary(csv_path: Path, max_samples: int = 3) -> dict[str, Any]:
    """Extract screening statistics and sample included papers from a CSV."""
    total_screened = 0
    total_included = 0
    total_excluded = 0
    sample_included: list[dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                total_screened += 1
                verdict = str(row.get("llm_record_screening_verdict", "") or "").strip()
                if verdict == "Included":
                    total_included += 1
                    if len(sample_included) < max_samples:
                        sample_included.append({
                            "title": str(row.get("Title", "") or ""),
                            "pmid": str(row.get("Paper_Index", "") or ""),
                            "year": str(row.get("Published", "") or "").split("-")[0],
                            "verdict": verdict,
                            "reason": str(row.get("llm_record_screening_reason", "") or "")[:200],
                        })
                elif verdict == "Excluded":
                    total_excluded += 1
    except Exception:
        pass
    return {
        "total_screened": total_screened,
        "total_included": total_included,
        "total_excluded": total_excluded,
        "total_selected": total_included,
        "sample_included": sample_included,
    }


def paper_title(record: dict[str, Any]) -> str:
    return str(record.get("title") or record.get("Title") or "(无标题)")


def paper_pmid(record: dict[str, Any]) -> str:
    return str(record.get("pmid") or record.get("Paper_Index") or "")


def paper_year(record: dict[str, Any]) -> str:
    return str(record.get("year") or record.get("Published") or "").split("-")[0]


def extract_record_included_summary(
    record_path: Path,
    total_screened: int | None = None,
    max_samples: int = 3,
) -> dict[str, Any]:
    records = read_json(record_path, [])
    if not isinstance(records, list):
        records = []
    sample = []
    for paper in records[:max_samples]:
        if not isinstance(paper, dict):
            continue
        sample.append({
            "title": paper_title(paper),
            "pmid": paper_pmid(paper),
            "year": paper_year(paper),
            "verdict": "Included",
            "reason": "",
        })
    included = len(records)
    return {
        "total_screened": total_screened if total_screened is not None else included,
        "total_included": included,
        "total_excluded": None,
        "total_selected": included,
        "sample_included": sample,
    }


def refresh_pdf_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    refreshed = copy.deepcopy(manifest)
    missing = refreshed.get("missing_pdfs", [])
    existing = refreshed.get("existing_pdfs", [])
    if not isinstance(missing, list):
        missing = []
    if not isinstance(existing, list):
        existing = []

    still_missing = []
    now_existing = list(existing)
    seen_existing_paths = {
        str(item.get("pdf_path") or item.get("expected_pdf_path") or "")
        for item in now_existing
        if isinstance(item, dict)
    }
    for item in missing:
        if not isinstance(item, dict):
            continue
        expected = Path(str(item.get("expected_pdf_path", "")))
        if expected.exists():
            existing_item = dict(item)
            existing_item["pdf_path"] = str(expected)
            if str(expected) not in seen_existing_paths:
                now_existing.append(existing_item)
                seen_existing_paths.add(str(expected))
        else:
            still_missing.append(item)

    refreshed["existing_pdfs"] = now_existing
    refreshed["missing_pdfs"] = still_missing
    refreshed["existing_pdf_count"] = len(now_existing)
    refreshed["missing_pdf_count"] = len(still_missing)
    refreshed["total_paper_count"] = refreshed.get("total_paper_count") or len(now_existing) + len(still_missing)
    return refreshed


def collect_screening_summary(task: dict[str, Any], config_path: Path) -> None:
    csv_path = find_screening_csv(config_path, task["pico_idx"])
    if csv_path:
        task["artifacts"]["screening_summary"] = extract_screening_summary(csv_path)
        task["artifacts"]["screening_summary"]["path"] = str(csv_path.resolve())
        persist_task(task)
        return

    record_path = phase3_record_path(config_path, task["pico_idx"])
    literature_total = task.get("artifacts", {}).get("literature_search", {}).get("total_count")
    try:
        literature_total = int(literature_total) if literature_total is not None else None
    except Exception:
        literature_total = None
    if record_path.exists():
        summary = extract_record_included_summary(record_path, total_screened=literature_total)
        summary["path"] = str(record_path.resolve())
        task["artifacts"]["screening_summary"] = summary
        persist_task(task)


def phase3_record_output_exists(config_path: Path, pico_idx: str) -> bool:
    return phase3_record_path(config_path, pico_idx).exists() or find_screening_csv(config_path, pico_idx) is not None


def phase3_full_text_output_exists(config_path: Path, pico_idx: str) -> bool:
    """Check if Phase3 full-text assessment has already produced output."""
    config = read_json(config_path)
    base = config.get("pipeline", {}).get("paths", {}).get("study_selection")
    if not base:
        return False
    for sub in ("paperinfo", "outcomeinfo"):
        d = ROOT_DIR / base / sub
        if d.exists() and list(d.glob(f"*PICO{pico_idx}*")):
            return True
    return False


def phase4_output_exists(config_path: Path, pico_idx: str) -> bool:
    """Check if Phase4 evidence assessment has already produced output."""
    config = read_json(config_path)
    base = config.get("pipeline", {}).get("paths", {}).get("evidence_assessment")
    if not base:
        return False
    for sub in ("paperinfo", "outcomeinfo"):
        d = ROOT_DIR / base / sub
        if d.exists() and list(d.glob(f"*PICO{pico_idx}*")):
            return True
    return False


def run_reasoning_until_pause(task_id: str, start_stage: str = "phase1") -> None:
    with TASK_LOCK:
        task = TASKS[task_id]
    try:
        config_path = prepare_task_config(task)

        if start_stage == "phase1":
            pico = read_pico_for_task(config_path, task["pico_idx"])
            if pico:
                task["artifacts"]["pico"] = {
                    "path": str(pico_file(config_path).resolve()),
                    "data": pico,
                }
                append_event(task, "检测到已有 PICO 分解结果，跳过 Phase1 计算，请确认或修改")
            else:
                run_phase(task, "phase1", "Phase1-question_decomposition.py", config_path)
                pico = read_pico_for_task(config_path, task["pico_idx"])
                task["artifacts"]["pico"] = {
                    "path": str(pico_file(config_path).resolve()),
                    "data": pico,
                }
                append_event(task, "Phase1 已完成，等待确认 PICO")
            set_task_state(task, "waiting", current_stage="phase1_review", awaiting="pico_review")
            return

        if start_stage == "phase2":
            ls_file = find_literature_search_file(config_path, task["pico_idx"])
            if ls_file:
                summary = extract_literature_search_sample(ls_file)
                task["artifacts"]["literature_search"] = {
                    "path": str(ls_file.resolve()),
                    "total_count": summary["total_count"],
                    "sample": summary["sample"],
                }
                append_event(
                    task,
                    f"检测到已有文献检索结果（共 {summary['total_count']} 篇），跳过 Phase2 计算，请确认"
                )
            else:
                run_phase(task, "phase2", "Phase2-literature_search.py", config_path)
                ls_file = find_literature_search_file(config_path, task["pico_idx"])
                summary = extract_literature_search_sample(ls_file) if ls_file else {"total_count": 0, "sample": []}
                task["artifacts"]["literature_search"] = {
                    "path": str(ls_file.resolve()) if ls_file else "",
                    "total_count": summary["total_count"],
                    "sample": summary["sample"],
                }
                append_event(task, f"Phase2 已完成，共检索到 {summary['total_count']} 篇文献，请确认")
            set_task_state(task, "waiting", current_stage="phase2_review", awaiting="phase2_review")
            return

        if start_stage == "phase3_record":
            record_path = phase3_record_path(config_path, task["pico_idx"])
            if phase3_record_output_exists(config_path, task["pico_idx"]):
                collect_phase3_artifacts(task, config_path)
                collect_screening_summary(task, config_path)
                append_event(task, "检测到已有题录筛选结果，跳过 Phase3 计算")
            else:
                run_phase(task, "phase3_record", "Phase3-study_selection.py", config_path)
                collect_phase3_artifacts(task, config_path)
                collect_screening_summary(task, config_path)
                append_event(task, "Phase3 题录筛选已完成")
            set_task_state(task, "waiting", current_stage="phase3_record_review", awaiting="pdf_upload")
            return

        if start_stage == "phase3_full_text":
            ran_any = False

            if phase3_full_text_output_exists(config_path, task["pico_idx"]):
                append_event(task, "检测到已有 Phase3 全文评估结果，跳过 Phase3_full_text 计算")
            else:
                run_phase(task, "phase3_full_text", "Phase3-full_text_assessment.py", config_path)
                ran_any = True

            if phase4_output_exists(config_path, task["pico_idx"]):
                append_event(task, "检测到已有 Phase4 证据评价结果，跳过 Phase4 计算")
            else:
                run_phase(task, "phase4", "Phase4-evidence_assessment.py", config_path)
                ran_any = True

            if latest_recommendation_file(config_path, task["pico_idx"]):
                append_event(task, "检测到已有 Phase5 推荐形成结果，跳过 Phase5 计算")
            else:
                run_phase(task, "phase5", "Phase5-recommendation_formulation.py", config_path)
                ran_any = True

            final = read_final_recommendation(latest_recommendation_file(config_path, task["pico_idx"]))
            task["artifacts"]["final_recommendation"] = final
            if ran_any:
                append_event(task, "推理任务完成")
            else:
                append_event(task, "所有阶段结果已存在，推理任务完成（跳过所有计算）")
            set_task_state(task, "completed", current_stage="completed")
            return

        raise ValueError(f"Unknown start_stage: {start_stage}")
    except Exception as exc:
        task["error"] = str(exc)
        task["traceback"] = traceback.format_exc()
        append_event(task, str(exc), level="error")
        set_task_state(task, "failed", current_stage=task.get("current_stage"))


def start_task_thread(task: dict[str, Any], start_stage: str) -> None:
    thread = threading.Thread(
        target=run_reasoning_until_pause,
        args=(task["task_id"], start_stage),
        daemon=True,
    )
    task["_thread"] = thread
    thread.start()


def route_decision(route_result: dict[str, Any]) -> str:
    return str(route_result.get("decision", route_result.get("判断", ""))).strip().lower()


def answer_from_candidates(route_result: dict[str, Any], retrieval_summary: dict[str, Any]) -> str:
    if route_decision(route_result) != "yes":
        return ""
    answer = route_result.get(
        "candidate_based_brief_answer",
        route_result.get("基于候选的简短答案", ""),
    )
    if answer:
        return answer
    ranks = route_result.get(
        "supporting_candidate_ranks",
        route_result.get("依据候选排名", []),
    )
    for rank in ranks:
        try:
            idx = int(rank) - 1
        except Exception:
            continue
        candidates = retrieval_summary.get("hybrid", [])
        if 0 <= idx < len(candidates):
            candidate_answer = candidates[idx].get("answer", "")
            if isinstance(candidate_answer, list):
                return "\n".join(str(item) for item in candidate_answer)
            return str(candidate_answer)
    return ""


def suppress_non_answerable_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    if route_decision(evidence.get("route", {})) == "yes":
        return evidence
    sanitized = copy.deepcopy(evidence)
    sanitized["answer"] = ""
    route = sanitized.get("route")
    if isinstance(route, dict):
        route["candidate_based_brief_answer"] = ""
        route["基于候选的简短答案"] = ""
    return sanitized


def create_reasoning_response(
    disease: str,
    question: str,
    evidence: dict[str, Any],
    event_message: str,
) -> tuple[int, dict[str, Any]]:
    evidence = suppress_non_answerable_evidence(evidence)
    task = create_task(disease, question)
    task["artifacts"]["evidenceqa"] = evidence
    append_event(task, event_message)
    start_task_thread(task, "phase1")
    return 200, {
        "ok": True,
        "mode": "reasoning",
        "task_id": task["task_id"],
        "task": public_task(task),
        "evidence": evidence,
    }


def run_evidenceqa(question: str, disease: str) -> dict[str, Any]:
    import evidenceqa_pipeline as ep

    ep.load_env_file(ROOT_DIR / ".env")
    model = os.getenv("DEEPSEEK_MODEL", ep.DEFAULT_LLM_MODEL)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", ep.DEFAULT_BASE_URL)
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env or environment.")

    llm = ep.build_llm(model=model, api_key=api_key, base_url=base_url, temperature=0.0)
    picos = ep.decompose_questions_batch(llm=llm, questions=[question], max_concurrency=2)
    pico = ep.normalize_pico(picos[0] if picos else ep.EMPTY_PICO)
    config = ep.RetrievalConfig(
        top_k=5,
        device=os.getenv("QUICKER_RETRIEVAL_DEVICE", "cpu"),
        qdrant_host=os.getenv("QUICKER_QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QUICKER_QDRANT_PORT", "6333")),
    )
    retriever = ep.ClinicalQAHybridRetriever(config)
    retrieval_results = retriever.retrieve(question, pico)
    unfiltered_counts = retrieval_counts(retrieval_results)

    kb_disease = disease_config(disease).get("kb_disease") or ""
    retrieval_results = filter_retrieval_results_by_disease(retrieval_results, disease)
    filtered_counts = retrieval_counts(retrieval_results)

    serialized = ep.serialize_retrieval_results(retrieval_results)
    candidates = ep.normalize_retrieval_results(
        serialized,
        retrieval_method="hybrid",
        max_candidates=5,
    )
    route_result = ep.judge_route_batch(
        llm=llm,
        cases=[
            {
                "question": question,
                "pico": pico,
                "candidates": candidates,
                "max_answer_chars": 4000,
            }
        ],
        max_concurrency=2,
    )[0]

    retrieval_summary = ep.summarize_retrieval_results(retrieval_results, max_answer_chars=1200)
    return {
        "question": question,
        "pico": pico,
        "retrieval": retrieval_summary,
        "route": route_result,
        "answer": answer_from_candidates(route_result, retrieval_summary),
        "kb_disease": kb_disease,
        "disease_aliases": sorted(disease_aliases(disease)),
        "retrieval_counts": {
            "before_disease_filter": unfiltered_counts,
            "after_disease_filter": filtered_counts,
        },
    }


def handle_ask(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    disease = str(payload.get("disease", "")).strip()
    question = str(payload.get("question", "")).strip()
    if not disease or not question:
        return api_error("disease and question are required")
    try:
        disease_config(disease)
    except ValueError as exc:
        return api_error(str(exc))

    evidence = run_evidenceqa(question, disease)
    if route_decision(evidence.get("route", {})) == "yes":
        return 200, {
            "ok": True,
            "mode": "knowledge_base",
            "answer": evidence.get("answer", ""),
            "evidence": evidence,
        }

    return create_reasoning_response(disease, question, evidence, "知识库不足以直接回答，已创建推理任务")


def get_task_or_error(task_id: str) -> dict[str, Any]:
    with TASK_LOCK:
        task = TASKS.get(task_id)
    if not task:
        raise KeyError(f"Task not found: {task_id}")
    return task


def handle_continue(task_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        task = get_task_or_error(task_id)
    except KeyError as exc:
        return api_error(str(exc), status=404)
    if task.get("status") != "waiting":
        return api_error("Task is not waiting for user input.")

    config_path = Path(task["config_path"])
    awaiting = task.get("awaiting")
    if awaiting == "pico_review":
        pico = payload.get("pico") or payload.get("data")
        if not isinstance(pico, dict):
            return api_error("pico must be a JSON object.")
        write_pico_for_task(config_path, task["pico_idx"], pico, task["question"])
        task["artifacts"]["pico"] = {
            "path": str(pico_file(config_path).resolve()),
            "data": read_pico_for_task(config_path, task["pico_idx"]),
        }
        append_event(task, "已保存用户确认的 PICO，继续 Phase2")
        set_task_state(task, "running", current_stage="phase2")
        start_task_thread(task, "phase2")
        return 200, {"ok": True, "task": public_task(task)}

    if awaiting == "phase2_review":
        append_event(task, "用户确认文献检索结果，继续 Phase3 题录筛选")
        set_task_state(task, "running", current_stage="phase3_record")
        start_task_thread(task, "phase3_record")
        return 200, {"ok": True, "task": public_task(task)}

    if awaiting == "pdf_upload":
        append_event(task, "用户确认 PDF 上传步骤，继续全文评估")
        set_task_state(task, "running", current_stage="phase3_full_text")
        start_task_thread(task, "phase3_full_text")
        return 200, {"ok": True, "task": public_task(task)}

    return api_error(f"Unsupported awaiting state: {awaiting}")


def match_upload_target(manifest: dict[str, Any], file_name: str, target_paper_uid: str | None) -> Path | None:
    missing = manifest.get("missing_pdfs", [])
    if not isinstance(missing, list):
        missing = []
    stem = Path(file_name).stem.lower()
    for item in missing:
        uid = str(item.get("paper_uid", ""))
        if target_paper_uid and uid == target_paper_uid:
            return Path(item["expected_pdf_path"])
        if uid and uid.lower() in stem:
            return Path(item["expected_pdf_path"])
    if len(missing) == 1:
        return Path(missing[0]["expected_pdf_path"])
    return None


def handle_upload(task_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        task = get_task_or_error(task_id)
    except KeyError as exc:
        return api_error(str(exc), status=404)
    config_path = Path(task["config_path"])
    manifest = read_json(phase3_manifest_path(config_path, task["pico_idx"]), {})
    files = payload.get("files")
    if not isinstance(files, list):
        return api_error("files must be a list.")
    saved = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "uploaded.pdf"))
        content_base64 = str(item.get("content_base64", ""))
        target_uid = item.get("target_paper_uid")
        if "," in content_base64 and content_base64.startswith("data:"):
            content_base64 = content_base64.split(",", 1)[1]
        try:
            content = base64.b64decode(content_base64)
        except Exception:
            continue
        target = match_upload_target(manifest, name, target_uid)
        if target is None:
            target = UPLOADS_DIR / task_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        saved.append(str(target.resolve()))
    task.setdefault("uploaded_files", []).extend(saved)
    task["artifacts"]["pdf_manifest"] = {
        "path": str(phase3_manifest_path(config_path, task["pico_idx"]).resolve()),
        "data": refresh_pdf_manifest(manifest),
    }
    append_event(task, f"已保存 {len(saved)} 个上传文件")
    return 200, {"ok": True, "saved": saved, "task": public_task(task)}


def route_api(method: str, path: str, query: dict[str, list[str]], body: Any) -> tuple[int, dict[str, Any]]:
    if method == "GET" and path == "/api/health":
        return 200, {"ok": True, "root": str(ROOT_DIR), "time": utc_now()}
    if method == "GET" and path == "/api/diseases":
        return 200, {"ok": True, "diseases": [meta["label"] for meta in DISEASES.values()]}
    if method == "POST" and path == "/api/ask":
        return handle_ask(body or {})

    parts = [part for part in path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "tasks":
        task_id = parts[2]
        if method == "GET" and len(parts) == 3:
            try:
                task = get_task_or_error(task_id)
            except KeyError as exc:
                return api_error(str(exc), status=404)
            return 200, {"ok": True, "task": public_task(task)}
        if method == "POST" and len(parts) == 4 and parts[3] == "continue":
            return handle_continue(task_id, body or {})
        if method == "POST" and len(parts) == 4 and parts[3] == "upload":
            return handle_upload(task_id, body or {})

    return api_error("Not found", status=404)


class WebHandler(BaseHTTPRequestHandler):
    server_version = "QuickerWeb/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (utc_now(), fmt % args))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            status, payload = route_api("GET", parsed.path, parse_qs(parsed.query), None)
            self._send_json(status, payload)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return
        try:
            status, payload = route_api("POST", parsed.path, parse_qs(parsed.query), body)
        except Exception as exc:
            status, payload = api_error(str(exc), status=500, details=traceback.format_exc())
        self._send_json(status, payload)

    def serve_static(self, request_path: str) -> None:
        if request_path in ("", "/"):
            file_path = FRONTEND_DIR / "index.html"
        else:
            safe_path = request_path.lstrip("/")
            file_path = (FRONTEND_DIR / safe_path).resolve()
            if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
                self.send_error(403)
                return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    ensure_runtime_dirs()
    load_runtime_tasks()
    host = os.getenv("QUICKER_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("QUICKER_WEB_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), WebHandler)
    print(f"Quicker web app running at http://{host}:{port}")
    print("Phase subprocess command: " + " ".join(phase_command("<script>", Path("<config>"))))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
