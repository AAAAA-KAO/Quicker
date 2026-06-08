import glob
from pathlib import Path
from typing import Any

from utils.cli_config import write_json_file


def pico_paper_library_path(paper_library_path: str, pico_idx: str) -> Path:
    return Path(paper_library_path) / f"PICO{pico_idx}"


def expected_pdf_folder(
    paper: dict,
    paper_library_path: str,
    pico_idx: str,
) -> Path:
    return pico_paper_library_path(paper_library_path, pico_idx) / str(
        paper["paper_uid"]
    )


def expected_pdf_path(
    paper: dict,
    paper_library_path: str,
    pico_idx: str,
) -> Path:
    folder = expected_pdf_folder(paper, paper_library_path, pico_idx)
    return folder / f"{paper['paper_uid']}.pdf"


def find_existing_pdf(
    paper: dict,
    paper_library_path: str,
    pico_idx: str,
) -> str | None:
    save_folder_path = paper.get("save_folder_path")
    candidate_folders = []
    if save_folder_path:
        candidate_folders.append(Path(save_folder_path))
    candidate_folders.append(expected_pdf_folder(paper, paper_library_path, pico_idx))

    seen = set()
    for folder in candidate_folders:
        folder = folder.expanduser()
        if folder in seen:
            continue
        seen.add(folder)
        pdf_files = sorted(glob.glob(str(folder / "*.pdf")))
        if pdf_files:
            return str(Path(pdf_files[0]).resolve())
    return None


def build_pdf_manifest(
    papers: list[dict],
    paper_library_path: str,
    pico_idx: str,
    stage: str,
) -> dict[str, Any]:
    missing_pdfs = []
    existing_pdfs = []

    for paper in papers:
        paper_uid = paper.get("paper_uid")
        if not paper_uid:
            raise ValueError("Every paper must contain paper_uid.")

        existing_pdf = find_existing_pdf(paper, paper_library_path, pico_idx)
        if existing_pdf:
            existing_pdfs.append(
                {
                    "paper_uid": paper_uid,
                    "title": paper.get("title"),
                    "pdf_path": existing_pdf,
                }
            )
            continue

        folder = expected_pdf_folder(paper, paper_library_path, pico_idx)
        preferred_path = expected_pdf_path(paper, paper_library_path, pico_idx)
        folder.mkdir(parents=True, exist_ok=True)
        missing_pdfs.append(
            {
                "paper_uid": paper_uid,
                "title": paper.get("title"),
                "pmid": paper.get("pmid"),
                "doi": paper.get("doi"),
                "expected_folder": str(folder.resolve()),
                "expected_pdf_path": str(preferred_path.resolve()),
            }
        )

    return {
        "stage": stage,
        "pico_idx": pico_idx,
        "paper_library_path": str(
            pico_paper_library_path(paper_library_path, pico_idx).resolve()
        ),
        "instruction": (
            "请用户自行下载缺失 PDF，并放入每篇文献的 expected_folder；"
            "推荐使用 expected_pdf_path 文件名。脚本不会自动下载 PDF。"
        ),
        "total_paper_count": len(papers),
        "existing_pdf_count": len(existing_pdfs),
        "missing_pdf_count": len(missing_pdfs),
        "existing_pdfs": existing_pdfs,
        "missing_pdfs": missing_pdfs,
    }


def write_pdf_manifest(
    manifest: dict[str, Any],
    json_path: str,
    markdown_path: str | None = None,
) -> tuple[Path, Path | None]:
    json_output = write_json_file(json_path, manifest)
    markdown_output = None
    if markdown_path:
        markdown_output = Path(markdown_path)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_pdf_manifest_markdown(manifest), encoding="utf-8")
    return json_output, markdown_output


def render_pdf_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        f"# Missing PDF Request: {manifest['stage']}",
        "",
        f"- PICO index: `{manifest['pico_idx']}`",
        f"- Missing PDF count: `{manifest['missing_pdf_count']}`",
        "",
        "请自行下载每篇文献 PDF，并放到下列目录。文件名推荐使用清单中的 `expected_pdf_path`。",
        "",
    ]
    for idx, paper in enumerate(manifest["missing_pdfs"], start=1):
        lines.extend(
            [
                f"## {idx}. {paper['paper_uid']}",
                "",
                f"- Title: {paper.get('title') or ''}",
                f"- PMID: {paper.get('pmid') or ''}",
                f"- DOI: {paper.get('doi') or ''}",
                f"- Folder: `{paper['expected_folder']}`",
                f"- Preferred file path: `{paper['expected_pdf_path']}`",
                "",
            ]
        )
    return "\n".join(lines)


def resolve_manifest_path(
    reports_path: str,
    filename_pattern: str,
    stage: str,
    pico_idx: str,
) -> str:
    return str(
        Path(reports_path)
        / filename_pattern.format(stage=stage, pico_idx=pico_idx)
    )

