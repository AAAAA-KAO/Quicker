"""Build a clinical-guideline QA knowledge base from MIMIC-CPG JSON files.

功能：
    从 data/mimic-cpg 下的 appendicitis.json、pancreatitis.json、
    cholecystitis.json 中抽取临床问题与对应推荐，并标准化为
    data/mimic-cpg/template.json 所示字段结构。

输入：
    --data-dir 指向包含 template.json 和 3 个疾病指南 JSON 的目录。

输出：
    默认写入 results/mimic_cpg_knowledge_base.json，内容为 QA 条目列表。
    每个条目包含 question_id、question、answer、disease、topic、pico、
    source、synonyms、created_at、search_text。

三阶段流程：
    1. 规则解析：按不同指南版式抽取 question、answer、disease、topic、
       source、created_at，并生成形如 f"{disease}-{id}" 的 question_id。
    2. LLM 批量补齐：使用 LangChain 调用 DeepSeek OpenAI-compatible API，
       通过 batch 调用抽取 pico 与 synonyms。
    3. 检索文本：按行拼接 question、disease、topic、pico、synonyms，
       写入 search_text 字段。

在项目根目录运行：
    conda run -n quicker python src/build_kb.py

常用示例：
    # 仅验证规则解析和输出结构，不调用 LLM
    conda run -n quicker python src/build_kb.py --skip-llm

    # 指定输出文件和批量大小
    conda run -n quicker python src/build_kb.py \
        --output-file results/mimic_cpg_qa.json \
        --batch-size 16 \
        --max-concurrency 4

命令行参数：
    --data-dir: 指南 JSON 所在目录，默认 data/mimic-cpg。
    --output-file: 标准化知识库输出路径，默认 results/mimic_cpg_knowledge_base.json。
    --env-file: 环境变量文件路径，默认 .env。
    --log-file: 日志文件路径；不传则自动写入 logs/build_kb.log。
    --batch-size: 每次发送给 LLM batch 的条目数，默认 16。
    --max-concurrency: LangChain batch 并发数，默认 4。
    --skip-llm: 跳过 LLM 阶段，pico/synonyms 使用空模板，便于本地快速验证。
    --only-disease: 只处理一个疾病，可选 appendicitis、pancreatitis、cholecystitis。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Optional

from utils.logging import log_step, setup_logging


DEFAULT_PICO = {"P": "", "I": "", "C": [], "O": {}}

OCR_JOIN_FIXES = {
    "crosssectional": "cross-sectional",
    "firstline": "first-line",
    "secondline": "second-line",
    "highrisk": "high-risk",
    "lowrisk": "low-risk",
    "videoassisted": "video-assisted",
    "singlestage": "single-stage",
    "gallstonerelated": "gallstone-related",
}

SOURCE_META = {
    "appendicitis": {
        "disease": "acute appendicitis",
        "title": "Diagnosis and treatment of acute appendicitis: 2020 update of the WSES Jerusalem guidelines",
        "year": 2020,
        "organization": "WSES",
        "source_file": "appendicitis.json",
    },
    "pancreatitis": {
        "disease": "severe acute pancreatitis",
        "title": "2019 WSES guidelines for the management of severe acute pancreatitis",
        "year": 2019,
        "organization": "WSES",
        "source_file": "pancreatitis.json",
    },
    "cholecystitis": {
        "disease": "acute calculus cholecystitis",
        "title": "2020 WSES updated guidelines for the diagnosis and treatment of acute calculus cholecystitis",
        "year": 2020,
        "organization": "WSES",
        "source_file": "cholecystitis.json",
    },
}

CHOLE_REC_TO_QUESTION_ORDINAL = {
    "1.1": 1,
    "1.2": 1,
    "1.3": 2,
    "1.4": 3,
    "2.1": 1,
    "2.2": 2,
    "2.3": 2,
    "2.4": 3,
    "2.5": 4,
    "2.6": 5,
    "2.7": 6,
    "2.8": 7,
    "3.1": 1,
    "3.2": 2,
    "3.3": 3,
    "3.4": 4,
    "3.5": 5,
    "4.1": 1,
    "4.2": 1,
    "5.1": 1,
    "6.1": 1,
    "6.2": 1,
    "6.3": 2,
    "6.4": 3,
    "6.5": 4,
    "6.6": 5,
    "6.7": 5,
    "6.8": 6,
    "6.9": 6,
    "7.1": 1,
    "7.2": 2,
    "7.3": 3,
}

PANCREATITIS_STATEMENT_TO_QUESTION = {
    0: {
        "severity grading": [1],
        "imaging": [2],
        "diagnostic laboratory parameters": [3],
        "diagnostics in idiopathic pancreatitis": [4],
        "risk scores": [5],
        "follow-up imaging": [6],
    },
    1: {
        "prophylactic antibiotics": [2],
        "infected necrosis and antibiotics": [1, 3],
        "type of antibiotics": [4],
    },
    2: {
        "monitoring": [1],
        "fluid resuscitation": [2],
        "pain control": [3],
        "mechanical ventilation": [4],
        "increased intra-abdominal pressure": [5],
        "pharmacological treatment": [5],
        "enteral nutrition": [6],
    },
    3: {
        "indications for emergent ercp": [1],
        "indications for percutaneous/endoscopic drainage": [3],
        "indications for surgical intervention": [4],
        "timing of surgery": [5],
        "surgical strategy": [2, 5],
        "timing of cholecystectomy": [6],
    },
    4: {
        "open abdomen": [1],
        "open abdomen management and temporary abdominal closure": [2],
        "timing of dressing changes": [3],
        "timing for abdominal closure": [4],
    },
}


@dataclass(frozen=True)
class PdfBlock:
    page_idx: int
    index: int
    block_type: str
    text: str


class TableParser(HTMLParser):
    """Small stdlib HTML table parser for MinerU table snippets."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
            self._in_cell = True

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            self._row.append(clean_text(" ".join(self._cell)))
            self._cell = []
            self._in_cell = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []


def clean_text(text: str) -> str:
    """Normalize OCR text while preserving clinical abbreviations."""

    text = unescape(text or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    for joined, fixed in OCR_JOIN_FIXES.items():
        text = text.replace(joined, fixed)
    return text.strip()


def normalize_topic(topic: str) -> str:
    topic = clean_text(topic)
    topic = re.sub(r"^\d+\.\s*", "", topic)
    return topic.strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def block_to_text(block: dict[str, Any]) -> str:
    chunks: list[str] = []

    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            if span.get("content"):
                chunks.append(str(span["content"]))
            if span.get("html"):
                chunks.append(str(span["html"]))

    for child_key in ("blocks", "sub_blocks"):
        for child in block.get(child_key, []) or []:
            child_text = block_to_text(child)
            if child_text:
                chunks.append(child_text)

    return clean_text(" ".join(chunks))


def flatten_pdf_blocks(data: dict[str, Any]) -> list[PdfBlock]:
    blocks: list[PdfBlock] = []
    for page in data.get("pdf_info", []) or []:
        page_idx = int(page.get("page_idx", -1))
        for block in page.get("para_blocks", []) or []:
            text = block_to_text(block)
            if not text:
                continue
            blocks.append(
                PdfBlock(
                    page_idx=page_idx,
                    index=int(block.get("index", -1)),
                    block_type=str(block.get("type", "")),
                    text=text,
                )
            )
    return blocks


def table_rows_from_text(text: str) -> list[list[str]]:
    match = re.search(r"(<table.*?</table>)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    parser = TableParser()
    parser.feed(match.group(1))
    return parser.rows


def split_questions_from_plain_cell(text: str) -> list[str]:
    pattern = re.compile(
        r"(?=(?:Which|Are|Is|How|When|Should|Can|What)\b)",
    )
    questions = [clean_text(part) for part in pattern.split(text) if clean_text(part)]
    return [q for q in questions if q.endswith("?")]


def split_numbered_items(text: str) -> list[str]:
    text = clean_text(text)
    starts = list(re.finditer(r"(?:^|\s)(\d+)\.\s+(?=[A-Z])", text))
    if not starts:
        return [text] if text else []

    items: list[str] = []
    for idx, start in enumerate(starts):
        item_start = start.start(1)
        item_end = starts[idx + 1].start(1) if idx + 1 < len(starts) else len(text)
        item = clean_text(text[item_start:item_end])
        if item:
            items.append(item)
    return items


def extract_appendicitis_question_table(blocks: list[PdfBlock]) -> dict[str, dict[str, str]]:
    questions: dict[str, dict[str, str]] = {}
    for block in blocks:
        if "Table 1 Research topics and key questions" not in block.text:
            continue
        for row in table_rows_from_text(block.text)[1:]:
            if len(row) < 2:
                continue
            topic = normalize_topic(row[0])
            for match in re.finditer(
                r"(Q\.\d+\.\d+):\s*(.*?)(?=Q\.\d+\.\d+:|$)",
                row[1],
                flags=re.DOTALL,
            ):
                code = match.group(1)
                question = clean_text(match.group(2))
                questions[code] = {"question": question, "topic": topic}
        if questions:
            break
    return questions


def extract_appendicitis_recommendations(text: str) -> list[str]:
    text = clean_text(text)
    starts = list(
        re.finditer(r"\bRecommendation\s+(\d+(?:\.\d+){1,2})\.?\s+", text, flags=re.IGNORECASE)
    )
    recommendations: list[str] = []

    for idx, start in enumerate(starts):
        end_candidates = [len(text)]
        if idx + 1 < len(starts):
            end_candidates.append(starts[idx + 1].start())
        for stop_pattern in (r"\bStatement\s+\d+(?:\.\d+){1,2}", r"\bComment:", r"\bQ\.\d+\.\d+:"):
            stop = re.search(stop_pattern, text[start.end() :], flags=re.IGNORECASE)
            if stop:
                end_candidates.append(start.end() + stop.start())
        rec = clean_text(text[start.start() : min(end_candidates)])
        evidence_match = re.search(
            r"^(.*?\[[^\]]*(?:QoE|Strength of recommendation|Strength of Recommendation)[^\]]*\])",
            rec,
            flags=re.IGNORECASE,
        )
        if evidence_match:
            rec = clean_text(evidence_match.group(1))
        if rec:
            recommendations.append(rec)

    if not recommendations and "No recommendation" in text:
        statement = re.search(
            r"\bStatement\s+\d+(?:\.\d+){1,2}.*?No recommendation\]",
            text,
            flags=re.IGNORECASE,
        )
        if statement:
            recommendations.append(clean_text(statement.group(0)))

    return dedupe_preserve_order(recommendations)


def parse_appendicitis(data: dict[str, Any], created_at: str, logger: Any) -> list[dict[str, Any]]:
    blocks = flatten_pdf_blocks(data)
    question_meta = extract_appendicitis_question_table(blocks)
    logger.debug("Appendicitis table questions=%d", len(question_meta))

    grouped: dict[str, list[str]] = {}
    current_code: Optional[str] = None
    current_buffer: list[str] = []

    def flush_current() -> None:
        nonlocal current_code, current_buffer
        if current_code:
            recommendations = extract_appendicitis_recommendations(" ".join(current_buffer))
            if recommendations:
                grouped.setdefault(current_code, []).extend(recommendations)
                logger.debug(
                    "Appendicitis %s recommendations=%d",
                    current_code,
                    len(recommendations),
                )
        current_buffer = []

    for block in blocks:
        match = re.match(r"^(Q\.\d+\.\d+):\s*(.+)", block.text)
        if match:
            flush_current()
            current_code = match.group(1)
            if current_code not in question_meta:
                question_meta[current_code] = {
                    "question": clean_text(match.group(2)),
                    "topic": "Uncategorized",
                }
            continue
        if current_code:
            current_buffer.append(block.text)
    flush_current()

    records: list[dict[str, Any]] = []
    for code in sorted(question_meta, key=appendicitis_sort_key):
        answers = dedupe_preserve_order(grouped.get(code, []))
        if not answers:
            logger.debug("Appendicitis %s skipped because no recommendation was found.", code)
            continue
        meta = question_meta[code]
        records.append(
            make_record(
                disease_key="appendicitis",
                ordinal=len(records) + 1,
                question=meta["question"],
                answers=answers,
                topic=meta["topic"],
                created_at=created_at,
            )
        )
    return records


def appendicitis_sort_key(code: str) -> tuple[int, int]:
    match = re.search(r"Q\.(\d+)\.(\d+)", code)
    if not match:
        return (999, 999)
    return (int(match.group(1)), int(match.group(2)))


def parse_cholecystitis_question_table(blocks: list[PdfBlock]) -> dict[tuple[int, int], dict[str, str]]:
    questions: dict[tuple[int, int], dict[str, str]] = {}
    for block in blocks:
        if "Table 1 Sections/topics, key questions and key words" not in block.text:
            continue
        for row in table_rows_from_text(block.text)[1:]:
            if len(row) < 2:
                continue
            section_match = re.match(r"^(\d+)\.\s*(.+)", row[0])
            if not section_match:
                continue
            section_id = int(section_match.group(1))
            topic = normalize_topic(section_match.group(2))
            for ordinal, question in enumerate(split_questions_from_plain_cell(row[1]), start=1):
                questions[(section_id, ordinal)] = {"question": question, "topic": topic}
        if questions:
            break
    return questions


def extract_cholecystitis_recommendations(blocks: list[PdfBlock]) -> dict[str, str]:
    body = " ".join(block.text for block in blocks if block.page_idx <= 19)
    body = clean_text(body)
    starts = list(re.finditer(r"\b([1-7]\.\d+)\s+(?=[A-Z])", body))
    recommendations: dict[str, str] = {}

    for idx, start in enumerate(starts):
        num = start.group(1)
        segment_end = starts[idx + 1].start() if idx + 1 < len(starts) else len(body)
        segment = clean_text(body[start.start() : segment_end])
        initial_window = segment[:900]

        if not re.search(
            r"\b(recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|advised)\b|#QoE|SoR",
            initial_window,
            flags=re.IGNORECASE,
        ):
            continue
        if "References" in segment[:200]:
            continue

        qoe_match = re.search(r"#QoE[^#]*#\.?", segment, flags=re.IGNORECASE)
        if qoe_match:
            segment = clean_text(segment[: qoe_match.end()])
        else:
            segment = re.split(r"\bDiscussion\b", segment, maxsplit=1)[0]
            segment = clean_text(segment)

        if num not in recommendations and is_cholecystitis_recommendation(segment):
            recommendations[num] = segment

    return recommendations


def is_cholecystitis_recommendation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|advised|preferred|should be considered|cannot suggest)\b",
            text,
            flags=re.I,
        )
    )


def parse_cholecystitis(data: dict[str, Any], created_at: str, logger: Any) -> list[dict[str, Any]]:
    blocks = flatten_pdf_blocks(data)
    question_meta = parse_cholecystitis_question_table(blocks)
    recommendations = extract_cholecystitis_recommendations(blocks)
    logger.debug(
        "Cholecystitis table questions=%d recommendations=%d",
        len(question_meta),
        len(recommendations),
    )

    grouped: dict[tuple[int, int], list[str]] = {}
    for rec_num, recommendation in sorted(
        recommendations.items(),
        key=lambda item: numbered_code_sort_key(item[0]),
    ):
        section_id = int(rec_num.split(".")[0])
        ordinal = CHOLE_REC_TO_QUESTION_ORDINAL.get(rec_num)
        if not ordinal:
            logger.debug("No cholecystitis question mapping for recommendation %s", rec_num)
            continue
        question_key = (section_id, ordinal)
        grouped.setdefault(question_key, []).append(recommendation)

    records: list[dict[str, Any]] = []
    for question_key in sorted(question_meta):
        answers = dedupe_preserve_order(grouped.get(question_key, []))
        if not answers:
            logger.debug("Cholecystitis %s skipped because no recommendation was found.", question_key)
            continue
        meta = question_meta[question_key]
        records.append(
            make_record(
                disease_key="cholecystitis",
                ordinal=len(records) + 1,
                question=meta["question"],
                answers=answers,
                topic=meta["topic"],
                created_at=created_at,
            )
        )
    return records


def parse_panc_questions_from_text(text: str) -> list[str]:
    text = clean_text(text)
    items = split_numbered_items(text)
    return [re.sub(r"^\d+\.\s*", "", item).strip() for item in items if item.endswith("?")]


def pancreatitis_statement_key(title: str) -> str:
    title = clean_text(title).lower()
    title = re.sub(r"^statements?\s*\(", "", title)
    title = re.sub(r"\)$", "", title)
    return title


def is_questions_heading(text: str) -> bool:
    return bool(re.match(r"^Questions:?\s*$", text, flags=re.IGNORECASE))


def is_statement_heading(text: str) -> bool:
    return bool(re.match(r"^Statements?\s*\(", text, flags=re.IGNORECASE))


def parse_pancreatitis(data: dict[str, Any], created_at: str, logger: Any) -> list[dict[str, Any]]:
    blocks = flatten_pdf_blocks(data)
    questions_by_section: dict[int, list[str]] = {}
    answers_by_section_question: dict[tuple[int, int], list[str]] = {}
    section_titles = {
        0: "Diagnosis and risk stratification",
        1: "Antimicrobial therapy",
        2: "ICU and medical management",
        3: "Surgical and operative management",
        4: "Open abdomen",
    }

    section_idx = -1
    current_statement_key: Optional[str] = None
    current_statement_section: Optional[int] = None
    current_statement_buffer: list[str] = []

    def flush_statement() -> None:
        nonlocal current_statement_key, current_statement_section, current_statement_buffer
        if current_statement_key is None or current_statement_section is None:
            current_statement_buffer = []
            return
        question_ordinals = PANCREATITIS_STATEMENT_TO_QUESTION.get(current_statement_section, {}).get(
            current_statement_key,
            [],
        )
        answers = split_numbered_items(" ".join(current_statement_buffer))
        answers = [answer for answer in answers if answer and not answer.lower().startswith("discussion")]
        logger.debug(
            "Pancreatitis section=%s statement=%s question_ordinals=%s answers=%d",
            current_statement_section,
            current_statement_key,
            question_ordinals,
            len(answers),
        )
        for question_ordinal in question_ordinals:
            answers_by_section_question.setdefault((current_statement_section, question_ordinal), []).extend(answers)
        current_statement_key = None
        current_statement_section = None
        current_statement_buffer = []

    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        text = block.text
        if block.page_idx >= 15 and text.startswith("References"):
            break

        if is_questions_heading(text):
            flush_statement()
            section_idx += 1
            question_parts: list[str] = []
            idx += 1
            while idx < len(blocks):
                next_text = blocks[idx].text
                if is_statement_heading(next_text):
                    idx -= 1
                    break
                if is_questions_heading(next_text):
                    idx -= 1
                    break
                question_parts.append(next_text)
                idx += 1
            questions_by_section[section_idx] = parse_panc_questions_from_text(" ".join(question_parts))
            logger.debug(
                "Pancreatitis section=%d questions=%d",
                section_idx,
                len(questions_by_section[section_idx]),
            )
        elif is_statement_heading(text):
            flush_statement()
            current_statement_key = pancreatitis_statement_key(text)
            current_statement_section = section_idx
        elif current_statement_key is not None:
            if text.startswith("Discussion"):
                flush_statement()
            elif text.startswith("References") or text.startswith("Ready to submit"):
                flush_statement()
                break
            else:
                current_statement_buffer.append(text)
        idx += 1
    flush_statement()

    records: list[dict[str, Any]] = []
    for section_id in sorted(questions_by_section):
        for ordinal, question in enumerate(questions_by_section[section_id], start=1):
            answers = dedupe_preserve_order(answers_by_section_question.get((section_id, ordinal), []))
            if not answers:
                logger.debug(
                    "Pancreatitis section=%d question=%d skipped because no statement was mapped.",
                    section_id,
                    ordinal,
                )
                continue
            records.append(
                make_record(
                    disease_key="pancreatitis",
                    ordinal=len(records) + 1,
                    question=question,
                    answers=answers,
                    topic=section_titles.get(section_id, "Uncategorized"),
                    created_at=created_at,
                )
            )
    return records


def numbered_code_sort_key(code: str) -> tuple[int, int]:
    major, minor = code.split(".", 1)
    return int(major), int(minor)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = clean_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def make_record(
    disease_key: str,
    ordinal: int,
    question: str,
    answers: list[str],
    topic: str,
    created_at: str,
) -> dict[str, Any]:
    source = dict(SOURCE_META[disease_key])
    disease = source.pop("disease")
    return {
        "question_id": f"{disease_key}-{ordinal}",
        "question": clean_text(question),
        "answer": dedupe_preserve_order(answers),
        "disease": disease,
        "topic": clean_text(topic),
        "pico": dict(DEFAULT_PICO),
        "source": source,
        "synonyms": {},
        "created_at": created_at,
        "search_text": "",
    }


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_llm_messages(record: dict[str, Any]) -> list[Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You are a clinical guideline information extraction assistant. "
        "Return only valid JSON. Do not include markdown fences."
    )
    human_prompt = {
        "task": "Extract PICO from the clinical question and generate keyword synonyms.",
        "schema": {
            "pico": {
                "P": "string population/patient group",
                "I": "string intervention/exposure/index test",
                "C": ["list of comparator/control strings"],
                "O": {
                    "comparator/control string": [
                        "list of clinically relevant outcome strings for this comparator"
                    ]
                },
            },
            "synonyms": {
                "keyword from question": ["synonym 1", "synonym 2"]
            },
        },
        "constraints": [
            "P and I must be strings.",
            "C must be a list.",
            "O must be an object. Its keys should be selected from C when comparators exist.",
            "synonyms keys must be important keywords or phrases from the question.",
            "Use English medical terms because the source questions are in English.",
        ],
        "record": {
            "question_id": record["question_id"],
            "question": record["question"],
            "disease": record["disease"],
            "topic": record["topic"],
            "answer": record["answer"],
        },
    }
    return [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(human_prompt, ensure_ascii=False))]


def normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pico = payload.get("pico") if isinstance(payload, dict) else {}
    if not isinstance(pico, dict):
        pico = {}

    c_value = pico.get("C", [])
    if isinstance(c_value, str):
        c_value = [c_value] if c_value.strip() else []
    if not isinstance(c_value, list):
        c_value = []
    c_value = [str(item).strip() for item in c_value if str(item).strip()]

    o_value = pico.get("O", {})
    if not isinstance(o_value, dict):
        o_value = {}
    normalized_o: dict[str, Any] = {}
    for key, value in o_value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, list):
            normalized_o[key_text] = [str(item).strip() for item in value if str(item).strip()]
        elif value:
            normalized_o[key_text] = [str(value).strip()]

    synonyms = payload.get("synonyms", {}) if isinstance(payload, dict) else {}
    if not isinstance(synonyms, dict):
        synonyms = {}
    normalized_synonyms: dict[str, list[str]] = {}
    for key, value in synonyms.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = []
        normalized_values = [str(item).strip() for item in values if str(item).strip()]
        normalized_synonyms[key_text] = dedupe_preserve_order(normalized_values)

    return {
        "pico": {
            "P": str(pico.get("P", "") or "").strip(),
            "I": str(pico.get("I", "") or "").strip(),
            "C": c_value,
            "O": normalized_o,
        },
        "synonyms": normalized_synonyms,
    }


def parse_llm_json(content: str) -> dict[str, Any]:
    content = clean_text(content)
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM response does not contain JSON object: {content[:200]}")
    return json.loads(match.group(0))


def enrich_with_llm(
    records: list[dict[str, Any]],
    batch_size: int,
    max_concurrency: int,
    logger: Any,
) -> None:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing. Set it in .env or environment variables.")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        batch_records = records[start:end]
        messages = [build_llm_messages(record) for record in batch_records]
        logger.debug("Calling LLM batch start=%d end=%d", start, end)
        responses = llm.batch(messages, config={"max_concurrency": max_concurrency})

        for record, response in zip(batch_records, responses):
            try:
                payload = normalize_llm_payload(parse_llm_json(str(response.content)))
                record["pico"] = payload["pico"]
                record["synonyms"] = payload["synonyms"]
            except Exception as exc:  # noqa: BLE001 - keep batch processing resilient.
                logger.exception("Failed to parse LLM response for %s: %s", record["question_id"], exc)


def fill_search_text(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["search_text"] = "\n".join(
            [
                f"question: {record['question']}",
                f"disease: {record['disease']}",
                f"topic: {record['topic']}",
                f"pico: {json.dumps(record['pico'], ensure_ascii=False)}",
                f"synonyms: {json.dumps(record['synonyms'], ensure_ascii=False)}",
            ]
        )


def parse_all_guidelines(data_dir: Path, only_disease: Optional[str], created_at: str, logger: Any) -> list[dict[str, Any]]:
    parsers = {
        "appendicitis": parse_appendicitis,
        "pancreatitis": parse_pancreatitis,
        "cholecystitis": parse_cholecystitis,
    }
    records: list[dict[str, Any]] = []
    disease_keys = [only_disease] if only_disease else list(parsers)

    for disease_key in disease_keys:
        source_file = data_dir / f"{disease_key}.json"
        log_step(logger, f"阶段一：规则解析 {source_file.name}")
        data = load_json(source_file)
        disease_records = parsers[disease_key](data, created_at, logger)
        records.extend(disease_records)
        logger.debug("%s records=%d", disease_key, len(disease_records))

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIMIC-CPG clinical guideline QA knowledge base.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mimic-cpg"))
    parser.add_argument("--output-file", type=Path, default=Path("results/mimic_cpg_knowledge_base.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--only-disease",
        choices=["appendicitis", "pancreatitis", "cholecystitis"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("build_mimic_cpg_kb", log_file=args.log_file)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    log_step(logger, "启动：构建 MIMIC-CPG 问答知识库")
    load_env_file(args.env_file)
    logger.debug("Arguments: %s", vars(args))

    records = parse_all_guidelines(args.data_dir, args.only_disease, created_at, logger)
    logger.debug("Total stage-1 records=%d", len(records))

    if args.skip_llm:
        log_step(logger, "阶段二：已跳过 LLM，使用空 pico/synonyms 模板")
    else:
        log_step(logger, "阶段二：批量调用 LLM 抽取 pico 与 synonyms")
        enrich_with_llm(records, args.batch_size, args.max_concurrency, logger)

    log_step(logger, "阶段三：生成 search_text")
    fill_search_text(records)

    log_step(logger, f"写入输出：{args.output_file}")
    write_json(args.output_file, records)
    log_step(logger, f"完成：共生成 {len(records)} 条问答")


if __name__ == "__main__":
    main()
