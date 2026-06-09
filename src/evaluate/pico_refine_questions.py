#!/usr/bin/env python3
"""Create PICO-perturbed evaluation questions with an LLM.

Input:
    data/evaluate/retrieve/original_questions.json

Output:
    data/evaluate/retrieve/pico_refine_questions.json

For each source record, this script randomly selects exactly one PICO element
from P, I, or C. When C is selected and has multiple elements, only one C
element is selected. The LLM then changes only the selected element to a
semantically different but type-valid element and writes a new English clinical
question from the modified PICO. If a C element changes, the corresponding O
key is moved to the new C string so O keys always match C elements.

Usage:
    conda run -n quicker python src/evaluate/pico_refine_questions.py

Optional:
    conda run -n quicker python src/evaluate/pico_refine_questions.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "evaluate" / "retrieve" / "original_questions.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluate" / "retrieve"
OUTPUT_PATH = OUTPUT_DIR / "pico_refine_questions.json"

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
SYSTEM_PROMPT = """You are a clinical evidence and PICO dataset generation expert.

Your job is to modify exactly one specified PICO element and generate a new English clinical question.

Rules:
1. The user will tell you which single element to modify: P, I, or one indexed element of C.
2. Modify ONLY that specified element.
3. The modified element must be semantically clearly different from the original, not just a synonym, spelling change, or word-order change.
4. Even after modification:
   - P must remain a patient group, population, or clinical population.
   - I must remain an intervention, exposure, diagnostic test, management strategy, or index approach.
   - C elements must remain comparators, controls, alternatives, or reference strategies.
5. Keep all O outcome lists unchanged.
6. If modifying C:
   - Replace only the selected C element.
   - Keep all other C elements unchanged and in the same order.
   - Move the selected element's O outcome list from the old C key to the new C key.
   - O keys must exactly match the final C list.
7. If modifying P or I, keep C and O exactly unchanged.
8. Generate a new clinical question in English that is faithful to the modified PICO, not the original PICO.
9. Return only valid JSON. Do not include markdown fences, explanations, or extra text.

Required JSON schema:
{
  "question": "new English clinical question based on the modified PICO",
  "pico": {
    "P": "modified or original population",
    "I": "modified or original intervention/exposure/test",
    "C": ["comparators/controls"],
    "O": {
      "each exact final C element": ["unchanged outcomes"]
    }
  },
  "modified_component": "P or I or C",
  "modified_c_index": null
}

When modified_component is C, modified_c_index must be the selected zero-based C index."""


@dataclass(frozen=True)
class MutationTarget:
    component: str
    c_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PICO-perturbed retrieval evaluation questions with DeepSeek through LangChain."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Input original_questions.json path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output pico_refine_questions.json path.")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for choosing P/I/C targets.")
    parser.add_argument("--retry", type=int, default=2, help="Retries per question after the first attempt.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds to sleep between successful requests.")
    return parser.parse_args()


def build_llm(model: str, api_key: str, base_url: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise TypeError(f"Input JSON top-level value must be a list, got {type(data).__name__}.")

    normalized_data: list[dict[str, Any]] = []
    for idx, entry in enumerate(data, start=1):
        normalized_data.append(normalize_source_entry(entry, idx))
    return normalized_data


def normalize_source_entry(entry: Any, idx: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError(f"Entry #{idx} must be a dict, got {type(entry).__name__}.")
    for key in ("question_id", "question", "pico"):
        if key not in entry:
            raise ValueError(f"Entry #{idx} is missing required key: {key}")

    question_id = entry["question_id"]
    question = entry["question"]
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError(f"Entry #{idx}.question_id must be a non-empty string.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Entry #{idx}.question must be a non-empty string.")

    pico = normalize_source_pico(entry["pico"], f"entry #{idx} source pico")
    validate_pico_shape(pico, f"entry #{idx} normalized source pico")
    return {
        "question_id": question_id.strip(),
        "question": question.strip(),
        "pico": pico,
    }


def normalize_source_pico(pico: Any, label: str) -> dict[str, Any]:
    if not isinstance(pico, dict):
        raise TypeError(f"{label} must be a dict, got {type(pico).__name__}.")
    if not isinstance(pico.get("P"), str) or not pico["P"].strip():
        raise ValueError(f"{label}.P must be a non-empty string.")
    if not isinstance(pico.get("I"), str) or not pico["I"].strip():
        raise ValueError(f"{label}.I must be a non-empty string.")
    if not isinstance(pico.get("C"), list):
        raise ValueError(f"{label}.C must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in pico["C"]):
        raise ValueError(f"{label}.C must contain only non-empty strings.")
    if not isinstance(pico.get("O"), dict):
        raise ValueError(f"{label}.O must be a dict.")

    c_values = [item.strip() for item in pico["C"]]
    if len(set(c_values)) != len(c_values):
        raise ValueError(f"{label}.C contains duplicate elements after trimming whitespace.")

    raw_outcomes = {str(key).strip(): value for key, value in pico["O"].items()}
    outcomes: dict[str, list[Any]] = {}
    for c_value in c_values:
        outcome_list = raw_outcomes.get(c_value, [])
        if not isinstance(outcome_list, list):
            raise ValueError(f"{label}.O[{c_value!r}] must be a list.")
        outcomes[c_value] = outcome_list

    return {
        "P": pico["P"].strip(),
        "I": pico["I"].strip(),
        "C": c_values,
        "O": outcomes,
    }


def validate_pico_shape(pico: Any, label: str) -> None:
    if not isinstance(pico, dict):
        raise TypeError(f"{label} must be a dict, got {type(pico).__name__}.")
    if not isinstance(pico.get("P"), str) or not pico["P"].strip():
        raise ValueError(f"{label}.P must be a non-empty string.")
    if not isinstance(pico.get("I"), str) or not pico["I"].strip():
        raise ValueError(f"{label}.I must be a non-empty string.")
    if not isinstance(pico.get("C"), list):
        raise ValueError(f"{label}.C must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in pico["C"]):
        raise ValueError(f"{label}.C must contain only non-empty strings.")
    if not isinstance(pico.get("O"), dict):
        raise ValueError(f"{label}.O must be a dict.")
    assert_o_keys_match_c(pico["C"], pico["O"], label)


def assert_o_keys_match_c(c_values: list[str], outcomes: dict[str, Any], label: str) -> None:
    if len(set(c_values)) != len(c_values):
        raise ValueError(f"{label}.C contains duplicate elements, so O keys cannot be matched unambiguously.")
    o_keys = list(outcomes.keys())
    if set(o_keys) != set(c_values) or len(o_keys) != len(c_values):
        raise ValueError(f"{label}.O keys must exactly match C elements.")
    for key, value in outcomes.items():
        if not isinstance(value, list):
            raise ValueError(f"{label}.O[{key!r}] must be a list.")


def clean_pico(pico: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pico, dict):
        raise TypeError(f"PICO must be a dict, got {type(pico).__name__}.")
    if not isinstance(pico.get("P"), str) or not pico["P"].strip():
        raise ValueError("PICO.P must be a non-empty string.")
    if not isinstance(pico.get("I"), str) or not pico["I"].strip():
        raise ValueError("PICO.I must be a non-empty string.")
    if not isinstance(pico.get("C"), list):
        raise ValueError("PICO.C must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in pico["C"]):
        raise ValueError("PICO.C must contain only non-empty strings.")
    if not isinstance(pico.get("O"), dict):
        raise ValueError("PICO.O must be a dict.")

    c_values = [item.strip() for item in pico["C"]]
    outcomes = {str(key).strip(): value for key, value in pico["O"].items()}
    return {
        "P": pico["P"].strip(),
        "I": pico["I"].strip(),
        "C": c_values,
        "O": outcomes,
    }


def choose_mutation_target(pico: dict[str, Any], rng: random.Random) -> MutationTarget:
    choices = ["P", "I"]
    if pico["C"]:
        choices.append("C")
    component = rng.choice(choices)
    if component == "C":
        return MutationTarget(component="C", c_index=rng.randrange(len(pico["C"])))
    return MutationTarget(component=component)


def build_human_payload(entry: dict[str, Any], target: MutationTarget) -> str:
    target_payload: dict[str, Any] = {
        "component_to_modify": target.component,
        "c_index_to_modify": target.c_index,
    }
    if target.component == "C" and target.c_index is not None:
        target_payload["c_value_to_modify"] = entry["pico"]["C"][target.c_index]

    payload = {
        "original_record": {
            "question_id": entry["question_id"],
            "question": entry["question"],
            "pico": entry["pico"],
        },
        "mutation_target": target_payload,
        "final_record_requirements": {
            "keep_question_id_out_of_response": True,
            "question_language": "English",
            "question_must_match_modified_pico": True,
            "do_not_change_non_target_pico_elements": True,
            "output_json_only": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def strip_markdown_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def parse_llm_json(content: str) -> dict[str, Any]:
    raw = strip_markdown_fences(content)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM response does not contain a JSON object: {content[:300]}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"LLM JSON response must be an object, got {type(payload).__name__}.")
    return payload


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def validate_modified_payload(
    payload: dict[str, Any],
    original_entry: dict[str, Any],
    target: MutationTarget,
) -> dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("LLM payload must contain a non-empty question string.")
    question = question.strip()
    if contains_cjk(question):
        raise ValueError("Generated question must be written in English and must not contain CJK text.")

    modified_pico = payload.get("pico")
    modified_pico = clean_pico(modified_pico)
    validate_pico_shape(modified_pico, "cleaned LLM payload pico")

    reported_component = payload.get("modified_component")
    if reported_component != target.component:
        raise ValueError(f"modified_component must be {target.component!r}, got {reported_component!r}.")
    reported_c_index = payload.get("modified_c_index")
    if target.component == "C":
        if reported_c_index != target.c_index:
            raise ValueError(f"modified_c_index must be {target.c_index}, got {reported_c_index!r}.")
    elif reported_c_index is not None:
        raise ValueError("modified_c_index must be null unless modified_component is C.")

    original_pico = original_entry["pico"]
    assert_single_target_changed(original_pico, modified_pico, target)

    return {
        "question_id": original_entry["question_id"],
        "question": question,
        "pico": {
            "P": modified_pico["P"],
            "I": modified_pico["I"],
            "C": modified_pico["C"],
            "O": modified_pico["O"],
        },
    }


def assert_single_target_changed(
    original_pico: dict[str, Any],
    modified_pico: dict[str, Any],
    target: MutationTarget,
) -> None:
    if target.component == "P":
        if normalize_text(modified_pico["P"]) == normalize_text(original_pico["P"]):
            raise ValueError("P was selected but was not meaningfully changed.")
        if modified_pico["I"] != original_pico["I"] or modified_pico["C"] != original_pico["C"]:
            raise ValueError("Only P may change; I and C must remain unchanged.")
        if modified_pico["O"] != original_pico["O"]:
            raise ValueError("O must remain unchanged when P changes.")
        return

    if target.component == "I":
        if normalize_text(modified_pico["I"]) == normalize_text(original_pico["I"]):
            raise ValueError("I was selected but was not meaningfully changed.")
        if modified_pico["P"] != original_pico["P"] or modified_pico["C"] != original_pico["C"]:
            raise ValueError("Only I may change; P and C must remain unchanged.")
        if modified_pico["O"] != original_pico["O"]:
            raise ValueError("O must remain unchanged when I changes.")
        return

    if target.component != "C" or target.c_index is None:
        raise ValueError(f"Unsupported mutation target: {target}")

    old_c = original_pico["C"]
    new_c = modified_pico["C"]
    if modified_pico["P"] != original_pico["P"] or modified_pico["I"] != original_pico["I"]:
        raise ValueError("Only C may change; P and I must remain unchanged.")
    if len(new_c) != len(old_c):
        raise ValueError("C length must remain unchanged when one C element is modified.")

    changed_indexes = [
        idx
        for idx, (old_value, new_value) in enumerate(zip(old_c, new_c, strict=True))
        if normalize_text(old_value) != normalize_text(new_value)
    ]
    if changed_indexes != [target.c_index]:
        raise ValueError(f"Exactly C[{target.c_index}] must change; changed indexes were {changed_indexes}.")

    old_key = old_c[target.c_index]
    new_key = new_c[target.c_index]
    if normalize_text(old_key) == normalize_text(new_key):
        raise ValueError("Selected C element was not meaningfully changed.")
    if set(modified_pico["O"].keys()) != set(new_c):
        raise ValueError("O keys must exactly match modified C elements.")

    for idx, c_value in enumerate(old_c):
        if idx == target.c_index:
            continue
        if modified_pico["O"].get(c_value) != original_pico["O"].get(c_value):
            raise ValueError(f"Outcome list for unchanged C element {c_value!r} must remain unchanged.")
    if modified_pico["O"].get(new_key) != original_pico["O"].get(old_key):
        raise ValueError("Outcome list for the modified C element must be moved unchanged to the new C key.")


def refine_single(
    llm: ChatOpenAI,
    entry: dict[str, Any],
    target: MutationTarget,
    retry: int,
) -> dict[str, Any]:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_human_payload(entry, target)),
    ]

    last_error: Exception | None = None
    for attempt in range(1, retry + 2):
        try:
            response = llm.invoke(messages)
            payload = parse_llm_json(str(response.content))
            return validate_modified_payload(payload, entry, target)
        except Exception as exc:
            last_error = exc
            if attempt <= retry:
                wait = 2 ** attempt
                print(f"  [RETRY] attempt {attempt} failed; retrying in {wait}s: {exc}")
                time.sleep(wait)
            else:
                print(f"  [FAIL] failed after {retry + 1} attempts: {exc}")

    raise last_error  # type: ignore[misc]


def refine_all(
    llm: ChatOpenAI,
    original_data: list[dict[str, Any]],
    rng: random.Random,
    retry: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    refined: list[dict[str, Any]] = []
    total = len(original_data)

    for idx, entry in enumerate(original_data, start=1):
        qid = entry["question_id"]
        target = choose_mutation_target(entry["pico"], rng)
        target_label = target.component if target.component != "C" else f"C[{target.c_index}]"

        print(f"\n[{idx}/{total}] question_id={qid} target={target_label}")
        result = refine_single(llm=llm, entry=entry, target=target, retry=retry)
        refined.append(result)
        print(f"  [OK] {result['question'][:140]}{'...' if len(result['question']) > 140 else ''}")

        if idx < total and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return refined


def save_output(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote {len(data)} records to: {path}")


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 72)
    print("PICO perturbation question generation")
    print(f"Model:  {args.model}")
    print(f"API key: {mask_secret(args.api_key)}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Seed:   {args.seed}")
    print("=" * 72)

    original_data = load_json_list(args.input)
    llm = build_llm(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    refined_data = refine_all(
        llm=llm,
        original_data=original_data,
        rng=rng,
        retry=args.retry,
        sleep_seconds=args.sleep,
    )
    save_output(refined_data, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
