#!/usr/bin/env python3
"""
Filter out hybrid_results entries whose question_id matches the top-level question_id.

Reads results/retrieve/refine_questions.json, and for each top-level dictionary,
removes entries from its "hybrid_results" list where
record.question_id == top-level question_id.

Saves the filtered result to results/retrieve/refine_questions-no.json.
"""

import json
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = project_root / "results" / "retrieve" / "refine_questions.json"
    output_path = project_root / "results" / "retrieve" / "refine_questions-no.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_removed = 0

    for item in data:
        own_qid = item.get("question_id")
        hybrid_results = item.get("hybrid_results", [])

        if not own_qid or not hybrid_results:
            continue

        before = len(hybrid_results)
        item["hybrid_results"] = [
            hr
            for hr in hybrid_results
            if hr.get("record", {}).get("question_id") != own_qid
        ]
        total_removed += before - len(item["hybrid_results"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. Removed {total_removed} self-matching entries across {len(data)} items.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
