#!/usr/bin/env python3
"""
Evaluate hybrid retrieval results from refine_questions.json.

Computes four metrics per item and aggregates across the full dataset:
  - Hit@1:   proportion where the correct question_id is the top-ranked candidate
  - Recall@3: proportion where the correct question_id appears in the top-3 candidates
  - Recall@5: proportion where the correct question_id appears in the top-5 candidates
  - MRR:     Mean Reciprocal Rank — average of 1/rank across all items
             (rank is 1-indexed; 0 contribution if the correct answer is absent)

Input:  results/retrieve/refine_questions.json
Output: results/evaluate/hybrid_retrieval_eval.json
"""

import json
import os
from datetime import datetime, timezone


def load_data(input_path: str) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(items: list[dict]) -> dict:
    """Compute Hit@1, Recall@3, Recall@5, and MRR across all items."""
    total = len(items)
    hit_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks: list[float] = []

    for item in items:
        target_qid = item["question_id"]
        candidates = item["hybrid_results"]  # list of {index, score, record}

        # Gather the ordered list of question_ids from hybrid_results
        candidate_qids = [c["record"]["question_id"] for c in candidates]

        # Determine the 1-based rank of the target (0 if not found)
        try:
            rank = candidate_qids.index(target_qid) + 1  # 1-indexed
        except ValueError:
            rank = 0

        # Hit@1
        if rank == 1:
            hit_at_1 += 1

        # Recall@3
        if 1 <= rank <= 3:
            recall_at_3 += 1

        # Recall@5
        if 1 <= rank <= 5:
            recall_at_5 += 1

        # MRR contribution
        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    return {
        "num_items": total,
        "hit_at_1": round(hit_at_1 / total, 6) if total > 0 else 0.0,
        "hit_at_1_count": hit_at_1,
        "recall_at_3": round(recall_at_3 / total, 6) if total > 0 else 0.0,
        "recall_at_3_count": recall_at_3,
        "recall_at_5": round(recall_at_5 / total, 6) if total > 0 else 0.0,
        "recall_at_5_count": recall_at_5,
        "mrr": round(mrr, 6),
    }


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    input_path = os.path.join(
        project_root, "results", "retrieve", "refine_questions.json"
    )
    output_dir = os.path.join(project_root, "results", "evaluate")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hybrid_retrieval_eval.json")

    items = load_data(input_path)
    metrics = compute_metrics(items)

    result = {
        "metadata": {
            "input_file": "results/retrieve/refine_questions.json",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Evaluation complete.")
    print(f"  Items evaluated: {metrics['num_items']}")
    print(f"  Hit@1:    {metrics['hit_at_1']:.4f}  ({metrics['hit_at_1_count']}/{metrics['num_items']})")
    print(f"  Recall@3: {metrics['recall_at_3']:.4f}  ({metrics['recall_at_3_count']}/{metrics['num_items']})")
    print(f"  Recall@5: {metrics['recall_at_5']:.4f}  ({metrics['recall_at_5_count']}/{metrics['num_items']})")
    print(f"  MRR:      {metrics['mrr']:.4f}")
    print(f"  Output -> {output_path}")


if __name__ == "__main__":
    main()
