import os
import re
from typing import List

import pandas as pd


def find_existing_screening_results(
    method: str,
    pico_idx: str,
    study_selection_base_path: str,
    exp_num: int,
) -> List[str]:
    '''
    Find the latest complete batch of saved record-screening results.

    A complete batch contains exp_0 through exp_{exp_num - 1} files with the same
    timestamp, matching the naming convention used by the screening functions.
    '''
    if exp_num <= 0:
        return []

    screening_results_dir = os.path.join(
        study_selection_base_path,
        'Results',
        'screening_records',
        method,
        pico_idx,
    )
    if not os.path.isdir(screening_results_dir):
        return []

    filename_pattern = re.compile(
        rf'^{re.escape(pico_idx)}_exp_(\d+)-(.+)\.csv$'
    )
    expected_exp_indices = set(range(exp_num))
    batches = {}

    for filename in os.listdir(screening_results_dir):
        match = filename_pattern.match(filename)
        if not match:
            continue

        exp_idx = int(match.group(1))
        timestamp = match.group(2)
        if exp_idx not in expected_exp_indices:
            continue

        batches.setdefault(timestamp, {})[exp_idx] = os.path.join(
            screening_results_dir, filename
        )

    complete_batches = {
        timestamp: batch
        for timestamp, batch in batches.items()
        if expected_exp_indices.issubset(batch)
    }
    if not complete_batches:
        return []

    latest_timestamp = max(complete_batches)
    latest_batch = complete_batches[latest_timestamp]
    return [latest_batch[i] for i in range(exp_num)]


def screen_records(
    method: str,
    search_results: pd.DataFrame,
    pico_idx: str,
    study_selection_base_path: str,
    disease: str,
    model,
    exp_num: int,
    clinical_question_with_pico: str,
    no_skip_set=None,
    return_no_skip_set=False,
):
    if method == 'basic':
        from utils.Study_Selection.simple import (
            screening_records_using_basic_prompt,
        )

        return screening_records_using_basic_prompt(
            search_results=search_results,
            pico_idx=pico_idx,
            study_selection_base_path=study_selection_base_path,
            disease=disease,
            model=model,
            exp_num=exp_num,
            clinical_question_with_pico=clinical_question_with_pico,
            no_skip_set=no_skip_set,
            return_no_skip_set=return_no_skip_set,
        )
    elif method == 'cot':
        from utils.Study_Selection.cot import screening_records_using_cot

        return screening_records_using_cot(
            search_results=search_results,
            pico_idx=pico_idx,
            study_selection_base_path=study_selection_base_path,
            disease=disease,
            model=model,
            exp_num=exp_num,
            clinical_question_with_pico=clinical_question_with_pico,
        )
    else:
        raise ValueError('Invalid method')
