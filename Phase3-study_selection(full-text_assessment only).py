import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
load_dotenv(".env")

from camel.types import TaskType, ModelType, ModelPlatformType
from utils.Study_Selection.record_screening import screen_records
from utils.General.quicker import Quicker, QuickerData, QuickerStage
from utils.Study_Selection.base import get_clinical_question_with_pico
from utils.logger import setup_loggers, get_detail_logger, get_workflow_logger


def main(args):
    # 解析超参数
    YOUR_CONFIG_PATH = args.YOUR_CONFIG_PATH
    YOUR_DATASET_PATH = args.YOUR_DATASET_PATH
    YOUR_QUESTION_DECOMPOSITION_PATH = args.YOUR_QUESTION_DECOMPOSITION_PATH
    YOUR_EVIDENCE_ASSESSMENT_PATH = args.YOUR_EVIDENCE_ASSESSMENT_PATH
    YOUR_LITERATURE_SEARCH_PATH = args.YOUR_LITERATURE_SEARCH_PATH
    YOUR_STUDY_SELECTION_PATH = args.YOUR_STUDY_SELECTION_PATH
    YOUR_PAPER_LIBRARY_PATH = args.YOUR_PAPER_LIBRARY_PATH
    disease = args.disease
    pico_idx = args.pico_idx
    record_screening_method = args.record_screening_method
    exp_num = args.exp_num
    t = args.t

    # 配置日志
    setup_loggers(log_file=os.path.join(os.getenv("LOG_DIR"), YOUR_DATASET_PATH.split('/')[-1], "Study_Selection", f"{pico_idx}.log"))
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    # 读取模型配置
    config_path = os.path.join(YOUR_CONFIG_PATH)
    with open(config_path, 'r', encoding="utf-8") as file:
        config = json.load(file)
    model_config = config['model']["study_selection_model"]
    model_name = model_config['model_name']
    base_url = model_config['BASE_URL']
    api_key = model_config['API_KEY']

    # 初始化模型
    wf_logger.info(f"Initializing study selection model: {model_name}")
    llm = ChatOpenAI(
        openai_api_key=api_key,
        base_url=base_url,
        model=model_name,
    )

    # 读取文献检索结果
    quickerdata_ls_path = os.path.join(YOUR_DATASET_PATH, f'quicker_data(PICO_IDX{pico_idx})_ls.json')
    with open(quickerdata_ls_path, 'r', encoding="utf-8") as f:
        quickerdata_ls = json.load(f)

    # 初始化 QuickerData 和 Quicker
    wf_logger.info(f"Initializing Quicker for {disease} with pico_idx {pico_idx}")
    quicker_data = QuickerData(disease=disease, pico_idx=pico_idx)
    quicker = Quicker(
        config_path=YOUR_CONFIG_PATH,
        question_deconstruction_database_path=YOUR_QUESTION_DECOMPOSITION_PATH,
        literature_search_database_path=YOUR_LITERATURE_SEARCH_PATH,
        study_selection_database_path=YOUR_STUDY_SELECTION_PATH,
        evidence_assessment_database_path=None,  # 不需要
        quicker_data=quicker_data,
        paper_library_base=YOUR_PAPER_LIBRARY_PATH,
    )

    # 准备数据
    data_dict = dict(
        clinical_question=quickerdata_ls['clinical_question'],
        population=quickerdata_ls['population'],
        intervention=quickerdata_ls['intervention'],
        comparison=quickerdata_ls['comparison'],
        outcome=quickerdata_ls['outcome'],
        study=['randomized clinical trial'],
        search_results=quickerdata_ls['search_results'],
    )

    # 添加数据到 quicker_data
    quicker._add_data_to_quickerdata_for_test(
        stage=QuickerStage.LITERATURE_SEARCH,
        default_value=data_dict,
    )

    # 设置纳入排除标准（可选）
    quicker.set_inclusion_exclusion_criteria(
        inclusion_criteria=args.inclusion_criteria,
        exclusion_criteria=args.exclusion_criteria
    )

    wf_logger.info("Run study selection")
    
    # 加载record_included_list
    screening_results_save_path_list = []
    for i in range(config['study_selection']['exp_num']):
        record_included_list_path = os.path.join(
            YOUR_STUDY_SELECTION_PATH,
            f'Results/screening_records/{record_screening_method}/{pico_idx}/{pico_idx}_exp_{i}-{t}.csv'
        )
        screening_results_save_path_list.append(record_included_list_path)
    
    record_included_list = quicker.get_full_text_assessment_paper_list(
        screening_results_save_path_list=screening_results_save_path_list,
        threshold=config['study_selection']['threshold'],
    )
    
    record_included_list, full_text_included_list, total_outcome_list = quicker.select_studies_by_full_text_assessment(record_included_list)

    quicker.quicker_data.update_data(
        dict(
            record_included_studies=record_included_list,
            full_text_included_studies=full_text_included_list,
            total_outcome_list=total_outcome_list,
        )
    )

    # 保存结果
    print(record_included_list)
    print(full_text_included_list)
    print(total_outcome_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Study Selection - Screen records for systematic review")
    parser.add_argument("--YOUR_CONFIG_PATH", type=str, default='config/config.json', help="The config path")
    parser.add_argument("--YOUR_DATASET_PATH", type=str, default='data/2021ACR RA', help="The dataset path")
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", type=str, default='data/2021ACR RA/Question_Decomposition', help="The question decomposition path")
    parser.add_argument("--YOUR_LITERATURE_SEARCH_PATH", type=str, default='data/2021ACR RA/Literature_Search', help="The literature search path")
    parser.add_argument("--YOUR_STUDY_SELECTION_PATH", type=str, default='data/2021ACR RA/Study_Selection', help="The study selection path")
    parser.add_argument("--YOUR_EVIDENCE_ASSESSMENT_PATH", type=str, default="data/2021ACR RA/Evidence_Assessment", help="your evidence assessment folder")
    parser.add_argument("--YOUR_PAPER_LIBRARY_PATH", type=str, default='data/2021ACR RA/Paper_Library', help="The paper library path")
    parser.add_argument("--disease", type=str, default='Rheumatoid Arthritis (RA)', help="The disease name")
    parser.add_argument("--pico_idx", type=str, default='dff23ac6', help="The PICO index of the question decomposition")
    parser.add_argument("--record_screening_method", type=str, default='basic', choices=['basic', 'cot'], help="The record screening method to use")
    parser.add_argument("--exp_num", type=int, default=10, help="The number of examples to use for few-shot learning")
    parser.add_argument("--inclusion_criteria", type=str, default='', help="The inclusion criteria for study selection")
    parser.add_argument("--exclusion_criteria", type=str, default='', help="The exclusion criteria for study selection")
    parser.add_argument("--t", type=str, default='2026-05-09-18-10-42', help="time stamp")
    args = parser.parse_args()
    main(args)