import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
load_dotenv(".env")

from camel.types import TaskType, ModelType,ModelPlatformType
from utils.Evidence_Retrieval.pubmedretrieval import PubMedRetrieval
from utils.logger import setup_loggers, get_detail_logger, get_workflow_logger


def main(args):
    # 解析超参数
    YOUR_CONFIG_PATH = args.YOUR_CONFIG_PATH
    YOUR_QUESTION_DECOMPOSITION_PATH = args.YOUR_QUESTION_DECOMPOSITION_PATH
    YOUR_DATASET_PATH = args.YOUR_DATASET_PATH
    save_base = args.save_base
    disease = args.disease
    pico_idx = args.pico_idx
    additional_parameters = args.additional_parameters
    filters = args.filters
    use_agent = args.use_agent
    invalid_publication_types = args.invalid_publication_types

    # 配置日志
    setup_loggers(log_file=os.path.join(os.getenv("LOG_DIR"), YOUR_DATASET_PATH.split('/')[-1], "Literature_Search", f"{pico_idx}.log"))
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    # 读取模型配置
    config_path = os.path.join(YOUR_CONFIG_PATH)
    with open(config_path, 'r', encoding="utf-8") as file:
        config = json.load(file)
    model_config = config['model']["literature_search_model"]
    model_name = model_config['model_name']
    base_url = model_config['BASE_URL']
    api_key = model_config['API_KEY']

    # 读取指定的pico_idx对应的PICO信息
    question_deconstruction_datapath = os.path.join(
    YOUR_QUESTION_DECOMPOSITION_PATH, 'PICO_Information.json'
    )
    question_deconstruction_data = pd.read_json(
        question_deconstruction_datapath, dtype={'Index': str}
    )
    question_deconstruction_data = question_deconstruction_data[
        question_deconstruction_data['Index'] == pico_idx
    ]
    original_qd_dict = question_deconstruction_data.to_dict(orient='records')  # 结果为一个Python字典

    # 初始化PubMedRetrieval
    wf_logger.info(f"Initializing PubMedRetrieval for {disease} with pico_idx {pico_idx}")
    model_setting = {'search_term_formation': model_name,'search_strategy_formation':model_name}
    clinical_question = original_qd_dict[0]['Question']
    population = original_qd_dict[0]['P']
    intervention = original_qd_dict[0]['I']
    comparison = original_qd_dict[0]['C']
    save_path = os.path.join(save_base, model_name, 'use_agent_' + str(use_agent))
    pico_idx = original_qd_dict[0]['Index']
    retriever = PubMedRetrieval(
        disease=disease,
        clinical_question=clinical_question,
        population=population,
        intervention=intervention,
        comparison=comparison,
        api_key=api_key,
        base_url=base_url,
        model_setting=model_setting,
        use_agent=use_agent,
        save_path=save_path,
        pico_idx=pico_idx,
        filters=filters, 
        additional_parameters=additional_parameters
    )

    # 执行检索
    # 该步骤会输出两个文件：检索策略.txt和检索结果.json
    wf_logger.info(f"Executing PubMedRetrieval for {disease} with pico_idx {pico_idx}")
    retriever.run()
    dt_logger.info(retriever.search_terms)

    # 加载文献检索结果
    save_results_path = os.path.join(save_path, f'PICO{pico_idx}.json')
    with open(save_results_path, 'r', encoding="utf8") as file:
        search_results = json.load(file)

    # Heuristic screening：移除重复和没有摘要的文献记录
    # remove records without abstract
    wf_logger.info(f"Remove duplicate or non-abstract or invalid publication types records")
    dt_logger.info('Total records: '+str(len(search_results)))
    search_results = [record for record in search_results if record['Abstract'] != None]
    dt_logger.info('Records with abstract: '+str(len(search_results)))
    # deduplicate
    pmid_set = {d["Paper_Index"] for d in search_results}
    for r in search_results:
        if r["Paper_Index"] in pmid_set:
            pmid_set.remove(r["Paper_Index"])
        else:
            search_results.remove(r)
    dt_logger.info('Records after deduplication: '+str(len(search_results)))

    # 查看文献类型
    publication_type_set = {type for record in search_results for type in record['Publication Types']}
    dt_logger.info(f"Publication types: {list(publication_type_set)}")

    # 移除所有无效的文献类型
    for record in search_results:
        if any(pt in invalid_publication_types for pt in record['Publication Types']):
            search_results.remove(record)
    dt_logger.info('Records after removing invalid publication types: ' + str(len(search_results)))

    # 保存第一、第二阶段的数据——PICO、文献检索结果
    quicker_data = {
        "disease": disease,
        "clinical_question": clinical_question,
        'pico_idx': pico_idx,
        "population": population,
        "intervention": intervention,
        "comparison": comparison,
        "search_results": search_results,
    }

    quicker_data_path = os.path.join(YOUR_DATASET_PATH, f'quicker_data(PICO_IDX{pico_idx})_ls.json')
    with open(quicker_data_path, 'w', encoding="utf8") as file:
        json.dump(quicker_data, file, indent=4)
    dt_logger.info(f"Quicker data saved to {quicker_data_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Literature search")
    parser.add_argument("--YOUR_CONFIG_PATH", type=str, default='config/config.json', help="The config path")
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", type=str, default='data/2021ACR RA/Question_Decomposition', help="The question decomposition path")
    parser.add_argument("--YOUR_DATASET_PATH", type=str, default='data/2021ACR RA', help="The dataset path")
    parser.add_argument("--save_base", type=str, default="data/2021ACR RA/Literature_Search/pubmed/Results", help="search results folder")
    parser.add_argument("--disease", type=str, default='Rheumatoid Arthritis (RA)', help="The disease name")
    parser.add_argument("--pico_idx", type=str, default='dff23ac6', help="The PICO index of the question decomposition")
    parser.add_argument("--additional_parameters", type=dict, default={'datetype': 'pdat', 'mindate': '1946', 'maxdate': '2025/03/30'}, help="The additional parameters to PubMedRetrieval")
    parser.add_argument("--filters", type=dict, default={"Just search for RCT":'''<search results> AND ("Randomized controlled trial"[pt] OR "Controlled clinical trial"[pt] OR Randomized[tiab] OR Placebo[tiab] OR "Drug therapy"[sh] OR Randomly[tiab] OR Trial[tiab] OR Groups[tiab])''', 'No review': "<search results> NOT review[pt]"}, help="The search filters to use")
    parser.add_argument("--use_agent", type=bool, default=True, help="Whether to use Agentic method or not")
    parser.add_argument("--invalid_publication_types", type=list, default=['Comment', 'Editorial', 'Case Reports', 'News', 'Interview','Published Erratum','Observational Study','Autobiography','Address','Meta-Analysis','Retracted Publication'], help="The invalid publication types to remove")  # you can modify this list according to your needs
    args = parser.parse_args()
    main(args)
