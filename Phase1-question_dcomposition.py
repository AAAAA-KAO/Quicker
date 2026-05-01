import json
import os
import hashlib
import argparse
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
load_dotenv(".env")

from utils.PICO.base import create_dataset
from utils.PICO.few_shot import match_few_shot, create_example_selector
from utils.logger import setup_loggers, get_workflow_logger, get_detail_logger
from utils.PICO.prompt import get_zero_shot_pipeline_prompt,get_few_shot_pipeline_prompt
from utils.PICO.pfe import generate_experience,generate_answer,combine_examples_with_experience


# Equipped the llm with structured output
class QuestionDecompositionOutput(BaseModel):
    P: list[str] = Field(description="The population of the question")
    I: list[str] = Field(description="The intervention of the question")
    C: list[str] = Field(description="The comparison of the question")
    O: list[str] = Field(description="The outcome of the question")


def main(args):
    # 解析超参数
    YOUR_CONFIG_PATH = args.YOUR_CONFIG_PATH
    YOUR_QUESTION_DECOMPOSITION_PATH = args.YOUR_QUESTION_DECOMPOSITION_PATH
    method = args.method
    dataset_name = args.dataset_name
    clinical_question = args.clinical_question

    # 配置额外参数
    question_index = hashlib.sha256((clinical_question + dataset_name).encode('utf-8')).hexdigest()[:8]

    # 配置日志
    setup_loggers(log_file=os.path.join(os.getenv("LOG_DIR"), dataset_name, "Question_Decomposition", f"{question_index}.log"))
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    # --------------------任务执行---------------------
    # 读取模型配置
    wf_logger.info(f"Start to decompose the question: {clinical_question}")
    wf_logger.info(f"【1/3】Config the model and prompt, Now you are using the {method} method")
    config_path = os.path.join(YOUR_CONFIG_PATH)
    with open(config_path, 'r', encoding="utf8") as file:
        config = json.load(file)
    model_config = config['model']
    provider = model_config[f'question_decomposition_model'].get('provider', 'OpenAI')
    model_name = model_config[f'question_decomposition_model']['model_name']
    api_key = model_config[f'question_decomposition_model']['API_KEY']
    api_base_URL = model_config[f'question_decomposition_model']['BASE_URL']

    # 创建模型
    qd_model = ChatOpenAI(openai_api_key=api_key, base_url=api_base_URL, model=model_name)
    qd_model = qd_model.with_structured_output(QuestionDecompositionOutput)

    # zero-shot：创建prompt
    pipeline_prompt = get_zero_shot_pipeline_prompt(dataset_name)
    
    # 构建langchain链
    later_zero_shot_exp = qd_model
    local_zero_shot_chain = pipeline_prompt | RunnableParallel(
        generation_chain=later_zero_shot_exp, prompt_value=RunnablePassthrough()
    )

    # 运行langchain链
    wf_logger.info(f"【2/3】Run the {method} chain")
    # Change the chain if you use the other two methods
    answer_dict = local_zero_shot_chain.invoke({"Question": clinical_question})
    dt_logger.info(f"The prompt is: {answer_dict['prompt_value']}")

    # 提取PICO元素
    population = answer_dict['generation_chain'].P
    intervention = answer_dict['generation_chain'].I
    comparison = answer_dict['generation_chain'].C
    outcome = answer_dict['generation_chain'].O
    dt_logger.info(f"Population: {population}")
    dt_logger.info(f"Intervention: {intervention}")
    dt_logger.info(f"Comparison: {comparison}")
    dt_logger.info(f"Outcome: {outcome}")

    # 读取已有的PICO
    pico_file_path = os.path.join(YOUR_QUESTION_DECOMPOSITION_PATH, 'PICO_Information.json')
    wf_logger.info(f"【3/3】Store the PICO to {pico_file_path}")
    if os.path.exists(pico_file_path):
        with open(pico_file_path, 'r', encoding="utf8") as file:
            pico_list = json.load(file)
    else:
        pico_list = []

    # 将本次新增的PICO添加到已有的PICO列表
    pico_dict = {}
    pico_dict['Index'] = question_index
    pico_dict['Question'] = clinical_question
    pico_dict['P'] = population
    pico_dict['I'] = intervention
    pico_dict['C'] = comparison
    pico_dict['O'] = outcome
    pico_list.append(pico_dict)

    if not os.path.exists(YOUR_QUESTION_DECOMPOSITION_PATH):
        os.makedirs(YOUR_QUESTION_DECOMPOSITION_PATH)

    # 保存更新后的PICO列表
    with open(os.path.join(YOUR_QUESTION_DECOMPOSITION_PATH,'PICO_Information.json'), 'w', encoding="utf8") as file:
        json.dump(pico_list, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decompose a question into PICO components")
    parser.add_argument("--YOUR_CONFIG_PATH", type=str, default='config/config.json', help="The config path")
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", type=str, default='data/2021ACR RA/Question_Decomposition', help="The question decomposition path")
    parser.add_argument("--method", type=str, default='zero-shot', help="The method to use")
    parser.add_argument("--dataset_name", type=str, default='2021ACR RA', help="The dataset name")
    parser.add_argument("--clinical_question", type=str, default="Should patients with RA on DMARDs who are in low disease activity gradually taper off DMARDs, abruptly withdraw DMARDs, or continue DMARDS at the same doses?", help="The clinical question to decompose")
    args = parser.parse_args()

    main(args)
