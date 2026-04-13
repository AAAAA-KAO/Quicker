from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import json
import os
import hashlib
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(".env")

from utils.PICO.prompt import get_zero_shot_pipeline_prompt,get_few_shot_pipeline_prompt
from utils.PICO.few_shot import match_few_shot, create_example_selector
from utils.PICO.pfe import generate_experience,generate_answer,combine_examples_with_experience
from utils.PICO.base import create_dataset
from utils.logger import setup_loggers, get_workflow_logger, get_detail_logger


# Hyperparameters
YOUR_CONFIG_PATH = 'config/config.json'  # your config path
YOUR_QUESTION_DECOMPOSITION_PATH =  'data/2021ACR RA/Question_Decomposition' # e.g. 'data/2021ACR RA/Question_Decomposition'

method = 'zero-shot'
dataset_name =  '2021ACR RA' # Choose from: 2021ACR RA, 2020EAN Dementia, 2024KDIGO CKD. If you want to use your own dataset, please modify the code accordingly.
clinical_question = "Should patients with RA on DMARDs who are in low disease activity gradually taper off DMARDs, abruptly withdraw DMARDs, or continue DMARDS at the same doses?" # Example: "In RA patients who have achieved sustained remission for over one year with DMARD monotherapy, is drug discontinuation advisable?"


# Equipped the llm with structured output
class QuestionDecompositionOutput(BaseModel):
    P: list[str] = Field(description="The population of the study")
    I: list[str] = Field(description="The intervention of the study")
    C: list[str] = Field(description="The comparison of the study")
    O: list[str] = Field(description="The outcome of the study")


def main():
    question_index = hashlib.sha256((clinical_question).encode('utf-8')).hexdigest()[:8]
    setup_loggers(log_file=os.path.join(os.getenv("LOG_DIR"), dataset_name, "Question_Decomposition", f"{question_index}.log"))
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

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
    qd_model = ChatOpenAI(openai_api_key=api_key, base_url=api_base_URL, model=model_name)
    qd_model = qd_model.with_structured_output(QuestionDecompositionOutput)

    # For zero-shot, you can use the following code:
    pipeline_prompt = get_zero_shot_pipeline_prompt(dataset_name)
    # output_parser = StrOutputParser()
    later_zero_shot_exp = qd_model
    local_zero_shot_chain = pipeline_prompt | RunnableParallel(
        generation_chain=later_zero_shot_exp, prompt_value=RunnablePassthrough()
    )

    wf_logger.info(f"【2/3】Run the {method} chain")
    # Change the chain if you use the other two methods
    answer_dict = local_zero_shot_chain.invoke({"Question": clinical_question})
    dt_logger.info(f"The prompt is: {answer_dict['prompt_value']}")
    dt_logger.info(f"The output PICO is: {answer_dict['generation_chain']}")

    
    # Extract the PICO from the answer_dict
    population = answer_dict['generation_chain'].P
    intervention = answer_dict['generation_chain'].I
    comparison = answer_dict['generation_chain'].C
    outcome = answer_dict['generation_chain'].O

    # Read pico_file
    pico_file_path = os.path.join(YOUR_QUESTION_DECOMPOSITION_PATH, 'PICO_Information.json')
    wf_logger.info(f"【3/3】Store the PICO to {pico_file_path}")
    if os.path.exists(pico_file_path):
        with open(pico_file_path, 'r', encoding="utf8") as file:
            pico_list = json.load(file)
    else:
        pico_list = []

    # Add a new question to the pico_file
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

    # Save to pico information file
    with open(os.path.join(YOUR_QUESTION_DECOMPOSITION_PATH,'PICO_Information.json'), 'w', encoding="utf8") as file:
        json.dump(pico_list, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
