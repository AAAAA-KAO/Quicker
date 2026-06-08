"""
Phase1：问题分解脚本。

运行示例：
    conda run -n quicker python Phase1-question_decomposition.py \
        --YOUR_CONFIG_PATH config/config.json

脚本功能：
    读取配置文件中的问题分解模型和临床问题，将临床问题拆解为 PICO
    （Population、Intervention、Comparison、Outcome）结构，并保存到
    PICO_Information.json。命令行参数会覆盖配置文件中的同名配置。

输入：
    1. --YOUR_CONFIG_PATH 指向的 JSON 配置文件。
    2. 临床问题、数据集名、输出目录等，来自命令行或配置文件。

输出：
    {YOUR_QUESTION_DECOMPOSITION_PATH}/PICO_Information.json

命令行参数：
    --YOUR_CONFIG_PATH：必填，项目配置文件路径。
    --YOUR_QUESTION_DECOMPOSITION_PATH：可选，Phase1 输出目录；未传入时读取
        config.pipeline.paths.question_decomposition。
    --dataset_name：可选，数据集名称；未传入时读取 config.pipeline.dataset_name。
    --clinical_question：可选，待分解的临床问题；未传入时读取
        config.pipeline.clinical_question。
    --method：可选，问题分解方法；未传入时读取
        config.pipeline.phase1_question_decomposition.method。
    --pico_idx：可选，PICO 编号；未传入或为 auto 时，根据临床问题和数据集名生成。
    --reuse_existing_pico / --no-reuse_existing_pico：可选，是否复用已存在的
        同编号 PICO；未传入时读取配置。
    --LOG_DIR：可选，日志目录；未传入时读取 config.logging.log_dir。
    --DOTENV_PATH：可选，.env 文件路径；未传入时不主动加载 .env。
"""

import argparse
import json
import os
from pathlib import Path

from utils.cli_config import (
    add_common_config_args,
    choose,
    config_path,
    get_nested,
    load_config,
    phase_config,
    prepare_environment,
    resolve_pico_idx,
)


def get_question_decomposition_output_model():
    from pydantic import BaseModel, Field

    class QuestionDecompositionOutput(BaseModel):
        P: list[str] = Field(description="The population of the question")
        I: list[str] = Field(description="The intervention of the question")
        C: list[str] = Field(description="The comparison of the question")
        O: dict[str, list[str]] = Field(description="The outcome of the question")

    return QuestionDecompositionOutput


def load_pico_list(pico_file_path: Path) -> list[dict]:
    if not pico_file_path.exists():
        return []
    with pico_file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{pico_file_path} must contain a JSON list.")
    return data


def resolve_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    phase_settings = phase_config(config, "phase1_question_decomposition")
    args.dataset_name = choose(
        args.dataset_name,
        get_nested(config, ("pipeline", "dataset_name")),
        "dataset_name/pipeline.dataset_name",
    )
    args.clinical_question = choose(
        args.clinical_question,
        get_nested(config, ("pipeline", "clinical_question")),
        "clinical_question/pipeline.clinical_question",
    )
    args.method = choose(
        args.method,
        phase_settings.get("method"),
        "method/pipeline.phase1_question_decomposition.method",
    )
    args.YOUR_QUESTION_DECOMPOSITION_PATH = choose(
        args.YOUR_QUESTION_DECOMPOSITION_PATH,
        config_path(config, "question_decomposition"),
        "YOUR_QUESTION_DECOMPOSITION_PATH/pipeline.paths.question_decomposition",
    )
    args.pico_idx = resolve_pico_idx(args.pico_idx, config)
    args.reuse_existing_pico = choose(
        args.reuse_existing_pico,
        phase_settings.get("reuse_existing_pico"),
        "reuse_existing_pico/pipeline.phase1_question_decomposition.reuse_existing_pico",
        required=False,
    )
    return args


def build_question_decomposition_model(config: dict):
    from langchain_openai import ChatOpenAI

    model_config = get_nested(config, ("model", "question_decomposition_model"), {})
    provider = model_config["provider"]
    if provider != "OpenAI":
        raise NotImplementedError(f"Provider {provider} is not implemented")

    model_kwargs = {
        "openai_api_key": model_config["API_KEY"],
        "base_url": model_config["BASE_URL"],
        "model": model_config["model_name"],
    }
    if model_config.get("temperature") is not None:
        model_kwargs["temperature"] = model_config["temperature"]
    output_model = get_question_decomposition_output_model()
    return ChatOpenAI(**model_kwargs).with_structured_output(
        output_model
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.YOUR_CONFIG_PATH)
    args = resolve_args(args, config)
    log_dir = prepare_environment(args, config)

    from utils.logger import get_detail_logger, get_workflow_logger, setup_loggers

    setup_loggers(
        log_file=os.path.join(
            log_dir,
            args.dataset_name,
            "Question_Decomposition",
            f"{args.pico_idx}.log",
        )
    )
    wf_logger = get_workflow_logger(__name__)
    dt_logger = get_detail_logger(__name__)

    if args.method != "zero-shot":
        raise NotImplementedError(
            "This script currently implements the zero-shot question decomposition path."
        )

    output_dir = Path(args.YOUR_QUESTION_DECOMPOSITION_PATH)
    pico_file_path = output_dir / "PICO_Information.json"
    pico_list = load_pico_list(pico_file_path)
    if args.reuse_existing_pico and any(
        str(item.get("Index")) == args.pico_idx for item in pico_list
    ):
        print(f"Existing PICO found and reused: {args.pico_idx}")
        return

    wf_logger.info("Start to decompose the question: %s", args.clinical_question)
    wf_logger.info("Use question decomposition method: %s", args.method)

    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from utils.PICO.prompt import get_zero_shot_pipeline_prompt

    qd_model = build_question_decomposition_model(config)
    pipeline_prompt = get_zero_shot_pipeline_prompt(args.dataset_name)
    local_zero_shot_chain = pipeline_prompt | RunnableParallel(
        generation_chain=qd_model,
        prompt_value=RunnablePassthrough(),
    )

    answer_dict = local_zero_shot_chain.invoke({"Question": args.clinical_question})
    dt_logger.info("The prompt is: %s", answer_dict["prompt_value"])

    generation = answer_dict["generation_chain"]
    pico_dict = {
        "Index": args.pico_idx,
        "Question": args.clinical_question,
        "P": generation.P,
        "I": generation.I,
        "C": generation.C,
        "O": generation.O,
    }
    dt_logger.info("PICO result: %s", pico_dict)

    pico_list.append(pico_dict)
    output_dir.mkdir(parents=True, exist_ok=True)
    with pico_file_path.open("w", encoding="utf-8") as file:
        json.dump(pico_list, file, indent=4, ensure_ascii=False)

    print(f"Saved PICO information to: {pico_file_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase1 question decomposition.")
    add_common_config_args(parser)
    parser.add_argument("--YOUR_QUESTION_DECOMPOSITION_PATH", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--clinical_question", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--pico_idx", default=None)
    parser.add_argument(
        "--reuse_existing_pico",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
