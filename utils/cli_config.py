import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(path: str | Path, data: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    return output_path


def get_nested(data: dict, keys: Iterable[str], default: Any = None) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def require_value(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"Missing required argument or config value: {name}")
    return value


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--YOUR_CONFIG_PATH",
        required=True,
        help="项目配置文件路径，例如 config/config.json。",
    )
    parser.add_argument(
        "--LOG_DIR",
        default=None,
        help="日志目录；未传入时读取 config.logging.log_dir。",
    )
    parser.add_argument(
        "--DOTENV_PATH",
        default=None,
        help="可选 .env 文件路径；未传入时不主动加载 .env。",
    )


def prepare_environment(args: argparse.Namespace, config: dict) -> str:
    dotenv_path = getattr(args, "DOTENV_PATH", None)
    if dotenv_path:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)

    log_dir = getattr(args, "LOG_DIR", None) or get_nested(
        config, ("logging", "log_dir")
    )
    require_value(log_dir, "LOG_DIR/config.logging.log_dir")
    os.environ["LOG_DIR"] = str(log_dir)

    runtime_environment = get_nested(
        config, ("pipeline", "runtime_environment"), {}
    )
    if isinstance(runtime_environment, dict):
        for key, value in runtime_environment.items():
            os.environ[str(key)] = str(value)

    return str(log_dir)


def config_path(config: dict, key: str) -> Any:
    return get_nested(config, ("pipeline", "paths", key))


def phase_config(config: dict, phase_name: str) -> dict:
    return get_nested(config, ("pipeline", phase_name), {}) or {}


def choose(cli_value: Any, config_value: Any, name: str, required: bool = True) -> Any:
    value = cli_value if cli_value is not None else config_value
    if required:
        return require_value(value, name)
    return value


def parse_json_arg(value: str | None) -> Any:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    candidate_path = Path(value)
    if candidate_path.exists():
        return load_json_file(candidate_path)
    return json.loads(value)


def choose_json(
    cli_value: str | None,
    config_value: Any,
    name: str,
    expected_type: type | tuple[type, ...] | None = None,
    required: bool = True,
) -> Any:
    value = parse_json_arg(cli_value)
    if value is None:
        value = config_value
    if required:
        value = require_value(value, name)
    if value is not None and expected_type and not isinstance(value, expected_type):
        raise ValueError(f"{name} must be {expected_type}, got {type(value)}")
    return value


def derive_pico_idx(config: dict) -> str:
    dataset_name = get_nested(config, ("pipeline", "dataset_name"))
    clinical_question = get_nested(config, ("pipeline", "clinical_question"))
    require_value(dataset_name, "pipeline.dataset_name")
    require_value(clinical_question, "pipeline.clinical_question")
    return hashlib.sha256(
        (clinical_question + dataset_name).encode("utf-8")
    ).hexdigest()[:8]


def resolve_pico_idx(cli_value: str | None, config: dict) -> str:
    configured_value = get_nested(config, ("pipeline", "pico_idx"))
    value = cli_value if cli_value is not None else configured_value
    if value in (None, "", "auto"):
        return derive_pico_idx(config)
    return str(value)


def resolve_dataset_path(args: argparse.Namespace, config: dict) -> str:
    return choose(
        getattr(args, "YOUR_DATASET_PATH", None),
        config_path(config, "dataset") or get_nested(config, ("pipeline", "dataset_path")),
        "YOUR_DATASET_PATH/pipeline.paths.dataset",
    )


def resolve_reports_path(args: argparse.Namespace, config: dict) -> str:
    return choose(
        getattr(args, "YOUR_REPORTS_PATH", None),
        config_path(config, "reports"),
        "YOUR_REPORTS_PATH/pipeline.paths.reports",
    )

