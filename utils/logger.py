import logging
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
from logging.handlers import RotatingFileHandler
load_dotenv(".env")

LOG_DIR = Path(os.getenv("LOG_DIR"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")


class PrintToLogger:
    def __init__(self, logger, level=logging.DEBUG):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        if not message:
            return

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
        self._buffer = ""


def setup_loggers(
    log_file: str = str(LOG_FILE),
    redirect_print: bool = False,
):
    """
    只在程序入口调用一次
    """
    Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 共享文件 handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 控制台 handler：只给 workflow 用
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # workflow logger
    workflow_logger = logging.getLogger("project.workflow")
    workflow_logger.setLevel(logging.DEBUG)
    workflow_logger.propagate = False

    # detail logger
    detail_logger = logging.getLogger("project.detail")
    detail_logger.setLevel(logging.DEBUG)
    detail_logger.propagate = False

    # 防止重复添加 handler
    if not workflow_logger.handlers:
        workflow_logger.addHandler(console_handler)
        workflow_logger.addHandler(file_handler)

    if not detail_logger.handlers:
        detail_logger.addHandler(file_handler)

    if redirect_print:
        sys.stdout = PrintToLogger(detail_logger, logging.DEBUG)
        sys.stderr = PrintToLogger(detail_logger, logging.DEBUG)
        sys.stdout = PrintToLogger(workflow_logger, logging.DEBUG)
        sys.stderr = PrintToLogger(workflow_logger, logging.DEBUG)

    return workflow_logger, detail_logger


def get_workflow_logger(module_name: str = None) -> logging.Logger:
    base_name = "project.workflow"
    if module_name:
        return logging.getLogger(f"{base_name}.{module_name}")
    return logging.getLogger(base_name)


def get_detail_logger(module_name: str = None) -> logging.Logger:
    base_name = "project.detail"
    if module_name:
        return logging.getLogger(f"{base_name}.{module_name}")
    return logging.getLogger(base_name)