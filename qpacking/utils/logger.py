import logging
import os
import sys
from datetime import datetime
from colorlog import ColoredFormatter


class LevelFilter(logging.Filter):
    def __init__(self, handler_level):
        super().__init__()
        self.handler_level = handler_level

    def filter(self, record):
        return record.levelno == self.handler_level


def setup_log(name, log_dir="./logs", enable_file_log=True):
    """
    Setup logger with optional file logging and colored console output.

    :param name: Logger name
    :param log_dir: Directory to store log files (only used if enable_file_log=True)
    :param enable_file_log: Whether to log messages to files
    :return: Configured logger
    """

    # 创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 确保日志目录存在（仅当需要文件日志时）
    if enable_file_log:
        os.makedirs(log_dir, exist_ok=True)

    # 获取运行的 Python 脚本名称（去掉 `.py` 后缀）
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_files = {
        "debug": os.path.join(log_dir, f"{script_name}_debug_{timestamp}.log"),
        "info": os.path.join(log_dir, f"{script_name}_info_{timestamp}.log"),
        "warning": os.path.join(log_dir, f"{script_name}_warning_{timestamp}.log"),
        "error": os.path.join(log_dir, f"{script_name}_error_{timestamp}.log"),
    }

    # 定义日志格式
    log_format = {
        "debug": '%(asctime)s [%(levelname)s] - File: %(filename)s - Func: %(funcName)s - Line: %(lineno)d - [%(message)s]',
        "info": '%(asctime)s [%(levelname)s]: %(message)s',
        "warning": '%(asctime)s [%(levelname)s] - File: %(filename)s - Func: %(funcName)s - Line: %(lineno)d - [%(message)s]',
        "error": '%(asctime)s [%(levelname)s] - File: %(filename)s - Func: %(funcName)s - Line: %(lineno)d - [%(message)s]',
    }

    # 创建日志处理器（仅在启用文件日志时添加）
    if enable_file_log:
        for level, log_file in log_files.items():
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=True)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(logging.Formatter(log_format[level]))
            file_handler.addFilter(LevelFilter(getattr(logging, level.upper())))
            logger.addHandler(file_handler)

    # 创建彩色控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    color_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]: %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    return logger
