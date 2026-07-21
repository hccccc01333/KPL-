"""
项目统一的 logger 工具。

用法：
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("hello")

为什么不用 print？
    1. 带时间戳，调试方便
    2. 可分级（DEBUG/INFO/WARNING/ERROR）按需开关
    3. 可同时输出到文件，复盘历史运行
"""
import logging
import sys
from pathlib import Path

# TODO（学完 notebook 01 后实现）：
#   1. 创建 logs/ 目录
#   2. 配置 handler：StreamHandler(stdout) + FileHandler(logs/run.log)
#   3. 统一格式：%(asctime)s | %(name)s | %(levelname)s | %(message)s
#   4. 暴露 get_logger(name) 函数


def get_logger(name: str = "kpl") -> logging.Logger:
    """获取一个配置好的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
