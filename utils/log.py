import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

# 日志格式常量
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO


def get_logger(log_dir: str, name: str, log_filename: str = None, level: int = LOG_LEVEL) -> logging.Logger:
    """创建并配置日志记录器
    
    Args:
        log_dir: 日志文件保存目录
        name: 日志记录器名称
        log_filename: 日志文件名（如果为None，则自动生成）
        level: 日志级别，默认为INFO
        
    Returns:
        配置好的Logger对象
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果未指定日志文件名，则使用时间戳自动生成
    if log_filename is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_filename = f"{timestamp}.log"
    else:
        log_filename = f"{log_filename}.log" if not log_filename.endswith('.log') else log_filename
    
    # 获取或创建记录器
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # 文件处理器（使用RotatingFileHandler实现日志轮转）
    log_path = os.path.join(log_dir, log_filename)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=50,  # 保留5个备份
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    @contextmanager
    def no_time():
        old_formatters = [h.formatter for h in logger.handlers]
        for h in logger.handlers:
            h.setFormatter(logging.Formatter("%(message)s"))
        try:
            yield
        finally:
            for h, f in zip(logger.handlers, old_formatters):
                h.setFormatter(f)
    
    logger.no_time = no_time

    with logger.no_time():
        logger.info( "=" * 25 + "   Settings   " + "=" * 25 )
    # 记录日志文件路径
    logger.info(f"Log File Path: {log_path}")
    
    return logger
