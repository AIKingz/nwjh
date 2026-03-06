import logging
from logging.handlers import TimedRotatingFileHandler
import os

# 日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径（每天新建一个）
log_file = os.path.join(LOG_DIR, "app.log")

# 日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 文件输出（按天切分，保留7天）
file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
file_handler.setFormatter(formatter)

# 配置全局 logger
logger = logging.getLogger("my_project")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
