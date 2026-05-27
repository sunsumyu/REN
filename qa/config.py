import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# 医药知识图谱 API 配置
GRAPH_API_URL = "https://ai.yzint.cn/api/knowledge/v1/graph/entity/random"
DEFAULT_KNOWLEDGE_BASE_ID = 201
DEFAULT_ENTITY_COUNT = 2
DEFAULT_HOP_COUNT = 2

# 大模型 API 配置
LLM_API_URL = "https://volley.yzint.cn/api/v1/chat/completions"
# 从环境变量中读取 API Key
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# 默认使用通用的大模型（支持同 GPT / 千问的调用）
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")

# 大模型池路由配置
MODEL_POOL_LIGHTWEIGHT = os.getenv("MODEL_POOL_LIGHTWEIGHT", "deepseek-v4-flash")
MODEL_POOL_PREMIUM = os.getenv("MODEL_POOL_PREMIUM", "deepseek-v4-pro")
MODEL_POOL_JUDGE = os.getenv("MODEL_POOL_JUDGE", "deepseek-v4-pro")

# 流程控制配置
MAX_RETRIES = 5  # 提升重试次数提升鲁棒性
RETRY_BACKOFF_FACTOR = 2.0  # 增加退避斜率以充分应对网关滑动窗口
CONCURRENT_QA_LIMIT = 4     # 并行解答的并发度限制

# 批处理高并发速率控制
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))                  # 单次并发生成的总样本数
BATCH_CONCURRENCY_LIMIT = int(os.getenv("BATCH_CONCURRENCY_LIMIT", "3"))  # 同时进行的多轮对话生成任务上限
GLOBAL_API_SEMAPHORE = int(os.getenv("GLOBAL_API_SEMAPHORE", "6"))      # 全局大模型并发限制（防429）

# 裁判打分阶段前的缓冲避让冷却（秒）
SOFT_DELAY_BEFORE_JUDGE = float(os.getenv("SOFT_DELAY_BEFORE_JUDGE", "3.0"))

# LLM 生成超参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.85"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.2"))

# 思维链提纯净化控制配置
PURIFY_LIMIT_RAW = os.getenv("PURIFY_LIMIT", "").strip()
PURIFY_LIMIT = int(PURIFY_LIMIT_RAW) if PURIFY_LIMIT_RAW and PURIFY_LIMIT_RAW.isdigit() else None

PURIFY_LINES_RAW = os.getenv("PURIFY_LINES", "").strip()
if PURIFY_LINES_RAW:
    import re
    # 去除中括号及空格，仅保留数字与逗号
    cleaned_lines = re.sub(r"[^\d,]", "", PURIFY_LINES_RAW)
    if cleaned_lines:
        PURIFY_LINES = [int(x) for x in cleaned_lines.split(",") if x]
    else:
        PURIFY_LINES = []
else:
    PURIFY_LINES = []

