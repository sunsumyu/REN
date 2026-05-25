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
LLM_API_URL = "https://volley.inner.yzint.cn/v1/chat/completions"
# 从环境变量中读取 API Key
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# 默认使用通用的大模型（支持同 GPT / 千问的调用）
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# 流程控制配置
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5  # 指数退避系数
CONCURRENT_QA_LIMIT = 4     # 并行解答的并发度限制
