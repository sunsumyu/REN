import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

# 确保代理白名单配置被正确注入到全局环境变量中（兼容大小写敏感的底层网络库）
_no_proxy_val = os.getenv("NO_PROXY", "").strip()
if _no_proxy_val:
    os.environ["NO_PROXY"] = _no_proxy_val
    os.environ["no_proxy"] = _no_proxy_val

# 医药知识图谱 API 配置
GRAPH_API_URL = "https://ai.yzint.cn/api/knowledge/v1/graph/entity/random"
DEFAULT_KNOWLEDGE_BASE_ID = 201
DEFAULT_ENTITY_COUNT = 2
DEFAULT_HOP_COUNT = 2

# 大模型 API 配置
LLM_API_URL = os.getenv("LLM_API_URL", "https://volley.yzint.cn/api/v1/chat/completions")
# 从环境变量中读取 API Key
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# 大模型方案一键切换。显式设置 LLM_MODEL / MODEL_POOL_* 时仍优先生效。
MODEL_PROFILE = os.getenv("MODEL_PROFILE", "deepseek").strip().lower()
MODEL_PROFILES = {
    "deepseek": {
        "default": "deepseek-v4-pro",
        "lightweight": "deepseek-v4-flash",
        "premium": "deepseek-v4-pro",
        "judge": "deepseek-v4-pro",
        "audit": "deepseek-v4-pro",
        "report": "deepseek-v4-flash",
    },
    "glm-audit": {
        "default": "deepseek-v4-pro",
        "lightweight": "deepseek-v4-flash",
        "premium": "deepseek-v4-pro",
        "judge": "glm-5.1",
        "audit": "Pro/zai-org/GLM-5.1",
        "report": "Pro/zai-org/GLM-4.7",
    },
    "glm-full": {
        "default": "glm-5.1",
        "lightweight": "glm-4.7",
        "premium": "glm-5.1",
        "judge": "glm-5.1",
        "audit": "Pro/zai-org/GLM-5.1",
        "report": "Pro/zai-org/GLM-4.7",
    },
}
_active_model_profile = MODEL_PROFILES.get(MODEL_PROFILE, MODEL_PROFILES["deepseek"])
_profile_locked = MODEL_PROFILE not in {"deepseek", "custom"}


def _model_setting(env_name: str, profile_key: str) -> str:
    if _profile_locked:
        return _active_model_profile[profile_key]
    return os.getenv(env_name, _active_model_profile[profile_key])

# 默认使用通用的大模型（支持同 GPT / 千问的调用）
LLM_MODEL = _model_setting("LLM_MODEL", "default")

# 大模型池路由配置
MODEL_POOL_LIGHTWEIGHT = _model_setting("MODEL_POOL_LIGHTWEIGHT", "lightweight")
MODEL_POOL_PREMIUM = _model_setting("MODEL_POOL_PREMIUM", "premium")
MODEL_POOL_JUDGE = _model_setting("MODEL_POOL_JUDGE", "judge")
AUDIT_MODEL = _model_setting("AUDIT_MODEL", "audit")
REPORT_MODEL = _model_setting("REPORT_MODEL", "report")

# 流程控制配置
MAX_RETRIES = 3  # 根据用户要求，大模型请求与网络重试次数严格限制在 3 次以内
RETRY_BACKOFF_FACTOR = 2.0  # 增加退避斜率以充分应对网关滑动窗口
CONCURRENT_QA_LIMIT = int(os.getenv("CONCURRENT_QA_LIMIT", "4"))     # 并行解答的并发度限制

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

# 提纯进程在全链路大并发时的最大并发度限制
PURIFY_CONCURRENCY = int(os.getenv("PURIFY_CONCURRENCY", "25"))

# 严格医疗事实/化学推演质检开关，开启时轻微幻觉或错词扣分将直接触发回滚
PURIFY_STRICT_RIGOR = os.getenv("PURIFY_STRICT_RIGOR", "false").strip().lower() in {"1", "true", "yes", "on"}

# 提纯失败时是否物理删除并隔离数据
PURIFY_DELETE_ON_FAIL = os.getenv("PURIFY_DELETE_ON_FAIL", "false").strip().lower() in {"1", "true", "yes", "on"}

# 净化起始行号 (1-based index)
PURIFY_START_LINE_RAW = os.getenv("PURIFY_START_LINE", "").strip()
PURIFY_START_LINE = int(PURIFY_START_LINE_RAW) if PURIFY_START_LINE_RAW and PURIFY_START_LINE_RAW.isdigit() else None

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

# PubMed (NCBI API) 访问与限流控制配置
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "").strip()
PUBMED_RATE_LIMIT = float(os.getenv("PUBMED_RATE_LIMIT", "10"))

