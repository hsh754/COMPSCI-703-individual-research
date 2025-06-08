# 全局参数配置
# 用于统一管理模型、路径、推理参数等

# 模型相关
MODEL_NAME = "Mistral-7b"
MODEL_PATH = "http://localhost:11434"  # Ollama 默认API地址

# 数据路径
DATA_ROOT = "data"
COMMONSENSEQA_PATH = f"{DATA_ROOT}/commonsenseqa"
COSE_PATH = f"{DATA_ROOT}/cose"
# Prompt模板路径
PROMPT_ROOT = "../prompts/templates"
EXAMPLES_ROOT = "../prompts/examples"

# 推理参数
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
ICL_MODE = "zero-shot-cot"  # 可选：zero-shot-cot、few-shot-cot、zero-shot-baseline

# 结果输出
OUTPUT_PATH = "outputs/results.jsonl"
LOG_PATH = "outputs/logs" 