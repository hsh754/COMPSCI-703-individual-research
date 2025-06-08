# LLM推理接口封装
# 支持OpenAI、LLaMA、Mistral等多种模型，统一推理参数

# TODO: 实现API调用与本地模型推理的统一接口

import requests
import time

def run_inference(prompt, model_name, temperature, max_new_tokens, icl_mode):
    """
    使用Ollama本地API进行推理。
    prompt: 输入文本
    model_name: 模型名称（如'llama-7b'）
    temperature: 采样温度
    max_new_tokens: 最大生成token数
    icl_mode: ICL模式（可忽略）
    返回：模型生成的文本
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens
        },
        "stream": False
    }
    for _ in range(3):  # 最多重试3次
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            print(f"Ollama API调用失败，重试中... 错误信息: {e}")
            time.sleep(2)
    return "[Ollama API调用失败]" 