# entailment.py
# 逻辑蕴含率评估模块 - 使用软阈值分类
from typing import List, Dict, Any
from src.utils.nli_client import get_nli_client
import re
# 软阈值设置：大于等于此值判为 ENTAILMENT，小于等于此值判为 CONTRADICTION
ENTAILMENT_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.2


def compute_entailment_ratio(
    steps: List[str],
    answer: str,
    nli_client=None
) -> Dict[str, Any]:
    """
    计算推理步骤与最终答案的逻辑蕴含比例，采用软阈值分类。
    参数：
        steps: 推理步骤列表
        answer: 标准或模型选的答案文本
        nli_client: NLI 客户端实例（可选）
    返回：
        包含评估信息的字典：
        - ratio: ENTAILMENT 步骤比例
        - step_details: 每步的分数、标签和 is_entail 标记
        - valid_steps: 有效步骤总数
        - entail_steps: 判定为 ENTAILMENT 的步骤数
        - hypothesis: 用于 NLI 的假设文本
    """
    # 若无步骤，直接返回
    if not steps:
        return {
            "ratio": 0.0,
            "step_details": [],
            "valid_steps": 0,
            "entail_steps": 0,
            "hypothesis": ""
        }

    # 构造假设句（完整自然语句有助于 NLI 判断）
    hypothesis = f"The final choice is {answer}."

    # 获取 NLI 客户端
    if nli_client is None:
        nli_client = get_nli_client()

    step_details = []
    entail_count = 0

    # 对每个步骤计算 ENTAILMENT 概率，并基于阈值分类
    for step in steps:
        prob = nli_client.entailment_score(step, hypothesis)
     # 1. 对 step 做轻量补全，去掉多余前缀
        cleaned = step.strip()
      # 如果以 "(X)" 或 "and " 等开头，可以去掉这些标记
        cleaned = re.sub(r'^\([A-E]\)\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^[Aa]nd\s+', '', cleaned)

     # 2. 给 NLI 一个完整的前提句，让模型更好理解
        premise_for_nli = f"This step says: {cleaned}."
        prob = nli_client.entailment_score(premise_for_nli, hypothesis)
        if prob >= ENTAILMENT_THRESHOLD:
            label = "ENTAILMENT"
        elif prob <= CONTRADICTION_THRESHOLD:
            label = "CONTRADICTION"
        else:
            label = "ENTAILMENT"

        is_entail = (label == "ENTAILMENT")
        if is_entail:
            entail_count += 1

        step_details.append({
            "step_text": step,
            "score": prob,
            "label": label,
            "is_entail": is_entail
        })

    # 计算比例
    ratio = entail_count / len(steps)

    return {
        "ratio": ratio,
        "step_details": step_details,
        "valid_steps": len(steps),
        "entail_steps": entail_count,
        "hypothesis": hypothesis,
    }
