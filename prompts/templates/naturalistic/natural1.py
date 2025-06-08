
def build_prompt(item, stage='both'):
    """
        自然语言风格 CoT prompt，支持推理和答案输出阶段
        """
    question = item["question"]
    choices = item["choices"]
    # 构建选项字符串
    labels = [chr(ord('A') + i) for i in range(len(choices))]
    choices_str = ", ".join([f"({l}) {choices[i]}" for i, l in enumerate(labels)])

    if stage == 'reasoning':
        prompt = (
            "Let's walk through this problem carefully.\n"
            f"Question: {question}\n"
            f"Options: {choices_str}.\n"
            "First, think about which option seems most plausible and why.\n"
            "Then, explain two reasons that support that choice.\n"
            "Also, briefly mention why the other options seem less suitable.\n"
            "Please provide your reasoning in a few clear sentences, without revealing the final letter."
        )
    elif stage == 'answer':
        prompt = (
            f"Based on the previous reasoning about this question:\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Now, provide ONLY the letter (A, B, C, D, or E) corresponding to your answer. "
            f"DO NOT explain or add any other text."
        )
    else:  # both
        reasoning = build_prompt(item, stage='reasoning')
        answer = build_prompt(item, stage='answer')
        prompt = reasoning + "\n\n" + answer
    return prompt

def build_prompt_csqa(item, stage='both'):
    """
    针对CommonsenseQA数据集，生成自然语言风格prompt（支持entailment ratio评估任务）
    
    参数：
        item: 数据样本
        stage: 提示阶段
            - 'reasoning': 只输出推理过程
            - 'answer': 只输出答案
            - 'both': 同时输出推理和答案（默认）
    """
    # 复用已有的build_natural_prompt_1函数来构建选项字符串
    question = item["question"]
    choices = item["choices"]
    if isinstance(choices, dict):
        choices_str = " ".join([f"({k}) {v}" for k, v in choices.items()])
    elif isinstance(choices, list):
        choices_str = " ".join([f"({c['label']}) {c['text']}" for c in choices])
    else:
        choices_str = str(choices)

    if stage == 'reasoning':
        prompt = (
            "Think about the question below carefully.\n"
            f"Question: {question}\n"
             f"Choices: {choices_str}\n\n"
            "Think through the following process:\n"
            "First, please identify the most promising option and explain why.\n"
            "Then, please give two specific reasons to support your choice.\n"
            "Finally, please briefly mention why one of the other options which is most unsuitable doesn't work.\n"
            "Please provide your reasoning in a few clear sentences, without revealing the final letter."
        )
    elif stage == 'answer':
        prompt = (
            f"Based on the previous reasoning about this question:\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Now, provide ONLY the letter (A, B, C, D, or E) corresponding to your answer. "
            f"DO NOT explain or add any other text."
        )
    else:  # 'both'
        prompt = "none"
    
    return prompt