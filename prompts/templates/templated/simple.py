

def build_prompt(item, stage='both'):
    """
    针对CommonsenseQA数据集，生成简单prompt（支持entailment ratio评估任务）
    """
    question = item["question"]
    choices = item["choices"]
    labels = [chr(ord('A') + i) for i in range(len(choices))]
    choices_str = " ".join([f"({label}) {text}" for label, text in zip(labels, choices)])

    if stage == 'reasoning':
        prompt = (
            f"Think about this question step by step.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Please explain your reasoning in simple steps."
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
        prompt = (
            f"Answer the following question.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n"
            f"Please answer with the letter A, B, C, D, or E without any explanation."
        )
    
    return prompt

def build_prompt_csqa(item, stage='both'):
    """
    针对CommonsenseQA数据集，生成简单prompt（支持entailment ratio评估任务）
    """
    # 复用已有的build_simple_prompt函数来构建选项字符串
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
            f"Let's think about this question step by step.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Please explain your reasoning in simple steps."
        )
    elif stage == 'answer':
        prompt = (
            f"For this question:\n"
            f"{question}\n"
            f"With choices: {choices_str}\n\n"
            f"Give ONLY the answer letter (A, B, C, D, or E)."
            f"DO NOT explain or add any other text."
        )
    else:  # 'both'
        prompt = (
            f"Answer the following question.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n"
            f"Please answer with the letter A, B, C, D, or E without any explanation."
        )
    return prompt