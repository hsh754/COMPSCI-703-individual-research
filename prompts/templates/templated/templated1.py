def build_templated_prompt_1(item):
    """
    根据CommonsenseQA的item生成标准prompt
    """
    question = item["question"]
    choices = item["choices"]
    if isinstance(choices, dict):
        choices_str = " ".join([f"({k}) {v}" for k, v in choices.items()])
    elif isinstance(choices, list):
        choices_str = " ".join([f"({c['label']}) {c['text']}" for c in choices])
    else:
        choices_str = str(choices)
    prompt = (
        f"Thinking the problem step by step, and answer the question.\n"
        f"Step 1: Read the question.\n"
        f"Step 2: Consider all the choices.\n"
        f"Step 3: Choose one option as the best answer.\n\n"
        f"Question: {question}\n"
        f"Choices: {choices_str}\n\n"
        f"Please answer ONLY with one letter (A, B, C, D, or E). Do NOT explain your answer. Do NOT output anything except the letter."
    )
    return prompt

def build_prompt(item, stage='both'):
    """
    针对cos-e数据集，生成结构化prompt（支持entailment ratio评估任务）
    
    参数：
        item: 数据样本
        stage: 提示阶段
            - 'reasoning': 只输出推理过程
            - 'answer': 只输出答案
            - 'both': 同时输出推理和答案（默认）
    """
    question = item["question"]
    choices = item["choices"]
    labels = [chr(ord('A')+i) for i in range(len(choices))]
    choices_str = " ".join([f"({label}) {text}" for label, text in zip(labels, choices)])

    if stage == 'reasoning':
        prompt = (
            f"Let's answer this question step by step.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Think briefly through the following steps:\n"
            f"Step 1: Restate the question and name the single choice you find most plausible.\n"
            f"Step 2: Give two concise factual statements that support why you chose that option.\n"
            f"Step 3: In ONLY ONE sentence, explain which one choice are the most unsuitable.\n\n"
            f"⚠️ Please ensure each 'Step X: …' is exactly one line."
            # f"Please briefly provide ONLY your step-by-step reasoning. DO NOT give the final answer."
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
        prompt = (
            f"Let's solve this question step by step.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"First, explain your reasoning:\n"
            f"Step 1: Understand the question carefully.\n"
            f"Step 2: Analyze each choice in one sentence.\n"
            f"Step 3: Explain your reasoning.\n\n"
            f"After explaining, provide your answer as a single letter (A, B, C, D, or E) on a new line."
        )
    
    return prompt

def build_prompt_csqa(item, stage='both'):
    """
    针对CommonsenseQA数据集，生成结构化prompt（支持entailment ratio评估任务）
    
    参数：
        item: 数据样本
        stage: 提示阶段
            - 'reasoning': 只输出推理过程
            - 'answer': 只输出答案
            - 'both': 同时输出推理和答案（默认）
    """
    # 复用已有的build_templated_prompt_1函数来构建选项字符串
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
            f"Let's solve this question step by step.\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Think through the following steps:\n"
            f"Step 1: Restate the question and name the single choice you find most plausible.\n"
            f"Step 2: Give two concise factual statements that support why you chose that option.\n"
            f"Step 3: In ONLY ONE sentence, explain which one choice are the most unsuitable.\n\n"
            f" Please ensure each 'Step X: …' is exactly one line."
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
        prompt = build_templated_prompt_1(item)
    
    return prompt