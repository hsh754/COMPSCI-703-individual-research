def build_fewshot_prompt_csqa(item, stage='both'):
    """
    构建 few-shot prompt，支持 reasoning / answer / both 阶段，示例内嵌，格式与 build_prompt 保持一致。
    """
    def format_choices(choices):
        if isinstance(choices, dict):
            return " ".join([f"({k}) {v}" for k, v in choices.items()])
        else:
            return " ".join([f"({c['label']}) {c['text']}" for c in choices])

    def format_reasoning_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"Step 1: {ex['step1']}\n"
            f"Step 2: {ex['step2']}\n"
            f"Step 3: {ex['step3']}\n"
        )

    def format_answer_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"Answer: {ex['answer']}\n"
        )

    # 嵌入示例，已手动构建
    k_examples = [
        {
            "question": "What is used to write on a blackboard?",
            "choices": {"A": "chalk", "B": "pen", "C": "crayon", "D": "marker", "E": "brush"},
            "step1": "Chalk seems most plausible.",
            "step2": "Two reasons support this: (1) Chalk writes clearly on blackboards. (2) Chalk doesn't permanently mark the surface.",
            "step3": "(E) Brushes are most unsuitable as they are for painting, not writing.",
            "answer": "A"
        },
        {
            "question": "What do you wear on your feet to walk outside?",
            "choices": {"A": "hat", "B": "scarf", "C": "gloves", "D": "shoes", "E": "glasses"},
            "step1": "Shoes are the best fit.",
            "step2": "Two reasons support this: (1) Shoes protect your feet from rough surfaces. (2) Shoes provide support and stability while walking.",
            "step3": "(A) Hats are clearly unrelated to walking with feet.",
            "answer": "D"
        }
    ]

    # 构建示例部分
    prompt = ""
    if stage == 'reasoning':
        prompt += "Here are some examples of step-by-step reasoning:\n\n"
        for ex in k_examples:
            prompt += format_reasoning_example(ex) + "\n"
        question = item["question"]
        choices = item["choices"]
        choices_str = format_choices(choices)
        prompt += (
            f"Now, let's reason through a new question step by step:\n\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Think briefly through the following steps:\n"
            f"Step 1: Name the single choice you find most plausible.\n"
            f"Step 2: Give two concise factual statements that support why you chose that option.\n"
            f"Step 3: In ONLY ONE sentence, explain which one choice are the most unsuitable.\n\n"
            f"⚠️ Please ensure each 'Step X: …' is exactly one line."
        )

    elif stage == 'answer':
        prompt += "Here are some examples of answers based on previous reasoning:\n\n"
        for ex in k_examples:
            prompt += format_answer_example(ex) + "\n"
        question = item["question"]
        choices = item["choices"]
        choices_str = format_choices(choices)
        prompt += (
            f"Based on the previous reasoning about this question:\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Now, provide ONLY the letter (A, B, C, D, or E) corresponding to your answer. "
            f"DO NOT explain or add any other text."
        )

    else:  # 'both'
        prompt = "none"

    return prompt

def build_fewshot_prompt_coes(item, stage='both'):
    """
    针对 CoE-S 数据集构建 few-shot prompt，支持 reasoning / answer 阶段。
    CoE-S 选项不带标签，需自动生成 (A)(B)... 标签。
    """
    def format_choices(choices_list):
        labels = [chr(ord('A') + i) for i in range(len(choices_list))]
        return " ".join([f"({label}) {choice}" for label, choice in zip(labels, choices_list)])

    def format_reasoning_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"Step 1: {ex['step1']}\n"
            f"Step 2: {ex['step2']}\n"
            f"Step 3: {ex['step3']}\n"
        )

    def format_answer_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"Answer: {ex['answer']}\n"
        )

    # 示例（注意：choices 是 list，不带标签）
    k_examples = [
        {
            "question": "What is used to write on a blackboard?",
            "choices": ["chalk", "pen", "crayon", "marker", "brush"],
            "step1": "Chalk seems most plausible.",
            "step2": "Two reasons support this: (1) Chalk writes clearly on blackboards. (2) Chalk doesn't permanently mark the surface.",
            "step3": "(E) Brushes are most unsuitable as they are for painting, not writing.",
            "answer": "A"
        },
        {
            "question": "What do you wear on your feet to walk outside?",
            "choices": ["hat", "scarf", "gloves", "shoes", "glasses"],
            "step1": "Shoes are the best fit.",
            "step2": "Two reasons support this: (1) Shoes protect your feet from rough surfaces. (2) Shoes provide support and stability while walking.",
            "step3": "(A) Hats are clearly unrelated to walking with feet.",
            "answer": "D"
        }
    ]

    # 构建示例部分
    prompt = ""
    if stage == 'reasoning':
        prompt += "Here are some examples of step-by-step reasoning:\n\n"
        for ex in k_examples:
            prompt += format_reasoning_example(ex) + "\n"
        question = item["question"]
        choices = item["choices"]  # assumed to be list
        choices_str = format_choices(choices)
        prompt += (
            f"Now, let's reason through a new question step by step:\n\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Think briefly through the following steps:\n"
            f"Step 1: Name the single choice you find most plausible.\n"
            f"Step 2: Give two concise factual statements that support why you chose that option.\n"
            f"Step 3: In ONLY ONE sentence, explain which one choice are the most unsuitable.\n\n"
            f"⚠️ Please ensure each 'Step X: …' is exactly one line."
        )

    elif stage == 'answer':
        prompt += "Here are some examples of answers based on previous reasoning:\n\n"
        for ex in k_examples:
            prompt += format_answer_example(ex) + "\n"
        question = item["question"]
        choices = item["choices"]
        choices_str = format_choices(choices)
        prompt += (
            f"Based on the previous reasoning about this question:\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Now, provide ONLY the letter (A, B, C, D, or E) corresponding to your answer. "
            f"DO NOT explain or add any other text."
        )

    else:  # 'both' 可根据需要自行扩展
        prompt = "none"

    return prompt

