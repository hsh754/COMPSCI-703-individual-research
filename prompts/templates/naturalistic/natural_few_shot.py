def build_fewshot_prompt_csqa(item, stage='both'):
    """
    构建 CSQA 数据集 natural 风格的 few-shot prompt，支持 reasoning / answer / both 两阶段。
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
            f"{ex['reasoning']}\n"
        )

    def format_answer_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"Answer: {ex['answer']}\n"
        )

    # few-shot 示例
    k_examples = [
        {
            "question": "What is used to write on a blackboard?",
            "choices": {"A": "chalk", "B": "pen", "C": "crayon", "D": "marker", "E": "brush"},
            "reasoning": (
                "First, the most promising option is (A) chalk."
                "There are two reasons support this choice: firstly, Chalk produces visible marks on blackboards. Secondly, Chalk is designed not to damage the board surface.\n"
                "The least suitable option is (E) brush, as brushes are used for painting, not writing."
            ),
            "answer": "A"
        },
        {
            "question": "What do you wear on your feet to walk outside?",
            "choices": {"A": "hat", "B": "scarf", "C": "gloves", "D": "shoes", "E": "glasses"},
            "reasoning": (
                "First, (D) shoes are the most reasonable choice."
                "There are two reasons support this choice: firstly,Shoes protect your feet outdoors. Secondly, Shoes are specifically made for walking and provide comfort and grip.\n"
                "The least fitting option is (A) hat, as it's worn on the head, not feet."
            ),
            "answer": "D"
        }
    ]

    prompt = ""
    if stage == 'reasoning':
        prompt += "Here are some natural language reasoning examples:\n\n"
        for ex in k_examples:
            prompt += format_reasoning_example(ex) + "\n"
        question = item["question"]
        choices_str = format_choices(item["choices"])
        prompt += (
            f"Now, think about the following question carefully:\n\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Think through the following process:\n"
            "First, please identify the most promising option and explain why.\n"
            "Then, please give two specific reasons to support your choice.\n"
            "Finally, please briefly mention why one of the other options which is most unsuitable doesn't work.\n"
            "Please provide your reasoning in a few clear sentences, without revealing the final letter."
        )

    elif stage == 'answer':
        prompt += "Here are some examples of final answers:\n\n"
        for ex in k_examples:
            prompt += format_answer_example(ex) + "\n"
        question = item["question"]
        choices_str = format_choices(item["choices"])
        prompt += (
            f"Based on the previous reasoning about this question:\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Now, provide ONLY the letter (A, B, C, D, or E) corresponding to your answer. "
            f"DO NOT explain or add any other text."
        )

    else:
        prompt = "none"

    return prompt

def build_fewshot_prompt_cose(item, stage='both'):
    """
    针对 CoE-S 数据集构建 natural 风格的 few-shot prompt，选项自动生成标签 A/B/C 等。
    """
    def format_choices(choices_list):
        labels = [chr(ord('A') + i) for i in range(len(choices_list))]
        return " ".join([f"({label}) {choice}" for label, choice in zip(labels, choices_list)])

    def format_reasoning_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"{ex['reasoning']}\n"
        )

    def format_answer_example(ex):
        return (
            f"Question: {ex['question']}\n"
            f"Choices: {format_choices(ex['choices'])}\n"
            f"Answer: {ex['answer']}\n"
        )

    k_examples = [
        {
            "question": "What is used to write on a blackboard?",
            "choices": ["chalk", "pen", "crayon", "marker", "brush"],
            "reasoning": (
                "First, (A) chalk is the most suitable."
                "There are two reasons support this choice: Firstly, Chalk produces visible marks on blackboards. Secondly, Chalk is designed not to damage the board surface."
                "The least suitable is (E) brush, because brushes are not used for writing."
            ),
            "answer": "A"
        },
        {
            "question": "What do you wear on your feet to walk outside?",
            "choices": ["hat", "scarf", "gloves", "shoes", "glasses"],
            "reasoning": (
                "First, the best option is (D) shoes."
                "There are two reasons support this choice: Firstly, Shoes protect your feet outdoors. Secondly, Shoes are specifically made for walking and provide comfort and grip."
                "The most unsuitable is (A) hat, which is worn on your head, not feet."
            ),
            "answer": "D"
        }
    ]

    prompt = ""
    if stage == 'reasoning':
        prompt += "Here are some natural language reasoning examples:\n\n"
        for ex in k_examples:
            prompt += format_reasoning_example(ex) + "\n"
        question = item["question"]
        choices_str = format_choices(item["choices"])
        prompt += (
            f"Now, think about the following question carefully:\n\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            "Think through the following process:\n"
            "First, please identify the most promising option and explain why.\n"
            "Then, please give two specific reasons to support your choice.\n"
            "Finally, please briefly mention why one of the other options which is most unsuitable doesn't work.\n"
            "Please provide your reasoning in a few clear sentences, without revealing the final letter."
        )

    elif stage == 'answer':
        prompt += "Here are some examples of final answers:\n\n"
        for ex in k_examples:
            prompt += format_answer_example(ex) + "\n"
        question = item["question"]
        choices_str = format_choices(item["choices"])
        prompt += (
            f"Based on the previous reasoning about this question:\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n\n"
            f"Now, provide ONLY the letter (A, B, C, D, or E) corresponding to your answer. "
            f"DO NOT explain or add any other text."
        )

    else:
        prompt = "none"

    return prompt
