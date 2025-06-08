# 实验主调度脚本
# 串联数据加载、Prompt生成、模型推理、推理链抽取、评估与结果存储

# TODO: 实现命令行参数解析、主流程调度

import json
import os
import re
from tqdm import tqdm
from src.datasets.loader import load_commonsenseqa
from src.inference.infer import run_inference
from prompts.templates.templated.templated_few_shot import build_fewshot_prompt_csqa as build_templated_prompt_csqa
from prompts.templates.naturalistic.natural_few_shot import build_fewshot_prompt_csqa as build_natural_prompt_csqa
from src.cot_extraction.extractor import extract_cot_steps
from src.evaluation.entailment import compute_entailment_ratio
from src.utils.nli_client import get_nli_client
from src.evaluation.accuracy import compute_accuracy


def evaluate_csqa_entailment(prompt_type='natural', sample_size=103, model_name="mistral:7b"):
    """
    使用entailment ratio评估CommonsenseQA数据集上的推理质量

    参数：
        prompt_type: 使用的prompt类型，'templated'或'natural'
        sample_size: 评估样本数量
        model_name: 使用的模型名称 (e.g., 'mistral:7b', 'falcon3:7b')
    """
    # 验证prompt_type
    if prompt_type not in ['templated', 'natural']:
        raise ValueError(f"不支持的prompt类型: {prompt_type}，必须是 'templated' 或 'natural'")

    # 1. 加载数据
    dataset = load_commonsenseqa()
    val_data = dataset["validation"]

    # 2. 选择prompt构建函数
    if prompt_type == 'templated':
        build_prompt = build_templated_prompt_csqa
    else:  # natural
        build_prompt = build_natural_prompt_csqa

    # 3. 初始化NLI客户端
    nli_client = get_nli_client()

    results = []
    predictions = []
    references = []
    total_ratio = 0.0

    # 4. 遍历样本
    for item in tqdm(val_data.select(range(sample_size)), desc="Evaluating"):
        # 第一阶段：获取推理过程
        reasoning_prompt = build_prompt(item, stage='reasoning')
        reasoning_output = run_inference(
            reasoning_prompt,
            model_name=model_name,
            temperature=0.7,
            max_new_tokens=256,
            icl_mode="few-shot-cot"
        )

        # 提取推理步骤
        steps = extract_cot_steps(reasoning_output, prompt_type=prompt_type)

        # 第二阶段：获取答案
        answer_prompt = build_prompt(item, stage='answer')
        answer_output = run_inference(
            answer_prompt,
            model_name=model_name,
            temperature=0.7,
            max_new_tokens=32,
            icl_mode="few-shot-cot"
        )

        # 提取答案标签
        model_label = extract_choice_commonsenseqa(answer_output)
        standard_label = item['answerKey']

        # 获取答案文本
        # 修改：直接使用数据集中的label和text列表
        choices = {}
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            choices[label] = text

        # 更严谨的answer_text获取逻辑
        valid_labels = ['A', 'B', 'C', 'D', 'E']
        if model_label in valid_labels and model_label in choices:
            answer_text = choices[model_label]
            used_label = model_label
        else:
            # 如果模型输出的标签无效，使用标准答案
            answer_text = choices[standard_label]
            used_label = standard_label

        # 收集预测/参考
        predictions.append(model_label)
        references.append(standard_label)

        # 计算entailment ratio
        entail_info = compute_entailment_ratio(
            steps,
            answer_text,
            nli_client
        )

        print("\n" + "-" * 40 + " each evaluation result " + "-" * 40)
        print(f"MA: {model_label} | SA: {standard_label}")
        print(f"hypothesis: {entail_info['hypothesis']}")
        print(f"Valid Steps: {entail_info['valid_steps']} | Supporting Steps: {entail_info['entail_steps']}")
        print(f"Entailment Ratio: {entail_info['ratio']:.2%}")

        # 记录结果
        result = {
            "id": item.get("id", ""),
            "question": item["question"],
            "choices": choices,
            "answer_label": standard_label,
            "answer": choices[standard_label],
            "model_reasoning": reasoning_output,
            "model_answer": model_label,
            "used_answer_text": answer_text,
            "extracted_steps": steps,
            "entailment_info": entail_info
        }
        results.append(result)
        total_ratio += entail_info['ratio']

    # 5. 保存详细结果
    output_dir = os.path.join("outputs", "few_shot")
    os.makedirs(output_dir, exist_ok=True)
    model_name_cleaned = model_name.replace(':', '_')
    output_file = os.path.join(output_dir, f"csqa_entail_results_{model_name_cleaned}_few-shot_{prompt_type}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6. 输出总体统计
    avg_ratio = total_ratio / len(results)
    acc = compute_accuracy(predictions, references)
    print(f"\nEvaluation completed!")
    print(f"Sample size: {len(results)}")
    print(f"Average Entailment Ratio: {avg_ratio:.2%}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Results saved to: {output_file}")


def extract_choice_commonsenseqa(output):
    """
    针对commonsenseQA数据集，从模型输出中宽松提取A/B/C/D/E选项字母
    """
    # 优先匹配单独一行只有A-E的情况
    lines = output.splitlines()
    for line in lines:
        line = line.strip()
        if line in ['A', 'B', 'C', 'D', 'E']:
            return line
    # 匹配第一个A-E
    match = re.search(r'([A-E])', output)
    if match:
        return match.group(1)
    # 兜底：取第一个非空字符
    for c in output:
        if c.strip():
            return c
    return ""


if __name__ == "__main__":
    # 从环境变量获取参数
    prompt_type = os.environ.get("PROMPT_TYPE", "natural")
    model_name = os.environ.get("MODEL_NAME", "mistral:7b")
    sample_size = int(os.environ.get("SAMPLE_SIZE", "103"))

    # prompt_type 验证
    if prompt_type not in ['templated', 'natural']:
        print(f"警告：未知的prompt类型 '{prompt_type}'，将使用 'natural'")
        prompt_type = 'natural'

    evaluate_csqa_entailment(prompt_type=prompt_type, sample_size=sample_size, model_name=model_name)
