import json
import os
from tqdm import tqdm
from src.datasets.loader import load_cose
from src.inference.infer import run_inference
from prompts.templates.naturalistic.natural_few_shot import build_fewshot_prompt_cose as build_natural_prompt_cose
from prompts.templates.templated.templated_few_shot import build_fewshot_prompt_coes as build_templated_prompt_cose
from src.cot_extraction.extractor import extract_cot_steps
from src.evaluation.entailment import compute_entailment_ratio
from src.utils.nli_client import get_nli_client
from src.evaluation.accuracy import compute_accuracy


def evaluate_cose_entailment(prompt_type='natural', sample_size=103, model_name="mistral:7b"):
    """
    使用entailment ratio评估cos-e数据集上的推理质量

    参数：
        prompt_type: 使用的prompt类型，'templated'或'natural'
        sample_size: 评估样本数量
        model_name: 使用的模型名称 (e.g., 'mistral:7b', 'falcon3:7b')
    """
    # 验证prompt_type
    if prompt_type not in ['templated', 'natural']:
        raise ValueError(f"不支持的prompt类型: {prompt_type}，必须是 'templated' 或 'natural'")

    # 1. 加载数据
    dataset = load_cose()
    val_data = dataset["validation"]

    # 2. 选择prompt构建函数
    if prompt_type == 'templated':
        build_prompt = build_templated_prompt_cose
    else:
        build_prompt = build_natural_prompt_cose
    # 3. 初始化NLI客户端（复用实例）
    nli_client = get_nli_client()

    results = []
    # 用于后续计算 accuracy
    predictions = []
    references = []
    total_ratio = 0.0

    # 4. 遍历样本
    for item in tqdm(val_data.select(range(sample_size)), desc="Evaluating"):
        # print("\n" + "="*80)
        # print(f"问题: {item['question']}")
        # print(f"选项: {item['choices']}")
        # print("-"*40 + " 第一阶段：推理过程 " + "-"*40)

        # 第一阶段：获取推理过程
        reasoning_prompt = build_prompt(item, stage='reasoning')
        # print("\nReasoning Output:")

        reasoning_output = run_inference(
            reasoning_prompt,
            model_name=model_name,
            temperature=0.7,
            max_new_tokens=256,
            icl_mode="few-shot-cot"
        )
        # print(reasoning_output)

        # 提取推理步骤
        steps = extract_cot_steps(reasoning_output, prompt_type=prompt_type)
        # print("\nThe extracted reasoning steps:")
        # for i, step in enumerate(steps, 1):
        #     print(f"Step {i}: {step}")
        #
        # print("\n" + "-"*40 + " 第二阶段：答案输出 " + "-"*40)
        # 第二阶段：获取答案
        answer_prompt = build_prompt(item, stage='answer')
        # print("\nAnswer Output:")

        answer_output = run_inference(
            answer_prompt,
            model_name=model_name,
            temperature=0.7,
            max_new_tokens=32,
            icl_mode="few-shot-cot"
        )
        # print(answer_output)
        #
        # 提取答案
        model_answer = answer_output.strip()

        # 标准答案标签
        standard_label = None
        for idx, choice in enumerate(item['choices']):
            if choice == item['answer']:
                standard_label = chr(ord('A') + idx)
                break

        # 根据模型输出字母匹配选项文本
        valid_labels = [chr(ord('A') + i) for i in range(len(item['choices']))]
        if model_answer in valid_labels:
            idx = ord(model_answer) - ord('A')
            # 若索引合法，使用对应文本；否则回退
            if 0 <= idx < len(item['choices']):
                answer_text = item['choices'][idx]
                model_label = model_answer
            else:
                answer_text = item['answer']
                model_label = standard_label
        else:
            # 非 A-E 输出回退到标准答案
            answer_text = item['answer']
            model_label = standard_label
            # 收集预测/参考
        predictions.append(model_label)
        references.append(standard_label)

        # 计算 entailment ratio，使用模型选的文本或标准答案文本
        entail_info = compute_entailment_ratio(
            steps,
            answer_text,
            nli_client
        )
        print("\n" + "-" * 40 + " each evaluation result " + "-" * 40)
        print(f"MA: {model_label} | SA: ({standard_label}) {item['answer']}")
        print(f"hypothesis: {entail_info['hypothesis']}")
        print(f"Valid Steps: {entail_info['valid_steps']} | Supporting Steps: {entail_info['entail_steps']}")
        print(f"Entailment Ratio: {entail_info['ratio']:.2%}")
        # print("="*80 + "\n")

        # 记录结果
        result = {
            "id": item["id"],
            "question": item["question"],
            "choices": item["choices"],
            "answer_label": standard_label,
            "answer": item["answer"],
            "model_reasoning": reasoning_output,
            "model_answer": model_label,
            'used_answer_text': answer_text,
            "extracted_steps": steps,
            "entailment_info": entail_info
        }
        results.append(result)
        total_ratio += entail_info['ratio']

    # 5. 保存详细结果
    output_dir = os.path.join("outputs", "few_shot")
    os.makedirs(output_dir, exist_ok=True)
    model_name_cleaned = model_name.replace(':', '_')
    output_file = os.path.join(output_dir, f"cose_entail_results_{model_name_cleaned}_few-shot_{prompt_type}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6. 输出总体统计
    avg_ratio = total_ratio / len(results)  # 7. 输出回答准确率
    acc = compute_accuracy(predictions, references)
    print(f"\nEvaluation completed!")
    print(f"Sample size: {len(results)}")
    print(f"Average Entailment Ratio: {avg_ratio:.2%}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # 从环境变量获取参数
    prompt_type = os.environ.get("PROMPT_TYPE", "natural")
    model_name = os.environ.get("MODEL_NAME", "mistral:7b")
    sample_size = int(os.environ.get("SAMPLE_SIZE", "103"))

    # prompt_type 验证
    if prompt_type not in ['templated', 'natural']:
        print(f"警告：未知的prompt类型 '{prompt_type}'，将使用 'natural'")
        prompt_type = 'natural'

    evaluate_cose_entailment(prompt_type=prompt_type, sample_size=sample_size, model_name=model_name)
