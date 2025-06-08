from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 只加载一次模型和分词器
model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 你下载的模型路径或huggingface hub名
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def run_inference(prompt, max_new_tokens=32, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 只返回新生成部分
    return output[len(prompt):].strip()