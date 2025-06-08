# NLI模型客户端
# 封装roberta-large-mnli等NLI模型的加载与推理接口

# TODO: 实现NLI模型加载与推理

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Union, Dict, Optional
import time

class NLIClient:
    def __init__(self,
                 model_name: str = "roberta-large-mnli",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # 初始化分词器与模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # 标签映射：0→ENTAILMENT, 1→NEUTRAL, 2→CONTRADICTION
        self.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

    def _prepare_input(self, premise: str, hypothesis: str) -> Dict[str, torch.Tensor]:
        """准备单条输入的张量，供单条预测和entailment_score使用"""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict_batch(self,
                      premises: List[str],
                      hypotheses: List[str],
                      batch_size: int = 32) -> List[str]:
        """对一组 premise-hypothesis 对进行批量预测"""
        results: List[str] = []
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i + batch_size]
            batch_hypotheses = hypotheses[i:i + batch_size]
            try:
                # 使用 tokenizer 批量编码
                inputs = self.tokenizer(
                    batch_premises,
                    batch_hypotheses,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    preds = outputs.logits.argmax(dim=-1)

                # 将预测索引映射为标签
                for idx in preds:
                    results.append(self.id2label[int(idx)])
            except Exception as e:
                print(f"批量预测出错: {e}，转为单条预测模式")
                # 回退到单条预测
                for p, h in zip(batch_premises, batch_hypotheses):
                    results.append(self._predict_single(p, h))
        return results

    def _predict_single(self,
                        premise: str,
                        hypothesis: str,
                        max_retries: int = 3) -> str:
        """对单条输入进行预测，带重试机制"""
        for attempt in range(max_retries):
            try:
                inputs = self._prepare_input(premise, hypothesis)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pred = outputs.logits.argmax(dim=-1).item()
                return self.id2label[pred]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"NLI预测失败: {e}")
                    return "NEUTRAL"
                time.sleep(1)

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """获取单条输入对的 ENTAILMENT 概率分数"""
        inputs = self._prepare_input(premise, hypothesis)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # 返回 ENTAILMENT 类别的概率
            return probs[0, 0].item()

# 全局单例
_nli_client = None

def get_nli_client() -> NLIClient:
    """获取NLI客户端单例"""
    global _nli_client
    if _nli_client is None:
        _nli_client = NLIClient()
    return _nli_client

def nli_entailment(premise: str, hypothesis: str) -> str:
    """兼容性接口"""
    client = get_nli_client()
    return client._predict_single(premise, hypothesis) 