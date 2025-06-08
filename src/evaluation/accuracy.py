# 准确率评估模块
# 判断模型输出的最终答案是否正确

def compute_accuracy(predictions, references):
    """
    输入：predictions（模型输出答案列表），references（标准答案列表）
    输出：准确率（float）
    """
    correct = sum([p == r for p, r in zip(predictions, references)])
    return correct / len(predictions) 