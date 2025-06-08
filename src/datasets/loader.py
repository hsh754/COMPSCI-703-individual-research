# src/datasets/loader.py
import os
from datasets import load_dataset, load_from_disk
from src.config import DATA_ROOT

def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def load_commonsenseqa():
    """
    加载 CommonsenseQA 数据集。
    本地目录： data/commonsenseqa
    """
    local_path = os.path.join(DATA_ROOT, "commonsenseqa")
    _ensure_dir(local_path)
    if os.path.exists(os.path.join(local_path, "dataset_info.json")):
        # 如果已经保存到磁盘，就直接载入
        return load_from_disk(local_path)
    # 否则从 Hugging Face 拉取并保存
    ds = load_dataset("commonsense_qa")
    ds.save_to_disk(local_path)
    return ds

def load_cose():
    """
    加载 CoS-E 数据集（Common Sense Explanations）。
    本地目录： data/cose
    注意：Hugging Face 上的名称可能是 "cos_e" 或类似，请确认。
    """
    local_path = os.path.join(DATA_ROOT, "cose")
    _ensure_dir(local_path)
    if os.path.exists(os.path.join(local_path, "dataset_info.json")):
        return load_from_disk(local_path)
    ds = load_dataset("cos_e","v1.11")   # 或者 "cos_e", "2017" 等，根据 Hugging Face Hub 上的版本标签
    ds.save_to_disk(local_path)
    return ds
