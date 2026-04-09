import os
import random
import numpy as np
import torch
import joblib  # 需要安装: pip install joblib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- 新增：保存和加载通用对象 (如Scaler) ---
def save_object(obj, filepath):
    # 确保存储目录存在
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(obj, filepath)
    print(f"[Info] Object saved to: {filepath}")

def load_object(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)