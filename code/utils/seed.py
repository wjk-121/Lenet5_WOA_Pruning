# 固定随机种子，确保结果可重复。
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 为了确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False