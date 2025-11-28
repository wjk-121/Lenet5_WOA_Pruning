# 全局配置：路径、超参数、WOA 设置等；自动创建所需目录。
from pathlib import Path
import os

# 以本文件为基准定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
MODEL_DIR = RESULTS_DIR / "models"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "reports"
TEST_IMG_DIR = PROJECT_ROOT / "test_imgs"

# 训练相关默认参数
TRAIN = {
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_workers": 2,
    "pin_memory": True,
    "save_best_only": True,
    "seed": 42,
}

# 剪枝与实验参数
PRUNE = {
    "fixed_ratios": [0.2, 0.3, 0.4, 0.5],         # 四组固定比例（分别应用于 conv1/conv2）
    "finetune_epochs": 2,                         # 剪枝后微调 epoch（建议 2,少量以节省时间）
    "finetune_lr": 0.005,
    "acc_drop_tolerance": 0.03,                   # 允许的精度下降（绝对值），例如 0.03 表示 3% 绝对精度点
}

# WOA 超参数（维度=2：conv1_ratio, conv2_ratio，均在[0, 0.9]范围内）
WOA = {
    "pop_size": 8,
    "iters": 10,
    "lb": [0.0, 0.0],
    "ub": [0.9, 0.9],
    "seed": 123,
}

# 设备与数据集
DEVICE_PREF = "cuda"  # "cuda" | "cpu"
DATASET = {
    "name": "MNIST",
    "root": str(DATA_DIR),
    "download": True,
}

def ensure_dirs():
    for p in [DATA_DIR, RESULTS_DIR, LOG_DIR, MODEL_DIR, FIG_DIR, REPORT_DIR, TEST_IMG_DIR]:
        os.makedirs(p, exist_ok=True)
    return {
        "data": DATA_DIR, "results": RESULTS_DIR, "logs": LOG_DIR,
        "models": MODEL_DIR, "figures": FIG_DIR, "reports": REPORT_DIR,
        "test_imgs": TEST_IMG_DIR,
    }
