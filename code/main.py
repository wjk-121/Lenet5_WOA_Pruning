import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from config import ensure_dirs, TRAIN, PRUNE
from train_baseline import train_baseline
from pruning_experiment import run_fixed_ratio_experiments, run_woa_search
from performance_compare import run_performance_compare
from utils.logger import setup_logger
from utils.dataset import get_dataloaders
import torch

def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all", choices=["train","prune","compare","all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # 全局日志
    from config import LOG_DIR, DATA_DIR
    logger, _ = setup_logger("main", str(LOG_DIR), "main")

    if args.step in ["train", "all"]:
        logger.info("步骤1：训练基线模型")
        train_baseline(epochs=args.epochs, batch_size=args.batch_size)

    if args.step in ["prune", "all"]:
        logger.info("步骤2：剪枝实验（固定比例 + WOA）")
        # 数据一次加载，复用
        train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR,batch_size=TRAIN["batch_size"],
                                                                num_workers=TRAIN["num_workers"],
                                                                pin_memory=TRAIN["pin_memory"],seed=TRAIN["seed"])
        # 固定比例
        run_fixed_ratio_experiments(logger, train_loader, val_loader, test_loader,
                                    torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # WOA 搜索
        run_woa_search(logger, train_loader, val_loader, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.step in ["compare", "all"]:
        logger.info("步骤3：性能对比与可视化")
        run_performance_compare()

    logger.info("流程完成。")

if __name__ == "__main__":
    main()