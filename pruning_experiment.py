# 运行四组固定比例剪枝实验与 WOA 搜索；保存模型与指标。
import os
import json
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from config import MODEL_DIR, LOG_DIR, FIG_DIR, REPORT_DIR, PRUNE, WOA as WCFG
from utils.logger import setup_logger
from utils.dataset import get_dataloaders
from utils.metrics import evaluate_accuracy, measure_inference_speed, count_parameters
from utils.seed import seed_everything
from utils.visualize import plot_woa_convergence
from models.lenet import LeNet5
from pruning import prune_lenet_l1
from woa_algorithm import WOA

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_baseline(device):
    path = os.path.join(MODEL_DIR, "baseline.pth")
    if not os.path.exists(path):
        raise FileNotFoundError("未找到基线模型，请先运行训练步骤。")
    ckpt = torch.load(path, map_location=device)
    model = LeNet5().to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model

def _finetune(model: nn.Module, train_loader, val_loader, device, epochs: int = 1, lr: float = 0.005, logger=None):
    # 简单微调：少量 epoch + 较小 LR
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    best_val = 0.0
    model.train()
    for e in range(1, epochs+1):
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        val_acc = evaluate_accuracy(model, val_loader, device)
        if logger: logger.info(f"微调 Epoch[{e}/{epochs}] loss={(running_loss/max(total,1)):.4f} val_acc={val_acc:.4f}")
        best_val = max(best_val, val_acc)
    return best_val

def run_fixed_ratio_experiments(logger, train_loader, val_loader, test_loader, device) -> Dict:
    results = {}
    baseline_model = _load_baseline(device)
    baseline_test_acc = evaluate_accuracy(baseline_model, test_loader, device)

    for r in PRUNE["fixed_ratios"]:
        logger.info(f"固定比例剪枝开始: ratio={r:.2f} (conv1/conv2 同比)")
        pruned, meta = prune_lenet_l1(baseline_model, r, r)
        # 可选微调
        if PRUNE["finetune_epochs"] > 0:
            _ = _finetune(pruned, train_loader, val_loader, device,
                          epochs=PRUNE["finetune_epochs"], lr=PRUNE["finetune_lr"], logger=logger)
        test_acc = evaluate_accuracy(pruned, test_loader, device)
        params = count_parameters(pruned)
        speed = measure_inference_speed(pruned, test_loader, device)["images_per_sec"]

        tag = f"ratio{int(r*100)}"
        save_path = os.path.join(MODEL_DIR, f"pruned_{tag}.pth")
        torch.save({
            "arch": "lenet5pruned",
            "state_dict": pruned.state_dict(),
            "meta": meta,
            "test_acc": test_acc,
            "params": params,
            "speed": speed,
        }, save_path)
        logger.info(f"剪枝完成[{tag}]: test_acc={test_acc:.4f}, params={params}, speed={speed:.1f} img/s, 保存: {save_path}")

        results[tag] = {
            "ratio": r,
            "test_acc": test_acc,
            "params": params,
            "speed": speed,
            "path": save_path
        }
    return {"baseline_acc": baseline_test_acc, "fixed": results}

def _woa_fitness_builder(baseline_model: nn.Module, train_loader, val_loader, device, logger):
    # 适应度 = 参数减少率 - 罚项；约束为精度下降<=阈值
    baseline_acc = evaluate_accuracy(baseline_model, val_loader, device)
    tol = PRUNE["acc_drop_tolerance"]

    def fitness(x: List[float]):
        r1, r2 = x[0], x[1]
        pruned, meta = prune_lenet_l1(baseline_model, r1, r2)
        # 轻微微调（加速可以设置 epochs=0）
        if PRUNE["finetune_epochs"] > 0:
            _ = _finetune(pruned, train_loader, val_loader, device,
                          epochs=1, lr=PRUNE["finetune_lr"], logger=None)
        acc = evaluate_accuracy(pruned, val_loader, device)
        # 绝对精度下降
        acc_drop = max(0.0, baseline_acc - acc)
        # 参数减少率
        import math
        from utils.metrics import count_parameters
        p_base = count_parameters(baseline_model)
        p_pruned = count_parameters(pruned)
        reduction = (p_base - p_pruned) / p_base
        # 罚项（超出阈值时）
        penalty = 0.0
        if acc_drop > tol:
            penalty = 10.0 * (acc_drop - tol)  # 强惩罚
        fit = reduction - penalty
        return fit, {"acc": acc, "acc_drop": acc_drop, "reduction": reduction, "r1": r1, "r2": r2}
    return fitness

def run_woa_search(logger, train_loader, val_loader, test_loader, device) -> Dict:
    baseline_model = _load_baseline(device)
    fitness_fn = _woa_fitness_builder(baseline_model, train_loader, val_loader, device, logger)
    woa = WOA(pop_size=WCFG["pop_size"], dim=2, lb=WCFG["lb"], ub=WCFG["ub"],
              iters=WCFG["iters"], fitness_fn=fitness_fn, seed=WCFG["seed"],logger=logger)
    logger.info(f"启动 WOA: pop={WCFG['pop_size']}, iters={WCFG['iters']}, lb={WCFG['lb']}, ub={WCFG['ub']}")
    result = woa.optimize()
    best = result["best_position"]
    logger.info(f"WOA 完成: best_ratio={best}, best_fitness={result['best_fitness']:.4f}")

    # 绘制收敛曲线
    fig_path = os.path.join(FIG_DIR, "woa_convergence.png")
    plot_woa_convergence(result["history"], fig_path)
    logger.info(f"WOA 收敛曲线保存: {fig_path}")

    # 用最优比率在 train+val 上微调后评估 test
    pruned, meta = prune_lenet_l1(baseline_model, best[0], best[1])
    _ = _finetune(pruned, train_loader, val_loader, device,
                  epochs=PRUNE["finetune_epochs"], lr=PRUNE["finetune_lr"], logger=logger)
    test_acc = evaluate_accuracy(pruned, test_loader, device)
    params = count_parameters(pruned)
    speed = measure_inference_speed(pruned, test_loader, device)["images_per_sec"]

    save_path = os.path.join(MODEL_DIR, "pruned_woa.pth")
    torch.save({
        "arch": "lenet5pruned",
        "state_dict": pruned.state_dict(),
        "meta": meta,
        "ratios": {"conv1": float(best[0]), "conv2": float(best[1])},
        "test_acc": test_acc,
        "params": params,
        "speed": speed,
    }, save_path)
    logger.info(f"WOA 剪枝模型保存: {save_path} (test_acc={test_acc:.4f}, params={params}, speed={speed:.1f})")

    return {
        "ratio": {"conv1": float(best[0]), "conv2": float(best[1])},
        "test_acc": test_acc, "params": params, "speed": speed, "path": save_path,
        "history": result["history"]
    }