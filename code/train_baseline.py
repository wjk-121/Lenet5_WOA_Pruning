# 训练基线模型 LeNet5；提供评估函数；保存最优模型与日志。
import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from config import MODEL_DIR, LOG_DIR, TRAIN
from utils.seed import seed_everything
from utils.logger import setup_logger
from utils.dataset import get_dataloaders
from utils.metrics import evaluate_accuracy, measure_inference_speed, count_parameters
from models.lenet import LeNet5

def get_device(pref: str = "cuda"):
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_baseline(epochs: int = None, batch_size: int = None, lr: float = None,
                   num_workers: int = None, pin_memory: bool = None, seed: int = None) -> Dict:
    # 读取配置或参数
    epochs = epochs or TRAIN["epochs"]
    batch_size = batch_size or TRAIN["batch_size"]
    lr = lr or TRAIN["lr"]
    num_workers = num_workers if num_workers is not None else TRAIN["num_workers"]
    pin_memory = TRAIN["pin_memory"] if pin_memory is None else pin_memory
    seed = seed or TRAIN["seed"]

    seed_everything(seed)
    logger, log_path = setup_logger("train", str(LOG_DIR), "train")

    device = get_device()
    logger.info(f"开始训练基线模型(LeNet5)，设备: {device}, epochs={epochs}, bs={batch_size}, lr={lr}")

    # 数据
    from config import DATA_DIR
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=batch_size,
                                                             num_workers=num_workers, pin_memory=pin_memory, seed=seed)

    # 模型与优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=TRAIN["momentum"], weight_decay=TRAIN["weight_decay"])

    best_val = 0.0
    best_path = os.path.join(MODEL_DIR, "baseline.pth")

    for epoch in range(1, epochs+1):
        model.train()
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
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        train_acc = correct / total if total else 0.0
        train_loss = running_loss / max(total, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)
        logger.info(f"Epoch [{epoch}/{epochs}] loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({
                "arch": "lenet5",
                "state_dict": model.state_dict(),
                "val_acc": best_val,
            }, best_path)
            logger.info(f"已保存更优模型 -> {best_path} (val_acc={best_val:.4f})")

    # 载入最佳并在 test 上评估
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_acc = evaluate_accuracy(model, test_loader, device)
    speed = measure_inference_speed(model, test_loader, device)
    params = count_parameters(model)
    logger.info(f"基线完成: test_acc={test_acc:.4f}, params={params}, speed={speed['images_per_sec']:.1f} img/s")

    torch.save({
        "arch": "lenet5",
        "state_dict": model.state_dict(),
        "val_acc": float(best_val),
        "test_acc": float(test_acc),
        "params": int(params),
        "speed": float(speed["images_per_sec"]),
    }, best_path)

    return {
        "path": best_path,
        "val_acc": best_val,
        "test_acc": test_acc,
        "params": params,
        "speed": speed["images_per_sec"],
        "log": log_path
    }