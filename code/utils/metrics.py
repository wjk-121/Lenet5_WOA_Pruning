# 评估函数：参数量统计、准确率、推理速度。
import time
from typing import Tuple, Dict
import torch
import torch.nn as nn

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, data_loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def measure_inference_speed(model: nn.Module, data_loader, device, warmup_batches: int = 3,
                            measure_batches: int = 20) -> Dict[str, float]:

    model.eval()
    # 热身
    it = iter(data_loader)
    for _ in range(warmup_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(data_loader)
            x, _ = next(it)
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 正式测量
    total_imgs = 0
    t0 = time.time()
    it = iter(data_loader)
    for _ in range(measure_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(data_loader)
            x, _ = next(it)
        bs = x.size(0)
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_imgs += bs
    t1 = time.time()
    elapsed = max(t1 - t0, 1e-6)
    ips = total_imgs / elapsed
    return {"images_per_sec": ips, "batches": measure_batches, "total_images": total_imgs, "elapsed_sec": elapsed}
