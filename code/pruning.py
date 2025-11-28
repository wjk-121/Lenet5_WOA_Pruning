# L1 过滤器剪枝：选择保留通道并构建新网络，拷贝权重，保持前向维度一致。
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import numpy as np

from models.lenet import LeNet5, LeNet5Pruned

def _l1_filter_norm(conv: nn.Conv2d) -> torch.Tensor:
    # 计算每个输出通道的 L1 范数用于重要性排序
    with torch.no_grad():
        w = conv.weight.detach().abs()  # [out_c, in_c, k, k]
        norms = w.sum(dim=(1,2,3))      # 每个 out 通道的 L1
    return norms

def _select_keep_indices(conv: nn.Conv2d, keep_ratio: float) -> List[int]:
    out_ch = conv.out_channels
    keep = max(1, int(round(out_ch * (1.0 - keep_ratio))))  # keep count = 剩余通道数
    norms = _l1_filter_norm(conv)  # 大的更重要
    sorted_idx = torch.argsort(norms, descending=True)
    keep_idx = sorted_idx[:keep].cpu().numpy().tolist()
    keep_idx.sort()  # 保持升序，便于权重切片
    return keep_idx

def _copy_conv_weights(new: nn.Conv2d, old: nn.Conv2d,
                       keep_out_idx: List[int], keep_in_idx: List[int] = None):
    # 拷贝 conv 权重：选择输出通道 & 输入通道
    with torch.no_grad():
        if keep_in_idx is None:
            w = old.weight[keep_out_idx, :, :, :].clone()
        else:
            w = old.weight[keep_out_idx, :, :, :][:, keep_in_idx, :, :].clone()
        new.weight.copy_(w)
        if old.bias is not None:
            new.bias.copy_(old.bias[keep_out_idx].clone())

def _expand_fc_in_indices(keep_conv2_out: List[int], spatial: int = 4*4) -> List[int]:
    # conv2 输出通道保留索引 -> 对应到 fc1 输入向量的列索引
    idx = []
    for j in keep_conv2_out:
        base = j * spatial
        idx.extend(list(range(base, base + spatial)))
    return idx


def prune_lenet_l1(model: LeNet5, ratio_c1: float, ratio_c2: float) -> Tuple[LeNet5Pruned, Dict]:
    """
    输入裁剪比例 r1/r2（去掉比例），输出新模型与剪枝元信息。
    """
    if ratio_c1 < 0 or ratio_c2 < 0 or ratio_c1 >= 1 or ratio_c2 >= 1:
        raise ValueError("Prune ratios must be in [0,1).")

    # 目标设备 = 原模型所在设备（确保新旧模型与权重在同一设备）
    device = next(model.parameters()).device

    keep_idx_c1 = _select_keep_indices(model.conv1, ratio_c1)
    keep_idx_c2 = _select_keep_indices(model.conv2, ratio_c2)

    c1_new = len(keep_idx_c1)
    c2_new = len(keep_idx_c2)

    # 新模型放到相同 device，后续 copy_ 不会因跨设备报错，推理也保持在该设备
    pruned = LeNet5Pruned(c1_new, c2_new).to(device)

    # 拷贝 conv1 权重
    _copy_conv_weights(pruned.conv1, model.conv1, keep_out_idx=keep_idx_c1)

    # 拷贝 conv2 权重（同时裁剪 in/out 通道）
    _copy_conv_weights(pruned.conv2, model.conv2,
                       keep_out_idx=keep_idx_c2,
                       keep_in_idx=keep_idx_c1)

    # 处理 fc1: 输入列选择 conv2 保留的通道展开
    with torch.no_grad():
        old_fc1_w = model.fc1.weight.detach().clone()
        old_fc1_b = model.fc1.bias.detach().clone()
        in_idx = _expand_fc_in_indices(keep_idx_c2, spatial=4*4)
        new_fc1_w = old_fc1_w[:, in_idx].clone()
        pruned.fc1.weight.copy_(new_fc1_w)
        pruned.fc1.bias.copy_(old_fc1_b)

        # 其余全连接层保持不变
        pruned.fc2.weight.copy_(model.fc2.weight.detach().clone())
        pruned.fc2.bias.copy_(model.fc2.bias.detach().clone())
        pruned.fc3.weight.copy_(model.fc3.weight.detach().clone())
        pruned.fc3.bias.copy_(model.fc3.bias.detach().clone())

    meta = {
        "arch": "lenet5pruned",
        "ratios": {"conv1": ratio_c1, "conv2": ratio_c2},
        "kept": {"conv1": keep_idx_c1, "conv2": keep_idx_c2},
        "channels": {"conv1": c1_new, "conv2": c2_new},
        "note": "L1 filter pruning",
    }
    return pruned, meta

