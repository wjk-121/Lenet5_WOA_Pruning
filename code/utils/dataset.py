# 数据集与 DataLoader：MNIST，划分 train/val/test。
from typing import Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_root: Path, batch_size: int = 64,
                    num_workers: int = 2, pin_memory: bool = True,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # 标准 MNIST 预处理
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(str(data_root), train=False, download=True, transform=tfm)

    # 划分 train/val
    n_total = len(train_full)  # 60000
    n_val = 5000
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
