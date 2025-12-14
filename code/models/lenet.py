# LeNet-5 基线与可变通道数的剪枝版。针对 MNIST 28x28。
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 28x28 -> conv5(valid)->24x24 -> pool2->12x12 -> conv5->8x8 -> pool2->4x4
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=True)
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, num_classes, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 12x12
        x = self.pool(F.relu(self.conv2(x)))  # 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5Pruned(nn.Module):

    def __init__(self, c1: int, c2: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=0, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, padding=0, bias=True)
        self.fc1 = nn.Linear(c2 * 4 * 4, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, num_classes, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x