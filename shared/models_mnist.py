"""
Shared MNIST model architectures — dùng chung cho cả 4 frameworks.
Không sửa file này trừ khi cả nhóm đồng ý.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Simple 2-layer neural network.
    Input: 784 (28x28 flattened) → Output: 10 classes
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepNN(nn.Module):
    """
    Deep dense network with Dropout.
    Input: 784 → 512 → 256 → 128 → 10
    """
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN(nn.Module):
    """
    Convolutional neural network with BatchNorm and MaxPooling.
    Input: 1x28x28 → Output: 10 classes
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Hyperparameters chuẩn — không đổi trên cả 4 frameworks
HPARAMS = {
    "lr":         0.001,
    "batch_size": 64,
    "epochs":     10,
    "optimizer":  "Adam",
}

MODEL_CLASSES = {
    "SimpleNN": SimpleNN,
    "DeepNN":   DeepNN,
    "CNN":      CNN,
}

# Models để chạy phần repeat (TC7) — cùng thứ tự trên mọi framework
MODELS = ["SimpleNN", "DeepNN", "CNN"]

# Số lần lặp mỗi model trong phần repeat
NUM_RUNS = 3

# TC2 — config sweep dùng SimpleNN
TC2_CONFIGS = [
    {"lr": 0.001, "batch_size": 32,  "epochs": 10},
    {"lr": 0.01,  "batch_size": 64,  "epochs": 10},
    {"lr": 0.05,  "batch_size": 128, "epochs": 10},
]
