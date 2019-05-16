import torch
from torch import nn


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(nn.Module):
    "Sigmoid module with range `(low,x_max)`"

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


class LogisticEnsemble(nn.Module):
    def __init__(self, in_dim, y_range):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.bn = nn.BatchNorm1d(in_dim)
        self.rn = SigmoidRange(*y_range)

    def forward(self, x):
        y = self.bn(x)
        y = self.fc(y)
        y = self.rn(y)
        return y
