import torch.nn as nn
import torch.nn.functional as F

import bitorch.layers as qnn


class BinarizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            qnn.QLinear(64, 32, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        output = F.log_softmax(x, dim=1)
        return output
