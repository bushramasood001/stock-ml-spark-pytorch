"""
model.py

# PURPOSE: Define PyTorch neural network model for stock return prediction.
"""

import torch
import torch.nn as nn


class StockReturnNN(nn.Module):
    """
    Simple Feedforward Neural Network for regression.
    """

    def __init__(self, input_dim: int):
        super(StockReturnNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
