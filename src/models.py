# src/models.py
import torch
import torch.nn as nn


# Logistic regression baseline (linear classifier)
class LRBaseline(nn.Module):
    """
    Multinomial logistic regression (softmax) as a single linear layer.
    Accepts (B, D) or (B, C, T).
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if x.ndim == 3:                   # (B, C, T)
            x = x.view(x.size(0), -1)     # -> (B, C*T)
        return self.fc(x)



# Small CNN for HAR
class SmallCNN(nn.Module):
    """
    Simple 1D CNN for HAR.
    Assumes input shape (B, C, T) where C is number of channels (e.g., 9).
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # global average pooling -> (B, 128, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, D) or (B, C, T)
        if x.ndim == 2:
            # if flat, we need to define how to reshape; for now assume already (B, C, T)
            raise ValueError("CNN expects input shape (B, C, T).")
        out = self.features(x)
        out = self.global_pool(out)
        out = self.classifier(out)
        return out



class MLPHar(nn.Module):
    def __init__(self, input_dim=561, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
