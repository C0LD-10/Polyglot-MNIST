"""
LeNet-5 variant adapted for Spanish MNIST (80 classes, 28x28 grayscale).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5Spanish(nn.Module):
    """
    LeNet-5 adapted for Spanish MNIST.

    Original LeNet-5 (LeCun et al., 1998) modified to:
      - Accept 28x28 input (vs original 32x32)
      - Output 80 classes (vs original 10)
      - Add BatchNorm for training stability

    Args:
        num_classes : Number of output classes (default: 80).
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28→28
        self.bn1   = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(2, 2)                           # 28→14

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)              # 14→10
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(2, 2)                           # 10→5

        # Fully connected
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(torch.tanh(self.bn1(self.conv1(x))))
        x = self.pool2(torch.tanh(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LeNet5Spanish("
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )


if __name__ == "__main__":
    model = LeNet5Spanish(num_classes=80)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(model)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
