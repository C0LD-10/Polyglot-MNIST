"""
CNN Model for Spanish MNIST classification.
A lightweight 3-layer convolutional network with BatchNorm and Dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    3-Layer CNN for 28x28 grayscale character classification.

    Architecture:
        Conv1(1→32) → BN → ReLU → MaxPool(2)
        Conv2(32→64) → BN → ReLU → MaxPool(2)
        Conv3(64→128) → BN → ReLU → MaxPool(2)
        FC(128→256) → Dropout → FC(256→num_classes)

    Args:
        num_classes : Number of output classes (default: 80).
        dropout     : Dropout probability (default: 0.5).
    """

    def __init__(self, num_classes: int = 80, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 28→14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 14→7

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)), # 7→3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CNNModel("
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )


if __name__ == "__main__":
    model = CNNModel(num_classes=80)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(model)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
