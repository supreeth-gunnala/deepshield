"""
DeepShield 2D CNN Baseline Classifier Module
Custom convolutional neural network for binary speech deepfake detection (REAL vs. AI-GENERATED).
"""

import os
import yaml
import torch
import torch.nn as nn

class DeepShieldCNN(nn.Module):
    """DeepShield 2D CNN Architecture for Log-Mel Spectrogram Speech Deepfake Detection."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        conv_channels: list[int] = None,
        dropout: float = 0.3,
        fc_hidden_dim: int = 128,
        config_path: str = None
    ):
        super().__init__()

        # Load parameters from YAML config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f).get("model", {})
            self.input_channels = cfg.get("input_channels", input_channels)
            self.num_classes = cfg.get("num_classes", num_classes)
            self.dropout_rate = cfg.get("dropout", dropout)
            conv_channels = cfg.get("conv_channels", conv_channels or [32, 64, 128])
            fc_hidden_dim = cfg.get("fc_hidden_dim", fc_hidden_dim)
        else:
            self.input_channels = input_channels
            self.num_classes = num_classes
            self.dropout_rate = dropout
            conv_channels = conv_channels or [32, 64, 128]
            fc_hidden_dim = fc_hidden_dim

        # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(self.input_channels, conv_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Conv -> BatchNorm -> ReLU -> MaxPool
        self.block3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive Pooling: Maps feature maps to fixed 4x4 spatial output regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened_dim = conv_channels[2] * 4 * 4 # 128 * 4 * 4 = 2048

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(flattened_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(fc_hidden_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor of shape (Batch, Input_Channels, Mel_Bins, Time_Frames) -> e.g. [B, 1, 128, 94]
        Output: Unnormalized logits of shape (Batch, Num_Classes) -> [B, 2]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits
