"""
DeepShield Advanced CRNN Classifier Module
Combines 2D Convolutional layers with Bidirectional Recurrent GRU layers to extract spatiotemporal features.
"""

import os
import yaml
import torch
import torch.nn as nn

class DeepShieldCRNN(nn.Module):
    """DeepShield Advanced CRNN Architecture (Conv2D + BiGRU + Dropout + Linear Classifier)."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        conv_channels: list[int] = None,
        rnn_hidden_dim: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3,
        config_path: str = None
    ):
        super().__init__()

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                full_cfg = yaml.safe_load(f) or {}
            cfg = full_cfg.get("advanced_model", {})
            self.input_channels = full_cfg.get("model", {}).get("input_channels", input_channels)
            self.num_classes = full_cfg.get("model", {}).get("num_classes", num_classes)
            conv_channels = cfg.get("conv_channels", conv_channels or [32, 64, 128, 256])
            self.rnn_hidden_dim = cfg.get("rnn_hidden_dim", rnn_hidden_dim)
            self.rnn_layers = cfg.get("rnn_layers", rnn_layers)
            self.dropout_rate = cfg.get("dropout", dropout)
        else:
            self.input_channels = input_channels
            self.num_classes = num_classes
            conv_channels = conv_channels or [32, 64, 128, 256]
            self.rnn_hidden_dim = rnn_hidden_dim
            self.rnn_layers = rnn_layers
            self.dropout_rate = dropout

        # 4 Convolutional Feature Extraction Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(self.input_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 128 -> 64
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 64 -> 32
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 32 -> 16
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 16 -> 8
        )

        # Recurrent Sequence Encoder (BiGRU)
        # Input shape to GRU: (Batch, Time_Frames, Feature_Dim)
        # Feature_Dim = conv_channels[3] * 8 = 256 * 8 = 2048
        gru_input_dim = conv_channels[3] * 8
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.rnn_layers > 1 else 0.0
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.rnn_hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor of shape (Batch, 1, Mel_Bins=128, Time_Frames=94)
        Output: Unnormalized logits of shape (Batch, 2)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) # Shape: (B, 256, 8, T')

        b, c, h, w = x.shape
        # Permute to sequence format: (Batch, Time_Frames, Channels * Height)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        # Pass through BiGRU
        gru_out, _ = self.gru(x) # (B, T', 2 * rnn_hidden_dim)

        # Temporal Average Pooling over time steps
        context = torch.mean(gru_out, dim=1) # (B, 2 * rnn_hidden_dim)

        logits = self.classifier(context)
        return logits
