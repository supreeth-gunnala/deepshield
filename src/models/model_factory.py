"""
DeepShield Model Factory Module
Supports building both CNN Baseline and CRNN Advanced models with hardware allocation and checkpoint isolation.
"""

import os
import yaml
import logging
import torch
from src.models.cnn_baseline import DeepShieldCNN
from src.models.advanced_model import DeepShieldCRNN

logger = logging.getLogger("DeepShield.ModelFactory")

def get_device() -> torch.device:
    """Returns PyTorch device (CUDA GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Allocated Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Allocated Device: CPU")
    return device

def count_parameters(model: torch.nn.Module) -> int:
    """Returns total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(
    model_name: str = "cnn_baseline",
    config_path: str = "config/default_config.yaml",
    device: torch.device = None
) -> torch.nn.Module:
    """Builds and returns the requested model instance (cnn_baseline or crnn_advanced)."""
    device = device or get_device()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if model_name == "cnn_baseline":
            model_name = cfg.get("model", {}).get("name", "cnn_baseline")

    if model_name in ["cnn_baseline", "CNN_Baseline", "cnn"]:
        model = DeepShieldCNN(config_path=config_path)
    elif model_name in ["crnn_advanced", "crnn", "advanced"]:
        model = DeepShieldCRNN(config_path=config_path)
    else:
        raise ValueError(f"Unknown model architecture '{model_name}'. Available: 'cnn_baseline', 'crnn_advanced'")

    model.to(device)
    num_params = count_parameters(model)
    logger.info(f"Built '{model_name}' with {num_params:,} parameters on device '{device}'.")
    return model
