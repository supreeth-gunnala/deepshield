"""
DeepShield Explainable AI (XAI) Module - Grad-CAM Implementation
Extracts spatial-spectral feature activations and backpropagated gradients from PyTorch Conv2D layers
to construct 2D model attention heatmaps over Log-Mel Spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

class GradCAMExplainer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for DeepShield models.
    Supports 2D CNN and CRNN architectures by targeting the final Conv2D layer.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: PyTorch model (DeepShieldCNN or DeepShieldCRNN).
            target_layer: Specific Conv2d module to hook. If None, automatically selects last Conv2d layer.
        """
        self.model = model
        self.model.eval()

        self.target_layer = target_layer or self._find_target_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []

        if self.target_layer is not None:
            self._register_hooks()

    def _find_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Automatically locates the last Conv2d layer in the model."""
        target = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input_tensor, output_tensor):
            self.activations = output_tensor.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Removes registered hooks from target layer."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Generates a 2D Grad-CAM heatmap normalized to [0, 1] matching input spectrogram dimensions.

        Args:
            input_tensor: Input tensor of shape (1, 1, Mel_Bins, Time_Frames).
            target_class: Target class index (0 for REAL, 1 for AI-GENERATED). If None, uses predicted class.

        Returns:
            heatmap: 2D numpy array of shape (Mel_Bins, Time_Frames) or None if unsupported.
            info: Metadata dictionary with target_class, pred_class, probabilities, and status.
        """
        if self.target_layer is None:
            return None, {"status": "Unsupported model architecture (No Conv2D layer found)"}

        # Ensure tensor is on correct device
        device = next(self.model.parameters()).device
        x = input_tensor.to(device)
        x.requires_grad_(True)

        self.model.zero_grad()

        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = pred_class

        # Target logit score
        score = logits[0, target_class]

        # Backward pass
        score.backward()

        if self.activations is None or self.gradients is None:
            return None, {
                "status": "Failed to capture activations or gradients",
                "pred_class": pred_class,
                "target_class": target_class,
                "probs": probs
            }

        # Channel-wise global average pooling of gradients
        # Gradients shape: (1, Channels, Height, Width)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to match input spatial dimensions (Mel_Bins, Time_Frames)
        input_shape = (x.shape[2], x.shape[3]) # (n_mels, time_frames)
        cam_resized = F.interpolate(cam, size=input_shape, mode="bilinear", align_corners=False)

        # Convert to numpy and normalize to [0, 1]
        heatmap = cam_resized.squeeze().cpu().numpy()

        cam_min, cam_max = heatmap.min(), heatmap.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (heatmap - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(heatmap)

        info = {
            "status": "Success",
            "pred_class": pred_class,
            "target_class": target_class,
            "target_class_name": "AI-GENERATED" if target_class == 1 else "REAL",
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
            "confidence": float(max(probs) * 100.0)
        }

        return heatmap, info

    def __del__(self):
        self.remove_hooks()
