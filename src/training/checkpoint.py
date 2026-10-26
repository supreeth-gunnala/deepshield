"""
DeepShield Checkpoint Manager Module
Saves and restores PyTorch model weights, optimizer states, metrics, and configurations.
"""

import os
import logging
import torch

logger = logging.getLogger("DeepShield.Checkpoint")

class CheckpointManager:
    """Manages model checkpoint serialization and restoration."""

    def __init__(self, checkpoint_dir: str = "models_saved/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        val_acc: float,
        filename: str = "best_model.pt"
    ) -> str:
        """Saves checkpoint to disk."""
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        torch.save(checkpoint_state, filepath)
        logger.info(f"Saved checkpoint '{filename}' (Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%).")
        return filepath

    def load_checkpoint(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: torch.device = None
    ) -> dict:
        """Loads checkpoint weights into model and optimizer."""
        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at: '{filepath}'")

        map_location = device if device else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Successfully loaded checkpoint '{filename}' from Epoch {checkpoint.get('epoch', 'N/A')}.")
        return checkpoint
