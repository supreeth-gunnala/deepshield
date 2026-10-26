"""
DeepShield Model Trainer Module
Executes training & validation loops, loss/accuracy tracking, device allocation, and early stopping.
"""

import time
import logging
import torch
import torch.nn as nn

from src.training.checkpoint import CheckpointManager

logger = logging.getLogger("DeepShield.Trainer")

class Trainer:
    """Handles PyTorch model training, validation, metric calculations, and checkpoint management."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_manager: CheckpointManager = None,
        early_stopping_patience: int = 5
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.patience = early_stopping_patience

    def train_epoch(self, train_loader) -> tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_features, batch_labels, _ in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(batch_features)
            loss = self.criterion(logits, batch_labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("Encountered NaN or Inf loss during training iteration!")

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_features.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == batch_labels).sum().item()
            total_samples += batch_features.size(0)

        epoch_loss = running_loss / max(total_samples, 1)
        epoch_acc = (correct_preds / max(total_samples, 1)) * 100.0
        return epoch_loss, epoch_acc

    def validate(self, val_loader) -> tuple[float, float]:
        """Runs validation step over validation DataLoader."""
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            for batch_features, batch_labels, _ in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_features)
                loss = self.criterion(logits, batch_labels)

                running_loss += loss.item() * batch_features.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_preds += (preds == batch_labels).sum().item()
                total_samples += batch_features.size(0)

        val_loss = running_loss / max(total_samples, 1)
        val_acc = (correct_preds / max(total_samples, 1)) * 100.0
        return val_loss, val_acc

    def fit(self, train_loader, val_loader, epochs: int = 5) -> dict:
        """Executes full multi-epoch training and validation loop."""
        best_val_loss = float("inf")
        patience_counter = 0

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

        print("\n==================================================")
        print("         STARTING DEEPSHIELD MODEL TRAINING        ")
        print("==================================================")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            elapsed_time = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed_time:.2f}s) | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save latest checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, epoch, val_loss, val_acc, filename="latest_model.pt"
            )

            # Check if best validation loss achieved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_acc, filename="best_model.pt"
                )
                print(f"  --> Saved new best model checkpoint! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[NOTICE] Early stopping triggered after {epoch} epochs without improvement.")
                    break

        print("==================================================")
        print("          TRAINING COMPLETED SUCCESSFULLY          ")
        print("==================================================\n")
        return history
