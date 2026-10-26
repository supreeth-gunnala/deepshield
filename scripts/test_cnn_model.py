"""
DeepShield Phase 6 Test Script
Verifies PyTorch 2D CNN forward pass, output shape, NaN/Inf checks, loss computation, and backpropagation.
"""

import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, count_parameters, get_device

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def run_cnn_model_test():
    print("==================================================")
    print("      DEEPSHIELD PHASE 6 CNN MODEL TEST          ")
    print("==================================================")

    device = get_device()

    # 1. Load one batch from Phase 5 DataLoader
    print(f"\n[STEP 1] Loading batch from DataLoader...")
    train_loader, _, _ = create_dataloaders(config_path=CONFIG_PATH)
    batch_features, batch_labels, batch_meta = next(iter(train_loader))

    batch_features = batch_features.to(device)
    batch_labels = batch_labels.to(device)

    batch_size = batch_features.shape[0]

    print("--------------------------------------------------")
    print("                INPUT BATCH STATS                 ")
    print("--------------------------------------------------")
    print(f"Batch Features Shape : {batch_features.shape} (Batch, C, Mels, Frames)")
    print(f"Batch Labels Shape   : {batch_labels.shape}")
    print(f"Batch Labels         : {batch_labels.tolist()}")
    print(f"Target Device        : {device}")
    print("--------------------------------------------------")

    # 2. Instantiate CNN Model
    print(f"\n[STEP 2] Building DeepShieldCNN Baseline Model...")
    model = build_model(config_path=CONFIG_PATH, device=device)
    num_params = count_parameters(model)

    print("\n--- MODEL ARCHITECTURE ---")
    print(model)
    print(f"\nTotal Trainable Parameters: {num_params:,}\n")

    # 3. Perform Forward Pass
    print(f"[STEP 3] Running Forward Pass...")
    model.eval()
    with torch.no_grad():
        logits = model(batch_features)

    print("--------------------------------------------------")
    print("                OUTPUT LOGITS STATS               ")
    print("--------------------------------------------------")
    print(f"Output Logits Shape  : {logits.shape} (Batch, Classes)")
    print(f"Output Logits Dtype  : {logits.dtype}")
    print(f"Sample Logit Values  :\n{logits[:3]}")
    print("--------------------------------------------------")

    # Forward Assertions
    assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN values detected in model output logits!"
    assert not torch.isinf(logits).any(), "Infinite values detected in model output logits!"

    # 4. Perform Backward Pass Test
    print(f"\n[STEP 4] Testing Loss Computation & Backward Pass (Backpropagation)...")
    model.train()
    model.zero_grad()

    criterion = nn.CrossEntropyLoss()
    train_logits = model(batch_features)
    loss = criterion(train_logits, batch_labels)

    print(f"Computed Dummy Loss  : {loss.item():.4f}")

    loss.backward()

    # Gradient Verification
    grads_exist = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"[FAIL] Missing gradient for parameter: {name}")
                grads_exist = False

    assert grads_exist, "Gradient check failed: some trainable parameters have None gradient!"
    print(f"[SUCCESS] Backward pass complete. Gradients successfully computed for all {len(list(model.parameters()))} parameter tensors.")

    print("\n==================================================")
    print("   DEEPSHIELD CNN MODEL TEST PASSED SUCCESSFULLY! ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_cnn_model_test()
    sys.exit(0 if success else 1)
