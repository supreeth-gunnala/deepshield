import os, sys, yaml
PROJECT_ROOT = r'C:\\Users\\user\\Downloads\\Ai Project'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.dataset.dataloader_factory import create_dataloaders
cfg_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
train_loader, val_loader, test_loader = create_dataloaders(config_path=cfg_path)
print('Train batches:', len(train_loader))
print('Val batches:', len(val_loader))
print('Test batches:', len(test_loader))
print('Train samples:', sum(len(batch[0]) for batch in train_loader))
print('Val samples:', sum(len(batch[0]) for batch in val_loader))
print('Test samples:', sum(len(batch[0]) for batch in test_loader))
