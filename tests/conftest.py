"""
Shared pytest fixtures for KitePy test suite.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tiny_linear_model():
    """Fast 2-layer MLP for core Engine/Callback tests."""
    return nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))


@pytest.fixture
def tiny_classification_loaders():
    """Deterministic train/val DataLoaders for fast core tests."""
    torch.manual_seed(42)
    X_train = torch.randn(32, 10)
    y_train = torch.randint(0, 3, (32,))
    X_val = torch.randn(16, 10)
    y_val = torch.randint(0, 3, (16,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8, shuffle=False)
    return train_loader, val_loader


@pytest.fixture
def sample_image_folder(tmp_path):
    """Create a temporary 3-class ImageFolder dataset on disk (no internet needed)."""
    data_root = tmp_path / "dataset"
    classes = ["cats", "dogs", "birds"]
    for cls_name in classes:
        cls_dir = data_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(4):
            arr = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{idx}.png")
    return data_root
