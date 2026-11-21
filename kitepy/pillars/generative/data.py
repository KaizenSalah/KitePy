"""
Generative data loading pipeline (unlabeled / image reconstruction & diffusion).
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from kitepy.core.config import DataConfig, TrainingConfig


class SyntheticGenerativeDataset(Dataset):
    """Synthetic image tensor dataset for VAE / DDPM testing."""

    def __init__(self, num_samples: int = 48, in_channels: int = 3, img_size: int = 32):
        self.images = torch.rand(num_samples, in_channels, img_size, img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images[idx]


def load_generative_data(
    dataset_name: Any,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 16
    img_size = data_config.img_size or 32
    in_channels = data_config.in_channels or 3

    if dataset_name == "synthetic" or dataset_name is None:
        train_ds = SyntheticGenerativeDataset(num_samples=48, in_channels=in_channels, img_size=img_size)
        val_ds = SyntheticGenerativeDataset(num_samples=16, in_channels=in_channels, img_size=img_size)
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
            None,
            {},
        )

    # Delegate standard vision datasets (cifar10, folder, etc.) to vision loader
    from kitepy.pillars.vision.data import load_vision_data_with_meta
    return load_vision_data_with_meta(dataset_name, data_config, train_config)


__all__ = ["SyntheticGenerativeDataset", "load_generative_data"]
