"""
Tabular data loading pipeline (CSV, DataFrame, NumPy, Tensor tuples, synthetic).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from kitepy.core.config import DataConfig, TrainingConfig


def load_tabular_data(
    dataset_name: Any,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 32
    task = (getattr(train_config, "task", None) or "classification").lower()
    metadata: Dict[str, Any] = {}

    if dataset_name == "synthetic" or dataset_name is None:
        n_samples, n_features = 128, 16
        X = torch.randn(n_samples, n_features)
        if task == "regression":
            y = (X[:, 0] * 2.0 - X[:, 1] * 0.5 + 0.1 * torch.randn(n_samples))
        else:
            y = (X[:, 0] + X[:, 1] > 0).long()
            metadata["num_classes"] = 2
        metadata["input_dim"] = n_features
        ds = TensorDataset(X, y)

    elif isinstance(dataset_name, tuple) and len(dataset_name) == 2:
        X_raw, y_raw = dataset_name
        X = torch.as_tensor(np.asarray(X_raw), dtype=torch.float32)
        if task == "regression":
            y = torch.as_tensor(np.asarray(y_raw), dtype=torch.float32)
        else:
            y = torch.as_tensor(np.asarray(y_raw), dtype=torch.long)
            metadata["num_classes"] = int(y.max().item()) + 1 if y.numel() > 0 else 2
        metadata["input_dim"] = X.shape[1]
        ds = TensorDataset(X, y)

    elif isinstance(dataset_name, str) and Path(dataset_name).is_file() and dataset_name.lower().endswith(".csv"):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for CSV loading. Install with: pip install pandas")
        df = pd.read_csv(dataset_name)
        target_col = df.columns[-1]
        y_series = df[target_col]
        X_df = pd.get_dummies(df.drop(columns=[target_col]), drop_first=False).astype(np.float32)
        # Standardize features
        vals = X_df.values
        mean = vals.mean(axis=0, keepdims=True)
        std = np.where(vals.std(axis=0, keepdims=True) < 1e-6, 1.0, vals.std(axis=0, keepdims=True))
        X = torch.tensor((vals - mean) / std, dtype=torch.float32)
        if task == "regression":
            y = torch.tensor(y_series.values, dtype=torch.float32)
        else:
            cats = y_series.astype("category")
            y = torch.tensor(cats.cat.codes.values, dtype=torch.long)
            metadata["class_names"] = [str(c) for c in cats.cat.categories.tolist()]
            metadata["num_classes"] = len(metadata["class_names"])
        metadata["input_dim"] = X.shape[1]
        ds = TensorDataset(X, y)
    else:
        raise ValueError(f"Unsupported tabular dataset: {dataset_name}")

    val_frac = data_config.val_split if isinstance(data_config.val_split, float) else 0.2
    val_len = max(1, int(len(ds) * val_frac)) if len(ds) >= 5 and val_frac > 0 else 0
    if val_len > 0:
        train_ds, val_ds = random_split(
            ds, [len(ds) - val_len, val_len],
            generator=torch.Generator().manual_seed(train_config.seed),
        )
    else:
        train_ds, val_ds = ds, None

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0) if val_ds is not None else None
    return train_loader, val_loader, None, metadata


__all__ = ["load_tabular_data"]
