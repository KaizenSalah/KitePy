"""
Tabular data loading pipeline (CSV, DataFrame, NumPy, Tensor tuples, synthetic).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from kitepy.core.config import DataConfig, TrainingConfig


def fit_tabular_preprocessor(X_df_raw: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit dummy encoding, median imputation, and z-score standardization on raw feature DataFrame."""
    import pandas as pd

    raw_columns = [str(c) for c in X_df_raw.columns]
    all_numeric = all(pd.api.types.is_numeric_dtype(X_df_raw[c]) for c in X_df_raw.columns)

    numeric_medians: Dict[str, float] = {}
    df_clean = X_df_raw.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            med_val = df_clean[col].median()
            med = float(med_val) if pd.notna(med_val) else 0.0
            numeric_medians[str(col)] = med
            df_clean[col] = df_clean[col].fillna(med)
        else:
            df_clean[col] = df_clean[col].fillna("missing").astype(str)

    X_encoded = pd.get_dummies(df_clean, drop_first=False).astype(np.float32)
    dummy_columns = [str(c) for c in X_encoded.columns]
    vals = np.nan_to_num(X_encoded.values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = vals.mean(axis=0, keepdims=True)
    raw_std = vals.std(axis=0, keepdims=True)
    std = np.where(raw_std < 1e-6, 1.0, raw_std)
    scaled = np.nan_to_num((vals - mean) / std, nan=0.0).astype(np.float32)

    preprocessor: Dict[str, Any] = {
        "raw_columns": raw_columns,
        "dummy_columns": dummy_columns,
        "numeric_medians": numeric_medians,
        "mean": mean.squeeze(0).astype(float).tolist(),
        "std": std.squeeze(0).astype(float).tolist(),
        "all_numeric": bool(all_numeric),
        "input_dim": int(scaled.shape[1]),
    }
    return torch.tensor(scaled, dtype=torch.float32), preprocessor


def transform_tabular_inputs(input_data: Any, preprocessor: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """Transform dict, list[dict], DataFrame, CSV path, or raw array using fitted Tabular preprocessor."""
    if preprocessor is not None:
        if isinstance(input_data, (str, Path)) and str(input_data).lower().endswith(".csv"):
            import pandas as pd
            input_data = pd.read_csv(input_data)

        is_row_dict = isinstance(input_data, dict)
        is_dict_list = isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], dict)
        is_dataframe = type(input_data).__name__ == "DataFrame"

        if is_row_dict or is_dict_list or is_dataframe:
            import pandas as pd

            df_in = pd.DataFrame([input_data]) if is_row_dict else (pd.DataFrame(input_data) if is_dict_list else input_data.copy())
            raw_cols: List[str] = preprocessor.get("raw_columns", [])
            medians: Dict[str, float] = preprocessor.get("numeric_medians", {})
            dummy_cols: List[str] = preprocessor.get("dummy_columns", [])

            for c in raw_cols:
                if c not in df_in.columns:
                    df_in[c] = medians.get(c, 0.0 if c in medians else "missing")
            if raw_cols:
                df_in = df_in[raw_cols]

            for c in df_in.columns:
                if str(c) in medians:
                    df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(medians[str(c)])
                else:
                    df_in[c] = df_in[c].fillna("missing").astype(str)

            encoded = pd.get_dummies(df_in, drop_first=False)
            encoded = encoded.reindex(columns=dummy_cols, fill_value=0.0).astype(np.float32)
            vals = np.nan_to_num(encoded.values, nan=0.0, posinf=0.0, neginf=0.0)
            mean = np.asarray(preprocessor["mean"], dtype=np.float32).reshape(1, -1)
            std = np.asarray(preprocessor["std"], dtype=np.float32).reshape(1, -1)
            scaled = np.nan_to_num((vals - mean) / std, nan=0.0).astype(np.float32)
            return torch.tensor(scaled, dtype=torch.float32)

    if isinstance(input_data, torch.Tensor):
        t = input_data.float()
    else:
        t = torch.as_tensor(np.asarray(input_data), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)

    if (
        preprocessor is not None
        and preprocessor.get("all_numeric", False)
        and t.shape[-1] == len(preprocessor.get("mean", []))
    ):
        mean_t = torch.tensor(preprocessor["mean"], dtype=torch.float32, device=t.device).view(1, -1)
        std_t = torch.tensor(preprocessor["std"], dtype=torch.float32, device=t.device).view(1, -1)
        t = torch.nan_to_num((t - mean_t) / std_t, nan=0.0)
    return t


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
        df = df.dropna(subset=[target_col])
        y_series = df[target_col]
        X, preprocessor = fit_tabular_preprocessor(df.drop(columns=[target_col]))
        preprocessor["target_column"] = str(target_col)
        metadata["preprocessor"] = preprocessor
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
    train_ds: Dataset
    val_ds: Optional[Dataset]
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


__all__ = ["fit_tabular_preprocessor", "transform_tabular_inputs", "load_tabular_data"]
