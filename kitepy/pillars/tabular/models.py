"""
Tabular Pillar - Tabular MLP & ResNet Models for Classification & Regression.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kitepy.core.adapters import get_adapter
from kitepy.core.base import BaseModel
from kitepy.core.config import TabularConfig
from .data import transform_tabular_inputs


class _TabularResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TabularResNet(nn.Module):
    """Modern ResNet-like architecture for tabular data."""

    def __init__(self, input_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        width = hidden_dims[0] if hidden_dims else 128
        self.input_proj = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*[_TabularResidualBlock(width, dropout) for _ in range(max(1, len(hidden_dims)))])
        self.norm = nn.BatchNorm1d(width)
        self.act = nn.SiLU()
        self.head = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.act(self.norm(h))
        return self.head(h)


class TabularMLP(nn.Module):
    """Clean MLP for tabular data."""

    def __init__(self, input_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.BatchNorm1d(h_d), nn.SiLU(), nn.Dropout(dropout)])
            in_d = h_d
        layers.append(nn.Linear(in_d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Tabular(BaseModel):
    """
    Tabular neural network API for CSV, NumPy, or Tensor structured data.

    Examples:
        >>> model = Tabular("tabular_resnet", task="classification")
        >>> model.train("iris.csv", epochs=10)
    """

    modality = "tabular"
    default_config_class = TabularConfig

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "tabular_resnet",
        config: Optional[Union[Dict, TabularConfig, str]] = None,
        task: str = "classification",
        **kwargs,
    ):
        self.task = task.lower()
        if self.task == "regression":
            kwargs.setdefault("loss", "mse")
        kwargs.setdefault("task", self.task)
        super().__init__(model, config, adapter=get_adapter(self.task), **kwargs)
        self.preprocessor: Optional[Dict[str, Any]] = None
        if self.task == "regression" and "num_classes" not in self._explicit_model_fields:
            self.config.num_classes = 1
            self._explicit_model_fields.add("num_classes")
        self.default_task = self.task

    def _auto_align_model_to_data(self, metadata: Dict[str, Any]):
        if metadata.get("preprocessor") is not None:
            self.preprocessor = metadata["preprocessor"]
        super()._auto_align_model_to_data(metadata)

    def _extra_save_metadata(self) -> Dict[str, Any]:
        return {"preprocessor": self.preprocessor} if self.preprocessor is not None else {}

    def _load_extra_metadata(self, checkpoint: Mapping[str, Any]) -> None:
        self.preprocessor = checkpoint.get("preprocessor")

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        else:
            out_dim = 1 if self.task == "regression" else self.config.num_classes
            if self.config.arch == "tabular_mlp":
                self.model = TabularMLP(self.config.input_dim, self.config.hidden_dims, out_dim, self.config.dropout)
            else:
                self.model = TabularResNet(self.config.input_dim, self.config.hidden_dims, out_dim, self.config.dropout)
        print(f"✓ Created Tabular model: {self.model_name} (in={self.config.input_dim}, task={self.task})")

    def _preprocess_input(self, input_data: Any) -> torch.Tensor:
        return transform_tabular_inputs(input_data, preprocessor=self.preprocessor)

    def _default_input_shape(self) -> Tuple[int, ...]:
        return (1, int(self.config.input_dim))


__all__ = ["Tabular", "TabularResNet", "TabularMLP"]
