"""
Audio Pillar - Audio & Speech Classification Models.

Supports:
- Built-in `AudioCNN` & `AudioTransformer`
- HuggingFace `AutoModelForAudioClassification` when `pretrained=True` with an HF repo ID
- Automatic log-STFT spectrogram extraction from raw 1D waveforms
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from kitepy.core.adapters import get_adapter
from kitepy.core.base import BaseModel
from kitepy.core.config import AudioConfig


class AudioCNN(nn.Module):
    """Lightweight spectrogram CNN for audio classification."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AudioTransformer(nn.Module):
    """Lightweight spectrogram patch Transformer (AST-style) for audio classification."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 192, num_classes: int = 10):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=8, stride=8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> (B, N_patches, D)
        h = self.patch_embed(x).flatten(2).transpose(1, 2)
        h = self.encoder(h)
        pooled = self.norm(h.mean(dim=1))
        return self.classifier(pooled)


class Audio(BaseModel):
    """
    Audio / Speech model API.

    Accepts spectrograms `(B, 1, F, T)`, raw 1D waveforms `(B, samples)`, or HuggingFace audio IDs.
    """

    modality = "audio"
    default_config_class = AudioConfig
    default_task = "classification"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "audio_cnn",
        config: Optional[Union[Dict, AudioConfig, str]] = None,
        **kwargs,
    ):
        super().__init__(model, config, adapter=get_adapter("classification"), **kwargs)
        self.feature_extractor = None

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        elif self.config.pretrained and self.model_name not in ("audio_cnn", "audio_transformer"):
            try:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            except ImportError:
                raise ImportError("transformers required for pretrained HF audio models.")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name, num_labels=self.config.num_classes, ignore_mismatched_sizes=True
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        elif self.config.arch == "audio_transformer" or self.model_name == "audio_transformer":
            self.model = AudioTransformer(
                in_channels=self.config.in_channels,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
            )
        else:
            self.model = AudioCNN(
                in_channels=self.config.in_channels,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
            )
        print(f"✓ Created Audio model: {self.model_name} ({sum(p.numel() for p in self.model.parameters()):,} params)")

    def _preprocess_input(self, input_data: Any) -> Any:
        if self.feature_extractor is not None and not isinstance(input_data, dict):
            arr = input_data.cpu().numpy() if isinstance(input_data, torch.Tensor) else input_data
            return self.feature_extractor(arr, sampling_rate=self.config.sample_rate, return_tensors="pt")

        if not isinstance(input_data, torch.Tensor):
            input_data = torch.as_tensor(input_data, dtype=torch.float32)

        if input_data.ndim == 1:
            input_data = input_data.unsqueeze(0)
        if input_data.ndim == 2:
            window = torch.hann_window(256, device=input_data.device)
            stft = torch.stft(input_data, n_fft=256, hop_length=128, window=window, return_complex=True)
            return torch.log1p(stft.abs()).unsqueeze(1)
        if input_data.ndim == 3:
            return input_data.unsqueeze(0)
        return input_data


__all__ = ["Audio", "AudioCNN", "AudioTransformer"]
