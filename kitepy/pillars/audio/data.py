"""
Audio data loading pipeline:
- Real audio folder structures (class subfolders of .wav files, or train/val splits)
- Real JSONL / CSV audio manifests ({"audio": "path.wav", "label": 0})
- Automatic log-STFT spectrogram extraction resized to (1, n_mels, num_frames)
- Synthetic spectrogram dataset for zero-download smoke testing
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import wave

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from kitepy.core.config import DataConfig, TrainingConfig


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def _load_wav_waveform(path: Path, target_samples: int = 16000) -> torch.Tensor:
    """Load a .wav file using Python built-in `wave` (or torchaudio if available for other formats)."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        dtype = np.int16 if sampwidth == 2 else (np.int8 if sampwidth == 1 else np.int32)
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        max_val = float(2 ** (8 * sampwidth - 1))
        arr = arr / max(1.0, max_val)
        wav = torch.from_numpy(arr)
    else:
        try:
            import torchaudio
            wav, _ = torchaudio.load(str(path))
            wav = wav.mean(dim=0)
        except ImportError:
            raise ImportError(
                f"Loading '{path.suffix}' audio requires torchaudio. Use .wav files or install torchaudio."
            )

    if wav.numel() < target_samples:
        wav = F.pad(wav, (0, target_samples - wav.numel()))
    elif wav.numel() > target_samples:
        wav = wav[:target_samples]
    return wav


def waveform_to_spectrogram(wav: torch.Tensor, n_mels: int = 64, num_frames: int = 128) -> torch.Tensor:
    """Convert 1D waveform into fixed-shape log-STFT spectrogram tensor (1, n_mels, num_frames)."""
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    window = torch.hann_window(256, device=wav.device)
    stft = torch.stft(wav, n_fft=256, hop_length=128, window=window, return_complex=True)
    spec = torch.log1p(stft.abs()).unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
    resized = F.interpolate(spec, size=(n_mels, num_frames), mode="bilinear", align_corners=False)
    return resized.squeeze(0)  # (1, n_mels, num_frames)


class AudioFolderDataset(Dataset):
    """Dataset of audio files labeled by subfolder or manifest."""

    def __init__(self, samples: List[Tuple[Path, int]], classes: List[str], n_mels: int = 64, num_frames: int = 128):
        self.samples = samples
        self.classes = classes
        self.n_mels = n_mels
        self.num_frames = num_frames
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = _load_wav_waveform(path)
        spec = waveform_to_spectrogram(wav, n_mels=self.n_mels, num_frames=self.num_frames)
        return spec, torch.tensor(label, dtype=torch.long)


class SyntheticAudioDataset(Dataset):
    """Synthetic spectrogram dataset for offline audio testing."""

    def __init__(self, num_samples: int = 64, n_mels: int = 64, num_frames: int = 128, num_classes: int = 10):
        self.specs = torch.randn(num_samples, 1, n_mels, num_frames)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]


def _scan_audio_folder(folder: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    classes = sorted([d.name for d in folder.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: List[Tuple[Path, int]] = []
    for cls_name in classes:
        for p in sorted((folder / cls_name).rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((p, class_to_idx[cls_name]))
    return samples, classes


def load_audio_data(
    dataset_name: Any,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 16
    num_classes = data_config.num_classes or 10

    if dataset_name == "synthetic" or dataset_name is None:
        train_ds = SyntheticAudioDataset(num_samples=48, num_classes=num_classes)
        val_ds = SyntheticAudioDataset(num_samples=16, num_classes=num_classes)
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
            None,
            {"num_classes": num_classes},
        )

    if isinstance(dataset_name, str):
        p = Path(dataset_name)
        if p.is_dir():
            train_dir, val_dir = p / "train", p / "val"
            if train_dir.is_dir() and val_dir.is_dir():
                tr_samples, classes = _scan_audio_folder(train_dir)
                va_samples, _ = _scan_audio_folder(val_dir)
                train_ds = AudioFolderDataset(tr_samples, classes)
                val_ds = AudioFolderDataset(va_samples, classes)
            else:
                samples, classes = _scan_audio_folder(p)
                full_ds = AudioFolderDataset(samples, classes)
                val_frac = data_config.val_split if isinstance(data_config.val_split, float) else 0.2
                val_len = max(1, int(len(full_ds) * val_frac)) if len(full_ds) >= 5 else 0
                if val_len > 0:
                    perm = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(train_config.seed)).tolist()
                    train_ds = Subset(full_ds, perm[val_len:])
                    val_ds = Subset(full_ds, perm[:val_len])
                else:
                    train_ds, val_ds = full_ds, None
            meta = {"num_classes": len(classes), "class_names": classes}
            return (
                DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
                DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0) if val_ds is not None else None,
                None,
                meta,
            )

        if p.is_file() and p.suffix.lower() in (".jsonl", ".csv"):
            records = []
            if p.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                with open(p, "r", encoding="utf-8") as f:
                    records = list(csv.DictReader(f))
            raw_labels = [str(r["label"]) for r in records]
            classes = sorted(set(raw_labels))
            c2i = {c: i for i, c in enumerate(classes)}
            samples = [(p.parent / r["audio"], c2i[str(r["label"])]) for r in records]
            full_ds = AudioFolderDataset(samples, classes)
            val_len = max(1, int(len(full_ds) * 0.2)) if len(full_ds) >= 5 else 0
            if val_len > 0:
                perm = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(train_config.seed)).tolist()
                train_ds, val_ds = Subset(full_ds, perm[val_len:]), Subset(full_ds, perm[:val_len])
            else:
                train_ds, val_ds = full_ds, None
            return (
                DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
                DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0) if val_ds is not None else None,
                None,
                {"num_classes": len(classes), "class_names": classes},
            )

    raise ValueError(f"Audio dataset '{dataset_name}' not found. Pass a .wav folder, .jsonl/.csv manifest, or 'synthetic'.")


__all__ = ["AudioFolderDataset", "SyntheticAudioDataset", "waveform_to_spectrogram", "load_audio_data"]
