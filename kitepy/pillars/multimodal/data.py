"""
Multimodal data loading:
- Real image-text manifests (.jsonl, .json, .csv) with {"image": "...", "text": "..."}
- Synthetic (image, token_ids) dataset for zero-download CLIP/VLM testing
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from kitepy.core.config import DataConfig, TrainingConfig
from kitepy.pillars.language.data import SimpleCharTokenizer


class ImageTextManifestDataset(Dataset):
    """Dataset loading (image_path, caption_text) pairs from a JSONL/JSON/CSV manifest."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        base_dir: Path,
        tokenizer: Any,
        img_size: int = 64,
        max_length: int = 64,
    ):
        from torchvision import transforms

        self.records = records
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image

        rec = self.records[idx]
        img_ref = Path(rec["image"])
        img_path = img_ref if img_ref.is_absolute() else (self.base_dir / img_ref)
        img = self.transform(Image.open(img_path).convert("RGB"))
        caption = str(rec.get("text", rec.get("caption", "")))
        ids = self.tokenizer(caption, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
        return img, ids[: self.max_length]


class SyntheticMultimodalDataset(Dataset):
    """Synthetic (image, text_ids) dataset for CLIP/VLM testing."""

    def __init__(self, num_samples: int = 64, img_size: int = 64, seq_len: int = 16, vocab_size: int = 260):
        self.images = torch.randn(num_samples, 3, img_size, img_size)
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.input_ids[idx]


def load_multimodal_data(
    dataset_name: Any,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 16
    img_size = data_config.img_size or 64

    if dataset_name == "synthetic" or dataset_name is None:
        train_ds = SyntheticMultimodalDataset(num_samples=48, img_size=img_size)
        val_ds = SyntheticMultimodalDataset(num_samples=16, img_size=img_size)
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
            None,
            {},
        )

    if isinstance(dataset_name, str) and Path(dataset_name).is_file():
        p = Path(dataset_name)
        suf = p.suffix.lower()
        if suf == ".jsonl":
            records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suf == ".json":
            parsed = json.loads(p.read_text(encoding="utf-8"))
            records = parsed if isinstance(parsed, list) else [parsed]
        elif suf == ".csv":
            with open(p, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
        else:
            raise ValueError(f"Unsupported multimodal manifest extension: {suf}")

        tok = getattr(data_config, "_tokenizer", None) or SimpleCharTokenizer(vocab_size=260)
        full_ds = ImageTextManifestDataset(records, p.parent, tok, img_size=img_size, max_length=min(64, data_config.max_length))
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
            {},
        )

    raise ValueError(
        f"Multimodal dataset '{dataset_name}' not found. Pass an image-text .jsonl/.json/.csv manifest or 'synthetic'."
    )


__all__ = ["ImageTextManifestDataset", "SyntheticMultimodalDataset", "load_multimodal_data"]
