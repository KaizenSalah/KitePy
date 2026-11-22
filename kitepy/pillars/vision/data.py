"""
Vision data loading engine.

Handles:
- Built-in datasets (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, ImageNet)
- Custom ImageFolder directories (pre-split or auto-split WITHOUT augmentation leak!)
- Synthetic datasets for zero-download smoke testing
- HuggingFace Datasets (`hf://dataset_name`)
- Automatic metadata extraction (num_classes, class_names, img_size, in_channels)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from kitepy.core.config import DataConfig, TrainingConfig
from kitepy.core.presets import DATASET_PRESETS


# ============================================================================
# SYNTHETIC DATA (For fast testing without downloads)
# ============================================================================

class SyntheticDataset(Dataset):
    """Synthetic dataset for testing training loops without downloads."""

    def __init__(
        self,
        num_samples: int = 100,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 10,
        transform=None,
        task: str = "classification",
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.transform = transform

        self.data = torch.randn(num_samples, in_channels, img_size, img_size)
        task_l = (task or "classification").lower()
        if task_l == "multilabel":
            self.targets = torch.randint(0, 2, (num_samples, num_classes)).float()
        elif task_l == "binary":
            self.targets = torch.randint(0, 2, (num_samples,))
        elif task_l == "regression":
            self.targets = torch.randn(num_samples)
        else:
            self.targets = torch.randint(0, max(1, num_classes), (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SyntheticSegmentationDataset(Dataset):
    """Synthetic (image, pixel_mask) dataset for semantic segmentation."""

    def __init__(self, num_samples: int = 32, img_size: int = 64, in_channels: int = 3, num_classes: int = 2):
        self.images = torch.randn(num_samples, in_channels, img_size, img_size)
        self.masks = torch.randint(0, num_classes, (num_samples, img_size, img_size))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class SyntheticDetectionDataset(Dataset):
    """Synthetic (image, box/label target dict) dataset for object detection."""

    def __init__(self, num_samples: int = 24, img_size: int = 64, in_channels: int = 3, num_classes: int = 10):
        self.images = torch.rand(num_samples, in_channels, img_size, img_size)
        self.targets = []
        for _ in range(num_samples):
            x1 = torch.randint(1, img_size // 2, (2,)).float()
            y1 = torch.randint(1, img_size // 2, (2,)).float()
            boxes = torch.stack([x1, y1, x1 + 10.0, y1 + 10.0], dim=1)
            labels = torch.randint(1, max(2, num_classes), (2,))
            self.targets.append({"boxes": boxes, "labels": labels})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def detection_collate_fn(batch):
    """Collate function for variable-length detection targets."""
    return [b[0] for b in batch], [b[1] for b in batch]


# ============================================================================
# CUSTOM IMAGE FOLDER LOADING
# ============================================================================

def _load_custom_folder(
    data_path: Path,
    train_transform,
    val_transform,
    train_config: TrainingConfig,
    data_config: DataConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """
    Load custom image folder dataset without leaking train_transform into validation.
    """
    from torchvision import datasets

    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"

    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 32

    if train_path.is_dir():
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        classes = list(train_dataset.classes)

        if val_path.is_dir():
            val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        else:
            # Split train folder into train/val using separate ImageFolder transforms
            val_frac = data_config.val_split if isinstance(data_config.val_split, float) else 0.2
            if val_frac > 0 and len(train_dataset) >= 5:
                n_total = len(train_dataset)
                val_len = max(1, int(n_total * val_frac))
                gen = torch.Generator().manual_seed(train_config.seed)
                perm = torch.randperm(n_total, generator=gen).tolist()
                val_idx, train_idx = perm[:val_len], perm[val_len:]
                val_full = datasets.ImageFolder(train_path, transform=val_transform)
                train_dataset = Subset(train_dataset, train_idx)
                val_dataset = Subset(val_full, val_idx)
            else:
                val_dataset = None

        test_dataset = datasets.ImageFolder(test_path, transform=val_transform) if test_path.is_dir() else None
    else:
        # Unsplit folder: create two ImageFolder instances so val NEVER gets train augmentations
        full_train = datasets.ImageFolder(data_path, transform=train_transform)
        full_val = datasets.ImageFolder(data_path, transform=val_transform)
        classes = list(full_train.classes)

        val_frac = data_config.val_split if isinstance(data_config.val_split, float) else 0.2
        n_total = len(full_train)
        val_size = max(1, int(n_total * val_frac)) if n_total >= 2 and val_frac > 0 else 0
        train_size = n_total - val_size

        gen = torch.Generator().manual_seed(train_config.seed)
        perm = torch.randperm(n_total, generator=gen).tolist()
        train_idx, val_idx = perm[:train_size], perm[train_size:]

        train_dataset = Subset(full_train, train_idx)
        val_dataset = Subset(full_val, val_idx) if val_idx else None
        test_dataset = None

    print(f"✓ Loaded custom dataset from {data_path} ({len(classes)} classes)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=data_config.shuffle,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=data_config.drop_last,
    )
    val_loader: Optional[DataLoader] = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    ) if val_dataset is not None else None
    test_loader: Optional[DataLoader] = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    ) if test_dataset is not None else None

    metadata = {"num_classes": len(classes), "class_names": classes}
    return train_loader, val_loader, test_loader, metadata


# ============================================================================
# VISION DATA LOADING
# ============================================================================

def load_vision_data_with_meta(
    dataset_name: Any,
    data_config: DataConfig,
    train_config: TrainingConfig,
    download: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """
    Load vision datasets and return (train_loader, val_loader, test_loader, metadata).
    """
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError("torchvision is required for vision datasets. Install with: pip install torchvision")

    metadata: Dict[str, Any] = {}
    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 32
    data_dir = getattr(data_config, "data_dir", "./data") or "./data"

    if not isinstance(dataset_name, str):
        raise ValueError(f"Unsupported vision dataset argument: {type(dataset_name)}")

    # Preset lookup
    preset = DATASET_PRESETS.get(dataset_name, {})
    target_img_size = data_config.img_size or preset.get("img_size", 224)
    in_channels = data_config.in_channels or 3
    num_classes = data_config.num_classes or preset.get("num_classes", 10)
    is_grayscale_dataset = dataset_name in ("mnist", "fashion_mnist", "fashion-mnist")

    if "class_names" in preset:
        metadata["class_names"] = list(preset["class_names"])
    if "num_classes" in preset:
        metadata["num_classes"] = int(preset["num_classes"])
    elif dataset_name == "synthetic":
        metadata["num_classes"] = int(num_classes)

    train_transform = get_train_transforms(
        target_img_size,
        data_config.augmentation,
        to_rgb=(is_grayscale_dataset and in_channels == 3),
        grayscale=(is_grayscale_dataset and in_channels == 1),
    )
    val_transform = get_val_transforms(
        target_img_size,
        to_rgb=(is_grayscale_dataset and in_channels == 3),
        grayscale=(is_grayscale_dataset and in_channels == 1),
    )

    # Custom folder path
    data_path = Path(dataset_name)
    if data_path.is_dir():
        return _load_custom_folder(data_path, train_transform, val_transform, train_config, data_config)

    # HuggingFace Hub vision dataset (`hf://...`)
    if dataset_name.startswith("hf://"):
        return _load_hf_vision_dataset(dataset_name[5:], train_transform, val_transform, bs, train_config)

    test_dataset = None

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=val_transform)
        metadata.setdefault("num_classes", 10)
        metadata.setdefault("class_names", list(train_dataset.classes))

    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=val_transform)
        metadata.setdefault("num_classes", 100)
        metadata.setdefault("class_names", list(train_dataset.classes))

    elif dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=train_transform)
        val_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=val_transform)
        metadata.setdefault("num_classes", 10)

    elif dataset_name in ("fashion_mnist", "fashion-mnist"):
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=train_transform)
        val_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=val_transform)
        metadata.setdefault("num_classes", 10)

    elif dataset_name in ("imagenet", "imagenet-1k"):
        imagenet_path = Path(data_dir) / "imagenet"
        if not imagenet_path.exists():
            raise ValueError(
                f"ImageNet not found at {imagenet_path}. "
                "Download ImageNet manually and place it in <data_dir>/imagenet"
            )
        train_dataset = datasets.ImageFolder(root=str(imagenet_path / "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(imagenet_path / "val"), transform=val_transform)
        metadata["num_classes"] = len(train_dataset.classes)
        metadata["class_names"] = list(train_dataset.classes)

    elif dataset_name == "synthetic":
        task_name = (getattr(train_config, "task", None) or "classification").lower()
        if task_name == "segmentation":
            sz = data_config.img_size if data_config.img_size and data_config.img_size != 224 else 64
            train_dataset = SyntheticSegmentationDataset(
                num_samples=32, img_size=sz, in_channels=in_channels, num_classes=num_classes
            )
            val_dataset = SyntheticSegmentationDataset(
                num_samples=max(8, bs), img_size=sz, in_channels=in_channels, num_classes=num_classes
            )
        elif task_name == "detection":
            sz = data_config.img_size if data_config.img_size and data_config.img_size != 224 else 64
            train_dataset = SyntheticDetectionDataset(
                num_samples=24, img_size=sz, in_channels=in_channels, num_classes=num_classes
            )
            val_dataset = SyntheticDetectionDataset(
                num_samples=max(8, bs), img_size=sz, in_channels=in_channels, num_classes=num_classes
            )
            det_train_loader = DataLoader(
                train_dataset, batch_size=bs, shuffle=data_config.shuffle,
                num_workers=train_config.num_workers, pin_memory=train_config.pin_memory,
                collate_fn=detection_collate_fn,
            )
            det_val_loader = DataLoader(
                val_dataset, batch_size=bs, shuffle=False,
                num_workers=train_config.num_workers, pin_memory=train_config.pin_memory,
                collate_fn=detection_collate_fn,
            )
            return det_train_loader, det_val_loader, None, metadata
        else:
            train_dataset = SyntheticDataset(
                num_samples=128,
                img_size=target_img_size,
                in_channels=in_channels,
                num_classes=num_classes,
                task=task_name,
            )
            val_dataset = SyntheticDataset(
                num_samples=max(32, bs * 2),
                img_size=target_img_size,
                in_channels=in_channels,
                num_classes=num_classes,
                task=task_name,
            )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_PRESETS.keys()) + ['synthetic']} or a local folder path"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=data_config.shuffle,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=data_config.drop_last,
    )
    val_loader: Optional[DataLoader] = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    ) if val_dataset is not None else None
    test_loader: Optional[DataLoader] = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    ) if test_dataset is not None else None

    return train_loader, val_loader, test_loader, metadata


def load_vision_data(
    dataset_name: str,
    data_config: DataConfig,
    train_config: TrainingConfig,
    download: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Backward-compatible 3-tuple vision data loader."""
    train_l, val_l, test_l, _ = load_vision_data_with_meta(
        dataset_name, data_config, train_config, download=download
    )
    return train_l, val_l, test_l


def _load_hf_vision_dataset(
    dataset_id: str,
    train_transform,
    val_transform,
    bs: int,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """Load an image classification dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets required for hf:// datasets. Install with: pip install datasets")

    ds = load_dataset(dataset_id)
    train_split = ds["train"]
    val_split = ds.get("validation") or ds.get("test")

    class _HFImageDataset(Dataset):
        def __init__(self, hf_ds, transform):
            self.ds = hf_ds
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item["image"].convert("RGB")
            label = item["label"]
            return self.transform(img), label

    features = train_split.features
    label_feat = features.get("label")
    classes = list(label_feat.names) if hasattr(label_feat, "names") else []
    meta = {"num_classes": len(classes) if classes else 10, "class_names": classes}

    train_loader = DataLoader(
        _HFImageDataset(train_split, train_transform),
        batch_size=bs, shuffle=True, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory,
    )
    val_loader: Optional[DataLoader] = DataLoader(
        _HFImageDataset(val_split, val_transform),
        batch_size=bs, shuffle=False, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory,
    ) if val_split is not None else None
    return train_loader, val_loader, None, meta


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transforms(
    img_size: int,
    augmentation: str = "auto",
    to_rgb: bool = False,
    grayscale: bool = False,
):
    """Get training transforms with augmentation."""
    from torchvision import transforms

    prefix = []
    if to_rgb:
        prefix.append(transforms.Grayscale(num_output_channels=3))

    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augmentation == "none":
        core = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    elif augmentation in ("light", "auto"):
        core = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif augmentation == "heavy":
        core = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    elif augmentation == "randaugment":
        core = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ]
    elif augmentation == "trivialaugment":
        core = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
        ]
    elif augmentation == "autoaugment":
        core = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        raise ValueError(
            f"Unknown augmentation: {augmentation}. "
            "Available: none, light, auto, heavy, randaugment, trivialaugment, autoaugment"
        )

    return transforms.Compose(prefix + core)


def get_val_transforms(
    img_size: int,
    to_rgb: bool = False,
    grayscale: bool = False,
):
    """Get validation/test/inference transforms (no augmentation)."""
    from torchvision import transforms

    prefix = []
    if to_rgb:
        prefix.append(transforms.Grayscale(num_output_channels=3))

    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose(
        prefix
        + [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class CustomDatasetWrapper(Dataset):
    """Wrapper for custom user datasets."""

    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.transform:
            item = self.transform(item)
        return item


__all__ = [
    "SyntheticDataset",
    "load_vision_data",
    "load_vision_data_with_meta",
    "get_train_transforms",
    "get_val_transforms",
    "CustomDatasetWrapper",
]
