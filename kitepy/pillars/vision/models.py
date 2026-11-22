"""
Vision model APIs - CNN, Transformer, and Vision.

Supports:
- 800+ timm models + custom nn.Module
- Real-world .predict() on Tensors, PIL Images, NumPy arrays, file paths, or lists
- Explicit multi-task Vision(task="classification" | "binary" | "multilabel" | "regression" | "segmentation" | "detection")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kitepy.core.adapters import get_adapter
from kitepy.core.base import BaseModel
from kitepy.core.config import CNNConfig, TransformerConfig
from .data import get_val_transforms
from .wrappers import create_model


# ============================================================================
# BUILT-IN SEGMENTATION & DETECTION LIGHTWEIGHT MODELS
# ============================================================================

class SimpleUNet(nn.Module):
    """Lightweight built-in U-Net for fast semantic segmentation out of the box."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_ch: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        u = self.up(e2)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec(torch.cat([u, e1], dim=1))


class TinyDetector(nn.Module):
    """
    Lightweight built-in object detector following the torchvision detection contract:
    - training: forward(images, targets) -> {"loss_box": Tensor, "loss_cls": Tensor}
    - eval:     forward(images) -> [{"boxes": Tensor[N,4], "labels": Tensor[N], "scores": Tensor[N]}, ...]
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.box_head = nn.Linear(64, 4)
        self.cls_head = nn.Linear(64, num_classes)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        x = torch.stack(images, dim=0) if isinstance(images, (list, tuple)) else images
        feats = self.backbone(x)
        pred_boxes = torch.sigmoid(self.box_head(feats)) * float(x.shape[-1])
        pred_logits = self.cls_head(feats)

        if self.training and targets is not None:
            gt_boxes = []
            gt_labels = []
            for t in targets:
                if len(t["boxes"]) > 0:
                    gt_boxes.append(t["boxes"][0].float())
                    gt_labels.append(t["labels"][0].long())
                else:
                    gt_boxes.append(torch.zeros(4, device=x.device))
                    gt_labels.append(torch.tensor(0, device=x.device, dtype=torch.long))
            gt_b = torch.stack(gt_boxes, dim=0)
            gt_l = torch.stack(gt_labels, dim=0)
            return {
                "loss_box": F.smooth_l1_loss(pred_boxes, gt_b),
                "loss_cls": F.cross_entropy(pred_logits, gt_l),
            }

        probs = torch.softmax(pred_logits, dim=-1)
        scores, labels = probs.max(dim=-1)
        results = []
        for i in range(x.size(0)):
            # Ensure valid [x1, y1, x2, y2] ordering
            b = pred_boxes[i]
            x1, y1 = torch.minimum(b[0], b[2]), torch.minimum(b[1], b[3])
            x2, y2 = torch.maximum(b[0], b[2]) + 1.0, torch.maximum(b[1], b[3]) + 1.0
            results.append({
                "boxes": torch.stack([x1, y1, x2, y2]).unsqueeze(0),
                "labels": labels[i].unsqueeze(0),
                "scores": scores[i].unsqueeze(0),
            })
        return results


# ============================================================================
# VISION MIXIN & PUBLIC CLASSES
# ============================================================================

class _VisionMixin:
    """Shared real-world image preprocessing for CNN, Transformer, and Vision."""

    def _preprocess_single_image(self, item: Any, img_size: int, in_channels: int) -> torch.Tensor:
        from PIL import Image

        if isinstance(item, torch.Tensor):
            return item

        if isinstance(item, (str, Path)):
            img = Image.open(item).convert("L" if in_channels == 1 else "RGB")
            return get_val_transforms(img_size, grayscale=(in_channels == 1))(img)

        if isinstance(item, Image.Image):
            img = item.convert("L" if in_channels == 1 else "RGB")
            return get_val_transforms(img_size, grayscale=(in_channels == 1))(img)

        if isinstance(item, np.ndarray):
            arr = item
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                img = Image.fromarray(arr).convert("L" if in_channels == 1 else "RGB")
                return get_val_transforms(img_size, grayscale=(in_channels == 1))(img)
            return torch.from_numpy(arr).float()

        if isinstance(item, (list, tuple)):
            t = torch.as_tensor(item, dtype=torch.float32)
            return t

        raise TypeError(f"Unsupported input type for vision prediction: {type(item)}")

    def _preprocess_input(self, input_data: Any) -> torch.Tensor:
        img_size = (
            getattr(self.data_config, "img_size", None)
            or getattr(self.config, "img_size", 224)
            or 224
        )
        in_channels = (
            getattr(self.data_config, "in_channels", None)
            or getattr(self.config, "in_channels", 3)
            or 3
        )

        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 3:
                return input_data.unsqueeze(0)
            return input_data

        if isinstance(input_data, np.ndarray) and input_data.ndim == 4:
            return torch.from_numpy(input_data).float()

        if isinstance(input_data, list):
            if input_data and isinstance(input_data[0], (int, float, list, tuple)):
                t = torch.as_tensor(input_data, dtype=torch.float32)
                return t.unsqueeze(0) if t.ndim == 3 else t
            tensors = [self._preprocess_single_image(item, img_size, in_channels) for item in input_data]
            return torch.stack(tensors, dim=0)

        single = self._preprocess_single_image(input_data, img_size, in_channels)
        if single.ndim == 3:
            single = single.unsqueeze(0)
        return single


class CNN(_VisionMixin, BaseModel):
    """Convolutional Neural Network for image tasks."""

    modality = "vision"
    default_config_class = CNNConfig
    default_task = "classification"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "resnet18",
        config: Optional[Union[Dict, CNNConfig, str]] = None,
        **kwargs,
    ):
        if isinstance(model, str):
            from .wrappers import validate_vision_model_name
            validate_vision_model_name(model)
        super().__init__(model, config, **kwargs)

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        else:
            self.model = create_model(
                model_name_or_module=self.model_name,
                modality=self.modality,
                config=self.config.to_dict(),
                pretrained=self.config.pretrained,
            )

        print(f"✓ Created CNN model: {self.model_name}")
        print(f"  - Architecture: {getattr(self.config, 'arch', 'custom')}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


class Transformer(_VisionMixin, BaseModel):
    """Vision Transformer for image tasks (ViT, DeiT, Swin, etc.)."""

    modality = "vision"
    default_config_class = TransformerConfig
    default_task = "classification"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "vit_base",
        config: Optional[Union[Dict, TransformerConfig, str]] = None,
        **kwargs,
    ):
        if isinstance(model, str):
            from .wrappers import validate_vision_model_name
            validate_vision_model_name(model)
        super().__init__(model, config, **kwargs)

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        else:
            self.model = create_model(
                model_name_or_module=self.model_name,
                modality=self.modality,
                config=self.config.to_dict(),
                pretrained=self.config.pretrained,
            )

        print(f"✓ Created Transformer model: {self.model_name}")
        print(f"  - Architecture: {getattr(self.config, 'arch', 'custom')}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


class Vision(_VisionMixin, BaseModel):
    """
    General Vision API with explicit task specification:
    - "classification": multi-class classification
    - "binary": binary classification (single logit + BCE)
    - "multilabel": multi-label classification (BCE)
    - "regression": image regression (MSE/MAE)
    - "segmentation": semantic segmentation (`unet`, `fcn_resnet50`, `deeplabv3_resnet50`, etc.)
    - "detection": object detection (`tiny_detector`, `fasterrcnn_resnet50_fpn`, `ssdlite320_mobilenet_v3_large`, etc.)
    """

    modality = "vision"
    default_config_class = CNNConfig

    TASK_DEFAULTS = {
        "classification": ("classification", "cross_entropy"),
        "binary": ("binary", "bce"),
        "multilabel": ("multilabel", "bce"),
        "regression": ("regression", "mse"),
        "segmentation": ("segmentation", "cross_entropy"),
        "detection": ("detection", "cross_entropy"),
    }

    def __init__(
        self,
        task: str = "classification",
        model: Optional[Union[str, nn.Module]] = None,
        config: Optional[Union[Dict, CNNConfig, str]] = None,
        **kwargs,
    ):
        self.task = task.lower()
        if self.task not in self.TASK_DEFAULTS:
            raise NotImplementedError(
                f"Task '{task}' not implemented. Supported: {list(self.TASK_DEFAULTS.keys())}."
            )

        if model is None:
            if self.task == "segmentation":
                model = "unet"
            elif self.task == "detection":
                model = "tiny_detector"
            else:
                model = "resnet18"

        adapter_name, default_loss = self.TASK_DEFAULTS[self.task]
        kwargs.setdefault("loss", default_loss)
        kwargs.setdefault("task", adapter_name)

        super().__init__(model, config, adapter=get_adapter(adapter_name), **kwargs)
        if "num_classes" not in self._explicit_model_fields:
            if self.task in ("binary", "regression"):
                self.config.num_classes = 1
            elif self.task == "segmentation":
                self.config.num_classes = 2
        self.default_task = adapter_name

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        elif self.task == "segmentation":
            self.model = self._build_segmentation_model()
        elif self.task == "detection":
            self.model = self._build_detection_model()
        else:
            self.model = create_model(
                model_name_or_module=self.model_name,
                modality=self.modality,
                config=self.config.to_dict(),
                pretrained=self.config.pretrained,
            )

        print(f"✓ Created Vision model for {self.task}: {self.model_name}")

    def _build_segmentation_model(self) -> nn.Module:
        if self.model_name == "unet":
            return SimpleUNet(in_channels=self.config.in_channels, num_classes=self.config.num_classes)
        from torchvision.models import segmentation as tv_seg
        if hasattr(tv_seg, self.model_name):
            fn = getattr(tv_seg, self.model_name)
            weights = "DEFAULT" if self.config.pretrained else None
            return fn(weights=weights, num_classes=self.config.num_classes)
        raise ValueError(f"Unknown segmentation model: {self.model_name}")

    def _build_detection_model(self) -> nn.Module:
        if self.model_name == "tiny_detector":
            return TinyDetector(in_channels=self.config.in_channels, num_classes=self.config.num_classes)
        from torchvision.models import detection as tv_det
        if hasattr(tv_det, self.model_name):
            fn = getattr(tv_det, self.model_name)
            weights = "DEFAULT" if self.config.pretrained else None
            return fn(weights=weights, num_classes=self.config.num_classes)
        raise ValueError(f"Unknown detection model: {self.model_name}")


__all__ = ["CNN", "Transformer", "Vision", "SimpleUNet", "TinyDetector"]
