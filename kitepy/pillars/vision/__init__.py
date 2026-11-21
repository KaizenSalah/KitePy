"""
Vision Pillar - Computer Vision Models & Data Pipeline
"""

from kitepy.core.adapters import register_pillar
from kitepy.core.presets import DATASET_PRESETS, MODEL_PRESETS
from kitepy.pillars.vision.data import (
    get_train_transforms,
    get_val_transforms,
    load_vision_data,
    load_vision_data_with_meta,
)
from kitepy.pillars.vision.models import CNN, SimpleUNet, TinyDetector, Transformer, Vision

# Register vision pillar with core BaseModel registry
register_pillar("vision", data_loader=load_vision_data_with_meta, default_task="classification")

__all__ = [
    "CNN",
    "Transformer",
    "Vision",
    "SimpleUNet",
    "TinyDetector",
    "load_vision_data",
    "load_vision_data_with_meta",
    "get_train_transforms",
    "get_val_transforms",
    "DATASET_PRESETS",
    "MODEL_PRESETS",
]
