"""
Core shared infrastructure for kitepy.

Contains:
- Adapters: Universal TaskAdapter protocol & Pillar Registry
- Engine: Universal training loop with callbacks, AMP, EMA, grad accum, accelerate
- Config: Typed dataclasses with nested YAML & kwargs routing
- Callbacks: Training hooks and loggers
- Metrics: Metric tracking
- Utils: Device, EMA, parameter grouping, summary utilities
- Base: BaseModel & compare()
"""

from kitepy.core.adapters import (
    ADAPTER_REGISTRY,
    BinaryClassificationAdapter,
    CausalLMAdapter,
    ClassificationAdapter,
    ContrastiveAdapter,
    CustomStepAdapter,
    DetectionAdapter,
    GenerativeAdapter,
    MultiLabelClassificationAdapter,
    RegressionAdapter,
    SegmentationAdapter,
    SequenceClassificationAdapter,
    TaskAdapter,
    get_adapter,
    get_pillar_data_loader,
    register_pillar,
)
from kitepy.core.base import BaseModel, compare
from kitepy.core.callbacks import (
    Callback,
    CallbackList,
    CSVLogger,
    EarlyStopping,
    LambdaCallback,
    LRMonitor,
    ModelCheckpoint,
    ProgressLogger,
    TensorBoardLogger,
    WandbLogger,
)
from kitepy.core.config import (
    AudioConfig,
    BaseConfig,
    CNNConfig,
    DataConfig,
    ExperimentConfig,
    GenerativeConfig,
    LLMConfig,
    ModelConfig,
    RNNConfig,
    TabularConfig,
    TrainingConfig,
    TransformerConfig,
    VLMConfig,
)
from kitepy.core.engine import Engine
from kitepy.core.logging import get_logger, set_log_level
from kitepy.core.lr_finder import LRFinder
from kitepy.core.metrics import MetricTracker, accuracy, precision_recall_f1
from kitepy.core.planner import ExecutionPlan, ExecutionPlanner, RuntimeGuard
from kitepy.core.serialization import atomic_torch_save, get_lineage_metadata, load_safe_bundle, save_safe_bundle
from kitepy.core.utils import (
    ModelEMA,
    print_device_info,
    print_version_info,
    set_seed,
)

__all__ = [
    # Adapters
    "TaskAdapter",
    "ClassificationAdapter",
    "BinaryClassificationAdapter",
    "MultiLabelClassificationAdapter",
    "RegressionAdapter",
    "CausalLMAdapter",
    "SequenceClassificationAdapter",
    "ContrastiveAdapter",
    "GenerativeAdapter",
    "SegmentationAdapter",
    "DetectionAdapter",
    "CustomStepAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "register_pillar",
    "get_pillar_data_loader",
    # Configs
    "BaseConfig",
    "TrainingConfig",
    "DataConfig",
    "ModelConfig",
    "CNNConfig",
    "TransformerConfig",
    "LLMConfig",
    "VLMConfig",
    "AudioConfig",
    "TabularConfig",
    "GenerativeConfig",
    "RNNConfig",
    "ExperimentConfig",
    # Engine & Base
    "Engine",
    "BaseModel",
    "compare",
    "ModelEMA",
    "get_logger",
    "set_log_level",
    "ExecutionPlan",
    "ExecutionPlanner",
    "RuntimeGuard",
    "atomic_torch_save",
    "get_lineage_metadata",
    "save_safe_bundle",
    "load_safe_bundle",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRMonitor",
    "ProgressLogger",
    "CSVLogger",
    "WandbLogger",
    "TensorBoardLogger",
    "LambdaCallback",
    # Metrics
    "MetricTracker",
    "accuracy",
    "precision_recall_f1",
    # LR Finder & Utils
    "LRFinder",
    "print_device_info",
    "print_version_info",
    "set_seed",
]
