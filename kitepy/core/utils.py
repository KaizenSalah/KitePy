"""
Utility functions for kitepy.

Device detection, logging, model summary, EMA, differential LR parameter groups, reproducibility.
"""

from copy import deepcopy
from pathlib import Path
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device(device: Union[str, int, torch.device] = "auto") -> torch.device:
    """Get the appropriate device for training/inference."""
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if isinstance(device, int):
        if torch.cuda.is_available() and 0 <= device < torch.cuda.device_count():
            return torch.device(f"cuda:{device}")
        raise ValueError(f"GPU {device} not available")

    return torch.device(device)


def get_model_device(
    model: Optional[nn.Module],
    explicit_device: Optional[torch.device] = None,
) -> torch.device:
    """Resolve model device safely, supporting parameterless or dynamically quantized modules."""
    if explicit_device is not None:
        return explicit_device
    if model is not None:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            return first_param.device
        first_buf = next(model.buffers(), None)
        if first_buf is not None:
            return first_buf.device
    return torch.device("cpu")


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    return torch.cuda.device_count()


def get_gpu_memory(device: int = 0) -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.set_device(device)
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def print_device_info():
    """Print information about available devices."""
    print("\n" + "=" * 70)
    print("Device Information")
    print("=" * 70)
    print("CPU: Available")

    if torch.cuda.is_available():
        print("CUDA: Available")
        print(f"  - GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            print(f"  - GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("CUDA: Not available")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available")

    print("=" * 70 + "\n")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# ============================================================================
# PARAMETER / TRANSFER LEARNING UTILITIES
# ============================================================================

def get_head_parameter_names(model: nn.Module) -> Set[str]:
    """
    Accurately identify classifier/head parameter names without false positives
    (e.g., avoids matching `blocks.*.mlp.fc1` inside Vision Transformers!).
    """
    head_names: Set[str] = set()

    # 1. timm models expose get_classifier()
    if hasattr(model, "get_classifier") and callable(model.get_classifier):
        try:
            clf = model.get_classifier()
            if isinstance(clf, nn.Module):
                clf_param_ids = {id(p) for p in clf.parameters()}
                for name, p in model.named_parameters():
                    if id(p) in clf_param_ids:
                        head_names.add(name)
                if head_names:
                    return head_names
        except Exception:
            pass

    # 2. Check common top-level or nested head module attributes
    head_attr_candidates = ("head", "classifier", "fc", "lm_head", "score", "last_linear")
    for attr in head_attr_candidates:
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
            sub_mod = getattr(model, attr)
            for sub_name, _ in sub_mod.named_parameters():
                head_names.add(f"{attr}.{sub_name}" if sub_name else attr)
    if head_names:
        return head_names

    # 3. Fallback: match final segment or penultimate module (e.g. "model.head.weight", NOT "blocks.0.mlp.fc1.weight")
    all_named = list(model.named_parameters())
    for name, _ in all_named:
        parts = name.split(".")
        # Check the owning module name (second-to-last part) or top-level module
        if any(seg in ("classifier", "head", "lm_head", "last_linear") for seg in parts[:-1]):
            head_names.add(name)
        elif len(parts) == 2 and parts[0] in ("fc", "score"):
            head_names.add(name)

    if not head_names and all_named:
        # Final fallback: last module's parameters (e.g. last Linear in Sequential)
        last_prefix = ".".join(all_named[-1][0].split(".")[:-1])
        for name, _ in all_named:
            if name.startswith(last_prefix + ".") or name == last_prefix:
                head_names.add(name)

    return head_names


def get_module_blocks(model: nn.Module) -> List[nn.Module]:
    """Extract ordered logical backbone blocks/layers for gradual unfreezing (excludes head/empty layers)."""
    target = getattr(model, "model", model)
    head_names = get_head_parameter_names(target)
    head_param_ids = {
        id(p) for n, p in target.named_parameters() if n in head_names
    }

    blocks: List[nn.Module] = []
    for child in target.children():
        if isinstance(child, (nn.Sequential, nn.ModuleList)) and len(child) > 0:
            items = list(child.children())
        else:
            items = [child]

        for mod in items:
            params = list(mod.parameters())
            if not params:
                continue  # Skip parameterless modules (Identity, Dropout, Pooling)
            if all(id(p) in head_param_ids for p in params):
                continue  # Skip classifier head (already trainable)
            blocks.append(mod)
    return blocks


def build_parameter_groups(
    model: nn.Module,
    lr: Union[float, Dict[str, float]],
    weight_decay: float = 0.01,
    backbone_lr_mult: Optional[float] = None,
    layer_decay: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Build optimizer parameter groups supporting:
    - Scalar LR
    - Dict LR: {"backbone": 1e-4, "head": 1e-3}
    - Differential LR multiplier: backbone_lr_mult=0.1
    - Layer-wise LR decay: layer_decay=0.75
    - Automatic weight_decay exclusion for bias & LayerNorm/BatchNorm params
    """
    head_names = get_head_parameter_names(model)
    no_decay_keywords = ("bias", "norm", "bn", "ln", "layernorm", "rmsnorm")

    if isinstance(lr, dict):
        head_lr = float(lr.get("head", lr.get("default", 1e-3)))
        backbone_lr = float(lr.get("backbone", head_lr * (backbone_lr_mult or 0.1)))
    else:
        head_lr = float(lr)
        backbone_lr = head_lr * backbone_lr_mult if backbone_lr_mult is not None else head_lr

    # Layer-wise decay detection (matches blocks.<i> or layers.<i> or layer<i>)
    layer_pattern = re.compile(r"(?:blocks|layers|h|encoder\.layer|decoder\.layers)\.(\d+)\b|layer(\d+)\b")
    layer_indices: Dict[str, int] = {}
    max_layer = 0
    if layer_decay is not None:
        for name, _ in model.named_parameters():
            m = layer_pattern.search(name)
            if m:
                idx = int(m.group(1) or m.group(2))
                layer_indices[name] = idx
                max_layer = max(max_layer, idx)

    grouped: Dict[Tuple[float, float], List[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = param.ndim <= 1 or any(k in name.lower() for k in no_decay_keywords)
        wd = 0.0 if is_no_decay else weight_decay

        if name in head_names:
            param_lr = head_lr
        elif layer_decay is not None and name in layer_indices:
            depth_from_top = (max_layer - layer_indices[name]) + 1
            param_lr = backbone_lr * (layer_decay ** depth_from_top)
        elif layer_decay is not None:
            # Embeddings / stem get lowest LR
            param_lr = backbone_lr * (layer_decay ** (max_layer + 2))
        else:
            param_lr = backbone_lr

        key = (param_lr, wd)
        grouped.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": p_lr, "weight_decay": p_wd}
        for (p_lr, p_wd), params in grouped.items()
    ]


# ============================================================================
# MODEL EMA (Exponential Moving Average)
# ============================================================================

class ModelEMA:
    """Maintains an exponential moving average of model parameters with warmup decay."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        param_names = {n for n, _ in model.named_parameters()}
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if k in msd:
                model_v = msd[k].detach().to(v.device)
                if k in param_names and v.dtype.is_floating_point:
                    v.mul_(d).add_(model_v, alpha=1.0 - d)
                else:
                    v.copy_(model_v)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]):
        self.decay = state.get("decay", self.decay)
        self.num_updates = state.get("num_updates", 0)
        if "shadow" in state:
            self.shadow.load_state_dict(state["shadow"])


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def print_model_summary(model: nn.Module, input_size: Optional[tuple] = None):
    """Print detailed model summary."""
    print("\n" + "=" * 70)
    print("Model Summary")
    print("=" * 70)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print(f"Non-trainable params:   {non_trainable_params:,}")
    print(f"Model size:             {get_model_size_mb(model):.2f} MB")
    print("\nArchitecture:")
    print(model)
    print("=" * 70 + "\n")


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Simple logger for training progress."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        formatted = f"[{level}] {message}"
        print(formatted)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")

    def info(self, message: str):
        self.log(message, "INFO")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def error(self, message: str):
        self.log(message, "ERROR")


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def get_checkpoint_path(
    checkpoint_dir: str,
    model_name: str,
    epoch: int,
    metric_name: str = "loss",
    metric_value: float = 0.0,
) -> str:
    """Generate checkpoint path with naming convention."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_epoch{epoch}_{metric_name}{metric_value:.4f}.pt"
    return str(Path(checkpoint_dir) / filename)


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_top_k: int = 3,
    metric_name: str = "loss",
    lower_is_better: bool = True,
):
    """Keep only top-k checkpoints based on metric."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists() or keep_top_k <= 0:
        return

    checkpoints = list(ckpt_dir.glob(f"*_{metric_name}*.pt"))
    if len(checkpoints) <= keep_top_k:
        return

    checkpoint_metrics = []
    for ckpt in checkpoints:
        try:
            metric_str = ckpt.stem.split(f"{metric_name}")[-1]
            metric_value = float(metric_str.split("_")[0])
            checkpoint_metrics.append((ckpt, metric_value))
        except Exception:
            continue

    checkpoint_metrics.sort(key=lambda x: x[1], reverse=not lower_is_better)

    for ckpt, _ in checkpoint_metrics[keep_top_k:]:
        ckpt.unlink()
        from .logging import get_logger
        get_logger("kitepy.utils").debug(f"Deleted old checkpoint: {ckpt.name}")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def print_training_header(config):
    """Print training configuration header."""
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Epochs:           {config.epochs}")
    if getattr(config, "max_steps", None):
        print(f"Max steps:        {config.max_steps}")
    print(f"Batch size:       {config.batch_size}")
    if getattr(config, "gradient_accumulation_steps", 1) > 1:
        print(f"Grad accum steps: {config.gradient_accumulation_steps}")
    print(f"Learning rate:    {config.lr}")
    print(f"Optimizer:        {config.optimizer}")
    print(f"Scheduler:        {config.scheduler}")
    print(f"Mixed precision:  {config.mixed_precision}")
    print(f"Device:           {config.devices}")
    if getattr(config, "ema", False):
        print(f"EMA:              True (decay={config.ema_decay})")
    print("=" * 70 + "\n")


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    train_acc: Optional[float] = None,
    val_acc: Optional[float] = None,
    lr: Optional[float] = None,
    epoch_time: Optional[float] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
):
    """Print summary for one epoch."""
    metrics = [f"Train Loss: {train_loss:.4f}"]

    if val_loss is not None:
        metrics.append(f"Val Loss: {val_loss:.4f}")
    if train_acc is not None:
        metrics.append(f"Train Acc: {train_acc:.2f}%")
    if val_acc is not None:
        metrics.append(f"Val Acc: {val_acc:.2f}%")
    if extra_metrics:
        skip = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "time"}
        for k, v in extra_metrics.items():
            if k not in skip and isinstance(v, (int, float)):
                metrics.append(f"{k}: {v:.4f}")
    if lr is not None:
        metrics.append(f"LR: {lr:.2e}")
    if epoch_time is not None:
        metrics.append(f"Time: {format_time(epoch_time)}")

    print(f"Epoch {epoch}/{total_epochs} | " + " | ".join(metrics))


# ============================================================================
# MEMORY & HELPERS
# ============================================================================

def get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    stats = {
        "cpu_memory_mb": 0,
        "gpu_memory_mb": 0,
        "gpu_memory_reserved_mb": 0,
    }
    if torch.cuda.is_available():
        stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
    return stats


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_auto_value(value: Union[str, int, float], resolver_fn) -> Union[int, float]:
    """Resolve 'auto' values in config."""
    if isinstance(value, str) and value.lower() == "auto":
        return resolver_fn()
    return value


def validate_path(path: Union[str, Path], create: bool = True) -> Path:
    """Validate and optionally create a path."""
    p = Path(path)
    if create and not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def print_version_info():
    """Print library and dependency versions."""
    print("\n" + "=" * 70)
    print("Environment Information")
    print("=" * 70)
    print(f"Python:        {os.sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")

    for mod_name, label in [
        ("timm", "timm"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
    ]:
        try:
            mod = __import__(mod_name)
            print(f"{label + ':':<15}{getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{label + ':':<15}Not installed")

    print("=" * 70 + "\n")


__all__ = [
    "get_device",
    "get_model_device",
    "get_num_gpus",
    "get_gpu_memory",
    "print_device_info",
    "set_seed",
    "get_head_parameter_names",
    "get_module_blocks",
    "build_parameter_groups",
    "ModelEMA",
    "count_parameters",
    "get_model_size_mb",
    "print_model_summary",
    "Logger",
    "get_checkpoint_path",
    "cleanup_old_checkpoints",
    "format_time",
    "get_lr",
    "print_training_header",
    "print_epoch_summary",
    "get_memory_usage",
    "clear_gpu_memory",
    "resolve_auto_value",
    "validate_path",
    "print_version_info",
]
