"""
Callback system for training hooks.

Provides extensible hooks for custom behavior during training, now fully wired into Engine.
"""

import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
import torch


class Callback:
    """Base callback class. Override methods to customize training behavior."""

    def on_train_start(self, trainer: Any, **kwargs):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: Any, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: Any, batch_idx: int, **kwargs):
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass

    def on_validation_start(self, trainer: Any, **kwargs):
        """Called at the start of validation."""
        pass

    def on_validation_end(self, trainer: Any, metrics: Dict[str, Any], **kwargs):
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")


class ModelCheckpoint(Callback):
    """Save model checkpoints."""

    def __init__(
        self,
        filepath: str = "checkpoints/model_{epoch:02d}.pt",
        monitor: str = "val_loss",
        save_best_only: bool = False,
        save_top_k: int = 3,
        mode: str = "min",
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: List[str] = []

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        current = metrics.get(self.monitor)
        if current is None:
            current = metrics.get("train_loss")
        if current is None:
            return

        if self.save_best_only:
            if self.mode == "min":
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if not improved:
                return
            self.best_value = current

        val_loss_val = metrics.get("val_loss")
        train_loss_val = metrics.get("train_loss")
        path = self.filepath.format(
            epoch=epoch,
            val_loss=val_loss_val if val_loss_val is not None else 0.0,
            train_loss=train_loss_val if train_loss_val is not None else 0.0,
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)
        print(f"  → Checkpoint saved: {path}")

        if len(self.saved_checkpoints) > self.save_top_k:
            old_path = self.saved_checkpoints.pop(0)
            if Path(old_path).exists():
                Path(old_path).unlink()


class LRMonitor(Callback):
    """Log learning rate at each epoch."""

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"  → Learning rate: {lr:.2e}")


class ProgressLogger(Callback):
    """Log training progress every N steps."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.running_loss = 0.0
        self.batch_count = 0

    def on_epoch_start(self, trainer, epoch, **kwargs):
        self.running_loss = 0.0
        self.batch_count = 0

    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        self.running_loss += loss
        self.batch_count += 1

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            avg_loss = self.running_loss / self.batch_count
            print(f"  Step {batch_idx + 1}: Loss = {avg_loss:.4f}")
            self.running_loss = 0.0
            self.batch_count = 0


class CSVLogger(Callback):
    """Zero-dependency experiment metrics logger writing to a CSV file."""

    def __init__(self, log_dir: str = "logs", filename: str = "metrics.csv"):
        self.log_dir = Path(log_dir)
        self.filepath = self.log_dir / filename
        self._fieldnames: Optional[List[str]] = None

    def on_train_start(self, trainer, **kwargs):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fieldnames = None

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        row = {"epoch": epoch}
        for k, v in metrics.items():
            if v is not None:
                row[k] = v

        write_header = not self.filepath.exists() or self._fieldnames is None
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class WandbLogger(Callback):
    """Log metrics to Weights & Biases."""

    def __init__(
        self,
        project: str = "kitepy",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.project = project
        self.name = name
        self.config = config
        self._wandb = None

    def on_train_start(self, trainer, **kwargs):
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config or trainer.config.to_dict(),
            )
            wandb.watch(trainer.model)
            print(f"✓ WandB initialized: {self.project}")
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._wandb:
            clean = {k: v for k, v in metrics.items() if v is not None}
            self._wandb.log(clean, step=epoch)

    def on_train_end(self, trainer, **kwargs):
        if self._wandb:
            self._wandb.finish()


class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self._writer = None

    def on_train_start(self, trainer, **kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self.log_dir)
            print(f"✓ TensorBoard logging to: {self.log_dir}")
        except ImportError:
            print("⚠️  TensorBoard not available")

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._writer:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    self._writer.add_scalar(name, value, epoch)

    def on_train_end(self, trainer, **kwargs):
        if self._writer:
            self._writer.close()


class LambdaCallback(Callback):
    """Create callback from callable hooks."""

    def __init__(
        self,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_epoch_start: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
        on_batch_start: Optional[Callable] = None,
        on_batch_end: Optional[Callable] = None,
        on_validation_start: Optional[Callable] = None,
        on_validation_end: Optional[Callable] = None,
    ):
        self._on_train_start = on_train_start
        self._on_train_end = on_train_end
        self._on_epoch_start = on_epoch_start
        self._on_epoch_end = on_epoch_end
        self._on_batch_start = on_batch_start
        self._on_batch_end = on_batch_end
        self._on_validation_start = on_validation_start
        self._on_validation_end = on_validation_end

    def on_train_start(self, trainer, **kwargs):
        if self._on_train_start:
            self._on_train_start(trainer, **kwargs)

    def on_train_end(self, trainer, **kwargs):
        if self._on_train_end:
            self._on_train_end(trainer, **kwargs)

    def on_epoch_start(self, trainer, epoch, **kwargs):
        if self._on_epoch_start:
            self._on_epoch_start(trainer, epoch, **kwargs)

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._on_epoch_end:
            self._on_epoch_end(trainer, epoch, metrics, **kwargs)

    def on_batch_start(self, trainer, batch_idx, **kwargs):
        if self._on_batch_start:
            self._on_batch_start(trainer, batch_idx, **kwargs)

    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        if self._on_batch_end:
            self._on_batch_end(trainer, batch_idx, loss, **kwargs)

    def on_validation_start(self, trainer, **kwargs):
        if self._on_validation_start:
            self._on_validation_start(trainer, **kwargs)

    def on_validation_end(self, trainer, metrics, **kwargs):
        if self._on_validation_end:
            self._on_validation_end(trainer, metrics, **kwargs)


class CallbackList:
    """Manage and dispatch multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Union[Callback, str]]] = None):
        self.callbacks: List[Callback] = []
        for cb in (callbacks or []):
            self.add(cb)

    def add(self, callback: Union[Callback, str]):
        if isinstance(callback, str):
            name = callback.lower()
            shortcuts = {
                "early_stopping": EarlyStopping,
                "lr_monitor": LRMonitor,
                "progress": ProgressLogger,
                "csv": CSVLogger,
                "tensorboard": TensorBoardLogger,
                "wandb": WandbLogger,
                "checkpoint": ModelCheckpoint,
            }
            if name not in shortcuts:
                raise ValueError(f"Unknown callback shortcut '{callback}'. Available: {list(shortcuts.keys())}")
            self.callbacks.append(shortcuts[name]())
        else:
            self.callbacks.append(callback)

    def on_train_start(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_start(trainer, **kwargs)

    def on_train_end(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_end(trainer, **kwargs)

    def on_epoch_start(self, trainer, epoch, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch, **kwargs)

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics, **kwargs)

    def on_batch_start(self, trainer, batch_idx, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_start(trainer, batch_idx, **kwargs)

    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss, **kwargs)

    def on_validation_start(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_validation_start(trainer, **kwargs)

    def on_validation_end(self, trainer, metrics, **kwargs):
        for cb in self.callbacks:
            cb.on_validation_end(trainer, metrics, **kwargs)

    def should_stop(self) -> bool:
        """Check if any callback requests stopping."""
        return any(getattr(cb, "should_stop", False) for cb in self.callbacks)


__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRMonitor",
    "ProgressLogger",
    "CSVLogger",
    "WandbLogger",
    "TensorBoardLogger",
    "LambdaCallback",
    "CallbackList",
]
