"""
Learning Rate Finder - Auto-tune learning rate via LR Range Test.

Supports any modality/batch format via TaskAdapter.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .adapters import TaskAdapter, ClassificationAdapter


class LRFinder:
    """Learning rate finder to automatically discover optimal learning rate."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        adapter: Optional[TaskAdapter] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.adapter = adapter or ClassificationAdapter()

        # Save original state cleanly via deepcopy
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.lrs: List[float] = []
        self.losses: List[float] = []

    def find(
        self,
        train_loader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        beta: float = 0.98,
    ) -> Tuple[list, list]:
        """Run learning rate range test."""
        self.model.train()
        self.lrs.clear()
        self.losses.clear()

        num_steps = max(2, num_steps)
        lr_schedule = np.geomspace(min_lr, max_lr, num_steps)

        best_loss = float("inf")
        avg_loss = 0.0

        print(f"\n🔍 Running LR Finder ({num_steps} steps)...")
        iterator = iter(train_loader)
        try:
            first_batch = next(iterator)
        except StopIteration as e:
            raise ValueError("Training DataLoader is empty (0 batches). Cannot run LRFinder.") from e

        try:
            for i, lr in enumerate(lr_schedule):
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = float(lr)

                if i == 0:
                    raw_batch = first_batch
                else:
                    try:
                        raw_batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_loader)
                        try:
                            raw_batch = next(iterator)
                        except StopIteration as e:
                            raise ValueError("Training DataLoader became empty during LRFinder.") from e

                batch = self.adapter.prepare_batch(raw_batch, self.device)

                self.optimizer.zero_grad()
                loss, _, _ = self.adapter.forward_and_loss(
                    self.model, batch, self.criterion, engine=None
                )

                loss_val = float(loss.item())
                avg_loss = beta * avg_loss + (1 - beta) * loss_val
                smoothed_loss = avg_loss / (1 - beta ** (i + 1))

                self.lrs.append(float(lr))
                self.losses.append(float(smoothed_loss))

                if i > 10 and (smoothed_loss > 4 * best_loss or np.isnan(smoothed_loss)):
                    print(f"  Stopping early at step {i} (loss diverged)")
                    break

                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss

                loss.backward()
                self.optimizer.step()

                if (i + 1) % 20 == 0:
                    print(f"  Step {i + 1}/{num_steps} | LR: {lr:.2e} | Loss: {smoothed_loss:.4f}")
        finally:
            # Always restore original weights & optimizer state even on interruption/error
            self.model.load_state_dict(self.model_state)
            self.optimizer.load_state_dict(self.optimizer_state)

        print("✓ LR Finder complete")
        return self.lrs, self.losses

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """Suggest optimal learning rate from collected loss curve."""
        if len(self.losses) == 0:
            raise RuntimeError("Must run find() first")

        if len(self.losses) <= skip_start + skip_end + 2:
            min_idx = int(np.argmin(self.losses))
            suggested_lr = float(self.lrs[max(0, min_idx - 1)])
            print(f"\n💡 Suggested learning rate: {suggested_lr:.2e}")
            return suggested_lr

        end_idx = -skip_end if skip_end > 0 else None
        losses = self.losses[skip_start:end_idx]
        lrs = self.lrs[skip_start:end_idx]

        gradients = np.gradient(losses)
        min_grad_idx = int(np.argmin(gradients))
        suggested_idx = max(0, min_grad_idx - 3)
        suggested_lr = float(lrs[suggested_idx])

        print(f"\n💡 Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr

    def plot(self, save_path: Optional[str] = None):
        """Plot learning rate vs loss curve."""
        if plt is None:
            print("Warning: matplotlib not installed, skipping plot.")
            return
        if len(self.losses) == 0:
            raise RuntimeError("Must run find() first")

        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, alpha=0.3)

        suggested_lr = self.suggest_lr()
        plt.axvline(suggested_lr, color="r", linestyle="--", label=f"Suggested: {suggested_lr:.2e}")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()


__all__ = ["LRFinder"]
