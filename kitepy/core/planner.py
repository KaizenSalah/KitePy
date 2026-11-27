"""
Constraint-Aware Execution Planner & Runtime Fault Recovery.

Provides:
1. Pre-flight ExecutionPlanner: estimates model/optimizer/activation memory footprint
   and solves for (micro_batch_size, gradient_accumulation_steps, amp_dtype, gradient_checkpointing)
   to preserve the target effective batch size under a hardware or explicit memory budget.
2. RuntimeGuard: handles CUDA/runtime OOM backoff (micro-batch splitting while preserving
   effective batch mathematics) and NaN/Inf loss or gradient recovery.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .adapters import TaskAdapter, infer_batch_size
from .logging import get_logger

logger = get_logger("kitepy.planner")


@dataclass
class ExecutionPlan:
    """Solved hardware-aware execution plan."""

    target_effective_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    amp_enabled: bool
    amp_dtype: str  # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool
    estimated_peak_memory_mb: float
    memory_budget_mb: float
    world_size: int = 1
    reason: str = "nominal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_static_memory_mb(model: nn.Module, bytes_per_param_state: float = 16.0) -> float:
    """
    Estimate static parameter + gradient + optimizer state memory in MB.
    AdamW fp32 states typically use ~16 bytes per trainable param (4B param + 4B grad + 8B m/v)
    plus ~4 bytes per frozen param.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_bytes = trainable * bytes_per_param_state + frozen * 4.0
    return float(total_bytes / (1024.0 * 1024.0))


def _slice_batch(batch: Any, start: int, end: int, batch_size: Optional[int] = None) -> Any:
    """Recursively slice tensors, nested dicts/mappings, or collated tuple/list batches along batch dim 0."""
    bs = batch_size if batch_size is not None else infer_batch_size(batch)
    if isinstance(batch, torch.Tensor):
        return batch[start:end] if batch.ndim > 0 else batch
    if isinstance(batch, Mapping):
        sliced = {}
        for k, v in batch.items():
            if isinstance(v, (torch.Tensor, Mapping)):
                sliced[k] = _slice_batch(v, start, end, batch_size=bs)
            elif isinstance(v, (list, tuple)) and len(v) == bs:
                sliced[k] = type(v)(v[start:end])
            else:
                sliced[k] = v
        return sliced
    if isinstance(batch, (tuple, list)):
        # Detection-style (list_of_images, list_of_target_dicts)
        if (
            len(batch) == 2
            and isinstance(batch[0], list)
            and isinstance(batch[1], list)
            and len(batch[1]) > 0
            and isinstance(batch[1][0], dict)
        ):
            det_items = [batch[0][start:end], batch[1][start:end]]
            return tuple(det_items) if isinstance(batch, tuple) else det_items
        # List of per-sample dicts
        if len(batch) > 0 and isinstance(batch[0], dict) and "input_ids" not in batch[0]:
            return batch[start:end]
        # Collated (inputs, targets) / [inputs, targets] from PyTorch default_collate
        items = [_slice_batch(v, start, end, batch_size=bs) for v in batch]
        return tuple(items) if isinstance(batch, tuple) else items
    return batch


def split_batch_into_micro_batches(batch: Any, num_chunks: int) -> List[Any]:
    """Split a batch into `num_chunks` micro-batches along batch dimension 0."""
    bs = infer_batch_size(batch)
    if bs <= 1 or num_chunks <= 1:
        return [batch]
    chunks = []
    step = max(1, math.ceil(bs / num_chunks))
    for start in range(0, bs, step):
        end = min(bs, start + step)
        chunks.append(_slice_batch(batch, start, end, batch_size=bs))
    return chunks


class _SingletonBatchNormGuard:
    """Temporarily switch active BatchNorm layers to eval() during bs==1 forward passes."""

    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, model: nn.Module, batch_size: int):
        self.model = model
        self.active = batch_size <= 1
        self._switched: List[nn.Module] = []

    def __enter__(self):
        if self.active:
            for mod in self.model.modules():
                if isinstance(mod, self._BN_TYPES) and mod.training:
                    mod.eval()
                    self._switched.append(mod)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mod in self._switched:
            mod.train()
        self._switched.clear()
        return False


class ExecutionPlanner:
    """
    Constraint-aware pre-flight execution planner.

    Measures/estimates per-sample activation memory and solves for micro-batch size,
    gradient accumulation steps, AMP dtype, and gradient checkpointing under a memory budget.
    """

    def __init__(self, safety_factor: float = 0.85):
        self.safety_factor = safety_factor

    def resolve_memory_budget_mb(
        self,
        device: torch.device,
        explicit_budget_mb: Optional[float] = None,
    ) -> float:
        if explicit_budget_mb is not None and explicit_budget_mb > 0:
            return float(explicit_budget_mb)
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device.index or 0)
                return float((free_bytes / (1024.0 * 1024.0)) * self.safety_factor)
            except Exception:
                return 8192.0 * self.safety_factor
        # Nominal budget on CPU/MPS
        return 8192.0

    def _probe_activation_mb_per_sample(
        self,
        model: nn.Module,
        sample_batch: Any,
        adapter: TaskAdapter,
        criterion: nn.Module,
        device: torch.device,
        amp_enabled: bool = False,
        amp_dtype: str = "fp32",
    ) -> float:
        """Estimate activation memory per sample using a single dry pass or tensor-size heuristic."""
        bs = max(1, infer_batch_size(sample_batch))
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                cpu_snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                micro = _slice_batch(sample_batch, 0, 1)
                prep = adapter.prepare_batch(micro, device)
                ac_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
                model.train()

                # Warm up cuDNN / allocator once before measuring delta
                with _SingletonBatchNormGuard(model, 1):
                    with torch.amp.autocast("cuda", dtype=ac_dtype, enabled=amp_enabled):
                        warm_loss, _, _ = adapter.forward_and_loss(model, prep, criterion, engine=None)
                warm_loss.backward()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats(device)
                base_mem = torch.cuda.memory_allocated(device)
                with _SingletonBatchNormGuard(model, 1):
                    with torch.amp.autocast("cuda", dtype=ac_dtype, enabled=amp_enabled):
                        loss, _, _ = adapter.forward_and_loss(model, prep, criterion, engine=None)
                loss.backward()
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                model.zero_grad(set_to_none=True)
                model.load_state_dict({k: v.to(device) for k, v in cpu_snap.items()})
                del cpu_snap
                torch.cuda.empty_cache()
                delta_mb = float(max(1.0, (peak - base_mem) / (1024.0 * 1024.0)))
                return delta_mb
            except Exception:
                pass

        # Deterministic analytical fallback: input tensor bytes + layer activation multiplier
        input_bytes = 0
        def _count_bytes(x):
            nonlocal input_bytes
            if isinstance(x, torch.Tensor):
                input_bytes += x.numel() * max(4, x.element_size())
            elif isinstance(x, dict):
                for v in x.values():
                    _count_bytes(v)
            elif isinstance(x, (tuple, list)):
                for v in x:
                    _count_bytes(v)

        _count_bytes(sample_batch)
        per_sample_input_mb = (input_bytes / bs) / (1024.0 * 1024.0)
        num_layers = max(1, sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear, nn.MultiheadAttention))))
        return max(0.5, per_sample_input_mb * num_layers * 2.0)

    def plan(
        self,
        model: nn.Module,
        device: torch.device,
        target_batch_size: int,
        sample_batch: Any,
        adapter: TaskAdapter,
        criterion: nn.Module,
        current_grad_accum: int = 1,
        mixed_precision: Any = True,
        memory_budget_mb: Optional[float] = None,
        world_size: int = 1,
    ) -> ExecutionPlan:
        """Solve for an OOM-safe ExecutionPlan preserving target effective batch size."""
        budget_mb = self.resolve_memory_budget_mb(device, memory_budget_mb)
        effective_target = max(1, target_batch_size * max(1, current_grad_accum) * max(1, world_size))

        # Choose AMP dtype
        amp_enabled = False
        amp_dtype = "fp32"
        if mixed_precision not in (False, "fp32", "no", None):
            if device.type == "cuda" and torch.cuda.is_available():
                amp_enabled = True
                if mixed_precision == "bf16" or (
                    hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                ):
                    amp_dtype = "bf16"
                else:
                    amp_dtype = "fp16"
            elif mixed_precision == "bf16":
                amp_enabled = True
                amp_dtype = "bf16"

        static_mb = estimate_static_memory_mb(model, bytes_per_param_state=12.0 if amp_enabled else 16.0)
        act_per_sample_mb = self._probe_activation_mb_per_sample(
            model, sample_batch, adapter, criterion, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype
        )
        if amp_enabled and device.type != "cuda":
            act_per_sample_mb *= 0.6

        available_act_mb = max(0.0, budget_mb - static_mb)
        max_micro = max(1, int(available_act_mb // max(0.1, act_per_sample_mb)))
        micro_bs = min(max(1, target_batch_size), max_micro)

        # Solve gradient accumulation steps to preserve target effective batch size
        per_step_global = max(1, micro_bs * max(1, world_size))
        grad_accum = max(1, math.ceil(effective_target / per_step_global))

        use_grad_ckpt = False
        est_peak = static_mb + micro_bs * act_per_sample_mb
        reason = "nominal"

        if micro_bs < target_batch_size:
            reason = f"memory_constrained_split (budget={budget_mb:.1f}MB, est_act={act_per_sample_mb:.2f}MB/sample)"
            if hasattr(model, "gradient_checkpointing_enable") and est_peak > budget_mb * 0.9:
                use_grad_ckpt = True
                est_peak = static_mb + micro_bs * (act_per_sample_mb * 0.4)

        plan = ExecutionPlan(
            target_effective_batch_size=effective_target,
            micro_batch_size=micro_bs,
            gradient_accumulation_steps=grad_accum,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            gradient_checkpointing=use_grad_ckpt,
            estimated_peak_memory_mb=round(est_peak, 2),
            memory_budget_mb=round(budget_mb, 2),
            world_size=world_size,
            reason=reason,
        )
        logger.info(
            f"ExecutionPlan: micro_bs={plan.micro_batch_size}, grad_accum={plan.gradient_accumulation_steps}, "
            f"amp={plan.amp_dtype}, est_peak={plan.estimated_peak_memory_mb:.1f}MB / {plan.memory_budget_mb:.1f}MB"
        )
        return plan


class RuntimeGuard:
    """Tracks and handles runtime OOM backoff & NaN/Inf gradient recovery."""

    def __init__(self, auto_recover: bool = True, max_consecutive_nan_steps: int = 5):
        self.auto_recover = auto_recover
        self.max_consecutive_nan_steps = max_consecutive_nan_steps
        self.oom_recoveries = 0
        self.nan_skips = 0
        self.consecutive_nans = 0

    @staticmethod
    def is_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else MemoryError) or (
            "out of memory" in msg
        )

    def check_finite_loss(self, loss: torch.Tensor) -> bool:
        if torch.isfinite(loss).all():
            self.consecutive_nans = 0
            return True
        self.nan_skips += 1
        self.consecutive_nans += 1
        logger.warning(f"Non-finite loss detected (skip #{self.nan_skips}); skipping optimizer step.")
        if self.consecutive_nans >= self.max_consecutive_nan_steps:
            raise RuntimeError(
                f"Training diverged: {self.consecutive_nans} consecutive non-finite loss steps."
            )
        return False

    def check_finite_gradients(self, model: nn.Module) -> bool:
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self.nan_skips += 1
                self.consecutive_nans += 1
                logger.warning(f"Non-finite gradient detected (skip #{self.nan_skips}); skipping optimizer step.")
                if self.consecutive_nans >= self.max_consecutive_nan_steps:
                    raise RuntimeError(
                        f"Training diverged: {self.consecutive_nans} consecutive non-finite gradient steps."
                    )
                return False
        return True

    def stats(self) -> Dict[str, int]:
        return {
            "oom_recoveries": self.oom_recoveries,
            "nan_skips": self.nan_skips,
        }


__all__ = [
    "ExecutionPlan",
    "ExecutionPlanner",
    "RuntimeGuard",
    "estimate_static_memory_mb",
    "split_batch_into_micro_batches",
]
