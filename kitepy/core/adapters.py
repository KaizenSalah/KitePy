"""
Task Adapters & Pillar Registry.

Decouples the universal Engine & BaseModel from any specific modality or task.
- Simple users get the right adapter automatically.
- Power users can override `forward_and_loss` or `compute_batch_metrics` for custom research.
"""

from collections.abc import Mapping
import inspect
import math
from typing import Any, Callable, Dict, Optional, Tuple, Type, cast
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in tuples, lists, or mappings (dict / HF BatchEncoding) to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def infer_batch_size(batch: Any) -> int:
    """Infer batch size from a tensor, tuple/list, or mapping batch."""
    if isinstance(batch, torch.Tensor):
        return int(batch.size(0))
    if isinstance(batch, Mapping):
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                return int(v.size(0))
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        return infer_batch_size(batch[0])
    return 1


# ============================================================================
# TASK ADAPTER BASE CLASS
# ============================================================================

class TaskAdapter:
    """
    Contract connecting any modality/task to the universal Engine.

    Override any method for custom training steps (GANs, contrastive, RLHF, detection, etc.).
    """

    task_name: str = "generic"
    primary_metric: str = "acc"

    def prepare_batch(self, batch: Any, device: torch.device) -> Any:
        """Move batch to device."""
        return move_to_device(batch, device)

    def unpack_batch(self, batch: Any) -> Tuple[Any, Any]:
        """Split batch into (inputs, targets) when applicable."""
        if isinstance(batch, Mapping):
            targets = batch.get("labels", batch.get("target"))
            return batch, targets
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        return batch, None

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: Any,
        criterion: nn.Module,
        engine: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any, Any]:
        """Run forward pass and compute loss. Returns (loss, outputs, targets)."""
        inputs, targets = self.unpack_batch(batch)
        if isinstance(inputs, Mapping):
            outputs = model(**dict(inputs))
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
                logits = getattr(outputs, "logits", outputs)
                return loss, logits, targets
            logits = getattr(outputs, "logits", outputs)
        else:
            logits = model(inputs)

        if targets is None:
            raise ValueError("Batch has no targets and model output did not provide .loss")
        loss = criterion(logits, targets)
        return loss, logits, targets

    def compute_batch_metrics(
        self,
        outputs: Any,
        targets: Any,
        loss_value: float,
        batch: Any,
    ) -> Dict[str, float]:
        """Return additive batch metric counters (e.g., correct, total)."""
        return {"total": float(infer_batch_size(batch))}

    def finalize_epoch_metrics(self, totals: Dict[str, float], num_batches: int) -> Dict[str, float]:
        """Convert additive epoch counters into final reported metrics."""
        return {}

    def predict_step(self, model: nn.Module, inputs: Any, device: torch.device) -> Any:
        """Run inference on prepared inputs."""
        prepared = move_to_device(inputs, device)
        if isinstance(prepared, Mapping):
            out = model(**dict(prepared))
            return getattr(out, "logits", out)
        return model(prepared)


# ============================================================================
# BUILT-IN TASK ADAPTERS
# ============================================================================

class ClassificationAdapter(TaskAdapter):
    """Standard single-label multi-class classification."""

    task_name = "classification"
    primary_metric = "acc"

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            return {"total": float(infer_batch_size(batch))}
        if targets.ndim > 1 and targets.dtype.is_floating_point:
            # Soft targets (e.g., MixUp/CutMix)
            hard_targets = targets.argmax(dim=1)
        else:
            hard_targets = targets
        preds = outputs.argmax(dim=1)
        correct = preds.eq(hard_targets).sum().item()
        total = hard_targets.size(0)
        return {"correct": float(correct), "total": float(total)}

    def finalize_epoch_metrics(self, totals: Dict[str, float], num_batches: int) -> Dict[str, float]:
        total = totals.get("total", 0.0)
        if total > 0 and "correct" in totals:
            return {"acc": 100.0 * totals["correct"] / total}
        return {}


class BinaryClassificationAdapter(TaskAdapter):
    """Binary classification (single logit output + BCEWithLogitsLoss)."""

    task_name = "binary"
    primary_metric = "acc"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        inputs, targets = self.unpack_batch(batch)
        outputs = model(**dict(inputs)) if isinstance(inputs, Mapping) else model(inputs)
        logits = getattr(outputs, "logits", outputs)
        logits_flat = logits.view(-1)
        targets_float = targets.float().view(-1)
        loss = criterion(logits_flat, targets_float)
        return loss, logits_flat, targets_float

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct = preds.eq(targets).sum().item()
        return {"correct": float(correct), "total": float(targets.numel())}

    def finalize_epoch_metrics(self, totals, num_batches):
        total = totals.get("total", 0.0)
        return {"acc": 100.0 * totals.get("correct", 0.0) / total} if total > 0 else {}


class MultiLabelClassificationAdapter(TaskAdapter):
    """Multi-label classification (B, C) logits + BCEWithLogitsLoss."""

    task_name = "multilabel"
    primary_metric = "acc"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        inputs, targets = self.unpack_batch(batch)
        outputs = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
        logits = getattr(outputs, "logits", outputs)
        if targets.ndim == 1 and logits.ndim == 2:
            targets_f = F.one_hot(
                targets.long().clamp(0, max(0, logits.size(-1) - 1)),
                num_classes=logits.size(-1),
            ).float()
        else:
            targets_f = targets.float()
        loss = criterion(logits, targets_f)
        return loss, logits, targets_f

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct = preds.eq(targets.float()).sum().item()
        total = float(targets.numel())
        return {"correct": float(correct), "total": total}

    def finalize_epoch_metrics(self, totals, num_batches):
        total = totals.get("total", 0.0)
        return {"acc": 100.0 * totals.get("correct", 0.0) / total} if total > 0 else {}


class RegressionAdapter(TaskAdapter):
    """Continuous regression tasks (computes MAE and RMSE)."""

    task_name = "regression"
    primary_metric = "mae"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        inputs, targets = self.unpack_batch(batch)
        outputs = model(**dict(inputs)) if isinstance(inputs, Mapping) else model(inputs)
        preds = getattr(outputs, "logits", outputs)
        targets_f = targets.float()
        if preds.ndim == 2 and preds.size(1) == 1 and targets_f.ndim == 1:
            preds = preds.squeeze(1)
        loss = criterion(preds, targets_f)
        return loss, preds, targets_f

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        diff = outputs.detach() - targets.detach()
        abs_err = diff.abs().sum().item()
        sq_err = (diff ** 2).sum().item()
        return {"abs_err": float(abs_err), "sq_err": float(sq_err), "total": float(targets.numel())}

    def finalize_epoch_metrics(self, totals, num_batches):
        total = totals.get("total", 0.0)
        if total <= 0:
            return {}
        return {
            "mae": totals["abs_err"] / total,
            "rmse": math.sqrt(totals["sq_err"] / total),
        }


def _accepts_keyword_input(model: nn.Module, key: str = "input_ids") -> bool:
    """Check whether model.forward accepts named keyword arguments (e.g., input_ids / **kwargs)."""
    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError):
        return True
    params = sig.parameters
    return key in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _filter_forward_kwargs(model: nn.Module, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter mapping kwargs to only arguments accepted by model.forward (unless it takes **kwargs)."""
    try:
        params = inspect.signature(model.forward).parameters
    except (ValueError, TypeError):
        return batch_dict
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return batch_dict
    return {k: v for k, v in batch_dict.items() if k in params}


class CausalLMAdapter(TaskAdapter):
    """Autoregressive causal language modeling (tracks loss & perplexity)."""

    task_name = "causal_lm"
    primary_metric = "perplexity"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        if isinstance(batch, Mapping):
            batch_dict = dict(batch)
            labels = batch_dict.get("labels", batch_dict.get("input_ids"))
            if _accepts_keyword_input(model, "input_ids"):
                outputs = model(**_filter_forward_kwargs(model, batch_dict))
            else:
                outputs = model(batch_dict["input_ids"])
            logits = getattr(outputs, "logits", outputs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                if labels is not None and (labels.size(1) <= 1 or not bool((labels[..., 1:] != -100).any().item())):
                    return logits.sum() * 0.0, logits, labels
                return outputs.loss, logits, labels
        else:
            inputs, labels = self.unpack_batch(batch)
            outputs = model(inputs)
            logits = getattr(outputs, "logits", outputs)

        if logits.ndim < 2 or logits.size(1) <= 1 or (labels is not None and labels.size(1) <= 1):
            return logits.sum() * 0.0, logits, labels

        # Standard next-token shift (align sequence length if model clamped context)
        if logits.size(1) != labels.size(1):
            min_t = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_t, :]
            labels = labels[:, :min_t]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if shift_labels.numel() == 0 or not bool((shift_labels != -100).any().item()):
            return logits.sum() * 0.0, logits, labels

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, logits, labels

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        return {"loss_sum": float(loss_value), "total": 1.0}

    def finalize_epoch_metrics(self, totals, num_batches):
        steps = max(1.0, totals.get("total", float(num_batches)))
        avg_loss = totals.get("loss_sum", 0.0) / steps
        ppl = math.exp(min(avg_loss, 20.0))
        return {"perplexity": ppl}


class SequenceClassificationAdapter(ClassificationAdapter):
    """Text sequence classification supporting dict/Mapping batches from HuggingFace tokenizers."""

    task_name = "sequence_classification"
    primary_metric = "acc"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        if isinstance(batch, Mapping):
            batch_dict = dict(batch)
            labels = batch_dict.get("labels")
            model_inputs = {k: v for k, v in batch_dict.items() if k != "labels"}
            if _accepts_keyword_input(model, "input_ids"):
                outputs = model(**batch_dict) if "labels" in batch_dict and hasattr(model, "config") else model(**model_inputs)
            else:
                outputs = model(batch_dict["input_ids"])
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss, getattr(outputs, "logits", outputs), labels
            logits = getattr(outputs, "logits", outputs)
            loss = criterion(logits, labels)
            return loss, logits, labels
        return super().forward_and_loss(model, batch, criterion, engine)


class ContrastiveAdapter(TaskAdapter):
    """CLIP-style symmetric contrastive image-text training."""

    task_name = "contrastive"
    primary_metric = "acc"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        if isinstance(batch, Mapping):
            outputs = model(**dict(batch))
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            outputs = model(batch[0], batch[1])
        else:
            outputs = model(batch)

        if isinstance(outputs, Mapping):
            logits_i = outputs["logits_per_image"]
            logits_t = outputs["logits_per_text"]
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            logits_i, logits_t = outputs
        else:
            raise ValueError("Contrastive model must return (logits_per_image, logits_per_text) or dict")

        labels = torch.arange(logits_i.size(0), device=logits_i.device)
        loss = 0.5 * (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels))
        return loss, logits_i, labels

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        preds = outputs.argmax(dim=1)
        correct = preds.eq(targets).sum().item()
        return {"correct": float(correct), "total": float(targets.size(0))}

    def finalize_epoch_metrics(self, totals, num_batches):
        total = totals.get("total", 0.0)
        return {"acc": 100.0 * totals.get("correct", 0.0) / total} if total > 0 else {}


class GenerativeAdapter(TaskAdapter):
    """Adapter for VAE (reconstruction + KL) or DDPM (noise prediction MSE) models."""

    task_name = "generative"
    primary_metric = "recon_loss"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(x, Mapping):
            x = x.get("pixel_values", x.get("image", next(iter(x.values()))))

        if hasattr(model, "compute_loss"):
            loss, aux = model.compute_loss(x)
            return loss, aux, x

        out = model(x)
        if isinstance(out, tuple) and len(out) == 3:
            # Standard VAE: (recon, mu, logvar)
            recon, mu, logvar = out
            recon_loss = F.mse_loss(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_w = getattr(model, "kl_weight", 0.001)
            return recon_loss + kl_w * kl_loss, recon, x

        loss = F.mse_loss(out, x)
        return loss, out, x

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        return {"loss_sum": float(loss_value), "total": 1.0}

    def finalize_epoch_metrics(self, totals, num_batches):
        steps = max(1.0, totals.get("total", float(num_batches)))
        return {"recon_loss": totals.get("loss_sum", 0.0) / steps}


class SegmentationAdapter(TaskAdapter):
    """Semantic segmentation adapter (pixel accuracy & mean IoU)."""

    task_name = "segmentation"
    primary_metric = "miou"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        inputs, targets = self.unpack_batch(batch)
        out = model(inputs)
        logits = out["out"] if isinstance(out, dict) and "out" in out else getattr(out, "logits", out)

        if logits.size(1) == 1 or (targets.ndim == 4 and targets.size(1) == 1):
            tgt = targets.float()
            if tgt.ndim == 3:
                tgt = tgt.unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, tgt)
        else:
            loss = F.cross_entropy(logits, targets.long())
        return loss, logits, targets

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        if outputs.size(1) == 1:
            preds = (torch.sigmoid(outputs.squeeze(1)) >= 0.5).long()
            tgt = targets.squeeze(1).long() if targets.ndim == 4 else targets.long()
            num_classes = 2
        else:
            preds = outputs.argmax(dim=1)
            tgt = targets.long()
            num_classes = outputs.size(1)

        correct = float(preds.eq(tgt).sum().item())
        total = float(tgt.numel())

        inter_sum = 0.0
        union_sum = 0.0
        valid_classes = 0.0
        for c in range(num_classes):
            p_c = preds == c
            t_c = tgt == c
            union = (p_c | t_c).sum().item()
            if union > 0:
                inter_sum += float((p_c & t_c).sum().item()) / float(union)
                valid_classes += 1.0
        return {"correct": correct, "total": total, "iou_sum": inter_sum, "iou_classes": valid_classes}

    def finalize_epoch_metrics(self, totals, num_batches):
        total = max(1.0, totals.get("total", 0.0))
        cls_cnt = max(1.0, totals.get("iou_classes", 0.0))
        return {
            "pixel_acc": 100.0 * totals.get("correct", 0.0) / total,
            "miou": totals.get("iou_sum", 0.0) / cls_cnt if totals.get("iou_classes", 0.0) > 0 else 0.0,
        }


class DetectionAdapter(TaskAdapter):
    """Object detection adapter compatible with torchvision detection models & custom detectors."""

    task_name = "detection"
    primary_metric = "detections"

    def forward_and_loss(self, model, batch, criterion, engine=None):
        images, targets = self.unpack_batch(batch)
        if isinstance(images, torch.Tensor):
            images = list(images)
        if isinstance(targets, Mapping):
            # Collate dict of batched tensors -> list of target dicts
            b = len(images)
            targets = [{k: v[i] for k, v in targets.items()} for i in range(b)]

        if model.training:
            out = model(images, targets)
            if isinstance(out, dict):
                loss = sum(v for v in out.values() if isinstance(v, torch.Tensor))
                return loss, out, targets
            if isinstance(out, torch.Tensor):
                return out, out, targets
            raise ValueError(f"Unexpected detection train output type: {type(out)}")
        else:
            preds = model(images)
            # Detection models in eval mode return predictions without losses; keep validation non-crashing
            device = images[0].device if isinstance(images, (list, tuple)) and len(images) > 0 else torch.device("cpu")
            loss = torch.tensor(0.0, device=device)
            return loss, preds, targets

    def compute_batch_metrics(self, outputs, targets, loss_value, batch):
        if isinstance(outputs, list):
            det_count = sum(len(d.get("boxes", [])) for d in outputs if isinstance(d, dict))
            return {"detections": float(det_count), "total": float(len(outputs))}
        return {"total": float(infer_batch_size(batch))}

    def finalize_epoch_metrics(self, totals, num_batches):
        total = max(1.0, totals.get("total", 1.0))
        if "detections" in totals:
            return {"avg_detections": totals["detections"] / total}
        return {}


class CustomStepAdapter(TaskAdapter):
    """Wraps a user-provided train_step(engine, batch) -> (loss, outputs, targets) function."""

    task_name = "custom"
    primary_metric = "loss"

    def __init__(self, step_fn: Callable):
        self.step_fn = step_fn

    def forward_and_loss(self, model, batch, criterion, engine=None):
        res = self.step_fn(engine, batch)
        if isinstance(res, torch.Tensor):
            return res, None, None
        return res


ADAPTER_REGISTRY: Dict[str, Type[TaskAdapter]] = {
    "classification": ClassificationAdapter,
    "binary": BinaryClassificationAdapter,
    "multilabel": MultiLabelClassificationAdapter,
    "regression": RegressionAdapter,
    "causal_lm": CausalLMAdapter,
    "sequence_classification": SequenceClassificationAdapter,
    "contrastive": ContrastiveAdapter,
    "generative": GenerativeAdapter,
    "segmentation": SegmentationAdapter,
    "detection": DetectionAdapter,
}


def get_adapter(
    task_or_adapter: Optional[Any] = None,
    modality: str = "vision",
    train_step: Optional[Callable] = None,
) -> TaskAdapter:
    """Resolve a TaskAdapter from an instance, task string, modality, or custom step function."""
    if train_step is not None:
        return CustomStepAdapter(train_step)
    if isinstance(task_or_adapter, TaskAdapter):
        return task_or_adapter
    if isinstance(task_or_adapter, str):
        key = task_or_adapter.lower()
        if key in ADAPTER_REGISTRY:
            return ADAPTER_REGISTRY[key]()
        raise ValueError(f"Unknown task '{task_or_adapter}'. Available: {list(ADAPTER_REGISTRY.keys())}")

    # Modality defaults
    if modality in ("vision", "audio"):
        return ClassificationAdapter()
    if modality == "language":
        return CausalLMAdapter()
    if modality == "multimodal":
        return ContrastiveAdapter()
    if modality == "generative":
        return GenerativeAdapter()
    if modality == "tabular":
        return ClassificationAdapter()
    return TaskAdapter()


# ============================================================================
# PILLAR REGISTRY (Decouples core BaseModel from individual pillars)
# ============================================================================

_PILLAR_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_pillar(
    name: str,
    data_loader: Optional[Callable] = None,
    default_task: str = "classification",
):
    """Register a domain pillar's data loader and defaults."""
    _PILLAR_REGISTRY[name] = {
        "data_loader": data_loader,
        "default_task": default_task,
    }


def get_pillar_data_loader(modality: str) -> Optional[Callable]:
    """Retrieve registered data loader for a modality."""
    entry = _PILLAR_REGISTRY.get(modality)
    if entry is not None:
        loader = entry.get("data_loader")
        if callable(loader):
            return cast(Callable, loader)
    return None


__all__ = [
    "move_to_device",
    "infer_batch_size",
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
]
