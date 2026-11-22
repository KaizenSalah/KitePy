"""
Universal training engine for all modalities and tasks.

Supports:
- TaskAdapter protocol (classification, regression, causal LM, contrastive, generative, custom train_step)
- Full CallbackList hooks + built-in CSVLogger / TensorBoard / WandB / EarlyStopping
- Gradient accumulation, max_steps, warmup_steps, step vs epoch vs plateau schedulers
- Differential LRs & layer-wise LR decay
- Mixed precision (fp16 / bf16), optional HuggingFace Accelerate distributed backend
- Exponential Moving Average (EMA) weights
- MixUp / CutMix batch augmentations
- LR auto-finder (`lr="auto"` / `lr_find=True`), OOM-safe auto batch size, overfit_batches, detect_anomaly
- Full checkpoint save & resume (model, optimizer, scheduler, scaler, EMA, epoch, history)
"""

from itertools import islice
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapters import TaskAdapter, get_adapter, infer_batch_size
from .callbacks import (
    CallbackList,
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,
)
from .config import TrainingConfig
from .logging import get_logger
from .lr_finder import LRFinder
from .planner import ExecutionPlan, ExecutionPlanner, RuntimeGuard, split_batch_into_micro_batches
from .serialization import atomic_torch_save, get_lineage_metadata
from .utils import (
    ModelEMA,
    build_parameter_groups,
    cleanup_old_checkpoints,
    get_checkpoint_path,
    get_device,
    get_lr,
    print_epoch_summary,
    print_training_header,
    set_seed,
)


# ============================================================================
# LOSS FUNCTIONS & AUGMENTATIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""

    def __init__(self, weight=None, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def _apply_mixup_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp or CutMix when inputs are 4D image tensors and targets are 1D class indices."""
    if inputs.ndim != 4 or targets.ndim != 1:
        return inputs, targets

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0:
        return inputs, targets

    lam = float(np.random.beta(alpha, alpha))
    batch_size = inputs.size(0)
    perm = torch.randperm(batch_size, device=inputs.device)

    y1 = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes).float()
    y2 = y1[perm]

    if use_cutmix:
        _, _, h, w = inputs.shape
        cut_rat = math.sqrt(max(0.0, 1.0 - lam))
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))
        x1 = max(0, cx - cut_w // 2)
        y1_box = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2_box = min(h, cy + cut_h // 2)

        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y1_box:y2_box, x1:x2] = inputs[perm, :, y1_box:y2_box, x1:x2]
        area = (x2 - x1) * (y2_box - y1_box)
        lam = 1.0 - (area / float(w * h))
    else:
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[perm]

    mixed_targets = lam * y1 + (1.0 - lam) * y2
    return mixed_inputs, mixed_targets


def _compute_balanced_class_weights(train_loader: DataLoader, device: torch.device) -> Optional[torch.Tensor]:
    """Compute inverse-frequency class weights from dataset targets if available."""
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None:
        return None
    targets = getattr(dataset, "targets", None)
    if targets is None and hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        inner = getattr(dataset, "dataset", None)
        if inner is not None:
            inner_targets = getattr(inner, "targets", None)
            if inner_targets is not None:
                targets = [inner_targets[i] for i in dataset.indices]
            elif hasattr(inner, "tensors") and len(inner.tensors) >= 2:
                targets = inner.tensors[1][dataset.indices]
    if targets is None and hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        targets = dataset.tensors[1]
    if targets is None:
        return None
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    if targets_t.ndim != 1 or targets_t.numel() == 0:
        return None
    counts = torch.bincount(targets_t).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class Engine:
    """Universal training engine supporting all KitePy modalities and advanced workflows."""

    STEP_SCHEDULERS = {"onecycle", "warmup_cosine", "cosine_warmup", "cosine", "linear", "polynomial"}
    EPOCH_SCHEDULERS = {"step", "multistep", "exponential"}

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        modality: str = "vision",
        adapter: Optional[TaskAdapter] = None,
    ):
        self.model = model
        self.config = config
        self.modality = modality
        self.adapter = adapter or get_adapter(
            task_or_adapter=getattr(config, "task", None),
            modality=modality,
            train_step=getattr(config, "train_step", None),
        )

        # Deterministic mode if requested
        if getattr(config, "deterministic", False):
            set_seed(config.seed, deterministic=True)

        if config.fast_dev_run:
            self.config.epochs = 1

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []

        # Optional Accelerate distributed backend
        self.accelerator = None
        if getattr(config, "use_accelerate", False) or getattr(config, "strategy", "auto") in ("ddp", "fsdp", "deepspeed"):
            try:
                from accelerate import Accelerator
                mp = "no"
                if config.mixed_precision in (True, "fp16"):
                    mp = "fp16"
                elif config.mixed_precision == "bf16":
                    mp = "bf16"
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=max(1, config.gradient_accumulation_steps),
                    mixed_precision=mp,
                )
                self.device = self.accelerator.device
            except ImportError:
                print("⚠️  accelerate not installed; falling back to single-device Engine")
                self.device = get_device(config.devices)
        else:
            self.device = get_device(config.devices)

        self.model = self.model.to(self.device)

        # Optional torch.compile
        if getattr(config, "compile", False) and hasattr(torch, "compile"):
            try:
                self.model = cast(nn.Module, torch.compile(self.model, mode=getattr(config, "compile_mode", "default")))
            except Exception as e:
                print(f"⚠️  torch.compile skipped: {e}")

        # AMP / mixed precision setup
        self.amp_enabled = False
        self.amp_dtype = torch.float16
        self.scaler: Optional[Any] = None
        self._configure_precision()

        # Optimizer, Scheduler, Criterion
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        self._scheduler_mode = "step"  # "step", "epoch", or "plateau"
        self.criterion = self._create_loss_function()

        # EMA
        self.ema = ModelEMA(self.model, decay=config.ema_decay) if getattr(config, "ema", False) else None

        # Logging, Execution Planner & Runtime Fault Recovery
        self.logger = get_logger(
            "kitepy.engine",
            level=getattr(config, "log_level", "INFO"),
            json_file=getattr(config, "structured_log_file", None),
        )
        self.planner = ExecutionPlanner()
        self.execution_plan: Optional[ExecutionPlan] = None
        self._planned_micro_batch_size: Optional[int] = None
        self.guard = RuntimeGuard(
            auto_recover=getattr(config, "auto_recover", True),
            max_consecutive_nan_steps=getattr(config, "max_consecutive_nan_steps", 5),
        )
        self.execution_report: Dict[str, Any] = {}

        # Callbacks
        self.callbacks = self._build_callbacks()

    def _is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    def _configure_precision(self):
        mp = self.config.mixed_precision
        if self.accelerator is not None:
            return  # Accelerate handles AMP

        if mp in (False, "fp32", "no", None):
            self.amp_enabled = False
            self.config.mixed_precision = False
            self.scaler = None
            return

        if self.device.type == "cuda":
            self.amp_enabled = True
            prefer_bf16 = (
                mp == "bf16"
                or self.config.amp_dtype == "bf16"
                or (
                    self.config.amp_dtype == "auto"
                    and mp is True
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                )
            )
            if prefer_bf16 and mp != "fp16":
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bf16 does not need GradScaler
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
        elif mp == "bf16":
            # PyTorch CPU supports bfloat16 autocast
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        else:
            self.amp_enabled = False
            self.config.mixed_precision = False
            self.scaler = None

    def _build_callbacks(self) -> CallbackList:
        cb_list = CallbackList(self.config.callbacks)
        csv_dir = getattr(self.config, "csv_log_dir", None)
        if isinstance(csv_dir, str) and csv_dir:
            cb_list.add(CSVLogger(log_dir=csv_dir))
        tb_dir = getattr(self.config, "tensorboard_dir", None)
        if isinstance(tb_dir, str) and tb_dir:
            cb_list.add(TensorBoardLogger(log_dir=tb_dir))
        wandb_proj = getattr(self.config, "wandb_project", None)
        if isinstance(wandb_proj, str) and wandb_proj:
            cb_list.add(WandbLogger(project=wandb_proj))
        return cb_list

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with optional differential LRs and layer-wise LR decay."""
        optimizer = self.config.optimizer

        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer

        raw_lr = self.config.lr
        base_lr: Union[float, Dict[str, float]] = 1e-3 if isinstance(raw_lr, str) else raw_lr

        param_groups = build_parameter_groups(
            model=self.model,
            lr=base_lr,
            weight_decay=self.config.weight_decay,
            backbone_lr_mult=getattr(self.config, "backbone_lr_mult", None),
            layer_decay=getattr(self.config, "layer_decay", None),
        )

        # Fallback if all parameters are frozen
        if not param_groups:
            param_groups = [{"params": list(self.model.parameters()), "lr": 1e-3, "weight_decay": 0.0}]

        kwargs = dict(self.config.optimizer_kwargs or {})

        if callable(optimizer) and not isinstance(optimizer, str):
            return cast(torch.optim.Optimizer, optimizer(param_groups, **kwargs))

        optimizer_name = optimizer.lower()
        opt_map: Dict[str, Callable[[], torch.optim.Optimizer]] = {
            "adam": lambda: torch.optim.Adam(param_groups, **kwargs),
            "adamw": lambda: torch.optim.AdamW(param_groups, **kwargs),
            "sgd": lambda: torch.optim.SGD(param_groups, momentum=kwargs.pop("momentum", 0.9), **kwargs),
            "rmsprop": lambda: torch.optim.RMSprop(param_groups, **kwargs),
            "adadelta": lambda: torch.optim.Adadelta(param_groups, **kwargs),
            "adagrad": lambda: torch.optim.Adagrad(param_groups, **kwargs),
            "adamax": lambda: torch.optim.Adamax(param_groups, **kwargs),
            "nadam": lambda: torch.optim.NAdam(param_groups, **kwargs),
            "radam": lambda: torch.optim.RAdam(param_groups, **kwargs),
        }

        if optimizer_name == "lion":
            try:
                from lion_pytorch import Lion  # type: ignore[import-not-found]
                return cast(torch.optim.Optimizer, Lion(param_groups, **kwargs))
            except ImportError:
                raise ImportError("Lion optimizer requires: pip install lion-pytorch")

        if optimizer_name not in opt_map:
            available = list(opt_map.keys()) + ["lion"]
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {available}")

        return opt_map[optimizer_name]()

    def _create_scheduler(self, total_opt_steps: int):
        """Create learning rate scheduler with optional warmup composition."""
        scheduler = self.config.scheduler

        if isinstance(scheduler, lr_scheduler.LRScheduler):
            self._scheduler_mode = "step"
            return scheduler

        if callable(scheduler) and not isinstance(scheduler, str):
            self._scheduler_mode = "step"
            return scheduler(self.optimizer, **dict(self.config.scheduler_kwargs or {}))

        scheduler_name = scheduler.lower()
        kwargs = dict(self.config.scheduler_kwargs or {})
        total_opt_steps = max(1, total_opt_steps)

        # Resolve warmup_steps (supports int or fraction)
        raw_warmup = getattr(self.config, "warmup_steps", 0) or kwargs.pop("warmup_steps", 0)
        if isinstance(raw_warmup, float) and 0 < raw_warmup < 1:
            warmup_steps = int(raw_warmup * total_opt_steps)
        else:
            warmup_steps = int(raw_warmup)

        if scheduler_name == "plateau":
            self._scheduler_mode = "plateau"
            patience = kwargs.pop("patience", 5)
            factor = kwargs.pop("factor", 0.1)
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=factor, **kwargs)

        if scheduler_name in self.EPOCH_SCHEDULERS:
            self._scheduler_mode = "epoch"
            if scheduler_name == "step":
                step_size = kwargs.pop("step_size", max(1, self.config.epochs // 3))
                gamma = kwargs.pop("gamma", 0.1)
                return lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma, **kwargs)
            if scheduler_name == "multistep":
                milestones = kwargs.pop("milestones", [max(1, self.config.epochs // 3), max(2, 2 * self.config.epochs // 3)])
                gamma = kwargs.pop("gamma", 0.1)
                return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma, **kwargs)
            if scheduler_name == "exponential":
                gamma = kwargs.pop("gamma", 0.95)
                return lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma, **kwargs)

        self._scheduler_mode = "step"

        if scheduler_name in ("none", "constant"):
            if warmup_steps > 0:
                return lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            return None

        if scheduler_name == "onecycle":
            max_lr = [pg["lr"] for pg in self.optimizer.param_groups]
            return lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr if len(max_lr) > 1 else max_lr[0],
                total_steps=total_opt_steps,
                **kwargs,
            )

        if scheduler_name in ("warmup_cosine", "cosine_warmup"):
            w_steps = warmup_steps if warmup_steps > 0 else max(1, int(0.1 * total_opt_steps))
            w_steps = min(w_steps, max(1, total_opt_steps - 1))

            def lr_lambda(step):
                if step < w_steps:
                    return float(step + 1) / float(max(1, w_steps))
                progress = float(step - w_steps) / float(max(1, total_opt_steps - w_steps))
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            return lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Remaining step schedulers (cosine, linear, polynomial)
        main_steps = max(1, total_opt_steps - warmup_steps)
        main_sched: lr_scheduler.LRScheduler
        if scheduler_name == "cosine":
            main_sched = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=main_steps, **kwargs)
        elif scheduler_name == "linear":
            main_sched = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=kwargs.pop("end_factor", 0.0),
                total_iters=main_steps,
                **kwargs,
            )
        elif scheduler_name == "polynomial":
            power = kwargs.pop("power", 1.0)
            main_sched = lr_scheduler.PolynomialLR(self.optimizer, total_iters=main_steps, power=power, **kwargs)
        else:
            available = [
                "none", "constant", "cosine", "linear", "step", "multistep",
                "exponential", "polynomial", "onecycle", "warmup_cosine", "plateau",
            ]
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {available}")

        if warmup_steps > 0 and main_steps > 0:
            warmup_sched = lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            )
            return lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_steps]
            )
        return main_sched

    def _create_loss_function(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Create loss function from config."""
        loss = getattr(self.config, "loss", "cross_entropy")

        if isinstance(loss, nn.Module):
            return loss
        if callable(loss) and not isinstance(loss, str):
            return cast(nn.Module, loss)

        loss_name = loss.lower() if isinstance(loss, str) else "cross_entropy"

        # Resolve class_weights from config if not explicitly passed
        weight = class_weights
        if weight is None and isinstance(getattr(self.config, "class_weights", None), (list, tuple, torch.Tensor)):
            weight = torch.as_tensor(self.config.class_weights, dtype=torch.float32)
        if weight is not None:
            weight = weight.to(self.device)

        label_smoothing = getattr(self.config, "label_smoothing", 0.0)

        loss_map = {
            "cross_entropy": lambda: nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing),
            "ce": lambda: nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing),
            "focal": lambda: FocalLoss(weight=weight, gamma=2.0),
            "bce": lambda: nn.BCEWithLogitsLoss(pos_weight=weight),
            "binary_cross_entropy": lambda: nn.BCEWithLogitsLoss(pos_weight=weight),
            "mse": lambda: nn.MSELoss(),
            "l2": lambda: nn.MSELoss(),
            "mae": lambda: nn.L1Loss(),
            "l1": lambda: nn.L1Loss(),
            "smooth_l1": lambda: nn.SmoothL1Loss(),
            "huber": lambda: nn.HuberLoss(),
            "kl_div": lambda: nn.KLDivLoss(reduction="batchmean"),
            "kld": lambda: nn.KLDivLoss(reduction="batchmean"),
            "nll": lambda: nn.NLLLoss(weight=weight),
            "margin": lambda: nn.MultiMarginLoss(),
        }

        if loss_name not in loss_map:
            raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_map.keys())}")
        return loss_map[loss_name]()

    def _maybe_run_lr_finder(self, train_loader: DataLoader):
        """Run LR range test if lr='auto' or lr_find=True."""
        should_find = getattr(self.config, "lr_find", False) or (
            isinstance(self.config.lr, str) and self.config.lr.lower() == "auto"
        )
        if not should_find:
            return
        finder = LRFinder(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            adapter=self.adapter,
        )
        finder.find(
            train_loader=train_loader,
            min_lr=self.config.lr_find_min,
            max_lr=self.config.lr_find_max,
            num_steps=min(self.config.lr_find_num_steps, max(10, len(train_loader) * 2)),
        )
        suggested = finder.suggest_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = suggested
        self.config.lr = suggested

    def _prepare_loader_subset(self, loader: DataLoader):
        """Handle overfit_batches and length-less IterableDataset loaders safely."""
        ob = getattr(self.config, "overfit_batches", 0)
        try:
            loader_len = len(loader)
        except TypeError:
            loader_len = self.config.max_steps or 50

        if not ob:
            return loader, loader_len
        if isinstance(ob, float) and 0 < ob < 1.0:
            n_batches = max(1, int(loader_len * ob))
        else:
            n_batches = max(1, int(ob))
        cached = list(islice(loader, n_batches))
        return cached, len(cached)

    def _resume_if_requested(self):
        """Restore training state from checkpoint if resume_from_checkpoint is set."""
        ckpt_path = getattr(self.config, "resume_from_checkpoint", None)
        if not ckpt_path:
            return 0
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] and self.ema:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0)
        print(f"✓ Resumed training from {ckpt_path} (epoch {start_epoch}, step {self.global_step})")
        return start_epoch

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        """Train the model."""
        # Handle "balanced" class weights automatically
        if getattr(self.config, "class_weights", None) == "balanced":
            bal_weights = _compute_balanced_class_weights(train_loader, self.device)
            if bal_weights is not None:
                self.criterion = self._create_loss_function(class_weights=bal_weights)

        # Auto LR finder before training
        self._maybe_run_lr_finder(train_loader)

        # Constraint-aware pre-flight ExecutionPlanner
        self._planned_micro_batch_size = None
        if getattr(self.config, "auto_plan", False) or getattr(self.config, "memory_budget_mb", None) is not None:
            try:
                sample_batch = next(iter(train_loader))
                observed_bs = infer_batch_size(sample_batch)
                target_bs = observed_bs if observed_bs > 0 else (
                    self.config.batch_size if isinstance(self.config.batch_size, int) else 32
                )
                world_sz = self.accelerator.num_processes if self.accelerator is not None else 1
                self.execution_plan = self.planner.plan(
                    model=self.model,
                    device=self.device,
                    target_batch_size=target_bs,
                    sample_batch=sample_batch,
                    adapter=self.adapter,
                    criterion=self.criterion,
                    current_grad_accum=getattr(self.config, "gradient_accumulation_steps", 1),
                    mixed_precision=self.config.mixed_precision,
                    memory_budget_mb=getattr(self.config, "memory_budget_mb", None),
                    world_size=world_sz,
                )
                # Sync planned precision & micro-batch size onto Engine
                if self.accelerator is None:
                    self.amp_enabled = self.execution_plan.amp_enabled
                    if self.execution_plan.amp_dtype == "bf16":
                        self.amp_dtype = torch.bfloat16
                        self.scaler = None
                    elif self.execution_plan.amp_dtype == "fp16" and self.device.type == "cuda":
                        self.amp_dtype = torch.float16
                        self.scaler = self.scaler or torch.amp.GradScaler("cuda")
                    else:
                        self.amp_dtype = torch.float32
                        self.scaler = None

                if self.execution_plan.micro_batch_size < target_bs:
                    self._planned_micro_batch_size = self.execution_plan.micro_batch_size

                if self.execution_plan.gradient_checkpointing:
                    grad_ckpt_fn = getattr(self.model, "gradient_checkpointing_enable", None)
                    if callable(grad_ckpt_fn):
                        grad_ckpt_fn()

                self.logger.info(
                    "execution_plan",
                    extra={"metrics": self.execution_plan.to_dict(), "structured_only": True},
                )
            except StopIteration:
                pass

        if self._is_main_process():
            print_training_header(self.config)

        train_iterable, steps_per_epoch = self._prepare_loader_subset(train_loader)
        accum_steps = max(1, int(getattr(self.config, "gradient_accumulation_steps", 1)))
        opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / accum_steps))
        total_opt_steps = self.config.max_steps or (self.config.epochs * opt_steps_per_epoch)

        self.scheduler = self._create_scheduler(total_opt_steps)

        if self.accelerator is not None:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            if isinstance(train_iterable, DataLoader):
                train_iterable = self.accelerator.prepare_data_loader(train_iterable)
            if val_loader is not None and isinstance(val_loader, DataLoader):
                val_loader = self.accelerator.prepare_data_loader(val_loader)
            if self.scheduler is not None:
                self.scheduler = self.accelerator.prepare(self.scheduler)

        start_epoch = self._resume_if_requested()
        self.callbacks.on_train_start(self)
        self.logger.info(
            "train_start",
            extra={
                "metrics": {"epochs": self.config.epochs, "amp_enabled": self.amp_enabled, "amp_dtype": str(self.amp_dtype)},
                "structured_only": True,
            },
        )

        max_epochs = self.config.epochs
        if self.config.max_steps is not None and max_epochs <= start_epoch:
            max_epochs = start_epoch + math.ceil(self.config.max_steps / opt_steps_per_epoch)

        for epoch in range(start_epoch + 1, max_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            self.callbacks.on_epoch_start(self, epoch)
            train_loss, train_metrics = self._train_epoch(
                train_iterable, steps_per_epoch, val_loader=val_loader
            )

            # Validation (respecting val_every_n_epochs)
            val_loss = None
            val_metrics: Dict[str, float] = {}
            val_every = max(1, int(getattr(self.config, "val_every_n_epochs", 1)))
            if val_loader is not None and (epoch % val_every == 0 or epoch == max_epochs):
                self.callbacks.on_validation_start(self)
                eval_model = self.ema.shadow if self.ema is not None else self.model
                val_loss, val_metrics = self._validate(val_loader, eval_model=eval_model)
                self.callbacks.on_validation_end(self, {"val_loss": val_loss, **val_metrics})

            # Epoch-level or Plateau scheduler step
            if self.scheduler is not None:
                if self._scheduler_mode == "epoch":
                    self.scheduler.step()
                elif self._scheduler_mode == "plateau":
                    metric = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric)

            epoch_time = time.time() - epoch_start
            train_acc = train_metrics.get("acc")
            val_acc = val_metrics.get("acc")

            history_entry: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": get_lr(self.optimizer),
                "time": epoch_time,
            }
            for k, v in train_metrics.items():
                if k != "acc":
                    history_entry[f"train_{k}"] = v
            for k, v in val_metrics.items():
                if k != "acc":
                    history_entry[f"val_{k}"] = v

            self.history.append(history_entry)

            print_epoch_summary(
                epoch=epoch,
                total_epochs=max_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=get_lr(self.optimizer),
                epoch_time=epoch_time,
                extra_metrics={
                    k: v for k, v in history_entry.items()
                    if k not in ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "time")
                },
            )

            self.callbacks.on_epoch_end(self, epoch, history_entry)
            self.logger.info(
                f"epoch_{epoch}_end",
                extra={"metrics": history_entry, "structured_only": True},
            )

            if self.config.save_every_n_epochs and epoch % self.config.save_every_n_epochs == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    self._save_checkpoint(epoch, val_loss if val_loss is not None else train_loss)

            # Built-in early stopping
            if self.config.early_stopping and val_loss is not None:
                if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch} epochs")
                        break

            if self.callbacks.should_stop():
                break

            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                print(f"\nReached max_steps={self.config.max_steps}, stopping training.")
                break

        self.callbacks.on_train_end(self)

        if self.accelerator is not None:
            self.model = self.accelerator.unwrap_model(self.model)

        # If EMA was enabled, copy EMA shadow weights into self.model for inference/saving
        if self.ema is not None:
            self.model.load_state_dict(self.ema.shadow.state_dict())

        self.execution_report = {
            "lineage": get_lineage_metadata(),
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
            "recovery": self.guard.stats(),
            "epochs_completed": len(self.history),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
        }
        self.logger.info("train_end", extra={"metrics": self.execution_report, "structured_only": True})

        return self.model, self.history

    def _optimizer_step(self):
        """Validate gradient finiteness, clip gradients, and step optimizer + scaler + EMA + step-scheduler."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Under FP16 GradScaler, normal initial overflow is handled automatically via scale backoff;
        # only trigger RuntimeGuard gradient abort/skip when unscaled (fp32/bf16) or scale has bottomed out.
        should_check_grads = self.scaler is None or self.scaler.get_scale() <= 1.0
        if self.guard.auto_recover and should_check_grads and not self.guard.check_finite_gradients(self.model):
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()
            return

        if self.config.max_grad_norm > 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        did_step = True
        if self.scaler is not None:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # If GradScaler skipped optimizer.step() due to inf/nan overflow, scale decreased;
            # do not advance scheduler/EMA on skipped steps.
            did_step = self.scaler.get_scale() >= scale_before
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        if did_step:
            if self.ema is not None:
                self.ema.update(self.model)

            if self.scheduler is not None and self._scheduler_mode == "step":
                self.scheduler.step()

            self.global_step += 1

    def _forward_backward_micro_batch(
        self,
        batch: Any,
        accum_steps: int,
    ) -> Tuple[float, Any, Any, bool]:
        """Run forward/backward on a batch, with planned micro-batching, OOM splitting, and NaN guarding."""
        batch_size = max(1, infer_batch_size(batch))
        chunks_to_try = 1
        if self._planned_micro_batch_size and self._planned_micro_batch_size < batch_size:
            chunks_to_try = max(1, math.ceil(batch_size / self._planned_micro_batch_size))

        while True:
            try:
                if chunks_to_try == 1:
                    sub_batches = [batch]
                else:
                    sub_batches = split_batch_into_micro_batches(batch, chunks_to_try)

                total_raw_loss = 0.0
                last_outputs, last_targets = None, None
                effective_div = accum_steps * len(sub_batches)

                for sb in sub_batches:
                    if self.accelerator is not None:
                        with self.accelerator.accumulate(self.model):
                            loss, outputs, targets = self.adapter.forward_and_loss(
                                self.model, sb, self.criterion, engine=self
                            )
                            if self.guard.auto_recover and not self.guard.check_finite_loss(loss):
                                self.optimizer.zero_grad(set_to_none=True)
                                return 0.0, outputs, targets, False
                            self.accelerator.backward(loss / len(sub_batches))
                    else:
                        with torch.amp.autocast(
                            device_type=self.device.type if self.device.type in ("cuda", "cpu") else "cpu",
                            dtype=self.amp_dtype,
                            enabled=self.amp_enabled,
                        ):
                            loss, outputs, targets = self.adapter.forward_and_loss(
                                self.model, sb, self.criterion, engine=self
                            )
                        if self.guard.auto_recover and not self.guard.check_finite_loss(loss):
                            self.optimizer.zero_grad(set_to_none=True)
                            return 0.0, outputs, targets, False
                        scaled_loss = loss / effective_div
                        if self.scaler is not None:
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                    total_raw_loss += float(loss.item())
                    last_outputs, last_targets = outputs, targets

                return total_raw_loss / len(sub_batches), last_outputs, last_targets, True

            except Exception as exc:
                if self.guard.auto_recover and self.guard.is_oom_error(exc) and chunks_to_try < max(8, batch_size):
                    self.guard.oom_recoveries += 1
                    chunks_to_try *= 2
                    self.optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.warning(
                        f"OOM detected; splitting batch into {chunks_to_try} micro-batches (recovery #{self.guard.oom_recoveries})"
                    )
                    continue
                raise

    def _train_epoch(
        self,
        train_loader: Any,
        steps_per_epoch: int,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        batches_run = 0
        metric_totals: Dict[str, float] = {}

        accum_steps = max(1, int(getattr(self.config, "gradient_accumulation_steps", 1)))
        mixup_a = float(getattr(self.config, "mixup_alpha", 0.0))
        cutmix_a = float(getattr(self.config, "cutmix_alpha", 0.0))

        # Mid-epoch validation interval
        val_interval = getattr(self.config, "val_check_interval", 1.0)
        val_every_n_batches = None
        if val_loader is not None and isinstance(val_interval, float) and 0 < val_interval < 1.0:
            val_every_n_batches = max(1, int(steps_per_epoch * val_interval))
        elif val_loader is not None and isinstance(val_interval, int) and val_interval > 0:
            val_every_n_batches = val_interval

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False, disable=not self._is_main_process())

        for batch_idx, raw_batch in enumerate(pbar):
            self.callbacks.on_batch_start(self, batch_idx)

            batch = self.adapter.prepare_batch(raw_batch, self.device)

            # Apply MixUp / CutMix if enabled on tensor tuple batches
            if (mixup_a > 0 or cutmix_a > 0) and isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                num_cls = None
                get_clf_fn = getattr(self.model, "get_classifier", None)
                if callable(get_clf_fn):
                    clf = get_clf_fn()
                    num_cls = getattr(clf, "out_features", None)
                if num_cls is None:
                    linears = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
                    if linears:
                        num_cls = linears[-1].out_features
                if num_cls is None:
                    num_cls = int(y.max().item()) + 1 if y.numel() > 0 else 2
                x_mix, y_mix = _apply_mixup_cutmix(x, y, num_cls, mixup_a, cutmix_a)
                batch = (x_mix, y_mix)

            ctx = torch.autograd.detect_anomaly() if getattr(self.config, "detect_anomaly", False) else _NullContext()
            with ctx:
                raw_loss_val, outputs, targets, is_valid = self._forward_backward_micro_batch(batch, accum_steps)

            if is_valid:
                if self.accelerator is not None:
                    if self.accelerator.sync_gradients:
                        self._optimizer_step()
                else:
                    is_boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == steps_per_epoch)
                    if is_boundary:
                        self._optimizer_step()

                total_loss += raw_loss_val
                batches_run += 1

                batch_metrics = self.adapter.compute_batch_metrics(outputs, targets, raw_loss_val, batch)
                for k, v in batch_metrics.items():
                    metric_totals[k] = metric_totals.get(k, 0.0) + v

            self.callbacks.on_batch_end(self, batch_idx, raw_loss_val)

            if (batch_idx + 1) % max(1, self.config.log_every_n_steps) == 0:
                pbar.set_postfix({"loss": f"{total_loss / batches_run:.4f}"})

            if (
                val_loader is not None
                and val_every_n_batches
                and (batch_idx + 1) % val_every_n_batches == 0
                and (batch_idx + 1) < steps_per_epoch
            ):
                v_loss, v_metrics = self._validate(val_loader)
                self.callbacks.on_validation_end(self, {"val_loss": v_loss, **v_metrics})
                self.model.train()

            if self.config.fast_dev_run:
                # Ensure optimizer stepped on fast_dev_run even if accum_steps > 1
                if self.global_step == 0 and self.accelerator is None:
                    self._optimizer_step()
                print("\n  [Fast Dev Run] Stopping after 1 batch")
                break

            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

        pbar.close()
        return self._reduce_distributed_metrics(total_loss, batches_run, metric_totals)

    def _reduce_distributed_metrics(
        self, total_loss: float, batches_run: int, metric_totals: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Aggregate loss and metric counters globally across DDP/FSDP ranks."""
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            loss_t = torch.tensor([total_loss, float(batches_run)], device=self.device)
            loss_t = self.accelerator.reduce(loss_t, reduction="sum")
            total_loss = float(loss_t[0].item())
            batches_run = int(loss_t[1].item())
            for k in sorted(metric_totals.keys()):
                kt = torch.tensor(metric_totals[k], device=self.device)
                metric_totals[k] = float(self.accelerator.reduce(kt, reduction="sum").item())

        avg_loss = total_loss / max(1, batches_run)
        epoch_metrics = self.adapter.finalize_epoch_metrics(metric_totals, max(1, batches_run))
        return avg_loss, epoch_metrics

    def _validate(
        self,
        val_loader: DataLoader,
        eval_model: Optional[nn.Module] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        target_model = eval_model if eval_model is not None else self.model
        target_model.eval()

        total_loss = 0.0
        batches_run = 0
        metric_totals: Dict[str, float] = {}

        with torch.no_grad():
            for raw_batch in val_loader:
                batch = self.adapter.prepare_batch(raw_batch, self.device)
                with torch.amp.autocast(
                    device_type=self.device.type if self.device.type in ("cuda", "cpu") else "cpu",
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    loss, outputs, targets = self.adapter.forward_and_loss(
                        target_model, batch, self.criterion, engine=self
                    )

                loss_val = float(loss.item())
                total_loss += loss_val
                batches_run += 1

                batch_metrics = self.adapter.compute_batch_metrics(outputs, targets, loss_val, batch)
                for k, v in batch_metrics.items():
                    metric_totals[k] = metric_totals.get(k, 0.0) + v

                if self.config.fast_dev_run:
                    break

        return self._reduce_distributed_metrics(total_loss, batches_run, metric_totals)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        test_loss, test_metrics = self._validate(test_loader)

        metrics: Dict[str, float] = {"test_loss": test_loss}
        if "acc" in test_metrics:
            metrics["test_accuracy"] = test_metrics["acc"]
        for k, v in test_metrics.items():
            metrics[f"test_{k}"] = v

        print("\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        if "acc" in test_metrics:
            print(f"  Accuracy: {test_metrics['acc']:.2f}%")
        for k, v in test_metrics.items():
            if k != "acc":
                print(f"  {k}: {v:.4f}")

        return metrics

    def _save_checkpoint(self, epoch: int, metric_value: float):
        """Save full resumable checkpoint."""
        checkpoint_path = get_checkpoint_path(
            checkpoint_dir=self.config.checkpoint_dir,
            model_name="model",
            epoch=epoch,
            metric_name="loss",
            metric_value=metric_value,
        )

        safe_cfg: Dict[str, Any] = {}
        for k, v in self.config.to_dict().items():
            if isinstance(v, (str, int, float, bool, type(None), dict)):
                safe_cfg[k] = v
            elif isinstance(v, (list, tuple)):
                safe_cfg[k] = [x if isinstance(x, (str, int, float, bool, type(None))) else repr(x) for x in v]
            else:
                safe_cfg[k] = repr(v)

        unwrapped_model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "ema_state_dict": self.ema.state_dict() if self.ema else None,
            "config": safe_cfg,
            "history": self.history,
        }

        atomic_torch_save(checkpoint, checkpoint_path)
        self.logger.info(f"  → Checkpoint saved: {checkpoint_path}")

        cleanup_old_checkpoints(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_top_k=self.config.save_top_k,
            metric_name="loss",
            lower_is_better=True,
        )


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


__all__ = ["Engine", "FocalLoss"]
