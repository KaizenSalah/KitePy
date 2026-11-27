"""
Base model class with shared logic for all modalities.

All public APIs (CNN, Transformer, Vision, LLM, VLM, Audio, Tabular, Generative) inherit from this.
Completely decoupled from any single modality via TaskAdapter and the Pillar Registry.
"""

from collections.abc import Mapping
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from .adapters import TaskAdapter, get_adapter, get_pillar_data_loader, move_to_device
from .config import (
    BaseConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    auto_batch_size,
    merge_configs,
    split_kwargs,
    validate_config,
)
from .presets import get_preset
from .utils import (
    get_device,
    get_gpu_memory,
    get_head_parameter_names,
    get_model_device,
    get_module_blocks,
    print_model_summary,
    set_seed,
)


def _is_single_predict_input(raw_inputs: Any, prepared: Any) -> bool:
    """Determine whether predict() input represents one sample rather than a batch list."""
    if not isinstance(raw_inputs, list):
        return True
    if raw_inputs and isinstance(raw_inputs[0], (int, float, list, tuple)) and isinstance(prepared, torch.Tensor):
        depth = 0
        cur: Any = raw_inputs
        while isinstance(cur, list):
            depth += 1
            cur = cur[0] if cur else None
        return bool(depth < prepared.ndim)
    return False


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors/arrays/scalars/subclasses into pure weights_only-safe Python primitives."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        return str(obj)  # strip str subclasses (e.g., torch.torch_version.TorchVersion)
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    return obj


class BaseModel:
    """
    Universal base class for all KitePy model APIs.

    Provides:
    - Progressive configuration (1-line defaults -> kwargs -> Config objects -> nested YAML)
    - .train() / .evaluate() / .predict()
    - .save() / .load() (including class names & inference metadata)
    - .freeze_backbone() / .unfreeze() (architecture-aware, safe for ViTs & CNNs)
    - .find_lr() / .tune() / .profile() / .compile() / .quantize() / .export()
    - .unwrap() escape hatch to raw PyTorch nn.Module
    """

    modality: str = "vision"
    default_config_class: Type[BaseConfig] = ModelConfig
    default_task: str = "classification"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = None,
        config: Optional[Union[Dict, BaseConfig, ExperimentConfig, str]] = None,
        adapter: Optional[TaskAdapter] = None,
        **kwargs,
    ):
        self.model_name = model if isinstance(model, str) else "custom"
        self.custom_model = model if isinstance(model, nn.Module) else None
        self.adapter = adapter

        self._explicit_model_fields: Set[str] = set()
        self.class_names: Optional[List[str]] = None
        self.last_input_shape: Optional[Tuple[int, ...]] = None

        # Partition kwargs across model, data, and training configs
        model_kw, data_kw, train_kw, unknown_kw = split_kwargs(kwargs, self.default_config_class)
        if unknown_kw:
            # Allow preset/wrapper-specific kwargs to flow into model config if applicable,
            # otherwise raise helpful error via strict validation
            raise ValueError(
                f"Unknown parameter(s) for {self.__class__.__name__}: {list(unknown_kw.keys())}"
            )

        self._explicit_model_fields.update(model_kw.keys())

        # Handle config sources (ExperimentConfig, YAML/JSON file, dict, or ModelConfig)
        user_model_cfg: Optional[Union[Dict[str, Any], BaseConfig]] = None
        user_data_cfg: Optional[Union[Dict[str, Any], BaseConfig]] = None
        user_train_cfg: Optional[Union[Dict[str, Any], BaseConfig]] = None

        if isinstance(config, ExperimentConfig):
            user_model_cfg = config.model
            user_data_cfg = config.data
            user_train_cfg = config.training
        elif isinstance(config, str):
            path_lower = config.lower()
            if path_lower.endswith((".yaml", ".yml")):
                import yaml
                with open(config, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                if any(k in raw for k in ("model", "training", "data")):
                    exp = ExperimentConfig.from_dict(raw, model_config_cls=self.default_config_class, strict=False)  # type: ignore[arg-type]
                    user_model_cfg = exp.model
                    user_data_cfg = exp.data
                    user_train_cfg = exp.training
                    if hasattr(user_model_cfg, "__dict__"):
                        self._explicit_model_fields.update(raw.get("model", {}).keys())
                else:
                    user_model_cfg = self.default_config_class.from_dict(raw, strict=False)
                    self._explicit_model_fields.update(raw.keys())
            elif path_lower.endswith(".json"):
                with open(config, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if any(k in raw for k in ("model", "training", "data")):
                    exp = ExperimentConfig.from_dict(raw, model_config_cls=self.default_config_class, strict=False)  # type: ignore[arg-type]
                    user_model_cfg = exp.model
                    user_data_cfg = exp.data
                    user_train_cfg = exp.training
                else:
                    user_model_cfg = self.default_config_class.from_dict(raw, strict=False)
                    self._explicit_model_fields.update(raw.keys())
            else:
                raise ValueError(f"Unknown config file format: {config}")
        elif isinstance(config, dict):
            if any(k in config for k in ("model", "training", "data")) and all(
                isinstance(config[k], (dict, BaseConfig)) for k in config if k in ("model", "training", "data")
            ):
                exp = ExperimentConfig.from_dict(config, model_config_cls=self.default_config_class, strict=False)  # type: ignore[arg-type]
                user_model_cfg = exp.model
                user_data_cfg = exp.data
                user_train_cfg = exp.training
            else:
                user_model_cfg = config
                self._explicit_model_fields.update(config.keys())
        elif isinstance(config, BaseConfig):
            user_model_cfg = config
            self._explicit_model_fields.add("num_classes")

        self.config: Any = self._build_config(model, user_model_cfg, **model_kw)
        self.data_config: DataConfig = merge_configs(DataConfig(), user_config=user_data_cfg, **data_kw)
        self.train_config: TrainingConfig = merge_configs(TrainingConfig(), user_config=user_train_cfg, **train_kw)

        # Runtime state
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []
        self.execution_report: Dict[str, Any] = {}
        self.best_metric: Optional[float] = None
        self.checkpoint_dir: Optional[str] = None

    def _build_config(
        self,
        model: Optional[Union[str, nn.Module]],
        config: Optional[Union[Dict, BaseConfig]],
        **kwargs,
    ) -> BaseConfig:
        """Build model configuration from defaults, preset, user config, and kwargs."""
        default = self.default_config_class()

        preset = None
        if isinstance(model, str):
            try:
                preset = get_preset(model, self._preset_modality())
            except ValueError:
                pass

        final_config = merge_configs(default, preset, config, **kwargs)
        validate_config(final_config)
        return final_config

    def _preset_modality(self) -> Optional[str]:
        """Return preset registry key for this model class."""
        return None if self.modality == "vision" else self.modality

    def _resolve_pillar_data_loader(self) -> Callable:
        """Look up the modality's data loader via the pillar registry (lazy-importing if needed)."""
        loader_fn = get_pillar_data_loader(self.modality)
        if loader_fn is None:
            try:
                importlib.import_module(f"kitepy.pillars.{self.modality}")
                loader_fn = get_pillar_data_loader(self.modality)
            except ImportError:
                pass
        if loader_fn is None:
            raise NotImplementedError(
                f"No data loader registered for modality '{self.modality}'. "
                "Pass PyTorch DataLoader(s) directly or register a pillar loader."
            )
        return loader_fn

    def _resolve_data(
        self,
        data: Any,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
        """
        Resolve any user data input into (train_loader, val_loader, test_loader, metadata).
        Supports:
        - Dataset preset name or folder/file path (str / Path)
        - (train_loader, val_loader) or (train_loader, val_loader, test_loader) tuples
        - PyTorch DataLoader
        - PyTorch Dataset (auto-split into train/val)
        - Dict / HuggingFace Dataset / custom objects routed to pillar loader
        """
        metadata: Dict[str, Any] = {}

        # 1. Tuple/list of loaders or datasets
        if isinstance(data, tuple) and len(data) in (2, 3) and isinstance(data[0], DataLoader):
            train_loader = data[0]
            val_loader = data[1]
            test_loader = data[2] if len(data) == 3 else None
            return train_loader, val_loader, test_loader, metadata

        # 2. Single PyTorch DataLoader
        if isinstance(data, DataLoader):
            val_loader = data_config.val_data if isinstance(data_config.val_data, DataLoader) else None
            return data, val_loader, None, metadata

        # 3. PyTorch Dataset (auto-split into train/val)
        if isinstance(data, Dataset):
            val_frac = data_config.val_split if isinstance(data_config.val_split, float) else 0.2
            if val_frac > 0 and len(data) >= 5:
                val_len = max(1, int(len(data) * val_frac))
                train_len = len(data) - val_len
                train_ds, val_ds = random_split(
                    data, [train_len, val_len],
                    generator=torch.Generator().manual_seed(train_config.seed),
                )
            else:
                train_ds, val_ds = data, None

            bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 32
            train_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=data_config.shuffle,
                num_workers=train_config.num_workers, pin_memory=train_config.pin_memory,
            )
            val_loader = DataLoader(
                val_ds, batch_size=bs, shuffle=False,
                num_workers=train_config.num_workers, pin_memory=train_config.pin_memory,
            ) if val_ds is not None else None
            if hasattr(data, "classes"):
                metadata["class_names"] = list(data.classes)
                metadata["num_classes"] = len(data.classes)
            return train_loader, val_loader, None, metadata

        # 4. Route through pillar data loader (handles str, Path, dict, HF Dataset, synthetic, etc.)
        data_arg = str(data) if isinstance(data, Path) else data
        if data_arg is None:
            data_arg = data_config.train_data

        # Ensure DataConfig carries model's expected img_size / in_channels / num_classes if set
        if getattr(self.config, "img_size", None) is not None and data_config.img_size is None:
            data_config.img_size = self.config.img_size
        if getattr(self.config, "in_channels", None) is not None and data_config.in_channels is None:
            data_config.in_channels = self.config.in_channels
        if data_config.num_classes is None:
            if "num_classes" in self._explicit_model_fields and getattr(self.config, "num_classes", None) is not None:
                data_config.num_classes = self.config.num_classes
            elif self.custom_model is not None:
                # Infer output class count from custom nn.Module's final Linear layer
                linears = [m for m in self.custom_model.modules() if isinstance(m, nn.Linear)]
                if linears:
                    data_config.num_classes = linears[-1].out_features

        loader_fn = self._resolve_pillar_data_loader()
        result = loader_fn(data_arg, data_config, train_config)

        if isinstance(result, tuple) and len(result) == 4:
            return result
        if isinstance(result, tuple) and len(result) == 3:
            train_l, val_l, test_l = result
            # Try to inspect dataset attributes for class metadata
            ds = getattr(train_l, "dataset", None)
            inner_ds = getattr(ds, "dataset", ds)
            if inner_ds is not None:
                if hasattr(inner_ds, "classes"):
                    metadata["class_names"] = list(inner_ds.classes)
                    metadata["num_classes"] = len(inner_ds.classes)
                elif hasattr(inner_ds, "num_classes"):
                    metadata["num_classes"] = int(inner_ds.num_classes)
            return train_l, val_l, test_l, metadata

        raise ValueError(f"Pillar data loader returned unexpected format: {type(result)}")

    def _auto_align_model_to_data(self, metadata: Dict[str, Any]):
        """Infer num_classes, class_names, etc. from dataset before lazy model build."""
        if metadata.get("class_names") and self.class_names is None:
            self.class_names = metadata["class_names"]

        if self.model is not None or self.custom_model is not None:
            return

        if "num_classes" in metadata and "num_classes" not in self._explicit_model_fields:
            task_hint = (getattr(self, "task", None) or getattr(self.train_config, "task", None) or "").lower()
            if task_hint not in ("binary", "regression") and hasattr(self.config, "num_classes"):
                inferred = int(metadata["num_classes"])
                if self.config.num_classes != inferred:
                    self.config.num_classes = inferred
                    print(f"ℹ️  Auto-detected {inferred} classes from dataset")

        if "num_classes" in metadata and "num_labels" not in self._explicit_model_fields:
            if hasattr(self.config, "num_labels"):
                self.config.num_labels = int(metadata["num_classes"])

        if "in_channels" in metadata and "in_channels" not in self._explicit_model_fields:
            if hasattr(self.config, "in_channels"):
                self.config.in_channels = int(metadata["in_channels"])

        if "input_dim" in metadata and "input_dim" not in self._explicit_model_fields:
            if hasattr(self.config, "input_dim"):
                self.config.input_dim = int(metadata["input_dim"])

    def _resolve_auto_batch_size(self, train_config: TrainingConfig):
        """Resolve batch_size='auto' safely based on hardware."""
        if isinstance(train_config.batch_size, str) and train_config.batch_size.lower() == "auto":
            if torch.cuda.is_available():
                free_gb = get_gpu_memory(0)
                train_config.batch_size = auto_batch_size("base", max(1.0, free_gb))
            else:
                train_config.batch_size = 32
            print(f"ℹ️  Auto-selected batch_size={train_config.batch_size}")

    def _adjust_runtime_defaults(self, train_config: TrainingConfig, explicit_train_kw: Optional[Dict[str, Any]] = None):
        """Normalize CPU and Windows DataLoader defaults safely unless explicitly overridden."""
        if not torch.cuda.is_available():
            if train_config.mixed_precision is True:
                train_config.mixed_precision = False
            train_config.pin_memory = False
            if sys.platform == "win32" and train_config.num_workers > 0 and "num_workers" not in (explicit_train_kw or {}):
                train_config.num_workers = 0

    def train(
        self,
        data: Optional[Any] = None,
        epochs: Optional[int] = None,
        **kwargs,
    ):
        """
        Train the model.

        Args:
            data: Dataset name, folder path, DataLoader, (train_loader, val_loader), Dataset, or "synthetic"
            epochs: Number of epochs (overrides config)
            **kwargs: Override any TrainingConfig or DataConfig parameters
        """
        from .engine import Engine

        if epochs is not None:
            kwargs["epochs"] = epochs

        _, data_kw, train_kw, unknown_kw = split_kwargs(kwargs, self.default_config_class)
        if unknown_kw:
            raise ValueError(
                f"Unknown train() parameter(s): {list(unknown_kw.keys())}"
            )

        train_config: TrainingConfig = merge_configs(self.train_config, **train_kw)
        data_config: DataConfig = merge_configs(self.data_config, **data_kw)
        validate_config(train_config)

        self._adjust_runtime_defaults(train_config, train_kw)

        self._resolve_auto_batch_size(train_config)
        set_seed(train_config.seed, deterministic=train_config.deterministic)

        # Resolve data FIRST so we can auto-align num_classes/in_channels before lazy _build_model()
        train_loader, val_loader, _, metadata = self._resolve_data(data, data_config, train_config)
        self._auto_align_model_to_data(metadata)

        # Initialize model lazily
        if self.model is None:
            self._build_model()
        assert self.model is not None

        if getattr(self.config, "freeze_backbone", False):
            self.freeze_backbone()

        if getattr(train_config, "gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")

        if self.device is None:
            self.device = get_device(train_config.devices)
        self.model = self.model.to(self.device)

        adapter = self.adapter or get_adapter(
            task_or_adapter=train_config.task or getattr(self, "default_task", None),
            modality=self.modality,
            train_step=train_config.train_step,
        )

        engine = Engine(
            model=self.model,
            config=train_config,
            modality=self.modality,
            adapter=adapter,
        )

        self.model, history = engine.train(train_loader=train_loader, val_loader=val_loader)
        self.device = engine.device
        self.training_history.extend(history)
        self.execution_report = engine.execution_report
        self.is_trained = True

        if history:
            last = history[-1]
            best_val = last.get("val_loss")
            if best_val is None:
                best_val = last.get("train_loss", "N/A")
            print(f"\n✓ Training complete! Final metric: {best_val}")

        return self

    def evaluate(
        self,
        data: Any,
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate the model on test/validation data."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call .train() first or load a checkpoint.")

        from .engine import Engine

        _, data_kw, train_kw, _ = split_kwargs(kwargs, self.default_config_class)
        train_config = merge_configs(self.train_config, **train_kw)
        data_config = merge_configs(self.data_config, test_data=data if isinstance(data, str) else None, **data_kw)
        self._adjust_runtime_defaults(train_config, train_kw)

        if isinstance(data, DataLoader):
            test_loader = data
        else:
            train_l, val_l, test_l, _ = self._resolve_data(data, data_config, train_config)
            test_loader = test_l or val_l or train_l

        adapter = self.adapter or get_adapter(
            task_or_adapter=train_config.task or getattr(self, "default_task", None),
            modality=self.modality,
        )

        engine = Engine(
            model=self.model,
            config=train_config,
            modality=self.modality,
            adapter=adapter,
        )
        return engine.evaluate(test_loader)

    def predict(
        self,
        inputs: Any,
        return_probs: bool = False,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Run inference on tensors, numpy arrays, PIL images, image paths, lists, or dicts.

        Returns raw logits tensor by default (backward compatible), or structured
        probabilities / top-k predictions when `return_probs=True` or `top_k` is specified.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Build, train, or load a model first.")

        self.model.eval()
        device = get_model_device(self.model, self.device)

        prepared = self._preprocess_input(inputs)
        prepared = move_to_device(prepared, device)

        if isinstance(prepared, torch.Tensor) and prepared.ndim >= 2:
            self.last_input_shape = tuple(prepared.shape[1:])

        from collections.abc import Mapping

        with torch.no_grad():
            if isinstance(prepared, Mapping):
                out = self.model(**dict(prepared))
                logits = getattr(out, "logits", out)
            else:
                logits = self.model(prepared)

        task_name = (
            getattr(self, "task", None)
            or getattr(self.train_config, "task", None)
            or getattr(self, "default_task", "classification")
            or "classification"
        ).lower()

        single_input = _is_single_predict_input(inputs, prepared)

        if not isinstance(logits, torch.Tensor):
            if isinstance(logits, list) and top_k is not None:
                formatted = []
                for det in logits:
                    if isinstance(det, Mapping) and "scores" in det:
                        scores_t = det["scores"]
                        k = min(top_k, len(scores_t))
                        order = torch.argsort(scores_t, descending=True)[:k]
                        formatted.append({dk: _to_jsonable(dv[order]) if isinstance(dv, torch.Tensor) else _to_jsonable(dv) for dk, dv in det.items()})
                    else:
                        formatted.append(_to_jsonable(det))
                return formatted[0] if len(formatted) == 1 and single_input else formatted
            return _to_jsonable(logits) if return_probs else logits

        if task_name == "regression":
            if top_k is not None:
                vals = logits.squeeze(-1).detach().cpu().tolist() if (logits.ndim == 2 and logits.size(-1) == 1) else logits.detach().cpu().tolist()
                if isinstance(vals, list) and len(vals) == 1 and single_input:
                    return vals[0]
                return vals
            return logits

        if top_k is not None:
            if task_name == "binary" and (logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1)):
                p_pos = torch.sigmoid(logits.reshape(-1, 1))
                probs = torch.cat([1.0 - p_pos, p_pos], dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
            k = min(top_k, probs.size(-1))
            values, indices = torch.topk(probs, k=k, dim=-1)
            results = []
            for row_vals, row_idxs in zip(values.cpu().tolist(), indices.cpu().tolist()):
                row_preds = []
                for score, idx in zip(row_vals, row_idxs):
                    label = self.class_names[idx] if (self.class_names and idx < len(self.class_names)) else str(idx)
                    row_preds.append({"class_id": idx, "label": label, "confidence": float(score)})
                results.append(row_preds)
            return results[0] if len(results) == 1 and single_input else results

        if return_probs:
            if task_name == "binary" and (logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1)):
                p_pos = torch.sigmoid(logits.reshape(-1, 1))
                return torch.cat([1.0 - p_pos, p_pos], dim=-1)
            return torch.softmax(logits, dim=-1)

        return logits

    def _extra_save_metadata(self) -> Dict[str, Any]:
        """Optional hook for subclasses to persist extra JSON-safe metadata."""
        return {}

    def _load_extra_metadata(self, checkpoint: Mapping[str, Any]) -> None:
        """Optional hook for subclasses to restore extra metadata from checkpoint."""
        return None

    def _save_external_artifacts(self, path: Path) -> None:
        """Optional hook for subclasses to persist external directories (e.g. PEFT adapters)."""
        return None

    def _load_external_artifacts(self, path: Path, checkpoint: Mapping[str, Any]) -> bool:
        """Optional hook returning True if model weights were already loaded externally."""
        return False

    def save(self, path: Union[str, Path], save_config: bool = True, safe: Optional[bool] = None):
        """
        Save model checkpoint atomically with lineage metadata.

        If `safe=True` or `path` ends with `.safetensors`, saves pickle-free `safetensors` + `.meta.json`.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        from .serialization import atomic_torch_save, get_lineage_metadata, save_safe_bundle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        use_safe = safe if safe is not None else (path.suffix.lower() == ".safetensors" or getattr(self.train_config, "safe_serialization", False))

        train_cfg_dict = {}
        for k, v in self.train_config.to_dict().items():
            if k in ("callbacks", "loss", "optimizer", "scheduler", "train_step", "class_weights") and not isinstance(
                v, (str, int, float, bool, type(None))
            ):
                train_cfg_dict[k] = repr(v) if not isinstance(v, list) else [repr(x) for x in v]
            else:
                train_cfg_dict[k] = v

        self._save_external_artifacts(path)

        metadata = _to_jsonable({
            "model_name": self.model_name,
            "modality": self.modality,
            "task": getattr(self, "task", getattr(self.train_config, "task", getattr(self, "default_task", None))),
            "config": self.config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "train_config": train_cfg_dict,
            "class_names": self.class_names,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
            "last_input_shape": list(self.last_input_shape) if self.last_input_shape is not None else None,
            "lineage": get_lineage_metadata(),
            **self._extra_save_metadata(),
        })

        if use_safe:
            st_path, meta_path = save_safe_bundle(self.model.state_dict(), metadata, path)
            print(f"✓ Safe model bundle saved to {st_path} (+ {meta_path.name})")
        else:
            checkpoint = {"model_state_dict": self.model.state_dict(), **metadata}
            atomic_torch_save(checkpoint, path)
            print(f"✓ Model saved to {path}")

        if save_config:
            config_path = path.with_suffix(".yaml")
            self.config.to_yaml(str(config_path))
            print(f"✓ Config saved to {config_path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
        custom_model: Optional[nn.Module] = None,
        allow_unsafe: bool = False,
    ):
        """
        Load model from `.safetensors` safe bundle or `.pt` checkpoint.

        For custom nn.Module architectures, pass `custom_model=YourModule()`.
        By default rejects untrusted pickle `.pt` files (`weights_only=True`); pass
        `allow_unsafe=True` only for trusted legacy checkpoints.
        """
        from .serialization import load_safe_bundle

        p = Path(path)
        if p.suffix.lower() == ".safetensors":
            state_dict, checkpoint = load_safe_bundle(p)
            checkpoint["model_state_dict"] = state_dict
        else:
            try:
                checkpoint = torch.load(p, map_location="cpu", weights_only=not allow_unsafe)
            except Exception as e:
                raise RuntimeError(
                    f"Refusing to unpickle checkpoint '{p}' with weights_only=False ({type(e).__name__}: {e}). "
                    "Re-save as a `.safetensors` safe bundle, or pass `allow_unsafe=True` only if you trust the source."
                ) from e

        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Invalid checkpoint payload type {type(checkpoint).__name__} in '{p}'.")

        if "config" not in checkpoint and custom_model is None:
            meta_hint = f" (missing sidecar '{p.with_suffix('.meta.json')}')" if p.suffix.lower() == ".safetensors" else ""
            raise ValueError(
                f"Cannot reconstruct model from weights-only checkpoint '{p}'{meta_hint}: "
                "required metadata key 'config' is absent. Save with `model.save(...)` or pass `custom_model=...`."
            )

        model_ref = custom_model if custom_model is not None else checkpoint.get("model_name")

        if model_ref == "custom":
            raise ValueError(
                "Checkpoint was saved from a custom nn.Module. "
                "Pass `custom_model=YourNet()` to load()."
            )

        cfg_dict = checkpoint.get("config") or {}
        saved_task = checkpoint.get("task") or (checkpoint.get("train_config") or {}).get("task")
        init_params = inspect.signature(cls.__init__).parameters
        if "task" in init_params and saved_task is not None:
            instance = cls(model=model_ref, config=cfg_dict, task=saved_task)
        else:
            instance = cls(model=model_ref, config=cfg_dict)
        if checkpoint.get("data_config"):
            instance.data_config = DataConfig.from_dict(checkpoint["data_config"], strict=False)
        if checkpoint.get("train_config"):
            instance.train_config = TrainingConfig.from_dict(checkpoint["train_config"], strict=False)

        instance.class_names = checkpoint.get("class_names")
        last_shape = checkpoint.get("last_input_shape")
        instance.last_input_shape = tuple(last_shape) if isinstance(last_shape, (list, tuple)) else None
        instance._load_extra_metadata(checkpoint)

        if not instance._load_external_artifacts(p, checkpoint):
            instance._build_model()
            assert instance.model is not None
            instance.model.load_state_dict(checkpoint["model_state_dict"])

        assert instance.model is not None
        instance.is_trained = checkpoint.get("is_trained", True)
        instance.training_history = checkpoint.get("training_history", [])

        if device:
            instance.device = get_device(device)
            instance.model = instance.model.to(instance.device)

        print(f"✓ Model loaded from {p}")
        return instance

    def describe(self):
        """Print model architecture and configuration."""
        print(f"\n{'=' * 70}")
        print(f"Model: {self.__class__.__name__}")
        print(f"Name: {self.model_name}")
        print(f"Modality: {self.modality}")
        print(f"Trained: {self.is_trained}")
        print(f"{'=' * 70}")
        print("\n📋 Configuration:")
        print(json.dumps(self.config.to_dict(), indent=2, default=str))

        if self.model is not None:
            print("\n🏗️  Architecture:")
            print_model_summary(self.model)
        else:
            print("\n⚠️  Model not initialized yet")

    def summary(self):
        """Print model parameter count and size."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        print_model_summary(self.model)

    def explain_config(self):
        """Print all available configuration options."""
        print(f"\n{'=' * 70}")
        print(f"Configuration Options for {self.__class__.__name__}")
        print(f"{'=' * 70}\n")
        default_config = self.default_config_class()
        print("You can override any of these values:")
        print(json.dumps(default_config.to_dict(), indent=2, default=str))

    def compile(self, mode: str = "default"):
        """Compile model with torch.compile (PyTorch 2.0+)."""
        if self.model is None:
            self._build_model()
        try:
            torch_version = torch.__version__.split("+")[0]
            major, _ = map(int, torch_version.split(".")[:2])
            if major < 2:
                print(f"⚠️  torch.compile requires PyTorch 2.0+, you have {torch_version}")
                return self
            print(f"🔧 Compiling model with mode='{mode}'...")
            self.model = torch.compile(self.model, mode=mode)
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")
        return self

    def quantize(self, bits: int = 8):
        """Quantize model for faster CPU inference (prefers torchao when available)."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _build_model() first.")
        try:
            if bits == 8:
                try:
                    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
                    quantize_(self.model, Int8DynamicActivationInt8WeightConfig())
                    print("✓ Model quantized to INT8 (torchao)")
                except ImportError:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        quant_fn = getattr(torch.ao.quantization, "quantize_dynamic", torch.quantization.quantize_dynamic)
                        self.model = quant_fn(self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
                    print("✓ Model quantized to INT8")
            else:
                print(f"⚠️  {bits}-bit quantization requires specialized libraries (bitsandbytes)")
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
        return self

    def freeze_backbone(self):
        """
        Freeze all backbone parameters cleanly while keeping the classifier/head trainable.
        Uses `timm.get_classifier()` / structural head detection (safe for ViT MLP fc1/fc2!).
        """
        if self.model is None:
            self._build_model()

        head_names = get_head_parameter_names(self.model)
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if name in head_names:
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        print(f"✓ Froze {frozen_count} backbone parameter tensors")
        if total > 0:
            print(f"  - Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
        return self

    def unfreeze(self, layers: Optional[int] = None, pattern: Optional[str] = None):
        """
        Unfreeze model layers for fine-tuning.

        Args:
            layers: Number of logical blocks/layers from the end to unfreeze (None = all)
            pattern: Optional substring/regex pattern to selectively unfreeze matching params
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        if pattern is not None:
            count = 0
            for name, param in self.model.named_parameters():
                if pattern in name:
                    param.requires_grad = True
                    count += 1
            print(f"✓ Unfroze {count} parameters matching '{pattern}'")
            return self

        if layers is None:
            for param in self.model.parameters():
                param.requires_grad = True
            print("✓ Unfroze all layers")
        else:
            blocks = get_module_blocks(self.model)
            if blocks:
                for block in blocks[-layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
            else:
                params = list(self.model.named_parameters())
                for _, param in params[-layers:]:
                    param.requires_grad = True
            print(f"✓ Unfroze last {layers} blocks/layers")

        return self

    def _default_input_shape(self) -> Tuple[int, ...]:
        """Infer expected single-sample input shape from model config or last seen input."""
        if self.last_input_shape is not None:
            return (1, *self.last_input_shape)
        c = getattr(self.config, "in_channels", 3)
        img_size = getattr(self.config, "img_size", 224)
        return (1, c, img_size, img_size)

    def profile(self, input_size: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """Profile model: parameters, memory, latency, throughput."""
        if self.model is None:
            self._build_model()
        assert self.model is not None

        if input_size is None:
            input_size = self._default_input_shape()

        print("\n" + "=" * 60)
        print("📊 Model Profile")
        print("=" * 60)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 * 1024)

        device = self.device if self.device else torch.device("cpu")
        self.model = self.model.to(device)
        self.model.eval()

        dummy_input = torch.randn(*input_size, device=device)

        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)

        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                times.append(time.perf_counter() - start)

        avg_latency = (sum(times) / len(times)) * 1000.0
        throughput = (1000.0 / avg_latency) * input_size[0] if avg_latency > 0 else 0.0

        print(f"\n📦 Parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"💾 Model size: {total_size_mb:.2f} MB")
        print(f"⚡ Latency ({device}): {avg_latency:.2f} ms | Throughput: {throughput:.1f} samples/sec")
        print("=" * 60)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": total_size_mb,
            "latency_ms": avg_latency,
            "throughput": throughput,
        }

    def export(
        self,
        path: Union[str, Path],
        format: str = "auto",
        input_shape: Optional[Tuple[int, ...]] = None,
        verify: bool = False,
    ):
        """
        Export model to TorchScript, ONNX, or SafeTensors.
        Automatically infers input_shape and handles device placement safely.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if format == "auto":
            if path.endswith(".onnx"):
                format = "onnx"
            elif path.endswith(".safetensors"):
                format = "safetensors"
            else:
                format = "torchscript"

        self.model.eval()
        original_device = get_model_device(self.model)
        shape = input_shape or self._default_input_shape()

        if format == "onnx":
            try:
                import inspect
                cpu_model = self.model.cpu()
                dummy_input = torch.randn(*shape)
                onnx_kwargs = {
                    "export_params": True,
                    "opset_version": 14,
                    "do_constant_folding": True,
                    "input_names": ["input"],
                    "output_names": ["output"],
                    "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                }
                # PyTorch 2.10+ defaults to dynamo=True requiring onnxscript; use legacy exporter for zero-extra-dep reliability
                if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                    onnx_kwargs["dynamo"] = False
                torch.onnx.export(cpu_model, dummy_input, path, **onnx_kwargs)
                print(f"✓ Exported to ONNX: {path}")
            except Exception as e:
                print(f"⚠️  ONNX export failed: {e}")
            finally:
                self.model = self.model.to(original_device)

        elif format == "torchscript":
            try:
                dummy_input = torch.randn(*shape, device=original_device)
                traced = torch.jit.trace(self.model, dummy_input)
                traced.save(path)
                print(f"✓ Exported to TorchScript: {path}")

                if verify:
                    loaded = torch.jit.load(path, map_location=original_device)
                    with torch.no_grad():
                        ref_out = self.model(dummy_input)
                        ts_out = loaded(dummy_input)
                    max_diff = (ref_out - ts_out).abs().max().item()
                    print(f"✓ Verified TorchScript export (max diff: {max_diff:.2e})")
            except Exception as e:
                print(f"⚠️  TorchScript export failed: {e}")

        elif format == "safetensors":
            try:
                from safetensors.torch import save_file
                from .serialization import prepare_safetensors_state_dict
                tensors = prepare_safetensors_state_dict(self.model.state_dict())
                save_file(tensors, path)
                print(f"✓ Exported weights to SafeTensors: {path}")
            except ImportError:
                raise ImportError("safetensors required. Install with: pip install safetensors")
        else:
            raise ValueError(f"Unknown format: {format}")

        return self

    def find_lr(self, data: Any = "synthetic", **kwargs) -> float:
        """Find optimal learning rate using the LR Range Test."""
        from .engine import create_loss_function
        from .lr_finder import LRFinder

        _, data_kw, train_kw, _ = split_kwargs(kwargs, self.default_config_class)
        train_config = merge_configs(self.train_config, **train_kw)
        data_config = merge_configs(self.data_config, **data_kw)
        self._adjust_runtime_defaults(train_config, train_kw)

        train_loader, _, _, metadata = self._resolve_data(data, data_config, train_config)
        self._auto_align_model_to_data(metadata)

        if self.model is None:
            self._build_model()
        assert self.model is not None

        device = get_device(train_config.devices)
        self.model = self.model.to(device)

        adapter = self.adapter or get_adapter(
            task_or_adapter=train_config.task or getattr(self, "default_task", None),
            modality=self.modality,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-7)
        criterion = create_loss_function(
            loss=getattr(train_config, "loss", "cross_entropy"),
            class_weights=getattr(train_config, "class_weights", None),
            label_smoothing=getattr(train_config, "label_smoothing", 0.0),
            device=device,
        )

        lr_finder = LRFinder(self.model, optimizer, criterion, device, adapter=adapter)
        lr_finder.find(train_loader)
        return lr_finder.suggest_lr()

    def tune(self, data: Any, trials: int = 20, **kwargs) -> Dict[str, Any]:
        """Hyperparameter tuning using Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna required. Install with: pip install optuna")

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

            model_copy = deepcopy(self)
            model_copy.model = None
            try:
                model_copy.train(
                    data=data,
                    epochs=kwargs.get("epochs", 3),
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    fast_dev_run=False,
                    **{k: v for k, v in kwargs.items() if k not in ("epochs", "lr", "batch_size", "weight_decay")},
                )
                if model_copy.training_history:
                    losses = [
                        h.get("val_loss") if h.get("val_loss") is not None else h.get("train_loss")
                        for h in model_copy.training_history
                    ]
                    valid = [x for x in losses if x is not None]
                    return min(valid) if valid else float("inf")
            except Exception as e:
                print(f"Trial failed: {e}")
                return float("inf")
            return float("inf")

        print(f"\n🔍 Starting hyperparameter search ({trials} trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)

        print("\n✓ Best hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print(f"   Best loss: {study.best_value:.4f}")
        best = study.best_params
        return dict(best)

    def to_fastapi(self):
        """Create a ready-to-run FastAPI inference app for this model."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel as PydanticBaseModel
        except ImportError:
            raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")

        if self.model is None:
            self._build_model()

        app = FastAPI(title=f"KitePy Inference API ({self.model_name})")

        class PredictRequest(PydanticBaseModel):
            inputs: Any
            top_k: Optional[int] = 5

        @app.get("/health")
        def health():
            return {"status": "ok", "model": self.model_name, "modality": self.modality}

        @app.post("/predict")
        def predict_endpoint(req: PredictRequest):
            try:
                raw_inputs = req.inputs
                if not isinstance(raw_inputs, (str, Mapping)):
                    try:
                        # Convert numeric JSON arrays back to Tensor to preserve single-item unwrapping
                        raw_inputs = torch.tensor(raw_inputs, dtype=torch.float32)
                    except (TypeError, ValueError, RuntimeError):
                        pass
                preds = self.predict(raw_inputs, top_k=req.top_k)
                return {"predictions": _to_jsonable(preds)}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        return app

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload KitePy model",
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """
        Push trained model checkpoint, YAML config, and model card to HuggingFace Hub.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

        if self.model is None:
            raise RuntimeError("No model to push. Build or train a model first.")

        import tempfile
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ckpt_path = tmp_dir / "model.pt"
            self.save(ckpt_path, save_config=True)

            card = tmp_dir / "README.md"
            card.write_text(
                f"---\nlibrary_name: kitepy\ntags:\n  - kitepy\n  - {self.modality}\n---\n\n"
                f"# {repo_id}\n\nTrained with **KitePy** (`{self.__class__.__name__}` / `{self.model_name}`).\n",
                encoding="utf-8",
            )
            url = api.upload_folder(
                folder_path=str(tmp_dir),
                repo_id=repo_id,
                commit_message=commit_message,
            )
        print(f"✓ Pushed model to HuggingFace Hub: {repo_id}")
        return str(url)

    def unwrap(self) -> nn.Module:
        """Escape hatch: return the underlying PyTorch nn.Module."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        return self.model

    def _build_model(self):
        raise NotImplementedError("Subclasses must implement _build_model()")

    def _preprocess_input(self, input_data: Any) -> Any:
        return input_data

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, trained={self.is_trained})"


def compare(
    models: List[Union[str, BaseModel]],
    data: Any,
    epochs: int = 3,
    cls: Optional[type] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Benchmark multiple model architectures side-by-side on the same dataset.

    Example:
        >>> results = kitepy.compare(["resnet18", "mobilenet_v3_small"], data="synthetic", epochs=1)
    """
    if cls is None:
        from kitepy.pillars.vision import CNN
        cls = CNN

    results: List[Dict[str, Any]] = []
    print("\n" + "=" * 75)
    print(f"🏁 KitePy Benchmark Comparison ({len(models)} models)")
    print("=" * 75)

    for m in models:
        instance = m if isinstance(m, BaseModel) else cls(m)
        t0 = time.perf_counter()
        instance.train(data, epochs=epochs, **kwargs)
        elapsed = time.perf_counter() - t0
        prof = instance.profile()
        last = instance.training_history[-1] if instance.training_history else {}
        results.append({
            "model": instance.model_name,
            "params": prof["total_params"],
            "size_mb": prof["model_size_mb"],
            "latency_ms": prof["latency_ms"],
            "train_loss": last.get("train_loss"),
            "val_loss": last.get("val_loss"),
            "val_acc": last.get("val_acc"),
            "train_time_s": elapsed,
            "instance": instance,
        })

    print("\n" + "-" * 75)
    print(f" {'Model':<22} {'Params':>10} {'Latency':>10} {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 75)
    for r in results:
        vloss = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "N/A"
        vacc = f"{r['val_acc']:.2f}%" if r["val_acc"] is not None else "N/A"
        print(f" {r['model']:<22} {r['params']:>10,} {r['latency_ms']:>8.1f}ms {vloss:>10} {vacc:>10}")
    print("-" * 75 + "\n")
    return results


__all__ = ["BaseModel", "compare"]
