"""
Configuration system for kitepy.

All models use these config dataclasses for:
- Strong defaults
- Type checking & helpful typo hints
- Easy serialization (nested & flat YAML/JSON)
- Clean routing across model, data, and training configs
"""

from dataclasses import dataclass, field, fields, asdict
from difflib import get_close_matches
from typing import Optional, Union, Literal, Any, Dict, List, Tuple, Type
import yaml
import json


# ============================================================================
# BASE CONFIGS
# ============================================================================

@dataclass
class BaseConfig:
    """Base configuration class with serialization and safe deserialization."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str):
        """Atomically save config to YAML file."""
        from .serialization import atomic_write_text
        atomic_write_text(yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False), path)

    def to_json(self, path: str):
        """Atomically save config to JSON file."""
        from .serialization import atomic_write_text
        atomic_write_text(json.dumps(self.to_dict(), indent=2), path)

    @classmethod
    def field_names(cls) -> set:
        """Return set of valid dataclass field names."""
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], strict: bool = True):
        """Create config from dictionary with helpful error messages for unknown keys."""
        if config_dict is None:
            return cls()
        valid_fields = cls.field_names()
        unknown = [k for k in config_dict if k not in valid_fields]
        if unknown:
            if strict:
                hints = []
                for k in unknown:
                    matches = get_close_matches(k, valid_fields, n=2, cutoff=0.6)
                    if matches:
                        hints.append(f"'{k}' (did you mean '{matches[0]}'?)")
                    else:
                        hints.append(f"'{k}'")
                raise ValueError(
                    f"Unknown {cls.__name__} parameter(s): {', '.join(hints)}"
                )
            config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str, strict: bool = True):
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls.from_dict(config_dict, strict=strict)

    @classmethod
    def from_json(cls, path: str, strict: bool = True):
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, strict=strict)


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """Universal training configuration for all modalities."""

    # Training duration
    epochs: int = 10
    max_steps: Optional[int] = None  # Stop after N optimizer steps if set

    # Batch size & gradient accumulation
    batch_size: Union[int, str] = 32  # int or "auto"
    gradient_accumulation_steps: int = 1

    # Learning rate & LR Finder
    lr: Union[float, str, Dict[str, float]] = 1e-3  # float, "auto", or {"backbone": ..., "head": ...}
    backbone_lr_mult: Optional[float] = None  # Differential LR multiplier for backbone
    layer_decay: Optional[float] = None  # Layer-wise LR decay (e.g., 0.75)
    lr_find: bool = False
    lr_find_min: float = 1e-7
    lr_find_max: float = 10.0
    lr_find_num_steps: int = 100
    weight_decay: float = 0.01
    warmup_steps: Union[int, float] = 0  # int steps or float fraction (0.1 = 10%)

    # Optimizer
    optimizer: Union[str, Any] = "adamw"  # "adam", "adamw", "sgd", "lion", etc. or class/instance
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Scheduler
    scheduler: Union[str, Any] = "cosine"  # "cosine", "linear", "step", "onecycle", "plateau", etc.
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Regularization
    max_grad_norm: float = 1.0
    dropout: float = 0.0
    label_smoothing: float = 0.0

    # Exponential Moving Average (EMA)
    ema: bool = False
    ema_decay: float = 0.9999

    # Precision & Compilation
    mixed_precision: Union[bool, str] = True  # True/"fp16", "bf16", False/"fp32"
    amp_dtype: Literal["auto", "fp16", "bf16"] = "auto"
    compile: bool = False
    compile_mode: str = "default"

    # Distributed training
    strategy: Literal["auto", "ddp", "fsdp", "deepspeed"] = "auto"
    use_accelerate: bool = False
    num_nodes: int = 1
    devices: Union[int, str] = "auto"  # "auto", "cpu", "cuda", "mps", or GPU index

    # Checkpointing
    save_every_n_epochs: Optional[int] = 1
    save_top_k: int = 3
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # Logging & Validation
    log_every_n_steps: int = 50
    val_check_interval: Union[int, float] = 1.0  # float fraction of epoch or int steps
    val_every_n_epochs: int = 1
    csv_log_dir: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Performance
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Debugging
    fast_dev_run: bool = False
    overfit_batches: Union[int, float] = 0
    detect_anomaly: bool = False

    # Gradient checkpointing
    gradient_checkpointing: bool = False

    # Task & Loss
    task: Optional[str] = None  # "classification", "binary", "multilabel", "regression", "causal_lm", etc.
    loss: Union[str, Any] = "cross_entropy"
    class_weights: Optional[Union[List[float], str, Any]] = None  # list, Tensor, or "balanced"

    # Vision Batch Augmentation
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # Execution Planning, Fault Recovery & Enterprise Logging
    auto_plan: bool = False  # Pre-flight memory-constrained micro-batch & precision planner
    memory_budget_mb: Optional[float] = None  # Explicit VRAM/RAM budget in MB
    auto_recover: bool = True  # Runtime OOM micro-batch splitting + NaN/Inf gradient guard
    max_consecutive_nan_steps: int = 5
    log_level: str = "INFO"
    structured_log_file: Optional[str] = None
    safe_serialization: bool = False  # Prefer pickle-free safetensors + JSON metadata

    # Custom hooks / Callbacks
    callbacks: Optional[List[Any]] = None
    train_step: Optional[Any] = None  # Optional custom callable(engine, batch) -> (loss, outputs, targets)


# ============================================================================
# DATA CONFIG
# ============================================================================

@dataclass
class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration."""

    # Data source
    train_data: Optional[Any] = None
    val_data: Optional[Any] = None
    test_data: Optional[Any] = None
    data_dir: str = "./data"

    # Data splits (if single dataset)
    train_split: Union[str, float] = "train"
    val_split: Union[str, float] = 0.2
    test_split: Union[str, float] = "test"

    # Vision / text parameters
    img_size: Optional[int] = None
    in_channels: Optional[int] = None
    num_classes: Optional[int] = None
    max_length: int = 512

    # Preprocessing & augmentation
    preprocessing: str = "auto"
    augmentation: str = "auto"  # "auto", "none", "light", "heavy", "randaugment", "trivialaugment", "autoaugment"

    # Sampling
    shuffle: bool = True
    drop_last: bool = False

    # Streaming (for large datasets)
    streaming: bool = False
    buffer_size: int = 10000


# ============================================================================
# MODEL CONFIGS
# ============================================================================

@dataclass
class ModelConfig(BaseConfig):
    """Base model configuration."""

    # Pretrained weights
    pretrained: bool = False
    pretrained_path: Optional[str] = None

    # Model modifications
    freeze_backbone: bool = False
    freeze_embeddings: bool = False

    # Regularization
    dropout: float = 0.0
    drop_path: float = 0.0


@dataclass
class CNNConfig(ModelConfig):
    """Configuration for CNN models (ResNet, VGG, EfficientNet, etc.)."""

    arch: str = "resnet"
    depth: Union[int, str] = 18
    width_multiplier: float = 1.0
    variant: Optional[str] = None

    # Task
    num_classes: int = 1000
    in_channels: int = 3
    img_size: int = 224

    # Architecture-specific
    groups: int = 1
    base_width: int = 64

    # Advanced
    global_pool: str = "avg"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Vision Transformers (ViT, Swin, DeiT, etc.)."""

    arch: str = "vit"

    # Image specs
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Transformer specs
    embed_dim: int = 768
    depth: Union[int, List[int]] = 12
    depths: Optional[List[int]] = None
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Task
    num_classes: int = 1000

    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Position embedding
    pos_embed_type: str = "learned"


@dataclass
class LLMConfig(ModelConfig):
    """Configuration for Language Models (GPT, LLaMA, BERT, etc.)."""

    arch: str = "gpt"
    model_type: str = "causal_lm"  # "causal_lm", "classification", "masked_lm", "seq2seq"

    # Model size
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024

    # Vocabulary
    vocab_size: int = 50257
    context_length: int = 512
    num_labels: int = 2

    # Attention
    attention_type: str = "multihead"
    n_kv_heads: Optional[int] = None

    # Positional encoding
    pos_encoding: str = "learned"

    # Normalization & Activation
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    activation: str = "gelu"

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Initialization & LoRA
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    trust_remote_code: bool = False


@dataclass
class VLMConfig(ModelConfig):
    """Configuration for Vision-Language Models (CLIP, BLIP, etc.)."""

    arch: str = "clip"
    vision_model: str = "resnet18"
    text_model: str = "gpt2"
    vision_config: Optional[TransformerConfig] = None
    text_config: Optional[LLMConfig] = None

    # Fusion
    fusion_type: str = "contrastive"
    projection_dim: int = 256
    img_size: int = 224
    vocab_size: int = 1000
    max_length: int = 64

    # Task & Loss
    task: str = "contrastive"
    temperature: float = 0.07


@dataclass
class AudioConfig(ModelConfig):
    """Configuration for Audio / Speech models."""

    arch: str = "audio_cnn"
    num_classes: int = 10
    in_channels: int = 1
    sample_rate: int = 16000
    n_mels: int = 64
    num_frames: int = 128
    hidden_dim: int = 128


@dataclass
class TabularConfig(ModelConfig):
    """Configuration for Tabular neural models (MLP, ResNet-Tabular)."""

    arch: str = "tabular_resnet"
    input_dim: int = 16
    num_classes: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    dropout: float = 0.1
    task: str = "classification"  # "classification" or "regression"


@dataclass
class GenerativeConfig(ModelConfig):
    """Configuration for Generative models (VAE, DDPM)."""

    arch: str = "vae"  # "vae" or "ddpm"
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 64
    base_channels: int = 32
    timesteps: int = 200
    kl_weight: float = 0.001


@dataclass
class RNNConfig(ModelConfig):
    """Configuration for RNN models (LSTM, GRU)."""

    rnn_type: str = "lstm"
    input_size: int = 768
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    bidirectional: bool = False
    num_classes: int = 1000


# ============================================================================
# COMBINED CONFIG
# ============================================================================

MODEL_CONFIG_MAP: Dict[str, Type[ModelConfig]] = {
    "cnn": CNNConfig,
    "transformer": TransformerConfig,
    "llm": LLMConfig,
    "vlm": VLMConfig,
    "audio": AudioConfig,
    "tabular": TabularConfig,
    "generative": GenerativeConfig,
    "rnn": RNNConfig,
}


@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration combining model, training, and data sub-configs."""

    model: ModelConfig = field(default_factory=CNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment metadata
    experiment_name: str = "default_experiment"
    tags: list = field(default_factory=list)
    notes: str = ""

    # Tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        model_config_cls: Optional[Type[ModelConfig]] = None,
        strict: bool = True,
    ):
        """Construct ExperimentConfig from a possibly nested dictionary."""
        if config_dict is None:
            return cls()

        data = dict(config_dict)
        model_cls = model_config_cls or CNNConfig

        # Support nested 'model', 'training', 'data' dicts
        model_raw = data.pop("model", None)
        training_raw = data.pop("training", None)
        data_raw = data.pop("data", None)

        if isinstance(model_raw, dict):
            model_cfg = model_cls.from_dict(model_raw, strict=strict)
        elif isinstance(model_raw, BaseConfig):
            model_cfg = model_raw
        else:
            model_cfg = model_cls()

        if isinstance(training_raw, dict):
            training_cfg = TrainingConfig.from_dict(training_raw, strict=strict)
        elif isinstance(training_raw, TrainingConfig):
            training_cfg = training_raw
        else:
            training_cfg = TrainingConfig()

        if isinstance(data_raw, dict):
            data_cfg = DataConfig.from_dict(data_raw, strict=strict)
        elif isinstance(data_raw, DataConfig):
            data_cfg = data_raw
        else:
            data_cfg = DataConfig()

        base_fields = {f.name for f in fields(cls)} - {"model", "training", "data"}
        top_kwargs = {k: v for k, v in data.items() if k in base_fields}
        unknown = [k for k in data if k not in base_fields]
        if unknown and strict:
            raise ValueError(f"Unknown ExperimentConfig field(s): {unknown}")

        return cls(model=model_cfg, training=training_cfg, data=data_cfg, **top_kwargs)

    @classmethod
    def from_yaml(cls, path: str, model_config_cls: Optional[Type[ModelConfig]] = None, strict: bool = True):
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls.from_dict(config_dict, model_config_cls=model_config_cls, strict=strict)


# ============================================================================
# CONFIG MERGING & ROUTING UTILITIES
# ============================================================================

def merge_configs(
    default: BaseConfig,
    preset: Optional[Dict[str, Any]] = None,
    user_config: Optional[Union[Dict[str, Any], BaseConfig]] = None,
    strict: bool = False,
    **kwargs
) -> BaseConfig:
    """
    Merge configurations with priority: kwargs > user_config > preset > default.
    """
    config_dict = default.to_dict()

    if preset:
        for k, v in preset.items():
            if k in config_dict:
                config_dict[k] = v

    if user_config:
        if isinstance(user_config, BaseConfig):
            user_dict = user_config.to_dict()
        else:
            user_dict = dict(user_config)
        for k, v in user_dict.items():
            if strict and k not in config_dict:
                raise ValueError(f"Unknown config key '{k}' for {type(default).__name__}")
            if k in config_dict:
                config_dict[k] = v

    for k, v in kwargs.items():
        if strict and k not in config_dict:
            matches = get_close_matches(k, config_dict.keys(), n=1, cutoff=0.6)
            hint = f" (did you mean '{matches[0]}'?)" if matches else ""
            raise ValueError(f"Unknown config parameter '{k}'{hint} for {type(default).__name__}")
        if k in config_dict:
            config_dict[k] = v

    return type(default).from_dict(config_dict, strict=False)


def split_kwargs(
    kwargs: Dict[str, Any],
    model_config_cls: Type[BaseConfig],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Partition flat kwargs into (model_kwargs, data_kwargs, train_kwargs, unknown_kwargs).
    """
    model_fields = {f.name for f in fields(model_config_cls)}
    data_fields = DataConfig.field_names()
    train_fields = TrainingConfig.field_names()

    model_kw, data_kw, train_kw, unknown_kw = {}, {}, {}, {}
    for k, v in kwargs.items():
        matched = False
        if k in model_fields:
            model_kw[k] = v
            matched = True
        if k in data_fields:
            data_kw[k] = v
            matched = True
        if k in train_fields:
            train_kw[k] = v
            matched = True
        if not matched:
            unknown_kw[k] = v

    return model_kw, data_kw, train_kw, unknown_kw


def validate_config(config: BaseConfig) -> bool:
    """Validate configuration values."""
    if isinstance(config, TrainingConfig):
        if config.epochs <= 0 and config.max_steps is None:
            raise ValueError("Either epochs or max_steps must be positive")

        if isinstance(config.batch_size, int) and config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if config.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        if isinstance(config.lr, (int, float)) and config.lr <= 0:
            raise ValueError("learning rate must be positive")
        elif isinstance(config.lr, dict):
            for group_name, val in config.lr.items():
                if val <= 0:
                    raise ValueError(f"learning rate for '{group_name}' must be positive")

    elif isinstance(config, DataConfig):
        if isinstance(config.val_split, float) and not (0.0 <= config.val_split < 1.0):
            raise ValueError("val_split must be in [0.0, 1.0)")
        if config.max_length is not None and config.max_length <= 0:
            raise ValueError("max_length must be positive")

    return True


# ============================================================================
# AUTO-TUNING UTILITIES
# ============================================================================

def auto_batch_size(model_size: str, available_memory_gb: float) -> int:
    """Auto-tune batch size based on model size and available GPU memory."""
    size_to_batch = {
        "small": 128,
        "base": 64,
        "large": 32,
        "xlarge": 16,
    }
    memory_factor = available_memory_gb / 24.0
    base_batch = size_to_batch.get(model_size, 32)
    return max(1, int(base_batch * memory_factor))


def auto_learning_rate(batch_size: int, base_lr: float = 1e-3) -> float:
    """Auto-scale learning rate with batch size (linear scaling rule)."""
    base_batch = 256
    return base_lr * (batch_size / base_batch)


__all__ = [
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
    "MODEL_CONFIG_MAP",
    "merge_configs",
    "split_kwargs",
    "validate_config",
    "auto_batch_size",
    "auto_learning_rate",
]
