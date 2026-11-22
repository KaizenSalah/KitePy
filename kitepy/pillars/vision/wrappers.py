"""
Wrappers for external model libraries (timm, HuggingFace, etc.).
"""

import inspect
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


TIMM_NAME_MAPPING = {
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "vit_huge": "vit_huge_patch14_224",
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "swin_large": "swin_large_patch4_window7_224",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "efficientnetv2_m": "tf_efficientnetv2_m",
    "efficientnetv2_l": "tf_efficientnetv2_l",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large",
    "wide_resnet50": "wide_resnet50_2",
    "wide_resnet101": "wide_resnet101_2",
    "resnext50": "resnext50_32x4d",
    "resnext101": "resnext101_32x8d",
    "mobilenet_v2": "mobilenetv2_100",
    "mobilenet_v3_small": "mobilenetv3_small_100",
    "mobilenet_v3_large": "mobilenetv3_large_100",
}


def validate_vision_model_name(user_name: str) -> None:
    """Validate that a vision model name exists in KitePy presets or timm, raising a helpful typo hint."""
    from kitepy.core.presets import CNN_PRESETS, TRANSFORMER_PRESETS
    if user_name in CNN_PRESETS or user_name in TRANSFORMER_PRESETS or user_name in TIMM_NAME_MAPPING:
        return
    try:
        import timm
        avail = set(timm.list_models())
        if user_name in avail:
            return
        from difflib import get_close_matches
        pool = list(CNN_PRESETS.keys()) + list(TRANSFORMER_PRESETS.keys()) + list(avail)
        hints = get_close_matches(user_name, pool, n=3, cutoff=0.55)
        hint_msg = f" Did you mean one of: {hints}?" if hints else ""
        raise ValueError(f"Unknown vision model '{user_name}'.{hint_msg}")
    except ImportError:
        return


def resolve_model_name(user_name: str) -> str:
    """Resolve user-friendly name to actual timm model name."""
    if user_name in TIMM_NAME_MAPPING:
        resolved = TIMM_NAME_MAPPING[user_name]
        print(f"ℹ️  Resolved '{user_name}' → '{resolved}'")
        return resolved
    return user_name


def create_timm_model(
    model_name: str,
    num_classes: int = 1000,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Create a vision model from timm."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for vision models. Install with: pip install timm")

    resolved_name = resolve_model_name(model_name)

    available_models = timm.list_models()
    if resolved_name not in available_models:
        similar = [m for m in available_models if model_name.split("_")[0] in m][:5]
        if similar:
            raise ValueError(
                f"Model '{model_name}' (resolved to '{resolved_name}') not found in timm. "
                f"Did you mean one of: {similar}?"
            )
        raise ValueError(f"Model '{resolved_name}' not available in timm")

    # Filter optional kwargs (like img_size / patch_size) so CNNs don't reject ViT args
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    is_vit_family = any(k in resolved_name for k in ("vit", "deit", "swin", "beit", "eva"))
    if not is_vit_family:
        clean_kwargs.pop("img_size", None)
        clean_kwargs.pop("patch_size", None)

    try:
        model = timm.create_model(
            resolved_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **clean_kwargs,
        )
    except TypeError:
        # Retry without optional kwargs if specific timm model doesn't accept them
        clean_kwargs.pop("img_size", None)
        clean_kwargs.pop("patch_size", None)
        model = timm.create_model(
            resolved_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=drop_rate,
        )

    return model


def list_timm_models(filter: Optional[str] = None) -> list:
    """List available timm models."""
    try:
        import timm
    except ImportError:
        return []
    models = timm.list_models()
    if filter:
        models = [m for m in models if filter.lower() in m.lower()]
    return models


def get_timm_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a timm model."""
    try:
        import timm
    except ImportError:
        return {}
    model = timm.create_model(resolve_model_name(model_name), pretrained=False)
    config = getattr(model, "default_cfg", {})
    return {
        "model_name": model_name,
        "input_size": config.get("input_size", (3, 224, 224)),
        "num_classes": config.get("num_classes", 1000),
        "architecture": config.get("architecture", "unknown"),
        "pretrained_available": config.get("has_weights", False),
    }


def create_hf_model(
    model_name: str,
    model_type: str = "causal_lm",
    num_labels: Optional[int] = None,
    pretrained: bool = False,
    trust_remote_code: bool = False,
    **kwargs,
) -> nn.Module:
    """Create a model from HuggingFace Transformers."""
    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
        )
    except ImportError:
        raise ImportError("transformers is required. Install with: pip install transformers")

    model_classes = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq": AutoModelForSeq2SeqLM,
        "classification": AutoModelForSequenceClassification,
        "base": AutoModel,
    }
    ModelClass = model_classes.get(model_type)
    if ModelClass is None:
        raise ValueError(f"Unknown model_type: {model_type}")

    extra = dict(kwargs)
    if num_labels is not None and model_type == "classification":
        extra["num_labels"] = num_labels

    if pretrained:
        return ModelClass.from_pretrained(model_name, trust_remote_code=trust_remote_code, **extra)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code, **extra)
    return ModelClass.from_config(config)


def create_hf_tokenizer(model_name: str, trust_remote_code: bool = False, **kwargs):
    """Create a tokenizer from HuggingFace."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers is required. Install with: pip install transformers")
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)


class CustomModelWrapper(nn.Module):
    """Wrapper for custom PyTorch models."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def wrap_custom_model(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")
    return CustomModelWrapper(model)


def create_model(
    model_name_or_module: Union[str, nn.Module],
    modality: str,
    config: Dict[str, Any],
    pretrained: bool = False,
) -> nn.Module:
    """Universal vision model factory."""
    if isinstance(model_name_or_module, nn.Module):
        return wrap_custom_model(model_name_or_module)

    model_name = model_name_or_module

    if modality == "vision":
        extra = {}
        # Only forward img_size/patch_size when relevant (ViTs)
        if any(k in model_name for k in ("vit", "deit", "swin", "beit")):
            if config.get("img_size") is not None:
                extra["img_size"] = config["img_size"]
            if config.get("patch_size") is not None and "swin" not in model_name:
                extra["patch_size"] = config["patch_size"]

        return create_timm_model(
            model_name=model_name,
            num_classes=config.get("num_classes", 1000),
            pretrained=pretrained,
            in_channels=config.get("in_channels", 3),
            drop_rate=config.get("drop_rate", 0.0),
            drop_path_rate=config.get("drop_path_rate", 0.0),
            **extra,
        )

    if modality == "language":
        return create_hf_model(
            model_name=model_name,
            model_type=config.get("model_type", "causal_lm"),
            pretrained=pretrained,
            num_labels=config.get("num_labels"),
        )

    raise ValueError(f"Unknown modality: {modality}")


def get_model_source(model_name: str) -> str:
    """Determine which library provides a model."""
    try:
        import timm
        if resolve_model_name(model_name) in timm.list_models():
            return "timm"
    except ImportError:
        pass
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
        return "huggingface"
    except Exception:
        pass
    return "unknown"


__all__ = [
    "TIMM_NAME_MAPPING",
    "resolve_model_name",
    "create_timm_model",
    "list_timm_models",
    "get_timm_model_info",
    "create_hf_model",
    "create_hf_tokenizer",
    "CustomModelWrapper",
    "wrap_custom_model",
    "create_model",
    "get_model_source",
]
