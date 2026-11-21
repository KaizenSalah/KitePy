"""
Model and dataset preset registry.

Users type "resnet18", "vit_tiny", "gpt2", "clip_base", etc. and we resolve it to a full config.
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# CNN PRESETS
# ============================================================================

CNN_PRESETS: Dict[str, Dict[str, Any]] = {
    # ResNet family
    "resnet18": {"arch": "resnet", "depth": 18},
    "resnet34": {"arch": "resnet", "depth": 34},
    "resnet50": {"arch": "resnet", "depth": 50},
    "resnet101": {"arch": "resnet", "depth": 101},
    "resnet152": {"arch": "resnet", "depth": 152},

    # Wide ResNet
    "wide_resnet50": {"arch": "resnet", "depth": 50, "width_multiplier": 2.0},
    "wide_resnet101": {"arch": "resnet", "depth": 101, "width_multiplier": 2.0},

    # ResNeXt
    "resnext50": {"arch": "resnext", "depth": 50, "groups": 32, "base_width": 4},
    "resnext101": {"arch": "resnext", "depth": 101, "groups": 32, "base_width": 8},

    # EfficientNet family
    "efficientnet_b0": {"arch": "efficientnet", "depth": 0},
    "efficientnet_b1": {"arch": "efficientnet", "depth": 1},
    "efficientnet_b2": {"arch": "efficientnet", "depth": 2},
    "efficientnet_b3": {"arch": "efficientnet", "depth": 3},
    "efficientnet_b4": {"arch": "efficientnet", "depth": 4},
    "efficientnet_b5": {"arch": "efficientnet", "depth": 5},
    "efficientnet_b6": {"arch": "efficientnet", "depth": 6},
    "efficientnet_b7": {"arch": "efficientnet", "depth": 7},

    # EfficientNetV2
    "efficientnetv2_s": {"arch": "efficientnetv2", "depth": "s"},
    "efficientnetv2_m": {"arch": "efficientnetv2", "depth": "m"},
    "efficientnetv2_l": {"arch": "efficientnetv2", "depth": "l"},

    # MobileNet
    "mobilenet_v2": {"arch": "mobilenetv2"},
    "mobilenet_v3_small": {"arch": "mobilenetv3", "variant": "small"},
    "mobilenet_v3_large": {"arch": "mobilenetv3", "variant": "large"},

    # VGG
    "vgg11": {"arch": "vgg", "depth": 11},
    "vgg13": {"arch": "vgg", "depth": 13},
    "vgg16": {"arch": "vgg", "depth": 16},
    "vgg19": {"arch": "vgg", "depth": 19},

    # DenseNet
    "densenet121": {"arch": "densenet", "depth": 121},
    "densenet169": {"arch": "densenet", "depth": 169},
    "densenet201": {"arch": "densenet", "depth": 201},

    # ConvNeXt
    "convnext_tiny": {"arch": "convnext"},
    "convnext_small": {"arch": "convnext"},
    "convnext_base": {"arch": "convnext"},
    "convnext_large": {"arch": "convnext"},
}


# ============================================================================
# TRANSFORMER PRESETS (Vision)
# ============================================================================

TRANSFORMER_PRESETS: Dict[str, Dict[str, Any]] = {
    "vit_tiny": {
        "arch": "vit",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
    },
    "vit_small": {
        "arch": "vit",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "vit_base": {
        "arch": "vit",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "vit_large": {
        "arch": "vit",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
    },
    "vit_huge": {
        "arch": "vit",
        "img_size": 224,
        "patch_size": 14,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
    },

    # DeiT
    "deit_tiny": {"arch": "deit", "img_size": 224, "patch_size": 16, "embed_dim": 192, "depth": 12, "num_heads": 3},
    "deit_small": {"arch": "deit", "img_size": 224, "patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6},
    "deit_base": {"arch": "deit", "img_size": 224, "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12},

    # Swin Transformer
    "swin_tiny": {"arch": "swin", "img_size": 224, "embed_dim": 96, "depths": [2, 2, 6, 2]},
    "swin_small": {"arch": "swin", "img_size": 224, "embed_dim": 96, "depths": [2, 2, 18, 2]},
    "swin_base": {"arch": "swin", "img_size": 224, "embed_dim": 128, "depths": [2, 2, 18, 2]},
    "swin_large": {"arch": "swin", "img_size": 224, "embed_dim": 192, "depths": [2, 2, 18, 2]},
}


# ============================================================================
# LLM PRESETS
# ============================================================================

LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny_gpt": {
        "arch": "gpt",
        "n_layers": 2,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "vocab_size": 260,
        "context_length": 128,
    },
    "gpt2": {
        "arch": "gpt",
        "n_layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "d_ff": 3072,
        "vocab_size": 50257,
        "context_length": 1024,
    },
    "gpt2_medium": {
        "arch": "gpt",
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 16,
        "d_ff": 4096,
        "vocab_size": 50257,
        "context_length": 1024,
    },
    "gpt2_large": {
        "arch": "gpt",
        "n_layers": 36,
        "d_model": 1280,
        "n_heads": 20,
        "d_ff": 5120,
        "vocab_size": 50257,
        "context_length": 1024,
    },
    "gpt_125m": {
        "arch": "gpt",
        "n_layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "d_ff": 3072,
        "vocab_size": 50257,
        "context_length": 2048,
    },
    "gpt_350m": {
        "arch": "gpt",
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 16,
        "d_ff": 4096,
        "vocab_size": 50257,
        "context_length": 2048,
    },
    "gpt_1b": {
        "arch": "gpt",
        "n_layers": 24,
        "d_model": 2048,
        "n_heads": 16,
        "d_ff": 8192,
        "vocab_size": 50257,
        "context_length": 2048,
    },
    "gpt_3b": {
        "arch": "gpt",
        "n_layers": 32,
        "d_model": 2560,
        "n_heads": 32,
        "d_ff": 10240,
        "vocab_size": 50257,
        "context_length": 2048,
    },
    "llama_1b": {
        "arch": "llama",
        "n_layers": 22,
        "d_model": 2048,
        "n_heads": 16,
        "d_ff": 5632,
        "vocab_size": 32000,
        "context_length": 2048,
        "norm_type": "rmsnorm",
        "activation": "swiglu",
        "pos_encoding": "rope",
    },
    "llama_3b": {
        "arch": "llama",
        "n_layers": 32,
        "d_model": 3200,
        "n_heads": 32,
        "d_ff": 8640,
        "vocab_size": 32000,
        "context_length": 2048,
        "norm_type": "rmsnorm",
        "activation": "swiglu",
        "pos_encoding": "rope",
    },
    "llama_7b": {
        "arch": "llama",
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
        "d_ff": 11008,
        "vocab_size": 32000,
        "context_length": 4096,
        "norm_type": "rmsnorm",
        "activation": "swiglu",
        "pos_encoding": "rope",
    },
    "mistral_7b": {
        "arch": "mistral",
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "d_ff": 14336,
        "vocab_size": 32000,
        "context_length": 8192,
        "norm_type": "rmsnorm",
        "activation": "swiglu",
        "pos_encoding": "rope",
    },
}


# ============================================================================
# VLM, AUDIO, TABULAR, GENERATIVE, RNN PRESETS
# ============================================================================

VLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "clip_tiny": {
        "arch": "clip",
        "vision_model": "resnet18",
        "text_model": "tiny_gpt",
        "projection_dim": 128,
        "task": "contrastive",
    },
    "clip_base": {
        "arch": "clip",
        "vision_model": "vit_base",
        "text_model": "gpt2",
        "projection_dim": 512,
        "task": "contrastive",
    },
    "clip_large": {
        "arch": "clip",
        "vision_model": "vit_large",
        "text_model": "gpt2_medium",
        "projection_dim": 768,
        "task": "contrastive",
    },
    "blip_base": {
        "arch": "blip",
        "vision_model": "vit_base",
        "text_model": "gpt2",
        "task": "captioning",
    },
}

AUDIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "audio_cnn": {"arch": "audio_cnn", "hidden_dim": 128},
    "audio_transformer": {"arch": "audio_transformer", "hidden_dim": 192},
}

TABULAR_PRESETS: Dict[str, Dict[str, Any]] = {
    "tabular_mlp": {"arch": "tabular_mlp", "hidden_dims": [128, 64]},
    "tabular_resnet": {"arch": "tabular_resnet", "hidden_dims": [128, 128, 64]},
}

GENERATIVE_PRESETS: Dict[str, Dict[str, Any]] = {
    "vae_small": {"arch": "vae", "latent_dim": 32, "base_channels": 32, "img_size": 32},
    "vae_base": {"arch": "vae", "latent_dim": 64, "base_channels": 64, "img_size": 64},
    "ddpm_small": {"arch": "ddpm", "base_channels": 32, "img_size": 32, "timesteps": 100},
}

RNN_PRESETS: Dict[str, Dict[str, Any]] = {
    "lstm_small": {"rnn_type": "lstm", "hidden_size": 256, "num_layers": 2},
    "lstm_base": {"rnn_type": "lstm", "hidden_size": 512, "num_layers": 3},
    "lstm_large": {"rnn_type": "lstm", "hidden_size": 1024, "num_layers": 4},
    "gru_base": {"rnn_type": "gru", "hidden_size": 512, "num_layers": 3},
}


# ============================================================================
# DATASET PRESETS
# ============================================================================

DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    # Vision datasets
    "cifar10": {
        "name": "cifar10",
        "num_classes": 10,
        "img_size": 32,
        "in_channels": 3,
        "modality": "vision",
        "source": "torchvision",
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
    },
    "cifar100": {
        "name": "cifar100",
        "num_classes": 100,
        "img_size": 32,
        "in_channels": 3,
        "modality": "vision",
        "source": "torchvision",
    },
    "imagenet": {
        "name": "imagenet-1k",
        "num_classes": 1000,
        "img_size": 224,
        "in_channels": 3,
        "modality": "vision",
        "source": "torchvision",
    },
    "imagenet21k": {
        "name": "imagenet-21k",
        "num_classes": 21843,
        "img_size": 224,
        "in_channels": 3,
        "modality": "vision",
        "source": "huggingface",
    },
    "mnist": {
        "name": "mnist",
        "num_classes": 10,
        "img_size": 28,
        "in_channels": 1,
        "modality": "vision",
        "source": "torchvision",
        "class_names": [str(i) for i in range(10)],
    },
    "fashion_mnist": {
        "name": "fashion_mnist",
        "num_classes": 10,
        "img_size": 28,
        "in_channels": 1,
        "modality": "vision",
        "source": "torchvision",
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ],
    },

    # Text datasets
    "wikitext": {
        "name": "wikitext",
        "subset": "wikitext-103-v1",
        "modality": "language",
        "source": "huggingface",
    },
    "openwebtext": {
        "name": "openwebtext",
        "modality": "language",
        "source": "huggingface",
    },
    "c4": {
        "name": "c4",
        "modality": "language",
        "source": "huggingface",
    },
    "pile": {
        "name": "EleutherAI/pile",
        "modality": "language",
        "source": "huggingface",
    },

    # Multimodal datasets
    "coco": {
        "name": "coco",
        "modality": "multimodal",
        "source": "huggingface",
        "task": "captioning",
    },
    "flickr30k": {
        "name": "flickr30k",
        "modality": "multimodal",
        "source": "huggingface",
        "task": "captioning",
    },
}


# ============================================================================
# REGISTRY FUNCTIONS
# ============================================================================

MODALITY_REGISTRIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "cnn": CNN_PRESETS,
    "transformer": TRANSFORMER_PRESETS,
    "llm": LLM_PRESETS,
    "language": LLM_PRESETS,
    "vlm": VLM_PRESETS,
    "multimodal": VLM_PRESETS,
    "audio": AUDIO_PRESETS,
    "tabular": TABULAR_PRESETS,
    "generative": GENERATIVE_PRESETS,
    "rnn": RNN_PRESETS,
}


def list_models(modality: str = "all") -> Any:
    """List all available model presets for a given modality."""
    if modality == "vision":
        return list(CNN_PRESETS.keys()) + list(TRANSFORMER_PRESETS.keys())
    if modality in MODALITY_REGISTRIES:
        return list(MODALITY_REGISTRIES[modality].keys())
    if modality == "all":
        return {
            "cnn": list(CNN_PRESETS.keys()),
            "transformer": list(TRANSFORMER_PRESETS.keys()),
            "llm": list(LLM_PRESETS.keys()),
            "vlm": list(VLM_PRESETS.keys()),
            "audio": list(AUDIO_PRESETS.keys()),
            "tabular": list(TABULAR_PRESETS.keys()),
            "generative": list(GENERATIVE_PRESETS.keys()),
            "rnn": list(RNN_PRESETS.keys()),
        }
    raise ValueError(f"Unknown modality: {modality}")


def list_datasets(modality: str = "all") -> List[str]:
    """List all available dataset presets."""
    if modality == "all":
        return list(DATASET_PRESETS.keys())

    datasets = []
    for name, info in DATASET_PRESETS.items():
        if info.get("modality") == modality:
            datasets.append(name)
        elif modality == "text" and info.get("modality") == "language":
            datasets.append(name)
    return datasets


def get_preset(name: str, modality: Optional[str] = None) -> Dict[str, Any]:
    """Get a preset configuration by name."""
    if modality is None or modality == "vision":
        if modality == "vision":
            pool = {**CNN_PRESETS, **TRANSFORMER_PRESETS}
        else:
            pool = {
                **CNN_PRESETS,
                **TRANSFORMER_PRESETS,
                **LLM_PRESETS,
                **VLM_PRESETS,
                **AUDIO_PRESETS,
                **TABULAR_PRESETS,
                **GENERATIVE_PRESETS,
                **RNN_PRESETS,
            }
        if name in pool:
            return pool[name]
        raise ValueError(f"Preset '{name}' not found")

    if modality not in MODALITY_REGISTRIES:
        raise ValueError(f"Unknown modality: {modality}")

    registry = MODALITY_REGISTRIES[modality]
    if name not in registry:
        raise ValueError(f"Preset '{name}' not found in {modality} registry")
    return registry[name]


def register_preset(
    name: str,
    config: Dict[str, Any],
    modality: str,
    overwrite: bool = False,
):
    """Register a custom preset."""
    if modality not in MODALITY_REGISTRIES:
        raise ValueError(f"Unknown modality: {modality}")
    registry = MODALITY_REGISTRIES[modality]

    if name in registry and not overwrite:
        raise ValueError(
            f"Preset '{name}' already exists. Set overwrite=True to replace."
        )

    registry[name] = config
    print(f"✓ Registered preset '{name}' in {modality} registry")


MODEL_PRESETS = {
    **CNN_PRESETS,
    **TRANSFORMER_PRESETS,
    **LLM_PRESETS,
    **VLM_PRESETS,
    **AUDIO_PRESETS,
    **TABULAR_PRESETS,
    **GENERATIVE_PRESETS,
    **RNN_PRESETS,
}

__all__ = [
    "CNN_PRESETS",
    "TRANSFORMER_PRESETS",
    "LLM_PRESETS",
    "VLM_PRESETS",
    "AUDIO_PRESETS",
    "TABULAR_PRESETS",
    "GENERATIVE_PRESETS",
    "RNN_PRESETS",
    "DATASET_PRESETS",
    "MODEL_PRESETS",
    "list_models",
    "list_datasets",
    "get_preset",
    "register_preset",
]
