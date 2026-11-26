"""
Multimodal Pillar - Vision-Language Models (CLIP-style contrastive VLM & HuggingFace CLIP).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kitepy.core.adapters import get_adapter
from kitepy.core.base import BaseModel
from kitepy.core.config import VLMConfig
from kitepy.core.utils import get_model_device
from kitepy.pillars.language.data import SimpleCharTokenizer


class DualEncoderCLIP(nn.Module):
    """Clean PyTorch CLIP-style dual encoder (vision + text -> shared contrastive space)."""

    def __init__(self, projection_dim: int = 128, vocab_size: int = 1000, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, projection_dim),
        )
        self.token_emb = nn.Embedding(vocab_size, 128)
        self.text_encoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vision_encoder(images), dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.token_emb(input_ids).mean(dim=1)
        return F.normalize(self.text_encoder(emb), dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor):
        img_feat = self.encode_image(images)
        txt_feat = self.encode_text(input_ids)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = scale * img_feat @ txt_feat.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


class VLM(BaseModel):
    """
    Vision-Language Model API:
    - Offline `DualEncoderCLIP` (`"clip_tiny"`, `"clip_base"`)
    - HuggingFace `CLIPModel` + `CLIPProcessor` when `pretrained=True` with an HF repo ID (e.g. `"openai/clip-vit-base-patch32"`)
    """

    modality = "multimodal"
    default_config_class = VLMConfig
    default_task = "contrastive"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "clip_tiny",
        config: Optional[Union[Dict, VLMConfig, str]] = None,
        **kwargs,
    ):
        kwargs.setdefault("task", "contrastive")
        super().__init__(model, config, adapter=get_adapter("contrastive"), **kwargs)
        self.tokenizer = SimpleCharTokenizer(vocab_size=self.config.vocab_size)
        self.processor = None

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        elif self.config.pretrained and "/" in self.model_name:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError:
                raise ImportError("transformers required for pretrained HuggingFace CLIP models.")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.model = DualEncoderCLIP(
                projection_dim=self.config.projection_dim,
                vocab_size=self.config.vocab_size,
                temperature=self.config.temperature,
            )
        print(f"✓ Created VLM model: {self.model_name} ({sum(p.numel() for p in self.model.parameters()):,} params)")

    @torch.no_grad()
    def predict(self, image: Any, texts: Optional[List[str]] = None, **kwargs) -> Any:
        """Compute zero-shot probabilities over candidate text descriptions (supports Tensor, PIL, or file path)."""
        if self.model is None:
            self._build_model()
        self.model.eval()
        device = get_model_device(self.model, self.device)

        if texts is None:
            raise ValueError("Provide `texts=[...]` for VLM zero-shot prediction.")

        if isinstance(image, (str, Path)):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        if self.processor is not None:
            inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        else:
            if not isinstance(image, torch.Tensor):
                from torchvision import transforms
                img_sz = getattr(self.config, "img_size", 64) or 64
                tf = transforms.Compose([transforms.Resize((img_sz, img_sz)), transforms.ToTensor()])
                image = tf(image)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            tok = self.tokenizer(texts, max_length=self.config.max_length, return_tensors="pt")["input_ids"].to(device)
            logits_per_image, _ = self.model(image, tok)

        probs = torch.softmax(logits_per_image, dim=-1)
        return probs if kwargs.get("return_probs", True) else logits_per_image


__all__ = ["VLM", "DualEncoderCLIP"]
