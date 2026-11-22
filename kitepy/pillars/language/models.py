"""
Language Pillar - LLM and TextClassifier APIs.

Supports:
- Pretrained HuggingFace LLMs ("Qwen/Qwen2.5-0.5B", "gpt2", local HF directories) with AutoTokenizer + PEFT/built-in LoRA
- Built-in pure-PyTorch TinyGPT ("tiny_gpt") for offline / from-scratch SLM training
- ChatML / Alpaca / JSONL / list / HuggingFace dataset fine-tuning in 1 line
- .generate() for text generation and .predict() for logits/classifications
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kitepy.core.adapters import get_adapter
from kitepy.core.base import BaseModel
from kitepy.core.config import DataConfig, LLMConfig, TrainingConfig
from .data import SimpleCharTokenizer


# ============================================================================
# BUILT-IN LIGHTWEIGHT LoRA & CAUSAL TRANSFORMER
# ============================================================================

class LoRALinear(nn.Module):
    """Lightweight built-in LoRA adapter wrapping an nn.Linear layer."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return base_out + self.scaling * lora_out


def apply_builtin_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> nn.Module:
    """Freeze base model and inject LoRA adapters into attention & MLP linear projections."""
    for p in model.parameters():
        p.requires_grad_(False)

    target_keywords = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "c_attn", "c_proj", "gate_proj", "up_proj", "down_proj", "proj", "qkv")
    replaced = 0
    for _, module in list(model.named_modules()):
        if isinstance(module, nn.MultiheadAttention):
            continue
        for child_name, child in list(module.named_children()):
            if type(child) is nn.Linear and any(k in child_name.lower() for k in target_keywords):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1

    if replaced == 0:
        for _, module in list(model.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                continue
            for child_name, child in list(module.named_children()):
                if type(child) is nn.Linear:
                    setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                    replaced += 1

    print(f"✓ Applied LoRA to {replaced} linear layers")
    return model


class _CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Pure-PyTorch GPT-style model supporting causal LM and sequence classification."""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [_CausalTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if cfg.model_type == "classification":
            self.head = nn.Linear(cfg.d_model, cfg.num_labels)
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_word_embeddings:
                self.head.weight = self.tok_emb.weight

        self._init_weights(cfg.initializer_range)

    def _init_weights(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        _, t = input_ids.shape
        t_clamped = min(t, self.cfg.context_length)
        input_ids = input_ids[:, :t_clamped]
        pos = torch.arange(0, t_clamped, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)
        x = self.ln_f(x)

        if self.cfg.model_type == "classification":
            return self.head(x[:, -1, :])

        return self.head(x)


# ============================================================================
# PUBLIC LLM & TEXT CLASSIFIER APIs
# ============================================================================

class LLM(BaseModel):
    """
    Large Language Model API supporting HuggingFace LLMs (e.g. `Qwen/Qwen2.5-0.5B`)
    with LoRA, as well as local `tiny_gpt` SLMs.

    Example:
        >>> LLM("Qwen/Qwen2.5-0.5B", pretrained=True, lora=True).train("my_chat.jsonl")
    """

    modality = "language"
    default_config_class = LLMConfig
    default_task = "causal_lm"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "tiny_gpt",
        config: Optional[Union[Dict, LLMConfig, str]] = None,
        task: str = "causal_lm",
        **kwargs,
    ):
        self.task = task.lower()
        model_type = "classification" if self.task in ("classification", "sequence_classification") else "causal_lm"
        kwargs.setdefault("model_type", model_type)
        adapter_name = "sequence_classification" if model_type == "classification" else "causal_lm"
        kwargs.setdefault("task", adapter_name)

        super().__init__(model, config, adapter=get_adapter(adapter_name), **kwargs)
        self.default_task = adapter_name
        self.tokenizer = None

    def _is_hf_model(self) -> bool:
        return bool(
            self.custom_model is None
            and self.model_name != "tiny_gpt"
            and (self.config.pretrained or "/" in self.model_name or Path(self.model_name).is_dir())
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        if self._is_hf_model():
            from kitepy.pillars.vision.wrappers import create_hf_tokenizer
            self.tokenizer = create_hf_tokenizer(self.model_name, trust_remote_code=self.config.trust_remote_code)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = SimpleCharTokenizer(vocab_size=self.config.vocab_size)
        return self.tokenizer

    def _resolve_data(self, data: Any, data_config: DataConfig, train_config: TrainingConfig):
        # Ensure the exact matching tokenizer (e.g., Qwen/GPT2 BPE) is ready BEFORE tokenizing the dataset
        tok = self._ensure_tokenizer()
        setattr(data_config, "_tokenizer", tok)
        if getattr(self.config, "context_length", None):
            data_config.max_length = min(data_config.max_length, self.config.context_length)
        return super()._resolve_data(data, data_config, train_config)

    def _auto_align_model_to_data(self, metadata: Dict[str, Any]):
        if self.task == "causal_lm":
            return
        super()._auto_align_model_to_data(metadata)

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        elif not self._is_hf_model():
            self.model = TinyGPT(self.config)
            self._ensure_tokenizer()
        else:
            from kitepy.pillars.vision.wrappers import create_hf_model
            self.model = create_hf_model(
                model_name=self.model_name,
                model_type=self.config.model_type,
                num_labels=self.config.num_labels if self.config.model_type == "classification" else None,
                pretrained=self.config.pretrained,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._ensure_tokenizer()

        if getattr(self.config, "lora", False):
            try:
                from peft import LoraConfig, TaskType, get_peft_model
                tt = TaskType.SEQ_CLS if self.config.model_type == "classification" else TaskType.CAUSAL_LM
                peft_cfg = LoraConfig(
                    task_type=tt,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                )
                self.model = get_peft_model(self.model, peft_cfg)
                print("✓ Applied PEFT LoRA")
            except Exception:
                self.model = apply_builtin_lora(
                    self.model, r=self.config.lora_r, alpha=self.config.lora_alpha, dropout=self.config.lora_dropout
                )

        print(f"✓ Created LLM model: {self.model_name} ({sum(p.numel() for p in self.model.parameters()):,} params)")

    def _preprocess_input(self, input_data: Any) -> Any:
        from collections.abc import Mapping
        if isinstance(input_data, torch.Tensor):
            return input_data
        if isinstance(input_data, Mapping):
            return dict(input_data)
        tok = self._ensure_tokenizer()
        return dict(
            tok(
                input_data,
                max_length=self.config.context_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        )

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 32, temperature: float = 0.7) -> str:
        """Generate text continuation from a prompt (or ChatML messages)."""
        if self.model is None:
            self._build_model()
        self.model.eval()
        device = self.device if self.device is not None else next(self.model.parameters()).device
        tok = self._ensure_tokenizer()

        if hasattr(self.model, "generate") and not isinstance(self.model, TinyGPT):
            enc = tok(prompt, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            if attn is not None:
                gen_kwargs["attention_mask"] = attn.to(device)
            out_ids = self.model.generate(ids, **gen_kwargs)
            return tok.decode(out_ids[0].tolist(), skip_special_tokens=True)

        enc = tok(prompt, max_length=self.config.context_length, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        eos_id = getattr(tok, "eos_token_id", 2)
        if ids.size(1) > 1 and ids[0, -1].item() == eos_id:
            ids = ids[:, :-1]

        for _ in range(max_new_tokens):
            ctx = ids[:, -self.config.context_length :]
            logits = self.model(ctx)
            logits = getattr(logits, "logits", logits)
            if temperature <= 0:
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break

        return tok.decode(ids[0].tolist())


class TextClassifier(LLM):
    """Convenience API for sequence/text classification."""

    def __init__(self, model: Optional[Union[str, nn.Module]] = "tiny_gpt", num_classes: int = 2, **kwargs):
        super().__init__(model=model, task="classification", num_labels=num_classes, **kwargs)


__all__ = ["LLM", "TextClassifier", "TinyGPT", "LoRALinear", "apply_builtin_lora"]
