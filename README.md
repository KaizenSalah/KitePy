# kitepy - Dead Simple Deep Learning 🚀

[![CI](https://github.com/KaizenSalah/kitepy/actions/workflows/ci.yml/badge.svg)](https://github.com/KaizenSalah/kitepy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Build and train deep learning models across **Vision, Language, Multimodal, Audio, Tabular, and Generative AI** with **one line of code**—while keeping full power-user control underneath.

```python
from kitepy import CNN
CNN("resnet18").train("cifar10")
```

Simple by default. Unstoppable when you need control. ✨

---

## Features

- **🎯 One-Line Training with Smart Auto-Inference**: Automatically infers `num_classes`, `class_names`, and `input_dim` from folders, CSVs, and datasets.
- **🧩 Universal `TaskAdapter` + `Engine`**: One battle-tested training loop supporting classification, binary, multi-label, regression, causal LM, contrastive VLM, generative (VAE/DDPM), or custom `train_step` hooks.
- **🔧 Zero-Friction Callbacks & Logging**: Built-in `EarlyStopping`, `ModelCheckpoint`, `CSVLogger`, `LRMonitor`, `TensorBoardLogger`, `WandbLogger`, and `LambdaCallback`.
- **⚡ Advanced Training Techniques**: Differential learning rates (`lr={"backbone": 1e-4, "head": 1e-3}`), layer-wise LR decay (`layer_decay`), EMA weights (`ema=True`), MixUp & CutMix, gradient accumulation, `max_steps`, `warmup_steps`, `lr="auto"` LR Finder, `bf16`/`fp16` AMP, and `accelerate` distributed support.
- **🌐 6 Complete Pillars**:
  - **Vision**: `CNN`, `Transformer` (ViT/DeiT/Swin), `Vision(task=...)` backed by 800+ `timm` models
  - **Language**: `LLM`, `TextClassifier` (GPT/HF Transformers + LoRA)
  - **Multimodal**: `VLM` (CLIP-style contrastive + zero-shot prediction)
  - **Audio**: `Audio` (spectrogram CNN + automatic STFT for raw waveforms)
  - **Tabular**: `Tabular` (`TabularResNet`, `TabularMLP` for CSV/NumPy/Tensor classification & regression)
  - **Generative**: `Generative` (`ConvVAE`, `SimpleDDPM` with `.sample()`)
- **🚀 Production Ready**: `.predict()` on file paths/PIL/NumPy/tensors with `top_k`, `.export()` to TorchScript/ONNX/SafeTensors, INT8 `.quantize()`, `.to_fastapi()`, and side-by-side `compare()`.

---

## Installation

```bash
# Core install
pip install kitepy

# With LLM & HuggingFace support
pip install kitepy[llm]

# Full install
pip install kitepy[all]

# Development install
git clone https://github.com/KaizenSalah/kitepy
cd kitepy
pip install -e .[dev]
```

---

## Quick Start

### Vision (CNN & ViT)

```python
from kitepy import CNN, Transformer

# Auto-detects classes from an ImageFolder directory!
model = CNN("resnet18").train("./my_dataset", epochs=10)

# Real-world inference on image path with class names
preds = model.predict("cat.jpg", top_k=3)

# Vision Transformer transfer learning
vit = Transformer("vit_tiny", pretrained=True, num_classes=10)
vit.freeze_backbone()  # Accurately freezes backbone & ViT MLP blocks, keeps head trainable
vit.train("cifar10", epochs=5, lr={"backbone": 1e-4, "head": 1e-3}, ema=True)
```

### Language (LLM & Text Classification)

```python
from kitepy import LLM, TextClassifier

llm = LLM("tiny_gpt", lora=True)
llm.train(["kitepy makes deep learning simple", ...], epochs=3)
print(llm.generate("kitepy", max_new_tokens=20))

clf = TextClassifier("tiny_gpt", num_classes=2)
clf.train("synthetic", epochs=2)
```

### Tabular, Audio, Multimodal, Generative

```python
from kitepy import Tabular, Audio, VLM, Generative

Tabular("tabular_resnet", task="classification").train("data.csv", epochs=10)
Audio("audio_cnn", num_classes=10).train("synthetic", epochs=5)
VLM("clip_tiny").train("synthetic", epochs=5)
gen = Generative("vae_small").train("synthetic", epochs=5)
samples = gen.sample(n=4)
```

---

## Power-User Controls

### Nested YAML Configuration

```yaml
# config.yaml
model:
  arch: resnet
  depth: 50
  num_classes: 100

training:
  epochs: 50
  batch_size: 64
  lr: 0.001
  gradient_accumulation_steps: 2
  ema: true
  mixup_alpha: 0.2
  csv_log_dir: ./logs

data:
  augmentation: randaugment
```

```python
from kitepy import CNN
model = CNN("resnet50", config="config.yaml").train("cifar100")
```

### Custom Training Step & Escape Hatch

```python
# Hook custom loss/step logic while keeping KitePy's AMP, callbacks, and checkpointing
def my_step(engine, batch):
    x, y = batch
    logits = engine.model(x)
    loss = my_custom_loss(logits, y)
    return loss, logits, y

CNN(my_nn_module).train(my_loader, train_step=my_step, callbacks=["early_stopping", "csv"])

# Or drop down to raw PyTorch anytime
pytorch_model = model.unwrap()
```

### Benchmark Models Side-by-Side

```python
from kitepy import compare
results = compare(["resnet18", "mobilenet_v3_small", "efficientnet_b0"], data="cifar10", epochs=3)
```

---

## Testing

The test suite is organized cleanly into unit and integration layers:

```
tests/
├── conftest.py
├── unit/
│   ├── core/          # Adapters, Callbacks, Config, Engine, LRFinder, Metrics, Presets, Utils
│   └── pillars/       # Vision, Language, Multimodal, Audio, Tabular, Generative
└── integration/       # End-to-end pipelines & advanced power-user workflows
```

Run all tests with:

```bash
pytest -v
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
