"""Power-User Escape Hatches: custom nn.Module, custom train_step, CSV logging, and .unwrap()."""

import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from kitepy import CNN

net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 5),
)

def custom_step(engine, batch):
    x, y = batch
    logits = engine.model(x)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    return loss, logits, y

with tempfile.TemporaryDirectory() as tmp:
    model = CNN(net)
    model.train(
        "synthetic",
        epochs=2,
        batch_size=8,
        train_step=custom_step,
        gradient_accumulation_steps=2,
        csv_log_dir=tmp,
        save_every_n_epochs=0,
        num_workers=0,
    )
    raw_pytorch = model.unwrap()
    print("Unwrapped type:", type(raw_pytorch).__name__)
