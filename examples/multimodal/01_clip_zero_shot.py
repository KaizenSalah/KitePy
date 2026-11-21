"""Multimodal CLIP-style VLM: Train contrastively and run zero-shot prediction."""

import tempfile
import torch
from kitepy import VLM

with tempfile.TemporaryDirectory() as tmp:
    vlm = VLM("clip_tiny")
    vlm.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=tmp, save_every_n_epochs=0)

    probs = vlm.predict(torch.randn(1, 3, 64, 64), texts=["a photo of a cat", "a red sports car", "a mountain"])
    print("Zero-shot probabilities:", probs.tolist())
