"""Hello KitePy: Dead simple training and inference in ~10 lines."""

import tempfile
import torch
from kitepy import CNN

with tempfile.TemporaryDirectory() as tmp:
    # 1. Train in one line (uses synthetic data here so nothing needs to download)
    model = CNN("resnet18", num_classes=4)
    model.class_names = ["cat", "dog", "bird", "car"]
    model.train("synthetic", epochs=1, batch_size=8, fast_dev_run=True, num_workers=0, checkpoint_dir=tmp)

    # 2. Predict with human-readable labels & confidences
    preds = model.predict(torch.randn(1, 3, 224, 224), top_k=3)
    print("Top predictions:", preds)
