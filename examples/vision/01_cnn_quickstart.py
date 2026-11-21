"""Vision Quickstart: Train, save, reload, and export a CNN."""

import tempfile
from pathlib import Path
import torch
from kitepy import CNN

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = Path(tmp) / "resnet18.pt"

    # Train (swap "synthetic" for "cifar10" or "./my_image_folder" for real data)
    model = CNN("resnet18", num_classes=10)
    model.train("synthetic", epochs=1, batch_size=8, fast_dev_run=True, num_workers=0, checkpoint_dir=tmp)

    # Save & reload
    model.save(ckpt_path)
    loaded = CNN.load(ckpt_path)

    # Inference & TorchScript export
    logits = loaded.predict(torch.randn(2, 3, 224, 224))
    print("Prediction shape:", logits.shape)

    loaded.export(Path(tmp) / "resnet18_scripted.pt", verify=True)
