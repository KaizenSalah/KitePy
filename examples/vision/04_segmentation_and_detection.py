"""Vision Segmentation & Object Detection in a few lines."""

import tempfile
import torch
from kitepy import Vision

with tempfile.TemporaryDirectory() as tmp:
    # 1. Semantic Segmentation (U-Net)
    seg = Vision(task="segmentation", model="unet", num_classes=3)
    seg.train("synthetic", epochs=1, batch_size=4, num_workers=0, checkpoint_dir=tmp, save_every_n_epochs=0)
    mask_logits = seg.predict(torch.randn(1, 3, 64, 64))
    print("Segmentation mask logits:", mask_logits.shape)

    # 2. Object Detection (TinyDetector or fasterrcnn_resnet50_fpn)
    det = Vision(task="detection", model="tiny_detector", num_classes=4)
    det.train("synthetic", epochs=1, batch_size=4, num_workers=0, checkpoint_dir=tmp, save_every_n_epochs=0)
    detections = det.predict(torch.rand(1, 3, 64, 64))
    print("Detection boxes:", detections[0]["boxes"].shape)
