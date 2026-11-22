"""Unit tests for Vision(task=...) multi-task support."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from kitepy import Vision


def test_vision_binary_multilabel_and_regression(tmp_path):
    x = torch.randn(16, 3, 64, 64)

    # Binary
    bin_loader = DataLoader(TensorDataset(x, torch.randint(0, 2, (16,))), batch_size=8)
    m_bin = Vision(task="binary", model="resnet18")
    m_bin.train(bin_loader, epochs=1, num_workers=0, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert m_bin.is_trained

    # Multi-label
    ml_loader = DataLoader(TensorDataset(x, torch.randint(0, 2, (16, 4)).float()), batch_size=8)
    m_ml = Vision(task="multilabel", model="resnet18", num_classes=4)
    m_ml.train(ml_loader, epochs=1, num_workers=0, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert m_ml.is_trained

    # Regression
    reg_loader = DataLoader(TensorDataset(x, torch.randn(16)), batch_size=8)
    m_reg = Vision(task="regression", model="resnet18")
    m_reg.train(reg_loader, epochs=1, num_workers=0, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert "train_mae" in m_reg.training_history[-1]

    # Synthetic multi-label & binary & regression must work directly with data="synthetic"
    syn_ml = Vision(task="multilabel", model="resnet18", num_classes=4)
    syn_ml.train("synthetic", epochs=1, img_size=32, batch_size=8, fast_dev_run=True, num_workers=0, save_every_n_epochs=0)
    assert syn_ml.is_trained


def test_vision_segmentation_and_detection(tmp_path):
    # Semantic Segmentation (U-Net)
    seg = Vision(task="segmentation", model="unet", num_classes=3)
    seg.train(
        "synthetic",
        epochs=1,
        batch_size=4,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert seg.is_trained
    assert "train_miou" in seg.training_history[-1]
    assert "train_pixel_acc" in seg.training_history[-1]
    mask_logits = seg.predict(torch.randn(1, 3, 64, 64))
    assert mask_logits.shape == (1, 3, 64, 64)

    # Object Detection (TinyDetector)
    det = Vision(task="detection", model="tiny_detector", num_classes=4)
    det.train(
        "synthetic",
        epochs=1,
        batch_size=4,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert det.is_trained
    assert "val_avg_detections" in det.training_history[-1]
    preds = det.predict(torch.rand(1, 3, 64, 64))
    assert isinstance(preds, list) and "boxes" in preds[0]
