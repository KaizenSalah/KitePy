"""Unit tests for Vision Transformers (ViT, DeiT, Swin) and ViT freeze_backbone fix."""

import torch

from kitepy import Transformer


def test_vit_creation_and_forward():
    m = Transformer("vit_tiny", num_classes=10)
    m._build_model()
    out = m.predict(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_vit_freeze_backbone_freezes_mlp_fc_layers():
    """Regression test: ViT blocks.*.mlp.fc1/fc2 must be frozen, only classifier head stays trainable."""
    vit = Transformer("vit_tiny", num_classes=10)
    vit._build_model()
    vit.freeze_backbone()

    trainable_names = [n for n, p in vit.model.named_parameters() if p.requires_grad]
    assert len(trainable_names) == 2  # head.weight and head.bias
    assert all("head" in n for n in trainable_names)
    assert not any("mlp.fc" in n for n in trainable_names)

    # Gradual unfreeze last block + full unfreeze
    vit.unfreeze(layers=1)
    assert sum(p.requires_grad for p in vit.model.parameters()) >= 2

    vit.unfreeze()
    assert all(p.requires_grad for p in vit.model.parameters())


def test_vit_training_synthetic(tmp_path):
    vit = Transformer("vit_tiny", num_classes=5)
    vit.train(
        "synthetic",
        epochs=1,
        batch_size=4,
        fast_dev_run=True,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert vit.is_trained
