"""Transfer Learning with Vision Transformers: freeze backbone, differential LRs, and EMA."""

import tempfile
from kitepy import Transformer, EarlyStopping

with tempfile.TemporaryDirectory() as tmp:
    vit = Transformer("vit_tiny", num_classes=10)

    # Freeze all backbone & transformer blocks; keep only the classification head trainable
    vit.freeze_backbone()

    # Fine-tune with differential LRs, EMA weights, and EarlyStopping
    vit.train(
        "synthetic",
        epochs=1,
        batch_size=8,
        lr={"backbone": 1e-4, "head": 1e-3},
        ema=True,
        callbacks=[EarlyStopping(patience=3)],
        fast_dev_run=True,
        num_workers=0,
        checkpoint_dir=tmp,
    )

    # Unfreeze last 2 transformer blocks for stage-2 fine-tuning
    vit.unfreeze(layers=2)
    print("Stage-2 trainable params:", sum(p.numel() for p in vit.unwrap().parameters() if p.requires_grad))
