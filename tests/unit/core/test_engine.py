"""Unit tests for universal Engine (callbacks, grad accum, schedulers, EMA, resume, mixup, etc.)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitepy import Callback, CSVLogger, EarlyStopping, Engine, TrainingConfig


def test_engine_fast_dev_run(tmp_path, tiny_linear_model, tiny_classification_loaders):
    train_loader, val_loader = tiny_classification_loaders
    cfg = TrainingConfig(
        fast_dev_run=True,
        checkpoint_dir=str(tmp_path),
        devices="cpu",
        mixed_precision=False,
    )
    engine = Engine(tiny_linear_model, cfg)
    _, history = engine.train(train_loader, val_loader)
    assert engine.global_step == 1
    assert len(history) == 1
    assert history[0]["train_loss"] > 0


def test_engine_callbacks_are_actually_invoked(tmp_path, tiny_linear_model, tiny_classification_loaders):
    train_loader, val_loader = tiny_classification_loaders
    events = []

    class TrackerCallback(Callback):
        def on_train_start(self, trainer, **kwargs):
            events.append("train_start")

        def on_epoch_start(self, trainer, epoch, **kwargs):
            events.append(f"epoch_start:{epoch}")

        def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
            events.append("batch_end")

        def on_validation_start(self, trainer, **kwargs):
            events.append("val_start")

        def on_validation_end(self, trainer, metrics, **kwargs):
            events.append("val_end")

        def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
            events.append(f"epoch_end:{epoch}")

        def on_train_end(self, trainer, **kwargs):
            events.append("train_end")

    cfg = TrainingConfig(
        epochs=2,
        checkpoint_dir=str(tmp_path),
        devices="cpu",
        mixed_precision=False,
        csv_log_dir=str(tmp_path / "logs"),
        callbacks=[TrackerCallback()],
    )
    Engine(tiny_linear_model, cfg).train(train_loader, val_loader)

    assert "train_start" in events
    assert "train_end" in events
    assert "epoch_start:1" in events and "epoch_end:2" in events
    assert "val_start" in events and "val_end" in events
    assert "batch_end" in events
    assert (tmp_path / "logs" / "metrics.csv").exists()


def test_engine_gradient_accumulation_and_max_steps(tmp_path, tiny_linear_model, tiny_classification_loaders):
    train_loader, _ = tiny_classification_loaders  # 4 batches per epoch
    cfg = TrainingConfig(
        epochs=5,
        gradient_accumulation_steps=2,
        max_steps=3,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
        devices="cpu",
        mixed_precision=False,
    )
    engine = Engine(tiny_linear_model, cfg)
    engine.train(train_loader)
    assert engine.global_step == 3


def test_engine_schedulers_including_plateau_and_warmup(tmp_path, tiny_linear_model, tiny_classification_loaders):
    train_loader, val_loader = tiny_classification_loaders
    for sched in ["cosine", "linear", "step", "multistep", "exponential", "polynomial", "onecycle", "warmup_cosine", "plateau"]:
        cfg = TrainingConfig(
            epochs=2,
            scheduler=sched,
            warmup_steps=0.2 if sched in ("cosine", "linear") else 0,
            checkpoint_dir=str(tmp_path),
            save_every_n_epochs=0,
            devices="cpu",
            mixed_precision=False,
        )
        engine = Engine(nn.Linear(10, 3), cfg)
        _, hist = engine.train(train_loader, val_loader)
        assert len(hist) == 2


def test_engine_ema_overfit_and_resume_checkpoint(tmp_path, tiny_linear_model, tiny_classification_loaders):
    train_loader, val_loader = tiny_classification_loaders
    ckpt_dir = tmp_path / "ckpts"

    cfg1 = TrainingConfig(
        epochs=2,
        ema=True,
        ema_decay=0.99,
        overfit_batches=1,
        checkpoint_dir=str(ckpt_dir),
        save_every_n_epochs=1,
        devices="cpu",
        mixed_precision=False,
    )
    engine1 = Engine(tiny_linear_model, cfg1)
    _, hist1 = engine1.train(train_loader, val_loader)
    assert len(hist1) == 2

    saved_ckpts = sorted(ckpt_dir.glob("*.pt"))
    assert len(saved_ckpts) >= 1

    # Resume to epoch 3
    cfg2 = TrainingConfig(
        epochs=3,
        ema=True,
        resume_from_checkpoint=str(saved_ckpts[-1]),
        checkpoint_dir=str(ckpt_dir),
        save_every_n_epochs=1,
        devices="cpu",
        mixed_precision=False,
    )
    engine2 = Engine(nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3)), cfg2)
    _, hist2 = engine2.train(train_loader, val_loader)
    assert len(hist2) == 3
    assert hist2[-1]["epoch"] == 3


def test_engine_differential_lr_and_class_weights_balanced(tmp_path):
    class NetWithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
            self.head = nn.Linear(8, 2)

        def forward(self, x):
            return self.head(self.backbone(x))

    ds = TensorDataset(torch.randn(20, 8), torch.tensor([0] * 16 + [1] * 4))
    loader = DataLoader(ds, batch_size=10)

    cfg = TrainingConfig(
        epochs=1,
        lr={"backbone": 1e-4, "head": 1e-2},
        class_weights="balanced",
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
        devices="cpu",
        mixed_precision=False,
    )
    engine = Engine(NetWithHead(), cfg)
    lrs = sorted({pg["lr"] for pg in engine.optimizer.param_groups})
    assert lrs == [1e-4, 1e-2]
    engine.train(loader)
    assert engine.criterion.weight is not None


def test_engine_callback_early_stopping(tmp_path, tiny_linear_model, tiny_classification_loaders):
    train_loader, val_loader = tiny_classification_loaders
    es = EarlyStopping(monitor="val_loss", patience=1, min_delta=999.0)
    cfg = TrainingConfig(
        epochs=10,
        callbacks=[es],
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
        devices="cpu",
        mixed_precision=False,
    )
    _, hist = Engine(tiny_linear_model, cfg).train(train_loader, val_loader)
    assert len(hist) <= 2
