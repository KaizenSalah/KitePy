"""Unit tests for Callback system (including CSVLogger & validation hooks)."""

import csv
import os
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from kitepy import (
    CallbackList,
    CSVLogger,
    EarlyStopping,
    LambdaCallback,
    LRMonitor,
    ModelCheckpoint,
    ProgressLogger,
    TrainingConfig,
)


class _DummyTrainer:
    def __init__(self):
        self.model = nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.config = TrainingConfig()


def test_early_stopping():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    es.on_epoch_end(None, 1, {"val_loss": 10.0})
    assert es.best_value == 10.0 and es.counter == 0
    es.on_epoch_end(None, 2, {"val_loss": 9.0})
    assert es.best_value == 9.0
    es.on_epoch_end(None, 3, {"val_loss": 9.5})
    assert es.counter == 1 and not es.should_stop
    es.on_epoch_end(None, 4, {"val_loss": 9.5})
    assert es.counter == 2 and es.should_stop


def test_model_checkpoint_handles_missing_val_loss(tmp_path):
    trainer = _DummyTrainer()
    ckpt_path = str(tmp_path / "model_{epoch}.pt")
    mc = ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", mode="min", save_best_only=True, save_top_k=2)

    mc.on_epoch_end(trainer, 1, {"val_loss": 10.0})
    assert os.path.exists(tmp_path / "model_1.pt")

    mc.on_epoch_end(trainer, 2, {"val_loss": 11.0})
    assert not os.path.exists(tmp_path / "model_2.pt")

    # Even if val_loss is None, fallback to train_loss without TypeError
    mc.on_epoch_end(trainer, 3, {"val_loss": None, "train_loss": 5.0})
    assert os.path.exists(tmp_path / "model_3.pt")


def test_csv_logger(tmp_path):
    trainer = _DummyTrainer()
    cb = CSVLogger(log_dir=str(tmp_path), filename="metrics.csv")
    cb.on_train_start(trainer)
    cb.on_epoch_end(trainer, 1, {"train_loss": 0.8, "val_loss": 0.9})
    cb.on_epoch_end(trainer, 2, {"train_loss": 0.4, "val_loss": 0.5})

    csv_file = tmp_path / "metrics.csv"
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert float(rows[1]["train_loss"]) == 0.4


def test_callback_list_shortcuts_and_validation_hooks():
    cbl = CallbackList(["early_stopping", "lr_monitor", "progress"])
    assert len(cbl.callbacks) == 3
    assert isinstance(cbl.callbacks[0], EarlyStopping)
    assert isinstance(cbl.callbacks[1], LRMonitor)
    assert isinstance(cbl.callbacks[2], ProgressLogger)

    mock_cb = MagicMock()
    cbl2 = CallbackList([mock_cb])
    cbl2.on_validation_start(None)
    cbl2.on_validation_end(None, {"val_loss": 0.5})
    mock_cb.on_validation_start.assert_called_once()
    mock_cb.on_validation_end.assert_called_once()


def test_lambda_callback():
    mock_fn = MagicMock()
    lc = LambdaCallback(on_epoch_end=mock_fn)
    lc.on_epoch_end("t", 1, {"train_loss": 0.1})
    mock_fn.assert_called_once_with("t", 1, {"train_loss": 0.1})
