"""Regression tests for core hardening, edge cases, and serialization safety."""

from collections import UserDict
import csv

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitepy import (
    CNN,
    LLM,
    Audio,
    Engine,
    LRFinder,
    ModelCheckpoint,
    Tabular,
    TextClassifier,
    TrainingConfig,
    Vision,
)
from kitepy.core.adapters import CausalLMAdapter, SequenceClassificationAdapter


_CANARY_EXECUTED = False


def _harmless_canary_trigger():
    global _CANARY_EXECUTED
    _CANARY_EXECUTED = True
    return {"harmless": 1}


class _MaliciousPickleCanary:
    def __reduce__(self):
        return (_harmless_canary_trigger, ())


def test_vision_and_tabular_save_load_preserve_task_and_weights(tmp_path):
    for task, model_name, nc, sample in [
        ("segmentation", "unet", 3, torch.randn(1, 3, 32, 32)),
        ("detection", "tiny_detector", 4, torch.randn(1, 3, 64, 64)),
        ("binary", "resnet18", 1, torch.randn(1, 3, 32, 32)),
    ]:
        vis = Vision(task=task, model=model_name, num_classes=nc)
        vis.train("synthetic", epochs=1, img_size=32 if task != "detection" else 64, batch_size=4, fast_dev_run=True, save_every_n_epochs=0)
        bundle_path = tmp_path / f"vis_{task}.safetensors"
        vis.save(bundle_path, safe=True)

        reloaded = Vision.load(bundle_path)
        assert reloaded.task == task
        assert reloaded.model_name == model_name
        assert reloaded.config.num_classes == nc
        out = reloaded.predict(sample)
        if task == "detection":
            assert isinstance(out, list) and "boxes" in out[0]
        elif task == "segmentation":
            assert out.shape == (1, nc, 32, 32)
        elif task == "binary":
            probs = reloaded.predict(sample, top_k=2)
            assert len(probs) == 2 and abs(sum(p["confidence"] for p in probs) - 1.0) < 1e-4

    tab = Tabular("tabular_mlp", task="regression")
    tab.train("synthetic", epochs=1, batch_size=8, fast_dev_run=True, save_every_n_epochs=0)
    tab_path = tmp_path / "tab_reg.safetensors"
    tab.save(tab_path, safe=True)
    reloaded_tab = Tabular.load(tab_path)
    assert reloaded_tab.task == "regression"
    preds = reloaded_tab.predict(np.random.randn(2, 16).astype(np.float32), top_k=3)
    assert isinstance(preds, list) and isinstance(preds[0], float)


def test_to_fastapi_text_detection_and_regression():
    clf = TextClassifier("tiny_gpt", num_classes=2, context_length=32)
    clf.train("synthetic", epochs=1, batch_size=4, fast_dev_run=True, save_every_n_epochs=0)
    resp_txt = TestClient(clf.to_fastapi()).post("/predict", json={"inputs": ["KitePy test sentence"], "top_k": 2})
    assert resp_txt.status_code == 200
    preds_txt = resp_txt.json()["predictions"]
    assert isinstance(preds_txt, list) and "confidence" in preds_txt[0][0]

    det = Vision(task="detection", model="tiny_detector", num_classes=3)
    det.train("synthetic", epochs=1, img_size=64, batch_size=4, fast_dev_run=True, save_every_n_epochs=0)
    resp_det = TestClient(det.to_fastapi()).post("/predict", json={"inputs": np.random.rand(1, 3, 64, 64).tolist(), "top_k": 3})
    assert resp_det.status_code == 200
    assert "boxes" in resp_det.json()["predictions"]

    cnn = Vision(task="classification", model="resnet18", num_classes=4)
    cnn._build_model()
    resp_cnn = TestClient(cnn.to_fastapi()).post("/predict", json={"inputs": np.random.randn(1, 3, 32, 32).tolist(), "top_k": 2})
    assert resp_cnn.status_code == 200 and len(resp_cnn.json()["predictions"]) == 2

    reg = Tabular("tabular_mlp", task="regression")
    reg.train("synthetic", epochs=1, batch_size=8, fast_dev_run=True, save_every_n_epochs=0)
    resp_reg = TestClient(reg.to_fastapi()).post("/predict", json={"inputs": np.random.randn(2, 16).tolist(), "top_k": 3})
    assert resp_reg.status_code == 200
    preds_reg = resp_reg.json()["predictions"]
    assert isinstance(preds_reg, list) and isinstance(preds_reg[0], float)


def test_find_lr_uses_task_loss_without_error():
    vis_bin = Vision(task="binary", model="resnet18", num_classes=1)
    lr_bin = vis_bin.find_lr("synthetic", img_size=32, batch_size=4, num_workers=0)
    assert lr_bin > 0

    tab_reg = Tabular("tabular_mlp", task="regression")
    lr_reg = tab_reg.find_lr("synthetic", batch_size=8, num_workers=0)
    assert lr_reg > 0


def test_model_checkpoint_eviction_csv_drift_and_resume_compat(tmp_path):
    class DummyTrainer:
        def __init__(self):
            self.model = nn.Linear(4, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

    best_path = tmp_path / "best.pt"
    cb1 = ModelCheckpoint(filepath=str(best_path), monitor="val_loss", save_best_only=True, save_top_k=2)
    t1 = DummyTrainer()
    for ep, vloss in [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4)]:
        cb1.on_epoch_end(t1, ep, {"val_loss": vloss, "train_loss": vloss})
    assert best_path.exists()

    fifo_dir = tmp_path / "keep_best"
    cb2 = ModelCheckpoint(filepath=str(fifo_dir / "ep_{epoch:02d}.pt"), monitor="val_loss", save_best_only=False, save_top_k=2)
    for ep, vloss in [(1, 0.05), (2, 0.90), (3, 1.50)]:
        cb2.on_epoch_end(t1, ep, {"val_loss": vloss, "train_loss": vloss})
    kept = sorted(p.name for p in fifo_dir.glob("*.pt"))
    assert "ep_01.pt" in kept and "ep_03.pt" not in kept

    ds = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))), batch_size=8)
    val_ds = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=8)
    csv_dir = tmp_path / "csv"
    eng = Engine(
        nn.Linear(4, 2),
        TrainingConfig(epochs=2, val_every_n_epochs=2, csv_log_dir=str(csv_dir), save_every_n_epochs=0, mixed_precision=False),
    )
    eng.train(ds, val_ds)
    with open(csv_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    assert "val_loss" in cols and rows[-1]["val_loss"] != ""

    cb_ckpt = tmp_path / "cb_epoch1.pt"
    ModelCheckpoint(filepath=str(cb_ckpt), save_best_only=False).on_epoch_end(t1, 1, {"val_loss": 0.5, "train_loss": 0.5})
    eng_res = Engine(
        nn.Linear(4, 2),
        TrainingConfig(epochs=2, resume_from_checkpoint=str(cb_ckpt), save_every_n_epochs=0, mixed_precision=False),
    )
    eng_res.train(ds, val_ds)
    assert eng_res.execution_report["epochs_completed"] == 1


def test_lm_and_seq_cls_adversarial_batches_and_greedy_generate():
    lm_ad = CausalLMAdapter()
    toy_lm = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16))
    for batch in [
        UserDict({"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.full((1, 4), -100)}),
        {"input_ids": torch.tensor([[5]]), "labels": torch.tensor([[5]])},
    ]:
        loss, _, _ = lm_ad.forward_and_loss(toy_lm, batch, nn.CrossEntropyLoss())
        assert torch.isfinite(loss)
        loss.backward()

    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 8)
            self.fc = nn.Linear(8, 2)

        def forward(self, input_ids, attention_mask=None, labels=None):
            return self.fc(self.emb(input_ids).mean(dim=1))

    seq_ad = SequenceClassificationAdapter()
    u_batch = UserDict({"input_ids": torch.randint(0, 32, (2, 5)), "labels": torch.tensor([0, 1])})
    loss, _, _ = seq_ad.forward_and_loss(SeqModel(), u_batch, nn.CrossEntropyLoss())
    assert torch.isfinite(loss)

    tiny = LLM("tiny_gpt", context_length=32)
    tiny.train("synthetic", epochs=1, batch_size=4, fast_dev_run=True, save_every_n_epochs=0)
    out1 = tiny.generate("kitepy", max_new_tokens=6, temperature=0.0)
    out2 = tiny.generate("kitepy", max_new_tokens=6, temperature=0.0)
    assert out1 == out2


def test_empty_loader_rejected_across_engine_and_lr_finder():
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 8), torch.empty(0, dtype=torch.long)), batch_size=4)
    m_empty = CNN(nn.Linear(8, 2))
    with pytest.raises(ValueError, match="empty"):
        m_empty.train(empty_loader, epochs=1, save_every_n_epochs=0)
    assert not m_empty.is_trained

    net = nn.Linear(8, 2)
    finder = LRFinder(net, torch.optim.AdamW(net.parameters(), lr=1e-7), nn.CrossEntropyLoss(), torch.device("cpu"))
    with pytest.raises(ValueError, match="empty"):
        finder.find(empty_loader, num_steps=5)


def test_serialization_blocks_pickle_rce_and_missing_sidecar_error(tmp_path):
    global _CANARY_EXECUTED
    _CANARY_EXECUTED = False

    evil_pt = tmp_path / "untrusted.pt"
    torch.save({"model_name": "resnet18", "config": {"num_classes": 2}, "model_state_dict": {}, "trap": _MaliciousPickleCanary()}, evil_pt)

    with pytest.raises(RuntimeError, match="Refusing to unpickle"):
        CNN.load(evil_pt)
    assert _CANARY_EXECUTED is False

    m = CNN("resnet18", num_classes=3)
    m._build_model()
    orphan_st = tmp_path / "orphan.safetensors"
    m.export(orphan_st, format="safetensors")
    with pytest.raises(ValueError, match="sidecar"):
        CNN.load(orphan_st)


def test_audio_short_and_empty_waveforms(tmp_path):
    for arch in ("audio_cnn", "audio_transformer"):
        au = Audio(arch, num_classes=4)
        au.train("synthetic", epochs=1, batch_size=4, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
        for n in (64, 10, 1):
            out = au.predict(torch.randn(n))
            assert out.shape == (1, 4)
        with pytest.raises(ValueError, match="empty"):
            au.predict(torch.empty(0))
