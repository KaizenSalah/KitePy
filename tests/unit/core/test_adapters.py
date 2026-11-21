"""Unit tests for TaskAdapter protocol and Pillar Registry."""

import pytest
import torch
import torch.nn as nn

from kitepy.core.adapters import (
    BinaryClassificationAdapter,
    CausalLMAdapter,
    ClassificationAdapter,
    ContrastiveAdapter,
    CustomStepAdapter,
    GenerativeAdapter,
    MultiLabelClassificationAdapter,
    RegressionAdapter,
    SequenceClassificationAdapter,
    get_adapter,
    move_to_device,
)


def test_move_to_device_nested():
    data = {"x": torch.zeros(2, 3), "meta": (torch.ones(2), "tag", [torch.tensor(1.0)])}
    moved = move_to_device(data, torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert moved["meta"][1] == "tag"


def test_classification_adapter():
    adapter = ClassificationAdapter()
    model = nn.Linear(4, 3)
    criterion = nn.CrossEntropyLoss()
    batch = (torch.randn(6, 4), torch.randint(0, 3, (6,)))
    loss, out, targets = adapter.forward_and_loss(model, batch, criterion)
    assert loss.ndim == 0
    bm = adapter.compute_batch_metrics(out, targets, float(loss.item()), batch)
    assert bm["total"] == 6.0
    final = adapter.finalize_epoch_metrics(bm, 1)
    assert 0.0 <= final["acc"] <= 100.0


def test_binary_and_multilabel_adapters():
    bin_ad = BinaryClassificationAdapter()
    m_bin = nn.Linear(4, 1)
    loss, out, tgt = bin_ad.forward_and_loss(m_bin, (torch.randn(5, 4), torch.randint(0, 2, (5,))), nn.BCEWithLogitsLoss())
    assert loss.ndim == 0
    bm = bin_ad.compute_batch_metrics(out, tgt, float(loss.item()), None)
    assert "acc" in bin_ad.finalize_epoch_metrics(bm, 1)

    ml_ad = MultiLabelClassificationAdapter()
    m_ml = nn.Linear(4, 3)
    loss2, out2, tgt2 = ml_ad.forward_and_loss(m_ml, (torch.randn(5, 4), torch.randint(0, 2, (5, 3))), nn.BCEWithLogitsLoss())
    assert loss2.ndim == 0
    bm2 = ml_ad.compute_batch_metrics(out2, tgt2, float(loss2.item()), None)
    assert "acc" in ml_ad.finalize_epoch_metrics(bm2, 1)


def test_regression_adapter():
    adapter = RegressionAdapter()
    model = nn.Linear(4, 1)
    batch = (torch.randn(8, 4), torch.randn(8))
    loss, out, tgt = adapter.forward_and_loss(model, batch, nn.MSELoss())
    assert loss.ndim == 0
    bm = adapter.compute_batch_metrics(out, tgt, float(loss.item()), batch)
    final = adapter.finalize_epoch_metrics(bm, 1)
    assert "mae" in final and "rmse" in final
    assert final["rmse"] >= 0.0


def test_causal_lm_and_sequence_classification_adapters():
    class DummyLM(nn.Module):
        def forward(self, input_ids, **kwargs):
            return torch.randn(input_ids.size(0), input_ids.size(1), 20)

    lm_ad = CausalLMAdapter()
    ids = torch.randint(0, 20, (4, 8))
    batch = {"input_ids": ids, "labels": ids.clone()}
    loss, out, tgt = lm_ad.forward_and_loss(DummyLM(), batch, nn.CrossEntropyLoss())
    assert loss.ndim == 0
    bm = lm_ad.compute_batch_metrics(out, tgt, float(loss.item()), batch)
    assert "perplexity" in lm_ad.finalize_epoch_metrics(bm, 1)

    class DummySeq(nn.Module):
        def forward(self, input_ids, **kwargs):
            return torch.randn(input_ids.size(0), 2)

    seq_ad = SequenceClassificationAdapter()
    batch_cls = {"input_ids": ids, "labels": torch.randint(0, 2, (4,))}
    loss2, out2, tgt2 = seq_ad.forward_and_loss(DummySeq(), batch_cls, nn.CrossEntropyLoss())
    assert loss2.ndim == 0


def test_contrastive_and_generative_and_custom_adapters():
    class DummyCLIP(nn.Module):
        def forward(self, img, txt):
            b = img.size(0)
            return torch.randn(b, b), torch.randn(b, b)

    c_ad = ContrastiveAdapter()
    loss, out, tgt = c_ad.forward_and_loss(DummyCLIP(), (torch.randn(4, 3), torch.randint(0, 5, (4, 6))), None)
    assert loss.ndim == 0

    class DummyVAE(nn.Module):
        def forward(self, x):
            return x.clone(), torch.zeros(x.size(0), 4), torch.zeros(x.size(0), 4)

    g_ad = GenerativeAdapter()
    loss_g, _, _ = g_ad.forward_and_loss(DummyVAE(), (torch.rand(4, 3, 8, 8), None), None)
    assert loss_g.ndim == 0

    custom = get_adapter(train_step=lambda engine, b: (b[0].sum(), b[0], b[1]))
    assert isinstance(custom, CustomStepAdapter)
    with pytest.raises(ValueError):
        get_adapter("nonexistent_task")
