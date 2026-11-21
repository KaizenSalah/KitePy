"""Unit tests for LRFinder, MetricTracker, Presets, and Utils."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitepy import (
    LRFinder,
    MetricTracker,
    accuracy,
    list_datasets,
    list_models,
    precision_recall_f1,
    register_preset,
)
from kitepy.core.presets import get_preset
from kitepy.core.utils import (
    ModelEMA,
    build_parameter_groups,
    cleanup_old_checkpoints,
    count_parameters,
    format_time,
    get_checkpoint_path,
    get_device,
    get_head_parameter_names,
    get_model_size_mb,
    resolve_auto_value,
    set_seed,
)


def test_lr_finder_restores_weights_and_suggests_lr():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    init_weight = model[0].weight.detach().clone()
    loader = DataLoader(TensorDataset(torch.randn(16, 8), torch.randint(0, 3, (16,))), batch_size=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-7)
    finder = LRFinder(model, opt, nn.CrossEntropyLoss(), torch.device("cpu"))
    lrs, losses = finder.find(loader, num_steps=15)
    assert len(lrs) > 0 and len(losses) == len(lrs)
    suggested = finder.suggest_lr(skip_start=2, skip_end=1)
    assert 0 < suggested <= 10.0
    assert torch.allclose(model[0].weight, init_weight)


def test_metrics_and_tracker():
    output = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    target = torch.tensor([1, 0, 2])
    (acc1,) = accuracy(output, target, topk=(1,))
    assert acc1 == 100.0

    prf = precision_recall_f1(output, target, average="macro")
    assert prf["f1"] == 1.0

    tracker = MetricTracker(num_classes=3)
    tracker.update(output, target, loss=0.5)
    m = tracker.compute()
    assert m["accuracy"] == 100.0 and m["loss"] == 0.5
    cm = tracker.get_confusion_matrix()
    assert cm.sum().item() == 3


def test_presets_registry():
    all_models = list_models("all")
    assert "cnn" in all_models and "llm" in all_models and "tabular" in all_models
    assert "cifar10" in list_datasets("vision")
    register_preset("custom_resnet_test", {"arch": "resnet", "depth": 50}, modality="cnn", overwrite=True)
    assert get_preset("custom_resnet_test", "cnn")["depth"] == 50
    from kitepy.core.presets import CNN_PRESETS
    CNN_PRESETS.pop("custom_resnet_test", None)


def test_utils_head_detection_ema_and_param_groups():
    class FakeViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8))])
            self.blocks[0].mlp = nn.Module()
            self.blocks[0].mlp.fc1 = nn.Linear(8, 16)
            self.blocks[0].mlp.fc2 = nn.Linear(16, 8)
            self.head = nn.Linear(8, 5)

        def get_classifier(self):
            return self.head

    vit = FakeViT()
    head_names = get_head_parameter_names(vit)
    assert "head.weight" in head_names and "head.bias" in head_names
    # Crucial: MLP fc1/fc2 must NOT be mistakenly classified as head!
    assert "blocks.0.mlp.fc1.weight" not in head_names
    assert "blocks.0.mlp.fc2.weight" not in head_names

    groups = build_parameter_groups(vit, lr=1e-3, backbone_lr_mult=0.1, layer_decay=0.8)
    assert len(groups) >= 2

    ema = ModelEMA(vit, decay=0.9)
    with torch.no_grad():
        vit.head.weight.add_(1.0)
    ema.update(vit)
    assert not torch.allclose(ema.shadow.head.weight, vit.head.weight)


def test_utils_basics(tmp_path):
    set_seed(42)
    a = torch.randn(4)
    set_seed(42)
    assert torch.allclose(a, torch.randn(4))
    assert get_device("cpu").type == "cpu"
    assert count_parameters(nn.Linear(4, 2)) == 10
    assert get_model_size_mb(nn.Linear(4, 2)) > 0
    assert format_time(90) == "1.5m"
    assert resolve_auto_value("auto", lambda: 7) == 7

    p1 = get_checkpoint_path(str(tmp_path), "m", 1, "loss", 0.5)
    p2 = get_checkpoint_path(str(tmp_path), "m", 2, "loss", 0.1)
    open(p1, "w").close()
    open(p2, "w").close()
    cleanup_old_checkpoints(str(tmp_path), keep_top_k=1, metric_name="loss", lower_is_better=True)
    assert not os.path.exists(p1) if False else True
    from pathlib import Path
    assert not Path(p1).exists() and Path(p2).exists()
