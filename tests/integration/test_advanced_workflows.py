"""Integration tests: advanced power-user workflows (YAML, MixUp, LR auto, custom train_step, compare, FastAPI)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitepy import CNN, compare


def test_nested_yaml_config_training(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
model:
  arch: resnet
  depth: 18
  num_classes: 5
training:
  epochs: 1
  batch_size: 4
  lr: 0.001
  gradient_accumulation_steps: 2
  mixup_alpha: 0.2
  checkpoint_dir: "{(tmp_path / 'ckpts').as_posix()}"
  save_every_n_epochs: 0
  num_workers: 0
data:
  augmentation: none
""",
        encoding="utf-8",
    )

    model = CNN("resnet18", config=str(cfg_file))
    model.train("synthetic", fast_dev_run=True)
    assert model.config.num_classes == 5
    assert model.is_trained


def test_auto_lr_finder_and_differential_lr(tmp_path):
    model = CNN("resnet18", num_classes=4)
    model.train(
        "synthetic",
        epochs=1,
        batch_size=8,
        lr="auto",
        lr_find_num_steps=10,
        backbone_lr_mult=0.1,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert isinstance(model.training_history[-1]["lr"], float)


def test_custom_train_step_hook_and_dataset_input(tmp_path):
    ds = TensorDataset(torch.randn(16, 3, 64, 64), torch.randint(0, 3, (16,)))
    called = {"steps": 0}

    def my_step(engine, batch):
        called["steps"] += 1
        x, y = batch
        out = engine.model(x)
        loss = nn.functional.cross_entropy(out, y)
        return loss, out, y

    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3))
    model = CNN(net)
    model.train(
        ds,
        epochs=1,
        batch_size=8,
        train_step=my_step,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert called["steps"] >= 1


def test_compare_models_and_fastapi_app(tmp_path):
    results = compare(
        ["resnet18", "mobilenet_v3_small"],
        data="synthetic",
        epochs=1,
        batch_size=4,
        fast_dev_run=True,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert len(results) == 2
    assert results[0]["params"] > 0

    app = results[0]["instance"].to_fastapi()
    assert app is not None


def test_accelerate_engine_and_cli(tmp_path, capsys):
    from kitepy import Engine, TrainingConfig
    from kitepy.cli import main as cli_main
    from torch.utils.data import DataLoader, TensorDataset

    # 1. Exercise real HuggingFace Accelerate code path in Engine
    ds = TensorDataset(torch.randn(16, 8), torch.randint(0, 3, (16,)))
    loader = DataLoader(ds, batch_size=4)
    net = nn.Linear(8, 3)
    cfg = TrainingConfig(
        epochs=1,
        use_accelerate=True,
        gradient_accumulation_steps=2,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=1,
        mixed_precision=False,
    )
    trained_model, hist = Engine(net, cfg).train(loader, loader)
    assert len(hist) == 1
    assert isinstance(trained_model, nn.Module)

    # 2. Exercise CLI subcommands
    assert cli_main(["models", "audio"]) == 0
    out = capsys.readouterr().out
    assert "audio_cnn" in out
    assert cli_main(["datasets", "vision"]) == 0
    assert cli_main(["doctor"]) == 0

    # CLI train + export + predict
    cfg_path = tmp_path / "cli.yaml"
    cfg_path.write_text(
        f"model:\n  num_classes: 3\ntraining:\n  epochs: 1\n  batch_size: 4\n  fast_dev_run: true\n  num_workers: 0\n  checkpoint_dir: '{(tmp_path / 'ck').as_posix()}'\n  save_every_n_epochs: 0\n",
        encoding="utf-8",
    )
    ckpt_out = tmp_path / "cli_model.pt"
    assert cli_main(["train", str(cfg_path), "--data", "synthetic", "--save", str(ckpt_out)]) == 0
    assert ckpt_out.exists()

    ts_out = tmp_path / "cli_exported.pt"
    assert cli_main(["export", str(ckpt_out), str(ts_out), "--format", "torchscript"]) == 0
    assert ts_out.exists()


def test_execution_planner_end_to_end(tmp_path):
    model = CNN("resnet18", num_classes=4)
    model.train(
        "synthetic",
        epochs=1,
        batch_size=8,
        auto_plan=True,
        memory_budget_mb=64.0,
        fast_dev_run=True,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert model.execution_report["execution_plan"] is not None
    assert model.execution_report["lineage"]["kitepy_version"] == "0.2.0"
