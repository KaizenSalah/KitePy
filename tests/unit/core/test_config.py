"""Unit tests for Configuration system (flat/nested YAML, kwargs routing, typo hints)."""

import pytest

from kitepy.core.config import (
    CNNConfig,
    DataConfig,
    ExperimentConfig,
    TrainingConfig,
    auto_batch_size,
    auto_learning_rate,
    merge_configs,
    split_kwargs,
    validate_config,
)


def test_training_config_defaults_and_validation():
    cfg = TrainingConfig()
    assert cfg.optimizer == "adamw"
    assert validate_config(cfg)

    with pytest.raises(ValueError):
        validate_config(TrainingConfig(epochs=-1, max_steps=None))
    with pytest.raises(ValueError):
        validate_config(TrainingConfig(gradient_accumulation_steps=0))
    with pytest.raises(ValueError):
        validate_config(TrainingConfig(lr={"backbone": -0.1, "head": 1e-3}))


def test_unknown_key_typo_hint():
    with pytest.raises(ValueError, match="did you mean 'epochs'"):
        TrainingConfig.from_dict({"epocs": 10}, strict=True)


def test_yaml_and_json_roundtrip(tmp_path):
    cfg = TrainingConfig(epochs=25, gradient_accumulation_steps=4, ema=True)
    yaml_path = tmp_path / "train.yaml"
    cfg.to_yaml(str(yaml_path))
    loaded = TrainingConfig.from_yaml(str(yaml_path))
    assert loaded.epochs == 25
    assert loaded.gradient_accumulation_steps == 4
    assert loaded.ema is True

    cnn_cfg = CNNConfig(depth=50, num_classes=12)
    json_path = tmp_path / "cnn.json"
    cnn_cfg.to_json(str(json_path))
    loaded_cnn = CNNConfig.from_json(str(json_path))
    assert loaded_cnn.depth == 50 and loaded_cnn.num_classes == 12


def test_nested_experiment_yaml(tmp_path):
    yaml_content = """
experiment_name: resnet_cifar
model:
  arch: resnet
  depth: 50
  num_classes: 100
  pretrained: true
training:
  epochs: 40
  lr: 0.002
  gradient_accumulation_steps: 2
data:
  augmentation: randaugment
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    exp = ExperimentConfig.from_yaml(str(p), model_config_cls=CNNConfig)
    assert exp.model.num_classes == 100
    assert exp.model.pretrained is True
    assert exp.training.epochs == 40
    assert exp.training.gradient_accumulation_steps == 2
    assert exp.data.augmentation == "randaugment"


def test_split_kwargs_and_merge():
    model_kw, data_kw, train_kw, unknown = split_kwargs(
        {"num_classes": 10, "augmentation": "heavy", "epochs": 5, "lr": 1e-4, "bogus": 1},
        CNNConfig,
    )
    assert model_kw == {"num_classes": 10}
    assert data_kw == {"num_classes": 10, "augmentation": "heavy"}
    assert train_kw == {"epochs": 5, "lr": 1e-4}
    assert unknown == {"bogus": 1}

    merged = merge_configs(TrainingConfig(epochs=10), {"epochs": 20}, {"lr": 1e-4}, epochs=30)
    assert merged.epochs == 30 and merged.lr == 1e-4


def test_auto_tuning_helpers():
    assert auto_batch_size("base", 24.0) == 64
    assert auto_batch_size("base", 12.0) == 32
    assert auto_learning_rate(256, 1e-3) == 1e-3
    assert auto_learning_rate(512, 1e-3) == 2e-3
