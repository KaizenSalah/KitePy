"""Unit tests for rank-aware structured logging, atomic serialization, and safetensors bundles."""

import json

import torch

from kitepy import CNN, get_logger, set_log_level
from kitepy.core.serialization import atomic_torch_save, load_safe_bundle, save_safe_bundle


def test_structured_logging_and_rank_filtering(tmp_path, monkeypatch):
    json_log = tmp_path / "events.jsonl"
    logger = get_logger("kitepy.test_logger", level="INFO", json_file=str(json_log))
    logger.info("hello enterprise", extra={"metrics": {"loss": 0.42}})

    lines = json_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    parsed = json.loads(lines[-1])
    assert parsed["message"] == "hello enterprise"
    assert parsed["metrics"]["loss"] == 0.42

    set_log_level("WARNING")
    monkeypatch.setenv("RANK", "1")
    from kitepy.core.logging import _RankZeroFilter
    assert _RankZeroFilter().filter(None) is False
    monkeypatch.delenv("RANK", raising=False)
    set_log_level("INFO")


def test_atomic_torch_save_and_safetensors_bundle_roundtrip(tmp_path):
    pt_path = tmp_path / "atomic.pt"
    atomic_torch_save({"a": torch.tensor([1.0, 2.0])}, pt_path)
    loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
    assert torch.allclose(loaded["a"], torch.tensor([1.0, 2.0]))

    # Pickle-free safetensors + .meta.json via CNN.save/load
    model = CNN("resnet18", num_classes=3)
    model.class_names = ["x", "y", "z"]
    model._build_model()

    st_path = tmp_path / "safe_model.safetensors"
    model.save(st_path, safe=True)
    assert st_path.exists()
    assert st_path.with_suffix(".meta.json").exists()

    _, meta = load_safe_bundle(st_path)
    assert "lineage" in meta and "torch_version" in meta["lineage"]

    reloaded = CNN.load(st_path)
    assert reloaded.class_names == ["x", "y", "z"]
    assert reloaded.predict(torch.randn(1, 3, 224, 224)).shape == (1, 3)
