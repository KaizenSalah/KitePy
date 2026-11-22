"""Unit tests for Constraint-Aware ExecutionPlanner and RuntimeGuard (OOM & NaN recovery)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitepy import Engine, ExecutionPlanner, TrainingConfig
from kitepy.core.adapters import ClassificationAdapter


def test_execution_planner_solves_micro_batch_and_grad_accum():
    model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    batch = (torch.randn(32, 64), torch.randint(0, 10, (32,)))
    planner = ExecutionPlanner()

    # Constrain memory budget tightly so target_batch_size=32 must split via gradient accumulation
    plan = planner.plan(
        model=model,
        device=torch.device("cpu"),
        target_batch_size=32,
        sample_batch=batch,
        adapter=ClassificationAdapter(),
        criterion=nn.CrossEntropyLoss(),
        memory_budget_mb=4.0,
    )
    assert plan.target_effective_batch_size == 32
    assert plan.micro_batch_size <= 32
    assert plan.micro_batch_size * plan.gradient_accumulation_steps >= 32
    assert plan.reason != "nominal"


def test_engine_executes_planned_micro_batches_when_loader_exceeds_plan(tmp_path):
    """If DataLoader yields batch=8 but ExecutionPlanner solves micro_bs<=4, Engine chunks each batch."""
    seen_batch_sizes = []

    class RecordingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            if self.training:
                seen_batch_sizes.append(x.size(0))
            return self.fc(x)

    ds = TensorDataset(torch.randn(16, 64), torch.randint(0, 10, (16,)))
    loader = DataLoader(ds, batch_size=8)

    cfg = TrainingConfig(
        epochs=1,
        batch_size=8,
        auto_plan=True,
        memory_budget_mb=0.05,  # Forces micro_batch_size=1
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
        devices="cpu",
        mixed_precision=False,
    )
    engine = Engine(RecordingNet(), cfg)
    engine.train(loader)
    assert engine.execution_plan is not None
    assert engine.execution_plan.micro_batch_size < 8
    assert max(seen_batch_sizes) <= engine.execution_plan.micro_batch_size
    assert engine.global_step == 2  # Preserves 2 optimizer steps (1 per 8-sample batch)


def test_engine_runtime_oom_microbatch_recovery(tmp_path):
    """Simulate CUDA/runtime OOM on batch > 2 and verify Engine automatically splits & recovers."""

    class OOMOnLargeBatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            if self.training and x.size(0) > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return self.fc(x)

    ds = TensorDataset(torch.randn(16, 8), torch.randint(0, 3, (16,)))
    loader = DataLoader(ds, batch_size=8)

    cfg = TrainingConfig(
        epochs=1,
        batch_size=8,
        auto_recover=True,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
        devices="cpu",
        mixed_precision=False,
    )
    engine = Engine(OOMOnLargeBatchNet(), cfg)
    _, hist = engine.train(loader)
    assert len(hist) == 1
    assert engine.guard.oom_recoveries >= 1
    assert engine.execution_report["recovery"]["oom_recoveries"] >= 1


def test_engine_nan_guard_skips_and_preserves_weights(tmp_path):
    class SometimesNaNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            out = self.fc(x)
            if self.calls == 1:
                return out * float("nan")
            return out

    ds = TensorDataset(torch.randn(16, 8), torch.randint(0, 3, (16,)))
    loader = DataLoader(ds, batch_size=8)

    net = SometimesNaNNet()
    cfg = TrainingConfig(
        epochs=1,
        auto_recover=True,
        max_consecutive_nan_steps=3,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
        devices="cpu",
        mixed_precision=False,
    )
    engine = Engine(net, cfg)
    _, hist = engine.train(loader)
    assert engine.guard.nan_skips == 1
    assert torch.isfinite(net.fc.weight).all()

    # Permanent NaN must raise after max_consecutive_nan_steps
    class AlwaysNaNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(x) * float("nan")

    with pytest.raises(RuntimeError, match="consecutive non-finite"):
        Engine(
            AlwaysNaNNet(),
            TrainingConfig(
                epochs=2,
                auto_recover=True,
                max_consecutive_nan_steps=2,
                checkpoint_dir=str(tmp_path),
                save_every_n_epochs=0,
                devices="cpu",
                mixed_precision=False,
            ),
        ).train(loader)
